"""End-user wall clock of cpu_attention against torch SDPA and vLLM's CPU attention.

    uv run python benchmarks/bench.py --json out.json \
        [--vllm-bf16 X.so --vllm-fp32 Y.so --vllm-isa-bf16 amx|vec|neon --vllm-isa-fp32 vec|neon]

Method (benchmarks/BENCHMARK.md has the details):

* Every arm is timed at its Python API, one call per sample, as a user calls it:
  ``flash_attn_func`` against ``F.scaled_dot_product_attention`` for prefill, and
  ``flash_attn_with_kvcache`` against vLLM's ``reshape_and_cache`` +
  ``cpu_attention_with_kv_cache`` for paged decode.
* 3 warmup calls, then 10 timed calls per arm, interleaved across arms in a rotated
  order; L2 and L3 are flushed (a write sweep of twice the LLC) before every call.
  The median decides; min and max are reported.
* Paged rows run each library at every page size it supports among 16 and 32 and
  keep its best (vLLM on x86 takes only multiples of 32, so 32 there).
* Threads = physical cores.
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cpu_attention  # noqa: E402
from cpu_attention import _kernel  # noqa: E402

DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
WARMUP, RUNS, PAGES = 3, 10, (16, 32)

# MHA shapes (GQA/MQA are not supported yet).
# Prefill: (name, batch, seq, heads, head_dim, causal).
FLASH = [
    ("short prompt", 1, 128, 16, 64, True),
    ("prompt 1k", 1, 1024, 16, 64, True),
    ("prompt 2k, d128", 1, 2048, 32, 128, True),
    ("long prompt 4k, d128", 1, 4096, 32, 128, True),
    ("encoder batch", 8, 512, 16, 64, False),
    ("encoder 1500 (Whisper)", 1, 1500, 20, 64, False),
    ("encoder, d32", 8, 256, 16, 32, False),
]
# Paged decode: (name, batch, context, heads, head_dim, append).
# `context` is a length or a list of per-sequence lengths; `append` writes the new
# token's K/V in the same call (self-attention), otherwise the cache is read only
# (cross-attention over an encoder's K/V).
PAGED = [
    ("b1 short", 1, 256, 16, 64, True),
    ("b1 long 4k, d128", 1, 4096, 32, 128, True),
    ("b8 1k", 8, 1024, 16, 64, True),
    ("b8 1k, d32", 8, 1024, 16, 32, True),
    ("b8 4k, d128", 8, 4096, 32, 128, True),
    ("b8 mixed 17-2047", 8, [17, 255, 1023, 1025, 33, 513, 2047, 64], 16, 64, True),
    ("b8 cross 1500 (Whisper)", 8, 1500, 20, 64, False),
    ("b32 512", 32, 512, 16, 64, True),
    ("b32 1k, d128", 32, 1024, 32, 128, True),
]


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def machine_type():
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
        headers={"Metadata-Flavor": "Google"})
    try:
        return urllib.request.urlopen(req, timeout=1).read().decode().rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        return platform.node()


def load_extension(path, _loaded={}):
    """A compiled extension, imported under its file name (its init symbol), once."""
    if path not in _loaded:
        name = os.path.basename(path).split(".")[0]
        spec = importlib.util.spec_from_file_location(name, path)
        _loaded[path] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_loaded[path])
    return _loaded[path]


def load_package(directory, name):
    """Another build of the package (its .py files and shared objects), imported under
    its own name from a private copy: no shared module state or dlopen handle."""
    dst = os.path.join(tempfile.mkdtemp(), name)
    shutil.copytree(directory, dst, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(dst, "__init__.py"), submodule_search_locations=[dst])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def measure(fns, flush):
    """3 warmup + 10 timed calls per arm, rotated, caches flushed before each call."""
    names = list(fns)
    for n in names:
        for _ in range(WARMUP):
            flush()
            fns[n]()
    us = {n: [] for n in names}
    for r in range(RUNS):
        for n in names[r % len(names):] + names[:r % len(names)]:
            flush()
            t0 = time.perf_counter_ns()
            fns[n]()
            us[n].append((time.perf_counter_ns() - t0) / 1000)
    return {n: dict(median=sorted(x)[len(x) // 2], min=min(x), max=max(x), samples=x)
            for n, x in us.items()}


def flash_row(spec, dtype, pkgs):
    _, b, s, h, d, causal = spec
    q, k, v = (torch.randn(b, s, h, d, dtype=dtype) for _ in range(3))
    out = {}
    fns = {name: (lambda p=p, name=name: out.__setitem__(
        name, p.flash_attn_func(q, k, v, causal=causal))) for name, p in pkgs.items()}
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    fns["sdpa"] = lambda: out.__setitem__("sdpa", F.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal).transpose(1, 2))
    return fns, out, "sdpa"


def paged_row(spec, dtype, pkgs, vllm):
    _, b, ctx, h, d, append = spec
    lens = ctx if isinstance(ctx, list) else [ctx] * b
    staged = [n - 1 if append else n for n in lens]  # an append call adds the last token
    kv = [(torch.randn(n, h, d, dtype=dtype), torch.randn(n, h, d, dtype=dtype))
          for n in staged]
    q = torch.randn(b, 1, h, d, dtype=dtype)
    kn, vn = (torch.randn(b, 1, h, d, dtype=dtype) for _ in range(2))
    before = torch.tensor(staged, dtype=torch.int32)
    out, fns = {}, {}
    for page in PAGES:
        nb = [-(-n // page) for n in lens]
        bt = torch.zeros(b, max(nb), dtype=torch.int32)
        for i, n in enumerate(nb):
            bt[i, :n] = torch.arange(sum(nb[:i]), sum(nb[:i]) + n, dtype=torch.int32)
        for name, p in pkgs.items():
            kc = torch.zeros(sum(nb), h, page, d, dtype=dtype)
            vc = torch.zeros_like(kc)
            for i, (k, v) in enumerate(kv):  # stage each prompt through the API
                if len(k):
                    p.flash_attn_with_kvcache(q[i:i + 1], kc, vc, k=k[None], v=v[None],
                                              cache_seqlens=0, page_table=bt[i:i + 1])
            arm = f"{name} p{page}"
            extra = dict(k=kn, v=vn) if append else {}
            fns[arm] = (lambda p=p, kc=kc, vc=vc, bt=bt, arm=arm, extra=extra: out.__setitem__(
                arm, p.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=before,
                                               page_table=bt, **extra)))
        if vllm is not None and (page % 32 == 0 or platform.machine() == "aarch64"):
            try:
                fns[f"vllm p{page}"] = vllm_paged(vllm, page, bt, kv, q, kn, vn, lens,
                                                  append, out)
            except Exception as e:  # noqa: BLE001
                print(f"  vllm p{page}: {str(e).splitlines()[0][:100]}", flush=True)
    return fns, out, "vllm"


def vllm_paged(vllm, page, bt, kv, q, kn, vn, lens, append, out):
    """vLLM's two calls per decode step: store the new token, attend. The scheduler
    metadata is per batch (once per step in vLLM), so it is built outside the timer."""
    mod, isa = vllm
    b, _, h, d = q.shape
    kc = torch.zeros(int(bt.max()) + 1, h, page, d, dtype=q.dtype)
    vc = torch.zeros_like(kc)
    slot = lambda i, t: int(bt[i, t // page]) * page + t % page  # noqa: E731
    slots = torch.tensor([slot(i, t) for i, (k, _) in enumerate(kv) for t in range(len(k))],
                         dtype=torch.int64)
    mod.cpu_attn_reshape_and_cache(torch.cat([k for k, _ in kv]), torch.cat([v for _, v in kv]),
                                   kc, vc, slots, isa)
    seqlens = torch.tensor(lens, dtype=torch.int32)
    qsl = torch.arange(b + 1, dtype=torch.int32)
    meta = mod.get_scheduler_metadata(b, h, h, d, seqlens, q.dtype, qsl, True, -1, isa,
                                      False, None, "auto")
    new_slots = torch.tensor([slot(i, n - 1) for i, n in enumerate(lens)], dtype=torch.int64)
    q3, kn3, vn3 = q.squeeze(1), kn.squeeze(1), vn.squeeze(1)
    o = torch.empty_like(q3)
    name = f"vllm p{page}"

    def f():
        if append:
            mod.cpu_attn_reshape_and_cache(kn3, vn3, kc, vc, new_slots, isa)
        mod.cpu_attention_with_kv_cache(q3, kc, vc, o, qsl, seqlens, d ** -0.5, True, None,
                                        -1, bt, 0.0, meta, None, None, 1.0, 1.0, "auto")
        out[name] = o.unsqueeze(1)
    f()
    return f


def best(timing, prefix):
    arms = [n for n in timing if n.split(" ")[0] == prefix]
    return min(arms, key=lambda n: timing[n]["median"]) if arms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="", help="substring filter on row names")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "both"), default="both")
    ap.add_argument("--threads", type=int, default=0, help="default: physical cores")
    ap.add_argument("--compare", default="", metavar="PACKAGE_DIR",
                    help="also time another build of the package (an A/B in one process)")
    ap.add_argument("--vllm-bf16", default="")
    ap.add_argument("--vllm-fp32", default="")
    ap.add_argument("--vllm-isa-bf16", default="vec", help="amx on AMX CPUs, neon on aarch64")
    ap.add_argument("--vllm-isa-fp32", default="vec", help="neon on aarch64")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or _kernel._physical_cores())
    pkgs = {"ours": cpu_attention}
    if args.compare:
        pkgs["compare"] = load_package(args.compare, "ca_compare")
    vllm = {t: (load_extension(getattr(args, f"vllm_{t}")),
                getattr(args, f"vllm_isa_{t}"))
            for t in DT if getattr(args, f"vllm_{t}")}
    flush_buf = torch.zeros(2 * _kernel._llc_bytes() // 4, dtype=torch.int32)
    env = dict(machine=machine_type(),
               cpu=sh("lscpu").split("Model name:")[-1].split("\n")[0].strip(),
               threads=torch.get_num_threads(),
               git=sh("git", "-C", ROOT, "rev-parse", "--short", "HEAD"),
               torch=torch.__version__,
               mojo=sh(os.path.join(os.path.dirname(sys.executable), "mojo"), "--version"),
               llc_bytes=_kernel._llc_bytes(),
               vllm={t: [getattr(args, f"vllm_{t}"), v[1]] for t, v in vllm.items()})
    print(json.dumps(env), flush=True)
    rows = []
    for kind, specs in (("flash", FLASH), ("paged", PAGED)):
        for spec in specs:
            for dt in DT:
                if args.only not in spec[0] or args.dtype not in ("both", dt):
                    continue
                torch.manual_seed(0)
                if kind == "flash":
                    fns, out, ref = flash_row(spec, DT[dt], pkgs)
                else:
                    fns, out, ref = paged_row(spec, DT[dt], pkgs, vllm.get(dt))
                timing = measure(fns, lambda: flush_buf.add_(1))
                ours, base = best(timing, "ours"), best(timing, ref)
                row = dict(kind=kind, name=spec[0], dtype=dt, shape=spec[1:], timing=timing,
                           ours=ours, ref=base)
                line = (f"{kind:5s} {spec[0]:26s} {dt}  ours {timing[ours]['median']:9.1f} us"
                        f" ({ours})")
                if base:
                    row["ratio"] = timing[ours]["median"] / timing[base]["median"]
                    # Sanity, not a gate (tests/ hold the gates): both must agree.
                    row["max_diff"] = (out[ours].float() - out[base].float()).abs().max().item()
                    line += (f"  {base} {timing[base]['median']:9.1f}  ratio {row['ratio']:.2f}"
                             f"  diff {row['max_diff']:.1e}")
                if "compare" in pkgs:
                    other = best(timing, "compare")
                    row["vs_compare"] = timing[ours]["median"] / timing[other]["median"]
                    line += f"  vs compare {row['vs_compare']:.2f}"
                print(line, flush=True)
                rows.append(row)
                if args.json:
                    with open(args.json, "w") as f:
                        json.dump(dict(environment=env, rows=rows), f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
