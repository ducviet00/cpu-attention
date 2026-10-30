"""The benchmark: the Mojo kernel against torch SDPA and vLLM, one process.

    scripts/pin_governor.sh uv run python benchmarks/bench.py --manifest full \
        --json /path/outside/the/repo/<rev>.json [--vllm-bf16 X.so --vllm-fp32 Y.so]

Rules:

* Every Mojo arm is an independently loaded copy of the package
  (benchmarks/arms.py): ``mojo_new`` (this checkout) and ``mojo_ctrl`` (a second,
  independent copy of it: the noise floor). ``--previous DIR`` adds
  ``mojo_previous`` and makes ``mojo_ctrl`` a second copy of DIR instead.
* The reference of a flash row is torch SDPA (``F.scaled_dot_product_attention``);
  of a paged row it is vLLM's CPU attention (``scripts/build_vllm_ref.sh``): the
  same append + paged decode, or the same ``reshape_and_cache`` for store rows.
* Arms are interleaved, the order rotated every round; the minimum decides and
  raw samples + median are kept. Per-arm env is applied outside the timer.
* Decode rows whose per-arm K/V working set exceeds a quarter of the LLC are
  timed cold: before every sample, outside the timer and identically for every
  arm, a read-only sweep of twice the LLC evicts it. Interleaved warm, such rows
  were scored on whichever arm's lines the LLC replacement happened to keep (the
  C4D: 16-33 MB per arm on a 32 MB L3, verdicts 0.9-1.7 for equal kernels); a
  serving KV cache is far larger than the LLC anyway. ``cold`` is in the JSON.
* Each arm stages its own cache from the same logical data; repeated append
  calls rewrite the same slot, so contexts never grow. Allocation is charged
  identically: ours include the pack and the output allocation (except
  ``engine_out`` rows); vLLM's cache is staged and its output preallocated
  outside the timer.
* Every timed arm's output is validated after timing: ours against the test
  gates, a reference against the ``flash_attn_func`` gates (fp32 2e-4, bf16
  3e-2; vLLM's bf16 decode is 1.8-2.3e-3 above the bf16 rounding on the Axion's
  heterogeneous contexts, outside our 1e-3). vLLM on x86 takes no page size
  that is not a multiple of 32; those rows have no reference.
* ``ratio = mojo_new / reference``. A row is a ``win`` below ``1 - band`` with
  ``band = max(0.03, |ctrl/new - 1|)``, a ``loss`` above ``1 + band``, and
  ``unresolved`` in between; ``INVALID`` when ``mojo_new`` fails its gate.
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import arms as A  # noqa: E402
import manifest as M  # noqa: E402
import reference as R  # noqa: E402

DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
ARM64 = platform.machine() in ("aarch64", "arm64")


# ---------------------------------------------------------------------------
# environment record
# ---------------------------------------------------------------------------
def _sh(*cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (result.stdout or result.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def environment(arms_, compiler):
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _sh("git", "-C", _ROOT, "rev-parse", "HEAD"),
        "git_status": _sh("git", "-C", _ROOT, "status", "--porcelain",
                           "cpu_attention"),
        "source_sha256": source_hashes(),
        "mojo": compiler,
        "torch": torch.__version__, "python": sys.version.split()[0],
        "torch_parallel": torch.__config__.parallel_info(),
        "gcc": _sh("gcc", "--version").split("\n")[0],
        "lscpu": _sh("lscpu"), "uname": _sh("uname", "-a"),
        "governor": open(gov).read().strip() if os.path.exists(gov) else None,
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": torch.get_num_threads(),
        "env": {k: v for k, v in os.environ.items()
                 if k.startswith(("CPU_ATTENTION", "OMP", "GOMP", "KMP", "MKL", "VLLM"))},
        "arms": {a.name: a.describe() for a in arms_},
    }


def source_hashes():
    paths = ["pyproject.toml", "uv.lock", "scripts/build.py",
             "benchmarks/bench.py", "benchmarks/arms.py", "benchmarks/manifest.py"]
    paths += [os.path.relpath(p, _ROOT) for p in
              glob.glob(os.path.join(_ROOT, "cpu_attention/_mojo/src/*.mojo"))]
    result = {}
    for path in sorted(paths):
        with open(os.path.join(_ROOT, path), "rb") as f:
            result[path] = hashlib.sha256(f.read()).hexdigest()
    return result


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_arms(fns, envs, rounds, calls, warmup=2, sweep=None):
    """Rotated round-robin; per-call microseconds. ``envs[name]()`` (and
    ``sweep()``, when given) runs before each sample, outside the timer."""
    names = list(fns)
    for n in names:
        envs[n]()
        for _ in range(warmup):
            fns[n]()
    samples = {n: [] for n in names}
    for r in range(rounds):
        k = r % len(names)
        for n in names[k:] + names[:k]:
            if sweep is not None:
                sweep()
            envs[n]()
            fn = fns[n]
            t0 = time.perf_counter_ns()
            for _ in range(calls):
                fn()
            samples[n].append((time.perf_counter_ns() - t0) / 1000.0 / calls)
    return samples


def stats(xs):
    s = sorted(xs)
    return {"min": s[0], "median": s[len(s) // 2], "n": len(s)}


# ---------------------------------------------------------------------------
# flash_attn_func rows
# ---------------------------------------------------------------------------
def flash_inputs(r, dtype):
    b, sq, sk, h, hkv, d = r["b"], r["sq"], r["sk"], r["h"], r["hkv"], r["d"]
    if r["layout"] == "fused":
        assert sq == sk and h == hkv
        buf = torch.randn(b, sq, 3 * h * d, dtype=dtype)
        q, k, v = (buf[..., i * h * d:(i + 1) * h * d].view(b, sq, h, d)
                   for i in range(3))
        return q, k, v
    return (torch.randn(b, sq, h, d, dtype=dtype),
            torch.randn(b, sk, hkv, d, dtype=dtype),
            torch.randn(b, sk, hkv, d, dtype=dtype))


def flash_row(r, arms_, vllm, args):
    dtype = DT[r["dtype"]]
    q, k, v = flash_inputs(r, dtype)
    scale = r["d"] ** -0.5
    causal = r["causal"]
    out = {}
    fns = {}
    for a in arms_:
        def f(a=a):
            out[a.name] = a.pkg.flash_attn_func(q, k, v, scale, causal)
        fns[a.name] = f
    qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    mask = None
    if causal and r["sq"] != r["sk"]:
        mask = (torch.arange(r["sk"])[None, :] <=
                torch.arange(r["sq"])[:, None] + (r["sk"] - r["sq"]))
    gqa = r["h"] != r["hkv"]

    def sdpa():
        out["sdpa"] = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=mask, is_causal=causal and mask is None,
            scale=scale, enable_gqa=gqa).transpose(1, 2)
    fns["sdpa"] = sdpa
    vl = vllm.get(r["dtype"]) if vllm else None
    if vl is not None and r["sq"] == r["sk"] and r["b"] == 1:
        fns["vllm"] = vllm_flash(vl, r, q, k, v, scale, out)
    ref = R.ref_attention(q, k, v, causal, scale=scale)
    tol = 3e-2 if r["dtype"] == "bf16" else 2e-4

    def check(name):
        got = out[name].float()
        err = (got - ref).abs().max().item()
        if not err <= tol and r["dtype"] == "fp32":
            truth = R.ref_attention(q, k, v, causal, scale=scale,
                                    dtype=torch.float64)
            if (ref.double() - truth).abs().max().item() > tol:
                err = (got.double() - truth).abs().max().item()
        return err, err <= tol
    return fns, check, args.rounds, 1


def vllm_flash(vl, r, q, k, v, scale, out):
    mod, isa = vl
    seq, h, hkv, d, dtype = r["sq"], r["h"], r["hkv"], r["d"], q.dtype
    bs = 64
    nb = (seq + bs - 1) // bs
    qf = q.reshape(seq, h, d).contiguous()
    kc = torch.zeros(nb, hkv, bs, d, dtype=dtype)
    vc = torch.zeros_like(kc)
    mod.cpu_attn_reshape_and_cache(k.reshape(seq, hkv, d).contiguous(),
                                   v.reshape(seq, hkv, d).contiguous(), kc, vc,
                                   torch.arange(seq, dtype=torch.int64), isa)
    lens = torch.tensor([seq], dtype=torch.int32)
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    meta = mod.get_scheduler_metadata(1, h, hkv, d, lens, dtype, qsl,
                                      r["causal"], -1, isa, False, None, "auto")
    o = torch.empty_like(qf)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)

    def f():
        mod.cpu_attention_with_kv_cache(qf, kc, vc, o, qsl, lens, scale,
                                        r["causal"], None, -1, bt, 0.0, meta,
                                        None, None, 1.0, 1.0, "auto")
        out["vllm"] = o.view(1, seq, h, d)
    return f


# ---------------------------------------------------------------------------
# flash_attn_with_kvcache / engine rows
# ---------------------------------------------------------------------------
def paged_inputs(r, dtype, page):
    b, h, hkv, d = r["b"], r["h"], r["hkv"], r["d"]
    ctxs = r["ctx"] if isinstance(r["ctx"], list) else [r["ctx"]] * b
    npages = [(c + page - 1) // page for c in ctxs]
    width = max(npages)
    bt = torch.zeros(b, width, dtype=torch.int32)
    nxt = 0
    for i, n in enumerate(npages):
        bt[i, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32)
        nxt += n
    lens = torch.tensor(ctxs, dtype=torch.int32)
    kall = [torch.randn(c, hkv, d, dtype=dtype) for c in ctxs]
    vall = [torch.randn(c, hkv, d, dtype=dtype) for c in ctxs]
    q = torch.randn(b, h, d, dtype=dtype)
    slots = torch.tensor([int(bt[i, (c - 1) // page]) * page + (c - 1) % page
                          for i, c in enumerate(ctxs)], dtype=torch.int64)
    allslots = torch.cat([bt[i, torch.arange(c) // page].long() * page +
                          torch.arange(c) % page for i, c in enumerate(ctxs)])
    return dict(q=q, kall=kall, vall=vall, bt=bt, lens=lens, slots=slots,
                allslots=allslots, shape=(nxt, hkv, page, d),
                kn=torch.stack([x[-1] for x in kall]),
                vn=torch.stack([x[-1] for x in vall]))


def paged_row(r, arms_, vllm, args, page):
    dtype = DT[r["dtype"]]
    x = paged_inputs(r, dtype, page)
    q, bt, lens, slots = x["q"], x["bt"], x["lens"], x["slots"]
    kn, vn = x["kn"], x["vn"]
    scale = r["d"] ** -0.5
    kflat, vflat = torch.cat(x["kall"]), torch.cat(x["vall"])
    out, fns, caches = {}, {}, {}
    entry, op = r["entry"], r["op"]
    stage_k, stage_v = kflat, vflat
    if op == "self":
        stage_k, stage_v = kflat.clone(), vflat.clone()
        last = lens.long().cumsum(0) - 1
        stage_k[last] = float("nan")
        stage_v[last] = float("nan")
    for a in arms_:
        kc = torch.zeros(x["shape"], dtype=dtype)
        vc = torch.zeros_like(kc)
        a.enter()
        a.pkg.reshape_and_cache(stage_k, stage_v, kc, vc, x["allslots"])
        if entry == "store":
            kc.fill_(float("nan"))
            vc.fill_(float("nan"))
        caches[a.name] = (kc, vc)
        o = torch.empty_like(q)
        p = a.pkg

        if entry == "store":
            def f(p=p, kc=kc, vc=vc):
                p.reshape_and_cache(kflat, vflat, kc, vc, x["allslots"])
        elif entry == "api":
            q4 = q.unsqueeze(1)
            if op == "self":
                k4, v4, pre = kn.unsqueeze(1), vn.unsqueeze(1), lens - 1

                def f(p=p, kc=kc, vc=vc, n=a.name):
                    out[n] = p.flash_attn_with_kvcache(
                        q4, kc, vc, k=k4, v=v4, cache_seqlens=pre,
                        page_table=bt, softmax_scale=scale).squeeze(1)
            else:
                def f(p=p, kc=kc, vc=vc, n=a.name):
                    out[n] = p.flash_attn_with_kvcache(
                        q4, kc, vc, cache_seqlens=lens, page_table=bt,
                        softmax_scale=scale).squeeze(1)
        else:
            dst = o if entry == "engine_out" else None
            if op == "self":
                def f(p=p, kc=kc, vc=vc, n=a.name, dst=dst):
                    out[n] = p.self_attention(q, kn, vn, kc, vc, slots, bt,
                                              lens, scale, out=dst)
            else:
                def f(p=p, kc=kc, vc=vc, n=a.name, dst=dst):
                    out[n] = p.cross_attention(q, kc, vc, bt, lens, scale,
                                               out=dst)
        fns[a.name] = f
    vl = vllm.get(r["dtype"]) if vllm else None
    if vl is not None:
        try:
            fns["vllm"] = vllm_paged(vl, r, x, q, kn, vn, scale, op, out)
        except Exception as e:  # noqa: BLE001
            print(f"  vllm unavailable for this row: {e}", flush=True)

    kref = torch.nn.utils.rnn.pad_sequence(x["kall"], batch_first=True)
    vref = torch.nn.utils.rnn.pad_sequence(x["vall"], batch_first=True)
    ref = R.ref_decode(q, kref.transpose(1, 2), vref.transpose(1, 2),
                       scale=scale, lens=lens)
    def store_check(name):
        kc, vc = caches[name]
        slot = x["allslots"].view(-1, 1, 1)
        head = torch.arange(r["hkv"]).view(1, -1, 1)
        dim = torch.arange(r["d"]).view(1, 1, -1)
        base = ((slot // page) * r["hkv"] + head) * page * r["d"]
        off = slot % page
        ko = ((dim // 2 * page + off) * 2 + dim % 2
              if r["dtype"] == "bf16" else dim * page + off)
        vo = ((off // 2 * r["d"] + dim) * 2 + off % 2
              if ARM64 and r["dtype"] == "bf16" else off * r["d"] + dim)
        ek = (kc.flatten()[base + ko].float() - kflat.float()).abs().max().item()
        ev = (vc.flatten()[base + vo].float() - vflat.float()).abs().max().item()
        error = max(ek, ev)
        return error, ek == 0 and ev == 0

    def check(name):
        got = out[name].float()
        if name == "vllm":  # a reference must be sane, not meet our gate
            err = (got - ref).abs().max().item()
            return err, err <= (2e-4 if r["dtype"] == "fp32" else 3e-2)
        if r["dtype"] == "fp32":
            err = (got - ref).abs().max().item()
            if err > 1e-6:
                truth = R.ref_decode(q, kref.transpose(1, 2),
                                     vref.transpose(1, 2), dtype=torch.float64,
                                     scale=scale, lens=lens)
                # The fp32 reference is itself up to ~9e-7 from the exact
                # answer, so a kernel 2e-7 from the truth can read 1.0e-6 from
                # it (seen on het contexts): the
                # gate is judged against fp64 once it is exceeded.
                err = (got.double() - truth).abs().max().item()
            return err, err <= 1e-6
        # bf16: 1e-3 above the bf16 output-rounding floor (the literal absolute
        # 1e-3 is below that floor; reference.bf16_excess), the tests' gate.
        err = R.bf16_excess(got, ref)
        return err, err <= 1e-3

    def check_all(names):
        if entry == "store":
            # vLLM's cache layout is its own: its store is judged by decoding
            # from it, ours by reading every staged element back.
            res = {n: store_check(n) for n in names if n != "vllm"}
            if "vllm" in names:
                vllm_decode()
                res["vllm"] = check("vllm")
            return res
        return {n: check(n) for n in names}
    vllm_decode = getattr(fns.get("vllm"), "decode", None)
    rounds = args.paged_rounds
    ws = 2 * kflat.numel() * kflat.element_size()  # one arm's K + V bytes
    return fns, check_all, rounds, args.block_calls, ws


def vllm_paged(vl, r, x, q, kn, vn, scale, op, out):
    mod, isa = vl
    b, h, hkv, d = r["b"], r["h"], r["hkv"], r["d"]
    kc = torch.zeros(x["shape"], dtype=q.dtype)
    vc = torch.zeros_like(kc)
    mod.cpu_attn_reshape_and_cache(torch.cat(x["kall"]), torch.cat(x["vall"]),
                                   kc, vc, x["allslots"], isa)
    qsl = torch.arange(b + 1, dtype=torch.int32)
    meta = mod.get_scheduler_metadata(b, h, hkv, d, x["lens"], q.dtype, qsl,
                                      True, -1, isa, False, None, "auto")
    o = torch.empty_like(q)
    bt, lens, slots = x["bt"], x["lens"], x["slots"]
    kflat, vflat = torch.cat(x["kall"]), torch.cat(x["vall"])

    def decode():
        mod.cpu_attention_with_kv_cache(q, kc, vc, o, qsl, lens, scale, True,
                                        None, -1, bt, 0.0, meta, None, None,
                                        1.0, 1.0, "auto")
        out["vllm"] = o

    if r["entry"] == "store":
        def f():
            mod.cpu_attn_reshape_and_cache(kflat, vflat, kc, vc, x["allslots"], isa)
        f.decode = decode
        decode()
        return f

    def f():
        if op == "self":
            mod.cpu_attn_reshape_and_cache(kn, vn, kc, vc, slots, isa)
        decode()
    f()
    return f


# ---------------------------------------------------------------------------
def build_arms(args):
    factories = []
    want = args.arms.split(",")
    if "mojo_new" in want:
        factories.append(lambda: A.current("mojo_new"))
    if "mojo_vec" in want:  # Mojo with the AMX policy off: the vector policy only
        factories.append(lambda: A.current("mojo_vec", {"CPU_ATTENTION_ISA": "vec"}))
    if "mojo_ctrl" in want:
        if args.previous:
            factories.append(lambda: A.snapshot("mojo_ctrl", args.previous))
        else:
            factories.append(lambda: A.current("mojo_ctrl"))
    if args.previous:
        factories.append(lambda: A.snapshot("mojo_previous", args.previous))
    random.Random(args.load_seed).shuffle(factories)
    return [factory() for factory in factories]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="full",
                    choices=("full", "historical", "extended"))
    ap.add_argument("--only", default="", help="substring filter on row labels")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "both"), default="both")
    ap.add_argument("--arms", default="mojo_new,mojo_ctrl,sdpa,vllm")
    ap.add_argument("--previous", default="", metavar="PACKAGE_DIR",
                    help="saved previous package (glue and libraries); load it twice "
                         "as mojo_previous and mojo_ctrl for build comparisons")
    ap.add_argument("--vllm-bf16", default=os.environ.get("VLLM_REF_BF16", ""),
                    help="vLLM build for bf16 rows (AMX build on AMX boxes)")
    ap.add_argument("--vllm-fp32", default=os.environ.get("VLLM_REF_FP32", ""))
    ap.add_argument("--vllm-isa-bf16", default="vec", help="amx on AMX boxes")
    ap.add_argument("--vllm-isa-fp32", default="vec")
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--paged-rounds", type=int, default=41)
    ap.add_argument("--block-calls", type=int, default=1)
    ap.add_argument("--page", type=int, default=16 if ARM64 else 32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--load-seed", type=int, default=0,
                    help="shuffle module load and cache allocation order reproducibly")
    ap.add_argument("--json", default="")
    ap.add_argument("--resume", action="store_true",
                    help="continue --json: skip the rows it already holds and keep them")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    mojo = os.path.join(os.path.dirname(sys.executable), "mojo")
    compiler = _sh(mojo, "--version") if os.path.exists(mojo) else "?"
    arms_ = build_arms(args)
    vllm = {}
    if "vllm" in args.arms:
        for tag in ("bf16", "fp32"):
            path = getattr(args, f"vllm_{tag}")
            mod = R.load_so(path) if path else None
            if mod is not None:
                vllm[tag] = (mod, getattr(args, f"vllm_isa_{tag}"))
    env = environment(arms_, compiler)
    env.update(argv=sys.argv, vllm={k: [getattr(args, f"vllm_{k}"), v[1]]
                                    for k, v in vllm.items()})
    print(json.dumps({k: env[k] for k in ("git", "mojo", "torch", "threads",
                                          "governor", "arms")}, indent=1),
          flush=True)
    rows = [r for r in M.rows(args.manifest)
            if args.dtype in ("both", r["dtype"]) and args.only in M.label(r)]
    results = []
    if args.resume and args.json and os.path.exists(args.json):
        with open(args.json) as f:
            results = json.load(f)["rows"]
        done = {x["label"] for x in results}
        rows = [r for r in rows if M.label(r) not in done]
        print(f"resuming: {len(results)} rows kept, {len(rows)} to run", flush=True)
    R.burn()
    torch.manual_seed(0)
    llc = arms_[0].pkg._kernel._llc_bytes()
    evict = torch.ones((2 * llc) // 8, dtype=torch.int64)
    for r in rows:
        label = M.label(r)
        cold = False
        if r["kind"] == "flash":
            fns, check, rounds, calls = flash_row(r, arms_, vllm, args)
            checks = None
        else:
            page = r["page"] or args.page
            fns, check_all, rounds, calls, ws = paged_row(r, arms_, vllm, args, page)
            check, checks = None, check_all
            cold = ws > llc // 4
        envs = {n: next((a.enter for a in arms_ if a.name == n), A.clear_env)
                for n in fns}
        if "sdpa" not in args.arms:
            fns.pop("sdpa", None)
        samples = time_arms(fns, envs, rounds, calls,
                            sweep=(lambda: evict.sum()) if cold else None)
        st = {n: stats(s) for n, s in samples.items()}
        valid = (checks(list(fns)) if checks else
                 {n: check(n) for n in fns})
        row = dict(r, label=label, rounds=rounds, block_calls=calls, cold=cold,
                   min_us={n: s["min"] for n, s in st.items()},
                   median_us={n: s["median"] for n, s in st.items()},
                   samples_us=samples, err={n: v[0] for n, v in valid.items()},
                   valid={n: v[1] for n, v in valid.items()},
                   loadavg=os.getloadavg())
        line = f"{label:48s}"
        base = "sdpa" if r["kind"] == "flash" else "vllm"
        if "mojo_new" in st:
            line += f" new {st['mojo_new']['min']:10.1f}"
        if base in st and "mojo_new" in st:
            ratio = st["mojo_new"]["min"] / st[base]["min"]
            control_peer = "mojo_previous" if args.previous else "mojo_new"
            spread = (abs(st["mojo_ctrl"]["min"] / st[control_peer]["min"] - 1)
                      if "mojo_ctrl" in st else 0.0)
            band = max(0.03, spread)
            verdict = ("win" if ratio < 1 - band else
                       "loss" if ratio > 1 + band else "unresolved")
            if not valid["mojo_new"][1]:
                verdict = "INVALID"
            row.update(baseline=base, ratio=ratio, ctrl_spread=spread, band=band,
                       verdict=verdict)
            line += (f" {base} {st[base]['min']:10.1f} ratio {ratio:5.3f} "
                     f"ctrl {spread:5.3f} {verdict}")
        if args.previous and "mojo_new" in st:
            row["new/previous"] = st["mojo_new"]["min"] / st["mojo_previous"]["min"]
            line += f" /previous {row['new/previous']:5.3f}"
        if r["kind"] == "flash" and "vllm" in st and "mojo_new" in st:
            row["new/vllm"] = st["mojo_new"]["min"] / st["vllm"]["min"]
            line += f" /vllm {row['new/vllm']:5.3f}"
        bad = [n for n, v in valid.items() if not v[1]]
        if bad:
            line += f"  INVALID {bad}"
        if cold:
            line += "  cold"
        print(line, flush=True)
        results.append(row)
        # The row's closures reference its (up to 270 MB per arm) K/V caches in
        # reference cycles; without this the process reached 8 GB by row 156.
        del fns, check, checks, samples, valid
        gc.collect()
        if args.json:  # rewritten every row: a spot preemption keeps the rest
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            pending = args.json + ".tmp"
            with open(pending, "w") as f:
                json.dump({"environment": env, "rows": results}, f)
            os.replace(pending, args.json)
    summarize(results)
    return 0


def summarize(rows):
    v = [r.get("verdict") for r in rows if "verdict" in r]
    print(f"\n{len(v)} rows: {v.count('win')} win, {v.count('unresolved')} "
          f"unresolved, {v.count('loss')} loss, {v.count('INVALID')} invalid")
    for r in rows:
        if r.get("verdict") not in ("win", None):
            print(f"  {r['verdict']:10s} {r['label']}  ratio {r['ratio']:.3f} "
                  f"vs {r['baseline']}")


if __name__ == "__main__":
    raise SystemExit(main())
