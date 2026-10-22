"""A/B the two AMX kernels (v1 `dense_amx.mojo`, v2 `dense_amx_v2.mojo`).

    .venv/bin/python tests/bench_amx_v2.py                    # parity + timing
    .venv/bin/python tests/bench_amx_v2.py --dtype bf16 --rounds 9
    .venv/bin/python tests/bench_amx_v2.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench_amx.so --isa amx

Everything is interleaved in one process (`time_rounds` rotates the arm order
every round, which the project measured as worth 3.4% on identical code) and the
ratios are min-based: interference can only make a sample slower, so the minimum
is the honest one on a shared box. `v2/v1 < 1` means v2 is faster; so does
`v2/vllm < 1`.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

SHAPES = [
    (128, 8, 64),
    (512, 8, 64),
    (1024, 8, 64),
    (2048, 8, 64),
    (4096, 8, 64),
    (256, 16, 128),
    (1024, 16, 128),
    (2048, 16, 128),
    (4096, 16, 128),
    (512, 32, 128),
    (1024, 8, 32),
]


def sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=q.shape[-1] ** -0.5,
    )


def load_vllm(path):
    if not path or not os.path.exists(path):
        return None
    name = os.path.basename(path).rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_rounds(variants, rounds):
    """Median and min per variant, interleaved, order rotated every round."""
    for fn in variants.values():
        for _ in range(2):
            fn()
    samples = {name: [] for name in variants}
    names = list(variants)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            t0 = time.perf_counter()
            variants[name]()
            samples[name].append((time.perf_counter() - t0) * 1e6)
    return ({n: statistics.median(v) for n, v in samples.items()},
            {n: min(v) for n, v in samples.items()})


def paged_case(seq, heads, dim, dtype, block=128):
    q = torch.randn(seq, heads, dim, dtype=dtype)
    k = torch.randn(seq, heads, dim, dtype=dtype)
    v = torch.randn(seq, heads, dim, dtype=dtype)
    blocks = (seq + block - 1) // block
    kc = torch.zeros(blocks, heads, block, dim, dtype=dtype)
    vc = torch.zeros_like(kc)
    bt = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
    pos = torch.arange(seq, dtype=torch.int64)
    slots = pos // block * block + pos % block
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    lens = torch.tensor([seq], dtype=torch.int32)
    return q, k, v, kc, vc, bt, slots, qsl, lens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32", "both"], default="both")
    ap.add_argument("--causal", choices=["0", "1", "both"], default="1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "amx"))
    ap.add_argument("--control", action="store_true",
                    help="add a duplicate v2 arm: ctrl/v2 is the noise floor")
    ap.add_argument("--kt", default="", help="CPU_ATTENTION_AMX_KT")
    args = ap.parse_args()
    if args.kt:
        os.environ["CPU_ATTENTION_AMX_KT"] = args.kt
    if args.threads:
        torch.set_num_threads(args.threads)
    if not K.amx_available() or "v2" not in K._AMX_ENTRIES:
        print("AMX v2 kernel unavailable here: skipped")
        return 0
    vllm = load_vllm(args.vllm)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}[args.dtype]
    causals = {"0": [False], "1": [True], "both": [True, False]}[args.causal]
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in SHAPES if s in want]
    bad = 0
    burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        burn @ burn
    del burn

    print(f"threads={torch.get_num_threads()} vllm={args.vllm or '-'} "
          f"isa={args.isa} kt={os.environ.get('CPU_ATTENTION_AMX_KT', '-')}")
    for causal in causals:
        print(f"\ncausal={causal}   us; v2/v1 and v2/vllm are min-based, "
              f"<1 = v2 faster")
        for seq, heads, dim in shapes:
            for dtype in dtypes:
                q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                    seq, heads, dim, dtype)
                out = torch.empty_like(q)
                q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
                res = {}

                def arm(name):
                    def f():
                        K._AMX_IMPL = name
                        res[name] = attn.flash_attn_func(q3, k3, v3, causal=causal)
                    return f

                variants = {"v1": arm("v1"), "v2": arm("v2")}
                if args.control:
                    variants["ctrl"] = arm("v2")
                if vllm is not None:
                    vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                    meta = vllm.get_scheduler_metadata(
                        len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                        -1, args.isa, False, None, "auto")
                    variants["vllm"] = (
                        lambda: vllm.cpu_attention_with_kv_cache(
                            q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None,
                            -1, bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))
                med, mn = time_rounds(variants, args.rounds)
                ref = sdpa(q3, k3, v3, causal)
                tol = 3e-2 if dtype == torch.bfloat16 else 2e-4
                d = {}
                for name in ("v1", "v2"):
                    d[name] = (res[name].transpose(1, 2).float()
                               - ref.float()).abs().max().item()
                ok = d["v2"] <= tol
                bad += 0 if ok else 1
                flops = 4.0 * heads * seq * seq * dim * (0.5 if causal else 1.0)
                tag = "bf16" if dtype == torch.bfloat16 else "fp32"
                line = (f"  S={seq:5d} H={heads:2d} D={dim:3d} {tag}: "
                        f"v1 {med['v1']:8.0f}  v2 {med['v2']:8.0f}"
                        f"  v2/v1 {mn['v2'] / mn['v1']:5.2f}")
                if "vllm" in med:
                    line += (f"  vllm {med['vllm']:8.0f}"
                             f"  v2/vllm {mn['v2'] / mn['vllm']:5.2f}"
                             f"  v1/vllm {mn['v1'] / mn['vllm']:5.2f}")
                if "ctrl" in med:
                    line += f"  ctrl {mn['ctrl'] / mn['v2']:5.2f}"
                line += (f"  [{flops / med['v2'] * 1e-6 / 1e9:4.0f} GFLOP/s]"
                         f"  d1 {d['v1']:.1e} d2 {d['v2']:.1e}"
                         f"{'' if ok else '  PARITY-FAIL'}")
                print(line, flush=True)
    print(f"{'FAIL' if bad else 'OK'}: {bad} parity failures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
