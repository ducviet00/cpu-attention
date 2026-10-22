"""A/B the fp32 V scratch copy (`vpack`) in one process.

`_kernel._dense_plan` copies V into a per-(batch, kv-head) contiguous slab when
`2 * V_bytes >= LLC`, on the theory that the P@V sweep cannot follow a stride of
`heads * head_dim` and the copy pays for itself. That gate was measured on the
Ryzen (32 MB L3); this box has 48 MB, so the shape where it first fires is
4096:16:128 (V = 33.5 MB).

    .venv/bin/python tests/bench_f32_vpack.py --shapes 4096:16:128,2048:16:128

Forces the gate on and off by replacing `_llc_bytes` (it is read only when a plan
is built, so each arm clears the plan cache) and interleaves the arms.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import sdpa  # noqa: E402

BIG = 1 << 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--shapes", default="4096:16:128,2048:16:128,4096:8:64")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    real = K._llc_bytes
    shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
    print(f"threads={torch.get_num_threads()} dtype={args.dtype} "
          f"llc={real()} bytes (plan gate: 2*V >= llc)")
    print(f"{'shape':16s} {'auto us':>9} {'nocopy us':>10} {'copy us':>9} "
          f"{'auto/nocopy':>11} {'auto/copy':>10}  d")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dt)
        k = torch.randn(1, seq, heads, dim, dtype=dt)
        v = torch.randn(1, seq, heads, dim, dtype=dt)
        ref = sdpa(q, k, v, True)
        got = {}

        def arm(name, llc):
            def f():
                K._llc_bytes = llc
                got[name] = attn.flash_attn_func(q, k, v, causal=True)
            return f

        variants = {
            "auto": arm("auto", real),
            "nocopy": arm("nocopy", lambda: BIG),
            "copy": arm("copy", lambda: 1),
        }
        # The LLC gate is read when a plan is built, so each arm needs its own
        # plan: clear the cache *before* the warmup, never inside the timing
        # loop (a cleared cache re-`torch.empty`s multi-megabyte scratch and the
        # page faults land in the measurement -- that alone moved 512:32:128 by
        # 19% between two arms that should be identical).
        for name, fn in variants.items():
            K.clear_scratch()
            fn()
            fn()
        samples = {n: [] for n in variants}
        for _ in range(args.rounds):
            for name, fn in variants.items():
                t0 = time.perf_counter()
                fn()
                samples[name].append((time.perf_counter() - t0) * 1e6)
                # No clear here: the LLC size is part of the plan key, so the
                # plans cannot cross between arms.
        med = {n: statistics.median(s) for n, s in samples.items()}
        tol = 3e-2 if dt is torch.bfloat16 else 2e-4
        dmax = max((got[n].transpose(1, 2).float() - ref.float()).abs().max().item()
                   for n in got)
        print(f"{seq:5d}:{heads}:{dim} {med['auto']:9.1f} {med['nocopy']:10.1f} "
              f"{med['copy']:9.1f} {med['auto'] / med['nocopy']:10.3f}x "
              f"{med['auto'] / med['copy']:9.3f}x  {dmax:.1e}"
              f"{'' if dmax <= tol else ' PARITY-FAIL'}", flush=True)
    K._llc_bytes = real
    K.clear_scratch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
