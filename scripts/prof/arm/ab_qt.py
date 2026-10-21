"""Interleaved A/B of the ARM dense kernel's row-block variants (one process).

    python scripts/prof/arm/ab_qt.py [--shapes 1024:8:64,...] [--rounds 7]
                                     [--variants 4,8] [--dtype bf16]

``_kernel._dense`` reads ``_QT_ENV`` on every call, so the kernel variant can be
switched between samples of the *same* process -- which is the only comparison
these boxes allow (see docs/PROTOCOL.md). On ARM bf16 ``qt`` selects the kernel
itself: 4 is v1 (online softmax every 16 keys, 4 rows), 8 is v2 (the tile's
logits in L1, one softmax per 64 keys, 8 rows). Outputs are compared per variant
against a reference before any timing is reported.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "tests"))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402

SHAPES = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
          (1024, 16, 128), (2048, 16, 128)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="",
                    help="comma separated S:H:D (default: a short sweep)")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--variants", default="4,8")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--causal", default="1")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    causal = args.causal == "1"
    variants = [int(v) for v in args.variants.split(",")]
    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tol = 3e-2 if dtype is torch.bfloat16 else 2e-4

    print(f"threads={torch.get_num_threads()} causal={causal} {args.dtype}")
    print("  shape            " + "".join(f"{'qt=' + str(v):>11s}" for v in variants)
          + "   ratios (vs first)   max|diff|")
    for S, H, D in shapes:
        q = torch.randn(1, S, H, D, dtype=dtype)
        k = torch.randn(1, S, H, D, dtype=dtype)
        v = torch.randn(1, S, H, D, dtype=dtype)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, scale=D ** -0.5)

        outs = {}
        for var in variants:
            K._QT_ENV = str(var)
            outs[var] = attn.flash_attn_func(q, k, v, causal=causal)

        def run(var):
            K._QT_ENV = str(var)
            return attn.flash_attn_func(q, k, v, causal=causal)

        for var in variants:
            run(var)
        samples = {var: [] for var in variants}
        for _ in range(args.rounds):
            for var in variants:
                t0 = time.perf_counter()
                run(var)
                samples[var].append((time.perf_counter() - t0) * 1e6)
        med = {var: statistics.median(s) for var, s in samples.items()}
        base = med[variants[0]]
        diffs = {v: float((outs[v].transpose(1, 2).float() - ref.float()).abs().max())
                 for v in variants}
        line = f"  {S:5d}:{H:3d}:{D:4d}"
        line += "".join(f"{med[v]:11.0f}" for v in variants)
        line += "   " + " ".join(f"{base / med[v]:5.2f}x" for v in variants)
        line += "   " + " ".join(f"{diffs[v]:.1e}" for v in variants)
        print(line, flush=True)
        for v in variants:
            if diffs[v] > tol:
                print(f"    !! parity {diffs[v]:.2e} > {tol:.0e} for variant {v}")
    K._QT_ENV = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
