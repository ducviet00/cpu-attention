"""Tiny debug harness for the ARM dense kernels: Sq/Sk/H/D sweep vs SDPA.

    python scripts/prof/arm/dbg.py [--dtype fp32] [--causal 1]

Prints max|diff| for a set of small shapes so a new kernel's failure can be
localised (single key, single tile, multi-tile, GQA) instead of guessed at.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/home/ubuntu/cpu-attention")
import cpu_attention as attn  # noqa: E402


def one(Sq, Sk, H, D, dtype, causal, Hk=None):
    Hk = Hk or H
    q = torch.randn(1, Sq, H, D, dtype=dtype)
    k = torch.randn(1, Sk, Hk, D, dtype=dtype)
    v = torch.randn(1, Sk, Hk, D, dtype=dtype)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=D ** -0.5)
    out = attn.flash_attn_func(q, k, v, causal=causal).transpose(1, 2)
    d = (out.float() - ref.float()).abs()
    return float(d.max()), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--causal", default="1")
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    causal = args.causal == "1"
    # Sq == Sk only: with a non-square causal case SDPA aligns top-left and
    # flash-attn bottom-right, so those shapes cannot be compared this way.
    for Sq, Sk, H, D, Hk in [
        (2, 2, 1, 64, 1), (3, 3, 1, 64, 1), (4, 4, 1, 64, 1),
        (5, 5, 1, 64, 1), (8, 8, 1, 64, 1), (9, 9, 1, 64, 1),
        (16, 16, 1, 64, 1), (17, 17, 1, 64, 1), (33, 33, 1, 64, 1),
        (64, 64, 1, 64, 1), (65, 65, 1, 64, 1), (100, 100, 1, 64, 1),
        (9, 9, 1, 128, 1), (9, 9, 1, 32, 1), (100, 100, 2, 128, 1),
    ]:
        d, _ = one(Sq, Sk, H, D, dtype, causal, Hk)
        print(f"  Sq={Sq:4d} Sk={Sk:4d} H={H} Hk={Hk} D={D:3d}: max|diff|={d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
