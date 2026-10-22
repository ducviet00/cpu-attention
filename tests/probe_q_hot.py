"""Timing probe: how much of a call is the *cold* Q read?

    python tests/probe_q_hot.py --shapes 4096:16:128,256:16:128

Every work item reads its own `qt` query rows once, 4 KB at D=128 / qt=8, and
nothing else in the call touches them -- a candidate for the per-item cost that
is not the panel build. Here the same call runs with two Q tensors:

  cold  a normal contiguous `q`
  hot   `q` with `stride(1) == 0`, so every row reads row 0 and the whole item's
        Q is 256 B that stays in L1. Wrong results by construction; the
        instruction count, the strides handed to the kernel and the output are
        identical.

The difference is the price of the Q read, i.e. the ceiling on what a prefetch
or a wider panel load can recover.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402

SHAPES = [(256, 16, 128), (512, 32, 128), (1024, 16, 128), (2048, 16, 128),
          (4096, 16, 128), (4096, 8, 64)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()
    burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        burn @ burn
    del burn
    print(f"threads={torch.get_num_threads()} us, medians of {args.rounds}")
    print(f"{'shape':>15}{'cold':>10}{'q-hot':>10}{'delta':>8}   what the Q read costs")
    for seq, heads, dim in SHAPES:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        # stride(1) == 0: every query row aliases row 0.
        q_hot = q.as_strided((1, seq, heads, dim), (seq * heads * dim, 0,
                                                   dim, 1))
        variants = {"cold": q, "hot": q_hot}
        samples = {n: [] for n in variants}
        fns = {n: (lambda t=t: attn.flash_attn_func(t, k, v, causal=True))
               for n, t in variants.items()}
        for _ in range(2):
            for n in variants:
                fns[n]()
        for _ in range(args.rounds):
            for n in variants:
                t = time.perf_counter()
                fns[n]()
                samples[n].append((time.perf_counter() - t) * 1e6)
        med = {n: statistics.median(s) for n, s in samples.items()}
        print(f"{seq:5d}:{heads:2d}:{dim:3d}{med['cold']:10.1f}{med['hot']:10.1f}"
              f"{med['cold'] - med['hot']:8.1f}   "
              f"{(med['cold'] - med['hot']) / med['cold'] * 100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
