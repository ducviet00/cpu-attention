"""Is short-S slow because of the fixed cost, or because the kernel is slow there?

At S=128/H=8/D=64 bf16 we run at 39% of the bfmmla ceiling against vLLM's 50%,
while at S=4096/H=16/D=128 we are at 85% against their 95%. Two very different
stories fit that: (a) the per-call fixed cost (K/V pack ~6 us, host ~3.5 us, one
fork-join) dominates a 52 us call, or (b) the row kernel itself is inefficient
when the key range is short.

They are separable without touching the kernel: **raise the head count at fixed
seq**. Work scales with heads, the key range does not, and every per-call
overhead is paid once. If cycles/row-key at S=128/H=64 falls to the S=4096 value,
the short-S loss is fixed cost and belongs to the pack/host; if it stays at the
S=128/H=8 value, the kernel is the problem and no amount of dispatch work helps.

    .venv/bin/python tests/bench_scale.py --threads 4

Ratios are `ours/baseline` (`<1` wins); `GFLOP/s` uses exact causal MACs and
`%ceil` is against 4-core `bfmmla` = 835 GFLOP/s (2.0/cycle x 32 flops x 3.26 GHz).
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402

PEAK = 2.0 * 32 * 3.26e9 / 1e9  # GFLOP/s per core


def bench(fn, rounds=9):
    for _ in range(3):
        fn()
    s = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    cores = torch.get_num_threads()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    peak = PEAK * cores

    _burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        _burn @ _burn
    del _burn

    print(f"threads={cores} {args.dtype} peak~{peak:.0f} GFLOP/s "
          f"({PEAK:.0f}/core)")
    print(f"{'S':>6s} {'H':>4s} {'D':>4s} {'us':>10s} {'GFLOP/s':>8s} "
          f"{'%ceil':>6s} {'cyc/rk':>7s} {'row-keys':>10s}")
    shapes = ((128, 8, 64), (128, 32, 64), (128, 64, 64),
              (512, 8, 64), (512, 32, 64),
              (2048, 8, 64), (2048, 32, 64))
    # Round-robin, not one shape at a time: this laptop throttles ~2x under a
    # sustained AVX-512 sweep, so a sequential sweep charges the first shape the
    # cool clock and the last one the hot clock.
    tensors = {s: torch.randn(1, s[0], s[1], s[2], dtype=dt) for s in shapes}
    fns = {s: (lambda q=tensors[s]: attn.flash_attn_func(q, q, q, causal=True))
           for s in shapes}
    for f in fns.values():
        for _ in range(2):
            f()
    samples = {s: [] for s in shapes}
    for _ in range(args.rounds):
        for s in shapes:
            t0 = time.perf_counter()
            fns[s]()
            samples[s].append((time.perf_counter() - t0) * 1e6)
    for seq, heads, dim in shapes:
        us = statistics.median(samples[(seq, heads, dim)])
        rk = heads * seq * (seq + 1) / 2.0
        flops = 4.0 * rk * dim
        gf = flops / (us * 1e-6) / 1e9
        # cycles per row-key, from the measured clock-independent GFLOP/s
        cyc = 4.0 * dim * 2 * 3.26e9 / 1e9 / gf if gf else 0.0
        print(f"{seq:6d} {heads:4d} {dim:4d} {us:10.1f} {gf:8.0f} "
              f"{gf / peak:6.0%} {cyc:7.2f} {rk:10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
