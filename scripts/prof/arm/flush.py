"""Pack and call cost with K/V **evicted** between timed calls.

    python scripts/prof/arm/flush.py --shape 128:8:64 --dtype bf16
    python scripts/prof/arm/flush.py --shape 2048:8:64 --evict-mb 600

`tests/bench_phases.py` (and any loop of our own calls) keeps K, V and the pack
scratch in L2/L3 between samples, so it reports *hot* numbers. The real matrix
interleaves SDPA and vLLM, whose traffic evicts them. This harness walks a
buffer larger than the LLC between timed calls, so each call re-reads K/V from
DRAM, and reports the pack and the full call in both states.

The walk's own cost is *not* subtracted from the pack: the pack is timed from
inside the call (the same `_PackTimer` wrapper `bench_phases.py` uses), so the
two states are directly comparable. The full call's wall time does include the
walk? No -- the walk runs before the timer starts; it only changes the cache
state.
"""

import argparse
import os
import statistics
import sys
import time

import torch

ROOT = "/home/ubuntu/cpu-attention"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("CPU_ATTENTION_NO_FUSE", "1")  # so the pack can be timed
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_phases import _PackTimer  # noqa: E402
from bench_vs_refs import paged_case  # noqa: E402


class Walker:
    """Stream a buffer bigger than the LLC so K/V leave the cache."""

    def __init__(self, mb: int):
        self.buf = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.float32)
        self.buf.fill_(1.0)

    def __call__(self) -> None:
        # A plain sum forces every line to be read; torch will not elide it.
        self.buf.sum()


def measure(fn, walk, rounds):
    """Median of (call) over `rounds`, with `walk` between calls when given."""
    for _ in range(3):
        if walk:
            walk()
        fn()
    calls, packs = [], []
    for _ in range(rounds):
        if walk:
            walk()
        t0 = time.perf_counter()
        pack = fn()
        calls.append((time.perf_counter() - t0) * 1e6)
        packs.append(pack)
    return statistics.median(calls), statistics.median(packs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="128:8:64")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--evict-mb", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args()

    seq, heads, dim = (int(x) for x in args.shape.split(":"))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.set_num_threads(4)
    q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(seq, heads, dim, dtype)
    q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

    timer = _PackTimer()
    timer.install()

    def call() -> float:
        timer.time = 0.0
        attn.flash_attn_func(q3, k3, v3, causal=True)
        return timer.time * 1e6

    walk = Walker(args.evict_mb)
    l3 = 0
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index3/size") as f:
            t = f.read().strip()
        l3 = int(t[:-1]) * (1024 * 1024 if t.endswith("M") else 1024)
    except OSError:
        pass
    walk()
    t0 = time.perf_counter()
    for _ in range(5):
        walk()
    walk_cost = (time.perf_counter() - t0) / 5 * 1e6

    hot = measure(call, None, args.rounds)
    cold = measure(call, walk, args.rounds)

    print(f"{args.shape} {args.dtype}  L3={l3 >> 20} MB  evict={args.evict_mb} MB"
          f"  (walk {walk_cost/1000:.2f} ms)")
    print(f"  {'':10s} {'full':>9s} {'pack':>9s} {'pack%':>7s}")
    for name, (full, pack) in (("hot", hot), ("evicted", cold)):
        print(f"  {name:10s} {full:9.1f} {pack:9.2f} {100*pack/full:6.1f}%")
    print(f"  evicted/hot: full {cold[0]/hot[0]:.2f}x  pack {cold[1]/hot[1]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
