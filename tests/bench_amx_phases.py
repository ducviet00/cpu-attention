"""Split one AMX dense call into the K/V pack and the row kernel.

Uses the C kernel's profiling bits (bit 16 = skip the pack, bit 17 = skip the
attention), interleaved in one process like every other measurement here.

    CPU_ATTENTION_AMX_MODE unset  -> full
    .venv/bin/python tests/bench_amx_phases.py
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import SHAPES, sdpa  # noqa: E402


def call(q3, k3, v3, causal):
    return K._dense(q3, k3, v3, q3.shape[-1] ** -0.5, causal)


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(2):
            f()
    s = {n: [] for n in fns}
    for _ in range(rounds):
        for n, f in fns.items():
            t0 = time.perf_counter()
            f()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: statistics.median(v) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in shapes if s in want]

    print(f"threads={torch.get_num_threads()}   bf16, causal, us")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        variants = {}
        for mode, name in (("", "full"), ("attn", "pack"), ("pack", "kernel")):
            def f(mode=mode):
                K._AMX_MODE = mode
                call(q, k, v, True)
            variants[name] = f
        variants["sdpa"] = lambda: sdpa(q, k, v, True)
        m = bench(variants, args.rounds)
        K._AMX_MODE = ""
        print(f"  S={seq:5d} H={heads:2d} D={dim:3d}: full={m['full']:8.0f} "
              f"pack={m['pack']:7.0f} ({100*m['pack']/m['full']:4.1f}%) "
              f"kernel={m['kernel']:8.0f} "
              f"sum-check={m['pack']+m['kernel']-m['full']:6.0f} "
              f"sdpa={m['sdpa']:8.0f}  o/sdpa {m['full']/m['sdpa']:4.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
