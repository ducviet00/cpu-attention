"""Interleaved sweep of `qt` (query rows per work item) in one process.

`CPU_ATTENTION_QT` is read per call, so this monkeypatches `_kernel._QT_ENV` and
times every arm round-robin in one process -- separate runs are not comparable
on these boxes (they throttle 2-2.5x).

    .venv/bin/python tests/bench_qt.py --dtype fp32 --qts 4,6,8,10
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

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}


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
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--qts", default="4,6,8,10")
    ap.add_argument("--causal", default="1")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="1024:8:64,2048:8:64,4096:8:64,512:8:64,1024:16:128,2048:16:128,4096:16:128,512:32:128")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    dt = DT[args.dtype]
    causal = args.causal == "1"
    qts = [int(x) for x in args.qts.split(",")]
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want]
    print(f"dtype={args.dtype} threads={torch.get_num_threads()} causal={causal} "
          f"us; win = fastest per row")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dt)
        k = torch.randn(1, seq, heads, dim, dtype=dt)
        v = torch.randn(1, seq, heads, dim, dtype=dt)
        ref = sdpa(q, k, v, causal)
        fns = {}
        for qt in qts:
            def f(qt=qt):
                K._QT_ENV = str(qt)
                K._dense(q, k, v, dim ** -0.5, causal)
            fns[f"qt{qt}"] = f
        m = bench(fns, args.rounds)
        K._QT_ENV = None
        best = min(m.values())
        # parity of the fastest arm, so a broken qt cannot win
        K._QT_ENV = str(next(n for n, t in m.items() if t == best)[2:])
        got = K._dense(q, k, v, dim ** -0.5, causal)
        K._QT_ENV = None
        d = (got.float() - ref.transpose(1, 2).float()).abs().max().item()
        tol = 3e-2 if dt == torch.bfloat16 else 2e-4
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}: "
        line += " ".join(f"{n}={m[n]:8.0f}" for n in fns)
        line += f"   best={best:8.0f} (x{max(m.values())/best:4.2f} over worst)"
        line += f"  d={d:.1e}{'' if d <= tol else ' PARITY FAIL'}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
