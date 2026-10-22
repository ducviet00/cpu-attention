"""Phase ablation for the AMX kernels: where does a call's time go?

    .venv/bin/python tests/bench_amx_phases2.py --dtype bf16 --impl v2
    .venv/bin/python tests/bench_amx_phases2.py --dtype bf16 --impl both

Uses the kernel's own profiling bits (`CPU_ATTENTION_AMX_MODE`, see `V2Job.mode`
/ `AmxJob.mode`) round-robin in one process. `marginal` is
`full - skip_that_phase`, so it is what actually disappears when the phase is
removed (phases overlap, so the shares can sum past 100% at the margins -- read
the marginals, not the shares).
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

PHASES = {
    "full": 0,
    "skip_pack": 1,
    "skip_attn": 2,
    "skip_kpack": 8,
    "skip_vpack": 4,
    "skip_qk": 16,
    "skip_softmax": 32,
    "skip_pv": 64,
    "skip_epilogue": 256,
}


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
            f()
    s = {n: [] for n in fns}
    names = list(fns)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            t0 = time.perf_counter()
            fns[n]()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: (statistics.median(v), min(v)) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--impl", default="v2", choices=["v1", "v2", "both"])
    ap.add_argument("--shapes", default="128:8:64,1024:16:128,4096:16:128")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want] or sorted(want)
    impls = ["v1", "v2"] if args.impl == "both" else [args.impl]
    burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        burn @ burn
    del burn
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dt)
        k = torch.randn(1, seq, heads, dim, dtype=dt)
        v = torch.randn(1, seq, heads, dim, dtype=dt)
        for impl in impls:
            fns = {}
            for name, bit in PHASES.items():
                def f(bit=bit, impl=impl):
                    K._AMX_MODE_V = bit
                    K._AMX_IMPL = impl
                    K._dense(q, k, v, dim ** -0.5, True)
                fns[name] = f
            m = bench(fns, args.rounds)
            K._AMX_MODE_V = 0
            full = m["full"][1]
            print(f"  S={seq:5d} H={heads:2d} D={dim:3d} {args.dtype} {impl}: "
                  f"full {full:8.1f} us (median {m['full'][0]:8.1f})", flush=True)
            for name in PHASES:
                if name == "full":
                    continue
                d = full - m[name][1]
                print(f"      {name:14s} {m[name][1]:8.1f} us   "
                      f"marginal {d:7.1f} us = {100 * d / full:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
