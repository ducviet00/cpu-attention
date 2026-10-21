"""Phase ablation of the AMX attention loop: Q@K^T vs softmax vs P@V.

The kernel has profiling bits (`CPU_ATTENTION_AMX_MODE`, see `_kernel.py`) that
skip one phase; the loop structure is otherwise unchanged, so the difference is
that phase's cost. Round-robin in one process: these boxes throttle, so a
cross-run comparison is meaningless.

    .venv/bin/python tests/bench_amx_ablate.py --dtype bf16
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

MODES = {"full": 0, "noPack": 1, "noAttn": 2, "noAll": 3,
         "noQK": 16, "noSM": 32, "noPV": 64}


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
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
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shapes", default="128:8:64,256:16:128,512:8:64,1024:16:128")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want] or sorted(want)
    print(f"threads={torch.get_num_threads()} {args.dtype}; shares are of the "
          f"attention part (total - pack - fixed)")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dt)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fns = {}
        for name, bit in MODES.items():
            def f(bit=bit):
                K._AMX_MODE_V = bit
                K._dense(q, k, v, dim ** -0.5, True)
            fns[name] = f
        fns["sdpa"] = lambda: sdpa(q, k, v, True)
        m = bench(fns, args.rounds)
        K._AMX_MODE_V = 0
        pack = m["full"] - m["noPack"]      # marginal cost of the pack
        attn = m["full"] - m["noAttn"]      # marginal cost of the attention loop
        fixed = m["noAll"]                  # neither phase: region + marshalling
        # each phase's marginal cost, as a share of the attention part
        qk = m["full"] - m["noQK"]
        sm = m["full"] - m["noSM"]
        pv = m["full"] - m["noPV"]
        print(f"  S={seq:5d} H={heads:2d} D={dim:3d}: total={m['full']:6.0f} "
              f"pack={pack:5.0f} ({100*pack/m['full']:3.0f}%) "
              f"fixed={fixed:4.0f} attn={attn:6.0f} | "
              f"qk={100*qk/attn:3.0f}% sm={100*sm/attn:3.0f}% "
              f"pv={100*pv/attn:3.0f}% resid={100*(1-(qk+sm+pv)/attn):4.0f}% "
              f"o/sdpa {m['full']/m['sdpa']:4.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
