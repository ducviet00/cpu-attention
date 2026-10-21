"""Phase ablation of the AMX attention loop: Q@K^T vs softmax vs P@V.

Interleaved, one process. Numbers come from the C kernel's profiling bits
(`CPU_ATTENTION_AMX_MODE=abl1` = no Q@K^T, `abl2` = no softmax, `abl4` = no
P@V), so the loop structure is unchanged -- only the phase is skipped.

    .venv/bin/python tests/bench_amx_ablate.py
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

    print(f"threads={torch.get_num_threads()}   bf16 causal, us; "
          "shares are of the attention part (full - pack)")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        modes = {"full": "", "noqk": "abl1", "nosm": "abl2", "nopv": "abl4",
                 "pack": "attn", "sdpa": None}
        fns = {}
        for name, mode in modes.items():
            if mode is None:
                fns[name] = lambda: sdpa(q, k, v, True)
            else:
                def f(mode=mode):
                    K._AMX_MODE = mode
                    K._dense(q, k, v, dim ** -0.5, True)
                fns[name] = f
        m = bench(fns, args.rounds)
        K._AMX_MODE = ""
        attn = m["full"] - m["pack"]
        qk = attn - (m["noqk"] - m["pack"])
        sm = attn - (m["nosm"] - m["pack"])
        pv = attn - (m["nopv"] - m["pack"])
        print(f"  S={seq:5d} H={heads:2d} D={dim:3d}: full={m['full']:7.0f} "
              f"pack={m['pack']:6.0f} attn={attn:7.0f} | "
              f"qk={qk:6.0f} ({100*qk/attn:3.0f}%) "
              f"sm={sm:6.0f} ({100*sm/attn:3.0f}%) "
              f"pv={pv:6.0f} ({100*pv/attn:3.0f}%) "
              f"resid={attn-qk-sm-pv:5.0f} | o/sdpa {m['full']/m['sdpa']:4.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
