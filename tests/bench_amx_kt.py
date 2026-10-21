"""A/B the AMX super-tile size (keys per softmax/PV step) on one box.

    .venv/bin/python tests/bench_amx_kt.py --kts 64,128,256
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
    ap.add_argument("--kts", default="64,128,256")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="1024:8:64,4096:8:64,1024:16:128,4096:16:128,128:8:64,256:16:128")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    kts = [int(x) for x in args.kts.split(",")]
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want]
    print(f"threads={torch.get_num_threads()}  bf16 causal us; ratio is vs kt={kts[0]}")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        fns = {}
        for kt in kts:
            def f(kt=kt):
                K._KT_ENV = str(kt)
                K._dense(q, k, v, dim ** -0.5, True)
            fns[f"kt{kt}"] = f
        fns["sdpa"] = lambda: sdpa(q, k, v, True)
        m = bench(fns, args.rounds)
        K._KT_ENV = "0"
        base = m[f"kt{kts[0]}"]
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}: "
        line += " ".join(f"{n}={m[n]:7.0f}({base/m[n]:5.2f}x)" for n in fns if n != "sdpa")
        line += f"  sdpa={m['sdpa']:8.0f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
