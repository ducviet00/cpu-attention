"""Sweep the AMX kernel's knobs in one process, min-based, order rotated.

`CPU_ATTENTION_AMX_KT` (keys per softmax + P@V super-tile) is read per call by
`_kernel._amx_dense`, so it can be swept inside one process -- which is the only
way to compare on these boxes.

    OMP_NUM_THREADS=4 .venv/bin/python tests/bench_amx_knob.py --kts 64,128,256,384

Why kt matters: the per-work-item working set is `32*kt*4` (logits) +
`32*kt*2` (prob) + `32*head_dim*4` (accumulator). At kt=256 and head_dim 128
that is 32 + 16 + 16 = 64 KB against a 48 KB L1d, so the softmax re-reads the
logits from L2. The sweep says whether that is what binds.
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
from bench_vs_refs import SHAPES  # noqa: E402


def bench(fns, rounds):
    names = list(fns)
    for n in names:
        for _ in range(3):
            fns[n]()
    s = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            t0 = time.perf_counter()
            fns[n]()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: (statistics.median(v), min(v)) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--kts", default="64,128,256,384")
    ap.add_argument("--shapes", default="128:8:64,256:16:128,512:8:64,1024:8:64,1024:16:128,4096:16:128")
    args = ap.parse_args()
    torch.set_num_threads(os.environ.get("OMP_NUM_THREADS") and
                          int(os.environ["OMP_NUM_THREADS"]) or torch.get_num_threads())
    kts = [int(x) for x in args.kts.split(",")]
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want]
    print(f"threads={torch.get_num_threads()} bf16 causal; min us and its ratio "
          f"against kt={kts[0]}")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        fns = {}
        for kt in kts:
            def f(kt=kt):
                K._AMX_KT = kt
                K._dense(q, k, v, dim ** -0.5, True)
            fns[f"kt{kt}"] = f
        ref = None
        m = bench(fns, args.rounds)
        base = m[f"kt{kts[0]}"][1]
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}: "
        line += "  ".join(
            f"kt{n[2:]}={m[n][1]:7.0f}({base/m[n][1]:5.3f})" for n in fns)
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
