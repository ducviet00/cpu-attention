"""Alternating-round A/B of one shape across N variants in N separate trees/env.

`ab_two_trees.py` hardcodes its shape list; this runs the same protocol for an
arbitrary shape: one short process per variant per round, variants alternated
old→new→…, median of the per-round medians. Used for the tile-geometry question
(`qt>4` with sub=16/dcp=8 vs the coupled (4,8)), where the comparison must be
made at a *fixed* qt so only the geometry differs.

    python scripts/prof/arm/ab_rounds.py --shape 4096:8:64 --dtype bf16 \
        --rounds 5 --variant old_qt8:/tmp/base:8 --variant new_qt8:$PWD:8 \
        --variant new_qt4:$PWD:4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

CHILD = r"""
import json, statistics, sys, time, torch
sys.path.insert(0, sys.argv[1])
import cpu_attention as A
torch.set_num_threads(1)
dt = torch.float32 if sys.argv[2] == "fp32" else torch.bfloat16
S, H, D = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
q = torch.randn(1, S, H, D, dtype=dt)
k = torch.randn(1, S, H, D, dtype=dt)
v = torch.randn(1, S, H, D, dtype=dt)
def once(f):
    t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) * 1e6
def med(f, n=3):
    for _ in range(2): f()
    return statistics.median([once(f) for _ in range(n)])
f = lambda: A.flash_attn_func(q, k, v, causal=True)
ours = med(f)
flops = 4.0 * H * S * S * D * 0.5
print(json.dumps({"ours": ours, "gf": flops / (ours * 1e-6) / 1e9}))
"""


def run(tree: str, dtype: str, shape, qt: str) -> dict:
    env = dict(os.environ)
    if qt:
        env["CPU_ATTENTION_QT"] = qt
    else:
        env.pop("CPU_ATTENTION_QT", None)
    out = subprocess.run(
        [sys.executable, "-c", CHILD, tree, dtype, *map(str, shape)],
        capture_output=True, text=True, env=env,
    )
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(out.stdout + out.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4096:8:64")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--variant", action="append", required=True,
                    help="name:tree:qt")
    args = ap.parse_args()
    shape = tuple(int(x) for x in args.shape.split(":"))
    vs = []
    for spec in args.variant:
        name, tree, qt = spec.split(":")
        vs.append((name, os.path.abspath(tree), qt))
    for name, tree, qt in vs:
        if "CPU_ATTENTION_QT" in os.environ:
            del os.environ["CPU_ATTENTION_QT"]
        print(f"{name}: qt={qt or 'default'} tree={tree}", flush=True)

    samples = {n: [] for n, _, _ in vs}
    for _ in range(args.rounds):
        for name, tree, qt in vs:
            samples[name].append(run(tree, args.dtype, shape, qt))
    med = {n: statistics.median(s["ours"] for s in v) for n, v in samples.items()}
    base = min(med.values())
    print(f"\nS={shape[0]} H={shape[1]} D={shape[2]} {args.dtype}:")
    for n in med:
        gf = statistics.median(s["gf"] for s in samples[n])
        print(f"  {n:12s} {med[n]:9.0f} us  {gf:6.0f} GF/s  "
              f"{base/med[n]:.3f}x of best  ({med[n]/med[list(med)[0]]:.2f}x of {list(med)[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
