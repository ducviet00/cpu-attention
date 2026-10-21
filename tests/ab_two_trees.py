"""A/B two cpu_attention trees (each with its own built ``_C.so``), fairly.

A single process cannot load two ``_C.so`` with different Python wrappers, so
this runs one short process per tree and alternates them: round 1 old, round 1
new, round 2 old, ... and reports the median of the per-round medians. That is
the only comparison that survives a box that throttles -- on the Ryzen a
sustained AVX-512 sweep drops from 4.8 to 1.9 GHz, which is a 2.5x swing in the
absolute number and a 1.6x swing in the *ratio* between two variants.

Why this exists: a worker rewrote the fp32 inner loop, measured a 2.2-4.7x win
against a baseline taken while the box was throttled, and the rewrite was in
fact 1.63x *slower* at head_dim 64 than the kernel it replaced. Blocking that
took one run of this script.

    python tests/ab_two_trees.py --old /path/to/tree_a --new /path/to/tree_b
    python tests/ab_two_trees.py --old . --new ../worktree --dtype fp32 --rounds 9

Both trees must have a built ``cpu_attention/_C.so``. ``--old`` defaults to the
working tree, ``--new`` is required.
"""

import argparse
import json
import statistics
import subprocess
import sys

CHILD = r"""
import json, statistics, sys, time, torch
sys.path.insert(0, sys.argv[1])
import cpu_attention as A
torch.set_num_threads(int(sys.argv[2]))
dt = torch.float32 if sys.argv[3] == "fp32" else torch.bfloat16
S, H, D = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
causal = sys.argv[7] == "1"
q = torch.randn(1, S, H, D, dtype=dt)
k = torch.randn(1, S, H, D, dtype=dt)
v = torch.randn(1, S, H, D, dtype=dt)

def once(f):
    t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) * 1e6

def med(f, n=5):
    for _ in range(2): f()
    return statistics.median([once(f) for _ in range(n)])

ours = med(lambda: A.flash_attn_func(q, k, v, causal=causal))
sdpa = med(lambda: torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    is_causal=causal, scale=D ** -0.5))
flops = 4.0 * H * S * S * D * (0.5 if causal else 1.0)
print(json.dumps({"ours": ours, "sdpa": sdpa, "gflops": flops / (ours * 1e-6) / 1e9}))
"""

SHAPES = [(1024, 8, 64), (2048, 8, 64), (1024, 16, 128), (2048, 16, 128)]


def run(tree, dtype, threads, shape, causal):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, tree, str(threads), dtype,
         *map(str, shape), "1" if causal else "0"],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(out.stdout + out.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=".")
    ap.add_argument("--new", required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "both"], default="both")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--causal", choices=["0", "1"], default="1")
    args = ap.parse_args()
    import os
    old = os.path.abspath(args.old)
    new = os.path.abspath(args.new)
    threads = args.threads or os.cpu_count()
    dtypes = {"fp32": ["fp32"], "bf16": ["bf16"],
              "both": ["fp32", "bf16"]}[args.dtype]
    causal = args.causal == "1"

    print(f"old={old}\nnew={new}\nthreads={threads} rounds={args.rounds}")
    print(f"{'shape':22s} {'dtype':5s} {'old us':>9s} {'new us':>9s} {'new/old':>8s} "
          f"{'old GF':>7s} {'new GF':>7s} {'new/SDPA':>9s}")
    for dtype in dtypes:
        for shape in SHAPES:
            samples = {"old": [], "new": []}
            for _ in range(args.rounds):
                samples["old"].append(run(old, dtype, threads, shape, causal))
                samples["new"].append(run(new, dtype, threads, shape, causal))
            med = {n: statistics.median(s["ours"] for s in v) for n, v in samples.items()}
            gf = {n: statistics.median(s["gflops"] for s in v) for n, v in samples.items()}
            sdpa = statistics.median(s["sdpa"] for s in samples["new"])
            print(f"S={shape[0]:5d} H={shape[1]:2d} D={shape[2]:3d} {dtype:5s} "
                  f"{med['old']:9.0f} {med['new']:9.0f} {med['old']/med['new']:7.2f}x "
                  f"{gf['old']:7.0f} {gf['new']:7.0f} {sdpa/med['new']:8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
