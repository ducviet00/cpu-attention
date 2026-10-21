"""Turn perf_matrix.sh + perf_baseline.sh CSVs into the report tables.

    python scripts/prof/x86/analyze.py /tmp/perf.csv /tmp/base.csv

`perf stat` wraps the whole process, so its totals include ~3.2 G cycles of
interpreter/torch start-up.  The baseline CSV is the same process with
`--reps 0 --warm 0`, i.e. everything except the calls we care about; the
per-call numbers below are `(total - baseline) / (warm + reps)` with
warm=3, reps from perf_matrix.sh's table.

Derived columns: IPC, FMA/cycle (vec-FMA = 16 MACs = 32 flops), D1 fill rate
and L2 miss rate per *load* (not per instruction), load-bound cycles and
store-forward stalls per call, and instructions per (row,key) MAC pair, which
is the shape-independent way to compare two kernels with different blocking.
"""

from __future__ import annotations

import argparse
import collections
import sys

WARM = 3
REPS = {  # must match reps_for() in perf_matrix.sh
    (2048, 8, 64): 6,
    (1024, 16, 128): 6,
    (512, 8, 64): 40,
    (128, 8, 64): 120,
}
GHZ = 3.80  # measured ~3.7-3.9 GHz with one core under AVX-512


def load(path):
    d = collections.defaultdict(dict)
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) != 5 or parts[0] == "variant":
            continue
        v, shape, dt, metric, val = parts
        try:
            d[(v, shape, dt)][metric] = float(val)
        except ValueError:
            pass
    return d


def pairs(shape, causal=True):
    """(row,key) MAC pairs the kernel must touch, for the causal protocol."""
    s, h, _ = (int(x) for x in shape.split(":"))
    # sum over q-blocks of 4 rows x keys-in-block; keys ~ q0+4 -> 4 * S^2/2 / 4
    return h * s * s / 2.0 * 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("perf")
    ap.add_argument("base")
    args = ap.parse_args()
    perf, base = load(args.perf), load(args.base)

    keys = sorted(perf, key=lambda k: (k[1], k[2], k[0]))
    print(f"{'variant':6s} {'shape':12s} {'dtype':5s} {'us':>9s} {'Mcyc':>8s} {'Minsn':>8s} "
          f"{'IPC':>5s} {'FMA/cyc':>7s} {'in/FMA':>6s} {'in/pair':>7s} {'D1fill%':>7s} "
          f"{'L2miss%':>7s} {'SF/key':>7s} {'Ldbnd%':>7s}")
    for k in keys:
        v, shape, dt = k
        if v.startswith("base_"):
            continue
        b = base.get(("base_" + v, shape, dt), {})
        if not b:
            # a qt/workers variant still has the same start-up cost as `ours`
            b = base.get(("base_ours", shape, dt), base.get(("base_vllm", shape, dt), {}))
        reps = REPS.get(tuple(int(x) for x in shape.split(":")), 10)
        div = WARM + reps
        got = {}
        for m in ("cycles", "instructions", "median_us", "mem_inst_retired.all_loads",
                  "l1d.replacement", "l2_rqsts.demand_data_rd_miss",
                  "ld_blocks.store_forward", "exe_activity.bound_on_loads",
                  "uops_issued.any", "branches", "branch-misses"):
            if m in perf[k]:
                got[m] = (perf[k][m] - b.get(m, 0.0)) / (1 if m == "median_us" else div)
        if "cycles" not in got or got["cycles"] <= 0:
            continue
        s, h, dim = (int(x) for x in shape.split(":"))
        flops = 4.0 * h * (s * s / 2.0) * dim
        vecfma = flops / 32.0
        ipc = got["instructions"] / got["cycles"]
        loads = max(1.0, got.get("mem_inst_retired.all_loads", 1.0))
        d1 = 100.0 * got.get("l1d.replacement", 0.0) / loads
        l2 = 100.0 * got.get("l2_rqsts.demand_data_rd_miss", 0.0) / loads
        sf = got.get("ld_blocks.store_forward", 0.0) / max(1.0, pairs(shape))
        ldb = 100.0 * got.get("exe_activity.bound_on_loads", 0.0) / got["cycles"]
        print(f"{v:6s} {shape:12s} {dt:5s} {got['median_us']:9.1f} "
              f"{got['cycles']/1e6:8.1f} {got['instructions']/1e6:8.1f} "
              f"{ipc:5.2f} {ipc*vecfma/got['instructions']:7.2f} "
              f"{got['instructions']/vecfma:6.2f} "
              f"{got['instructions']/max(1.0,pairs(shape)):7.2f} "
              f"{d1:7.1f} {l2:7.2f} {sf:7.3f} {ldb:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
