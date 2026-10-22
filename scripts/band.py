"""Per-row band across repeat runs of the same tree: min..max of both ratios.

The scorecard answers "did this row win"; a row at 1.00-1.03 answers differently
on consecutive runs, and this project has already published one such number that
did not survive (`NOTES-X86.md`'s 0.98 that re-measured at 1.14-1.21). So a
result is quoted as a band over R runs of the same build, and a row only counts
as a win if it is below 1 in every run.

    .venv/bin/python tests/bench_vs_refs.py --vllm ... --json /tmp/r1.json
    .venv/bin/python tests/bench_vs_refs.py --vllm ... --json /tmp/r2.json
    .venv/bin/python scripts/band.py /tmp/r1.json /tmp/r2.json

Both ratios are `ours/baseline`, so below 1 wins. `wins` counts rows that are
< 1 in every run; `may` counts rows that are < 1 in at least one (i.e. rows
sitting on the line).
"""

import json
import sys


def key(r):
    return (r["seq"], r["heads"], r["dim"], r["dtype"], r["causal"])


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    runs = []
    for path in argv[1:]:
        with open(path) as f:
            runs.append({key(r): r for r in json.load(f)})
    keys = sorted(set().union(*(r.keys() for r in runs)),
                  key=lambda k: (k[0], k[1], k[2], k[3], not k[4]))
    print(f"{len(runs)} runs: {', '.join(p.rsplit('/', 1)[-1] for p in argv[1:])}")
    print(f"{'shape':22s} {'dtype':5s} {'o/sdpa band':>15s} {'o/vllm band':>15s}  verdict  (band = medians; min = per-run minimum)")
    print("-" * 78)
    wins = fails = may = 0
    for k in keys:
        seq, heads, dim, dtype, causal = k
        sd, vl, vm = [], [], []
        for r in runs:
            row = r.get(k)
            if row is None:
                continue
            sd.append(row["o_sdpa"])
            if row["o_vllm"] is not None:
                vl.append(row["o_vllm"])
                # min-based ratio: interference can only slow a sample down, so
                # the worst run's *min* is the honest upper bound on this row
                vm.append(row.get("o_vllm_min", row["o_vllm"]))
        line = f"S={seq:<5d}H={heads:<3d}D={dim:<4d} {dtype:5s} "
        line += f"{min(sd):6.2f}..{max(sd):5.2f}  " if sd else f"{'-':>15s}  "
        if vl:
            line += f"{min(vl):6.2f}..{max(vl):5.2f}"
        else:
            line += f"{'-':>13s}"
        if vl:
            if max(vm) < 1.0:
                verdict = "win"
                wins += 1
            elif min(vl) < 1.0:
                verdict = "on the line"
                may += 1
            else:
                verdict = "loss"
                fails += 1
        else:
            verdict = ""
        if vm:
            line += f"   worst-run min {max(vm):5.2f}"
        print(f"{line}  {verdict}")
    n = wins + fails + may
    print(f"\nbeats vLLM in every run: {wins}/{n}; on the line (<1 somewhere): {may}; "
          f"losses: {fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
