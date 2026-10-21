"""Turn bench_vs_refs.py JSON output into the campaign scorecard.

    .venv/bin/python tests/bench_vs_refs.py --vllm ... --json /tmp/x86.json
    .venv/bin/python scripts/scorecard.py /tmp/x86.json /tmp/arm.json

One file per run; the title of each column is the file's stem. Rows are marked
`win` when `ours/baseline < 1` (the project convention) in both columns, and the
parity column says whether the winning kernel was inside tolerance. This is the
only definition of "done" the campaign has, so it is computed, not read off a
table by eye.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        rows = json.load(f)
    return {(r["seq"], r["heads"], r["dim"], r["dtype"], r["causal"]): r
            for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--show-parity", action="store_true")
    args = ap.parse_args()

    tables = [(p.rsplit("/", 1)[-1].rsplit(".", 1)[0], load(p)) for p in args.jsons]
    keys = sorted(set().union(*(t.keys() for _, t in tables)),
                  key=lambda k: (k[0], k[1], k[2], k[3], not k[4]))
    head = f"{'shape':22s} {'dtype':5s} {'causal':6s}"
    for name, _ in tables:
        head += f" | {name + ' o/sdpa':>16s} {name + ' o/vllm':>16s}"
    print(head)
    print("-" * len(head))
    wins = {name: [0, 0, 0] for name, _ in tables}
    for k in keys:
        seq, heads, dim, dtype, causal = k
        line = f"S={seq:<5d}H={heads:<3d}D={dim:<4d} {dtype:5s} {str(causal):6s}"
        for name, t in tables:
            r = t.get(k)
            if r is None:
                line += f" | {'-':>16s} {'-':>16s}"
                continue
            o_s, o_v = r["o_sdpa"], r["o_vllm"]
            flag = "" if r["parity_ok"] else "!"
            line += f" | {o_s:14.2f}{flag:2s}"
            if o_v is None:
                line += f" {'-':>16s}"
            else:
                wins[name][2] += 1
                line += f" {o_v:14.2f}{flag:2s}"
                if o_v < 1.0:
                    wins[name][1] += 1
            if o_s < 1.0:
                wins[name][0] += 1
        print(line)
    print()
    for name, t in tables:
        n = len(t)
        nv = wins[name][2]
        print(f"{name}: beats sdpa {wins[name][0]}/{n}, "
              f"beats vllm {wins[name][1]}/{nv if nv else n}")
    bad = [(name, k, r) for name, t in tables for k, r in t.items()
           if not r["parity_ok"]]
    if bad:
        print(f"\n{len(bad)} parity failures:")
        for name, k, r in bad:
            print(f"  {name} S={k[0]} H={k[1]} D={k[2]} {k[3]} causal={k[4]}"
                  f" max|diff|={r['maxdiff']:.2e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
