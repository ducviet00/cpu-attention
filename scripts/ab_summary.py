"""Summarize a `benchmarks/bench.py --previous ... --json` run: verdicts by README row
family, new/previous medians per kind x dtype, and every loss.

    python scripts/ab_summary.py full.json [more.json ...]
"""

import collections
import json
import statistics
import sys

FAMILIES = ("flash", "engine", "engine_out", "api", "store")


def family(r):
    if r["kind"] == "flash":
        return "flash"
    return "store" if r.get("op") == "store" else r["entry"]


def summarize(path):
    rows = json.load(open(path))["rows"]
    print(f"== {path} ({len(rows)} rows)")
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[family(r)][r["verdict"]] += 1
        counts["all"][r["verdict"]] += 1
    for f in FAMILIES + ("all",):
        c = counts[f]
        extra = f"  invalid {c['invalid']}" if c["invalid"] else ""
        print(f"  {f:11s} {c['win']:3d} / {c['unresolved']:3d} / {c['loss']:3d}{extra}")
    for kind in ("flash", "paged"):
        for dt in ("bf16", "fp32"):
            np_ = [r["new/previous"] for r in rows if r["kind"] == kind and r["dtype"] == dt]
            if np_:
                print(f"  {kind} {dt}: median new/previous {statistics.median(np_):.3f}")
    for r in rows:
        if r["verdict"] == "loss":
            print(f"  loss {r['label']:45s} {r['ratio']:.3f}  new/previous {r['new/previous']:.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
