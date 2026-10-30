"""Summarize a `benchmarks/bench.py --json` run: verdicts against the reference (SDPA for
flash rows, vLLM for paged rows) by README row family, median ratios per kind x dtype
(and new/previous with `--previous`), and every loss.

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
        v = r.get("verdict", "no reference")
        counts[family(r)][v] += 1
        counts["all"][v] += 1
    for f in FAMILIES + ("all",):
        c = counts[f]
        extra = f"  invalid {c['INVALID']}" if c["INVALID"] else ""
        extra += f"  no reference {c['no reference']}" if c["no reference"] else ""
        print(f"  {f:11s} {c['win']:3d} / {c['unresolved']:3d} / {c['loss']:3d}{extra}")
    for kind in ("flash", "paged"):
        for dt in ("bf16", "fp32"):
            sel = [r for r in rows if r["kind"] == kind and r["dtype"] == dt]
            for key in ("ratio", "new/previous"):
                xs = [r[key] for r in sel if key in r]
                if xs:
                    print(f"  {kind} {dt}: median {key} {statistics.median(xs):.3f}")
    for r in rows:
        if r.get("verdict") == "loss":
            prev = f"  new/previous {r['new/previous']:.3f}" if "new/previous" in r else ""
            print(f"  loss {r['label']:45s} {r['ratio']:.3f} vs {r['baseline']}{prev}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
