"""Summarize saved same-process runs; never compare timings across JSON files."""

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
names = ["paged-full-afbea06.json", "paged-full-bfdot.json",
         "paged-full-bfdot-hilo.json", "paged-full-hilo-scalar-peer.json",
         "paged-full-hilo-rotated.json",
         "flash-full-rq128.json"]
lines = ["# Per-row Axion results", "",
         "Each ratio uses arms timed in the same process. No cross-run speedups.", ""]
for name in names:
    path = ROOT / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    rows = data["rows"]
    counts = collections.Counter(r.get("verdict", "no valid baseline") for r in rows)
    invalid = [r["label"] for r in rows if not r["valid"]["mojo_new"]]
    lines += [f"## `{name}`", "", f"Rows: {len(rows)}; {dict(counts)}.",
              f"Invalid new outputs: {invalid}.", "",
              "| Row | Verdict | New / valid C++ | New / previous | Control spread |",
              "|---|---|---:|---:|---:|"]
    for r in rows:
        ratio = r.get("ratio")
        previous = r.get("new/previous")
        if previous is None and "mojo_before" in r["min_us"]:
            previous = r["min_us"]["mojo_new"] / r["min_us"]["mojo_before"]
        spread = r.get("ctrl_spread")
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        lines.append(f"| {r['label']} | {r.get('verdict', 'no valid baseline')} | "
                     f"{fmt(ratio)} | {fmt(previous)} | {fmt(spread)} |")
    lines += [""]
(ROOT / "ROWS.md").write_text("\n".join(lines))
