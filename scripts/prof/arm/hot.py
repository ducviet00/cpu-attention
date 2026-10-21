"""Rank the hottest instructions inside one symbol of a ``perf annotate`` dump.

    SYM=dense::_dense_task[...]
    sudo perf annotate -i /tmp/p.data --stdio -n -s "$SYM" > /tmp/ann.txt
    python3 scripts/prof/arm/hot.py /tmp/ann.txt [_dense_task] [top] [mnemonic]

The dump has one section per symbol and one line per instruction with its sample
count; this prints, for the matching symbols: the mnemonic share of the samples
(so ``bfmmla`` vs ``ldr/str [sp]`` vs the exp polynomial is visible at a glance)
and the top individual instructions. Shares are of the symbol's own samples.
"""

import collections
import re
import sys

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "_dense_task"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
only = sys.argv[4] if len(sys.argv) > 4 else None

sections = []
cur = None
for line in open(path, errors="replace"):
    m = re.search(r"([0-9a-f]+) <(.*)>:\s*$", line)
    if m and not re.match(r"^\s*\d+ :\s+[0-9a-f]+:", line):
        cur = {"name": m.group(2), "rows": []}
        sections.append(cur)
        continue
    m = re.match(r"^\s*(\d+) :\s+([0-9a-f]+):\s+(\S+)\s*(.*)", line)
    if m and cur is not None:
        cur["rows"].append((int(m.group(1)), int(m.group(2), 16), m.group(3),
                            m.group(4)[:64]))

cand = [s for s in sections if want in s["name"]]
if not cand:
    print("no section matching", want)
    sys.exit(1)
cand.sort(key=lambda s: -sum(r[0] for r in s["rows"]))
for s in cand[:2]:
    tot = sum(r[0] for r in s["rows"])
    if not tot:
        continue
    print(f"\n=== {s['name'][:110]}\n    {tot} samples, {len(s['rows'])} insns")
    hist = collections.Counter()
    for p, a, mn, rest in s["rows"]:
        hist[mn] += p
    print("    share:", " ".join(
        f"{k}:{100*v/tot:.0f}%" for k, v in hist.most_common(12)))
    rows = sorted(s["rows"], key=lambda r: -r[0])
    shown = 0
    for p, a, mn, rest in rows:
        if p == 0:
            break
        if only and only != mn:
            continue
        print(f"  {100*p/tot:5.1f}% {p:5d} {a:#08x} {mn:10s} {rest}")
        shown += 1
        if shown >= top:
            break
