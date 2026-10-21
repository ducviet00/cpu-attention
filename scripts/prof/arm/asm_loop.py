"""Per-loop instruction histograms for one Mojo kernel instantiation.

The kernel's cost per key tile is what matters, so this finds every loop that
contains `bfmmla` inside a named instantiation, recovers each loop body from the
backward branch, and histograms mnemonics -- separating arithmetic from stack
traffic (`[sp, ...]`), which is the thing that decides whether the softmax
overlaps the matmul.

usage: asm_loop.py arm.s [substring-of-instantiation] [--all]
"""

from __future__ import annotations

import collections
import re
import sys

FUNC = re.compile(r'\t\.type\t"(.*)",@function')
LABEL = re.compile(r"^\s*(\.LBB[\w]+):")
# b<cond> .LBB / b .LBB -- backward branch closes a loop
BR = re.compile(r"^\s+(b\.[a-z]+|b)\s+(\.LBB[\w]+)\s*$")


def mnemonic(line: str) -> str:
    s = line.strip()
    if not s or s.startswith((".", "#", "//")) or s.endswith(":"):
        return ""
    return s.split()[0]


def histogram(body: list[str]) -> tuple[collections.Counter, int, int]:
    hist = collections.Counter(m for m in map(mnemonic, body) if m)
    total = sum(hist.values())
    sp = sum(1 for ln in body if "[sp" in ln)
    return hist, total, sp


def main() -> int:
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "head_dim=64,qt=4"
    show_all = "--all" in sys.argv
    lines = open(path).read().splitlines()

    bounds = [(i, m.group(1)) for i, ln in enumerate(lines)
              if (m := FUNC.match(ln))]
    bounds.append((len(lines), ""))
    target = None
    for k in range(len(bounds) - 1):
        if bounds[k][1].count(want) and "dense_task" in bounds[k][1]:
            target = (bounds[k][0], bounds[k + 1][0], bounds[k][1])
    if target is None:
        print(f"no function matching {want}", file=sys.stderr)
        return 2
    lo, hi, name = target
    print(f"function: ...{name[name.find('dense::'):][:70]}  lines {lo}-{hi}")

    body = lines[lo:hi]
    labels = {m.group(1): i for i, ln in enumerate(body) if (m := LABEL.match(ln))}
    loops = []
    for i, ln in enumerate(body):
        m = BR.match(ln)
        if m and m.group(2) in labels and labels[m.group(2)] <= i:
            loops.append((labels[m.group(2)], i))
    loops.sort(key=lambda ab: ab[1] - ab[0])
    print(f"{len(loops)} loops in this function\n")
    seen = set()
    for a, b in loops:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        h, total, sp = histogram(body[a:b + 1])
        if not h.get("bfmmla") and not show_all:
            continue
        print(f"loop body lines {lo+a}-{lo+b}: {total} instr, "
              f"{h.get('bfmmla',0)} bfmmla, {h.get('fmla',0)} fmla, "
              f"{h.get('fmul',0)} fmul, {sp} [sp] ({100*sp/max(total,1):.0f}%)")
        top = " ".join(f"{m}={c}" for m, c in h.most_common(10))
        print(f"    {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
