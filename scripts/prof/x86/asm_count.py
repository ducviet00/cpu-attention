"""Static instruction accounting for one instantiation of the dense kernel.

    objdump -d --line-numbers C_dbg.so | python scripts/prof/x86/asm_count.py \
        --sym 'dtype=f32,head_dim=128,qt=4' --bucket 505-517:QK \
        --bucket 519-530:mask --bucket 532-558:softmax --bucket 560-574:PV

Reads objdump output on stdin (built with `-g1` so every instruction carries a
`dense.mojo:LINE` attribution), keeps only the instructions that belong to the
named `_dense_task` instantiation, and prints:

  * instructions per source line, with the opcode classes that matter here
    (FMA, load, store, broadcast, shuffle, compare/select, spill)
  * a bucket roll-up, so the per-phase split is the tool's output, not a guess

Dynamic counts come from the source's loop structure (QK = head_dim
iterations, P@V = one per key, softmax = one per row per key tile); this script
only produces the static multiplicities, and `PROFILE_X86.md` multiplies.

`spill` counts instructions whose memory operand is rsp/rbp-relative: with
`-O3` Mojo only uses the stack frame for `stack_allocation` arrays and real
register spills, so a high count in the inner loop is spill traffic.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

FMA = re.compile(r"\bv(fn?m(add|sub)\d*(ss|ps|sd|pd)|fmadd\d+\w*)\b")
LOAD = re.compile(r"\b(vmov\w*(ps|pd|dqu|dqa|dq|q|d|ups|aps|dqu16|dqu32)\w*)\b")
BCAST = re.compile(r"\bvbroadcast\w*\b|\bvpbroadcast\w*\b")
SHUF = re.compile(r"\bv(shuf\w*|permi\w*|permt\w*|perm\w*|unpck\w*|insert\w*|extract\w*)\b")
CMP = re.compile(r"\bv(cmpp\w*|pcmp\w*|blend\w*|pternlog\w*|pord|pand|kortest\w*|kand\w*|kor\w*|kshift\w*)\b")
STORE = re.compile(r"\bvmov\w*(ps|pd|dqu|dqa|dq|q|d|ups|aps)\b")
BR = re.compile(r"^\s*[0-9a-f]+:\s+.*\b(j\w+|call\w*|ret\w*)\s")
REDUCE = re.compile(r"\bv(reduce|maxps|minps|maxss|minss|maxpd)\w*|\bvshuf\w*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", required=True)
    ap.add_argument("--bucket", action="append", default=[],
                    help="lo-hi:NAME over dense.mojo line numbers")
    ap.add_argument("--lib", default="")
    args = ap.parse_args()

    buckets = []
    for b in args.bucket:
        rng, name = b.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        buckets.append((lo, hi, name))

    cur_func = ""
    in_func = False
    line = 0
    per_line: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    spill_total = 0
    insn_total = 0
    for raw in sys.stdin:
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", raw)
        if m:
            cur_func = m.group(1)
            in_func = args.sym in cur_func
            continue
        if not in_func:
            continue
        m = re.search(r"dense\.mojo:(\d+)", raw)
        if m:
            line = int(m.group(1))
        m = re.match(r"^\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(\w+)", raw)
        if not m:
            continue
        op = m.group(3)
        insn_total += 1
        total[op] += 1
        cls = "other"
        if FMA.search(op):
            cls = "fma"
        elif BCAST.search(op):
            cls = "bcast"
        elif SHUF.search(op):
            cls = "shuffle"
        elif CMP.search(op):
            cls = "cmp/sel"
        elif LOAD.search(op):
            cls = "load" if "(" in raw.split("\t")[-1] or "[" in raw else "regmove"
        elif STORE.search(op):
            cls = "store" if ("(" in raw or "[" in raw) else "regmove"
        if re.search(r"\((%rsp|%rbp)", raw):
            cls = "spill"
            spill_total += 1
        per_line[line][cls] += 1
        per_line[line]["n"] += 1

    print(f"# instructions in {args.sym}: {insn_total}, stack-frame/spill: {spill_total} "
          f"({100.0*spill_total/max(1,insn_total):.1f}%)")
    print(f"# top opcodes: " + " ".join(f"{k}={v}" for k, v in total.most_common(12)))
    print("line  " + "  ".join(f"{c:>7s}" for c in
                               ("n", "fma", "load", "store", "bcast", "shuffle", "cmp/sel", "spill")))
    for ln in sorted(per_line):
        c = per_line[ln]
        if c["n"] == 0:
            continue
        print(f"{ln:5d} " + "  ".join(f"{c[k]:7d}" for k in
                                      ("n", "fma", "load", "store", "bcast", "shuffle", "cmp/sel", "spill")))
    if buckets:
        print("\nbucket                      n     fma    load   store   bcast shuffle cmp/sel   spill")
        for lo, hi, name in buckets:
            c = collections.Counter()
            for ln in range(lo, hi + 1):
                c.update(per_line.get(ln, {}))
            print(f"{name:22s} {c['n']:6d} {c['fma']:7d} {c['load']:7d} {c['store']:7d} "
                  f"{c['bcast']:7d} {c['shuffle']:7d} {c['cmp/sel']:7d} {c['spill']:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
