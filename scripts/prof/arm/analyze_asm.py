"""Count instructions inside the hot loops of a disassembled kernel.

    objdump -d --no-show-raw-insn _C.so > /tmp/ours.asm
    python3 scripts/prof/arm/analyze_asm.py /tmp/ours.asm [function-substring]

Finds every backward branch inside the matching function(s), takes the loop it
closes (the instructions between the branch target and the branch), and reports
the loops that contain ``bfmmla`` (or, with ``--any``, the biggest ones): body
instruction count, bfmmla count, and an instruction histogram. That is the
static counterpart to ``pk.py`` -- instructions per row-key is the number that
binds, and this says which instructions they are.
"""

import collections
import re
import sys

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "bf16,head_dim=64,qt=4"
any_loop = "--any" in sys.argv

lines = open(path).read().splitlines()
func = None
insns = []  # (addr, mnemonic, text, func)
for line in lines:
    m = re.match(r"^([0-9a-f]+) <(.*)>:$", line)
    if m:
        func = m.group(2)
        continue
    m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
    if m:
        insns.append((int(m.group(1), 16), m.group(2), m.group(3), func))

by_func = collections.defaultdict(list)
for i in insns:
    by_func[i[3]].append(i)

targets = [f for f in by_func if want in f]
if not targets:
    print("no function matching", want)
    sys.exit(1)

for f in targets:
    body = by_func[f]
    addr_index = {ins[0]: k for k, ins in enumerate(body)}
    print(f"\n=== {f[:100]}  ({len(body)} instructions)")
    loops = []
    for k, (addr, mn, text, _) in enumerate(body):
        if not mn.startswith("b"):
            continue
        m = re.search(r"^([0-9a-f]+)", text)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt in addr_index and addr_index[tgt] < k:
            loops.append((addr_index[tgt], k))
    for start, end in loops:
        blk = body[start : end + 1]
        nbf = sum(1 for i in blk if "bfmmla" in i[1])
        if not any_loop and nbf == 0:
            continue
        hist = collections.Counter(i[1] for i in blk)
        sp = sum(1 for i in blk if "[sp" in i[2])
        print(
            f"\n  loop {blk[0][0]:#x}-{blk[-1][0]:#x}: {len(blk)} insns, "
            f"{nbf} bfmmla, {sp} stack refs"
        )
        print("   ", " ".join(f"{k}:{v}" for k, v in hist.most_common(14)))
