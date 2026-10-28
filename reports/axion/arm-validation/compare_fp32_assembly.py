"""Compare instruction counts in unchanged fp32 paged bodies across builds."""

import collections
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def bodies(path):
    text = subprocess.check_output(["objdump", "-dC", str(path)], text=True)
    result = {}
    current = None
    for line in text.splitlines():
        header = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if header:
            name = header[1]
            current = name if "kvcache::_body" in name and "dtype=f32" in name else None
            if current:
                result[current] = []
        elif current:
            ins = re.match(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)", line)
            if ins:
                result[current].append(ins[1])
    return result


old = bodies(ROOT / "before/_C_aarch64-bf16.so")
new = bodies(ROOT.parents[2] / "cpu_attention/_C_aarch64-bf16.so")
for name, seq in old.items():
    counts = collections.Counter(seq)
    other = collections.Counter(new[name])
    print(name.rsplit(",dtype=", 1)[-1], len(seq), len(new[name]),
          "same mnemonic sequence", seq == new[name], "count diff", counts != other)
