"""Differential ``perf stat``: instructions / cycles per row-key.

    sudo python3 scripts/prof/arm/pk.py <one.py|vllm_one.py> <S> <H> <D> \
        <bf16|fp32> <0|1> [iters_big] [iters_small]

Runs the same script twice with different iteration counts inside one process
and subtracts, so the interpreter start-up (torch import, tensor allocation, and
for vLLM the cache staging) cancels out and what is left is the kernel. Needs
``perf``; on the m9g box the PMU is only readable through sudo.

The interesting derived numbers are *instructions per row-key* (>8 of which are
``bfmmla``/``bfdot``/FMA on the two matmul phases at bf16) and IPC.
"""

import os
import subprocess
import sys

script, S, H, D, dt, causal = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
)
big = int(sys.argv[7]) if len(sys.argv) > 7 else 60
small = int(sys.argv[8]) if len(sys.argv) > 8 else 10
EV = [
    "cycles",
    "instructions",
    "l1d_cache",
    "l1d_cache_refill",
    "l2d_cache_refill",
    "stall_backend",
    "br_mis_pred",
]

PY = "/home/ubuntu/cpu-attention/.venv/bin/python"
HOME = "/home/ubuntu"


def run(iters):
    cmd = ["sudo", "-n"]
    # sudo resets the environment, so any kernel knob has to be re-exported
    # through it explicitly (CPU_ATTENTION_QT selects the ARM kernel variant).
    knob = os.environ.get("CPU_ATTENTION_QT")
    if knob:
        cmd += ["env", "CPU_ATTENTION_QT=" + knob]
    cmd += ["perf", "stat", "-x,", "-e", ",".join(EV),
            PY, script, S, H, D, dt, causal, str(iters)]
    env = dict(os.environ, HOME=HOME)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stderr
    vals = {}
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[2] in EV:
            try:
                vals[parts[2]] = float(parts[0])
            except ValueError:
                pass
    if not vals:
        print(out[-2000:])
    return vals


a, b = run(big), run(small)
n = big - small
Si, Hi = int(S), int(H)
rk = Hi * (Si * (Si + 1) // 2 if causal == "1" else Si * Si) * n
print(f"delta over {n} iters, {rk} row-keys")
for e in EV:
    if e in a and e in b:
        print(f"  {e:20s} {(a[e] - b[e]) / rk:9.4f} /row-key")
if "instructions" in a and "cycles" in a:
    di = a["instructions"] - b["instructions"]
    dc = a["cycles"] - b["cycles"]
    if dc > 0:
        print(f"  IPC {di/dc:.2f}")
