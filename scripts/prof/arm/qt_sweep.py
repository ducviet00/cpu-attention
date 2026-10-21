"""Interleaved `qt` sweep for the ARM dense path, long sequences.

`qt` is the number of query rows one work item processes, so it sets how many
times each K/V byte is reused: traffic per row-key scales as 1/qt. `_kernel`
reads its `_QT_ENV` module global on every call, so every `qt` can be timed
inside ONE process, round-robin -- which is the only way to survive a throttling
box. This is the ARM analogue of the x86 `CPU_ATTENTION_QT` finding (qt=8 worth
1.20-1.34x at long S with L2 miss rate 6.34% -> 2.62%).

    taskset -c 3 env OMP_NUM_THREADS=1 \
      .venv/bin/python scripts/prof/arm/qt_sweep.py --shape 2048:8:64 --dtype bf16

Also checks every `qt` against `qt=4` output, so a "fast" variant that is wrong
cannot be reported.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "x86"))
import _common as C  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="2048:8:64")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "both"])
    ap.add_argument("--qts", default="2,4,6,8")
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()

    C.set_single_thread()
    seq, heads, dim = (int(x) for x in args.shape.split(":"))
    qts = [int(x) for x in args.qts.split(",")]
    dtypes = ["bf16", "fp32"] if args.dtype == "both" else [args.dtype]

    for name in dtypes:
        c = C.case(seq, heads, dim, C.DTYPES[name])
        q, k, v = (c[x].unsqueeze(0) for x in ("q", "k", "v"))
        torch.set_num_threads(1)

        def run(qt: int):
            K._QT_ENV = str(qt)
            return C.ours_fn(c, True)()

        # parity: every qt must reproduce qt=4 (same kernel, different row block)
        K._QT_ENV = "4"
        ref = run(4)
        for qt in qts:
            if qt == 4:
                continue
            o = run(qt)
            d = (o.float() - ref.float()).abs().max().item()
            assert d < 2e-2, f"qt={qt} disagrees with qt=4: max|d|={d}"

        variants = {}
        for qt in qts:
            def fn(qt=qt):
                K._QT_ENV = str(qt)
                return C.ours_fn(c, True)()
            variants[f"qt{qt}"] = fn
        med = C.timer(variants, rounds=args.rounds)
        K._QT_ENV = None
        base = med.get("qt4") or min(med.values())
        print(f"S={seq} H={heads} D={dim} {name}: " + " ".join(
            f"{n}={t:9.1f}us({base/t:.2f}x)" for n, t in med.items()
        ) + f"  | {C.flops(seq, heads, dim, True)/1e9/ (min(med.values())*1e-6):.0f} GF/s best",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
