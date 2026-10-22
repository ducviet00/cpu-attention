"""Interleaved A/B of runtime knobs (``qt``, the short-sequence path) per shape.

Every variant is timed in one process with the arm order rotated each round and
the *minimum* reported, because these boxes drift (docs/FINDINGS.md §5). The
first variant is the control: its ratio to itself over two slots is the noise
floor of the run, so a claim under 3% can be checked against it.

    .venv/bin/python tests/bench_knobs.py --shapes 128:8:64,256:16:128 \
        --variants ship:8:256,qt8:8:100000,qt12:12:256,smalloff:8:0 --rounds 15

`--variants name:qt:small` -- ``qt`` 0 = the `_qt_for` table, ``small`` is
`CPU_ATTENTION_SMALL_MAXK` (0 disables the short-sequence kernel).
Append ``:ctrl`` to a name to run the *first* variant's settings under that
name (a duplicate arm, i.e. the noise floor).
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import sdpa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="128:8:64,256:16:128,512:32:128")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--causal", choices=["0", "1"], default="1")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--variants", default="ship:0:256,ctrl:0:256:ctrl")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    causal = args.causal == "1"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    spec = []
    for v in args.variants.split(","):
        f = v.split(":")
        spec.append((f[0], int(f[1]), int(f[2]), len(f) > 3 and f[3] == "ctrl"))
    base = spec[0]

    def arm(s):
        name, qt, small, ctrl = s
        if ctrl:
            name, qt, small = base[0] + "dup", base[1], base[2]
        return name, qt, small

    _burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        _burn @ _burn
    del _burn

    print(f"threads={torch.get_num_threads()} causal={causal} "
          f"dtype={args.dtype}  (min of {args.rounds}, rotated)")
    for sh in args.shapes.split(","):
        S, H, D = (int(x) for x in sh.split(":"))
        q = torch.randn(1, S, H, D, dtype=dt)
        k = torch.randn(1, S, H, D, dtype=dt)
        v = torch.randn(1, S, H, D, dtype=dt)
        fns = {}
        for s in spec:
            name, qt, small = arm(s)
            # The knob has to be set *inside* the timed callable: `qt` and
            # `CPU_ATTENTION_SMALL_MAXK` are read while the plan is built, and
            # every variant's plan is in the cache after the warmup, so the
            # toggle costs a dict key and nothing else.
            fns[name] = (
                lambda qt=qt, small=small: (
                    K.set_qt(qt),
                    K.set_small_maxk(small),
                    attn.flash_attn_func(q, k, v, causal=causal),
                )[-1]
            )
        ref = sdpa(q, k, v, causal)
        for name, f in fns.items():
            for _ in range(2):
                o = f()
            d = (o.transpose(1, 2).float() - ref.float()).abs().max().item()
            tol = 3e-2 if dt is torch.bfloat16 else 2e-4
            if d > tol:
                print(f"  !! {name} parity {d:.1e} > {tol:.0e}")
        samples = {n: [] for n in fns}
        names = list(fns)
        for r in range(args.rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for name in order:
                t0 = time.perf_counter()
                fns[name]()
                samples[name].append((time.perf_counter() - t0) * 1e6)
        mn = {n: min(v_) for n, v_ in samples.items()}
        md = {n: statistics.median(v_) for n, v_ in samples.items()}
        first = names[0]
        line = f"{S:5d}:{H}:{D} {args.dtype} "
        for n in names:
            line += f" {n} {mn[n]:7.1f}/{md[n]:7.1f} ({mn[n]/mn[first]:5.3f})"
        print(line, flush=True)
    K.set_qt(0)
    K.set_small_maxk(K._DEFAULT_SMALL_MAXK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
