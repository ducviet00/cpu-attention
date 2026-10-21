"""One call, one variant, under callgrind -- for exact Ir / D1 / LL / branch counts.

The box has no PMU (`perf_event_paranoid=4`), so cycle counts come from
`cntvct_el0` in `arm_cycles.py` and *instruction* attribution comes from here.
Run it under valgrind, annotate the result, and join the two: callgrind gives
the split of a measured call across source lines, so `Ir(line)/Ir(call)` times
the measured cycles is a per-phase cycle estimate that cannot dead-code-eliminate
anything (unlike a phase-ablation build).

    taskset -c 3 env OMP_NUM_THREADS=1 \
      valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
               --callgrind-out-file=/tmp/cg.ours.512.bf16 \
      .venv/bin/python scripts/prof/arm_callgrind.py --variant ours \
               --shape 512:8:64 --dtype bf16 --calls 1

Keep S small (<=512): callgrind is ~50-100x slower than native and the whole
point is a tractable, exact count.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "x86"))
import _common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["ours", "vllm", "sdpa"])
    ap.add_argument("--shape", default="512:8:64")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--isa", default="neon")
    ap.add_argument("--calls", type=int, default=1)
    ap.add_argument("--causal", default="1")
    ap.add_argument("--vllm", default=C.VLLM_DEFAULT)
    args = ap.parse_args()

    C.set_single_thread()
    causal = args.causal == "1"
    seq, heads, dim = (int(x) for x in args.shape.split(":"))
    c = C.case(seq, heads, dim, C.DTYPES[args.dtype])

    if args.variant == "ours":
        fn = C.ours_fn(c, causal)
    elif args.variant == "sdpa":
        fn = C.sdpa_fn(c, causal)
    else:
        vllm = C.load_vllm(args.vllm)
        if vllm is None:
            print("no vllm reference build", file=sys.stderr)
            return 2
        fn = C.build_vllm(vllm, c, heads, dim, C.DTYPES[args.dtype], args.isa,
                          causal)

    fn()  # one warm call so first-call allocation is not in the count

    # Start collection only now, so torch's import/startup is not in the count.
    # `--toggle-collect` cannot bind a Mojo @export symbol by name, so drive
    # instrumentation from inside the process instead.
    def ctl(cmd: str) -> None:
        # `--instr-atstart=no` means nothing is counted until "on", and we do
        # NOT zero afterwards (that would discard what we just measured).
        os.system(f"callgrind_control -i {cmd} >/dev/null 2>&1")

    ctl("on")
    print(f"CALLGRIND-BEGIN {args.variant} S={seq} H={heads} D={dim} "
          f"{args.dtype} causal={causal} calls={args.calls}", flush=True)
    for _ in range(args.calls):
        fn()
    ctl("off")
    print("CALLGRIND-END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
