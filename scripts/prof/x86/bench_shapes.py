"""Headline wall-clock table: ours vs vLLM csrc/cpu vs torch SDPA.

    taskset -c 3 env OMP_NUM_THREADS=1 python scripts/prof/bench_shapes.py
    taskset -c 3 ... --dtype fp32 --json /tmp/bench_fp32.json

Single-threaded by protocol (the point is instructions-per-op, not scaling;
`components.py` covers the threaded fixed cost).  Variants are interleaved in
one process and reported as medians, because these boxes throttle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="both", choices=["fp32", "bf16", "both"])
    ap.add_argument("--causal", default="1", choices=["0", "1"])
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--isa", default="vec")
    ap.add_argument("--vllm", default=C.VLLM_DEFAULT)
    ap.add_argument("--with-sdpa", action="store_true")
    ap.add_argument("--kv-split", action="store_true",
                    help="pass enable_kv_split to vLLM's scheduler")
    ap.add_argument("--shapes", default="",
                    help="comma-separated S:H:D overrides for one-off probes")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    C.set_single_thread()
    causal = args.causal == "1"
    vllm = C.load_vllm(args.vllm)
    shapes = C.SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
    dtypes = (["fp32", "bf16"] if args.dtype == "both" else [args.dtype])
    print(f"threads={C.torch.get_num_threads()} causal={causal} "
          f"vllm={os.path.basename(args.vllm) if vllm else '-'} isa={args.isa} "
          f"kv_split={args.kv_split}", flush=True)

    rows = []
    for seq, heads, dim in shapes:
        for name in dtypes:
            c = C.case(seq, heads, dim, C.DTYPES[name])
            variants = {"ours": C.ours_fn(c, causal)}
            if vllm is not None:
                variants["vllm"] = C.build_vllm(vllm, c, heads, dim,
                                                C.DTYPES[name], args.isa, causal,
                                                args.kv_split)
            if args.with_sdpa:
                variants["sdpa"] = C.sdpa_fn(c, causal)
            # parity first: never report a timing for a variant that is wrong
            ref = C.sdpa_fn(c, causal)().transpose(1, 2).contiguous() if args.with_sdpa else None
            got = {}
            for n, f in variants.items():
                o = f()
                got[n] = o if o is not None else getattr(f, "out", None)
            if ref is not None:
                for n, o in got.items():
                    if n == "sdpa" or o is None:
                        continue
                    d = (o.float() - ref.float()).abs().max().item()
                    rel = d / max(1e-6, ref.float().abs().max().item())
                    print(f"    parity {n}: max|d|={d:.3e} rel={rel:.2e}", flush=True)
            med = C.timer(variants, rounds=args.rounds)
            rows.append(C.report(name, med, seq, heads, dim, causal))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
