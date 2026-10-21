"""Run exactly one (variant, shape, dtype) N times -- the unit `perf stat` and
`valgrind --tool=callgrind` measure.

    # hardware counters (needs sudo for the PMU on this box)
    sudo perf stat -e cycles,instructions,... taskset -c 3 python scripts/prof/run_one.py \
        --variant ours --shape 2048:8:64 --dtype fp32 --reps 20

    # instruction-level attribution
    valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
        --callgrind-out-file=/tmp/cg.out \
        python scripts/prof/run_one.py --variant ours --shape 128:8:64 --reps 1 --no-warm

Print the wall time per call on stdout so the perf run has a denominator that
was measured in the same process.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ours",
                    choices=["ours", "vllm", "sdpa", "transpose", "nokt"])
    ap.add_argument("--shape", default="2048:8:64")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--causal", default="1")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3,
                    help="warm-up calls (included in the perf counters; see "
                         "perf_baseline.sh)")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--isa", default="vec")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU_ATTENTION_WORKERS equivalent, 0 = automatic")
    ap.add_argument("--qt", type=int, default=0)
    ap.add_argument("--v-stride", type=int, default=0,
                    help="SYNTHETIC, same idea for V: override the element stride "
                         "between cache lines of the V sweep (0 = real layout).")
    ap.add_argument("--kt-stride", type=int, default=0,
                    help="SYNTHETIC address-pattern probe: override the element "
                         "stride between K^T head-dim rows (0 = the real layout). "
                         "A small stride makes the whole K^T panel L1/L2-resident "
                         "without changing one instruction, at the price of wrong "
                         "numbers. Isolates 'misses' from 'loop structure'.")
    ap.add_argument("--vllm", default=C.VLLM_DEFAULT)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seq, heads, dim = (int(x) for x in args.shape.split(":"))
    dtype = C.DTYPES[args.dtype]
    causal = args.causal == "1"
    K.set_num_workers(args.workers)
    if args.qt:
        os.environ["CPU_ATTENTION_QT"] = str(args.qt)
        K._QT_ENV = os.environ["CPU_ATTENTION_QT"]

    c = C.case(seq, heads, dim, dtype)
    if args.variant == "ours":
        fn = C.ours_fn(c, causal)
    elif args.variant == "sdpa":
        fn = C.sdpa_fn(c, causal)
    elif args.variant in ("transpose", "nokt"):
        # the two halves of a dense call, to price the pass separately
        q = c["q"].unsqueeze(0).contiguous()
        k = c["k"].unsqueeze(0).contiguous()
        v = c["v"].unsqueeze(0).contiguous()
        if dtype == torch.bfloat16:
            kt = torch.empty((1, heads, dim // 2, seq, 2), dtype=k.dtype)
            mode = 1
        else:
            kt = torch.empty((1, heads, dim, seq), dtype=k.dtype)
            mode = 0
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        tr = K._transpose_attention()
        dn = K._dense_attention()
        scale = dim ** -0.5
        workers = K._workers(heads * (-(-seq // K._QT)), 8)
        rows = 1
        if args.variant == "transpose":
            fn = lambda: tr(k.data_ptr(), kt.data_ptr(), 0, 0, 1, heads, seq,
                            dim, k.stride(0), k.stride(1), k.stride(2),
                            kt.stride(1), kt.stride(2), v.stride(0),
                            v.stride(1), v.stride(2), 0, 0,
                            K._DTYPE_CODES[k.dtype], mode, workers)
        else:
            qt_used = args.qt or K._QT
            tasks = 1 * heads * (-(-seq // qt_used))
            ktb, ktd, ktt = kt.stride(1), kt.stride(2), kt.stride(3)
            kt_use = kt
            if args.kt_stride:
                # SYNTHETIC: same instruction stream, same tile count, same load
                # count; only the element stride between head-dim rows changes, so
                # the whole K^T panel becomes L1/L2-resident. Numbers are wrong on
                # purpose -- this measures the address pattern, nothing else.
                need = dim * args.kt_stride + seq + 256
                kt_use = torch.zeros(need, dtype=k.dtype)
                ktb, ktd, ktt = 0, args.kt_stride, 1
            vb, vtok, vh = v.stride(0), v.stride(1), v.stride(2)
            v_use = v
            if args.v_stride:
                # same instruction stream, V confined to a small buffer
                need = dim * args.v_stride + 256
                v_use = torch.zeros(need, dtype=v.dtype)
                vb, vtok, vh = 0, args.v_stride, 0
            fn = lambda: dn(q.data_ptr(), kt_use.data_ptr(), v_use.data_ptr(),
                            out.data_ptr(), 1, seq, seq, heads, heads, dim,
                            q.stride(0), q.stride(1), q.stride(2),
                            ktb, ktd, ktt,
                            vb, vtok, vh,
                            out.stride(0), out.stride(1), out.stride(2),
                            scale, causal, K._DTYPE_CODES[q.dtype],
                            qt_used, K._workers(tasks, 8))
        if args.variant == "transpose":
            fn()  # once, so --variant nokt (below) has a transposed K
    else:
        if args.variant != "vllm":
            raise SystemExit("bad variant")
        vllm = C.load_vllm(args.vllm)
        if vllm is None:
            raise SystemExit(f"no vLLM extension at {args.vllm}")
        fn = C.build_vllm(vllm, c, heads, dim, dtype, args.isa, causal,
                          os.environ.get("VLLM_KV_SPLIT") == "1")

    if not args.no_warm:
        for _ in range(args.warm):
            fn()
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(ts) if ts else 0.0
    f = C.flops(seq, heads, dim, causal)
    print(f"variant={args.variant} S={seq} H={heads} D={dim} {args.dtype} "
          f"threads={args.threads} reps={args.reps} median_us={med:.2f} "
          f"gflops={f/(med*1e-6)/1e9:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
