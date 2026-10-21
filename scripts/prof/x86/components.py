"""Fixed-cost breakdown of one dense call, short S.

The engine's encoder path runs S=128-256, where a call is tens of microseconds
and everything that is not the inner loop matters.  vLLM has *none* of our
fixed cost: no K^T transpose (its cache is pre-paged), no per-call scratch
allocation, one extension call.  This driver measures, in one process and
interleaved:

  full        attn.flash_attn_func(...)            -- what the engine calls
  wrapper     _kernel._dense(...)                  -- the same minus validation
  transpose   the K^T (+ V repack) pass alone
  alloc_kt    torch.empty of the scratch
  alloc_out   torch.empty_like(out)
  dense_call  the 27-arg extension call, buffers preallocated
  null_call   the same with batch=0 (pure marshalling, no work)
  vllm        cpu_attention_with_kv_cache, cache pre-staged

and, with --workers, the same call inline (workers=1) vs on the pool, which is
the only honest way to price the GOMP fork-join: identical work both times.

    taskset -c 3 env OMP_NUM_THREADS=1 python scripts/prof/components.py --dtype fp32
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402


def parts(seq, heads, dim, dtype, causal):
    """Return (dict of thunks, closure that runs the real work)."""
    c = C.case(seq, heads, dim, dtype)
    q = c["q"].unsqueeze(0)
    q = q.contiguous()
    k = c["k"].unsqueeze(0).contiguous()
    v = c["v"].unsqueeze(0).contiguous()
    if dtype == torch.bfloat16:
        kt = torch.empty((1, heads, dim // 2, seq, 2), dtype=k.dtype)
        vt = None
        mode = 1
    else:
        kt = torch.empty((1, heads, dim, seq), dtype=k.dtype)
        vt = None
        mode = 0
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    scale = dim ** -0.5
    tr = K._transpose_attention()
    dn = K._dense_attention()
    workers = K._workers(1 * heads * (-(-seq // K._QT)), 8)

    def transpose():
        tr(k.data_ptr(), kt.data_ptr(), 0, 0, 1, heads, seq, dim,
           k.stride(0), k.stride(1), k.stride(2),
           kt.stride(1), kt.stride(2), v.stride(0), v.stride(1), v.stride(2),
           0, 0, K._DTYPE_CODES[k.dtype], mode, workers)

    def dense(batch=1, qt=K._QT):
        tasks = batch * heads * (-(-seq // qt))
        dn(q.data_ptr(), kt.data_ptr(), v.data_ptr(), out.data_ptr(),
           batch, seq, seq, heads, heads, dim,
           q.stride(0), q.stride(1), q.stride(2),
           kt.stride(1), kt.stride(2), kt.stride(3),
           v.stride(0), v.stride(1), v.stride(2),
           out.stride(0), out.stride(1), out.stride(2),
           scale, causal, K._DTYPE_CODES[q.dtype], qt,
           K._workers(tasks, 8))

    thunks = {
        "full": C.ours_fn(c, causal),
        "wrapper": lambda: K._dense(q, k, v, scale, causal),
        "transpose": transpose,
        "alloc_kt": lambda: torch.empty_like(kt),
        "alloc_out": lambda: torch.empty_like(q, memory_format=torch.contiguous_format),
        "dense_call": dense,
        "null_call": lambda: dense(batch=0),
    }
    return thunks, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--causal", default="1")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--shapes", default="128:8:64,256:8:64,512:8:64,1024:16:128")
    ap.add_argument("--vllm", default=C.VLLM_DEFAULT)
    ap.add_argument("--pool", action="store_true",
                    help="also time ours inline (workers=1) vs pool (workers=4)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.threads > 1:
        torch.set_num_interop_threads(1)
    dtype = C.DTYPES[args.dtype]
    causal = args.causal == "1"
    vllm = C.load_vllm(args.vllm)
    shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
    print(f"threads={args.threads} causal={causal}", flush=True)

    for seq, heads, dim in shapes:
        thunks, c = parts(seq, heads, dim, dtype, causal)
        variants = dict(thunks)
        if vllm is not None:
            variants["vllm"] = C.build_vllm(vllm, c, heads, dim, dtype, "vec",
                                            causal)
        if args.pool:
            K.set_num_workers(1)
            variants["ours_inline"] = C.ours_fn(c, causal)
            K.set_num_workers(4)
            variants["ours_pool4"] = C.ours_fn(c, causal)
            K.set_num_workers(0)
        med = C.timer(variants, rounds=args.rounds)
        base = med["full"]
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}: "
        line += " ".join(f"{n}={t:7.2f}" for n, t in med.items())
        print(line, flush=True)
        # the two numbers that matter: what is not the attention pass, and what
        # vLLM does not pay at all
        overhead = med["full"] - med["dense_call"]
        print(f"      non-kernel (wrap+check+2 allocs+transpose) = {overhead:7.2f} us"
              f" = {100*overhead/base:4.1f}% of the call;"
              f" transpose alone {med['transpose']:6.2f} us;"
              f" marshalling {med['null_call']:6.2f} us", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
