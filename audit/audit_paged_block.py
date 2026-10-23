"""Paged audit, laptop method: per-arm best-of-N blocks (kernel_ab style).

The interleaved round method is right on a server; on the Ryzen the clock/power
state changes between single calls, so a 21-round median is dominated by the
power state and even the min is a lucky sample. This script times one arm for
40 consecutive calls after 8 warmups (the method of `benchmarks/kernel_ab.py`)
and alternates blocks vllm/ours/ours/vllm to keep drift symmetric.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as C  # noqa: E402
from audit_paged import CASES, HEADS, DIM, make_case, stage  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cpu_attention as attn  # noqa: E402


def timed(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.path.expanduser(
        "~/vllm_cpu_build/vllm_cpu_attn_bench.so"))
    ap.add_argument("--isa", default="vec")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    vllm = C.load_so(args.vllm)
    dtypes = [{"bf16": torch.bfloat16, "fp32": torch.float32}[t]
              for t in args.dtypes.split(",")]
    rows = []
    for dtype in dtypes:
        tag = "bf16" if dtype is torch.bfloat16 else "fp32"
        for batch, ctx in CASES:
            q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens, pages = \
                make_case(batch, ctx, args.block, dtype)
            kc_o, vc_o = torch.zeros_like(kc), torch.zeros_like(vc)
            stage(k_all, v_all, kc_o, vc_o, args.block, batch, ctx, args.isa, None)
            if vllm is not None:
                stage(k_all, v_all, kc, vc, args.block, batch, ctx, args.isa, vllm)
            scale = DIM ** -0.5
            qsl = torch.arange(batch + 1, dtype=torch.int32)
            out = torch.empty_like(q)

            def ours_self():
                attn.self_attention(q, k_new, v_new, kc_o, vc_o, slots, blocks,
                                    lens, scale)

            def ours_cross():
                attn.cross_attention(q, kc_o, vc_o, blocks, lens, scale)

            meta = vllm.get_scheduler_metadata(
                batch, HEADS, HEADS, DIM, lens, dtype, qsl, True, -1,
                args.isa, False, None, "auto") if vllm else None

            def vllm_self():
                vllm.cpu_attn_reshape_and_cache(k_new, v_new, kc, vc, slots, args.isa)
                vllm.cpu_attention_with_kv_cache(
                    q, kc, vc, out, qsl, lens, scale, True, None, -1, blocks,
                    0.0, meta, None, None, 1.0, 1.0, "auto")

            def vllm_cross():
                vllm.cpu_attention_with_kv_cache(
                    q, kc, vc, out, qsl, lens, scale, True, None, -1, blocks,
                    0.0, meta, None, None, 1.0, 1.0, "auto")

            for op, a, b in (("self", ours_self, vllm_self),
                             ("cross", ours_cross, vllm_cross)):
                ts = {"ours": [], "vllm": []}
                for arm, fn in (("vllm", b), ("ours", a), ("ours", a), ("vllm", b)):
                    ts[arm].append(timed(fn, args.iters))
                o, v = min(ts["ours"]), min(ts["vllm"])
                row = {"op": op, "batch": batch, "ctx": ctx, "dtype": tag,
                       "block": args.block, "ours_us": o, "vllm_us": v,
                       "o_vllm": o / v}
                rows.append(row)
                print(f"{op:5s} b={batch:<2} ctx={ctx:<5} {tag}: ours {o:9.1f} "
                      f"vllm {v:9.1f}  o/vllm {o / v:5.2f}", flush=True)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
