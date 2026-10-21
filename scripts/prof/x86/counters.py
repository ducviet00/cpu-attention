"""Run one dense-attention variant in a tight loop, for `perf stat` to count.

    perf stat -e cycles,instructions,... \
      .venv/bin/python scripts/prof/x86/counters.py --variant ours --shape 1024:16:128

Both variants are driven from the same shapes/inputs as `tests/bench_vs_refs.py`
(vLLM's cache staged outside the loop, ours charged its pack), so the counter
ratios are comparable. Per row-key figures are printed at the end: they are the
number to compare, not the raw totals, because the two kernels walk different
amounts of work per call only through the trailing partial tile.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "tests"))
import cpu_attention as attn  # noqa: E402
from bench_vs_refs import load_vllm, paged_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["ours", "vllm", "sdpa"], required=True)
    ap.add_argument("--shape", default="1024:16:128")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    seq, heads, dim = (int(x) for x in args.shape.split(":"))
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    causal = bool(args.causal)
    q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(seq, heads, dim, dtype)
    q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = torch.empty_like(q)

    if args.variant == "ours":
        fn = lambda: attn.flash_attn_func(q3, k3, v3, causal=causal)
    elif args.variant == "sdpa":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(
            q3.transpose(1, 2), k3.transpose(1, 2), v3.transpose(1, 2),
            is_causal=causal, scale=dim ** -0.5)
    else:
        vllm = load_vllm(args.vllm)
        if vllm is None:
            raise SystemExit("--vllm is required for the vllm variant")
        vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
        meta = vllm.get_scheduler_metadata(
            len(lens), heads, heads, dim, lens, dtype, qsl, causal, -1,
            args.isa, False, None, "auto")
        fn = lambda: vllm.cpu_attention_with_kv_cache(
            q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1, bt, 0.0,
            meta, None, None, 1.0, 1.0, "auto")

    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fn()
    dt = time.perf_counter() - t0
    # causal work is half of seq^2 keys per row; non-causal is all of it
    row_keys = seq * seq * (0.5 if causal else 1.0)
    print(f"{args.variant} S={seq} H={heads} D={dim} {args.dtype} "
          f"causal={causal}: {dt / args.iters * 1e6:.1f} us/call, "
          f"row-keys/call={heads * row_keys:.3e}, threads={torch.get_num_threads()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
