"""Dense attention vs torch SDPA and vLLM's csrc/cpu kernel, drift-controlled.

Every variant is timed in a round-robin loop inside one process and the median
per variant is reported, because these boxes throttle: on the laptop a
sustained AVX-512 sweep drops to ~2.0 GHz and the same binary measures 0.63-1.09
against SDPA depending on when it runs. Never compare two runs of this script
against each other -- compare the ratios inside one run.

    uv run python tests/bench_vs_refs.py                      # bf16 + fp32, causal
    uv run python tests/bench_vs_refs.py --dtype fp32 --causal 0
    uv run python tests/bench_vs_refs.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so

vLLM's kernel is paged, so its K/V is staged into a block-table cache before
timing and the measured call is attention only; ours includes the K^T transpose
and its allocations, i.e. ours is charged for work vLLM does not do. Report both
`ours` and `ours/sdpa`; `vllm` is the reference to beat.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402

SHAPES = [
    # (seqlen, nheads, head_dim)
    (128, 8, 64),
    (512, 8, 64),
    (1024, 8, 64),
    (2048, 8, 64),
    (4096, 8, 64),
    (256, 16, 128),
    (1024, 16, 128),
    (2048, 16, 128),
    (4096, 16, 128),
    (512, 32, 128),
    (1024, 8, 32),
]


def load_vllm(path):
    """Load the standalone csrc/cpu extension (see the build recipe in AGENTS.md)."""
    if not path or not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("vllm_cpu_attn_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paged_case(seq, heads, dim, dtype, block=128):
    """q/k/v plus a paged cache, block table, slot map and lengths for one request.

    Same construction as the vLLM-side benchmark: every token's K/V goes into the
    paged cache, so the timed call is pure attention.
    """
    q = torch.randn(seq, heads, dim, dtype=dtype)
    k = torch.randn(seq, heads, dim, dtype=dtype)
    v = torch.randn(seq, heads, dim, dtype=dtype)
    blocks = (seq + block - 1) // block
    kc = torch.zeros(blocks, heads, block, dim, dtype=dtype)
    vc = torch.zeros_like(kc)
    bt = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
    pos = torch.arange(seq, dtype=torch.int64)
    slots = pos // block * block + pos % block
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    lens = torch.tensor([seq], dtype=torch.int32)
    return q, k, v, kc, vc, bt, slots, qsl, lens


def time_rounds(variants, rounds):
    """Median of `rounds` interleaved samples per variant."""
    for fn in variants.values():
        for _ in range(2):
            fn()
    samples = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            t0 = time.perf_counter()
            fn()
            samples[name].append((time.perf_counter() - t0) * 1e6)
    return {name: statistics.median(v) for name, v in samples.items()}


def sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=q.shape[-1] ** -0.5,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32", "both"], default="both")
    ap.add_argument("--causal", choices=["0", "1", "both"], default="1")
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""),
                    help="path to the standalone csrc/cpu .so (optional)")
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"),
                    help="vLLM ISA hint: vec|amx|neon|...")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    vllm = load_vllm(args.vllm)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}[args.dtype]
    causals = {"0": [False], "1": [True], "both": [True, False]}[args.causal]

    print(f"threads={torch.get_num_threads()} vllm={args.vllm or '-'} isa={args.isa}")
    for causal in causals:
        print(f"\ncausal={causal}   us: ours / sdpa" + (" / vllm" if vllm else "")
              + "   (GFLOP/s ours, vllm)")
        for seq, heads, dim in SHAPES:
            for dtype in dtypes:
                q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                    seq, heads, dim, dtype)
                out = torch.empty_like(q)
                q3 = q.unsqueeze(0)
                k3 = k.unsqueeze(0)
                v3 = v.unsqueeze(0)
                variants = {
                    "ours": lambda: attn.flash_attn_func(q3, k3, v3, causal=causal),
                    "sdpa": lambda: sdpa(q3, k3, v3, causal),
                }
                if vllm is not None:
                    vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                    meta = vllm.get_scheduler_metadata(
                        len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                        -1, args.isa, False, None, "auto")
                    variants["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                        q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1,
                        bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))
                m = time_rounds(variants, args.rounds)
                flops = 4.0 * heads * seq * seq * dim * (0.5 if causal else 1.0)
                gflops = {n: flops / (t * 1e-6) / 1e9 for n, t in m.items()}
                tag = "bf16" if dtype is torch.bfloat16 else "fp32"
                line = (f"  S={seq:5d} H={heads:2d} D={dim:3d} {tag}: "
                        f"{m['ours']:9.0f} / {m['sdpa']:9.0f}")
                if "vllm" in m:
                    line += f" / {m['vllm']:9.0f}"
                line += (f"   x{m['sdpa']/m['ours']:5.2f} vs sdpa "
                         f"({gflops['ours']:4.0f} GFLOP/s")
                if "vllm" in m:
                    line += (f", {m['ours']/m['vllm']:5.2f}x vllm time, "
                             f"{gflops['vllm']:4.0f} GFLOP/s vllm")
                print(line + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
