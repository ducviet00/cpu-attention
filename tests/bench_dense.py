"""Dense attention: cpu_attention.flash_attn_func vs torch SDPA.

    uv run python tests/bench_dense.py            # both dtypes
    uv run python tests/bench_dense.py --fp32     # fp32 only
    THREADS=8 CPU_ATTENTION_QT=8 uv run python tests/bench_dense.py

Reports the median of a fixed time budget per case, two sweeps back to back (the
band matters: these boxes drift). ``GFLOP/s`` counts the causal half of QK and PV.
"""

import os
import sys
import time
from functools import partial

import torch

import cpu_attention as attn


def _time(fn, budget_s=0.15, min_iters=5) -> float:
    """Median microseconds per call, enough iterations to fill ``budget_s``."""
    for _ in range(3):
        fn()
    iters, total = 0, 0.0
    ts = []
    while total < budget_s or iters < min_iters:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        ts.append(dt * 1e6)
        total += dt
        iters += 1
    ts.sort()
    return ts[len(ts) // 2]


def sdpa(q, k, v, causal):
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    heads, kv_heads = q.shape[2], k.shape[2]
    if kv_heads != heads:
        kk = kk.repeat_interleave(heads // kv_heads, 1)
        vv = vv.repeat_interleave(heads // kv_heads, 1)
    return torch.nn.functional.scaled_dot_product_attention(
        qq, kk, vv, is_causal=causal, scale=q.shape[3] ** -0.5
    )


CASES = [
    # (batch, seqlen, nheads, nheads_k, head_dim)
    (1, 128, 8, 8, 64),
    (1, 512, 8, 8, 64),
    (1, 1024, 8, 8, 64),
    (1, 2048, 8, 8, 64),
    (1, 4096, 8, 8, 64),
    (1, 256, 16, 16, 128),
    (1, 1024, 16, 16, 128),
    (1, 2048, 16, 16, 128),
    (1, 512, 32, 32, 128),
    (1, 1024, 8, 8, 32),
    (2, 1024, 8, 8, 64),
    (1, 512, 16, 4, 64),  # GQA: 4 query heads per KV head
]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", torch.get_num_threads())))
    dtypes = [torch.float32] if "--fp32" in sys.argv else []
    if "--bf16" in sys.argv:
        dtypes = [torch.bfloat16]
    if not dtypes:
        dtypes = [torch.bfloat16, torch.float32]
    print(
        f"threads={torch.get_num_threads()} qt={os.environ.get('CPU_ATTENTION_QT', 'auto')}"
    )

    for causal in (True, False):
        print(f"\ncausal={causal}")
        head = f"{'case':26s}"
        for dt in dtypes:
            name = "bf16" if dt is torch.bfloat16 else "fp32"
            head += f" | {name + ' ours':>11s} {name + ' sdpa':>11s} {'x':>5s}"
        print(head)
        print("-" * len(head))
        for batch, seq, heads, kv_heads, dim in CASES:
            line = f"B={batch} S={seq:5d} H={heads:2d} Hk={kv_heads:2d} D={dim:3d}"
            flops = 4.0 * batch * heads * seq * seq * dim * (0.5 if causal else 1.0)
            row = []
            for dt in dtypes:
                q = torch.randn(batch, seq, heads, dim, dtype=dt)
                k = torch.randn(batch, seq, kv_heads, dim, dtype=dt)
                v = torch.randn(batch, seq, kv_heads, dim, dtype=dt)
                ours = _time(partial(attn.flash_attn_func, q, k, v, causal=causal))
                ref = _time(partial(sdpa, q, k, v, causal))
                row += [ours, ref]
            text = ""
            for i in range(0, len(row), 2):
                text += (
                    f" | {row[i]:11.0f} {row[i + 1]:11.0f} {row[i + 1] / row[i]:5.2f}"
                )
            print(line + text)

    # One absolute number to compare across machines: GFLOP/s at a fixed shape.
    print("\nGFLOP/s (causal, B=1 S=2048 H=8 Hk=8 D=64 / H=16 D=128):")
    for dt in dtypes:
        for heads, dim in ((8, 64), (16, 128)):
            q = torch.randn(1, 2048, heads, dim, dtype=dt)
            k = torch.randn(1, 2048, heads, dim, dtype=dt)
            v = torch.randn(1, 2048, heads, dim, dtype=dt)
            us = _time(partial(attn.flash_attn_func, q, k, v, causal=True), 0.3)
            flops = 4.0 * heads * 2048 * 2048 * dim * 0.5
            print(
                f"  {'bf16' if dt is torch.bfloat16 else 'fp32'} H={heads:2d} D={dim:3d}: "
                f"{flops / (us * 1e-6) / 1e9:7.0f} GFLOP/s ({us:.0f} us)"
            )


if __name__ == "__main__":
    main()
