"""Time one shape through one path, warm clock, min-based (ablation driver).

    .venv/bin/python tests/bench_one.py --shape 128:8:64 --small 1 --rounds 21

`--small 1` forces `dense_small`'s path on (threshold 1e5), `0` forces it off.
Prints median and min; the min is the number to compare (see docs/FINDINGS.md).
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="128:8:64")
    ap.add_argument("--small", type=int, default=1)
    ap.add_argument("--qt", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    S, H, D = (int(x) for x in args.shape.split(":"))
    dt = torch.bfloat16 if D <= 64 else torch.bfloat16
    q = torch.randn(1, S, H, D, dtype=dt)
    k = torch.randn(1, S, H, D, dtype=dt)
    v = torch.randn(1, S, H, D, dtype=dt)
    # warm the clock for one second (these boxes idle at a fraction of boost)
    b = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        b @ b
    del b
    K.set_small_maxk(100000 if args.small else 0)
    if args.qt:
        K.set_qt(args.qt)
    f = lambda: attn.flash_attn_func(q, k, v, causal=True)
    f()
    f()
    ts = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        out = f()
        ts.append((time.perf_counter() - t0) * 1e6)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True, scale=D ** -0.5)
    d = (out.transpose(1, 2).float() - ref.float()).abs().max().item()
    K.set_qt(0)
    K.set_small_maxk(K._DEFAULT_SMALL_MAXK)
    print(f"{args.shape} small={args.small} qt={args.qt}  "
          f"median {statistics.median(ts):6.2f} us  min {min(ts):6.2f}  d={d:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
