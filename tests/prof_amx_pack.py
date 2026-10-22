"""What the AMX K/V pack costs on its own, hot and cold, 1..8 threads.

`mode=2` in the AMX entry runs the pack and skips the attention loop, so the
call is the pack plus marshal plus the OpenMP region. This separates three
things the phase-ablation cannot: the pack's own rate, whether it is
latency-bound (cold) or instruction-bound (hot), and how it scales with
threads.

    OMP_NUM_THREADS=4 .venv/bin/python tests/prof_amx_pack.py

`hot` re-packs the same K/V every call (L2 resident after the first one);
`cold` rotates over 8 independent K/V sets so nothing survives in L2. If hot is
much faster than cold the pack is cold-read-bound and the fix is memory-level
parallelism, not fewer instructions.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention._kernel as K  # noqa: E402

SHAPES = [(128, 8, 64), (256, 16, 128), (512, 8, 64), (1024, 8, 64),
          (1024, 16, 128), (4096, 16, 128)]


def med(fn, rounds=15):
    for _ in range(4):
        fn()
    s = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(s), min(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shapes", default="")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in SHAPES if s in want]
    fp32 = dt == torch.float32
    assert K.amx_available(), 'no AMX here'
    print(f"{args.dtype}: pack-only us (median/min); GB/s counts K+V read once")
    for seq, heads, dim in shapes:
        kb32 = -(-seq // 32)
        kb16 = kb32 * 2
        dc32, dc16 = dim // 32, dim // 16
        slab = 1024 if fp32 else 512
        kp = torch.empty(heads * kb16 * dc32 * slab, dtype=torch.bfloat16)
        vp = torch.empty(heads * kb32 * dc16 * slab, dtype=torch.bfloat16)
        out = torch.empty(1, seq, heads, dim, dtype=dt)
        sets = [(torch.randn(1, seq, heads, dim, dtype=dt),
                 torch.randn(1, seq, heads, dim, dtype=dt)) for _ in range(8)]
        k0, v0 = sets[0]
        gb = 2.0 * seq * heads * dim * k0.element_size() / 1e9
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}:"
        for th in [int(x) for x in args.threads.split(",")]:
            K._AMX_MODE_V = 2
            torch.set_num_threads(th)

            def pk(k=k0, v=v0, th=th):
                K._AMX_ENTRY(
                    k.data_ptr(), kp.data_ptr(), vp.data_ptr(), out.data_ptr(),
                    k.data_ptr(), v.data_ptr(),
                    1, seq, seq, heads, heads, dim,
                    k.stride(0), k.stride(1), k.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    kb16 * dc32 * slab, kb32 * dc16 * slab, kb16, kb32,
                    dc32, dc16, dim ** -0.5, 1, 256, K._workers(heads * 1, 8),
                    int(fp32), 2)
            mh, lh = med(pk, args.rounds)
            i = [0]

            def pc():
                i[0] = (i[0] + 1) % 8
                k, v = sets[i[0]]
                pk(k, v)
            mc, lc = med(pc, args.rounds)
            line += (f"  {th}T hot={mh:6.1f} ({gb/mh*1e6:5.0f} GB/s)"
                     f" cold={mc:6.1f}")
            K._AMX_MODE_V = 0
        torch.set_num_threads(4)
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
