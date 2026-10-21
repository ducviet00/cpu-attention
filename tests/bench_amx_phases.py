"""Split an AMX call into the K/V pack and the attention loop, plus the
per-call fixed cost, at the shapes where the call is short enough for it to
matter.

Uses the kernel's profiling bits (`CPU_ATTENTION_AMX_MODE=skip_pack|skip_attn`)
round-robin in one process, which is the only comparison these throttling boxes
allow. `fixed` is what is left when *both* are skipped: the fork-join, the tile
config and the per-call marshalling.

    .venv/bin/python tests/bench_amx_phases.py --dtype bf16
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import SHAPES, sdpa  # noqa: E402

MODES = {"both": 0, "skip_pack": 1, "skip_attn": 2, "skip_all": 3}


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
            f()
    s = {n: [] for n in fns}
    for _ in range(rounds):
        for n, f in fns.items():
            t0 = time.perf_counter()
            f()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: statistics.median(v) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shapes", default="128:8:64,256:8:64,512:8:64,256:16:128,512:32:128,1024:16:128")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
    shapes = [s for s in SHAPES if s in want] or sorted(want)
    print(f"threads={torch.get_num_threads()} {args.dtype} causal us  "
          f"(pack = both-skip_pack, attn = both-skip_attn, fixed = skip_all)")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dt)
        k = torch.randn(1, seq, heads, dim, dtype=dt)
        v = torch.randn(1, seq, heads, dim, dtype=dt)
        fns = {}
        for name, bit in MODES.items():
            def f(bit=bit):
                K._AMX_MODE_V = bit
                K._dense(q, k, v, dim ** -0.5, True)
            fns[name] = f
        fns["sdpa"] = lambda: sdpa(q, k, v, True)
        m = bench(fns, args.rounds)
        K._AMX_MODE_V = 0
        pack = m["both"] - m["skip_pack"]
        attn = m["both"] - m["skip_attn"]
        fixed = m["skip_all"]
        print(f"  S={seq:5d} H={heads:2d} D={dim:3d}: total={m['both']:6.0f} "
              f"pack={pack:5.0f} ({100*pack/m['both']:4.1f}%) "
              f"attn={attn:6.0f} ({100*attn/m['both']:4.1f}%) "
              f"fixed={fixed:5.1f} "
              f"resid={m['both']-pack-attn-fixed:5.0f} "
              f"o/sdpa {m['both']/m['sdpa']:4.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
