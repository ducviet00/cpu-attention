"""AMX lever A/B: on/off, v1/v2, per-head vs phase-split, in one process.

Toggles the Python-side switches (`_AMX_ENV`, `_AMX_FP32_ENV`, `_AMX_IMPL`,
`_AMX_MODE_V`) between arms, which is exactly what the kernel reads per call.
"""

import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as C  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402

SHAPES = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
          (4096, 8, 64), (256, 16, 128), (1024, 16, 128), (2048, 16, 128),
          (4096, 16, 128), (512, 32, 128), (1024, 8, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--which", default="onoff,v1v2,phase",
                    help="comma list of onoff,v1v2,phase")
    ap.add_argument("--shapes", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in shapes if s in want]
    C.burn()
    print(f"threads={torch.get_num_threads()} dtype={args.dtype} "
          f"amx={K.amx_available()} which={args.which}", flush=True)
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        runs = {}
        if "onoff" in args.which:
            def amx_on():
                K._AMX_ENV = True
                return attn.flash_attn_func(q, k, v, causal=True)

            def amx_off():
                K._AMX_ENV = False
                return attn.flash_attn_func(q, k, v, causal=True)
            amx_on(); amx_off()
            runs["amx_on"] = amx_on
            runs["amx_off"] = amx_off
        if "v1v2" in args.which:
            def v1():
                K._AMX_ENV = True
                K._AMX_IMPL = "v1"
                return attn.flash_attn_func(q, k, v, causal=True)

            def v2():
                K._AMX_ENV = True
                K._AMX_IMPL = "v2"
                return attn.flash_attn_func(q, k, v, causal=True)
            v1(); v2()
            runs["v1"] = v1
            runs["v2"] = v2
        if "phase" in args.which:
            def per_head():
                K._AMX_ENV = True
                K._AMX_IMPL = "v1"
                K._AMX_MODE_V = 0
                return attn.flash_attn_func(q, k, v, causal=True)

            def split():
                K._AMX_ENV = True
                K._AMX_IMPL = "v1"
                K._AMX_MODE_V = 512
                return attn.flash_attn_func(q, k, v, causal=True)
            per_head(); split()
            runs["per_head"] = per_head
            runs["split"] = split
        s = C.time_arms(runs, args.rounds)
        st = {n: C.stats(v) for n, v in s.items()}
        line = f"S={seq:5d} H={heads:2d} D={dim:3d}:"
        for n in runs:
            line += f" {n} {st[n]['min']:8.1f}"
        if "amx_off" in st:
            line += f"  amx_speedup {st['amx_off']['min']/st['amx_on']['min']:5.2f}"
        if "v2" in st:
            line += f"  v1/v2 {st['v2']['min']/st['v1']['min']:5.2f}"
        if "split" in st:
            line += f"  per_head/split {st['split']['min']/st['per_head']['min']:5.2f}"
        print(line, flush=True)
        K._AMX_ENV = True
        K._AMX_IMPL = "v1"
        K._AMX_MODE_V = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
