"""Where the AMX dense call's *non-kernel* time goes (the "fixed" cost).

The scorecard charges us for everything inside ``flash_attn_func``: the checked
wrapper, the plan/scratch lookups, the output allocation, the 36-argument
extension marshal, and the OpenMP region. At S=128 that was measured at ~19 us
around a ~42 us kernel body, i.e. it is worth more than the K/V pack.

    OMP_NUM_THREADS=4 .venv/bin/python tests/prof_amx_callpath.py --dtype bf16

Prints, per shape, the median us of each level, interleaved in one process:

  pub     flash_attn_func (the scorecard's "ours" arm)
  dense   _dense() -- the same minus the checked wrapper
  ext0    _AMX_ENTRY(.., mode=0) with the same Python prep, nothing skipped
  ext3    ... mode=3: skip pack and attention, i.e. prep + marshal + region
  noop    an empty extension call (marshal only, no region): the Python side
  alloc   torch.empty_like(q) alone
  pre     the Python prep in _amx_dense without the extension call

Nothing here is a performance claim about the kernel; it exists to size the
overhead the kernel work has to beat.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention._kernel as K  # noqa: E402
import cpu_attention as attn  # noqa: E402

SHAPES = [(128, 8, 64), (256, 16, 128), (512, 8, 64), (1024, 8, 64),
          (1024, 16, 128), (4096, 16, 128)]


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
            f()
    s = {n: [] for n in fns}
    names = list(fns)
    for r in range(rounds):
        for n in names[r % len(names):] + names[:r % len(names)]:
            t0 = time.perf_counter()
            fns[n]()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: (statistics.median(v), min(v)) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--shapes", default="")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in SHAPES if s in want]
    print(f"threads={torch.get_num_threads()} {args.dtype}  (median / min, us)")
    for seq, heads, dim in shapes:
        q1 = torch.randn(1, seq, heads, dim, dtype=dt)
        k1 = torch.randn_like(q1)
        v1 = torch.randn_like(q1)
        q, k, v = q1, k1, v1
        scale = dim ** -0.5
        fp32 = dt == torch.float32
        kb32 = -(-seq // 32)
        kb16 = kb32 * 2
        dc32, dc16 = dim // 32, dim // 16
        slab = 1024 if fp32 else 512
        kp = K._scratch("amx_kp", (heads * kb16 * dc32 * slab,), torch.bfloat16)
        vp = K._scratch("amx_vp", (heads * kb32 * dc16 * slab,), torch.bfloat16)
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        qs, ks, vs, os_ = q.stride(), k.stride(), v.stride(), out.stride()
        tasks = heads * (-(-seq // 32))

        def ext(mode):
            K._AMX_ENTRY(
                q.data_ptr(), kp.data_ptr(), vp.data_ptr(), out.data_ptr(),
                k.data_ptr(), v.data_ptr(),
                1, seq, seq, heads, heads, dim,
                qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], vs[0], vs[1], vs[2],
                os_[0], os_[1], os_[2],
                kb16 * dc32 * slab, kb32 * dc16 * slab, kb16, kb32, dc32, dc16,
                scale, 1, 256, K._workers(tasks, 8), int(fp32), mode)

        fns = {
            "pub": lambda: attn.flash_attn_func(q1, k1, v1, causal=True),
            "dense": lambda: K._dense(q, k, v, scale, True),
            "ext0": lambda: ext(0),
            "ext3": lambda: ext(3),
            "alloc": lambda: torch.empty_like(q, memory_format=torch.contiguous_format),
        }
        m = bench(fns, args.rounds)
        tot = m["pub"][0]
        print(f"  S={seq:5d} H={heads:2d} D={dim:3d}: pub={tot:6.1f} "
              f"dense={m['dense'][0]:6.1f} ext0={m['ext0'][0]:6.1f} "
              f"ext3={m['ext3'][0]:6.1f} alloc={m['alloc'][0]:5.2f} | "
              f"wrapper={tot-m['dense'][0]:5.2f} "
              f"prep={m['dense'][0]-m['ext0'][0]:5.2f} "
              f"region+pack+attn={m['ext3'][0]:5.2f} "
              f"(min pub={m['pub'][1]:.1f} ext0={m['ext0'][1]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
