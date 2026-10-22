"""A/B the per-head dispatch against the (head, q-pair) round robin, in-process.

Both variants are the same built kernel; only the work-item shape changes, so
they can be interleaved inside one process (the only fair way on these boxes).

    .venv/bin/python tests/bench_head.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --isa neon --shapes 128:8:64,512:8:64,1024:8:64 --dtype both --rounds 7

Columns: `base` = shipped dispatch, `head` = one thread owns whole heads, and
`head/base` is the ratio the change buys (<1 is faster). `d=` is max|diff| vs
SDPA for each variant -- the per-head path must be bit-identical to the base one
(same kernel, same order of reductions per row), so any nonzero difference there
is a bug, not a tolerance question.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import SHAPES, load_vllm, paged_case, sdpa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32", "both"], default="both")
    ap.add_argument("--causal", choices=["0", "1"], default="1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--maxk", type=int, default=100000)
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    causal = args.causal == "1"
    vllm = load_vllm(args.vllm)
    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}[args.dtype]

    _burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        _burn @ _burn
    del _burn

    print(f"threads={torch.get_num_threads()} causal={causal} "
          f"vllm={args.vllm or '-'} maxk={args.maxk}")
    header = (f"{'S':>6} {'H':>3} {'D':>4} dtype  {'base us':>9} {'head us':>9} "
              f"{'head/base':>9}  {'o/vllm b':>8} {'o/vllm h':>8}  d_base d_head")
    print(header)
    for seq, heads, dim in shapes:
        for dtype in dtypes:
            q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                seq, heads, dim, dtype)
            q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            out = torch.empty_like(q)
            ref = sdpa(q3, k3, v3, causal)
            got = {}

            def base():
                K.set_head_maxk(0)
                got["base"] = attn.flash_attn_func(q3, k3, v3, causal=causal)

            def head():
                K.set_head_maxk(args.maxk)
                got["head"] = attn.flash_attn_func(q3, k3, v3, causal=causal)

            variants = {"base": base, "head": head}
            if vllm is not None:
                vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                meta = vllm.get_scheduler_metadata(
                    len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                    -1, args.isa, False, None, "auto")
                variants["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                    q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1,
                    bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))
            for fn in variants.values():
                fn()
                fn()
            samples = {n: [] for n in variants}
            for _ in range(args.rounds):
                for name, fn in variants.items():
                    t0 = time.perf_counter()
                    fn()
                    samples[name].append((time.perf_counter() - t0) * 1e6)
            med = {n: statistics.median(v) for n, v in samples.items()}
            tol = 3e-2 if dtype is torch.bfloat16 else 2e-4
            d = {}
            for name in ("base", "head"):
                diff = (got[name].transpose(1, 2).float()
                        - ref.float()).abs().max().item()
                d[name] = diff
            tag = "bf16" if dtype is torch.bfloat16 else "fp32"
            line = (f"{seq:6d} {heads:3d} {dim:4d} {tag}  {med['base']:9.1f} "
                    f"{med['head']:9.1f} {med['base']/med['head']:8.3f}x")
            if "vllm" in med:
                line += (f"  {med['base']/med['vllm']:8.2f} "
                         f"{med['head']/med['vllm']:8.2f}")
            line += f"  {d['base']:.1e}{'' if d['base'] <= tol else '!'}"
            line += f" {d['head']:.1e}{'' if d['head'] <= tol else '!'}"
            if d["base"] != d["head"]:
                line += "  NOT-IDENTICAL"
            print(line, flush=True)
    K.set_head_maxk(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
