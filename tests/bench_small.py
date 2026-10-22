"""A/B the short-sequence kernel (`dense_small.mojo`) against the shipped one.

Both are the same built library; only `_SMALL_MAXK` changes, so they can be
interleaved inside one process (the only fair comparison on these boxes). The
two paths used to be bit-identical; since the short path moved to the
bf16-precision degree-2 exponential (`_exp2_bf16`) they differ by ~1 bf16 ulp,
so each is checked against SDPA inside the fixed bf16 tolerance and the
`ship/small` column is the one that decides.

    .venv/bin/python tests/bench_small.py --shapes 128:8:64,256:16:128 --rounds 15
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
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="128:8:64,256:16:128,512:8:64,1024:16:128")
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
          f"vllm={args.vllm or '-'}")
    print(f"{'S':>6} {'H':>3} {'D':>4} dtype {'ship us':>9} {'small us':>9} "
          f"{'ship/small':>10} {'o/vllm s':>8} {'o/vllm m':>8}  ship min  small min  parity")
    for seq, heads, dim in shapes:
        for dtype in dtypes:
            q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                seq, heads, dim, dtype)
            q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            out = torch.empty_like(q)
            ref = sdpa(q3, k3, v3, causal)
            got = {}

            def ship():
                K.set_small_maxk(0)
                got["ship"] = attn.flash_attn_func(q3, k3, v3, causal=causal)

            def small():
                K.set_small_maxk(100000)
                got["small"] = attn.flash_attn_func(q3, k3, v3, causal=causal)

            variants = {"ship": ship, "small": small}
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
            s = {n: [] for n in variants}
            items = list(variants.items())
            for rnd in range(args.rounds):
                # Rotate the arm order every round: with a fixed order the first
                # arm of each round is systematically faster (two identical arms
                # measured 3.4% apart on a quiet box), and every ratio in this
                # file would carry that bias.
                for name, fn in (items if rnd % 2 == 0 else items[::-1]):
                    t0 = time.perf_counter()
                    fn()
                    s[name].append((time.perf_counter() - t0) * 1e6)
            med = {n: statistics.median(v) for n, v in s.items()}
            tol = 3e-2 if dtype is torch.bfloat16 else 2e-4
            d = {n: (got[n].transpose(1, 2).float() - ref.float()).abs().max().item()
                 for n in got}
            tag = "bf16" if dtype is torch.bfloat16 else "fp32"
            line = (f"{seq:6d} {heads:3d} {dim:4d} {tag} {med['ship']:9.1f} "
                    f"{med['small']:9.1f} {med['ship'] / med['small']:9.3f}x")
            if "vllm" in med:
                line += (f" {med['ship'] / med['vllm']:8.2f} "
                         f"{med['small'] / med['vllm']:8.2f}")
            line += f" {min(s['ship']):9.1f} {min(s['small']):10.1f}"
            line += f"  {d['ship']:.1e}/{d['small']:.1e}"
            line += " SAME-EXP" if d["ship"] == d["small"] else " diff-exp"
            if d["small"] > tol:
                line += " PARITY-FAIL"
            print(line, flush=True)
    K.set_small_maxk(K._DEFAULT_SMALL_MAXK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
