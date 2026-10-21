"""Can fp32 attention run on AMX by splitting each operand into two bf16 halves?

AMX has no fp32 mode and there is no 512-bit fp32 instruction beyond FMA, so the
only structural win left for fp32 on a Xeon is to do the two products in bf16
and pay for the precision with extra terms::

    x  = hi + lo,  hi = bf16(x), lo = bf16(x - hi)          |lo| <= 2^-8 |x|
    x@y ~= hi_x@hi_y + hi_x@lo_y + lo_x@hi_y                 (drops lo@lo ~ 2^-16)

This script measures the error of exactly that arithmetic -- each product is
`bf16 x bf16 -> fp32 accumulate`, which is what `tdpbf16ps` does, and the sum of
the three products is fp32 -- against an fp32 reference, on the benchmark's
shapes. It needs no AMX hardware.

    .venv/bin/python tests/three_term_error.py            # all shapes
    .venv/bin/python tests/three_term_error.py --shapes 1024:16:128

Gate: our parity test allows max|diff| <= 2e-4 against SDPA in fp32, and the
current fp32 kernel sits at 1e-6, so the whole budget is available.
"""

import argparse
import sys

import torch

sys.path.insert(0, "tests")
from bench_vs_refs import SHAPES  # noqa: E402


def split(x, terms):
    """(hi, lo, ...) bf16-valued fp32 halves of ``x``, ``terms`` deep."""
    out = []
    rest = x
    for _ in range(terms - 1):
        hi = rest.to(torch.bfloat16).float()
        out.append(hi)
        rest = rest - hi
    out.append(rest.to(torch.bfloat16).float())
    return out


def dot(a, b, terms):
    ah = split(a, terms)
    bh = split(b, terms)
    acc = None
    for i, x in enumerate(ah):
        for j, y in enumerate(bh):
            # drop the product with the most error first: lo*lo, then lo*hi
            if i + j > terms - 1:
                continue
            p = x @ y
            acc = p if acc is None else acc + p
    return acc


def reference(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=q.shape[-1] ** -0.5)


def emulated(q, k, v, causal, terms, pv_terms=None):
    d = q.shape[-1]
    qk = dot(q * d ** -0.5, k.transpose(-1, -2), terms)
    if causal:
        n = qk.shape[-1]
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        qk = qk.masked_fill(~mask, float("-inf"))
    p = torch.softmax(qk, dim=-1)
    return dot(p, v, terms if pv_terms is None else pv_terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--terms", default="2,3,4")
    args = ap.parse_args()
    causal = bool(args.causal)
    terms = [int(t) for t in args.terms.split(",")]
    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    print(f"causal={causal}  max|diff| vs fp32 reference (gate 2e-4), "
          f"worst of {args.seeds} seeds")
    print(f"{'shape':20s} " + "  ".join(f"terms={t}" for t in terms))
    worst = {t: 0.0 for t in terms}
    for seq, heads, dim in shapes:
        row = []
        for t in terms:
            err = 0.0
            for seed in range(args.seeds):
                g = torch.Generator().manual_seed(seed)
                q = torch.randn(1, heads, seq, dim, generator=g)
                k = torch.randn(1, heads, seq, dim, generator=g)
                v = torch.randn(1, heads, seq, dim, generator=g)
                ref = reference(q, k, v, causal)
                got = emulated(q, k, v, causal, t)
                err = max(err, (got - ref).abs().max().item())
            row.append(err)
            worst[t] = max(worst[t], err)
        print(f"S={seq:<5d}H={heads:<3d}D={dim:<4d} "
              + "  ".join(f"{e:8.2e}" for e in row))
    print()
    for t in terms:
        print(f"terms={t}: worst {worst[t]:.2e} "
              f"({'PASS' if worst[t] <= 2e-4 else 'FAIL'} vs the 2e-4 gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
