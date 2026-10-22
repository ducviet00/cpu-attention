"""Would int8 Q@K^T meet the bf16 parity gate, and on which inputs does it break?

The ARM `smmla` (i8mm) does twice the MACs per instruction that `bfmmla` does,
and Q@K^T is half the matmul in a dense attention call, so an int8 Q@K^T with a
bf16 P@V is worth ~15-20% -- which is the size of the remaining gap on the ARM
rows. The kernel agent measured 19% and a parity failure at 3.1e-2 against a
3e-2 gate, concentrated in the first ~128 rows; this script asks the question
that decides whether that path may ship: *how much margin is there, and what
input breaks it?*

It simulates exactly what the instruction computes -- int8 x int8 with an exact
int32 accumulator, one fp32 rescale per logit -- and leaves the rest of the
pipeline as the current bf16 kernel runs it (P rounded to bf16, V bf16, fp32
accumulate), so the number it reports is comparable to the bf16 kernel's own
7.8e-3..1.6e-2 against SDPA.

Scales are the ones a kernel can actually factor out of the dot product: one per
query row for Q (it multiplies the row's logits) and either one per key or one
per 64-key tile for K (both multiply a single logit, or a whole tile of them).

    .venv/bin/python tests/int8_qk_error.py                 # benchmark shapes
    .venv/bin/python tests/int8_qk_error.py --stress        # + adversarial inputs

Gate: bf16 parity is max|diff| <= 3e-2 vs SDPA. A scheme that only passes on
`torch.randn` is not shippable: real q/k have per-channel outliers, and a
per-row max scale loses bits exactly when one channel dominates.
"""

import argparse
import sys

import torch

sys.path.insert(0, "tests")
from bench_vs_refs import SHAPES  # noqa: E402

TOL = 3e-2


def q_sym(x, dim):
    """Symmetric int8 with one scale over `dim` (the axis a kernel can factor)."""
    s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-30) / 127.0
    return (x / s).round().clamp_(-127, 127), s


def q_tile(x, tile):
    """Symmetric int8 with one scale per `tile` keys along dim -2 (keys)."""
    orig = x
    n = x.shape[-2]
    pad = (-n) % tile
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    blocks = x.reshape(*x.shape[:-2], -1, tile, x.shape[-1])
    s = blocks.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-30) / 127.0
    q = (blocks / s).round().clamp_(-127, 127)
    q = q.reshape(x.shape)[..., :n, :]
    # one scale per key, flattened back to (..., n, 1)
    sk = s.squeeze(-1).repeat_interleave(tile, dim=-2)[..., :n, :]
    return q, sk


def bf16_ref(q, k, v, causal, scale):
    """What the current bf16 kernel computes: bf16 x bf16, fp32 accumulate."""
    qk = (q * scale).to(torch.bfloat16).float() @ k.to(torch.bfloat16).float().mT
    if causal:
        n = qk.shape[-1]
        qk = qk.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    p = torch.softmax(qk, dim=-1)
    return p.to(torch.bfloat16).float() @ v.to(torch.bfloat16).float()


def int8_qk(q, k, v, causal, scale, k_scale, hybrid_rows=0):
    """int8 Q@K^T (exact int32 accumulate, fp32 rescale), bf16 P@V."""
    q8, sq = q_sym(q, dim=-1)                       # (b, h, s, d) -> per row
    if k_scale == "key":
        k8, sk = q_sym(k, dim=-1)                   # per key
    else:
        k8, sk = q_tile(k, 64)                      # per 64-key tile
    qk = (q8.double() @ k8.double().mT) * (sq.double() @ sk.double().mT) * scale
    if hybrid_rows:
        ref_qk = (q * scale).to(torch.bfloat16).float() @ k.to(
            torch.bfloat16).float().mT
        qk[..., :hybrid_rows, :] = ref_qk[..., :hybrid_rows, :]
    if causal:
        n = qk.shape[-1]
        qk = qk.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    p = torch.softmax(qk.float(), dim=-1)
    return p.to(torch.bfloat16).float() @ v.to(torch.bfloat16).float()


def cases(kind):
    """Yield (name, q, k, v, causal) inputs of the given kind, at one shape.

    Every family is rescaled so the fp32 logits have std ~2 (a realistic
    attention range: max |logit| ~ 10). Unscaled 30x-outlier inputs produce
    logits of ~100, where the fp32 reference itself is a hard argmax and *any*
    rounding of q/k -- bf16 included -- picks a different key, which measures
    saturation rather than the quantizer under test.
    """
    g = torch.Generator().manual_seed(0)
    seq, heads, dim = 512, 8, 64
    for causal in (True, False):
        q = torch.randn(heads, seq, dim, generator=g)
        k = torch.randn(heads, seq, dim, generator=g)
        v = torch.randn(heads, seq, dim, generator=g)
        if kind == "plain":
            pass
        elif kind == "outlier":
            # Per-channel scale spread: activation outliers are per-channel in
            # real models, and a per-row max scale is dominated by the biggest
            # channel, which costs the other channels ~4 bits.
            sc = torch.ones(dim)
            sc[0] = 20.0
            sc[1:4] = 5.0
            q = q * sc
            k = k * sc
        elif kind == "peaked":
            # Repeated tokens: q == k on a few rows is what padding and copy
            # heads look like, and it is the peaked case the gate is sensitive to.
            k = k.clone()
            k[:2] = q[:2]
        elif kind == "rowscale":
            # Query rows whose magnitude spans 100x.
            r = torch.logspace(0, 2, seq).reshape(seq, 1)
            q = q * r
        elif kind == "heavy":
            # Student-t (df=1.5), heavy tailed across dims and rows.
            q = torch.randn(heads, seq, dim, generator=g) / torch.sqrt(
                torch.rand(heads, seq, 1, generator=g) * 0.6 + 0.05)
        # Normalise the *whole* problem so the logit std is ~2 in every family.
        logits = (q @ k.mT) * (dim ** -0.5)
        sigma = logits.reshape(-1).std().clamp_min(1e-6)
        yield f"{kind} causal={int(causal)}", q, k * (2.0 / sigma), v, causal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--stress", action="store_true",
                    help="also run the adversarial input families at S=512")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    print(f"max|diff| vs fp32 SDPA. bf16 gate {TOL:.0e}; the bf16 kernel alone "
          f"sits at 7.8e-3..1.6e-2.")
    print(f"{'case':34s} {'bf16':>9s} {'i8 per-key':>11s} {'i8 per-tile':>12s} "
          f"{'i8 hyb256':>10s}")
    worst = {}

    def report(name, q, k, v, causal):
        scale = q.shape[-1] ** -0.5
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=scale)
        vals = [
            (bf16_ref(q, k, v, causal, scale) - ref).abs().max().item(),
            (int8_qk(q, k, v, causal, scale, "key") - ref).abs().max().item(),
            (int8_qk(q, k, v, causal, scale, "tile") - ref).abs().max().item(),
            (int8_qk(q, k, v, causal, scale, "key", hybrid_rows=256)
             - ref).abs().max().item(),
        ]
        for i, key in enumerate(("bf16", "key", "tile", "hyb")):
            worst[key] = max(worst.get(key, 0.0), vals[i])
        flag = "" if max(vals[1:]) <= TOL else "  <-- int8 FAILS"
        print(f"{name:34s} {vals[0]:9.2e} {vals[1]:11.2e} {vals[2]:12.2e} "
              f"{vals[3]:10.2e}{flag}")

    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    for seq, heads, dim in shapes:
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed)
            q = torch.randn(heads, seq, dim, generator=g)
            k = torch.randn(heads, seq, dim, generator=g)
            v = torch.randn(heads, seq, dim, generator=g)
            report(f"S={seq} H={heads} D={dim} s={seed}", q, k, v, True)

    if args.stress:
        print("\n-- adversarial inputs (S=512 H=8 D=64) --")
        for kind in ("plain", "outlier", "peaked", "rowscale", "heavy"):
            for name, q, k, v, causal in cases(kind):
                report(name, q, k, v, causal)

    print()
    for key, label in (("bf16", "current bf16 kernel"), ("key", "int8 QK, per-key K scale"),
                       ("tile", "int8 QK, per-64-key K scale"),
                       ("hyb", "int8 QK, bf16 on rows 0-255")):
        verdict = "passes" if worst[key] <= TOL else "FAILS"
        print(f"{label:32s} worst {worst[key]:.2e}  {verdict} the {TOL:.0e} gate"
              f"  ({TOL / max(worst[key], 1e-30):.1f}x margin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
