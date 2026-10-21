"""Would int8 P@V meet the bf16 parity gate? Simulated before any kernel work.

ARM's `smmla`/`usmmla` (i8mm) and x86's `amx_int8` do twice the MACs per
instruction that `bfmmla` / `tdpbf16ps` do, and P@V is ~31% of the ARM call, so
int8 P@V would be worth ~15% overall -- which is the size of the remaining gap on
most ARM rows. Nobody has tried it, and the reason to check before writing it is
accuracy: P is already rounded to bf16 (an 8-bit mantissa) for P@V, so an int8 P
with a per-row scale is the *same precision class*; the risk is V, which is bf16
with an exponent and would become int8 with a per-scale window.

This simulates exactly what the instruction computes -- int8 x int8 with int32
accumulation, then one fp32 rescale per output element -- and measures
max|diff| against an fp32 reference on the benchmark's shapes.

    .venv/bin/python tests/int8_pv_error.py

Gate: bf16 parity is max|diff| <= 3e-2 against SDPA, and the current bf16 kernel
sits at 7.8e-3..1.6e-2, so int8 P@V has to stay under ~2e-2 to leave headroom.
"""

import argparse
import sys

import torch

sys.path.insert(0, "tests")
from bench_vs_refs import SHAPES  # noqa: E402


def q_i8(x, dim):
    """Symmetric int8 quantization with one scale over `dim`.

    The scale has to be *factorable out of the dot product*, which is why the
    axis matters: for P@V = sum_j P[r,j] V[j,d], a scale on P may depend on r
    and a scale on V may depend on d, but neither may depend on j -- a per-key
    scale sits inside the sum and cannot be applied afterwards.
    """
    s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-30) / 127.0
    return (x / s).round().clamp_(-127, 127), s


def bf16_ref_emulation(q, k, v, causal):
    """What the current bf16 kernel does: bf16 x bf16, fp32 accumulate."""
    d = q.shape[-1]
    qk = (q * d ** -0.5).to(torch.bfloat16).float() @ k.to(
        torch.bfloat16).float().transpose(-1, -2)
    if causal:
        n = qk.shape[-1]
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        qk = qk.masked_fill(~mask, float("-inf"))
    p = torch.softmax(qk, dim=-1)
    return p, p.to(torch.bfloat16).float() @ v.to(torch.bfloat16).float()


def int8_pv(p, v, v_scale):
    """P@V with both operands int8, an exact integer accumulator, then rescaled.

    P: one scale per query row (over the keys). V: `v_scale` is either "dim"
    (one scale per output dim, over the keys) or "all" (one global scale) -- the
    two shapes a pack pass can produce and that still factor out of the sum.
    The products are accumulated in float64 so the simulation is exact for int8
    operands; a real kernel uses int32, which is exact to the same range.
    """
    pq, ps = q_i8(p, dim=-1)
    if v_scale == "dim":
        vq, vs = q_i8(v, dim=-2)          # (b, h, 1, dim)
    else:
        vs = v.abs().amax().clamp_min(1e-30) / 127.0
        vq = (v / vs).round().clamp_(-127, 127)
    acc = pq.double() @ vq.double()
    return (acc * (ps.double() * vs.double())).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--causal", type=int, default=1)
    args = ap.parse_args()
    causal = bool(args.causal)
    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    print(f"causal={causal}, max|diff| vs fp32 reference; gate for bf16 is 3e-2 "
          f"and the current bf16 kernel is at 7.8e-3..1.6e-2")
    print(f"{'shape':20s} {'bf16 (current)':>15s} {'int8 PV (per-dim V)':>20s} "
          f"{'int8 PV (global V)':>20s}")
    worst = {"bf16": 0.0, "row": 0.0, "key": 0.0}
    for seq, heads, dim in shapes:
        row = []
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed)
            q = torch.randn(1, heads, seq, dim, generator=g)
            k = torch.randn(1, heads, seq, dim, generator=g)
            v = torch.randn(1, heads, seq, dim, generator=g)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=dim ** -0.5)
            p, bf16_out = bf16_ref_emulation(q, k, v, causal)
            e_bf16 = (bf16_out - ref).abs().max().item()
            e_row = (int8_pv(p, v, "dim") - ref).abs().max().item()
            e_key = (int8_pv(p, v, "all") - ref).abs().max().item()
            row.append((e_bf16, e_row, e_key))
        e_bf16, e_row, e_key = (max(r[i] for r in row) for i in range(3))
        worst["bf16"] = max(worst["bf16"], e_bf16)
        worst["row"] = max(worst["row"], e_row)
        worst["key"] = max(worst["key"], e_key)
        print(f"S={seq:<5d}H={heads:<3d}D={dim:<4d} {e_bf16:15.2e} "
              f"{e_row:20.2e} {e_key:20.2e}")
    print()
    for name, label in (("bf16", "current bf16 kernel"),
                        ("row", "int8 PV, per-dim V scale"),
                        ("key", "int8 PV, one global V scale")):
        verdict = "passes" if worst[name] <= 3e-2 else "FAILS"
        print(f"{label:28s} worst {worst[name]:.2e}  {verdict} the 3e-2 gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
