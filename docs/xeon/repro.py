"""Single-case repro for the dense kernel, printing where a NaN/garbage lands."""

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

import audit_common as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402


def case(sq, sk, h, kvh, d, causal, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, sq, h, d, dtype=dtype)
    k = torch.randn(1, sk, kvh, d, dtype=dtype)
    v = torch.randn(1, sk, kvh, d, dtype=dtype)
    scale = d ** -0.5
    _kernel.set_backend("cpp")
    got = attn.flash_attn_func(q, k, v, scale, causal)
    g = h // kvh
    kr = k.repeat_interleave(g, dim=2) if g > 1 else k
    vr = v.repeat_interleave(g, dim=2) if g > 1 else v
    ref = C.ref_attention(q, kr, vr, causal, dtype=torch.float32)
    diff = (got.float() - ref).abs()
    bad = torch.isnan(got) | torch.isinf(got)
    print(f"sq={sq} sk={sk} h={h}/{kvh} d={d} causal={int(causal)}: "
          f"max|diff|={diff.max().item():.3e} bad={(bad.sum().item())}")
    if bad.any() or diff.max().item() > 1e-2:
        idx = (bad | (diff > 1e-2)).nonzero()
        rows = sorted(set(int(i[1]) for i in idx))
        print(f"  bad rows ({len(rows)}): {rows[:12]}{' ...' if len(rows) > 12 else ''}")
        print(f"  bad heads: {sorted(set(int(i[2]) for i in idx))}")
        r = rows[0]
        print(f"  row {r}: got {got[0, r, 0, :4].tolist()}")
        print(f"          ref {ref[0, r, 0, :4].tolist()}")
        print(f"  row {r}: got max {got[0, r].abs().max().item():.3e} ref max "
              f"{ref[0, r].abs().max().item():.3e}")
    return got, ref


if __name__ == "__main__":
    for sq, sk, h, kvh, d, c in [(129, 43, 4, 2, 128, True),
                                 (129, 43, 8, 1, 64, True),
                                 (65, 21, 2, 1, 64, True),
                                 (129, 43, 4, 4, 128, True),
                                 (129, 43, 4, 2, 128, False),
                                 (129, 43, 4, 2, 64, True),
                                 (200, 43, 4, 2, 128, True)]:
        case(sq, sk, h, kvh, d, c, torch.float32)
    print("--- bf16")
    for sq, sk, h, kvh, d, c in [(129, 43, 8, 1, 64, True),
                                 (65, 21, 2, 1, 64, True)]:
        case(sq, sk, h, kvh, d, c, torch.bfloat16)
