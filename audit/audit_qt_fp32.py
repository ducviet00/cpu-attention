"""FP32 QT sweep on ARM (legal values are multiples of the NEON width, 4)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as C  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402

C.burn()
for seq, heads, dim in [(128, 8, 64), (512, 8, 64), (1024, 8, 64),
                        (1024, 16, 128), (2048, 16, 128)]:
    q = torch.randn(1, seq, heads, dim, dtype=torch.float32)
    arms = {}
    for qt in (4, 8, 12):
        def f(qt=qt):
            K._QT_ENV = str(qt)
            return attn.flash_attn_func(q, q, q, causal=True)
        arms[str(qt)] = f
    for n in arms:
        arms[n]()
    s = C.time_arms(arms, 7)
    st = {n: C.stats(v)["min"] for n, v in s.items()}
    base = st["4"]
    print(f"fp32 {seq}:{heads}:{dim}  " + "  ".join(
        f"qt{n} {st[n]:9.1f} ({st[n] / base:4.2f})" for n in ("4", "8", "12")),
        flush=True)
