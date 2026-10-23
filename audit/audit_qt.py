"""QT sweep: rows per work item, in one process (the plan key carries _QT_ENV)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as C  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402


def sweep(dtype, shapes, qts, rounds):
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=dtype)
        arms = {}
        for qt in qts:
            def f(qt=qt):
                K._QT_ENV = str(qt)
                return attn.flash_attn_func(q, q, q, causal=True)
            arms[str(qt)] = f
        for n in arms:
            arms[n]()
        s = C.time_arms(arms, rounds)
        st = {n: C.stats(v)["min"] for n, v in s.items()}
        base = st[str(qts[0])]
        best = min(st.values())
        tag = "bf16" if dtype is torch.bfloat16 else "fp32"
        line = f"{tag} {seq}:{heads}:{dim}"
        for qt in qts:
            line += f"  qt{qt} {st[str(qt)]:8.1f} ({st[str(qt)] / base:4.2f})"
        line += "   best=" + ",".join(k for k, v in st.items() if v == best)
        print(line, flush=True)
    K._QT_ENV = None


def main():
    rounds = 7
    sweep(torch.bfloat16,
          [(128, 8, 64), (256, 16, 128), (512, 8, 64), (512, 32, 128),
           (1024, 16, 128)], (6, 8, 12, 16), rounds)
    # fp32 on ARM only accepts qt multiples of the NEON width (4) and lives in
    # audit_qt_fp32.py; on x86 the AMX kernel owns fp32 and qt does not apply.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
