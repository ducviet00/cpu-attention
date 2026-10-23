"""NaN rate for shapes where some query rows can see no key at all.

`seq_q > seq_k` puts the first rows' causal limit below key 0: those rows have an
all-masked softmax and the whole reduction is over an empty set, so the only
correct answer is 0. Random data is drawn fresh each trial, so the rate has to be
measured, not observed once.
"""

import sys
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402

SHAPES = [(129, 43, 8, 1, 64), (129, 43, 4, 2, 128), (65, 21, 2, 1, 64),
          (200, 43, 4, 2, 128), (130, 66, 8, 8, 64)]


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    K.set_backend("cpp")
    for (sq, sk, h, kvh, d) in SHAPES:
        for dtype in (torch.float32, torch.bfloat16):
            nan = 0
            for t in range(trials):
                q = torch.randn(1, sq, h, d, dtype=dtype)
                k = torch.randn(1, sk, kvh, d, dtype=dtype)
                v = torch.randn(1, sk, kvh, d, dtype=dtype)
                out = attn.flash_attn_func(q, k, v, d ** -0.5, True)
                bad = torch.isnan(out.float()) | torch.isinf(out.float())
                if bad.any():
                    nan += 1
                    if nan == 1:
                        idx = bad.nonzero()
                        rows = sorted(set(int(i[1]) for i in idx))
                        heads = sorted(set(int(i[2]) for i in idx))
                        print(f"    first bad: rows {rows[:8]} heads {heads} "
                              f"n={int(bad.sum())}")
            print(f"sq={sq} sk={sk} h={h}/{kvh} d={d} {str(dtype)[6:]:8s}: "
                  f"{nan}/{trials} trials with NaN/inf", flush=True)


if __name__ == "__main__":
    main()
