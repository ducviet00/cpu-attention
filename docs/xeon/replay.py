"""Replay specific `fuzz_dense` cases and print the reference against our output.

`fuzz_dense.py` reports `max|diff|` but not which side is NaN. This replays the
same `random.Random(0)` stream and prints both.
"""

import os
import random
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import cpu_attention as attn  # noqa: E402
import fuzz_dense as F  # noqa: E402


def main(cases, want_idx):
    rng = random.Random(0)
    for i in range(cases):
        dtype = rng.choice([torch.float32, torch.bfloat16])
        q, k, v, causal, scale, kind = F.make_case(rng, 900, dtype)
        if i not in want_idx:
            continue
        want = F.reference(q, k, v, causal, scale).float()
        got = attn.flash_attn_func(q, k, v, softmax_scale=scale,
                                   causal=causal).float()
        d = (got - want).abs()
        print(f"case {i} {kind} q={tuple(q.shape)} k={tuple(k.shape)} "
              f"causal={int(causal)}")
        print(f"  want: nan={int(torch.isnan(want).sum().item())} "
              f"inf={int(torch.isinf(want).sum().item())} "
              f"max={want.nan_to_num().abs().max().item():.3e}")
        print(f"  got : nan={int(torch.isnan(got).sum().item())} "
              f"inf={int(torch.isinf(got).sum().item())} "
              f"max={got.nan_to_num().abs().max().item():.3e}")
        bad = torch.isnan(d) | torch.isinf(d)
        print(f"  max|diff|={d.max().item():.3e}  non-finite diffs="
              f"{int(bad.sum().item())}")
        if bad.any():
            idx = bad.nonzero()[0]
            b, s, h, dd = (int(x) for x in idx)
            print(f"  first at b={b} s={s} h={h}: want={want[b, s, h, :4].tolist()}")
            print(f"                          got ={got[b, s, h, :4].tolist()}")
            print(f"  row max: want={want[b, s, h].abs().max().item():.3e} "
                  f"got={got[b, s, h].abs().max().item():.3e}")


if __name__ == "__main__":
    idx = [int(x) for x in (sys.argv[1] or "46,53,55").split(",")]
    main(max(idx) + 1, set(idx))
