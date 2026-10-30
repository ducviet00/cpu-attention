"""The slow, obviously correct reference: numpy in float64, one (batch, head) at a time."""

import numpy as np
import torch


def f64(t):
    return t.detach().double().numpy()


def attention(q, k, v, scale, causal=False):
    """q [b, sq, h, d], k/v [b, sk, hk, d] (numpy float64) -> [b, sq, h, d].

    Query head ``i`` reads kv head ``i // (h // hk)``; ``causal`` is bottom-right
    aligned (query ``i`` sees keys ``j <= i + sk - sq``), and a row that sees no
    key is zero.
    """
    b, sq, h, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    out = np.zeros((b, sq, h, d))
    for bi in range(b):
        for hi in range(h):
            kh = hi // (h // hk)
            s = q[bi, :, hi] @ k[bi, :, kh].T * scale
            if causal:
                s[np.arange(sk)[None, :] > np.arange(sq)[:, None] + sk - sq] = -np.inf
            for i in range(sq):
                if np.isinf(s[i]).all():
                    continue
                p = np.exp(s[i] - s[i].max())
                out[bi, i, hi] = p @ v[bi, :, kh] / p.sum()
    return out


def check(got, ref, dtype, paged=False):
    """Max error of ``got`` (a torch tensor) against the float64 ``ref``, and its limit.

    flash_attn_func: fp32 2e-4, bf16 3e-2. Paged decode: fp32 1e-6; bf16 1e-3 beyond
    the unavoidable rounding of the exact result to bf16 (rounding alone exceeds an
    absolute 1e-3 on many cases, so the literal gate is unattainable). The fp32 limits
    hold at softmax_scale <= 1/sqrt(head_dim): fp32 softmax error grows with the
    logits (c4a-standard-8, d=128, scale 0.2 = 2.3x that: up to 2.2e-6; at the model
    scale 1.4-2.2e-7).
    """
    err = np.abs(f64(got) - ref)
    if dtype == torch.float32:
        return err.max(), (1e-6 if paged else 2e-4)
    if not paged:
        return err.max(), 3e-2
    rounding = np.abs(ref - f64(torch.from_numpy(ref).bfloat16()))
    return (err - rounding).max(), 1e-3
