"""The slow, obviously correct reference: eager attention in float64 (the shape of
transformers' ``eager_attention_forward``: matmul, additive mask, softmax, matmul)."""

import torch


def attention(q, k, v, scale, causal=False):
    """q [b, sq, h, d], k/v [b, sk, h, d] (MHA) -> float64 [b, sq, h, d].

    ``causal`` is bottom-right aligned (query ``i`` sees keys ``j <= i + sk - sq``),
    and a row that sees no key is zero.
    """
    q, k, v = (x.double().transpose(1, 2) for x in (q, k, v))  # [b, h, s, d]
    weights = torch.matmul(q, k.transpose(2, 3)) * scale
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        mask = torch.zeros(sq, sk, dtype=torch.float64)
        mask[torch.arange(sk)[None, :] > torch.arange(sq)[:, None] + sk - sq] = float("-inf")
        weights = weights + mask
    weights = torch.softmax(weights, dim=-1).nan_to_num(0.0)  # a fully masked row is 0/0
    return torch.matmul(weights, v).transpose(1, 2).contiguous()


def check(got, ref, dtype, paged=False):
    """Max error of ``got`` against the float64 ``ref``, and its limit.

    flash_attn_func: fp32 2e-4, bf16 3e-2. Paged decode: fp32 1e-6; bf16 1e-3 beyond
    the unavoidable rounding of the exact result to bf16 (rounding alone exceeds an
    absolute 1e-3 on many cases, so the literal gate is unattainable). The fp32 limits
    hold at softmax_scale <= 1/sqrt(head_dim): fp32 softmax error grows with the
    logits (c4a-standard-8, d=128, scale 0.2 = 2.3x that: up to 2.2e-6; at the model
    scale 1.4-2.2e-7).
    """
    err = (got.double() - ref).abs()
    if dtype == torch.float32:
        return err.max().item(), (1e-6 if paged else 2e-4)
    if not paged:
        return err.max().item(), 3e-2
    rounding = (ref - ref.bfloat16().double()).abs()
    return (err - rounding).max().item(), 1e-3
