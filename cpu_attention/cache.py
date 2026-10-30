"""Writing the paged KV cache.

The cache layout is this library's own (head-major, K transposed inside the page),
so only the library writes it: ``reshape_and_cache`` (vLLM's signature), or
``flash_attn_with_kvcache(k=..., v=...)``.
"""

import torch

from . import _kernel as K

_PLANS = {}  # geometry key -> (plan, copy)


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter K/V of ``num_tokens`` tokens into the paged cache.

    ``key``/``value`` are ``[num_tokens, nheads, headdim]`` (any token stride;
    heads and dims contiguous, so chunks of a fused projection work as is), the
    caches ``[num_blocks, nheads, page_block_size, headdim]``. Token ``i`` goes to
    slot ``slot_mapping[i]`` (int64): page ``slot // page_block_size``, position
    ``slot % page_block_size``; a negative slot skips the token (padding).
    """
    geom = (key.shape, key.stride(), key.dtype, value.shape, value.stride(), value.dtype,
            k_cache.shape, k_cache.stride(), k_cache.dtype, v_cache.shape, v_cache.stride(),
            v_cache.dtype, slot_mapping.shape, slot_mapping.stride(), slot_mapping.dtype,
            torch.get_num_threads())
    entry = _PLANS.get(geom)
    if entry is None:
        entry = _PLANS[geom] = _plan(key, value, k_cache, v_cache, slot_mapping)
    plan, copy = entry
    if not key.shape[0]:
        return
    if int(slot_mapping.max()) >= k_cache.shape[0] * k_cache.shape[2]:
        raise ValueError("cpu_attention: slot_mapping points past the end of the cache")
    if copy:
        key, value = key.contiguous(), value.contiguous()
    K.run_paged(plan, None, key, value, k_cache, v_cache, None, slot_mapping, None, None, 1.0)


def _plan(key, value, k_cache, v_cache, slots):
    """Validate one geometry and plan it: (plan, copy)."""
    if key.dim() != 3 or value.shape != key.shape:
        raise ValueError("cpu_attention: key/value must both be [num_tokens, nheads, headdim]")
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("cpu_attention: k_cache/v_cache must be "
                         "[num_blocks, nheads, page_block_size, headdim] and match")
    if key.shape[1:] != (k_cache.shape[1], k_cache.shape[3]):
        raise ValueError("cpu_attention: key/value heads or head_dim do not match the cache")
    if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("cpu_attention: the paged caches must be contiguous")
    if key.dtype not in K.DTYPES or not key.dtype == value.dtype == k_cache.dtype == v_cache.dtype:
        raise ValueError("cpu_attention: key/value and the caches must share a dtype (fp32 or bf16)")
    if key.shape[2] not in K.HEAD_DIMS or k_cache.shape[2] % 16:
        raise ValueError("cpu_attention: head_dim must be 32/64/128 and the page a multiple of 16")
    if slots.dtype != torch.int64 or slots.shape != (key.shape[0],) or not slots.is_contiguous():
        raise ValueError("cpu_attention: slot_mapping must be a contiguous int64 [num_tokens]")
    copy = not (key.stride(-1) == 1 and key.stride(-2) == key.shape[2]
                and value.stride() == key.stride())
    if copy:
        key = key.contiguous()
    return K.paged_plan(None, key, k_cache, 0, True, False), copy
