"""Writing the paged KV cache.

The cache layout is this library's own (head-major, K transposed inside the page),
so only the library writes it: ``reshape_and_cache`` (vLLM's signature), or
``flash_attn_with_kvcache(k=..., v=...)``.
"""

import torch

from . import _kernel as K


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter K/V of ``num_tokens`` tokens into the paged cache.

    ``key``/``value`` are ``[num_tokens, nheads, headdim]`` (any token stride;
    heads and dims contiguous, so chunks of a fused projection work as is), the
    caches ``[num_blocks, nheads, page_block_size, headdim]``. Token ``i`` goes to
    slot ``slot_mapping[i]`` (int64): page ``slot // page_block_size``, position
    ``slot % page_block_size``; a negative slot skips the token (padding).
    """
    if K.module().store(K.runtime(), key, value, k_cache, v_cache, slot_mapping) is None:
        key, value, slot_mapping = _check(key, value, k_cache, v_cache, slot_mapping)
        if K.module().store(K.runtime(), key, value, k_cache, v_cache, slot_mapping) is None:
            raise RuntimeError("cpu_attention: the kernel declined arguments that passed validation")


def _check(key, value, k_cache, v_cache, slots):
    """Raise the precise error for a call the kernel declined, or return key/value
    and the slots in layouts it reads in place."""
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
    if slots.dtype != torch.int64 or slots.shape != (key.shape[0],):
        raise ValueError("cpu_attention: slot_mapping must be int64 [num_tokens]")
    if key.shape[0] and int(slots.max()) >= k_cache.shape[0] * k_cache.shape[2]:
        raise ValueError("cpu_attention: slot_mapping points past the end of the cache")
    key, value, slots = (x.as_subclass(torch.Tensor) if type(x) not in (torch.Tensor, torch.nn.Parameter)
                         else x for x in (key, value, slots))
    if not (key.stride(-1) == 1 and key.stride(-2) == key.shape[2] and value.stride() == key.stride()):
        key, value = key.contiguous(), value.contiguous()
    return key, value, slots.contiguous()
