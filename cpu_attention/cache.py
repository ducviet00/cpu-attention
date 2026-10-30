"""Writing the paged KV cache.

The cache layout is this library's own (head-major, K transposed inside the page),
so nothing but the library can write it: ``reshape_and_cache`` is the public way,
with vLLM's signature. ``flash_attn_with_kvcache(k=..., v=...)`` appends through
the same kernel.
"""

import torch

from . import _kernel


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter K/V of ``num_tokens`` tokens into the paged cache.

    ``key``/``value`` are ``[num_tokens, nheads_k, headdim]`` (any token stride,
    heads and dims contiguous: chunks of a fused projection work as is). The
    caches are ``[num_blocks, nheads_k, page_block_size, headdim]``, contiguous.
    Token ``i`` goes to slot ``slot_mapping[i]`` (int64), i.e. page
    ``slot // page_block_size``, position ``slot % page_block_size``; a negative
    slot skips the token (padding), as in vLLM.
    """
    if key.dim() != 3 or value.shape != key.shape:
        raise ValueError("cpu_attention: key/value must both be [num_tokens, nheads_k, headdim]")
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("cpu_attention: k_cache/v_cache must be "
                         "[num_blocks, nheads_k, page_block_size, headdim] and match")
    if key.shape[1:] != (k_cache.shape[1], k_cache.shape[3]):
        raise ValueError("cpu_attention: key/value heads or head_dim do not match the cache")
    if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("cpu_attention: the paged caches must be contiguous")
    if not (key.dtype == value.dtype == k_cache.dtype == v_cache.dtype):
        raise ValueError("cpu_attention: key/value and the caches must share a dtype")
    if not _kernel.supported(key.dtype, k_cache.shape[2], key.shape[2]):
        raise ValueError(f"cpu_attention cannot run dtype={key.dtype}, "
                         f"page={k_cache.shape[2]}, head_dim={key.shape[2]}")
    if slot_mapping.dtype != torch.int64 or tuple(slot_mapping.shape) != (key.shape[0],):
        raise ValueError("cpu_attention: slot_mapping must be int64 [num_tokens]")
    if key.stride(-1) != 1 or key.stride(-2) != key.shape[2]:
        key = key.contiguous()
    if value.stride(-1) != 1 or value.stride(-2) != value.shape[2]:
        value = value.contiguous()
    if key.shape[0] and int(slot_mapping.max()) >= k_cache.shape[0] * k_cache.shape[2]:
        raise ValueError("cpu_attention: slot_mapping points past the end of the cache")
    if key.shape[0]:
        _kernel._kv_store(key, value, k_cache, v_cache, slot_mapping.contiguous())
