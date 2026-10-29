"""Low-level paged-attention entry points (the engine's per-layer calls).

These take three-dimensional q/k/v and precomputed slot mappings, so they skip
the argument validation ``flash_attn_with_kvcache`` does: the caller is expected
to have built the tensors (fast-encdec's runner does, once per step).
"""

from . import _kernel
from ._kernel import _DTYPE_CODES, _attention, _paged, _workers


def self_attention(
    query,
    key,
    value,
    k_cache,
    v_cache,
    slot_mapping,
    block_table,
    context_lens,
    scale,
    out=None,
):
    """Write new K/V into the paged cache and attend over the cached keys.

    ``query``/``key``/``value`` are ``[num_seqs, num_heads, head_dim]`` with the
    token stride taken from the tensors (a fused q/k/v projection hands out
    chunks of one buffer).  ``context_lens`` is the cached length *after* the
    store, and ``slot_mapping[b]`` is where sequence ``b``'s new token goes.

    ``out`` writes into a caller-owned ``[num_seqs, num_heads, head_dim]``
    tensor instead of allocating one. At batch 1 the allocation is 1.3 us of a
    6.8 us call, so an engine that keeps a scratch buffer per layer step should
    pass it.
    """
    if _kernel._BACKEND == "mojo":
        table = getattr(_kernel._KV_TABLE, "t", None)
        if table is not None and _kernel._KV_TABLE.gen == _kernel._KV_GEN[0]:
            res = _kernel._KV_TABLE.fn(
                _kernel._KV_TABLE.addr,
                query,
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                block_table,
                context_lens,
                scale,
                out,
            )
            if res is not None:
                return res
    return _paged(
        query,
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        block_table,
        context_lens,
        scale,
        do_store=True,
        out=out,
    )


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale,
                    out=None):
    """Attend over a paged, pre-staged encoder K/V cache (no store).

    ``out`` as in :func:`self_attention`.
    """
    if _kernel._BACKEND == "mojo":
        table = getattr(_kernel._KV_TABLE, "t", None)
        if table is not None and _kernel._KV_TABLE.gen == _kernel._KV_GEN[0]:
            res = _kernel._KV_TABLE.fn(
                _kernel._KV_TABLE.addr,
                query,
                None,
                None,
                k_cache,
                v_cache,
                None,
                block_table,
                cache_seqlens,
                scale,
                out,
            )
            if res is not None:
                return res
    return _paged(
        query,
        None,
        None,
        k_cache,
        v_cache,
        None,
        block_table,
        cache_seqlens,
        scale,
        do_store=False,
        out=out,
    )


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter ``[num_tokens, num_heads, head_dim]`` K/V into the paged cache."""
    if _kernel.backend() == "mojo":
        _kernel._kv_store(key, value, k_cache, v_cache, slot_mapping)
        return
    # The q/dst/lens slots of the shared entry point are unused here: the kernel
    # only stores when ``do_attend`` is False.
    attention = _attention()
    num_tokens, num_heads, head_dim = key.shape
    if key.stride()[-1] != 1 or key.stride()[-2] != head_dim:
        key, value = key.contiguous(), value.contiguous()
    attention(
        key.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        0,
        slot_mapping.data_ptr(),
        0,
        0,
        num_tokens,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        0,
        0,
        key.stride(0),
        0,
        _DTYPE_CODES[key.dtype],
        1.0,
        _workers(num_tokens * num_heads),
        True,
        False,
    )
