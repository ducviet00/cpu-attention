"""Example: an inference engine's per-layer paged-attention calls.

An engine that already holds validated tensors and its own slot mapping (as
fast-encdec's runner does, once per decode step) can skip
``flash_attn_with_kvcache``'s argument checks and call the kernel plumbing
directly. These take three-dimensional q/k/v; they use ``cpu_attention._kernel``,
which is internal and may change between releases.
"""

from cpu_attention import _kernel
from cpu_attention._kernel import _paged


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
    """``cpu_attention.reshape_and_cache`` without its argument checks."""
    _kernel._kv_store(key, value, k_cache, v_cache, slot_mapping)
