"""Example: an inference engine's per-layer paged-attention calls.

An engine that already holds validated tensors and its own slot mapping (as
fast-encdec's runner does, once per decode step) can skip the public functions'
checks and launch the kernel directly. q/k/v are ``[tokens, heads, head_dim]``
(heads and dims contiguous; a token stride is fine). This uses
``cpu_attention._kernel``, which is internal and may change between releases.
"""

import torch

from cpu_attention import _kernel as K

_PLANS = {}


def _plan(query, key, k_cache, block_table, store, attend):
    geom = (None if query is None else (query.shape, query.stride()),
            None if key is None else (key.shape, key.stride()), k_cache.shape,
            k_cache.dtype, None if block_table is None else block_table.shape,
            store, attend, torch.get_num_threads())
    plan = _PLANS.get(geom)
    if plan is None:
        npages = block_table.shape[1] if block_table is not None else 0
        plan = _PLANS[geom] = K.paged_plan(query, key, k_cache, npages, store, attend)
    return plan


def self_attention(query, key, value, k_cache, v_cache, slot_mapping, block_table,
                   context_lens, scale, out=None):
    """Store each sequence's new K/V at ``slot_mapping`` and attend over its cache;
    ``context_lens`` is the length after the store. ``out``, when given, is a
    caller-owned ``[seqs, heads, head_dim]`` output (it saves the allocation)."""
    out = torch.empty_like(query) if out is None else out
    K.run_paged(_plan(query, key, k_cache, block_table, True, True), query, key, value,
                k_cache, v_cache, out, slot_mapping, block_table, context_lens, scale)
    return out


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale, out=None):
    """Attend over a pre-staged (encoder) K/V cache; no store."""
    out = torch.empty_like(query) if out is None else out
    K.run_paged(_plan(query, None, k_cache, block_table, False, True), query, None, None,
                k_cache, v_cache, out, None, block_table, cache_seqlens, scale)
    return out


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping):
    """``cpu_attention.reshape_and_cache`` without its checks."""
    K.run_paged(_plan(None, key, k_cache, None, True, False), None, key, value,
                k_cache, v_cache, None, slot_mapping, None, None, 1.0)
