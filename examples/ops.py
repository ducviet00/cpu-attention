"""Example: an inference engine's per-layer paged-attention calls.

An engine that keeps its own slot mapping and a scratch output per layer (as
fast-encdec's runner does) can call the kernel's decode entry directly: q/k/v are
``[seqs, heads, head_dim]`` (heads and dims contiguous; a token stride is fine).
The entry validates its arguments and returns None for anything it does not take.
``cpu_attention._kernel`` is internal and may change between releases.
"""

from cpu_attention import _kernel as K


def _checked(result):
    if result is None:
        raise ValueError("cpu_attention: the kernel declined these arguments")
    return result


def self_attention(query, key, value, k_cache, v_cache, slot_mapping, block_table,
                   context_lens, scale, out=None):
    """Store each sequence's new K/V at ``slot_mapping`` and attend over its cache;
    ``context_lens`` is the length after the store. ``out``, when given, is a
    caller-owned ``[seqs, heads, head_dim]`` output (it saves the allocation)."""
    return _checked(K.module().decode(K.runtime(), query, k_cache, v_cache, key, value,
                                      slot_mapping, context_lens, block_table, scale, out))


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale, out=None):
    """Attend over a pre-staged (encoder) K/V cache; no store."""
    return _checked(K.module().decode(K.runtime(), query, k_cache, v_cache, None, None, None,
                                      cache_seqlens, block_table, scale, out))


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping):
    """``cpu_attention.reshape_and_cache`` without the error explanations."""
    _checked(K.module().store(K.runtime(), key, value, k_cache, v_cache, slot_mapping))
