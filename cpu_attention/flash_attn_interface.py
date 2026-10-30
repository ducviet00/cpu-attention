"""flash-attn's interface over the Mojo paged kernel.

``flash_attn_with_kvcache`` and ``flash_attn_func`` take flash-attn's arguments;
what the kernel cannot do raises ``NotImplementedError`` instead of being
silently ignored.  The rest of the module mirrors upstream's file of the same
name so ``from cpu_attention.flash_attn_interface import flash_attn_func`` works.
"""

import torch

from . import _kernel
from ._kernel import _DTYPE_CODES, _HEAD_DIMS, _dense, _paged, supported

_SCALES = {}
"""``head_dim ** -0.5`` per head dim: a float pow is ~20 ns and the value cannot
change, so the default scale is looked up rather than recomputed."""


def _lens_tensor(cache_seqlens, batch: int, device):
    """Return ``cache_seqlens`` as the int32 ``[batch]`` lengths the kernel reads."""
    if cache_seqlens is None:
        raise ValueError("cpu_attention: `cache_seqlens` is required")
    if isinstance(cache_seqlens, int):
        return torch.full((batch,), cache_seqlens, dtype=torch.int32, device=device)
    if cache_seqlens.dtype != torch.int32 or tuple(cache_seqlens.shape) != (batch,):
        raise ValueError(
            "cpu_attention: `cache_seqlens` must be an int or an int32 [batch] tensor"
        )
    return cache_seqlens


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    block_table=None,
    slot_mapping=None,
    check_inputs=True,
):
    """``flash_attn_with_kvcache`` over this backend's paged cache.

    Same arguments and defaults as flash-attn's function; the unsupported ones
    raise ``NotImplementedError`` rather than being silently ignored (``qv``,
    ``rotary_*``, ``cache_leftpad``, ``cu_seqlens_*``, ``max_seqlen_q``,
    ``*_descale``, ``window_size``, ``softcap``, ``attention_chunk``,
    ``return_softmax_lse``).  The tuning hints ``num_splits``, ``pack_gqa``,
    ``sm_margin``, ``scheduler_metadata`` and ``rotary_interleaved`` are
    accepted and ignored.  ``block_table`` is this repo's name for flash's
    ``page_table``.

    Deliberate differences:

    * ``q`` must be ``[batch, 1, nheads, headdim]``: the kernel decodes one
      query token per sequence, so ``seqlen`` may only be 1.
    * The cache is head-major ``[num_blocks, nheads_k, page_block_size,
      headdim]`` (``attention.self_cache_shape``), not flash's page-major
      layout, because K is stored transposed inside the page.  Must be
      contiguous.
    * ``causal`` is accepted and has no effect: the one query sits at the last
      cached position, where the bottom-right-aligned mask keeps every key.
    * With ``k``/``v``, they are written at ``cache_seqlens`` (the length
      *before* the write, flash's convention).  Passing ``slot_mapping``
      instead writes them at those slots and reads ``cache_seqlens`` as the
      length *after* the write -- the engine's Context convention, and it skips
      computing the slots per call.
    * ``check_inputs=False`` skips the argument validation for callers whose
      tensors are already known-good (the engine's per-layer path): the checks
      cost a few microseconds per call, which at 24 calls per decode step is
      measurable end to end.
    """
    if (
        qv is None
        and rotary_cos is None
        and rotary_sin is None
        and cache_leftpad is None
        and cu_seqlens_q is None
        and cu_seqlens_k_new is None
        and max_seqlen_q is None
        and rotary_seqlens is None
        and q_descale is None
        and k_descale is None
        and v_descale is None
        and cache_batch_idx is None
        and window_size == (-1, -1)
        and not softcap
        and not attention_chunk
        and not return_softmax_lse
        and (page_table is None) != (block_table is None)
    ):
        table = getattr(_kernel._KV_TABLE, "t", None)
        if table is not None and _kernel._KV_TABLE.gen == _kernel._KV_GEN[0]:
            pt = page_table if page_table is not None else block_table
            res = _kernel._KV_TABLE.fn(
                _kernel._KV_TABLE.addr,
                q,
                k,
                v,
                k_cache,
                v_cache,
                slot_mapping,
                pt,
                cache_seqlens,
                softmax_scale,
                None,
            )
            if res is not None:
                return res
    if check_inputs:
        if (
            qv is not None
            or rotary_cos is not None
            or rotary_sin is not None
            or cache_leftpad is not None
            or cu_seqlens_q is not None
            or cu_seqlens_k_new is not None
            or max_seqlen_q is not None
            or rotary_seqlens is not None
            or q_descale is not None
            or k_descale is not None
            or v_descale is not None
        ):
            raise NotImplementedError(
                "cpu_attention: qv/rotary_*/cache_leftpad/cu_seqlens_*/"
                "max_seqlen_q/*_descale are not supported"
            )
        if window_size != (-1, -1):
            raise NotImplementedError(
                "cpu_attention: sliding windows are not supported"
            )
        if softcap:
            raise NotImplementedError("cpu_attention: softcapping is not supported")
        if attention_chunk:
            raise NotImplementedError(
                "cpu_attention: `attention_chunk` is not supported"
            )
        if return_softmax_lse:
            raise NotImplementedError(
                "cpu_attention: `return_softmax_lse` is not supported"
            )
        if q.dim() != 4:
            raise ValueError(
                "cpu_attention: q must be [batch, seqlen, nheads, headdim]"
            )
        batch, q_len, num_heads, head_dim = q.shape
        if q_len != 1:
            raise NotImplementedError(
                "cpu_attention decodes one query token per sequence; "
                f"got seqlen={q_len}"
            )
        if page_table is not None and block_table is not None:
            raise ValueError("cpu_attention: pass page_table or block_table, not both")
        page_table = page_table if page_table is not None else block_table
        if page_table is None:
            raise ValueError("cpu_attention: pass `page_table` (or `block_table`)")
        if cache_batch_idx is not None:
            page_table = page_table.index_select(0, cache_batch_idx.long())
        if page_table.dtype != torch.int32 or page_table.dim() != 2:
            raise ValueError(
                "cpu_attention: page_table must be int32 [batch, max_blocks]"
            )
        if page_table.shape[0] != batch:
            raise ValueError("cpu_attention: page_table has the wrong batch size")
        page_table = page_table.contiguous()
        if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
            raise ValueError(
                "cpu_attention: k_cache/v_cache must be "
                "[num_blocks, nheads_k, page_block_size, headdim] and match"
            )
        num_kv_heads, block_size, cache_dim = k_cache.shape[1:]
        if cache_dim != head_dim:
            raise ValueError(
                "cpu_attention: head_dim mismatch between q and the caches"
            )
        if num_heads % num_kv_heads:
            raise ValueError("cpu_attention: nheads must be divisible by nheads_k")
        if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
            raise ValueError(
                "cpu_attention: the paged caches must be contiguous "
                "(head-major layout, see `attention.self_cache_shape`)"
            )
        if q.dtype != k_cache.dtype or v_cache.dtype != q.dtype:
            raise ValueError("cpu_attention: q and the caches must share a dtype")
        if not supported(q.dtype, block_size, head_dim):
            raise ValueError(
                f"cpu_attention cannot run dtype={q.dtype}, page={block_size}, "
                f"head_dim={head_dim}"
            )
        if k is not None and (
            k.dim() != 4
            or k.shape[:2] != (batch, 1)
            or k.shape[2] != num_kv_heads
            or v.shape != k.shape
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError(
                "cpu_attention: k/v must be [batch, 1, nheads_k, headdim] and "
                "match the cache"
            )
        if slot_mapping is not None and (
            k is None or slot_mapping.dtype != torch.int64
        ):
            raise ValueError("cpu_attention: slot_mapping needs k/v and must be int64")
    else:
        batch, _, num_heads, head_dim = q.shape
        page_table = page_table if page_table is not None else block_table
        num_kv_heads, block_size = k_cache.shape[1], k_cache.shape[2]
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    query = q.squeeze(1)
    if k is not None:
        if slot_mapping is None:
            before = _lens_tensor(cache_seqlens, batch, q.device)
            slots = None
            lens = before
        else:
            slots = slot_mapping
            lens = _lens_tensor(cache_seqlens, batch, q.device)
        out = _paged(
            query,
            k.squeeze(1),
            v.squeeze(1),
            k_cache,
            v_cache,
            slots,
            page_table,
            lens,
            scale,
            do_store=True,
        )
    else:
        out = _paged(
            query,
            None,
            None,
            k_cache,
            v_cache,
            None,
            page_table,
            _lens_tensor(cache_seqlens, batch, q.device),
            scale,
            do_store=False,
        )
    return out.unsqueeze(1)


def flash_attn_func(q, k, v, softmax_scale=None, causal=False):
    """Dense attention, flash-attn's ``flash_attn_func``.

    ``q`` is ``[batch, seqlen_q, nheads, headdim]`` and ``k``/``v`` are
    ``[batch, seqlen_k, nheads_k, headdim]``; MQA/GQA is ``nheads % nheads_k ==
    0`` with query head ``h`` reading KV head ``h // (nheads // nheads_k)``.
    ``causal`` uses flash-attn's bottom-right alignment (query ``i`` sees keys
    ``j <= i + seqlen_k - seqlen_q``), so rows with no visible key come back
    zero.  ``softmax_scale`` defaults to ``1 / sqrt(headdim)``.

    Dropped from flash-attn's signature rather than accepted-and-rejected:
    ``dropout_p``, ``window_size``, ``softcap``, ``alibi_slopes``,
    ``deterministic`` and ``return_attn_probs`` (inference-only forward pass, no
    local attention, no bias terms).
    """
    # Mojo fast path: a layout a validated call already registered runs with
    # the tensors read in the extension (no Python checks or plan lookup).
    table = _kernel._FLASH.__dict__.get("t")
    if (table is not None and _kernel._FLASH.gen == _kernel._flash_gen() and
            type(q) is torch.Tensor):
        res = _kernel._FLASH.fn(_kernel._FLASH.addr, q, k, v,
                                softmax_scale, causal)
        if res is not None:
            return res
    q0, k0, v0 = q, k, v
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "cpu_attention: q/k/v must be [batch, seqlen, nheads, headdim]"
        )
    # One `.shape`/`.stride()`/`.dtype` read per tensor for the whole call: each
    # is a ~0.1 us Python call, the checks below need them anyway, and `_dense`
    # takes them as its cache key (`geom`) instead of re-reading the tensors.
    qs, ks, vs = q.shape, k.shape, v.shape
    batch, q_len, num_heads, head_dim = qs
    kv_len, num_kv_heads = ks[1], ks[2]
    if (
        kv_len < 1
        or q_len < 1
        or ks[0] != batch
        or ks[3] != head_dim
        or vs != ks
        or num_heads % num_kv_heads
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError(
            "cpu_attention: k/v must be [batch, seqlen_k, nheads_k, headdim] and "
            "match q (nheads divisible by nheads_k)"
        )
    if head_dim not in _HEAD_DIMS:
        raise ValueError(
            f"cpu_attention cannot run head_dim={head_dim} (32, 64 or 128)"
        )
    if q.dtype not in _DTYPE_CODES:
        raise ValueError(f"cpu_attention cannot run dtype={q.dtype}")
    if softmax_scale is None:
        scale = _SCALES.get(head_dim)
        if scale is None:
            scale = _SCALES[head_dim] = head_dim**-0.5
    else:
        scale = softmax_scale
    # The kernel reads the head dim contiguously and takes every other stride
    # explicitly, so a [batch, seq, nheads*headdim] slice of a projection works
    # as is; only a strided head dim needs a copy.
    stq, stk, stv = q.stride(), k.stride(), v.stride()
    if stq[3] != 1:
        q = q.contiguous()
        stq = q.stride()
    if stk[3] != 1:
        k = k.contiguous()
        stk = k.stride()
    if stv[3] != 1:
        v = v.contiguous()
        stv = v.stride()
    out = _dense(q, k, v, scale, causal, (qs, stq, stk, stv, q.dtype))
    if q is q0 and k is k0 and v is v0:
        _kernel._flash_register(q, k, v, softmax_scale, causal)
    return out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
):
    """Not implemented: variable-length packing is not ported (use padded q/k/v)."""
    raise NotImplementedError(
        "cpu_attention: variable-length attention is not ported; "
        "use flash_attn_func for dense padded q/k/v, or flash_attn_with_kvcache "
        "for the paged decode path"
    )


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    """Not implemented (dense packed attention)."""
    raise NotImplementedError("cpu_attention: packed dense attention is not ported")


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    """Not implemented (dense packed attention)."""
    raise NotImplementedError("cpu_attention: packed dense attention is not ported")


def flash_attn_varlen_qkvpacked_func(qkv, *args, **kwargs):
    """Not implemented (dense packed attention)."""
    raise NotImplementedError("cpu_attention: packed dense attention is not ported")


def flash_attn_varlen_kvpacked_func(q, kv, *args, **kwargs):
    """Not implemented (dense packed attention)."""
    raise NotImplementedError("cpu_attention: packed dense attention is not ported")
