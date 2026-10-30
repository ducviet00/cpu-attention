"""flash-attn's interface over the Mojo kernel.

Same names and arguments as flash-attn; what the kernel cannot do raises
``NotImplementedError`` instead of being silently ignored. A call is validated and
planned once per geometry (shapes, strides, dtypes, thread count); later calls
with that geometry look the plan up and launch.
"""

import torch

from . import _kernel as K
from .cache import reshape_and_cache

_MHA_ONLY = "cpu_attention supports MHA only (nheads == nheads_k); GQA/MQA are not supported yet"
_PLANS = {}  # geometry key -> plan; plans are read-only, so threads share them


def _remember(key, plan):
    if len(_PLANS) > 1024:
        _PLANS.clear()
    _PLANS[key] = plan
    return plan


def _layout_ok(x):
    """Heads and dims contiguous (any token stride): the kernel reads it in place."""
    return x.stride(-1) == 1 and x.stride(-2) == x.shape[-1]


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
):
    """Paged decode: one query token per sequence against its cached K/V.

    flash-attn's arguments; ``qv``, ``rotary_*``, ``cache_leftpad``,
    ``cu_seqlens_*``, ``max_seqlen_q``, ``*_descale``, ``window_size``,
    ``softcap``, ``attention_chunk`` and ``return_softmax_lse`` raise
    ``NotImplementedError``; the tuning hints are ignored. Differences:

    * ``q`` is ``[batch, 1, nheads, headdim]``, MHA (``nheads_k == nheads``).
    * The cache is head-major ``[num_blocks, nheads, page_block_size, headdim]``
      (K is stored transposed inside the page), contiguous, and written only by
      this library: ``reshape_and_cache``, or ``k``/``v`` here -- any number of
      new tokens, appended at ``cache_seqlens`` (the length before the write).
    * ``causal`` has no effect: the query sits at the last cached position.
    """
    if (qv is not None or rotary_cos is not None or rotary_sin is not None
            or cache_leftpad is not None or cu_seqlens_q is not None
            or cu_seqlens_k_new is not None or max_seqlen_q is not None
            or rotary_seqlens is not None or q_descale is not None
            or k_descale is not None or v_descale is not None
            or window_size != (-1, -1) or softcap or attention_chunk or return_softmax_lse):
        raise NotImplementedError(
            "cpu_attention: qv, rotary, cache_leftpad, varlen, descale, sliding "
            "windows, softcap, attention_chunk and return_softmax_lse are not supported")
    if page_table is None:
        raise ValueError("cpu_attention: pass `page_table`")
    if cache_batch_idx is not None:
        page_table = page_table.index_select(0, cache_batch_idx.long())
    if not isinstance(cache_seqlens, torch.Tensor):
        if cache_seqlens is None:
            raise ValueError("cpu_attention: `cache_seqlens` is required")
        cache_seqlens = torch.full((q.shape[0],), cache_seqlens, dtype=torch.int32)
    if k is not None and k.shape[1] != 1:
        # Several new tokens (a prompt): store them at cache_seqlens + j, then
        # decode over the grown cache.
        n = k.shape[1]
        page = k_cache.shape[2]
        pos = cache_seqlens.long()[:, None] + torch.arange(n)[None]
        slots = page_table.long().gather(1, pos // page) * page + pos % page
        reshape_and_cache(k.reshape(-1, *k.shape[2:]), v.reshape(-1, *v.shape[2:]),
                          k_cache, v_cache, slots.flatten())
        cache_seqlens, k, v = cache_seqlens + n, None, None
    new = None if k is None else (k.shape, k.stride(), k.dtype, v.shape, v.stride(), v.dtype)
    key = (q.shape, q.stride(), q.dtype, k_cache.shape, k_cache.stride(), k_cache.dtype,
           v_cache.shape, v_cache.stride(), v_cache.dtype, page_table.shape,
           page_table.stride(), page_table.dtype, cache_seqlens.shape,
           cache_seqlens.stride(), cache_seqlens.dtype, new, torch.get_num_threads())
    plan = _PLANS.get(key) or _remember(key, _kvcache_plan(q, k_cache, v_cache, k, v,
                                                           page_table, cache_seqlens))
    buf, addr, copy, scale = plan
    if copy:
        q = q.contiguous()
        k, v = (None, None) if k is None else (k.contiguous(), v.contiguous())
    out = torch.empty_like(q)
    K.run_paged(plan, q, k, v, k_cache, v_cache, out, None, page_table, cache_seqlens,
                scale if softmax_scale is None else softmax_scale)
    return out


def _kvcache_plan(q, k_cache, v_cache, k, v, page_table, lens):
    """Validate one geometry and plan it: (buffer, address, copy, default scale)."""
    if q.dim() != 4 or q.shape[1] != 1:
        if q.dim() == 4:
            raise NotImplementedError("cpu_attention decodes one query token per sequence")
        raise ValueError("cpu_attention: q must be [batch, 1, nheads, headdim]")
    batch, _, heads, hd = q.shape
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("cpu_attention: k_cache/v_cache must be "
                         "[num_blocks, nheads, page_block_size, headdim] and match")
    if k_cache.shape[3] != hd or heads % k_cache.shape[1]:
        raise ValueError("cpu_attention: q and the caches disagree on heads or head_dim")
    if k_cache.shape[1] != heads:
        raise NotImplementedError(_MHA_ONLY)
    if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("cpu_attention: the paged caches must be contiguous")
    if q.dtype not in K.DTYPES or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("cpu_attention: q and the caches must share a dtype (fp32 or bf16)")
    if hd not in K.HEAD_DIMS or k_cache.shape[2] % 16:
        raise ValueError(f"cpu_attention: head_dim must be 32/64/128 (got {hd}) and the "
                         f"page a multiple of 16 (got {k_cache.shape[2]})")
    if (page_table.dtype != torch.int32 or page_table.dim() != 2
            or page_table.shape[0] != batch or not page_table.is_contiguous()):
        raise ValueError("cpu_attention: page_table must be a contiguous int32 [batch, max_blocks]")
    if lens.dtype != torch.int32 or lens.shape != (batch,) or not lens.is_contiguous():
        raise ValueError("cpu_attention: cache_seqlens must be an int or an int32 [batch] tensor")
    if (k is None) != (v is None) or k is not None and (
            k.shape != (batch, 1, heads, hd) or v.shape != k.shape
            or k.dtype != q.dtype or v.dtype != q.dtype):
        raise ValueError("cpu_attention: k/v must both be [batch, seqlen_new, nheads, headdim]")
    copy = not _layout_ok(q) or k is not None and not (_layout_ok(k) and v.stride() == k.stride())
    if copy:
        q = q.contiguous()
        k = None if k is None else k.contiguous()
    buf, addr = K.paged_plan(q, k, k_cache, page_table.shape[1], k is not None, True)
    return buf, addr, copy, hd ** -0.5


def flash_attn_func(q, k, v, softmax_scale=None, causal=False):
    """Dense attention over whole sequences, flash-attn's ``flash_attn_func``.

    q/k/v are ``[batch, seqlen, nheads, headdim]`` (MHA; GQA/MQA raise
    ``NotImplementedError``). ``causal`` is bottom-right aligned (query ``i`` sees
    keys ``j <= i + seqlen_k - seqlen_q``); a row that sees no key is zero.
    ``softmax_scale`` defaults to ``1 / sqrt(headdim)``. Inference only: flash-attn's
    dropout, window, softcap, alibi and ``return_attn_probs`` are not offered.
    """
    key = (q.shape, q.stride(), q.dtype, k.shape, k.stride(), k.dtype, v.shape,
           v.stride(), v.dtype, softmax_scale, causal, torch.get_num_threads())
    plan = _PLANS.get(key) or _remember(key, _flash_plan(q, k, v, softmax_scale, causal))
    buf, addr, ws, copy, scale = plan
    if copy:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    K.module().flash_forward(addr, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                             out.data_ptr(), K.workspace(ws), scale)
    return out


def _flash_plan(q, k, v, softmax_scale, causal):
    """Validate one geometry and plan it: (buffer, address, workspace bytes, copy, scale)."""
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape:
        raise ValueError("cpu_attention: q/k/v must be [batch, seqlen, nheads, headdim]")
    batch, seq_q, heads, hd = q.shape
    if (k.shape[0] != batch or k.shape[3] != hd or heads % k.shape[2]
            or seq_q < 1 or k.shape[1] < 1):
        raise ValueError("cpu_attention: k/v must be [batch, seqlen_k, nheads, headdim] matching q")
    if k.shape[2] != heads:
        raise NotImplementedError(_MHA_ONLY)
    if q.dtype not in K.DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("cpu_attention: q/k/v must share a dtype (fp32 or bf16)")
    if hd not in K.HEAD_DIMS:
        raise ValueError(f"cpu_attention: head_dim must be 32, 64 or 128 (got {hd})")
    # The kernel takes every stride explicitly except the head dim's.
    copy = q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1
    if copy:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    scale = hd ** -0.5 if softmax_scale is None else float(softmax_scale)
    return (*K.flash_plan(q, k, v, scale, bool(causal)), copy, scale)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, softmax_scale=None, causal=False):
    """Not ported: use padded q/k/v with ``flash_attn_func``, or the paged cache."""
    raise NotImplementedError("cpu_attention: variable-length attention is not ported")


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    """Not ported (pass q/k/v views of the packed tensor to ``flash_attn_func``)."""
    raise NotImplementedError("cpu_attention: packed attention is not ported")


flash_attn_kvpacked_func = flash_attn_varlen_qkvpacked_func = \
    flash_attn_varlen_kvpacked_func = flash_attn_qkvpacked_func
