"""flash-attn's interface over the Mojo kernel.

Same names and arguments as flash-attn; what the kernel cannot do raises
``NotImplementedError`` instead of being silently ignored. Each call goes straight
to the kernel's native entry, which validates the tensors itself; only when it
declines (``None``) does Python look again, to raise the precise error or to copy a
layout the kernel cannot read in place.
"""

import torch

from . import _kernel as K
from .cache import reshape_and_cache

_MHA_ONLY = "cpu_attention supports MHA only (nheads == nheads_k); GQA/MQA are not supported yet"
_NATIVE = []  # (runtime address, decode, flash), bound on the first call


def _native():
    if not _NATIVE:
        mod = K.module()
        _NATIVE[:] = [K.runtime(), mod.decode, mod.flash]
    return _NATIVE


def _plain(x):
    """A tensor subclass as a plain tensor (the kernel reads only plain ones)."""
    plain = type(x) in (torch.Tensor, torch.nn.Parameter) or not isinstance(x, torch.Tensor)
    return x if plain else x.as_subclass(torch.Tensor)


def _rows(x):
    """``x`` with heads and dims contiguous (any token stride is read in place)."""
    return x if x.stride(-1) == 1 and x.stride(-2) == x.shape[-1] else x.contiguous()


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
    alibi_slopes=None,
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
    * ``block_table`` (flash-attn 2's name for ``page_table``) is accepted;
      ``alibi_slopes`` (flash-attn 2) raises ``NotImplementedError``.
    """
    if (qv is not None or rotary_cos is not None or rotary_sin is not None
            or cache_leftpad is not None or cu_seqlens_q is not None
            or cu_seqlens_k_new is not None or max_seqlen_q is not None
            or rotary_seqlens is not None or q_descale is not None
            or k_descale is not None or v_descale is not None
            or window_size != (-1, -1) or softcap or attention_chunk or return_softmax_lse
            or alibi_slopes is not None):
        raise NotImplementedError(
            "cpu_attention: qv, rotary, cache_leftpad, varlen, descale, sliding "
            "windows, softcap, attention_chunk, alibi and return_softmax_lse are not supported")
    if block_table is not None:
        if page_table is not None:
            raise ValueError("cpu_attention: pass `page_table` or `block_table`, not both")
        page_table = block_table
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
        n, page = k.shape[1], k_cache.shape[2]
        pos = cache_seqlens.long()[:, None] + torch.arange(n)[None]
        slots = page_table.long().gather(1, pos // page) * page + pos % page
        reshape_and_cache(k.reshape(-1, *k.shape[2:]), v.reshape(-1, *v.shape[2:]),
                          k_cache, v_cache, slots.flatten())
        cache_seqlens, k, v = cache_seqlens + n, None, None
    rt, decode, _ = _NATIVE or _native()
    out = decode(rt, q, k_cache, v_cache, k, v, None, cache_seqlens, page_table, softmax_scale, None)
    if out is None:
        q, k_cache, v_cache, k, v, page_table, cache_seqlens = _check_kvcache(
            q, k_cache, v_cache, k, v, page_table, cache_seqlens)
        out = decode(rt, q, k_cache, v_cache, k, v, None, cache_seqlens, page_table,
                     softmax_scale, None)
        if out is None:
            raise RuntimeError("cpu_attention: the kernel declined arguments that passed validation")
    return out


def _check_kvcache(q, k_cache, v_cache, k, v, page_table, lens):
    """Raise the precise error for a call the kernel declined, or return its
    tensors with the layouts it reads in place."""
    q, k_cache, v_cache, k, v, page_table, lens = map(_plain, (q, k_cache, v_cache, k, v, page_table, lens))
    if q.dim() != 4:
        raise ValueError("cpu_attention: q must be [batch, 1, nheads, headdim]")
    if q.shape[1] != 1:
        raise NotImplementedError("cpu_attention decodes one query token per sequence")
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
    if page_table.dtype != torch.int32 or page_table.dim() != 2 or page_table.shape[0] != batch:
        raise ValueError("cpu_attention: page_table must be int32 [batch, max_blocks]")
    if lens.dtype != torch.int32 or lens.shape != (batch,):
        raise ValueError("cpu_attention: cache_seqlens must be an int or an int32 [batch] tensor")
    if (k is None) != (v is None) or k is not None and (
            k.shape != (batch, 1, heads, hd) or v.shape != k.shape
            or k.dtype != q.dtype or v.dtype != q.dtype):
        raise ValueError("cpu_attention: k/v must both be [batch, seqlen_new, nheads, headdim]")
    if int(lens.max()) + (k is not None) > page_table.shape[1] * k_cache.shape[2]:
        raise ValueError("cpu_attention: cache_seqlens exceed the page table's capacity")
    if k is not None:
        k, v = (k, v) if _rows(k) is k and v.stride() == k.stride() else (k.contiguous(), v.contiguous())
    return _rows(q), k_cache, v_cache, k, v, page_table.contiguous(), lens.contiguous()


def flash_attn_func(q, k, v, softmax_scale=None, causal=False):
    """Dense attention over whole sequences, flash-attn's ``flash_attn_func``.

    q/k/v are ``[batch, seqlen, nheads, headdim]`` (MHA; GQA/MQA raise
    ``NotImplementedError``). ``causal`` is bottom-right aligned (query ``i`` sees
    keys ``j <= i + seqlen_k - seqlen_q``); a row that sees no key is zero.
    ``softmax_scale`` defaults to ``1 / sqrt(headdim)``. Inference only: flash-attn's
    dropout, window, softcap, alibi and ``return_attn_probs`` are not offered.
    """
    out = _flash(q, k, v, softmax_scale, causal)
    if out is None:
        q, k, v = _check_flash(q, k, v)
        out = _flash(q, k, v, softmax_scale, bool(causal))
        if out is None:
            raise RuntimeError("cpu_attention: the kernel declined arguments that passed validation")
    return out


def _flash(q, k, v, softmax_scale, causal):
    rt, _, flash = _NATIVE or _native()
    ws, nbytes = K.workspace()
    out = flash(rt, q, k, v, softmax_scale, causal, ws, nbytes)
    if type(out) is int:  # the workspace was short: grow it once
        ws, nbytes = K.workspace(out)
        out = flash(rt, q, k, v, softmax_scale, causal, ws, nbytes)
    return out


def _check_flash(q, k, v):
    """Raise the precise error for a call the kernel declined, or return q/k/v
    with the head dim contiguous."""
    q, k, v = map(_plain, (q, k, v))
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
    return (x if x.stride(3) == 1 else x.contiguous() for x in (q, k, v))


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, softmax_scale=None, causal=False):
    """Not ported: use padded q/k/v with ``flash_attn_func``, or the paged cache."""
    raise NotImplementedError("cpu_attention: variable-length attention is not ported")


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    """Not ported (pass q/k/v views of the packed tensor to ``flash_attn_func``)."""
    raise NotImplementedError("cpu_attention: packed attention is not ported")


flash_attn_kvpacked_func = flash_attn_varlen_qkvpacked_func = \
    flash_attn_varlen_kvpacked_func = flash_attn_qkvpacked_func
