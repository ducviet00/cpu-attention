"""What the kernel cannot do is rejected, never silently ignored."""

import pytest
import torch

import cpu_attention as C


def paged():
    q = torch.randn(2, 1, 4, 64)
    kc = torch.zeros(4, 2, 16, 64)
    return q, kc, kc.clone(), torch.arange(4, dtype=torch.int32).view(2, 2)


@pytest.mark.parametrize("kwargs", [
    dict(window_size=(8, 0)), dict(softcap=30.0), dict(return_softmax_lse=True),
    dict(attention_chunk=64), dict(rotary_cos=torch.ones(1)), dict(qv=torch.ones(1)),
])
def test_kvcache_unsupported_options(kwargs):
    q, kc, vc, bt = paged()
    with pytest.raises(NotImplementedError):
        C.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=4, page_table=bt, **kwargs)


def test_kvcache_one_query_token_only():
    q, kc, vc, bt = paged()
    with pytest.raises(NotImplementedError):
        C.flash_attn_with_kvcache(q.expand(2, 3, 4, 64), kc, vc, cache_seqlens=4,
                                  page_table=bt)


@pytest.mark.parametrize("bad", ["dtype", "head_dim", "page", "heads", "no_table"])
def test_kvcache_bad_arguments(bad):
    q, kc, vc, bt = paged()
    if bad == "dtype":
        q = q.half()
    elif bad == "head_dim":
        q, kc, vc = q[..., :48], kc[..., :48].contiguous(), vc[..., :48].contiguous()
    elif bad == "page":
        kc, vc = kc[:, :, :8].contiguous(), vc[:, :, :8].contiguous()
    elif bad == "heads":
        q = q[:, :, :3]
    elif bad == "no_table":
        bt = None
    with pytest.raises(ValueError):
        C.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=4, page_table=bt)


@pytest.mark.parametrize("bad", ["dtype", "head_dim", "heads", "batch"])
def test_flash_bad_arguments(bad):
    q, k = torch.randn(2, 8, 4, 64), torch.randn(2, 8, 2, 64)
    if bad == "dtype":
        q, k = q.half(), k.half()
    elif bad == "head_dim":
        q, k = q[..., :48], k[..., :48]
    elif bad == "heads":
        q = q[:, :, :3]
    elif bad == "batch":
        k = k[:1]
    with pytest.raises(ValueError):
        C.flash_attn_func(q, k, k)


@pytest.mark.parametrize("fn", [C.flash_attn_varlen_func, C.flash_attn_qkvpacked_func,
                                C.flash_attn_kvpacked_func,
                                C.flash_attn_varlen_qkvpacked_func,
                                C.flash_attn_varlen_kvpacked_func])
def test_not_ported(fn):
    with pytest.raises(NotImplementedError):
        fn(*[torch.zeros(1)] * 7)
