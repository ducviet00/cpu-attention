"""flash_attn_with_kvcache (paged decode) against the float64 reference."""

import random
import threading

import numpy as np
import pytest
import torch

from cpu_attention import flash_attn_with_kvcache
from reference import attention, check, f64

DTYPES = [torch.float32, torch.bfloat16]
HEADS = [(8, 8), (8, 2), (8, 1)]


def stage(lens, h, hk, d, page, dtype, seed):
    """Caches holding ``lens[i]`` random tokens per sequence, staged through the API in
    two appends each. Everything a kernel must not read is NaN: unused slots and pages,
    and the page-table padding (which points at a poisoned page)."""
    g = torch.Generator().manual_seed(seed)
    npages = [-(-n // page) for n in lens]
    total = sum(npages) + 1
    order = torch.randperm(total, generator=g).to(torch.int32)
    bt = torch.full((len(lens), max(npages) + 1), int(order[-1]), dtype=torch.int32)
    kc = torch.full((total, hk, page, d), float("nan"), dtype=dtype)
    vc = kc.clone()
    q = torch.randn(len(lens), 1, h, d, generator=g).to(dtype)
    ks, vs, start = [], [], 0
    for i, n in enumerate(lens):
        bt[i, :npages[i]] = order[start:start + npages[i]]
        start += npages[i]
        k, v = (torch.randn(1, n, hk, d, generator=g).to(dtype) for _ in range(2))
        for a, z in ((0, n // 2), (n // 2, n)):
            if z > a:
                flash_attn_with_kvcache(q[i:i + 1], kc, vc, k=k[:, a:z], v=v[:, a:z],
                                        cache_seqlens=a, page_table=bt[i:i + 1])
        ks.append(k)
        vs.append(v)
    return q, kc, vc, bt, ks, vs


def reference(q, ks, vs, scale):
    return np.concatenate([attention(f64(q[i:i + 1]), f64(k), f64(v), scale)
                           for i, (k, v) in enumerate(zip(ks, vs))])


def decode(lens, h, hk, d, page, dtype, seed, scale=None):
    q, kc, vc, bt, ks, vs = stage(lens, h, hk, d, page, dtype, seed)
    seqlens = torch.tensor(lens, dtype=torch.int32)
    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=seqlens, page_table=bt,
                                  softmax_scale=scale)
    return check(out, reference(q, ks, vs, scale or d ** -0.5), dtype, paged=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("page", [16, 32, 48, 64])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("h,hk", HEADS)
def test_decode(dtype, page, d, h, hk):
    err, tol = decode([0, 1, page - 1, page, page + 1, 255, 257], h, hk, d, page, dtype,
                      seed=page + d)
    assert err <= tol


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("h,hk", HEADS)
def test_append_one_token(dtype, h, hk):
    """k/v of one new token are written at cache_seqlens and attended in the same call."""
    lens, d, page = [5, 31, 64, 300], 64, 16
    q, kc, vc, bt, ks, vs = stage(lens, h, hk, d, page, dtype, seed=3)
    kn, vn = (torch.randn(len(lens), 1, hk, d).to(dtype) for _ in range(2))
    seqlens = torch.tensor(lens, dtype=torch.int32)
    out = flash_attn_with_kvcache(q, kc, vc, k=kn, v=vn, cache_seqlens=seqlens,
                                  page_table=bt)
    ks = [torch.cat([k, kn[i:i + 1]], 1) for i, k in enumerate(ks)]
    vs = [torch.cat([v, vn[i:i + 1]], 1) for i, v in enumerate(vs)]
    ref = reference(q, ks, vs, d ** -0.5)
    err, tol = check(out, ref, dtype, paged=True)
    assert err <= tol
    again = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=seqlens + 1, page_table=bt)
    err, tol = check(again, ref, dtype, paged=True)
    assert err <= tol


@pytest.mark.parametrize("threads", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("h,hk", HEADS)
def test_threads_uneven_lengths(threads, h, hk):
    """Work is split across threads by context length: lopsided batches, odd counts."""
    saved = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for lens in ([0, 2047, 1, 17, 64], [2047, 16], [5, 3, 1024]):
            err, tol = decode(lens, h, hk, 64, 16, torch.float32, seed=threads + hk)
            assert err <= tol
    finally:
        torch.set_num_threads(saved)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_projection_strides(dtype):
    """q and the new k/v as chunks of one fused [b, 1, (h + 2 hk) d] projection."""
    lens, h, hk, d = [40, 200], 8, 2, 64
    q, kc, vc, bt, ks, vs = stage(lens, h, hk, d, 32, dtype, seed=4)
    qkv = torch.randn(len(lens), 1, (h + 2 * hk) * d).to(dtype)
    qn = qkv[..., :h * d].view(len(lens), 1, h, d)
    kn = qkv[..., h * d:(h + hk) * d].view(len(lens), 1, hk, d)
    vn = qkv[..., (h + hk) * d:].view(len(lens), 1, hk, d)
    out = flash_attn_with_kvcache(qn, kc, vc, k=kn, v=vn, page_table=bt,
                                  cache_seqlens=torch.tensor(lens, dtype=torch.int32))
    ks = [torch.cat([k, kn[i:i + 1]], 1) for i, k in enumerate(ks)]
    vs = [torch.cat([v, vn[i:i + 1]], 1) for i, v in enumerate(vs)]
    err, tol = check(out, reference(qn, ks, vs, d ** -0.5), dtype, paged=True)
    assert err <= tol


@pytest.mark.parametrize("seed", range(30))
def test_fuzz(seed):
    r = random.Random(seed)
    h, hk = r.choice(HEADS)
    lens = [r.randint(0, 1100) for _ in range(r.randint(1, 6))]
    err, tol = decode(lens, h, hk, r.choice([32, 64, 128]), r.choice([16, 32, 48, 64]),
                      r.choice(DTYPES), seed, scale=r.choice([None, 0.05, 0.2]))
    assert err <= tol


def test_concurrent_callers():
    """Python threads calling at once each get their own correct result."""
    cases = [stage([100 + 50 * i, 33], 8, 2, 64, 16, torch.float32, seed=i) for i in range(4)]
    refs = [reference(q, ks, vs, 0.125) for q, _, _, _, ks, vs in cases]
    errors = []

    def worker(i):
        q, kc, vc, bt, _, _ = cases[i]
        seqlens = torch.tensor([100 + 50 * i, 33], dtype=torch.int32)
        for _ in range(20):
            out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=seqlens, page_table=bt)
            errors.append(check(out, refs[i], torch.float32, paged=True))

    pool = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert len(errors) == 80 and all(err <= tol for err, tol in errors)
