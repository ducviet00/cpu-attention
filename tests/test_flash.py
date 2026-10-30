"""flash_attn_func against the float64 reference."""

import random

import pytest
import torch

from cpu_attention import flash_attn_func
from reference import attention, check, f64

DTYPES = [torch.float32, torch.bfloat16]
# (seq_q, seq_k): tile edges (16/64/128 rows), single rows, unequal lengths both ways.
LENGTHS = [(1, 1), (17, 17), (64, 64), (129, 129), (1, 300), (100, 37), (37, 100), (257, 513)]
HEADS = [(8, 8), (8, 2), (8, 1)]


def run(b, sq, sk, h, hk, d, dtype, causal, seed, scale=None):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(b, n, m, d, generator=g).to(dtype)
               for n, m in ((sq, h), (sk, hk), (sk, hk)))
    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    ref = attention(f64(q), f64(k), f64(v), scale or d ** -0.5, causal)
    return check(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("sq,sk", LENGTHS)
@pytest.mark.parametrize("h,hk", HEADS)
@pytest.mark.parametrize("causal", [True, False])
def test_shapes(dtype, d, sq, sk, h, hk, causal):
    err, tol = run(1, sq, sk, h, hk, d, dtype, causal, seed=sq * 1000 + sk)
    assert err <= tol


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_and_scale(dtype):
    err, tol = run(3, 70, 150, 8, 2, 64, dtype, True, seed=1, scale=0.07)
    assert err <= tol


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_qkv_strides(dtype):
    """q/k/v as chunks of one [b, s, 3*h*d] projection (no copy is needed)."""
    b, s, h, d = 2, 200, 4, 64
    qkv = torch.randn(b, s, 3 * h * d).to(dtype)
    q, k, v = (qkv[..., i * h * d:(i + 1) * h * d].view(b, s, h, d) for i in range(3))
    ref = attention(f64(q), f64(k), f64(v), d ** -0.5, True)
    err, tol = check(flash_attn_func(q, k, v, causal=True), ref, dtype)
    assert err <= tol


@pytest.mark.parametrize("seed", range(40))
def test_fuzz(seed):
    r = random.Random(seed)
    h, hk = r.choice(HEADS)
    err, tol = run(r.randint(1, 3), r.randint(1, 900), r.randint(1, 900), h, hk,
                   r.choice([32, 64, 128]), r.choice(DTYPES), r.random() < 0.5, seed)
    assert err <= tol


def test_repeated_calls_follow_changes():
    """The cached call plan must follow every change of scale, causal and shape."""
    for scale, causal, s in ((0.05, True, 64), (0.125, False, 64), (0.05, False, 65)):
        err, tol = run(1, s, s, 4, 4, 64, torch.float32, causal, seed=s, scale=scale)
        assert err <= tol
