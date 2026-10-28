"""Reference attention and timing helpers shared by the tests and the benchmark.

The references are computed from first principles in fp32 (never SDPA or a
fused kernel), so a test can disagree with every backend it checks.
"""

import importlib.util
import os
import statistics
import time

import torch


def load_so(path):
    if not path or not os.path.exists(path):
        return None
    name = os.path.basename(path).rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expand_kv(x, heads):
    """Repeat KV heads for GQA/MQA: [..., kv_heads, D] -> [..., heads, D]."""
    kv_heads = x.shape[-2]
    if kv_heads == heads:
        return x
    assert heads % kv_heads == 0, (heads, kv_heads)
    return x.repeat_interleave(heads // kv_heads, dim=-2)


def ref_attention(q, k, v, causal, block=128, dtype=torch.float32, scale=None):
    """Ground-truth attention from first principles, blockwise over queries.

    q is [B, S_q, H, D] and k/v are [B, S_k, H_kv, D] with H % H_kv == 0
    (GQA/MQA). The causal mask is bottom-right aligned: query i keeps keys
    j <= i + (S_k - S_q). A row with no key left is defined as 0, as the API
    returns. Computed in ``dtype`` (fp32, or fp64 to arbitrate) with an
    explicit mask; never calls torch SDPA or any fused kernel.
    """
    qf = q.to(dtype)
    kf = _expand_kv(k.to(dtype), q.shape[2])
    vf = _expand_kv(v.to(dtype), q.shape[2])
    s_q, s_k = qf.shape[1], kf.shape[1]
    scale = qf.shape[-1] ** -0.5 if scale is None else scale
    out = torch.zeros_like(qf)
    for i0 in range(0, s_q, block):
        i1 = min(i0 + block, s_q)
        scores = torch.einsum("bqhd,bkhd->bhqk", qf[:, i0:i1], kf) * scale
        if causal:
            qi = torch.arange(i0, i1).unsqueeze(1)
            ki = torch.arange(s_k).unsqueeze(0)
            keep = ki <= qi + (s_k - s_q)
            scores = scores.masked_fill(~keep, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        # softmax over an all -inf row is NaN; such a row attends nothing.
        p = torch.nan_to_num(p, nan=0.0)
        out[:, i0:i1] = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return out


def ref_decode(q, k, v, dtype=torch.float32, scale=None, lens=None):
    """One query token per sequence: q [B,H,D], k/v [B,H_kv,S,D] -> [B,H,D].

    ``lens`` [B] limits sequence b to its first lens[b] keys (heterogeneous
    contexts); ``scale`` defaults to D**-0.5.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    heads = q.shape[1]
    kf = k.to(dtype).transpose(1, 2)
    vf = v.to(dtype).transpose(1, 2)
    kf = _expand_kv(kf, heads).transpose(1, 2)
    vf = _expand_kv(vf, heads).transpose(1, 2)
    scores = torch.einsum("bhd,bhsd->bhs", q.to(dtype), kf) * scale
    if lens is not None:
        pos = torch.arange(kf.shape[2]).unsqueeze(0)
        keep = pos < torch.as_tensor(lens).long().unsqueeze(1)
        scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhs,bhsd->bhd", p, vf)


def bf16_excess(got, ref):
    """Largest error of `got` against the fp32 `ref` beyond the unavoidable
    bf16 output rounding: max(|got - ref| - |ref - bf16(ref)|).

    The ARM paged bf16 gate reads "<= 1e-3 against the reference". Taken as an
    absolute error to the real-valued result it cannot be met by any bf16
    output: rounding the exact output to bf16 alone exceeds 1e-3 on 77/252 of
    the Axion sweep cases (worst 6.6e-3, docs/Mojo.md). This measures the error
    the kernel adds on top of that floor; the raw |got - ref| is reported too.
    """
    floor = (ref - ref.bfloat16().float()).abs()
    return ((got.float() - ref).abs() - floor).max().item()


def stats(samples):
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "p90": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "n": len(samples),
    }


def time_arms(arms, rounds, warmup=2):
    """Round-robin, rotated-order timing in one process.

    Returns per-arm samples in microseconds. Rotation removes the
    first-arm-in-a-round bias; the caller judges on the minimum, and the
    median is kept to expose how much the box drifts.
    """
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    samples = {name: [] for name in arms}
    names = list(arms)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            fn = arms[name]
            t0 = time.perf_counter_ns()
            fn()
            samples[name].append((time.perf_counter_ns() - t0) / 1000.0)
    return samples


def burn(seconds=1.2):
    """Raise the clock before the first timed row."""
    a = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ a
    del a
