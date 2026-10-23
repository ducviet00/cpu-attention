"""Shared helpers for the independent audit harness.

Deliberately does not import tests/bench_vs_refs.py: the audit has to be able
to disagree with the repo's own measurement code.
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


def ref_attention(q, k, v, causal, block=128, dtype=torch.float32):
    """Ground-truth attention from first principles, blockwise over queries.

    q/k/v are [1, S, H, D]. Computed in fp32 with an explicit mask; never calls
    torch SDPA or any fused kernel.
    """
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    s_q, s_k = qf.shape[1], kf.shape[1]
    scale = qf.shape[-1] ** -0.5
    out = torch.empty_like(qf)
    for i0 in range(0, s_q, block):
        i1 = min(i0 + block, s_q)
        scores = torch.einsum("bqhd,bkhd->bhqk", qf[:, i0:i1], kf) * scale
        if causal:
            qi = torch.arange(i0, i1).unsqueeze(1)
            ki = torch.arange(s_k).unsqueeze(0)
            keep = ki <= qi + (s_k - s_q)
            scores = scores.masked_fill(~keep, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out[:, i0:i1] = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return out


def ref_decode(q, k, v):
    """One query token per sequence: q [B,H,D], k/v [B,H,S,D] -> [B,H,D]."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), k.float()) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", p, v.float())


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
    """Raise the clock before the first timed row (see bench_vs_refs)."""
    a = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ a
    del a
