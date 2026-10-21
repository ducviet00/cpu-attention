"""Shared setup for the x86 profiling drivers (see PROFILE_X86.md).

Everything here is measurement scaffolding: shapes, a loader for the standalone
vLLM `csrc/cpu` extension, the paged-cache construction that kernel needs, and
an interleaved timer.  Nothing in this file may change kernel behaviour -- the
point is to measure the tree as it stands.

Protocol (fixed across platforms so the numbers are comparable):

  * one core (`taskset -c 3`), `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`
  * every variant built once, warmed twice, then sampled round-robin so drift
    can only ever favour the variant that runs first in a round, not one side
  * medians of >= 9 samples, in ONE process; never compare separate runs
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cpu_attention as attn  # noqa: E402

# (seq, heads, head_dim) -- the four fixed protocol shapes.  S=128 is the
# fixed-cost case, S=1024/H=16/D=128 the register-wall case.
SHAPES = [
    (2048, 8, 64),
    (1024, 16, 128),
    (512, 8, 64),
    (128, 8, 64),
]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

VLLM_DEFAULT = os.environ.get(
    "VLLM_REF", os.path.expanduser("~/vllm_cpu_build/vllm_cpu_attn_bench.so")
)


def set_single_thread() -> None:
    """One thread, and make the kernel agree (it reads torch's thread count)."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised by a previous call


def load_vllm(path: str | None = None):
    path = path or VLLM_DEFAULT
    if not path or not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("vllm_cpu_attn_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case(seq: int, heads: int, dim: int, dtype, block: int = 128):
    """q/k/v plus the paged cache, block table, slot map and lengths.

    vLLM's kernel is paged, so K/V are staged once (outside the timed region)
    and the timed call is attention only.  Ours is charged for its K^T
    transpose pass; `components.py` splits that out.
    """
    q = torch.randn(seq, heads, dim, dtype=dtype)
    k = torch.randn(seq, heads, dim, dtype=dtype)
    v = torch.randn(seq, heads, dim, dtype=dtype)
    blocks = (seq + block - 1) // block
    kc = torch.zeros(blocks, heads, block, dim, dtype=dtype)
    vc = torch.zeros_like(kc)
    bt = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
    pos = torch.arange(seq, dtype=torch.int64)
    slots = pos // block * block + pos % block
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    lens = torch.tensor([seq], dtype=torch.int32)
    return dict(q=q, k=k, v=v, kc=kc, vc=vc, bt=bt, slots=slots, qsl=qsl, lens=lens)


def build_vllm(vllm, c, heads: int, dim: int, dtype, isa: str, causal: bool,
               kv_split: bool = False):
    """Stage the cache and make scheduler metadata once; return the call."""
    vllm.cpu_attn_reshape_and_cache(c["k"], c["v"], c["kc"], c["vc"],
                                    c["slots"], isa)
    meta = vllm.get_scheduler_metadata(
        len(c["lens"]), heads, heads, dim, c["lens"], dtype, c["qsl"], causal,
        -1, isa, kv_split, None, "auto")
    out = torch.empty_like(c["q"])

    def call():
        return vllm.cpu_attention_with_kv_cache(
            c["q"], c["kc"], c["vc"], out, c["qsl"], c["lens"], dim ** -0.5,
            causal, None, -1, c["bt"], 0.0, meta, None, None, 1.0, 1.0, "auto")

    call.out = out
    call.meta = meta
    return call


def sdpa_fn(c, causal: bool):
    q, k, v = c["q"].unsqueeze(0), c["k"].unsqueeze(0), c["v"].unsqueeze(0)
    scale = q.shape[-1] ** -0.5
    return lambda: torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=scale)


def ours_fn(c, causal: bool):
    q, k, v = c["q"].unsqueeze(0), c["k"].unsqueeze(0), c["v"].unsqueeze(0)
    return lambda: attn.flash_attn_func(q, k, v, causal=causal)


def timer(variants: dict, rounds: int = 9, warm: int = 2) -> dict:
    """Median microseconds per variant, interleaved inside one process."""
    for fn in variants.values():
        for _ in range(warm):
            fn()
    samples = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            t0 = time.perf_counter_ns()
            fn()
            samples[name].append((time.perf_counter_ns() - t0) / 1e3)
    return {name: statistics.median(v) for name, v in samples.items()}


def flops(seq: int, heads: int, dim: int, causal: bool, seq_k: int | None = None):
    seq_k = seq if seq_k is None else seq_k
    n = seq * seq_k * (0.5 if causal else 1.0)
    return 4.0 * heads * n * dim


def report(tag: str, med: dict, seq: int, heads: int, dim: int, causal: bool,
           ref: str = "sdpa") -> dict:
    f = flops(seq, heads, dim, causal)
    gf = {n: f / (t * 1e-6) / 1e9 for n, t in med.items()}
    line = f"  S={seq:5d} H={heads:2d} D={dim:3d} {tag}: " + " ".join(
        f"{n}={t:8.1f}us" for n, t in med.items())
    if ref in med:
        line += "  |"
        for n in med:
            if n == ref:
                continue
            line += f" {n}/{ref}={med[n]/med[ref]:5.2f}x"
    line += "  (" + " ".join(f"{n} {gf[n]:5.0f} GF/s" for n in med) + ")"
    print(line, flush=True)
    return dict(tag=tag, seq=seq, heads=heads, dim=dim, us=med, gflops=gf)
