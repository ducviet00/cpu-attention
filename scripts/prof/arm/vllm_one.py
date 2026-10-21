"""Same shape loop as one.py, but through vLLM's csrc/cpu kernel only.

The paged cache is staged (``reshape_and_cache``) and the scheduler metadata is
built *before* the timed loop, so the measurement is attention only -- that is
the contract ``bench_vs_refs.py`` and the campaign objective use.

    python scripts/prof/arm/vllm_one.py <S> <H> <D> <bf16|fp32> <causal 0|1> <iters>
"""

import importlib.util
import os
import sys
import time

import torch

S, H, D, dt, causal, iters = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5] == "1",
    int(sys.argv[6]),
)
dtype = torch.bfloat16 if dt == "bf16" else torch.float32
torch.set_num_threads(4)
if os.environ.get("MAX_THREADS"):
    torch.set_num_threads(int(os.environ["MAX_THREADS"]))

SO = "/home/ubuntu/vllm_cpu_build/vllm_cpu_attn_bench.so"
spec = importlib.util.spec_from_file_location("vllm_cpu_attn_bench", SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

block = 128
q = torch.randn(S, H, D, dtype=dtype)
k = torch.randn(S, H, D, dtype=dtype)
v = torch.randn(S, H, D, dtype=dtype)
out = torch.empty_like(q)
blocks = (S + block - 1) // block
kc = torch.zeros(blocks, H, block, D, dtype=dtype)
vc = torch.zeros_like(kc)
bt = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
pos = torch.arange(S, dtype=torch.int64)
slots = pos // block * block + pos % block
qsl = torch.tensor([0, S], dtype=torch.int32)
lens = torch.tensor([S], dtype=torch.int32)
m.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, "neon")
meta = m.get_scheduler_metadata(
    len(lens), H, H, D, lens, dtype, qsl, causal, -1, "neon", False, None, "auto"
)
f = lambda: m.cpu_attention_with_kv_cache(  # noqa: E731
    q, kc, vc, out, qsl, lens, D**-0.5, causal, None, -1, bt, 0.0, meta,
    None, None, 1.0, 1.0, "auto",
)
for _ in range(3):
    f()
t0 = time.perf_counter()
for _ in range(iters):
    f()
t = (time.perf_counter() - t0) / iters * 1e6
rk = H * (S * (S + 1) // 2 if causal else S * S)
print(
    f"vllm {S}x{H}x{D} {dt} causal={causal}: {t:.1f} us  "
    f"{t*1e3/rk:.3f} ns/row-key"
)
