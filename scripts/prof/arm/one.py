"""Time and/or profile the cpu_attention dense kernel on one shape.

    python scripts/prof/arm/one.py <S> <H> <D> <bf16|fp32> <causal 0|1> <iters>

Prints the per-call time and the ns per *row-key* (one query row attending one
key), which is the kernel's only meaningful unit of work: ``row-keys =
H * S*(S+1)/2`` causal, ``H*S*S`` otherwise.

Used with ``scripts/prof/arm/pk.py`` (differential ``perf stat``) to get
instructions and cycles per row-key.
"""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/cpu-attention")
import cpu_attention as attn  # noqa: E402

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
q = torch.randn(1, S, H, D, dtype=dtype)
k = torch.randn(1, S, H, D, dtype=dtype)
v = torch.randn(1, S, H, D, dtype=dtype)
f = lambda: attn.flash_attn_func(q, k, v, causal=causal)  # noqa: E731
for _ in range(3):
    f()
t0 = time.perf_counter()
for _ in range(iters):
    f()
t = (time.perf_counter() - t0) / iters * 1e6
rk = H * (S * (S + 1) // 2 if causal else S * S)
print(
    f"{S}x{H}x{D} {dt} causal={causal}: {t:.1f} us  {rk} row-keys  "
    f"{t*1e3/rk:.3f} ns/row-key"
)
