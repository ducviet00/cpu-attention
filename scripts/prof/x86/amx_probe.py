"""Is AMX actually usable in this process/VM?

torch's oneDNN bf16 GEMM is AMX-backed on a server part: it reaches ~350 GFLOP/s
per core, which is above the 250 GFLOP/s AVX-512 fp32 FMA ceiling and 3x the
120 GFLOP/s `vdpbf16ps` ceiling measured here. So a single-threaded bf16 matmul
is a direct probe of "did the CPU give us AMX".

    .venv/bin/python scripts/prof/x86/amx_probe.py
"""

import torch, time, statistics

torch.set_num_threads(1)


def bench(fn, rounds=7):
    for _ in range(3):
        fn()
    s = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


for n in (512, 1024, 2048):
    a = torch.randn(n, n, dtype=torch.bfloat16)
    b = torch.randn(n, n, dtype=torch.bfloat16)
    t = bench(lambda: a @ b)
    print(f"bf16 matmul {n:5d}: {2*n**3/t/1e9:7.1f} GFLOP/s")

af = torch.randn(1024, 1024)
bf = torch.randn(1024, 1024)
t = bench(lambda: af @ bf)
print(f"fp32 matmul  1024: {2*1024**3/t/1e9:7.1f} GFLOP/s")

# attention, for reference (the thing we are chasing)
for (s, h, d) in ((1024, 16, 128), (2048, 8, 64)):
    q = torch.randn(1, h, s, d, dtype=torch.bfloat16)
    k = torch.randn(1, h, s, d, dtype=torch.bfloat16)
    v = torch.randn(1, h, s, d, dtype=torch.bfloat16)
    f = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    t = bench(f)
    print(f"sdpa bf16 {s}:{h}:{d}: {4*h*s*s*d*0.5/t/1e9:7.1f} GFLOP/s")

print("torch amx flag:", torch.cpu._is_amx_tile_supported())
print("mkldnn:", torch.backends.mkldnn.is_available())
