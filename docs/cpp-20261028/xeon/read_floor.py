"""Read-bandwidth floor for the paged fp32 rows: torch.sum over the same bytes,
4 threads, min of 41 (the K+V caches of b=1/4096 and b=8/256, D=64, 16 heads)."""
import time

import torch

torch.set_num_threads(4)
for name, mb in (("b1/4096", 32), ("b8/256", 16)):
    x = torch.randn(mb * 1024 * 1024 // 4)
    for _ in range(5):
        x.sum()
    ts = []
    for _ in range(41):
        t0 = time.perf_counter_ns()
        x.sum()
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    t = min(ts)
    print(f"{name}: {mb} MB read in {t:.1f} us = {mb * 1.048576e6 / t / 1e3:.1f} GB/s")
