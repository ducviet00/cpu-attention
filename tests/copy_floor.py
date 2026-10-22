import sys, time, statistics, torch
sys.path.insert(0, ".")
torch.set_num_threads(4)
for name, nbytes in (("S=128 (K+V 256 KB)", 256*1024), ("S=512 (K+V 1 MB)", 1024*1024)):
    buf = torch.randn(nbytes // 2, dtype=torch.bfloat16)
    dst = torch.empty_like(buf)
    f = lambda: dst.copy_(buf)
    for _ in range(5):
        f()
    s = []
    for _ in range(21):
        t0 = time.perf_counter(); f(); s.append((time.perf_counter() - t0) * 1e6)
    t = statistics.median(s)
    print(f"{name:22s} copy {t:7.2f} us -> {2*nbytes/t/1e3:6.1f} GB/s (r+w)")
