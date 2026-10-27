import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "tests")]
import torch
import bench_cpp as bench
import cpu_attention as attn

torch.set_num_threads(8)
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
dtype = torch.bfloat16 if len(sys.argv) > 2 else torch.float32
q, kn, vn, ka, va, kc, vc, slots, bt, lens = bench.paged_case(1, ctx, 32, dtype)
bench.stage(ka, va, kc, vc, 32, 1, ctx, "cpp", None, "cpp")
for _ in range(8000 if ctx > 256 else 50000):
    attn.cross_attention(q, kc, vc, bt, lens, 0.125)
