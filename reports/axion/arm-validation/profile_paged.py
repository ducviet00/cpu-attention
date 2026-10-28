import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "tests")]
import bench
import cpu_attention as ca

torch.set_num_threads(8)
torch.manual_seed(0)
dtype = torch.bfloat16 if sys.argv[1] == "bf16" else torch.float32
r = dict(b=8, h=16, hkv=16, d=64, ctx=1024)
x = bench.paged_inputs(r, dtype, 16)
kc = torch.zeros(x["shape"], dtype=dtype)
vc = torch.zeros_like(kc)
ca.reshape_and_cache(torch.cat(x["kall"]), torch.cat(x["vall"]), kc, vc, x["allslots"])
out = torch.empty_like(x["q"])
end = time.monotonic() + 8
n = 0
while time.monotonic() < end:
    ca.cross_attention(x["q"], kc, vc, x["bt"], x["lens"], 0.125, out=out)
    n += 1
print("calls", n, "dtype", dtype)
