"""Call one dense row in a loop for perf record: loop.py 1024:8:32:fp32 [secs] [backend]."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path[:0] = [ROOT]
import torch  # noqa: E402
import cpu_attention as A  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

torch.set_num_threads(4)
s, h, d, tag = sys.argv[1].split(":")
s, h, d = int(s), int(h), int(d)
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
K.set_backend(sys.argv[3] if len(sys.argv) > 3 else "cpp")
dt = torch.float32 if tag == "fp32" else torch.bfloat16
q, k, v = [torch.randn(1, s, h, d, dtype=dt) for _ in range(3)]
n, t0 = 0, time.time()
while time.time() - t0 < secs:
    A.flash_attn_func(q, k, v, d ** -.5, True)
    n += 1
print(n, "calls", (time.time() - t0) / n * 1e6, "us/call")
