"""Minor page faults per dense call, per backend (is the output allocation faulting?)."""
import os
import resource
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, ROOT + "/audit"]
import torch  # noqa: E402
import cpu_attention as A  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

torch.set_num_threads(4)
for r in sys.argv[1].split(","):
    s, h, d, tag = r.split(":")
    s, h, d = int(s), int(h), int(d)
    dt = torch.float32 if tag == "fp32" else torch.bfloat16
    q, k, v = [torch.randn(1, s, h, d, dtype=dt) for _ in range(3)]
    for be in ("cpp", "mojo", "cpp", "mojo"):
        K.set_backend(be)
        for _ in range(5):
            A.flash_attn_func(q, k, v, d ** -.5, True)
        f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        for _ in range(50):
            A.flash_attn_func(q, k, v, d ** -.5, True)
        f1 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        print(r, be, (f1 - f0) / 50, "faults/call", flush=True)
