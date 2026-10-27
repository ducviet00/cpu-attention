"""Phase ablation of the C++ AMX dense kernel, Mojo as the in-process drift arm.

    python abl.py --mode 4 --rows 128:8:64:fp32 --rounds 41

CPU_ATTENTION_AMX_MODE is read lazily by the C++ kernel on its first call and
by Mojo at import, so setting it here (after the import) ablates only C++.
Results are garbage when mode != 0; only the time is reported.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, ROOT + "/audit"]
import torch  # noqa: E402
import audit_common as C  # noqa: E402
import cpu_attention as A  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", type=int, default=0)
p.add_argument("--rows", default="128:8:64:fp32")
p.add_argument("--rounds", type=int, default=41)
p.add_argument("--json", default="")
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(0)
os.environ["CPU_ATTENTION_AMX_MODE"] = str(a.mode)
C.burn()
out = []
for r in a.rows.split(","):
    s, h, d, tag = r.split(":")
    s, h, d = int(s), int(h), int(d)
    dt = torch.float32 if tag == "fp32" else torch.bfloat16
    q, k, v = [torch.randn(1, s, h, d, dtype=dt) for _ in range(3)]

    def arm(be):
        def f():
            K.set_backend(be)
            A.flash_attn_func(q, k, v, d ** -.5, True)
        return f
    smp = C.time_arms({"cpp": arm("cpp"), "mojo": arm("mojo")}, a.rounds, 3)
    st = {n: C.stats(x) for n, x in smp.items()}
    row = dict(row=r, mode=a.mode, cpp=st["cpp"]["min"],
               cpp_med=st["cpp"]["median"], mojo=st["mojo"]["min"],
               mojo_med=st["mojo"]["median"])
    print(json.dumps(row), flush=True)
    print(f"# {r:18s} mode {a.mode:3d} cpp {row['cpp']:8.1f} ({row['cpp_med']:8.1f})"
          f" mojo {row['mojo']:8.1f} ({row['mojo_med']:8.1f})"
          f" c/m {row['cpp'] / row['mojo']:.3f}"
          f" med {row['cpp_med'] / row['mojo_med']:.3f}", flush=True)
    out.append(row)
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
