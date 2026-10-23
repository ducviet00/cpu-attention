"""Shape sweep: where does the per-call fixed cost come from.

For one (heads, dim, dtype) walk `seq` upward; the slope of the min against the
causal logit count is the per-logit cost and the intercept is the per-call
fixed cost. Both backends and a duplicate `cpp` arm in one process.
"""

import argparse
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

import audit_common as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="32,64,128,256,512")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rounds", type=int, default=31)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    C.burn()
    rows = []
    for seq in (int(x) for x in args.seqs.split(",")):
        q = torch.randn(1, seq, args.heads, args.dim, dtype=dt)
        k = torch.randn(1, seq, args.heads, args.dim, dtype=dt)
        v = torch.randn(1, seq, args.heads, args.dim, dtype=dt)
        scale = args.dim ** -0.5

        def carm():
            _kernel.set_backend("cpp")
            return attn.flash_attn_func(q, k, v, scale, True)

        def marm():
            _kernel.set_backend("mojo")
            return attn.flash_attn_func(q, k, v, scale, True)

        s = C.time_arms({"cpp": carm, "ctrl": carm, "mojo": marm}, args.rounds)
        st = {n: C.stats(x) for n, x in s.items()}
        logits = (seq * (seq + 1) / 2) * args.heads
        row = {"seq": seq, "logits": logits,
               "cpp": st["cpp"]["min"], "ctrl": st["ctrl"]["min"],
               "mojo": st["mojo"]["min"]}
        rows.append(row)
        print(f"seq {seq:5d} logits {logits:9.0f}: cpp {row['cpp']:8.1f} "
              f"ctrl {row['ctrl']:8.1f} mojo {row['mojo']:8.1f}  "
              f"cpp/mojo {row['cpp'] / row['mojo']:5.3f}  "
              f"cpp {1000 * row['cpp'] / logits:6.2f} ns/logit  "
              f"mojo {1000 * row['mojo'] / logits:6.2f}", flush=True)
    # least squares on the two backends
    for k_ in ("cpp", "mojo", "ctrl"):
        n = len(rows)
        sx = sum(r["logits"] for r in rows)
        sy = sum(r[k_] for r in rows)
        sxx = sum(r["logits"] ** 2 for r in rows)
        sxy = sum(r["logits"] * r[k_] for r in rows)
        a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        b = (sy - a * sx) / n
        print(f"  {k_}: slope {1000 * a:.3f} ns/logit  intercept {b:6.1f} us")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
