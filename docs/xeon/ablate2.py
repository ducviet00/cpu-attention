"""Phase ablation with an unablated reference arm in the same process.

`CPU_ATTENTION_AMX_MODE` is read by both kernels, but at different times: the
Mojo backend reads it in Python **at import** (`_AMX_MODE_V`) while the C++
kernel reads it lazily through `getenv` on its first call. Setting the variable
after the import therefore ablates only the C++ kernel, and the Mojo arm stays a
fixed reference against which the drift of each process divides out.

One process per mode (`env_int` caches after the first call); `ctrl` is a
duplicate `cpp` arm and gives the floor.

    python docs/xeon/ablate2.py --shape 128:8:64 --dtype bf16 --reps 3

Bits: 1 pack, 2 Q@K^T, 4 softmax, 8 P@V, 16 the Q hi/lo panel, 32 the mask.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

import audit_common as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402

MODES = [(0, "full"), (1, "-pack"), (2, "-qk"), (4, "-softmax"), (8, "-pv"),
         (16, "-qpanel"), (32, "-mask")]


def inner(shape, dtype, rounds, mode):
    # The import already happened, so only the C++ kernel sees this value.
    os.environ["CPU_ATTENTION_AMX_MODE"] = str(mode)
    seq, heads, dim = (int(x) for x in shape.split(":"))
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    q = torch.randn(1, seq, heads, dim, dtype=dt)
    k = torch.randn(1, seq, heads, dim, dtype=dt)
    v = torch.randn(1, seq, heads, dim, dtype=dt)
    scale = dim ** -0.5

    def carm():
        _kernel.set_backend("cpp")
        return attn.flash_attn_func(q, k, v, scale, True)

    def marm():
        _kernel.set_backend("mojo")
        return attn.flash_attn_func(q, k, v, scale, True)

    C.burn()
    s = C.time_arms({"cpp": carm, "ctrl": carm, "mojo": marm}, rounds)
    return {n: C.stats(x) for n, x in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="128:8:64")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    got = {tag: [] for _, tag in MODES}
    for _ in range(args.reps):
        for mode, tag in MODES:
            r = subprocess.run(
                [sys.executable, __file__, "--inner", "--shape", args.shape,
                 "--dtype", args.dtype, "--rounds", str(args.rounds),
                 "--mode", str(mode)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:])
                raise SystemExit(1)
            got[tag].append(json.loads(r.stdout.strip().splitlines()[-1]))

    rows = []
    base = None
    print(f"{args.shape} {args.dtype}  rounds={args.rounds} reps={args.reps}")
    for mode, tag in MODES:
        rs = got[tag]
        cpp = statistics.median(x["cpp"]["min"] for x in rs)
        mojo = statistics.median(x["mojo"]["min"] for x in rs)
        ratio = statistics.median(x["cpp"]["min"] / x["mojo"]["min"] for x in rs)
        floor = statistics.median(x["ctrl"]["min"] / x["cpp"]["min"] for x in rs)
        if base is None:
            base = cpp
        row = {"mode": mode, "tag": tag, "cpp": cpp, "mojo": mojo,
               "cpp/mojo": ratio, "ctrl/cpp": floor}
        rows.append(row)
        # The Mojo arm is not ablated, so `ratio` is proportional to our time:
        # the share is drift-free even though each mode is its own process.
        print(f"  {tag:9s} mode={mode:2d} cpp {cpp:9.1f} mojo {mojo:9.1f} "
              f"cpp/mojo {ratio:5.3f} ctrl/cpp {floor:5.3f}   phase "
              f"{100 * (1 - ratio / rows[0]['cpp/mojo']):6.1f}% of the call")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"shape": args.shape, "dtype": args.dtype, "rows": rows},
                      f, indent=1)
    return 0


if __name__ == "__main__":
    if "--inner" in sys.argv:
        a = sys.argv[sys.argv.index("--shape") + 1]
        d = sys.argv[sys.argv.index("--dtype") + 1]
        r = int(sys.argv[sys.argv.index("--rounds") + 1])
        m = int(sys.argv[sys.argv.index("--mode") + 1])
        print(json.dumps(inner(a, d, r, m)))
    else:
        raise SystemExit(main())
