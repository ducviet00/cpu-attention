"""A/B two builds of the AMX softmax in one process, via two library copies.

`ab_two_trees.py` compares two *trees*; this does the same trick for one file:
build the old `_csrc_kernels.so` and the new one under different names and let
`CPU_ATTENTION_CSRC` pick between them, interleaved in one process (these boxes
throttle 2-2.5x, so a cross-run comparison is meaningless).

    python scripts/prof/x86/amx_ab.py old.so new.so [--shapes ...]
"""

import argparse
import importlib
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from bench_vs_refs import SHAPES, sdpa  # noqa: E402


def load(path):
    """A fresh cpu_attention._kernel bound to one library copy."""
    os.environ["CPU_ATTENTION_CSRC"] = path
    for name in [m for m in list(sys.modules) if m.startswith("cpu_attention")]:
        del sys.modules[name]
    mod = importlib.import_module("cpu_attention._kernel")
    assert mod.amx_available(), path
    return mod


def call(K, q, k, v, dim, causal=True):
    return K._dense(q, k, v, dim ** -0.5, causal)


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
            f()
    s = {n: [] for n in fns}
    for _ in range(rounds):
        for n, f in fns.items():
            t0 = time.perf_counter()
            f()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: statistics.median(v) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("libs", nargs="+")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    mods = [(os.path.basename(p).replace("_csrc_kernels", "").strip("_.so") or "base",
             load(p)) for p in args.libs]
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in shapes if s in want]
    print(f"threads={torch.get_num_threads()}  bf16 causal us; ratios vs {mods[0][0]}")
    for seq, heads, dim in shapes:
        q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
        ref = sdpa(q, k, v, True)
        fns = {}
        for name, K in mods:
            fns[name] = (lambda K=K: call(K, q, k, v, dim))
        m = bench(fns, args.rounds)
        # parity of every variant, so a fast-but-wrong one cannot win
        bad = []
        for name, K in mods:
            got = call(K, q, k, v, dim).transpose(1, 2)
            d = (got.float() - ref.float()).abs().max().item()
            if d > 3e-2:
                bad.append(f"{name}={d:.1e}")
        base = m[mods[0][0]]
        line = f"  S={seq:5d} H={heads:2d} D={dim:3d}: "
        line += "  ".join(f"{n}={m[n]:7.0f} ({base/m[n]:5.1%} saved)"
                          for n, _ in mods)
        if bad:
            line += "  PARITY FAIL " + ",".join(bad)
        print(line)
    os.environ.pop("CPU_ATTENTION_CSRC", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
