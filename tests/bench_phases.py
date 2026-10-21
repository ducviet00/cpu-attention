"""Where the time goes in one dense call: K/V pack vs the row kernel.

`ours` is charged for the K/V pack (transposed K, sometimes a repacked V) and
its scratch allocations, while vLLM's number is attention only (its cache is
staged before the timer). That is the contract, so it is also a place to win:
this script splits our call into the part that is the pack and the part that is
the kernel, interleaved with SDPA and vLLM, and prints the pack share.

It works by wrapping `cpu_attention._kernel._transpose_attention`, which
`_kernel._dense` resolves at call time, so the numbers come from the real call
path and cannot drift from the shipped layout logic.

    .venv/bin/python tests/bench_phases.py --dtype both
    .venv/bin/python tests/bench_phases.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa neon

Columns: full = flash_attn_func (what the engine calls); pack = the K(+V) pack
inside it; kernel = full - pack; pack% = pack/full. `o/sdpa`, `o/vllm` are the
project-wide ours/baseline ratios (<1 wins), computed from `full`.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The pack and the kernel share one OpenMP region (one fork-join instead of
# two), so the wrapper below has nothing to time unless the two calls are
# separated: `CPU_ATTENTION_NO_FUSE=1` restores them.
os.environ.setdefault("CPU_ATTENTION_NO_FUSE", "1")
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import SHAPES, load_vllm, paged_case, sdpa  # noqa: E402


class _PackTimer:
    """Time every `dense_transpose` call made inside one dense call."""

    def __init__(self) -> None:
        self.time = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._inner(*args, **kwargs)
        self.time += time.perf_counter() - t0
        self.calls += 1
        return out

    def install(self) -> None:
        self._inner = K._transpose_attention()
        K._transpose_attention = lambda: self


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32", "both"], default="both")
    ap.add_argument("--causal", choices=["0", "1"], default="1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="", help="e.g. 128:8:64,1024:16:128")
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    causal = args.causal == "1"
    vllm = load_vllm(args.vllm)
    timer = _PackTimer()
    timer.install()
    shapes = ([(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}[args.dtype]

    print(f"threads={torch.get_num_threads()} causal={causal} "
          f"vllm={args.vllm or '-'} isa={args.isa}")
    print("  S      H   D  dtype   full    pack  kernel  pack%   o/sdpa  o/vllm")
    for seq, heads, dim in shapes:
        for dtype in dtypes:
            q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                seq, heads, dim, dtype)
            q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            out = torch.empty_like(q)

            def ours():
                timer.time = 0.0
                t0 = time.perf_counter()
                attn.flash_attn_func(q3, k3, v3, causal=causal)
                total = time.perf_counter() - t0
                return total, timer.time

            variants = {"ours": ours, "sdpa": lambda: sdpa(q3, k3, v3, causal)}
            if vllm is not None:
                vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                meta = vllm.get_scheduler_metadata(
                    len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                    -1, args.isa, False, None, "auto")
                variants["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                    q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1,
                    bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))

            for name, fn in variants.items():
                fn()
            samples = {name: [] for name in variants}
            packs = []
            for _ in range(args.rounds):
                for name, fn in variants.items():
                    t0 = time.perf_counter()
                    res = fn()
                    dt = (time.perf_counter() - t0) * 1e6
                    if name == "ours":
                        packs.append(res[1] * 1e6)
                        dt = res[0] * 1e6
                    samples[name].append(dt)
            med = {n: statistics.median(v) for n, v in samples.items()}
            pack = statistics.median(packs)
            tag = "bf16" if dtype is torch.bfloat16 else "fp32"
            line = (f"{seq:5d} {heads:4d} {dim:4d} {tag}  {med['ours']:7.0f}"
                    f" {pack:7.0f} {med['ours'] - pack:7.0f}"
                    f" {100 * pack / med['ours']:5.1f}%"
                    f"   {med['ours'] / med['sdpa']:6.2f}")
            if "vllm" in med:
                line += f"  {med['ours'] / med['vllm']:6.2f}"
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
