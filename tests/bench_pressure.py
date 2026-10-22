"""Does the *neighbouring* arm change our number? (cache-pressure ablation)

`tests/bench_vs_refs.py` reports this kernel at ~214 us for 256:16:128 bf16
while the same binary timed alone in one process reads ~185 us. The arms differ:
the matrix interleaves SDPA and vLLM with every sample, so our samples start
cache-cold. This script measures ours in three contexts inside one process,
at a pinned `qt`, min-based and rotated:

    alone          ours only
    with-sdpa      ours interleaved with SDPA (the matrix's worst neighbour)
    with-vllm      ours interleaved with vLLM's paged kernel

If `alone` is much faster than the others, the scorecard's numbers are traffic,
not arithmetic, and per-row-key K/V traffic (`qt`) is the lever, not the tile.

    .venv/bin/python tests/bench_pressure.py --shape 256:16:128 --qts 8,16
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402
from bench_vs_refs import load_vllm, paged_case, sdpa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="256:16:128")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--qts", default="8,16")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default="neon")
    args = ap.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    S, H, D = (int(x) for x in args.shape.split(":"))
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vllm = load_vllm(args.vllm)
    q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(S, H, D, dt)
    q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = torch.empty_like(q)
    if vllm is not None:
        vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
        meta = vllm.get_scheduler_metadata(
            len(lens), H, H, D, lens, dt, qsl, True, -1, args.isa, False, None,
            "auto")

    def vl():
        vllm.cpu_attention_with_kv_cache(
            q, kc, vc, out, qsl, lens, D ** -0.5, True, None, -1, bt, 0.0,
            meta, None, None, 1.0, 1.0, "auto")

    _burn = torch.randn(768, 768)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.2:
        _burn @ _burn
    del _burn

    for qt in [int(x) for x in args.qts.split(",")]:
        K.set_qt(qt)
        ours = lambda: attn.flash_attn_func(q3, k3, v3, causal=True)
        variants = {"alone": (ours,)}
        variants["with-sdpa"] = (ours, lambda: sdpa(q3, k3, v3, True))
        if vllm is not None:
            variants["with-vllm"] = (ours, vl)
        for name, arms in variants.items():
            for f in arms:
                f()
                f()
            s = [[] for _ in arms]
            for r in range(args.rounds):
                for j in range(len(arms)):
                    t0 = time.perf_counter()
                    arms[j]()
                    s[j].append((time.perf_counter() - t0) * 1e6)
            print(f"qt={qt:2d} {name:10s} ours min {min(s[0]):7.1f} "
                  f"med {statistics.median(s[0]):7.1f}"
                  + (f"   other min {min(s[1]):7.1f}" if len(arms) > 1 else ""),
                  flush=True)
    K.set_qt(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
