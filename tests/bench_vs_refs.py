"""Dense attention vs torch SDPA and vLLM's csrc/cpu kernel, drift-controlled.

Every variant is timed in a round-robin loop inside one process and the median
per variant is reported, because these boxes throttle: on the laptop a
sustained AVX-512 sweep drops to ~2.0 GHz and the same binary measures 0.63-1.09
against SDPA depending on when it runs. Never compare two runs of this script
against each other -- compare the ratios inside one run.

    uv run python tests/bench_vs_refs.py                      # bf16 + fp32, causal
    uv run python tests/bench_vs_refs.py --dtype fp32 --causal 0
    uv run python tests/bench_vs_refs.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so

**Ratio convention (project-wide, see AGENTS.md §0): `ours / baseline`, so a
ratio below 1 means we are FASTER than that baseline.** `o/sdpa` and `o/vllm`
are printed in that order, and `<1` is a win in both columns.

vLLM's kernel is paged, so its K/V is staged into a block-table cache before
timing and the measured call is attention only; ours includes the K/V pack and
its allocations, i.e. ours is charged for work vLLM does not do once per call.
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402

SHAPES = [
    # (seqlen, nheads, head_dim)
    (128, 8, 64),
    (512, 8, 64),
    (1024, 8, 64),
    (2048, 8, 64),
    (4096, 8, 64),
    (256, 16, 128),
    (1024, 16, 128),
    (2048, 16, 128),
    (4096, 16, 128),
    (512, 32, 128),
    (1024, 8, 32),
]


def load_vllm(path):
    """Load the standalone csrc/cpu extension (see the build recipe in AGENTS.md).

    The module name has to match the file stem: the AMX variant is built with
    ``-DTORCH_EXTENSION_NAME=vllm_cpu_attn_bench_amx`` so that both references
    can live in one process, and its init function is named after it.
    """
    if not path or not os.path.exists(path):
        return None
    name = os.path.basename(path).rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paged_case(seq, heads, dim, dtype, block=128):
    """q/k/v plus a paged cache, block table, slot map and lengths for one request.

    Same construction as the vLLM-side benchmark: every token's K/V goes into the
    paged cache, so the timed call is pure attention.
    """
    q = torch.randn(seq, heads, dim, dtype=dtype)
    k = torch.randn(seq, heads, dim, dtype=dtype)
    v = torch.randn(seq, heads, dim, dtype=dtype)
    blocks = (seq + block - 1) // block
    kc = torch.zeros(blocks, heads, block, dim, dtype=dtype)
    vc = torch.zeros_like(kc)
    bt = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
    pos = torch.arange(seq, dtype=torch.int64)
    slots = pos // block * block + pos % block
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    lens = torch.tensor([seq], dtype=torch.int32)
    return q, k, v, kc, vc, bt, slots, qsl, lens


def time_rounds(variants, rounds, full=False):
    """Median of `rounds` interleaved samples per variant (and the min).

    The median is the published number, but on a shared box a burst of load
    from another job can land inside one variant's sample and move a median by
    15% (measured: 512:32:128 fp32 read 0.85 twice, then 1.01 twice, while an
    in-process interleaved probe of the same binary read 4046 us both times).
    The *minimum* is far more robust -- interference can only make a sample
    slower -- so it is reported beside the median and used when deciding
    whether a change helped.
    """
    for fn in variants.values():
        for _ in range(2):
            fn()
    samples = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            t0 = time.perf_counter()
            fn()
            samples[name].append((time.perf_counter() - t0) * 1e6)
    med = {name: statistics.median(v) for name, v in samples.items()}
    if full:
        return med, {name: min(v) for name, v in samples.items()}
    return med


def sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=q.shape[-1] ** -0.5,
    )


def parity(out, ref):
    """Max abs difference against the SDPA reference, at the dtype's tolerance.

    Printed beside every row so a fast-but-wrong kernel cannot look like a win:
    the tolerance is the one tests/test_dense.py asserts (fp32 2e-4, bf16 3e-2).
    """
    diff = (out.float() - ref.float()).abs().max().item()
    tol = 3e-2 if out.dtype == torch.bfloat16 else 2e-4
    return diff, diff <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32", "both"], default="both")
    ap.add_argument("--causal", choices=["0", "1", "both"], default="1")
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""),
                    help="path to the standalone csrc/cpu .so (optional)")
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"),
                    help="vLLM ISA hint: vec|amx|neon|...")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="", help="filter, e.g. 1024:16:128,512:32:128")
    ap.add_argument("--json", default="", help="write the table as JSON here")
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the max|diff| vs SDPA column (faster)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or torch.get_num_threads())
    vllm = load_vllm(args.vllm)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}[args.dtype]
    causals = {"0": [False], "1": [True], "both": [True, False]}[args.causal]
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in SHAPES if s in want]
    rows = []

    # Raise the clock before the first timed shape, for *one second*. These boxes
    # idle at a fraction of their sustained clock and two warmup calls per
    # variant do not ramp them, so the first row of a sweep otherwise reports the
    # idle number -- every variant, not just ours (Graviton, S=128 bf16: ours
    # 395 us in row 1 vs 54 us from row 2 on, vLLM 258 vs 40, SDPA 361 vs 89; a
    # 0.2 s burn was not enough either). A time budget, not an iteration count,
    # because the boxes differ by ~4x in throughput.
    _burn = torch.randn(768, 768)
    _t0 = time.perf_counter()
    while time.perf_counter() - _t0 < 1.2:
        _burn @ _burn
    del _burn, _t0

    print(f"threads={torch.get_num_threads()} vllm={args.vllm or '-'} isa={args.isa}")
    for causal in causals:
        print(f"\ncausal={causal}   us (ratios are ours/baseline: <1 = we win)")
        for seq, heads, dim in shapes:
            for dtype in dtypes:
                q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(
                    seq, heads, dim, dtype)
                out = torch.empty_like(q)
                q3 = q.unsqueeze(0)
                k3 = k.unsqueeze(0)
                v3 = v.unsqueeze(0)
                ours_out = {}

                def ours():
                    ours_out["v"] = attn.flash_attn_func(
                        q3, k3, v3, causal=causal)

                variants = {
                    "ours": ours,
                    "sdpa": lambda: sdpa(q3, k3, v3, causal),
                }
                if vllm is not None:
                    vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                    meta = vllm.get_scheduler_metadata(
                        len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                        -1, args.isa, False, None, "auto")
                    variants["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                        q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1,
                        bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))
                m, mmin = time_rounds(variants, args.rounds, full=True)
                d_ours, ok_ours = (float("nan"), True)
                if not args.no_parity:
                    ref = sdpa(q3, k3, v3, causal)
                    d_ours, ok_ours = parity(ours_out["v"].transpose(1, 2), ref)
                flops = 4.0 * heads * seq * seq * dim * (0.5 if causal else 1.0)
                gflops = {n: flops / (t * 1e-6) / 1e9 for n, t in m.items()}
                tag = "bf16" if dtype is torch.bfloat16 else "fp32"
                line = (f"  S={seq:5d} H={heads:2d} D={dim:3d} {tag}: "
                        f"{m['ours']:9.0f} us")
                if "vllm" in m:
                    line += f" (sdpa {m['sdpa']:8.0f}, vllm {m['vllm']:8.0f})"
                else:
                    line += f" (sdpa {m['sdpa']:8.0f})"
                line += f"   o/sdpa {m['ours']/m['sdpa']:5.2f}"
                if "vllm" in m:
                    line += f"  o/vllm {m['ours']/m['vllm']:5.2f}"
                line += f"   [{gflops['ours']:4.0f} GFLOP/s]"
                # min-based ratios: interference can only slow a sample down,
                # so these are the honest ones on a shared box (see time_rounds)
                line += f"  min o/vllm {mmin['ours']/mmin['vllm']:5.2f}" if "vllm" in mmin else f"  min o/sdpa {mmin['ours']/mmin['sdpa']:5.2f}"
                if not args.no_parity:
                    line += f"  d={d_ours:.1e}{'' if ok_ours else ' PARITY-FAIL'}"
                print(line, flush=True)
                rows.append({
                    "seq": seq, "heads": heads, "dim": dim, "dtype": tag,
                    "causal": causal, "us": m["ours"], "sdpa_us": m["sdpa"],
                    "vllm_us": m.get("vllm"),
                    "o_sdpa": m["ours"] / m["sdpa"],
                    "o_vllm": (m["ours"] / m["vllm"]) if "vllm" in m else None,
                    "gflops": gflops["ours"], "maxdiff": d_ours,
                    "parity_ok": ok_ours,
                    "threads": torch.get_num_threads(),
                })
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json}")
    bad = [r for r in rows if not r["parity_ok"]]
    if bad:
        print(f"\nPARITY FAILURES: {len(bad)} rows")
        for r in bad:
            print(f"  S={r['seq']} H={r['heads']} D={r['dim']} {r['dtype']}"
                  f" causal={r['causal']} d={r['maxdiff']:.2e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
