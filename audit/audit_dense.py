"""Independent dense-attention audit: ours vs vLLM csrc/cpu vs torch SDPA.

Truth comes from a first-principles blockwise reference, not from SDPA.

    python audit_dense.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --isa neon --dtypes bf16,fp32 --rounds 11 --json /tmp/out.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as C  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cpu_attention as attn  # noqa: E402

SHAPES = [
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


def paged_case(seq, heads, dim, dtype, block=128):
    """Identical construction to the repo harness, so vLLM sees the same tensors."""
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


def sdpa_arm(q3, k3, v3, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q3.transpose(1, 2), k3.transpose(1, 2), v3.transpose(1, 2),
        is_causal=causal, scale=q3.shape[-1] ** -0.5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    causal = bool(args.causal)
    vllm = C.load_so(args.vllm)
    dtypes = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
              "both": [torch.bfloat16, torch.float32]}
    if args.dtypes in dtypes:
        dtypes = dtypes[args.dtypes]
    else:
        dtypes = [{"bf16": torch.bfloat16, "fp32": torch.float32}[t] for t in args.dtypes.split(",")]
    shapes = SHAPES
    if args.shapes:
        want = {tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")}
        shapes = [s for s in SHAPES if s in want]

    C.burn()
    print(f"threads={torch.get_num_threads()} causal={causal} "
          f"vllm={args.vllm or '-'} isa={args.isa}", flush=True)
    rows = []
    for seq, heads, dim in shapes:
        for dtype in dtypes:
            tag = "bf16" if dtype is torch.bfloat16 else "fp32"
            q, k, v, kc, vc, bt, slots, qsl, lens = paged_case(seq, heads, dim, dtype)
            q3, k3, v3 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            out = torch.empty_like(q)
            ours_out = {}

            def ours():
                ours_out["v"] = attn.flash_attn_func(q3, k3, v3, causal=causal)

            arms = {"ours": ours, "sdpa": lambda: sdpa_arm(q3, k3, v3, causal),
                    "ctrl": ours}
            if vllm is not None:
                vllm.cpu_attn_reshape_and_cache(k, v, kc, vc, slots, args.isa)
                meta = vllm.get_scheduler_metadata(
                    len(lens), heads, heads, dim, lens, dtype, qsl, causal,
                    -1, args.isa, False, None, "auto")
                arms["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                    q, kc, vc, out, qsl, lens, dim ** -0.5, causal, None, -1,
                    bt, 0.0, meta, None, None, 1.0, 1.0, "auto"))
            samples = C.time_arms(arms, args.rounds)
            st = {name: C.stats(s) for name, s in samples.items()}
            m = {name: v["min"] for name, v in st.items()}
            med = {name: v["median"] for name, v in st.items()}
            row = {
                "seq": seq, "heads": heads, "dim": dim, "dtype": tag,
                "causal": causal, "rounds": args.rounds,
                "ours_us": m["ours"], "ours_med": med["ours"],
                "sdpa_us": m["sdpa"], "sdpa_med": med["sdpa"],
                "ctrl_us": m["ctrl"], "ctrl_med": med["ctrl"],
                "o_sdpa": m["ours"] / m["sdpa"],
                "o_sdpa_med": med["ours"] / med["sdpa"],
            }
            if "vllm" in m:
                row.update(vllm_us=m["vllm"], vllm_med=med["vllm"],
                           o_vllm=m["ours"] / m["vllm"],
                           o_vllm_med=med["ours"] / med["vllm"])
            if not args.no_ref:
                ref = C.ref_attention(q3, k3, v3, causal)
                d_ours = (ours_out["v"].float() - ref).abs().max().item()
                d_vllm = (out.float() - ref).abs().max().item()
                d_sdpa = (sdpa_arm(q3, k3, v3, causal).transpose(1, 2).float()
                          - ref).abs().max().item()
                row.update(d_ours=d_ours, d_vllm=d_vllm, d_sdpa=d_sdpa)
            line = (f"S={seq:5d} H={heads:2d} D={dim:3d} {tag}: "
                    f"ours {m['ours']:8.1f} (med {med['ours']:8.1f}) "
                    f"sdpa {m['sdpa']:8.1f}")
            if "vllm" in m:
                line += (f" vllm {m['vllm']:8.1f}  o/vllm {row['o_vllm']:5.2f}"
                         f" (med {row['o_vllm_med']:5.2f})")
            line += f"  o/sdpa {row['o_sdpa']:5.2f}  ctrl {m['ctrl']/m['ours']:5.2f}"
            if not args.no_ref:
                line += (f"  d(ours) {row['d_ours']:.1e} d(vllm) {row['d_vllm']:.1e}"
                         f" d(sdpa) {row['d_sdpa']:.1e}")
            print(line, flush=True)
            rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
