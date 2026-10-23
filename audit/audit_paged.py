"""Independent paged (decoder-step) audit: ours vs vLLM csrc/cpu.

Mirrors the engine's shapes: heads=16, head_dim=64, one new token per sequence
(``self``) and a pre-staged encoder cache (``cross``). Ground truth is a direct
softmax over the cache contents, not SDPA.

    python audit_paged.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --isa neon --block 16 --dtypes bf16,fp32 --rounds 11 --json /tmp/p.json
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

HEADS, DIM = 16, 64
CASES = [(1, 256), (8, 256), (8, 1024), (32, 1024), (1, 4096)]


def make_case(batch, ctx, block, dtype):
    pages = (ctx + block - 1) // block
    blocks = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    slots = blocks[:, (ctx - 1) // block].long() * block + (ctx - 1) % block
    lens = torch.full((batch,), ctx, dtype=torch.int32)
    shape = (batch * pages, HEADS, block, DIM)
    q = torch.randn(batch, HEADS, DIM, dtype=dtype)
    k_new = torch.randn(batch, HEADS, DIM, dtype=dtype)
    v_new = torch.randn(batch, HEADS, DIM, dtype=dtype)
    k_all = torch.randn(batch, ctx, HEADS, DIM, dtype=dtype)
    v_all = torch.randn(batch, ctx, HEADS, DIM, dtype=dtype)
    k_all[:, ctx - 1] = k_new
    v_all[:, ctx - 1] = v_new
    kc = torch.zeros(shape, dtype=dtype)
    vc = torch.zeros(shape, dtype=dtype)
    return q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens, pages


def stage(k_all, v_all, kc, vc, block, batch, ctx, isa, so=None):
    """Write every token of every sequence into the paged cache (once)."""
    k_flat = k_all.reshape(batch * ctx, HEADS, DIM)
    v_flat = v_all.reshape(batch * ctx, HEADS, DIM)
    b = torch.arange(batch).repeat_interleave(ctx)
    t = torch.arange(ctx).repeat(batch)
    slots = b * ((ctx + block - 1) // block) * block + t
    slots = slots.to(torch.int64)
    if so is not None:
        so.cpu_attn_reshape_and_cache(k_flat, v_flat, kc, vc, slots, isa)
        return
    attn.reshape_and_cache(k_flat, v_flat, kc, vc, slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    vllm = C.load_so(args.vllm)
    dtypes = [{"bf16": torch.bfloat16, "fp32": torch.float32}[t] for t in args.dtypes.split(",")]
    C.burn()
    print(f"threads={torch.get_num_threads()} block={args.block} "
          f"vllm={args.vllm or '-'} isa={args.isa}", flush=True)
    rows = []
    for dtype in dtypes:
        tag = "bf16" if dtype is torch.bfloat16 else "fp32"
        for batch, ctx in CASES:
            q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens, pages = \
                make_case(batch, ctx, args.block, dtype)
            kc_o = torch.zeros_like(kc)
            vc_o = torch.zeros_like(vc)
            stage(k_all, v_all, kc_o, vc_o, args.block, batch, ctx, args.isa, None)
            if vllm is not None:
                stage(k_all, v_all, kc, vc, args.block, batch, ctx, args.isa, vllm)
            scale = DIM ** -0.5
            qsl = torch.arange(batch + 1, dtype=torch.int32)
            out = torch.empty_like(q)
            ours_out = {}

            def ours_self():
                ours_out["v"] = attn.self_attention(
                    q, k_new, v_new, kc_o, vc_o, slots, blocks, lens, scale)

            def ours_cross():
                ours_out["c"] = attn.cross_attention(
                    q, kc_o, vc_o, blocks, lens, scale)

            arms = {"ours": ours_self, "ctrl": ours_cross}
            if vllm is not None:
                meta = vllm.get_scheduler_metadata(
                    batch, HEADS, HEADS, DIM, lens, dtype, qsl, True, -1,
                    args.isa, False, None, "auto")

                def vllm_self():
                    vllm.cpu_attn_reshape_and_cache(
                        k_new, v_new, kc, vc, slots, args.isa)
                    vllm.cpu_attention_with_kv_cache(
                        q, kc, vc, out, qsl, lens, scale, True, None, -1,
                        blocks, 0.0, meta, None, None, 1.0, 1.0, "auto")

                def vllm_cross():
                    vllm.cpu_attention_with_kv_cache(
                        q, kc, vc, out, qsl, lens, scale, True, None, -1,
                        blocks, 0.0, meta, None, None, 1.0, 1.0, "auto")

                arms["vllm"] = vllm_self
                cross_arms = {"ours": ours_cross, "vllm": vllm_cross}
            else:
                cross_arms = {"ours": ours_cross}

            # ---- self (store + attend) ----
            s = C.time_arms(arms, args.rounds)
            st = {n: C.stats(v) for n, v in s.items()}
            row = {"op": "self", "batch": batch, "ctx": ctx, "heads": HEADS,
                   "dim": DIM, "block": args.block, "dtype": tag,
                   "ours_us": st["ours"]["min"], "ours_med": st["ours"]["median"],
                   "ctrl_us": st["ctrl"]["min"],
                   "ctrl_med": st["ctrl"]["median"]}
            if "vllm" in st:
                row.update(vllm_us=st["vllm"]["min"],
                           vllm_med=st["vllm"]["median"],
                           o_vllm=st["ours"]["min"] / st["vllm"]["min"],
                           o_vllm_med=st["ours"]["median"] / st["vllm"]["median"])
            if not args.no_ref:
                ref = C.ref_decode(q, k_all.transpose(1, 2), v_all.transpose(1, 2))
                row["d_ours"] = (ours_out["v"].float() - ref).abs().max().item()
                if vllm is not None:
                    row["d_vllm"] = (out.float() - ref).abs().max().item()
            line = (f"self  b={batch:<3} ctx={ctx:<5} {tag}: ours {row['ours_us']:9.1f}"
                    f" (med {row['ours_med']:9.1f})")
            if "vllm_us" in row:
                line += (f"  vllm {row['vllm_us']:9.1f}  o/vllm {row['o_vllm']:5.2f}"
                         f" (med {row['o_vllm_med']:5.2f})")
            if not args.no_ref:
                line += f"  d(ours) {row['d_ours']:.1e}"
                if "d_vllm" in row:
                    line += f" d(vllm) {row['d_vllm']:.1e}"
            print(line, flush=True)
            rows.append(row)

            # ---- cross (attend only) ----
            s = C.time_arms(cross_arms, args.rounds)
            st = {n: C.stats(v) for n, v in s.items()}
            row = {"op": "cross", "batch": batch, "ctx": ctx, "heads": HEADS,
                   "dim": DIM, "block": args.block, "dtype": tag,
                   "ours_us": st["ours"]["min"], "ours_med": st["ours"]["median"]}
            ours_out.clear()
            ours_cross()
            if "vllm" in st:
                row.update(vllm_us=st["vllm"]["min"],
                           vllm_med=st["vllm"]["median"],
                           o_vllm=st["ours"]["min"] / st["vllm"]["min"],
                           o_vllm_med=st["ours"]["median"] / st["vllm"]["median"])
            if not args.no_ref:
                ref = C.ref_decode(q, k_all.transpose(1, 2), v_all.transpose(1, 2))
                row["d_ours"] = (ours_out["c"].float() - ref).abs().max().item()
                if vllm is not None:
                    row["d_vllm"] = (out.float() - ref).abs().max().item()
            line = (f"cross b={batch:<3} ctx={ctx:<5} {tag}: ours {row['ours_us']:9.1f}"
                    f" (med {row['ours_med']:9.1f})")
            if "vllm_us" in row:
                line += (f"  vllm {row['vllm_us']:9.1f}  o/vllm {row['o_vllm']:5.2f}"
                         f" (med {row['o_vllm_med']:5.2f})")
            if not args.no_ref:
                line += f"  d(ours) {row['d_ours']:.1e}"
                if "d_vllm" in row:
                    line += f" d(vllm) {row['d_vllm']:.1e}"
            print(line, flush=True)
            rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
