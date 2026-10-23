"""Per-row-ISA matrix for the Xeon AMX box: cpp vs mojo vs vLLM, one process.

`tests/bench_cpp.py` takes a single `--isa` for the whole run, but the two
baselines this box is judged against are different builds of vLLM: its AMX
kernel is the bf16 bar and its VEC kernel is the fp32 bar (the AMX kernel
`TORCH_CHECK`s on an fp32 query, so one process cannot hold both). This driver
takes the isa **per row** from the dtype and nothing else differs from
`bench_cpp.py`: same shapes, same rotation, same duplicate `ctrl` arm, same
`audit_common.time_arms`.

    python docs/xeon/matrix.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --rounds 41 --paged-rounds 15 --json /tmp/m.json

`cpp/mojo` and `cpp/vllm` are `ours/baseline`; below 1 is faster.
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

DENSE = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
         (4096, 8, 64), (256, 16, 128), (1024, 16, 128), (2048, 16, 128),
         (4096, 16, 128), (512, 32, 128), (1024, 8, 32)]
PAGED = [(1, 256), (8, 256), (8, 1024), (32, 1024), (1, 4096)]
HEADS, DIM = 16, 64

# The best vLLM build per dtype on this box (docs/amx/AMX_PAGED.md §1).
ISA = {"bf16": "amx", "fp32": "vec"}


def paged_case(batch, ctx, block, dtype):
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
    return q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens


def stage(k_all, v_all, kc, vc, block, batch, ctx, isa, so, backend):
    k_flat = k_all.reshape(batch * ctx, HEADS, DIM)
    v_flat = v_all.reshape(batch * ctx, HEADS, DIM)
    b = torch.arange(batch).repeat_interleave(ctx)
    t = torch.arange(ctx).repeat(batch)
    slots = (b * ((ctx + block - 1) // block) * block + t).to(torch.int64)
    if so is not None:
        so.cpu_attn_reshape_and_cache(k_flat, v_flat, kc, vc, slots, isa)
        return
    _kernel.set_backend(backend)
    attn.reshape_and_cache(k_flat, v_flat, kc, vc, slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--paged-rounds", type=int, default=15)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="dense,paged")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    vllm = C.load_so(args.vllm)
    C.burn()
    _kernel.set_backend("cpp")
    print(f"threads={torch.get_num_threads()} cpp-isa={cpp_isa()}"
          f" vllm={'yes' if vllm else '-'}", flush=True)
    rows = []
    dtypes = [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]

    if "dense" in args.only:
        for seq, heads, dim in DENSE:
            for dtype, tag in dtypes:
                isa = ISA[tag]
                q = torch.randn(1, seq, heads, dim, dtype=dtype)
                k = torch.randn(1, seq, heads, dim, dtype=dtype)
                v = torch.randn(1, seq, heads, dim, dtype=dtype)
                scale = dim ** -0.5
                res = {}

                def arm(be):
                    def f():
                        _kernel.set_backend(be)
                        res[be] = attn.flash_attn_func(q, k, v, scale, True)
                    return f

                arms = {"cpp": arm("cpp"), "mojo": arm("mojo"), "ctrl": arm("cpp")}
                if vllm is not None:
                    bs = 64
                    nb = (seq + bs - 1) // bs
                    kf, vf, qf = k.reshape(seq, heads, dim), v.reshape(seq, heads, dim), q.reshape(seq, heads, dim)
                    kc = torch.zeros(nb, heads, bs, dim, dtype=dtype)
                    vc = torch.zeros_like(kc)
                    vllm.cpu_attn_reshape_and_cache(
                        kf, vf, kc, vc, torch.arange(seq, dtype=torch.int64), isa)
                    lens = torch.tensor([seq], dtype=torch.int32)
                    qsl = torch.tensor([0, seq], dtype=torch.int32)
                    meta = vllm.get_scheduler_metadata(
                        1, heads, heads, dim, lens, dtype, qsl, True, -1,
                        isa, False, None, "auto")
                    out = torch.empty_like(qf)
                    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
                    arms["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                        qf, kc, vc, out, qsl, lens, scale, True, None, -1, bt,
                        0.0, meta, None, None, 1.0, 1.0, "auto"))
                s = C.time_arms(arms, args.rounds)
                st = {n: C.stats(x) for n, x in s.items()}
                _kernel.set_backend("cpp")
                got = attn.flash_attn_func(q, k, v, scale, True)
                ref = C.ref_attention(q, k, v, True, dtype=torch.float32)
                row = {"kind": "dense", "seq": seq, "heads": heads, "dim": dim,
                       "dtype": tag, "isa": isa,
                       "cpp_us": st["cpp"]["min"], "cpp_med": st["cpp"]["median"],
                       "mojo_us": st["mojo"]["min"], "mojo_med": st["mojo"]["median"],
                       "ctrl_us": st["ctrl"]["min"], "ctrl_med": st["ctrl"]["median"],
                       "d_cpp": (got.float() - ref).abs().max().item()}
                if "vllm" in st:
                    row.update(vllm_us=st["vllm"]["min"], vllm_med=st["vllm"]["median"])
                rows.append(row)
                print(fmt(row), flush=True)

    if "paged" in args.only:
        for dtype, tag in dtypes:
            isa = ISA[tag]
            for batch, ctx in PAGED:
                q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens = \
                    paged_case(batch, ctx, args.block, dtype)
                scale = DIM ** -0.5
                kcm, vcm = torch.zeros_like(kc), torch.zeros_like(vc)
                kcc, vcc = torch.zeros_like(kc), torch.zeros_like(vc)
                stage(k_all, v_all, kcm, vcm, args.block, batch, ctx, "mojo", None, "mojo")
                stage(k_all, v_all, kcc, vcc, args.block, batch, ctx, "cpp", None, "cpp")
                if vllm is not None:
                    stage(k_all, v_all, kc, vc, args.block, batch, ctx, isa, vllm, None)
                    qsl = torch.arange(batch + 1, dtype=torch.int32)
                    meta = vllm.get_scheduler_metadata(
                        batch, HEADS, HEADS, DIM, lens, dtype, qsl, True, -1,
                        isa, False, None, "auto")
                    vout = torch.empty_like(q)

                def arm_self(be):
                    def f():
                        _kernel.set_backend(be)
                        if be == "cpp":
                            return attn.self_attention(q, k_new, v_new, kcc, vcc, slots, blocks, lens, scale)
                        return attn.self_attention(q, k_new, v_new, kcm, vcm, slots, blocks, lens, scale)
                    return f

                def arm_cross(be):
                    def f():
                        _kernel.set_backend(be)
                        if be == "cpp":
                            return attn.cross_attention(q, kcc, vcc, blocks, lens, scale)
                        return attn.cross_attention(q, kcm, vcm, blocks, lens, scale)
                    return f

                for op, mk in (("self", arm_self), ("cross", arm_cross)):
                    arms = {"cpp": mk("cpp"), "mojo": mk("mojo"), "ctrl": mk("cpp")}
                    if vllm is not None:
                        if op == "self":
                            def vf():
                                vllm.cpu_attn_reshape_and_cache(k_new, v_new, kc, vc, slots, isa)
                                vllm.cpu_attention_with_kv_cache(
                                    q, kc, vc, vout, qsl, lens, scale, True, None, -1,
                                    blocks, 0.0, meta, None, None, 1.0, 1.0, "auto")
                        else:
                            def vf():
                                vllm.cpu_attention_with_kv_cache(
                                    q, kc, vc, vout, qsl, lens, scale, True, None, -1,
                                    blocks, 0.0, meta, None, None, 1.0, 1.0, "auto")
                        arms["vllm"] = vf
                    s = C.time_arms(arms, args.paged_rounds)
                    st = {n: C.stats(x) for n, x in s.items()}
                    _kernel.set_backend("cpp")
                    got = (attn.self_attention(q, k_new, v_new, kcc, vcc, slots, blocks, lens, scale)
                           if op == "self" else
                           attn.cross_attention(q, kcc, vcc, blocks, lens, scale))
                    ref = C.ref_decode(q, k_all.transpose(1, 2), v_all.transpose(1, 2))
                    row = {"kind": "paged", "op": op, "batch": batch, "ctx": ctx,
                           "dtype": tag, "block": args.block, "isa": isa,
                           "cpp_us": st["cpp"]["min"], "cpp_med": st["cpp"]["median"],
                           "mojo_us": st["mojo"]["min"], "mojo_med": st["mojo"]["median"],
                           "ctrl_us": st["ctrl"]["min"], "ctrl_med": st["ctrl"]["median"],
                           "d_cpp": (got.float() - ref).abs().max().item()}
                    if "vllm" in st:
                        row.update(vllm_us=st["vllm"]["min"], vllm_med=st["vllm"]["median"])
                    rows.append(row)
                    print(fmt(row), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


def fmt(r):
    name = (f"dense {r['seq']:5d}:{r['heads']:2d}:{r['dim']:3d}"
            if r["kind"] == "dense" else
            f"paged {r['op']:5s} b={r['batch']:<3} ctx={r['ctx']:<5}")
    out = (f"{name} {r['dtype']:4s} [{r['isa']:3s}]: "
           f"cpp {r['cpp_us']:9.1f} (med {r['cpp_med']:9.1f}) "
           f"mojo {r['mojo_us']:8.1f} cpp/mojo {r['cpp_us'] / r['mojo_us']:5.3f} "
           f"ctrl/ours {r['ctrl_us'] / r['cpp_us']:5.3f}")
    if "vllm_us" in r:
        out += (f"  vllm {r['vllm_us']:8.1f} (med {r['vllm_med']:8.1f}) "
                f"cpp/vllm {r['cpp_us'] / r['vllm_us']:5.3f}")
    return out + f"  d {r['d_cpp']:.1e}"


def cpp_isa():
    try:
        return _kernel._entry("isa")()
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    raise SystemExit(main())
