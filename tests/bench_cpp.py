"""Three-arm A/B for the C++ backend: cpp vs mojo vs vLLM, one process.

The Mojo backend and the C++ backend are both loaded and alternated inside this
process (``_kernel.set_backend``), because two separate runs on these boxes drift
5-10% -- more than most of the effects being measured. A duplicate ``cpp`` arm is
always kept as the noise floor.

    python tests/bench_cpp.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --isa vec --rounds 21 --json /tmp/cpp.json

``ours/baseline``, below 1 is faster.
"""

import argparse
import json
import os
import platform
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def one(name, fn, rounds, warmup=2):
    out = C.time_arms({"x": fn}, rounds, warmup)["x"]
    return C.stats(out)


def time_arms(arms, rounds, calls):
    # Blocks keep short laptop calls in one power state. Divide every sample,
    # including the duplicate control, by the same number of calls.
    def block(fn):
        def run():
            for _ in range(calls):
                fn()
        return run

    samples = C.time_arms({n: block(fn) for n, fn in arms.items()}, rounds)
    return {n: [x / calls for x in xs] for n, xs in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--paged-rounds", type=int, default=41)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="dense,paged")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "both"), default="both")
    ap.add_argument("--preallocated-out", action="store_true",
                    help="give BOTH paged backends out= (default: both allocate)")
    ap.add_argument("--block-calls", type=int, default=1,
                    help="calls per timed arm; use 40 for short laptop calls")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if min(args.rounds, args.paged_rounds, args.block_calls) < 1:
        ap.error("rounds and block-calls must be positive")

    if args.threads:
        torch.set_num_threads(args.threads)
    vllm = C.load_so(args.vllm)
    C.burn()
    print(f"threads={torch.get_num_threads()} isa={args.isa} "
          f"cpp-isa={_kernel.set_backend('cpp') or attn_backend_isa()} "
          f"vllm={'yes' if vllm else '-'}", flush=True)
    rows = []
    dtypes = [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]
    dtypes = [(dt, tag) for dt, tag in dtypes
              if args.dtype in ("both", tag)]
    torch.manual_seed(0)

    def save(row):
        row.update(threads=torch.get_num_threads(), isa=args.isa,
                   machine=platform.machine(), rounds=(args.rounds
                       if row["kind"] == "dense" else args.paged_rounds),
                   block_calls=args.block_calls,
                   output="preallocated" if row["kind"] == "paged" and
                       args.preallocated_out else "allocated",
                   loadavg=os.getloadavg(),
                   samples_us=s)
        rows.append(row)
        # A spot preemption must not destroy all preceding rows.
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)

    if "dense" in args.only:
        for seq, heads, dim in DENSE:
            for dtype, tag in dtypes:
                q = torch.randn(1, seq, heads, dim, dtype=dtype)
                k = torch.randn(1, seq, heads, dim, dtype=dtype)
                v = torch.randn(1, seq, heads, dim, dtype=dtype)
                q3, k3, v3 = q, k, v
                scale = dim ** -0.5
                res = {}

                def arm(be):
                    def f():
                        _kernel.set_backend(be)
                        res[be] = attn.flash_attn_func(q3, k3, v3, scale, True)
                    return f

                arms = {"cpp": arm("cpp"), "mojo": arm("mojo"),
                        "ctrl": arm("cpp")}
                if vllm is not None:
                    bs = 64
                    nb = (seq + bs - 1) // bs
                    kf = k.reshape(seq, heads, dim)
                    vf = v.reshape(seq, heads, dim)
                    qf = q.reshape(seq, heads, dim)
                    kc = torch.zeros(nb, heads, bs, dim, dtype=dtype)
                    vc = torch.zeros_like(kc)
                    vllm.cpu_attn_reshape_and_cache(
                        kf, vf, kc, vc, torch.arange(seq, dtype=torch.int64),
                        args.isa)
                    lens = torch.tensor([seq], dtype=torch.int32)
                    qsl = torch.tensor([0, seq], dtype=torch.int32)
                    meta = vllm.get_scheduler_metadata(
                        1, heads, heads, dim, lens, dtype, qsl, True, -1,
                        args.isa, False, None, "auto")
                    out = torch.empty_like(qf)
                    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
                    arms["vllm"] = (lambda: vllm.cpu_attention_with_kv_cache(
                        qf, kc, vc, out, qsl, lens, scale, True, None, -1, bt,
                        0.0, meta, None, None, 1.0, 1.0, "auto"))
                s = time_arms(arms, args.rounds, args.block_calls)
                st = {n: C.stats(x) for n, x in s.items()}
                ref = None
                _kernel.set_backend("cpp")
                got = attn.flash_attn_func(q3, k3, v3, scale, True)
                ref = C.ref_attention(q3, k3, v3, True, dtype=torch.float32)
                row = {"kind": "dense", "seq": seq, "heads": heads, "dim": dim,
                       "dtype": tag,
                       "cpp_us": st["cpp"]["min"], "cpp_med": st["cpp"]["median"],
                       "mojo_us": st["mojo"]["min"],
                       "mojo_med": st["mojo"]["median"],
                        "ctrl_us": st["ctrl"]["min"],
                       "ctrl_med": st["ctrl"]["median"],
                       "d_cpp": (got.float() - ref).abs().max().item()}
                if "vllm" in st:
                    row.update(vllm_us=st["vllm"]["min"],
                               vllm_med=st["vllm"]["median"])
                row["cpp/mojo"] = row["cpp_us"] / row["mojo_us"]
                row["ctrl/mojo"] = row["ctrl_us"] / row["mojo_us"]
                line = (f"dense {seq:5d}:{heads:2d}:{dim:3d} {tag}: "
                        f"cpp {row['cpp_us']:9.1f} (med {row['cpp_med']:9.1f}) "
                        f"mojo {row['mojo_us']:9.1f}  cpp/mojo "
                        f"{row['cpp/mojo']:5.3f}  ctrl/mojo "
                        f"{row['ctrl/mojo']:5.3f}")
                if "vllm_us" in row:
                    row["cpp/vllm"] = row["cpp_us"] / row["vllm_us"]
                    row["mojo/vllm"] = row["mojo_us"] / row["vllm_us"]
                    line += (f"  vllm {row['vllm_us']:9.1f} cpp/vllm "
                             f"{row['cpp/vllm']:5.3f} mojo/vllm "
                             f"{row['mojo/vllm']:5.3f}")
                print(line + f"  d {row['d_cpp']:.1e}", flush=True)
                if not row["d_cpp"] <= (3e-2 if tag == "bf16" else 2e-4):
                    raise AssertionError(f"dense parity failed: {row}")
                save(row)

    if "paged" in args.only:
        for dtype, tag in dtypes:
            for batch, ctx in PAGED:
                q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks, lens = \
                    paged_case(batch, ctx, args.block, dtype)
                scale = DIM ** -0.5
                kcm = torch.zeros_like(kc)
                vcm = torch.zeros_like(vc)
                kcc = torch.zeros_like(kc)
                vcc = torch.zeros_like(vc)
                stage(k_all, v_all, kcm, vcm, args.block, batch, ctx, "mojo",
                      None, "mojo")
                stage(k_all, v_all, kcc, vcc, args.block, batch, ctx, "cpp",
                      None, "cpp")
                if vllm is not None:
                    stage(k_all, v_all, kc, vc, args.block, batch, ctx,
                          args.isa, vllm, None)
                    qsl = torch.arange(batch + 1, dtype=torch.int32)
                    meta = vllm.get_scheduler_metadata(
                        batch, HEADS, HEADS, DIM, lens, dtype, qsl, True, -1,
                        args.isa, False, None, "auto")
                    vout = torch.empty_like(q)

                # Allocation policy must be identical for cpp and Mojo. vLLM's
                # output remains preallocated, as in the original audit.
                ocpp = torch.empty_like(q)
                omojo = torch.empty_like(q)

                def arm_self(be):
                    kk, vv = (kcc, vcc) if be == "cpp" else (kcm, vcm)
                    out = (ocpp if be == "cpp" else omojo) \
                        if args.preallocated_out else None
                    def f():
                        _kernel.set_backend(be)
                        return attn.self_attention(q, k_new, v_new, kk, vv,
                                                   slots, blocks, lens, scale,
                                                   out=out)
                    return f

                def arm_cross(be):
                    kk, vv = (kcc, vcc) if be == "cpp" else (kcm, vcm)
                    out = (ocpp if be == "cpp" else omojo) \
                        if args.preallocated_out else None
                    def f():
                        _kernel.set_backend(be)
                        return attn.cross_attention(q, kk, vv, blocks, lens,
                                                    scale, out=out)
                    return f

                for op, mk in (("self", arm_self), ("cross", arm_cross)):
                    res = {}
                    arms = {"cpp": mk("cpp"), "mojo": mk("mojo"),
                            "ctrl": mk("cpp")}
                    if vllm is not None:
                        if op == "self":
                            def vf():
                                vllm.cpu_attn_reshape_and_cache(
                                    k_new, v_new, kc, vc, slots, args.isa)
                                vllm.cpu_attention_with_kv_cache(
                                    q, kc, vc, vout, qsl, lens, scale, True,
                                    None, -1, blocks, 0.0, meta, None, None,
                                    1.0, 1.0, "auto")
                        else:
                            def vf():
                                vllm.cpu_attention_with_kv_cache(
                                    q, kc, vc, vout, qsl, lens, scale, True,
                                    None, -1, blocks, 0.0, meta, None, None,
                                    1.0, 1.0, "auto")
                        arms["vllm"] = vf
                    s = time_arms(arms, args.paged_rounds, args.block_calls)
                    st = {n: C.stats(x) for n, x in s.items()}
                    _kernel.set_backend("cpp")
                    got = (attn.self_attention(q, k_new, v_new, kcc, vcc, slots,
                                               blocks, lens, scale)
                           if op == "self" else
                           attn.cross_attention(q, kcc, vcc, blocks, lens,
                                                scale))
                    ref = C.ref_decode(q, k_all.transpose(1, 2),
                                       v_all.transpose(1, 2))
                    row = {"kind": "paged", "op": op, "batch": batch,
                           "ctx": ctx, "dtype": tag, "block": args.block,
                           "cpp_us": st["cpp"]["min"],
                           "cpp_med": st["cpp"]["median"],
                           "mojo_us": st["mojo"]["min"],
                           "mojo_med": st["mojo"]["median"],
                            "ctrl_us": st["ctrl"]["min"],
                            "ctrl_med": st["ctrl"]["median"],
                           "d_cpp": (got.float() - ref).abs().max().item()}
                    if "vllm" in st:
                        row.update(vllm_us=st["vllm"]["min"],
                                   vllm_med=st["vllm"]["median"])
                    row["cpp/mojo"] = row["cpp_us"] / row["mojo_us"]
                    line = (f"paged {op:5s} b={batch:<3} ctx={ctx:<5} {tag}: "
                            f"cpp {row['cpp_us']:8.1f} (med {row['cpp_med']:8.1f})"
                            f" mojo {row['mojo_us']:8.1f}  cpp/mojo "
                            f"{row['cpp/mojo']:5.3f} ctrl/mojo "
                            f"{row['ctrl_us'] / row['mojo_us']:5.3f}")
                    if "vllm_us" in row:
                        row["cpp/vllm"] = row["cpp_us"] / row["vllm_us"]
                        row["mojo/vllm"] = row["mojo_us"] / row["vllm_us"]
                        line += (f"  vllm {row['vllm_us']:8.1f} cpp/vllm "
                                 f"{row['cpp/vllm']:5.3f} mojo/vllm "
                                 f"{row['mojo/vllm']:5.3f}")
                    print(line + f"  d {row['d_cpp']:.1e}", flush=True)
                    if tag == "fp32" and not row["d_cpp"] <= 1e-6:
                        raise AssertionError(f"paged parity failed: {row}")
                    save(row)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


def attn_backend_isa():
    try:
        return _kernel._entry("isa")()
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    raise SystemExit(main())
