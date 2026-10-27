"""One dense row, four arms, one process.

The Ryzen throttles 20-50% over a ten-minute sweep, so a full-matrix run in one
process biases the later rows. This runs a single row per process: the arms are
still interleaved and rotated inside it (so the ratios are drift-free), and the
absolute numbers are taken while the core is cool.

    python tests/bench_row.py --seq 1024 --heads 16 --dim 128 --dtype fp32 \
        --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --rounds 21
    python tests/bench_matrix.py          # the whole matrix, one process per row
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


def dense_row(seq, heads, dim, dtype, vllm, isa, rounds, bsz=64):
    q = torch.randn(1, seq, heads, dim, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = dim ** -0.5

    def arm(be):
        def f():
            _kernel.set_backend(be)
            attn.flash_attn_func(q, k, v, scale, True)
        return f

    arms = {"cpp": arm("cpp"), "mojo": arm("mojo"), "ctrl": arm("cpp")}
    if vllm is not None:
        nb = (seq + bsz - 1) // bsz
        qf, kf, vf = q.reshape(seq, heads, dim), k.reshape(seq, heads, dim), \
            v.reshape(seq, heads, dim)
        kc = torch.zeros(nb, heads, bsz, dim, dtype=dtype)
        vc = torch.zeros_like(kc)
        vllm.cpu_attn_reshape_and_cache(kf, vf, kc, vc,
                                        torch.arange(seq, dtype=torch.int64), isa)
        lens = torch.tensor([seq], dtype=torch.int32)
        qsl = torch.tensor([0, seq], dtype=torch.int32)
        meta = vllm.get_scheduler_metadata(1, heads, heads, dim, lens, dtype,
                                          qsl, True, -1, isa, False, None,
                                          "auto")
        out = torch.empty_like(qf)
        bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
        arms["vllm"] = lambda: vllm.cpu_attention_with_kv_cache(
            qf, kc, vc, out, qsl, lens, scale, True, None, -1, bt, 0.0, meta,
            None, None, 1.0, 1.0, "auto")
    s = C.time_arms(arms, rounds)
    st = {n: C.stats(x) for n, x in s.items()}
    _kernel.set_backend("cpp")
    got = attn.flash_attn_func(q, k, v, scale, True)
    ref = C.ref_attention(q, k, v, True, dtype=torch.float32)
    row = {"kind": "dense", "seq": seq, "heads": heads, "dim": dim,
           "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
           "cpp_us": st["cpp"]["min"], "cpp_med": st["cpp"]["median"],
           "mojo_us": st["mojo"]["min"], "mojo_med": st["mojo"]["median"],
            "ctrl_us": st["ctrl"]["min"],
            "ctrl_med": st["ctrl"]["median"],
            "samples_us": s, "rounds": rounds,
            "threads": torch.get_num_threads(), "machine": platform.machine(),
            "loadavg": os.getloadavg(), "isa": isa, "output": "allocated",
            "d_cpp": (got.float() - ref).abs().max().item()}
    if not row["d_cpp"] <= (3e-2 if dtype == torch.bfloat16 else 2e-4):
        raise AssertionError(f"dense parity failed: {row}")
    if "vllm" in st:
        row["vllm_us"] = st["vllm"]["min"]
        row["vllm_med"] = st["vllm"]["median"]
    row["cpp/mojo"] = row["cpp_us"] / row["mojo_us"]
    row["ctrl/mojo"] = row["ctrl_us"] / row["mojo_us"]
    if "vllm_us" in row:
        row["cpp/vllm"] = row["cpp_us"] / row["vllm_us"]
        row["mojo/vllm"] = row["mojo_us"] / row["vllm_us"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    vllm = C.load_so(args.vllm)
    C.burn()
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    row = dense_row(args.seq, args.heads, args.dim, dt, vllm, args.isa,
                    args.rounds)
    line = (f"dense {row['seq']:5d}:{row['heads']:2d}:{row['dim']:3d} "
            f"{row['dtype']}: cpp {row['cpp_us']:9.1f} "
            f"(med {row['cpp_med']:9.1f}) mojo {row['mojo_us']:9.1f}  cpp/mojo "
            f"{row['cpp/mojo']:5.3f}  ctrl/mojo {row['ctrl/mojo']:5.3f}")
    if "vllm_us" in row:
        line += (f"  vllm {row['vllm_us']:9.1f} cpp/vllm {row['cpp/vllm']:5.3f}"
                 f" mojo/vllm {row['mojo/vllm']:5.3f}")
    print(line + f"  d {row['d_cpp']:.1e}", flush=True)
    print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([row], f, indent=1)
    return row


if __name__ == "__main__":
    main()
