"""Paired-process A/B between two builds of `_C_cpp.so`.

A code variant cannot share a process with its baseline (one `.so` is loaded per
backend), so the two builds are installed alternately and each process times
`cpp`, a duplicate `ctrl`, and `mojo`. The Mojo arm is identical in both builds
and is what the drift is divided out by; the reported number is the median over
processes of `cpp/mojo`, which is a paired statistic.

    python docs/xeon/ab_so.py --sos /tmp/a.so,/tmp/b.so --shape 512:8:64 \
        --dtype fp32 --rounds 41 --pairs 5
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

LIB = os.path.join(_ROOT, "cpu_attention", "_C_cpp.so")


def inner(shape, dtype, rounds, block=None, op=None):
    import torch
    import audit_common as C
    import cpu_attention as attn
    from cpu_attention import _kernel

    if block is None:
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
    else:
        # paged: shape is "batch:ctx", HEADS/DIM as in tests/bench_cpp.py
        heads, dim = 16, 64
        batch, ctx = (int(x) for x in shape.split(":"))
        dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        pages = (ctx + block - 1) // block
        blocks = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
        slots = blocks[:, (ctx - 1) // block].long() * block + (ctx - 1) % block
        lens = torch.full((batch,), ctx, dtype=torch.int32)
        q = torch.randn(batch, heads, dim, dtype=dt)
        k_all = torch.randn(batch, ctx, heads, dim, dtype=dt)
        v_all = torch.randn(batch, ctx, heads, dim, dtype=dt)
        kc = torch.zeros((batch * pages, heads, block, dim), dtype=dt)
        vc = torch.zeros_like(kc)
        scale = dim ** -0.5
        caches = {}
        for be in ("cpp", "mojo"):
            kk, vv = torch.zeros_like(kc), torch.zeros_like(vc)
            _kernel.set_backend(be)
            attn.reshape_and_cache(k_all.reshape(-1, heads, dim),
                                   v_all.reshape(-1, heads, dim), kk, vv,
                                   torch.arange(batch * ctx, dtype=torch.int64))
            caches[be] = (kk, vv)

        def carm():
            _kernel.set_backend("cpp")
            return attn.cross_attention(q, caches["cpp"][0], caches["cpp"][1],
                                        blocks, lens, scale)

        def marm():
            _kernel.set_backend("mojo")
            return attn.cross_attention(q, caches["mojo"][0], caches["mojo"][1],
                                        blocks, lens, scale)

    C.burn()
    s = C.time_arms({"cpp": carm, "ctrl": carm, "mojo": marm}, rounds)
    return {n: C.stats(x) for n, x in s.items()}


def run_variant(so, shape, dtype, rounds, block, op):
    shutil.copyfile(so, LIB)
    r = subprocess.run([sys.executable, __file__, "--inner", "--shape", shape,
                        "--dtype", dtype, "--rounds", str(rounds),
                        "--block", str(block or 0)], capture_output=True,
                       text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sos", required=True)
    ap.add_argument("--shape", default="512:8:64")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--block", type=int, default=0, help="paged when set")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    sos = args.sos.split(",")
    order = []
    for i in range(args.pairs):
        order += sos if i % 2 == 0 else list(reversed(sos))
        order += list(reversed(sos)) if i % 2 == 0 else sos
    got = {s: [] for s in sos}
    for s in order:
        got[s].append(run_variant(s, args.shape, args.dtype, args.rounds,
                                  args.block, None))

    print(f"{args.shape} {args.dtype} rounds={args.rounds} "
          f"n={len(got[sos[0]])}")
    out = []
    for s in sos:
        rows = got[s]
        cpp = statistics.median(r["cpp"]["min"] for r in rows)
        ratio = statistics.median(r["cpp"]["min"] / r["mojo"]["min"] for r in rows)
        floor = statistics.median(r["ctrl"]["min"] / r["cpp"]["min"] for r in rows)
        out.append({"so": s, "cpp": cpp, "cpp/mojo": ratio, "ctrl/cpp": floor,
                    "mojo": statistics.median(r["mojo"]["min"] for r in rows),
                    "samples": [round(r["cpp"]["min"], 1) for r in rows],
                    "ratios": [round(r["cpp"]["min"] / r["mojo"]["min"], 3)
                               for r in rows]})
        print(f"  {os.path.basename(s):14s} cpp {cpp:9.1f} mojo "
              f"{out[-1]['mojo']:9.1f}  cpp/mojo {ratio:5.3f} "
              f"ctrl/cpp {floor:5.3f}  ratios {out[-1]['ratios']}")
    if len(sos) == 2:
        a, b = out
        print(f"  >> B/A paired {b['cpp/mojo'] / a['cpp/mojo']:5.3f}   "
              f"raw {b['cpp'] / a['cpp']:5.3f}   (<1 = B faster)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"shape": args.shape, "dtype": args.dtype, "rows": out}, f,
                      indent=1)
    return 0


if __name__ == "__main__":
    if "--inner" in sys.argv:
        a = sys.argv[sys.argv.index("--shape") + 1]
        d = sys.argv[sys.argv.index("--dtype") + 1]
        r = int(sys.argv[sys.argv.index("--rounds") + 1])
        b = int(sys.argv[sys.argv.index("--block") + 1])
        print(json.dumps(inner(a, d, r, b or None)))
    else:
        raise SystemExit(main())
