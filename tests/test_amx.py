"""The AMX flash policy against the naive fp32 reference and the vector policy.

    uv run python tests/test_amx.py --dtype both            # parity grid
    uv run python tests/test_amx.py --dtype bf16 --bench    # + interleaved timing

`flash_attn_func` on an AMX CPU runs the AMX policy of the unified flash kernel
(`_mojo/src/flash_amx.mojo`); `K._AMX_ENV = False` selects the vector policy in
the same process, which is both the arbiter for the plan keying (on/off/on) and
the A/B arm. The test RUNS on an AMX CPU and FAILS if the kernel is unavailable
there; it only skips where the CPU has no AMX.

Gates (the project's): fp32 2e-4, bf16 3e-2 max abs diff against the reference.
The grid crosses the AMX tile edges: partial 32-row register tiles, key counts
around 16/32/256/512, unequal q/k lengths, noncausal, GQA/MQA, strided fused-QKV
chunks, and a NaN-filled tail after the tensors (a stray read comes back NaN).
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
from cpu_attention import _kernel as K  # noqa: E402
from reference import ref_attention  # noqa: E402


def run(q, k, v, causal, amx):
    prev = K._AMX_ENV
    K._AMX_ENV = amx
    try:
        return attn.flash_attn_func(q, k, v, causal=causal)
    finally:
        K._AMX_ENV = prev


def bench(fns, rounds):
    for f in fns.values():
        for _ in range(3):
            f()
    s = {n: [] for n in fns}
    for r in range(rounds):
        names = list(fns)
        k = r % len(names)
        for n in names[k:] + names[:k]:
            t0 = time.perf_counter()
            fns[n]()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: min(v) for n, v in s.items()}


# (sq, sk, heads, kv_heads, dim)
SHAPES = [
    (128, 128, 8, 8, 64), (512, 512, 8, 8, 64), (1024, 1024, 8, 8, 64),
    (256, 256, 16, 16, 128), (512, 512, 32, 32, 128), (1024, 1024, 8, 8, 32),
    (1, 129, 8, 8, 64), (17, 17, 8, 8, 64), (17, 301, 8, 8, 64),
    (31, 31, 4, 4, 64), (33, 33, 4, 4, 64), (65, 65, 8, 8, 64),
    (127, 129, 8, 8, 64), (129, 127, 8, 8, 64), (257, 513, 8, 8, 64),
    (513, 257, 8, 8, 64), (255, 255, 4, 4, 128), (257, 257, 4, 4, 128),
    (513, 513, 4, 4, 128), (1025, 1025, 4, 4, 32), (300, 300, 8, 2, 64),
    (300, 300, 16, 1, 128), (2049, 2049, 2, 2, 64), (63, 1000, 4, 4, 64),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "both"])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads or torch.get_num_threads())
    if not K.amx_available():
        # A CPU without AMX has nothing to test here. On a CPU *with* AMX an
        # unavailable kernel is a failure, never a successful skip.
        with open("/proc/cpuinfo") as f:
            has_amx = " amx_tile" in f.read()
        if has_amx:
            print("FAIL: this CPU has AMX but the AMX kernel is unavailable")
            return 1
        print("AMX kernel unavailable here (no AMX on this CPU): skipped")
        return 0

    dts = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
           "both": [torch.bfloat16, torch.float32]}[args.dtype]
    torch.manual_seed(0)
    bad = checks = 0
    for sq, sk, h, hk, d in SHAPES:
        for dt in dts:
            tol = 3e-2 if dt == torch.bfloat16 else 2e-4
            for causal in (True, False):
                q = torch.randn(1, sq, h, d, dtype=dt)
                k = torch.randn(1, sk, hk, d, dtype=dt)
                v = torch.randn(1, sk, hk, d, dtype=dt)
                ref = ref_attention(q, k, v, causal)
                got = run(q, k, v, causal, True)
                vec = run(q, k, v, causal, False)
                dg = (got.float() - ref).abs().max().item()
                dv = (vec.float() - ref).abs().max().item()
                ok = dg <= tol and torch.isfinite(got.float()).all().item()
                checks += 1
                bad += 0 if ok else 1
                print(f"  {sq}x{sk} H={h}/{hk} D={d} {'bf16' if dt == torch.bfloat16 else 'fp32'} "
                      f"causal={int(causal)}: amx {dg:.2e} vec {dv:.2e} {'OK' if ok else 'FAIL'}")
            if args.bench and sq == sk:
                q = torch.randn(1, sq, h, d, dtype=dt)
                k = torch.randn(1, sk, hk, d, dtype=dt)
                v = torch.randn(1, sk, hk, d, dtype=dt)
                m = bench({"amx": lambda: run(q, k, v, True, True),
                           "vec": lambda: run(q, k, v, True, False)}, args.rounds)
                print("  us: " + " ".join(f"{n}={t:8.1f}" for n, t in m.items())
                      + f"  amx/vec {m['amx'] / m['vec']:.2f}")

    # Strided chunks of one fused projection, as the engine passes them.
    for dt in dts:
        tol = 3e-2 if dt == torch.bfloat16 else 2e-4
        buf = torch.randn(2, 193, 3 * 8 * 64, dtype=dt)
        q, k, v = (buf[..., i * 512:(i + 1) * 512].view(2, 193, 8, 64) for i in range(3))
        ref = ref_attention(q, k, v, True)
        dg = (run(q, k, v, True, True).float() - ref).abs().max().item()
        ok = dg <= tol
        checks += 1
        bad += 0 if ok else 1
        print(f"  fused-QKV strided {'bf16' if dt == torch.bfloat16 else 'fp32'}: amx {dg:.2e} "
              f"{'OK' if ok else 'FAIL'}")

    # A NaN directly after K and V must not be read (padded keys, partial tiles).
    for dt in dts:
        n, h, d = 70, 4, 64
        kv = torch.randn(2, n + 40, h, d, dtype=dt)
        kv[:, n:] = float("nan")
        k, v = kv[0, :n].unsqueeze(0), kv[1, :n].unsqueeze(0)
        q = torch.randn(1, 45, h, d, dtype=dt)
        for causal in (True, False):
            out = run(q, k, v, causal, True)
            ok = torch.isfinite(out.float()).all().item()
            checks += 1
            bad += 0 if ok else 1
            print(f"  NaN past the tensors causal={int(causal)} "
                  f"{'bf16' if dt == torch.bfloat16 else 'fp32'}: {'OK' if ok else 'FAIL'}")

    print(f"{'FAIL' if bad else 'OK'}: {bad} failures in {checks} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
