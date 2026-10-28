"""AMX dense kernel vs torch SDPA and vs the Mojo kernel: parity + timing.

    .venv/bin/python tests/test_amx.py            # parity on a shape grid
    .venv/bin/python tests/test_amx.py --bench    # + interleaved timing

Parity tolerance is the same as `tests/test_dense.py` for bf16 (3e-2 max abs
diff on the output), because the AMX kernel does the same arithmetic: bf16
inputs, fp32 accumulate for Q@K^T, bf16 probabilities, fp32 P@V accumulate.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402


def sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=q.shape[-1] ** -0.5,
    )


def run(q, k, v, causal, amx):
    """One variant, forced to the AMX or the Mojo path."""
    prev = K._AMX_ENV
    K._AMX_ENV = amx
    try:
        return attn.flash_attn_func(q, k, v, causal=causal)
    finally:
        K._AMX_ENV = prev


def run_entry(q, k, v, causal, plan):
    """One AMX call through the 5-argument plan entry or the 36-argument one.

    The plan buffer carries ~30 geometry fields that used to cross the boundary
    as Python arguments; a wrong slot in it is silent (the kernel reads a
    plausible integer and writes the wrong place), so both entries must agree
    bit for bit.
    """
    prev = K._AMX_PLAN
    K._AMX_PLAN = K._AMX_PLAN if plan else None
    K.clear_scratch()
    try:
        return attn.flash_attn_func(q, k, v, causal=causal)
    finally:
        K._AMX_PLAN = prev
        K.clear_scratch()


def bench(fns, rounds=5):
    for f in fns.values():
        for _ in range(2):
            f()
    s = {n: [] for n in fns}
    for _ in range(rounds):
        for n, f in fns.items():
            t0 = time.perf_counter()
            f()
            s[n].append((time.perf_counter() - t0) * 1e6)
    return {n: statistics.median(v) for n, v in s.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "both"])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shapes", default="")
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

    shapes = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
              (256, 16, 128), (1024, 16, 128), (512, 32, 128), (1024, 8, 32)]
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]

    dts = {"bf16": [torch.bfloat16], "fp32": [torch.float32],
           "both": [torch.bfloat16, torch.float32]}[args.dtype]
    bad = 0
    for seq, heads, dim in shapes:
      for dt in dts:
        for causal in (True, False):
            q = torch.randn(1, seq, heads, dim, dtype=dt)
            k = torch.randn(1, seq, heads, dim, dtype=dt)
            v = torch.randn(1, seq, heads, dim, dtype=dt)
            ref = sdpa(q, k, v, causal).transpose(1, 2).contiguous()
            got = run_entry(q, k, v, causal, True)
            alt = run_entry(q, k, v, causal, False)
            mojo = run(q, k, v, causal, False)
            d_amx = (got.float() - ref.float()).abs().max().item()
            d_mojo = (mojo.float() - ref.float()).abs().max().item()
            d_plan = (got.float() - alt.float()).abs().max().item()
            ok = d_amx <= (3e-2 if dt == torch.bfloat16 else 2e-4)
            ok = ok and d_plan == 0.0
            bad += 0 if ok else 1
            tag = "bf16" if dt == torch.bfloat16 else "fp32"
            print(f"  S={seq:5d} H={heads:2d} D={dim:3d} {tag} causal={int(causal)}: "
                  f"amx max|d|={d_amx:.2e} mojo={d_mojo:.2e} "
                  f"plan-vs-36arg={d_plan:.1e} {'OK' if ok else 'FAIL'}")
            if not ok:
                # where does it go wrong?
                diff = (got.float() - ref.float()).abs()
                idx = diff.argmax()
                print("    worst at flat", int(idx), "of", diff.numel(),
                      " row frac", (int(idx) // dim) % seq)
        if args.bench:
            q = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
            k = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
            v = torch.randn(1, seq, heads, dim, dtype=torch.bfloat16)
            m = bench({
                "amx": lambda: run(q, k, v, True, True),
                "mojo": lambda: run(q, k, v, True, False),
                "sdpa": lambda: sdpa(q, k, v, True),
            }, args.rounds)
            f = 4.0 * heads * seq * seq * dim * 0.5
            print(f"  S={seq:5d} H={heads:2d} D={dim:3d} us: "
                  + " ".join(f"{n}={t:8.0f}" for n, t in m.items())
                  + f"   amx/sdpa {m['amx']/m['sdpa']:4.2f}"
                  f"  speedup vs mojo {m['mojo']/m['amx']:4.2f}x"
                  f"  [{f/m['amx']*1e-6/1e9*1e6:4.0f} GFLOP/s]")
    print(f"{'FAIL' if bad else 'OK'}: {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
