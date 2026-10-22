"""Randomised parity fuzz for the dense path, against SDPA.

`tests/test_dense.py` covers the shapes the kernel specialises on, but it is a
fixed list, and the failure mode this file exists for is a *silent* wrong answer
on a shape nobody wrote down: the per-geometry call plan (`_kernel._plans`)
caches the kernel path, `qt`, the worker count and the scratch pointers, so a
stale or mismatched entry corrupts numerics without an exception, and the
weight-packing paths have per-dtype and per-head-dim layouts that a new shape
can cross in a way the fixed list does not.

Two more things it checks that a straight parity list does not: the same call
repeated (the plan cache is hit), and the same call interleaved with other
shapes and with `clear_scratch()` (the plan cache is invalidated and rebuilt).

    .venv/bin/python tests/fuzz_dense.py --cases 200
    .venv/bin/python tests/fuzz_dense.py --seed 7 --max-seq 2000

Tolerances are the project's: fp32 2e-4, bf16 3e-2 against SDPA. A case that
exceeds it is printed with its shape and seed so it can be reproduced.
"""

import argparse
import random
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import cpu_attention as attn  # noqa: E402
import cpu_attention._kernel as K  # noqa: E402


def reference(q, k, v, causal, scale):
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    heads, kv_heads = q.shape[2], k.shape[2]
    if kv_heads != heads:
        kk = kk.repeat_interleave(heads // kv_heads, 1)
        vv = vv.repeat_interleave(heads // kv_heads, 1)
    seq_q, seq_k = q.shape[1], k.shape[1]
    if causal and seq_q != seq_k:
        keep = torch.arange(seq_k)[None, :] <= (
            torch.arange(seq_q)[:, None] + (seq_k - seq_q)
        )
        return torch.nn.functional.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=keep, scale=scale
        ).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        qq, kk, vv, is_causal=causal, scale=scale
    ).transpose(1, 2)


def make_case(rng, max_seq, dtype):
    """One random (q, k, v, causal) triple, sometimes strided."""
    batch = rng.choice([1, 1, 1, 2, 3])
    heads = rng.choice([1, 2, 3, 4, 8])
    kv_heads = rng.choice([h for h in (1, 2, 4, 8) if heads % h == 0])
    dim = rng.choice([32, 64, 128])
    seq_q = rng.choice([1, 2, 3, 7, 8, 9, 16, 17, 31, 64, 65, 127, 128,
                        129, 255, 300, 511, 512, max_seq])
    seq_q = max(1, min(seq_q, max_seq))
    seq_k = rng.choice([seq_q, seq_q, max(1, seq_q // 3), seq_q + rng.randint(1, 40),
                        rng.randint(1, max_seq)])
    seq_k = max(1, min(seq_k, max_seq))
    kind = rng.choice(["plain", "plain", "plain", "qkv-slice", "bad-stride"])
    if kind == "qkv-slice":
        # A fused q/k/v projection: head dim contiguous, but the token stride is
        # 3 * dim, and the slice is not a contiguous tensor.
        q = torch.randn(batch, seq_q, heads, 3 * dim, dtype=dtype)[..., :dim]
        k = torch.randn(batch, seq_k, kv_heads, 3 * dim, dtype=dtype)[..., :dim]
        v = torch.randn(batch, seq_k, kv_heads, 3 * dim, dtype=dtype)[..., :dim]
    elif kind == "bad-stride":
        # Head dim not contiguous: exercises the `.contiguous()` copy.
        q = torch.randn(batch, seq_q, dim, heads, dtype=dtype).transpose(2, 3)
        k = torch.randn(batch, seq_k, dim, kv_heads, dtype=dtype).transpose(2, 3)
        v = torch.randn(batch, seq_k, dim, kv_heads, dtype=dtype).transpose(2, 3)
    else:
        q = torch.randn(batch, seq_q, heads, dim, dtype=dtype)
        k = torch.randn(batch, seq_k, kv_heads, dim, dtype=dtype)
        v = torch.randn(batch, seq_k, kv_heads, dim, dtype=dtype)
    causal = rng.random() < 0.7
    scale = dim ** -0.5 if rng.random() < 0.7 else rng.uniform(0.05, 0.5)
    return q, k, v, causal, scale, kind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    fails = 0
    worst = 0.0
    for i in range(args.cases):
        dtype = rng.choice([torch.float32, torch.bfloat16])
        tol = 2e-4 if dtype is torch.float32 else 3e-2
        q, k, v, causal, scale, kind = make_case(rng, args.max_seq, dtype)
        tag = (f"case {i:3d} {str(dtype)[6:]:8s} q={tuple(q.shape)} "
               f"k={tuple(k.shape)} causal={int(causal)} {kind}")
        try:
            want = reference(q, k, v, causal, scale)
            got = attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
            # hit the plan cache again, then force a rebuild
            got2 = attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
            K.clear_scratch()
            got3 = attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
        except Exception as exc:  # noqa: BLE001 - a raise here is a bug of its own
            print(f"[RAISE] {tag}: {type(exc).__name__}: {exc}")
            fails += 1
            continue
        d = max((got.float() - want.float()).abs().max().item(),
                (got2.float() - want.float()).abs().max().item(),
                (got3.float() - want.float()).abs().max().item())
        worst = max(worst, d / tol)
        ok = d <= tol
        if not ok:
            fails += 1
            print(f"[FAIL] {tag}: max|diff|={d:.2e} (tol {tol:.0e})")
        elif args.verbose:
            print(f"[OK]   {tag}: max|diff|={d:.2e}")
    print(f"fuzz dense: {args.cases - fails}/{args.cases} cases within tolerance, "
          f"worst was {worst:.2f}x its tolerance")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
