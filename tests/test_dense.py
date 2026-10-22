"""Dense attention (``flash_attn_func``) against torch SDPA.

Covers the shapes the kernel specialises on (head_dim 32/64/128, fp32/bf16),
MHA and GQA, causal and non-causal, ``seqlen_q != seqlen_k`` (the bottom-right
causal alignment), a query block that is not a multiple of ``qt``, and rows whose
causal mask is empty (all-zero output, per flash-attn's contract).

    uv run python tests/test_dense.py
"""

import torch

import cpu_attention as attn

# (batch, seqlen_q, seqlen_k, nheads, nheads_k, head_dim)
SHAPES = [
    (1, 128, 128, 8, 8, 64),
    (1, 1, 9, 4, 4, 64),  # one query token, unaligned KV length
    (1, 200, 200, 6, 2, 128),  # GQA, head_dim 128
    (1, 64, 96, 8, 8, 32),  # seqlen_q < seqlen_k, head_dim 32
    (1, 37, 37, 3, 3, 128),  # not a multiple of qt
    (2, 5, 2, 2, 2, 64),  # seqlen_q > seqlen_k: leading rows fully masked
    (1, 300, 300, 8, 8, 64),  # several 64-key tiles
    (1, 17, 301, 4, 4, 64),  # short query against a long cache
    (1, 257, 257, 8, 2, 64),  # GQA + odd length
    # head_dim 128 with seq_k >= 1024: 8 rows x 128 dims is the only shape that
    # leaves the register kernel for the L1-accumulator one (`_dense_rows_mem`),
    # and 1088 = 17 x 64 exercises a partial last K^T block.
    (1, 1088, 1088, 2, 2, 128),
    (1, 1100, 1030, 2, 1, 128),  # GQA, seq_q != seq_k, both non-multiples
]


def _reference(q, k, v, causal, scale):
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


def main() -> int:
    torch.manual_seed(0)
    fails = 0
    checks = 0

    def check(name, got, want, tol):
        nonlocal fails, checks
        checks += 1
        err = (got.float() - want.float()).abs().max().item()
        ok = err <= tol
        if not ok:
            fails += 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: max|diff|={err:.3e} (tol {tol:.0e})")

    for dtype in (torch.float32, torch.bfloat16):
        # bf16 rounds the probabilities-free path only; 1 ulp of the output.
        tol = 2e-4 if dtype is torch.float32 else 3e-2
        for batch, seq_q, seq_k, heads, kv_heads, dim in SHAPES:
            q = torch.randn(batch, seq_q, heads, dim, dtype=dtype)
            k = torch.randn(batch, seq_k, kv_heads, dim, dtype=dtype)
            v = torch.randn(batch, seq_k, kv_heads, dim, dtype=dtype)
            for causal in (True, False):
                out = attn.flash_attn_func(q, k, v, causal=causal)
                want = _reference(q, k, v, causal, dim**-0.5)
                tag = "bf16" if dtype is torch.bfloat16 else "fp32"
                check(
                    f"{tag} B={batch} Sq={seq_q} Sk={seq_k} H={heads}/{kv_heads} "
                    f"D={dim} causal={int(causal)}",
                    out,
                    want,
                    tol,
                )

    # Explicit softmax_scale, and a strided (non-contiguous) head dim.
    q = torch.randn(1, 64, 4, 64, dtype=torch.float32)
    k = torch.randn(1, 64, 4, 64, dtype=torch.float32)
    v = torch.randn(1, 64, 4, 64, dtype=torch.float32)
    check(
        "fp32 explicit softmax_scale",
        attn.flash_attn_func(q, k, v, softmax_scale=0.3, causal=True),
        _reference(q, k, v, True, 0.3),
        2e-4,
    )
    qkv = torch.randn(1, 64, 4, 3 * 64)
    qn, kn, vn = qkv[..., :64], qkv[..., 64:128], qkv[..., 128:]
    check(
        "fp32 strided q/k/v (fused qkv slice)",
        attn.flash_attn_func(qn, kn, vn, causal=True),
        _reference(qn, kn, vn, True, 64**-0.5),
        2e-4,
    )

    # A fully masked row must be zero, not NaN (flash-attn's contract).
    q = torch.randn(1, 4, 2, 32)
    k = torch.randn(1, 2, 2, 32)
    v = torch.randn(1, 2, 2, 32)
    out = attn.flash_attn_func(q, k, v, causal=True)
    leading = out[0, :2]
    checks += 1
    if torch.count_nonzero(leading) == 0:
        print("[OK] fully masked rows are zero")
    else:
        fails += 1
        print(f"[FAIL] fully masked rows are zero: got {leading}")

    # --- the per-geometry call plan (`_kernel._plans`) ---------------------
    # A plan caches the kernel path, `qt`, the worker count and the *scratch
    # buffer pointers* for a geometry, so the two ways it can go wrong are a
    # stale entry (same geometry, reallocated buffers) and a plan reused for a
    # different geometry. Both are silent numeric corruption, so they are tested
    # rather than argued about.
    import cpu_attention._kernel as K

    for dtype, tol in ((torch.float32, 2e-4), (torch.bfloat16, 3e-2)):
        a = [torch.randn(1, 300, 4, 64, dtype=dtype) for _ in range(3)]
        b = [torch.randn(1, 130, 4, 64, dtype=dtype) for _ in range(3)]
        want_a = _reference(*a, True, 0.125)
        want_b = _reference(*b, True, 0.125)
        tag = "bf16" if dtype is torch.bfloat16 else "fp32"
        # interleaved geometries, twice, so the second round hits the cache
        for round_ in range(2):
            check(f"{tag} plan A round {round_}",
                  attn.flash_attn_func(*a, causal=True), want_a, tol)
            check(f"{tag} plan B round {round_}",
                  attn.flash_attn_func(*b, causal=True), want_b, tol)
        # fresh tensors with the same geometry (new data pointers)
        a2 = [t.clone() for t in a]
        check(f"{tag} plan reused with new pointers",
              attn.flash_attn_func(*a2, causal=True),
              _reference(*a2, True, 0.125), tol)
        # causal is part of the key: same geometry, other mask
        check(f"{tag} plan causal=False after causal=True",
              attn.flash_attn_func(*a, causal=False),
              _reference(*a, False, 0.125), tol)
        # dropping the workspace must drop the plans with it
        K.clear_scratch()
        check(f"{tag} plan rebuilt after clear_scratch",
              attn.flash_attn_func(*a, causal=True), want_a, tol)

    # Two threads with the same geometry: the plans and the buffers are
    # per-thread, so neither may scribble on the other's K/V pack.
    import threading

    errs = []

    def _worker(seed):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 256, 4, 64, generator=g)
        k = torch.randn(1, 256, 4, 64, generator=g)
        v = torch.randn(1, 256, 4, 64, generator=g)
        want = _reference(q, k, v, True, 0.125)
        for _ in range(20):
            got = attn.flash_attn_func(q, k, v, causal=True)
            errs.append((got - want).abs().max().item())

    threads = [threading.Thread(target=_worker, args=(s,)) for s in (1, 2, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    checks += 1
    if errs and max(errs) <= 2e-4:
        print(f"[OK] threads share no plan state: max|diff|={max(errs):.1e}")
    else:
        fails += 1
        print(f"[FAIL] threads share no plan state: max|diff|={max(errs):.1e}")

    # Rejections the kernel really cannot do.
    for name, fn in (        (
            "window_size is gone from the signature",
            lambda: attn.flash_attn_func(q, k, v, window_size=(-1, -1)),
        ),
        (
            "varlen is not ported",
            lambda: attn.flash_attn_varlen_func(q, k, v, None, None, 4, 2),
        ),
    ):
        checks += 1
        try:
            fn()
        except TypeError:
            print(f"[OK] {name}")
        except NotImplementedError:
            print(f"[OK] {name}")
        else:
            fails += 1
            print(f"[FAIL] {name}")

    # The extension has two entries for the fused dense call: the 42-argument
    # `dense_forward` and the 7-argument `dense_forward_plan`, which reads the
    # geometry out of an int64 buffer the call plan owns. They must agree bit for
    # bit -- a wrong index in that buffer is silent in the shapes it happens to
    # get right and a segfault in the ones it does not, which is exactly how the
    # first cut of it failed.
    import cpu_attention._kernel as K

    abi_bad = 0
    abi_cases = 0
    for dt in (torch.bfloat16, torch.float32):
        for (s_, h_, d_) in ((64, 4, 64), (130, 8, 64), (192, 16, 128),
                             (250, 16, 128), (512, 8, 64), (512, 32, 128)):
            for causal in (True, False):
                qq = torch.randn(1, s_, h_, d_, dtype=dt)
                kk = torch.randn_like(qq)
                vv = torch.randn_like(qq)
                K._PLAN_ABI = True
                aa = attn.flash_attn_func(qq, kk, vv, causal=causal)
                K._PLAN_ABI = False
                bb = attn.flash_attn_func(qq, kk, vv, causal=causal)
                K._PLAN_ABI = True
                abi_cases += 1
                if (aa.float() - bb.float()).abs().max().item() != 0.0:
                    abi_bad += 1
    checks += 1
    if abi_bad:
        fails += 1
        print(f"[FAIL] plan-buffer entry differs from the 42-argument entry "
              f"on {abi_bad}/{abi_cases} cases")
    else:
        print(f"[OK] plan-buffer entry is bit-identical on {abi_cases} cases")

    print(f"cpu_attention dense: {checks - fails}/{checks} checks passed")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
