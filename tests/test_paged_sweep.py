"""Paged parity sweep: shapes, dtypes, page sizes, GQA/MQA, strided inputs.

Covers the axes the cache layouts and the store/attend path can break:
head_dim 32/64/128, page 16/32/48/64, context 1..2049, GQA ratios 1/2/4,
a block table whose pages are not in cache order, and a query / new-K/V that
are strided chunks of a fused projection (what the engine actually passes).

    CPU_ATTENTION_BACKEND=cpp python tests/test_paged_sweep.py
    CPU_ATTENTION_BACKEND=mojo python tests/test_paged_sweep.py
"""

import itertools
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import reference as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402


def strided_chunk(t, proj_width):
    """`t` [b, n, proj_width] sliced out of a 3*proj_width-wide row.

    That is the shape the engine hands over: chunks of one fused qkv buffer, so
    the token stride is 3*proj_width and not the tensor width.
    """
    buf = torch.zeros(t.shape[0], 1, proj_width * 3, dtype=t.dtype)
    buf[:, :, :proj_width] = t.reshape(t.shape[0], 1, proj_width)
    return buf[:, :, :proj_width].view(t.shape[0], 1, proj_width)


def run_case(batch, ctx, heads, kvh, hd, page, dtype, strided, seed,
             reference_dtype=torch.float32):
    # Both backends must see the same tensors, so the data comes from a seed
    # rather than from the global generator.
    torch.manual_seed(seed)
    group = heads // kvh
    pages = (ctx + page - 1) // page
    shape = (batch * pages, kvh, page, hd)
    kc = torch.zeros(shape, dtype=dtype)
    vc = torch.zeros(shape, dtype=dtype)
    k_all = torch.randn(batch, ctx, kvh, hd, dtype=dtype)
    v_all = torch.randn(batch, ctx, kvh, hd, dtype=dtype)
    # A block table that is a permutation: cache order != sequence order.
    bt = torch.randperm(batch * pages).to(torch.int32).reshape(batch, pages)

    # Stage the first ctx-1 tokens through reshape_and_cache, scattered.
    if ctx > 1:
        kpre = k_all[:, :ctx - 1].reshape(batch * (ctx - 1), kvh, hd)
        vpre = v_all[:, :ctx - 1].reshape(batch * (ctx - 1), kvh, hd)
        b = torch.arange(batch).repeat_interleave(ctx - 1)
        t = torch.arange(ctx - 1).repeat(batch)
        slot = torch.empty(batch * (ctx - 1), dtype=torch.int64)
        for s in range(batch):
            m = b == s
            slot[m] = bt[s, t[m] // page].to(torch.int64) * page + t[m] % page
        attn.reshape_and_cache(kpre, vpre, kc, vc, slot)

    # The query and the new token, optionally strided like a fused qkv chunk.
    q = torch.randn(batch, heads, hd, dtype=dtype)
    kn = k_all[:, ctx - 1].reshape(batch, kvh, hd)
    vn = v_all[:, ctx - 1].reshape(batch, kvh, hd)
    if strided:
        q = strided_chunk(q, heads * hd).view(batch, heads, hd)
        kn = strided_chunk(kn, kvh * hd).view(batch, kvh, hd)
        vn = strided_chunk(vn, kvh * hd).view(batch, kvh, hd)
    slot_new = torch.empty(batch, dtype=torch.int64)
    for s in range(batch):
        pg = (ctx - 1) // page
        slot_new[s] = int(bt[s, pg]) * page + (ctx - 1) % page
    lens = torch.full((batch,), ctx, dtype=torch.int32)
    scale = hd ** -0.5

    got = attn.self_attention(q, kn, vn, kc, vc, slot_new, bt, lens, scale)
    # Cross-attention over the same cache must agree (no second store).
    got_x = attn.cross_attention(q, kc, vc, bt, lens, scale)

    kk = k_all.transpose(1, 2)
    vv = v_all.transpose(1, 2)
    if group > 1:
        kk = kk.repeat_interleave(group, dim=1)
        vv = vv.repeat_interleave(group, dim=1)
    ref = C.ref_decode(q.reshape(batch, heads, hd), kk, vv, dtype=reference_dtype)
    return got, got_x, ref


def main():
    # No options: the sweep is the point.
    torch.manual_seed(0)
    torch.set_num_threads(8)
    print(f"backend {_kernel.backend()}", flush=True)
    cases = bad = 0
    reference_checks = 0
    worst = {torch.float32: 0.0, torch.bfloat16: 0.0}
    worst_at = {}
    # bf16 is judged against the other backend in bf16 ulps (the shipped x86
    # kernel uses a degree-2 exponential that is inside a bf16 store, so an
    # fp32-accurate reference legitimately disagrees by ~4e-3); fp32 against the
    # naive fp32 reference in absolute terms.
    other = "mojo" if _kernel.backend() == "cpp" else "cpp"
    shapes = [(1, 1, False), (17, 1, True), (33, 3, False), (256, 2, True),
              (257, 1, False), (1024, 1, True), (2049, 2, False)]
    for hd, page, (heads, kvh) in itertools.product(
        (32, 64, 128), (16, 32, 48, 64), ((16, 16), (16, 8), (16, 4))
    ):
        for ctx, batch, strided in shapes:
            for dtype in (torch.float32, torch.bfloat16):
                # Python hashes dtype objects by address, so the old seed changed
                # with ASLR. A failing case must reproduce in a fresh process.
                seed = (batch * 3 + ctx * 5 + heads * 7 + kvh * 11 +
                        hd * 13 + page * 17 + int(dtype == torch.bfloat16))
                got, got_x, ref = run_case(batch, ctx, heads, kvh, hd, page,
                                           dtype, strided, seed)
                # Same case and the same tensors through the other backend.
                _kernel.set_backend(other)
                got_o, got_xo, _ = run_case(batch, ctx, heads, kvh, hd, page,
                                            dtype, strided, seed)
                _kernel.set_backend("cpp" if other == "mojo" else "mojo")
                if dtype == torch.float32:
                    err = max((got.float() - ref).abs().max().item(),
                              (got_x.float() - ref).abs().max().item())
                    tol = 1e-6
                    if err > tol:
                        # A sequential fp32 dot in the reference can itself lose
                        # more than the gate (seen on Axion at D=128/33 keys).
                        # Keep that comparison, but resolve it against fp64,
                        # never by widening the kernel's absolute tolerance.
                        _, _, truth = run_case(batch, ctx, heads, kvh, hd, page,
                                               dtype, strided, seed,
                                               reference_dtype=torch.float64)
                        ref_err = (ref.double() - truth).abs().max().item()
                        if ref_err > tol:
                            raw_err = err
                            err = max((got.double() - truth).abs().max().item(),
                                      (got_x.double() - truth).abs().max().item())
                            reference_checks += 1
                            print(f"[reference precision] hd={hd} page={page} "
                                  f"ctx={ctx}: kernel/fp32={raw_err:.3e}, "
                                  f"fp32/fp64={ref_err:.3e}, "
                                  f"kernel/fp64={err:.3e}", flush=True)
                else:
                    # bfs ulps at the row's peak output: one ulp of a bf16 value
                    # near `peak` is `peak * 2^-8`, which is the scale the
                    # scorecard quotes ("0-2 ulp at max outputs"). An
                    # element-wise ulp would count a cancellation-level value of
                    # 1e-4 as hundreds of ulps for the same 1e-3 error.
                    peak = ref.abs().max().clamp_min(1e-30) * 2 ** -8
                    err = max((got.float() - got_o.float()).abs().max().item(),
                              (got_x.float() - got_xo.float()).abs().max().item())
                    err /= peak.item()
                    tol = 2.0
                if err > worst[dtype]:
                    worst[dtype] = err
                    worst_at[dtype] = (hd, page, heads, kvh, ctx, batch, strided)
                cases += 1
                if not (err <= tol):
                    bad += 1
                    print(f"FAIL hd={hd} page={page} h={heads}/{kvh} ctx={ctx} "
                          f"b={batch} strided={strided} "
                          f"{'fp32' if dtype == torch.float32 else 'bf16'}: "
                           f"{err:.3e} > {'1e-6' if dtype == torch.float32 else '2 ulp'}",
                          flush=True)
    print(f"paged sweep: {cases - bad}/{cases} within tolerance")
    print(f"  fp64-resolved reference precision cases: {reference_checks}")
    print(f"  worst fp32 {worst[torch.float32]:.2e} at {worst_at.get(torch.float32)}")
    print(f"  worst bf16 {worst[torch.bfloat16]:.2f} ulp at "
          f"{worst_at.get(torch.bfloat16)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
