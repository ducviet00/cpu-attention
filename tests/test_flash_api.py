"""Tests for the flash-attn-shaped API, against an SDPA reference.

Run with:  PYTHONPATH=. python tests/test_flash_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cpu_attention as attn


def main():
    if not attn.is_available():
        # A gate that cannot run has not passed.
        print(
            "FAIL: the selected backend is not built (uv run --extra build python scripts/build.py)"
        )
        return 1

    fails = flash_api()
    try:
        attn.flash_attn_varlen_func(None, None, None, None, None, 0, 0)
    except NotImplementedError:
        print("[OK] flash_api rejects flash_attn_varlen_func")
    else:
        fails += 1
        print("[MISMATCH] flash_api accepted flash_attn_varlen_func")
    if fails:
        raise SystemExit(f"{fails} mismatches")
    print(f"cpu_attention {attn.__version__}: all checks passed")


def flash_api():
    """The flash-attn-shaped entry points: semantics, GQA, cache writes, rejections."""
    torch.manual_seed(7)
    D, H, BLOCK, BLOCKS, B, T = 32, 4, 16, 4, 3, 12
    KVH = 2
    scale = D**-0.5
    fails = 0

    def check(tag, have, want, tol=2e-3):
        nonlocal fails
        diff = (have.float() - want.float()).abs().max().item()
        ok = diff <= tol
        fails += not ok
        print(f"[{'OK' if ok else 'MISMATCH'}] flash_api {tag}: max|diff|={diff:.2e}")

    def rejects(tag, fn):
        nonlocal fails
        try:
            fn()
        except (NotImplementedError, ValueError):
            print(f"[OK] flash_api rejects {tag}")
            return
        fails += 1
        print(f"[MISMATCH] flash_api accepted {tag}")

    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
        ).transpose(1, 2)

    def caches(heads=H):
        pages = torch.arange(B * BLOCKS, dtype=torch.int32).reshape(B, BLOCKS)
        shape = (B * BLOCKS, heads, BLOCK, D)
        return torch.zeros(shape), torch.zeros(shape), pages

    q = torch.randn(B, 1, H, D)
    keys = torch.randn(B, T, H, D)
    vals = torch.randn(B, T, H, D)

    # 1. Incremental write+attend with flash's pre-append `cache_seqlens`.
    k_cache, v_cache, page_table = caches()
    for t in range(T):
        out = attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=keys[:, t : t + 1],
            v=vals[:, t : t + 1],
            cache_seqlens=t,
            page_table=page_table,
            causal=True,
            softmax_scale=scale,
        )
        check(f"write+attend t={t}", out, sdpa(q, keys[:, : t + 1], vals[:, : t + 1]))

    # 2. The `slot_mapping` fast path matches the pre-append path exactly.
    ka, va, pages = caches()
    kb, vb, _ = caches()
    out_a = attn.flash_attn_with_kvcache(
        q,
        ka,
        va,
        k=keys[:, 5:6],
        v=vals[:, 5:6],
        cache_seqlens=5,
        page_table=pages,
        causal=True,
        softmax_scale=scale,
    )
    slots = pages[:, 5 // BLOCK].long() * BLOCK + 5 % BLOCK
    out_b = attn.flash_attn_with_kvcache(
        q,
        kb,
        vb,
        k=keys[:, 5:6],
        v=vals[:, 5:6],
        cache_seqlens=6,
        slot_mapping=slots,
        page_table=pages,
        causal=True,
        softmax_scale=scale,
    )
    check("slot_mapping output", out_b, out_a)
    check("slot_mapping cache", kb, ka)

    # 3. Cross-attention over a staged cache, per-sequence int32 lengths.
    lens = torch.tensor([3, 7, 12], dtype=torch.int32)
    kc, vc, pages = caches()
    for b in range(B):
        n = int(lens[b])
        pos = torch.arange(n)
        slot = pages[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        attn.reshape_and_cache(keys[b, :n], vals[b, :n], kc, vc, slot)
    ref = torch.cat(
        [
            sdpa(
                q[b : b + 1],
                keys[b : b + 1, : int(lens[b])],
                vals[b : b + 1, : int(lens[b])],
            )
            for b in range(B)
        ]
    )
    out = attn.flash_attn_with_kvcache(
        q,
        kc,
        vc,
        cache_seqlens=lens,
        page_table=pages,
        causal=False,
        softmax_scale=scale,
    )
    check("cross, tensor lengths", out, ref)

    # 4. `cache_batch_idx` gathers the page-table rows (flash's remap semantics).
    #    Every sequence is filled to T so that a remapped row never reads slots
    #    that were never written.
    perm = torch.tensor([2, 0, 1], dtype=torch.int32)
    kcp, vcp, pagesp = caches()
    pos = torch.arange(T)
    for b in range(B):
        slot = pagesp[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        attn.reshape_and_cache(keys[b], vals[b], kcp, vcp, slot)
    out = attn.flash_attn_with_kvcache(
        q,
        kcp,
        vcp,
        cache_seqlens=lens,
        page_table=pagesp,
        cache_batch_idx=perm,
        softmax_scale=scale,
    )
    want = torch.cat(
        [
            sdpa(
                q[b : b + 1],
                keys[int(perm[b]) : int(perm[b]) + 1, : int(lens[b])],
                vals[int(perm[b]) : int(perm[b]) + 1, : int(lens[b])],
            )
            for b in range(B)
        ]
    )
    check("cache_batch_idx", out, want)

    # 5. GQA: 4 query heads over 2 KV heads.
    kg = torch.randn(B, 6, KVH, D)
    vg = torch.randn(B, 6, KVH, D)
    kq = torch.randn(B, 1, H, D)
    kcg, vcg, pagesg = caches(KVH)
    pos = torch.arange(6)
    for b in range(B):
        slot = pagesg[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        attn.reshape_and_cache(kg[b], vg[b], kcg, vcg, slot)
    out = attn.flash_attn_with_kvcache(
        kq, kcg, vcg, cache_seqlens=6, page_table=pagesg, softmax_scale=scale
    )
    check(
        "gqa",
        out,
        sdpa(
            kq,
            kg.repeat_interleave(H // KVH, dim=2),
            vg.repeat_interleave(H // KVH, dim=2),
        ),
    )

    # 6. `flash_attn_func` works for the single query token (kv_len 9, not a page multiple).
    dk = torch.randn(B, 9, H, D)
    dv = torch.randn(B, 9, H, D)
    check(
        "flash_attn_func L=1",
        attn.flash_attn_func(q, dk, dv, softmax_scale=scale),
        sdpa(q, dk, dv),
    )

    # 7. Engine aliases are the same call.
    kc1, vc1, pages1 = caches()
    kc2, vc2, _ = caches()
    ones = torch.ones(B, dtype=torch.int32)
    alias = attn.self_attention(
        q.squeeze(1),
        keys[:, 0],
        vals[:, 0],
        kc1,
        vc1,
        pages1[:, 0].long() * BLOCK,
        pages1,
        ones,
        scale,
    )
    direct = attn.flash_attn_with_kvcache(
        q,
        kc2,
        vc2,
        k=keys[:, 0:1],
        v=vals[:, 0:1],
        cache_seqlens=1,
        slot_mapping=pages1[:, 0].long() * BLOCK,
        page_table=pages1,
        causal=True,
        softmax_scale=scale,
    ).squeeze(1)
    check("self_attention alias", alias, direct)
    check(
        "cross_attention alias",
        attn.cross_attention(q.squeeze(1), kc, vc, pages, lens, scale),
        ref.squeeze(1),
    )

    # 8. Unsupported flash-attn features fail loudly, not silently.
    kc, vc, pages = caches()
    fresh = {
        "k": keys[:, :1],
        "v": vals[:, :1],
        "cache_seqlens": 0,
        "page_table": pages,
    }
    rejects(
        "seqlen=2",
        lambda: attn.flash_attn_with_kvcache(q.repeat(1, 2, 1, 1), kc, vc, **fresh),
    )
    rejects(
        "two new tokens",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, k=keys[:, :2], v=vals[:, :2], cache_seqlens=0, page_table=pages
        ),
    )
    rejects(
        "window_size",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, window_size=(4, 0)
        ),
    )
    rejects(
        "softcap",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, softcap=1.0
        ),
    )
    rejects(
        "rotary_cos",
        lambda: attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            rotary_cos=torch.zeros(4, 8),
        ),
    )
    rejects(
        "qv",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, qv=q
        ),
    )
    rejects(
        "cache_leftpad",
        lambda: attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            cache_leftpad=torch.zeros(B, dtype=torch.int32),
        ),
    )
    rejects(
        "return_softmax_lse",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, return_softmax_lse=True
        ),
    )
    rejects(
        "page_table+block_table",
        lambda: attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, block_table=pages
        ),
    )
    rejects(
        "slot_mapping without k/v",
        lambda: attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            slot_mapping=torch.zeros(B, dtype=torch.int64),
        ),
    )
    rejects(
        "dense head_dim=96",
        lambda: attn.flash_attn_func(
            torch.zeros(B, 8, H, 96),
            torch.zeros(B, 8, H, 96),
            torch.zeros(B, 8, H, 96),
        ),
    )
    rejects(
        "dense nheads not divisible by nheads_k",
        lambda: attn.flash_attn_func(
            torch.zeros(B, 8, 6, 64),
            torch.zeros(B, 8, 4, 64),
            torch.zeros(B, 8, 4, 64),
        ),
    )
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
