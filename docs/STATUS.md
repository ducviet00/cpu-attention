# Where the campaign stands

This file is the honest, measured state. `docs/PROTOCOL.md` says how numbers are
taken; `docs/BENCHMARKS.md` holds the full tables; this is the summary and the
log of what actually moved.

## The one number

`ours / baseline < 1`, for both baselines, on both platforms, for all 22 rows of
`tests/bench_vs_refs.py`. Below 1 is a win.

## Verified tables, 2026-10-22

`scripts/verify.sh` on each box: clean rebuild from scratch, both parity suites
(`test_dense` 49/49, `test_flash_api`, and on x86 `test_amx` for both dtypes),
then the full 22-row matrix with the reference kernel in one process. Ratios are
`ours / baseline`; below 1 wins. Ours includes the K/V pack, vLLM's timed region
excludes theirs, which is the campaign's stated asymmetry.

### x86 — Xeon 6975P (m8i, 2 physical cores), bf16 via AMX, fp32 via 3-term AMX

| | beats SDPA | beats vLLM |
|---|---|---|
| **VEC reference** (bf16 + fp32) | **22 / 22** | **21 / 22** |
| AMX reference (bf16 only; vLLM's AMX path has no fp32) | **11 / 11** | 4 / 11 |

fp32 is 0.33-0.54 of SDPA and 0.54-1.18 of VEC; bf16 is 0.49-0.80 of SDPA and
0.21-0.36 of VEC. The single VEC loss is `128:8:64 fp32` at 1.18. Against
vLLM's **AMX** build the bf16 losses are all short/medium S (1.03-1.62) and all
of them are the pack: 10-25% of our call is K/V packing that their timed region
does not contain, and at S=128 the whole call is 60 us.

### ARM — Graviton 5 (m9g, 4 cores), NEON + `bfmmla`

| | beats SDPA | beats vLLM |
|---|---|---|
| NEON reference | **22 / 22** | 1 / 22 |

fp32 is 0.35-0.94 of SDPA and 0.94-1.18 of vLLM; bf16 0.50-0.88 of SDPA and
1.06-1.42 of vLLM. Twelve of the 22 rows are within 15% of the vLLM line and
the campaign's first row has crossed it.

### Where the remaining work is

1. **ARM bf16** (1.06-1.42) and **ARM fp32 at short S** (1.12-1.18). The ARM
   kernel just moved to vLLM's *structure* (L1 logits tile, one softmax per
   64-key tile, 8 rows per work item), which took bf16 from 1.39-2.35 to
   1.07-1.32 and fp32 from 1.19-1.60 to 0.96-1.19. Perf says we are now at
   vLLM's instruction count for bf16 (20.99 vs 20.88 per row-key) and behind on
   IPC (3.62 vs 4.07); the next measured items are the two fork-joins (18% of an
   S=128 call), the bf16 D=128 probability-panel loads, and fp32's `ldp`/address
   stalls.
2. **x86 bf16 at short/medium S against vLLM's AMX build** (1.03-1.62). The pack
   is memory-bound at 88% of memcpy, so this needs the kernel, not the pack.
3. **x86 `128:8:64 fp32`** (1.18 against VEC), the only VEC row left.

## The laptop (Ryzen 7840H, 8 cores, AVX-512 + avx512_bf16, no AMX), 2026-10-21
