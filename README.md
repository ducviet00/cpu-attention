# cpu-attention

Paged and dense attention kernels for CPUs, with flash-attn's API. One source
tree builds for x86-64 (AVX-512, AMX) and aarch64 (NEON + BF16). It is the
attention backend of [fast-encdec](https://github.com/ducviet00/fast-encdec)
and works as a standalone library.

* `flash_attn_with_kvcache`: paged decode (self- and cross-attention), one
  query token per sequence.
* `flash_attn_func`: dense attention over whole prompts, causal (bottom-right
  aligned) or not.
* fp32 and bf16, MHA / MQA / GQA, head_dim 32/64/128, page size a multiple of
  16.
* One kernel, written in Mojo (`cpu_attention/_mojo/`,
  [`docs/Mojo.md`](docs/Mojo.md)): one driver per op for every CPU, a target
  differing only by compile-time constants and a trait-plugged inner kernel.
* Linux, PyTorch, Python >= 3.10.
* Not supported, and rejected with `NotImplementedError` rather than silently
  ignored: rotary, ALiBi, softcap, sliding windows, varlen/packed QKV,
  dropout, FP8, backward.

## Install and build

```bash
uv sync --extra build                                    # torch + the mojo wheel
uv run --extra build python scripts/build.py --tests     # one _C_<tier>.so per ISA tier
```

The build writes one shared object per ISA tier for the host architecture
(x86-64: AVX-512-BF16 + AMX, AVX2, baseline; aarch64: BF16, NEON), each compiled
for an explicit target rather than the build host. The loader picks the best
tier the CPU reports; `CPU_ATTENTION_ISA=amx|vec|avx2|scalar|neon` forces one
and is refused if the CPU lacks it.

## Use

```python
from cpu_attention import flash_attn_func, flash_attn_with_kvcache

# Dense: q/k/v [batch, seq, heads, head_dim]
out = flash_attn_func(q, k, v, causal=True)

# Paged decode: one query token per sequence
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,      # q [B, 1, H, D]; caches [blocks, kv_heads, page, D]
    k=k_new, v=v_new,         # written at cache_seqlens (the length before the write)
    cache_seqlens=lens,       # int32 [B]
    page_table=pages,         # int32 [B, max_blocks]
)
```

An engine's per-layer path should call the thin aliases. They skip the
argument checks the runner already did, and accept a preallocated `out=`.

```python
from cpu_attention import self_attention, cross_attention, reshape_and_cache

out = self_attention(q, k, v, k_cache, v_cache, slot_mapping, block_table, context_lens, scale)
out = cross_attention(q, k_cache, v_cache, block_table, cache_seqlens, scale)
reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)   # stage a cache
```

Notes on the API:
* **Cache layout.** The cache is this library's head-major layout (K is stored
  transposed inside the page). Stage a cache through `reshape_and_cache`; a
  cache in another library's layout (vLLM's included) is not readable.
* **Strides.** q/k/v may be strided chunks of a fused projection: any token
  stride, with heads and dims contiguous.
* **`causal` on the paged path** is accepted and has no effect, because one
  query at the last position sees every key.

## Performance

The kernel is measured against **torch SDPA** (`F.scaled_dot_product_attention`)
on every `flash_attn_func` row, and against **vLLM's CPU attention** (`csrc/cpu`
at `db3b0f0`, built by `scripts/build_vllm_ref.sh`) on every paged row. Times are
microseconds per call (the minimum over interleaved samples), and the ratio is
**ours / reference**: below 1 means we are faster. These numbers are the single
source of truth; `docs/Mojo.md` explains the design and the experiments.

**Setup** (Mojo 1.2.0 `2016d0cd`, torch 2.14.1+cpu, commit `453760b`, Oct 2026).
Three 8-vCPU GCP spot VMs, threads = physical cores:

| box | CPU | threads | vLLM ISA |
|---|---|---:|---|
| C4D | c4d-standard-8: AMD EPYC 9B45 (Zen 5), 4 cores / 8 threads, AVX-512 + BF16, no AMX | 4 | VEC |
| Axion | c4a-standard-8: Google Axion Neoverse-V2, 8 cores | 8 | NEON |
| Xeon | c4-standard-8: Intel Xeon Platinum 8581C, 4 cores / 8 threads, AMX | 4 | AMX for bf16, VEC for fp32 |

**Method** (`benchmarks/bench.py`; its docstring has the full rules).
* **The manifest** (`benchmarks/manifest.py`, fixed before optimizing) has 170
  rows, 85 shapes in both bf16 and fp32: odd lengths around tile and page
  boundaries (17 ... 4097), unequal query/key lengths, noncausal, GQA/MQA,
  head_dim 32/64/128, fused-QKV strided inputs, page sizes 16/48/64,
  heterogeneous contexts, store-only, a preallocated `out=` and the public
  `flash_attn_with_kvcache` call. Shapes are `seq_q:heads:head_dim` for flash
  rows (`sq/sk` when they differ, `h=Q/KV` heads for GQA/MQA).
* **One process per box**, with our build loaded twice independently (the second
  copy is the noise floor), SDPA and vLLM: interleaved, order rotated every
  round, and the **minimum** decides. Rounds: 61 for `flash_attn_func` and 121
  for decode, one call per sample. Every arm's output is validated.
* **Equal work.** Ours includes the K/V pack and the output allocation (except
  `out=` rows). For a decode row, vLLM does the same append and paged decode on
  its own cache, staged outside the timer, and its output is preallocated. For a
  store row, both run their `reshape_and_cache`.
* **Cold decode rows.** A decode row whose per-arm K/V working set exceeds a
  quarter of the LLC is timed cold (marked ᶜ): before every sample, outside the
  timer and identically for every arm, a read-only sweep of twice the LLC evicts
  it. A serving KV cache is far larger than the LLC.
* **Verdict per row.** With `band = max(3%, spread between our two copies)`, a
  row is a **win** below `1 - band`, a **loss** above `1 + band` (marked ▲), and
  **unresolved** in between.

**Results** (win / unresolved / loss):

| rows | reference | C4D | Axion | Xeon |
|---|---|---|---|---|
| `flash_attn_func` (78) | SDPA | 78 / 0 / 0 | 78 / 0 / 0 | 78 / 0 / 0 |
| decode, allocating the output (60) | vLLM | 41 / 15 / 0 (4 no ref.) | 53 / 3 / 4 | 30 / 20 / 6 (4 no ref.) |
| decode with `out=` (20) | vLLM | 14 / 6 / 0 | 18 / 2 / 0 | 10 / 7 / 3 |
| public `flash_attn_with_kvcache` decode (8) | vLLM | 5 / 2 / 1 | 6 / 1 / 1 | 5 / 2 / 1 |
| store-only (`reshape_and_cache`) (4) | vLLM | 3 / 0 / 1 | 3 / 1 / 0 | 4 / 0 / 0 |
| **all (170)** | | **141 / 23 / 2 (4 no ref.)** | **158 / 7 / 5** | **127 / 29 / 10 (4 no ref.)** |

Median ours / reference (bf16, fp32):

| rows | C4D | Axion | Xeon |
|---|---|---|---|
| `flash_attn_func` | 0.17, 0.30 | 0.46, 0.48 | 0.41, 0.32 |
| decode, allocating the output | 0.84, 0.98 | 0.86, 0.88 | 0.96, 0.94 |
| decode with `out=` | 0.84, 0.97 | 0.85, 0.88 | 0.91, 0.98 |
| public `flash_attn_with_kvcache` decode | 0.83, 0.99 | 0.91, 0.94 | 0.92, 0.83 |
| store-only (`reshape_and_cache`) | 0.46, 0.86 | 0.94, 0.77 | 0.61, 0.58 |

<details><summary>`flash_attn_func`: µs per call, ours / SDPA (78 rows)</summary>

| row | C4D ours | C4D SDPA | C4D ratio | Axion ours | Axion SDPA | Axion ratio | Xeon ours | Xeon SDPA | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `128:8:64 bf16` | 17.8 | 128.4 | 0.14 | 21.4 | 61.9 | 0.35 | 19.1 | 50.7 | 0.38 |
| `128:8:64 fp32` | 27.0 | 125.7 | 0.21 | 44.4 | 165.4 | 0.27 | 36.9 | 118.6 | 0.31 |
| `512:8:64 bf16` | 202.9 | 1614.5 | 0.13 | 210.7 | 608.6 | 0.35 | 147.7 | 416.3 | 0.35 |
| `512:8:64 fp32` | 335.4 | 1535.7 | 0.22 | 568.7 | 1775.8 | 0.32 | 302.5 | 1336.2 | 0.23 |
| `1024:8:64 bf16` | 777.9 | 4475.8 | 0.17 | 799.0 | 1645.6 | 0.49 | 474.5 | 1180.5 | 0.40 |
| `1024:8:64 fp32` | 1318.3 | 4361.5 | 0.30 | 2195.7 | 4605.7 | 0.48 | 1019.6 | 3174.8 | 0.32 |
| `2048:8:64 bf16` | 3059.5 | 14899.7 | 0.21 | 3101.4 | 5323.1 | 0.58 | 1617.9 | 3673.1 | 0.44 |
| `2048:8:64 fp32` | 5463.9 | 14473.3 | 0.38 | 8608.6 | 14336.3 | 0.60 | 3717.5 | 10525.9 | 0.35 |
| `4096:8:64 bf16` | 12166.6 | 53587.0 | 0.23 | 12177.8 | 18888.9 | 0.64 | 5909.2 | 12562.7 | 0.47 |
| `4096:8:64 fp32` | 22617.0 | 52088.1 | 0.43 | 34520.4 | 48266.7 | 0.72 | 14028.9 | 37208.2 | 0.38 |
| `256:16:128 bf16` | 212.2 | 1524.9 | 0.14 | 196.2 | 544.2 | 0.36 | 179.8 | 359.6 | 0.50 |
| `256:16:128 fp32` | 365.4 | 1515.2 | 0.24 | 542.6 | 1331.9 | 0.41 | 376.2 | 1284.9 | 0.29 |
| `1024:16:128 bf16` | 3068.4 | 16711.5 | 0.18 | 2628.4 | 5236.4 | 0.50 | 1482.3 | 3376.5 | 0.44 |
| `1024:16:128 fp32` | 5716.1 | 16572.2 | 0.34 | 7965.6 | 13678.4 | 0.58 | 3508.6 | 10884.3 | 0.32 |
| `2048:16:128 bf16` | 12049.0 | 56116.4 | 0.21 | 10231.4 | 17652.4 | 0.58 | 5099.7 | 10831.1 | 0.47 |
| `2048:16:128 fp32` | 22460.4 | 54368.4 | 0.41 | 31463.2 | 43325.1 | 0.73 | 12773.4 | 35716.2 | 0.36 |
| `4096:16:128 bf16` | 47063.3 | 205989.1 | 0.23 | 42005.8 | 62507.5 | 0.67 | 19670.9 | 38231.3 | 0.51 |
| `4096:16:128 fp32` | 86773.7 | 198862.5 | 0.44 | 125173.3 | 149934.8 | 0.83 | 66588.7 | 137427.2 | 0.48 |
| `512:32:128 bf16` | 1597.5 | 12164.1 | 0.13 | 1386.0 | 4075.0 | 0.34 | 948.6 | 2156.3 | 0.44 |
| `512:32:128 fp32` | 2943.9 | 12068.5 | 0.24 | 4126.8 | 10346.7 | 0.40 | 2263.5 | 10391.3 | 0.22 |
| `1024:8:32 bf16` | 431.2 | 2584.2 | 0.17 | 540.8 | 1182.7 | 0.46 | 359.7 | 1000.9 | 0.36 |
| `1024:8:32 fp32` | 706.5 | 2462.7 | 0.29 | 1329.4 | 2762.2 | 0.48 | 667.2 | 1945.5 | 0.34 |
| `17:8:64 bf16` | 2.5 | 11.0 | 0.23 | 5.2 | 17.6 | 0.30 | 5.5 | 15.3 | 0.36 |
| `17:8:64 fp32` | 2.8 | 11.5 | 0.24 | 5.7 | 20.0 | 0.28 | 5.8 | 14.5 | 0.40 |
| `65:8:64 bf16` | 7.2 | 55.6 | 0.13 | 11.2 | 31.1 | 0.36 | 12.4 | 36.9 | 0.34 |
| `65:8:64 fp32` | 10.0 | 44.3 | 0.23 | 17.3 | 47.6 | 0.36 | 18.2 | 54.7 | 0.33 |
| `129:8:64 bf16` | 18.8 | 162.3 | 0.12 | 22.0 | 66.3 | 0.33 | 24.5 | 65.4 | 0.38 |
| `129:8:64 fp32` | 28.2 | 133.4 | 0.21 | 46.0 | 127.3 | 0.36 | 50.3 | 149.2 | 0.34 |
| `257:8:64 bf16` | 59.0 | 489.0 | 0.12 | 62.5 | 182.1 | 0.34 | 58.4 | 183.4 | 0.32 |
| `257:8:64 fp32` | 94.9 | 422.4 | 0.22 | 155.7 | 425.7 | 0.37 | 127.3 | 417.2 | 0.31 |
| `513:8:64 bf16` | 206.8 | 1726.8 | 0.12 | 214.6 | 618.1 | 0.35 | 166.3 | 462.9 | 0.36 |
| `513:8:64 fp32` | 342.8 | 1563.0 | 0.22 | 573.1 | 1591.2 | 0.36 | 356.3 | 1537.2 | 0.23 |
| `1025:8:64 bf16` | 785.9 | 4699.1 | 0.17 | 804.0 | 1661.7 | 0.48 | 538.3 | 1322.7 | 0.41 |
| `1025:8:64 fp32` | 1328.5 | 4370.1 | 0.30 | 2203.3 | 4100.1 | 0.54 | 1104.0 | 3603.6 | 0.31 |
| `2049:8:64 bf16` | 3069.7 | 15346.0 | 0.20 | 3103.6 | 5347.1 | 0.58 | 1713.0 | 3720.4 | 0.46 |
| `2049:8:64 fp32` | 5223.3 | 14458.8 | 0.36 | 8597.5 | 12682.8 | 0.68 | 3735.7 | 10652.0 | 0.35 |
| `4097:8:64 bf16` | 12138.3 | 54627.3 | 0.22 | 12254.2 | 19455.6 | 0.63 | 6007.6 | 12538.6 | 0.48 |
| `4097:8:64 fp32` | 20794.1 | 51904.0 | 0.40 | 34556.5 | 42608.9 | 0.81 | 14081.4 | 37515.4 | 0.38 |
| `129:16:128 bf16` | 68.6 | 634.9 | 0.11 | 65.4 | 192.3 | 0.34 | 77.5 | 166.2 | 0.47 |
| `129:16:128 fp32` | 111.8 | 466.6 | 0.24 | 161.4 | 403.8 | 0.40 | 176.6 | 479.2 | 0.37 |
| `257:16:128 bf16` | 221.1 | 1913.4 | 0.12 | 201.1 | 574.3 | 0.35 | 195.5 | 563.4 | 0.35 |
| `257:16:128 fp32` | 380.8 | 1562.4 | 0.24 | 557.4 | 1364.7 | 0.41 | 425.0 | 1444.8 | 0.29 |
| `1025:16:128 bf16` | 3095.3 | 18158.9 | 0.17 | 2655.7 | 5298.1 | 0.50 | 1571.4 | 3521.6 | 0.45 |
| `1025:16:128 fp32` | 5765.1 | 16574.9 | 0.35 | 8013.7 | 13704.0 | 0.58 | 3689.3 | 11162.6 | 0.33 |
| `2049:16:128 bf16` | 12139.5 | 60387.1 | 0.20 | 10298.1 | 17583.6 | 0.59 | 5077.8 | 10770.7 | 0.47 |
| `2049:16:128 fp32` | 20977.0 | 54803.5 | 0.38 | 31604.5 | 43268.5 | 0.73 | 13043.3 | 36610.3 | 0.36 |
| `1025:8:32 bf16` | 435.6 | 2698.7 | 0.16 | 540.1 | 1180.3 | 0.46 | 382.6 | 1004.9 | 0.38 |
| `1025:8:32 fp32` | 711.8 | 2482.3 | 0.29 | 1329.6 | 2759.7 | 0.48 | 669.7 | 2029.0 | 0.33 |
| `1x129:8:64 bf16` | 3.5 | 35.1 | 0.10 | 4.8 | 21.3 | 0.23 | 6.5 | 13.6 | 0.48 |
| `1x129:8:64 fp32` | 3.8 | 10.7 | 0.36 | 7.0 | 22.5 | 0.31 | 5.6 | 13.3 | 0.42 |
| `17x301:8:64 bf16` | 14.6 | 58.7 | 0.25 | 13.8 | 44.3 | 0.31 | 15.0 | 38.8 | 0.39 |
| `17x301:8:64 fp32` | 19.4 | 57.3 | 0.34 | 31.4 | 61.3 | 0.51 | 34.9 | 61.2 | 0.57 |
| `127x129:8:64 bf16` | 18.5 | 150.1 | 0.12 | 20.4 | 88.9 | 0.23 | 22.0 | 91.8 | 0.24 |
| `127x129:8:64 fp32` | 28.0 | 147.5 | 0.19 | 44.7 | 150.2 | 0.30 | 40.6 | 153.0 | 0.27 |
| `129x127:8:64 bf16` | 18.8 | 196.4 | 0.10 | 19.8 | 89.5 | 0.22 | 22.1 | 96.2 | 0.23 |
| `129x127:8:64 fp32` | 27.5 | 165.6 | 0.17 | 43.5 | 153.0 | 0.28 | 39.7 | 159.9 | 0.25 |
| `257x513:8:64 bf16` | 153.6 | 1022.5 | 0.15 | 156.0 | 394.4 | 0.40 | 108.0 | 319.9 | 0.34 |
| `257x513:8:64 fp32` | 252.8 | 850.5 | 0.30 | 424.4 | 799.2 | 0.53 | 243.4 | 817.0 | 0.30 |
| `513x257:8:64 bf16` | 58.3 | 920.6 | 0.06 | 60.4 | 368.9 | 0.16 | 71.3 | 345.7 | 0.21 |
| `513x257:8:64 fp32` | 95.4 | 872.5 | 0.11 | 153.0 | 776.5 | 0.20 | 134.6 | 787.2 | 0.17 |
| `1025x2049:8:64 bf16` | 2298.2 | 13232.2 | 0.17 | 2321.7 | 5012.7 | 0.46 | 1265.2 | 4109.1 | 0.31 |
| `1025x2049:8:64 fp32` | 3898.1 | 12611.6 | 0.31 | 6484.5 | 11380.9 | 0.57 | 2757.9 | 9967.0 | 0.28 |
| `2049x1025:8:64 bf16` | 792.5 | 12652.8 | 0.06 | 803.9 | 4597.2 | 0.17 | 561.6 | 3265.3 | 0.17 |
| `2049x1025:8:64 fp32` | 1362.2 | 12103.1 | 0.11 | 2208.3 | 10744.6 | 0.21 | 1125.5 | 9398.7 | 0.12 |
| `1024:8:64 noncausal bf16` | 1505.0 | 5997.1 | 0.25 | 1532.1 | 2142.4 | 0.72 | 790.6 | 1488.7 | 0.53 |
| `1024:8:64 noncausal fp32` | 2554.2 | 5744.6 | 0.44 | 4263.5 | 4437.9 | 0.96 | 1726.3 | 4210.0 | 0.41 |
| `1025:8:64 noncausal bf16` | 1550.3 | 6323.3 | 0.25 | 1566.7 | 2196.5 | 0.71 | 852.2 | 1576.6 | 0.54 |
| `1025:8:64 noncausal fp32` | 2602.3 | 5820.4 | 0.45 | 4310.8 | 4505.6 | 0.96 | 1768.8 | 4392.6 | 0.40 |
| `1024:16:128 noncausal bf16` | 5829.3 | 22130.4 | 0.26 | 5027.5 | 6924.6 | 0.73 | 2436.3 | 4327.3 | 0.56 |
| `1024:16:128 noncausal fp32` | 11253.3 | 21890.5 | 0.51 | 15481.9 | 16158.7 | 0.96 | 5715.1 | 14323.7 | 0.40 |
| `1024:32/8:128 bf16` | 6036.5 | 33279.6 | 0.18 | 5221.4 | 10505.0 | 0.50 | 2839.1 | 6320.2 | 0.45 |
| `1024:32/8:128 fp32` | 11068.4 | 32649.5 | 0.34 | 15936.9 | 27250.0 | 0.58 | 6908.3 | 21664.8 | 0.32 |
| `1024:16/1:64 bf16` | 1563.7 | 8923.7 | 0.18 | 1602.9 | 3237.8 | 0.50 | 938.5 | 2269.6 | 0.41 |
| `1024:16/1:64 fp32` | 2644.8 | 8617.2 | 0.31 | 4402.2 | 8032.6 | 0.55 | 1761.8 | 6098.0 | 0.29 |
| `1024:8:64 fused bf16` | 784.0 | 4535.4 | 0.17 | 801.5 | 1672.7 | 0.48 | 498.3 | 1209.6 | 0.41 |
| `1024:8:64 fused fp32` | 1323.3 | 4351.6 | 0.30 | 2194.8 | 4081.1 | 0.54 | 971.0 | 3242.9 | 0.30 |
| `257:16:128 fused bf16` | 222.1 | 1923.4 | 0.12 | 202.1 | 576.4 | 0.35 | 217.5 | 593.4 | 0.37 |
| `257:16:128 fused fp32` | 392.5 | 1595.9 | 0.25 | 567.5 | 1365.8 | 0.42 | 447.8 | 1570.6 | 0.29 |

</details>

<details><summary>decode, allocating the output: µs per call, ours / vLLM (60 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `engine self b=1 ctx=256 bf16` | 3.9 | 7.4 | 0.52 | 6.4 | 10.9 | 0.59 | 5.8 | 11.9 | 0.49 |
| `engine self b=1 ctx=256 fp32` | 7.2 | 9.3 | 0.77 | 9.4 | 16.4 | 0.57 | 10.7 | 21.2 | 0.51 |
| `engine cross b=1 ctx=256 bf16` | 3.8 | 5.9 | 0.64 | 6.1 | 7.9 | 0.77 | 6.4 | 8.2 | 0.78 |
| `engine cross b=1 ctx=256 fp32` | 7.1 | 7.9 | 0.90 | 9.3 | 12.6 | 0.74 | 9.7 | 17.0 | 0.57 |
| `engine self b=8 ctx=256 bf16` | 46.3 | 67.5 | 0.69 | 36.0 | 86.0 | 0.42 | 90.7 | 113.2 | 0.80 |
| `engine self b=8 ctx=256 fp32` | 313.6ᶜ | 325.8 | 0.96 | 97.0 | 147.1 | 0.66 | 166.3 | 176.7 | 0.94 |
| `engine cross b=8 ctx=256 bf16` | 43.8 | 64.9 | 0.67 | 33.9 | 83.5 | 0.41 | 90.7 | 103.5 | 0.88 |
| `engine cross b=8 ctx=256 fp32` | 313.2ᶜ | 314.4 | 1.00 | 89.6 | 139.1 | 0.64 | 165.9 | 165.0 | 1.01 |
| `engine self b=8 ctx=1024 bf16` | 634.3ᶜ | 741.0 | 0.86 | 396.8ᶜ | 422.2 | 0.94 | 411.0 | 420.3 | 0.98 |
| `engine self b=8 ctx=1024 fp32` | 1235.7ᶜ | 1257.1 | 0.98 | 723.0ᶜ | 780.3 | 0.93 | 886.2 | 985.5 | 0.90 |
| `engine cross b=8 ctx=1024 bf16` | 632.5ᶜ | 735.8 | 0.86 | 385.0ᶜ | 404.5 | 0.95 | 402.2 | 411.8 | 0.98 |
| `engine cross b=8 ctx=1024 fp32` | 1233.6ᶜ | 1245.4 | 0.99 | 700.3ᶜ | 771.8 | 0.91 | 860.4 | 943.4 | 0.91 |
| `engine self b=32 ctx=1024 bf16` | 2521.0ᶜ | 3028.9 | 0.83 | 1525.7ᶜ | 1817.4 | 0.84 | 2959.2ᶜ | 2848.8 | 1.04 ▲ |
| `engine self b=32 ctx=1024 fp32` | 4946.1ᶜ | 5076.9 | 0.97 | 3012.5ᶜ | 3249.9 | 0.93 | 5222.0ᶜ | 5133.2 | 1.02 |
| `engine cross b=32 ctx=1024 bf16` | 2512.1ᶜ | 2992.6 | 0.84 | 1587.0ᶜ | 1855.8 | 0.86 | 2953.0ᶜ | 2744.3 | 1.08 ▲ |
| `engine cross b=32 ctx=1024 fp32` | 4920.5ᶜ | 5030.7 | 0.98 | 2875.7ᶜ | 3275.3 | 0.88 | 5191.5ᶜ | 4967.1 | 1.05 ▲ |
| `engine self b=1 ctx=4096 bf16` | 316.7ᶜ | 347.7 | 0.91 | 137.3 | 156.9 | 0.88 | 224.6 | 234.8 | 0.96 |
| `engine self b=1 ctx=4096 fp32` | 630.5ᶜ | 624.5 | 1.01 | 337.3ᶜ | 348.9 | 0.97 | 384.9 | 367.0 | 1.05 |
| `engine cross b=1 ctx=4096 bf16` | 317.7ᶜ | 358.3 | 0.89 | 140.3 | 151.3 | 0.93 | 227.6 | 246.1 | 0.92 |
| `engine cross b=1 ctx=4096 fp32` | 633.1ᶜ | 625.0 | 1.01 | 342.0ᶜ | 356.0 | 0.96 | 381.6 | 385.4 | 0.99 |
| `engine self b=8 ctx=255 bf16` | 45.7 | 61.4 | 0.74 | 36.9 | 75.3 | 0.49 | 93.4 | 119.3 | 0.78 |
| `engine self b=8 ctx=255 fp32` | 314.3ᶜ | 330.3 | 0.95 | 88.9 | 147.2 | 0.60 | 171.0 | 178.7 | 0.96 |
| `engine cross b=8 ctx=255 bf16` | 44.0 | 65.3 | 0.67 | 35.9 | 74.7 | 0.48 | 91.0 | 107.9 | 0.84 |
| `engine cross b=8 ctx=255 fp32` | 313.2ᶜ | 318.5 | 0.98 | 84.8 | 136.4 | 0.62 | 169.6 | 166.7 | 1.02 |
| `engine self b=8 ctx=257 bf16` | 174.6ᶜ | 220.1 | 0.79 | 38.8 | 81.8 | 0.47 | 96.2 | 132.2 | 0.73 |
| `engine self b=8 ctx=257 fp32` | 326.3ᶜ | 367.4 | 0.89 | 101.6 | 144.3 | 0.70 | 179.8 | 209.3 | 0.86 |
| `engine cross b=8 ctx=257 bf16` | 173.7ᶜ | 212.9 | 0.82 | 38.3 | 70.0 | 0.55 | 98.8 | 120.9 | 0.82 |
| `engine cross b=8 ctx=257 fp32` | 326.0ᶜ | 357.0 | 0.91 | 93.3 | 133.0 | 0.70 | 177.0 | 187.2 | 0.95 |
| `engine self b=8 ctx=1023 bf16` | 629.4ᶜ | 730.8 | 0.86 | 388.5ᶜ | 415.7 | 0.93 | 444.6 | 449.3 | 0.99 |
| `engine self b=8 ctx=1023 fp32` | 1234.0ᶜ | 1257.5 | 0.98 | 707.5ᶜ | 743.5 | 0.95 | 1026.8 | 1071.8 | 0.96 |
| `engine cross b=8 ctx=1023 bf16` | 631.8ᶜ | 720.9 | 0.88 | 373.1ᶜ | 412.7 | 0.90 | 442.2 | 455.7 | 0.97 |
| `engine cross b=8 ctx=1023 fp32` | 1231.0ᶜ | 1251.9 | 0.98 | 706.4ᶜ | 792.8 | 0.89 | 995.9 | 1057.1 | 0.94 |
| `engine self b=8 ctx=1025 bf16` | 640.4ᶜ | 759.2 | 0.84 | 346.2ᶜ | 421.6 | 0.82 | 471.0 | 477.3 | 0.99 |
| `engine self b=8 ctx=1025 fp32` | 1244.6ᶜ | 1299.4 | 0.96 | 689.1ᶜ | 769.7 | 0.90 | 1137.6 | 1136.2 | 1.00 |
| `engine cross b=8 ctx=1025 bf16` | 638.0ᶜ | 750.8 | 0.85 | 362.5ᶜ | 411.8 | 0.88 | 463.4 | 507.9 | 0.91 |
| `engine cross b=8 ctx=1025 fp32` | 1245.1ᶜ | 1289.8 | 0.97 | 691.9ᶜ | 763.1 | 0.91 | 1127.7 | 1099.8 | 1.03 |
| `engine self b=1 ctx=4097 bf16` | 318.7ᶜ | 369.1 | 0.86 | 138.2 | 159.3 | 0.87 | 224.5 | 247.1 | 0.91 |
| `engine self b=1 ctx=4097 fp32` | 633.1ᶜ | 633.2 | 1.00 | 344.7ᶜ | 349.2 | 0.99 | 373.1 | 393.3 | 0.95 |
| `engine cross b=1 ctx=4097 bf16` | 319.5ᶜ | 371.6 | 0.86 | 140.3 | 153.7 | 0.91 | 236.8 | 235.4 | 1.01 |
| `engine cross b=1 ctx=4097 fp32` | 633.2ᶜ | 631.9 | 1.00 | 348.5ᶜ | 350.3 | 1.00 | 378.0 | 386.1 | 0.98 |
| `engine self b=8 ctx=1024 page=16 bf16` | 631.3ᶜ | - | - | 369.7ᶜ | 425.2 | 0.87 | 524.2 | - | - |
| `engine self b=8 ctx=1024 page=16 fp32` | 1230.1ᶜ | - | - | 678.2ᶜ | 769.8 | 0.88 | 816.8 | - | - |
| `engine self b=8 ctx=1024 page=48 bf16` | 634.6ᶜ | - | - | 279.4ᶜ | 317.0 | 0.88 | 372.8 | - | - |
| `engine self b=8 ctx=1024 page=48 fp32` | 1228.4ᶜ | - | - | 517.2ᶜ | 627.4 | 0.82 | 778.0 | - | - |
| `engine self b=8 ctx=1024 page=64 bf16` | 632.1ᶜ | 721.2 | 0.88 | 282.2ᶜ | 328.2 | 0.86 | 383.1 | 385.5 | 0.99 |
| `engine self b=8 ctx=1024 page=64 fp32` | 1230.0ᶜ | 1256.6 | 0.98 | 518.3ᶜ | 638.6 | 0.81 | 901.2 | 1012.7 | 0.89 |
| `engine self b=8 ctx=1024 h=16/4 bf16` | 72.7 | 92.5 | 0.79 | 79.8 | 74.4 | 1.07 ▲ | 122.8 | 116.0 | 1.06 ▲ |
| `engine self b=8 ctx=1024 h=16/4 fp32` | 329.6ᶜ | 359.2 | 0.92 | 117.6 | 149.2 | 0.79 | 210.3 | 238.8 | 0.88 |
| `engine cross b=8 ctx=1024 h=16/4 bf16` | 74.4 | 102.5 | 0.73 | 80.2 | 68.9 | 1.16 ▲ | 122.1 | 108.6 | 1.12 ▲ |
| `engine cross b=8 ctx=1024 h=16/4 fp32` | 331.3ᶜ | 350.8 | 0.94 | 124.5 | 158.6 | 0.79 | 207.1 | 222.1 | 0.93 |
| `engine cross b=8 ctx=1024 h=16/1 bf16` | 34.6 | 107.8 | 0.32 | 66.5 | 37.7 | 1.77 ▲ | 75.0 | 35.4 | 2.12 ▲ |
| `engine cross b=8 ctx=1024 h=16/1 fp32` | 44.5 | 138.8 | 0.32 | 87.2 | 83.4 | 1.05 ▲ | 89.6 | 207.5 | 0.43 |
| `engine self b=8 ctx=het bf16` | 391.4ᶜ | 475.6 | 0.82 | 154.1 | 215.7 | 0.71 | 256.1 | 275.2 | 0.93 |
| `engine self b=8 ctx=het fp32` | 761.8ᶜ | 799.2 | 0.95 | 430.1ᶜ | 490.1 | 0.88 | 451.8 | 507.8 | 0.89 |
| `engine cross b=8 ctx=het bf16` | 391.0ᶜ | 467.7 | 0.84 | 148.0 | 209.9 | 0.71 | 257.4 | 267.4 | 0.96 |
| `engine cross b=8 ctx=het fp32` | 759.8ᶜ | 787.0 | 0.97 | 424.0ᶜ | 477.9 | 0.89 | 459.2 | 505.8 | 0.91 |
| `engine cross b=8 ctx=1024 d=32 bf16` | 325.2ᶜ | 379.7 | 0.86 | 89.0 | 118.5 | 0.75 | 205.4 | 231.4 | 0.89 |
| `engine cross b=8 ctx=1024 d=32 fp32` | 626.6ᶜ | 636.2 | 0.98 | 408.0ᶜ | 430.3 | 0.95 | 373.5 | 364.5 | 1.02 |
| `engine cross b=8 ctx=1024 d=128 bf16` | 623.9ᶜ | 721.0 | 0.87 | 269.2ᶜ | 360.2 | 0.75 | 409.2 | 413.1 | 0.99 |
| `engine cross b=8 ctx=1024 d=128 fp32` | 1232.5ᶜ | 1247.0 | 0.99 | 567.9ᶜ | 623.1 | 0.91 | 857.9 | 955.0 | 0.90 |

</details>

<details><summary>decode with `out=`: µs per call, ours / vLLM (20 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `engine_out self b=1 ctx=256 bf16` | 3.6 | 7.4 | 0.49 | 5.3 | 10.0 | 0.53 | 5.6 | 14.5 | 0.39 |
| `engine_out self b=1 ctx=256 fp32` | 6.7 | 9.0 | 0.74 | 9.2 | 16.3 | 0.57 | 9.4 | 27.2 | 0.35 |
| `engine_out cross b=1 ctx=256 bf16` | 3.5 | 5.9 | 0.60 | 5.6 | 7.2 | 0.77 | 4.9 | 7.2 | 0.68 |
| `engine_out cross b=1 ctx=256 fp32` | 6.7 | 7.7 | 0.87 | 8.6 | 12.6 | 0.68 | 10.0 | 15.1 | 0.66 |
| `engine_out self b=8 ctx=256 bf16` | 46.3 | 68.4 | 0.68 | 34.7 | 83.1 | 0.42 | 89.8 | 116.5 | 0.77 |
| `engine_out self b=8 ctx=256 fp32` | 311.6ᶜ | 324.0 | 0.96 | 84.5 | 141.3 | 0.60 | 165.6 | 180.3 | 0.92 |
| `engine_out cross b=8 ctx=256 bf16` | 43.5 | 73.3 | 0.59 | 34.5 | 80.3 | 0.43 | 87.8 | 107.1 | 0.82 |
| `engine_out cross b=8 ctx=256 fp32` | 311.2ᶜ | 307.7 | 1.01 | 87.3 | 137.7 | 0.63 | 165.5 | 164.1 | 1.01 |
| `engine_out self b=8 ctx=1024 bf16` | 631.1ᶜ | 731.7 | 0.86 | 373.6ᶜ | 404.9 | 0.92 | 398.3 | 422.7 | 0.94 |
| `engine_out self b=8 ctx=1024 fp32` | 1233.9ᶜ | 1262.9 | 0.98 | 694.9ᶜ | 795.2 | 0.87 | 924.2 | 972.8 | 0.95 |
| `engine_out cross b=8 ctx=1024 bf16` | 631.1ᶜ | 735.5 | 0.86 | 366.7ᶜ | 407.8 | 0.90 | 431.5 | 427.0 | 1.01 |
| `engine_out cross b=8 ctx=1024 fp32` | 1235.9ᶜ | 1252.0 | 0.99 | 713.1ᶜ | 761.7 | 0.94 | 883.1 | 973.3 | 0.91 |
| `engine_out self b=32 ctx=1024 bf16` | 2522.9ᶜ | 3016.1 | 0.84 | 1545.4ᶜ | 1774.8 | 0.87 | 2903.3ᶜ | 2794.8 | 1.04 ▲ |
| `engine_out self b=32 ctx=1024 fp32` | 4953.5ᶜ | 5115.2 | 0.97 | 2926.1ᶜ | 3259.5 | 0.90 | 5234.9ᶜ | 5099.4 | 1.03 |
| `engine_out cross b=32 ctx=1024 bf16` | 2519.1ᶜ | 2977.7 | 0.85 | 1471.3ᶜ | 1784.8 | 0.82 | 2938.2ᶜ | 2750.2 | 1.07 ▲ |
| `engine_out cross b=32 ctx=1024 fp32` | 4944.6ᶜ | 5088.1 | 0.97 | 2816.6ᶜ | 3208.0 | 0.88 | 5103.5ᶜ | 4942.3 | 1.03 ▲ |
| `engine_out self b=1 ctx=4096 bf16` | 316.6ᶜ | 373.3 | 0.85 | 138.6 | 156.6 | 0.88 | 219.8 | 241.0 | 0.91 |
| `engine_out self b=1 ctx=4096 fp32` | 630.1ᶜ | 629.4 | 1.00 | 336.9ᶜ | 334.4 | 1.01 | 381.7 | 375.5 | 1.02 |
| `engine_out cross b=1 ctx=4096 bf16` | 316.2ᶜ | 370.0 | 0.85 | 137.5 | 152.6 | 0.90 | 218.2 | 238.5 | 0.91 |
| `engine_out cross b=1 ctx=4096 fp32` | 629.8ᶜ | 624.5 | 1.01 | 336.6ᶜ | 341.0 | 0.99 | 382.2 | 373.7 | 1.02 |

</details>

<details><summary>public `flash_attn_with_kvcache` decode: µs per call, ours / vLLM (8 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `api self b=1 ctx=256 bf16` | 4.9 | 7.5 | 0.65 | 10.0 | 12.6 | 0.79 | 7.2 | 12.5 | 0.57 |
| `api self b=1 ctx=256 fp32` | 8.0 | 9.0 | 0.89 | 14.2 | 19.0 | 0.74 | 11.4 | 26.9 | 0.42 |
| `api cross b=1 ctx=256 bf16` | 4.8 | 6.0 | 0.81 | 9.4 | 8.5 | 1.10 ▲ | 6.7 | 7.5 | 0.89 |
| `api cross b=1 ctx=256 fp32` | 8.1 | 7.6 | 1.06 ▲ | 13.1 | 13.1 | 1.00 | 11.8 | 15.2 | 0.78 |
| `api self b=8 ctx=1024 bf16` | 639.9ᶜ | 747.3 | 0.86 | 361.3ᶜ | 399.6 | 0.90 | 446.8 | 420.6 | 1.06 ▲ |
| `api self b=8 ctx=1024 fp32` | 1245.7ᶜ | 1263.5 | 0.99 | 711.2ᶜ | 762.6 | 0.93 | 891.5 | 1009.7 | 0.88 |
| `api cross b=8 ctx=1024 bf16` | 645.5ᶜ | 734.8 | 0.88 | 371.2ᶜ | 406.6 | 0.91 | 396.1 | 417.0 | 0.95 |
| `api cross b=8 ctx=1024 fp32` | 1240.5ᶜ | 1250.9 | 0.99 | 723.2ᶜ | 763.9 | 0.95 | 854.1 | 895.0 | 0.95 |

</details>

<details><summary>store-only (`reshape_and_cache`): µs per call, ours / vLLM (4 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `store store b=32 ctx=1 bf16` | 3.5 | 5.4 | 0.64 | 5.0 | 5.1 | 1.00 | 9.3 | 11.0 | 0.84 |
| `store store b=32 ctx=1 fp32` | 11.9 | 10.2 | 1.17 ▲ | 6.9 | 7.4 | 0.93 | 24.3 | 34.7 | 0.70 |
| `store store b=1 ctx=1024 bf16` | 32.2 | 115.9 | 0.28 | 56.1 | 63.1 | 0.89 | 85.9 | 222.8 | 0.39 |
| `store store b=1 ctx=1024 fp32` | 124.1 | 226.3 | 0.55 | 75.6 | 126.3 | 0.60 | 188.8 | 406.1 | 0.47 |

</details>

**Where we do not win.**
* **GQA/MQA decode** (`h=16/4`, `h=16/1`, bf16) on the Axion and the Xeon: vLLM is
  up to 2.1x faster on MQA (`cross b=8 ctx=1024 h=16/1` bf16: 66.5 vs 37.7 us on the
  Axion, 75.0 vs 35.4 us on the Xeon). On the C4D these rows win or tie.
* **Xeon `b=32 ctx=1024`** decode: 1.03-1.08. This VM reports its socket's 260 MB L3,
  so only 8 Xeon decode rows cross the cold threshold; the rest are timed warm, and
  their control spreads reach 11%.
* **Single small calls**: the public API at `b=1 ctx=256` (Axion bf16 9.4 vs 8.5 us,
  C4D fp32 8.1 vs 7.6) and C4D store-only `b=32 ctx=1` fp32 (11.9 vs 10.2 us).
* **No reference**: vLLM on x86 takes only page sizes that are a multiple of 32, so
  the four `page=16` / `page=48` decode rows have no vLLM time on the C4D and the Xeon.

**Mixed context lengths** (`ctx=het`: 8 sequences of 17 ... 2047 tokens) were the
largest loss until `453760b` (Axion 1.55-2.74x slower than vLLM): decode tasks were
split across workers by count, so on 8 threads each worker took one whole sequence
and the one holding 2047 keys did 3.3x the mean. Split by context length, they now
win (11 of 12 rows; Xeon `cross` bf16 0.96, unresolved; Axion fp32 `self` 1221 -> 430
us, vLLM 490). vLLM's bf16 output on
these rows is 1.8-2.3e-3 above the bf16 rounding, outside our 1e-3 gate but inside the
reference sanity gate (3e-2), so they are scored.

**Caveats.**
* `flash_attn_func` against vLLM's prefill path is in the JSON (`new/vllm`, 62
  batch-1 equal-length rows): 0.44-1.09 on the C4D, 0.59-0.93 on the Axion, and
  0.38-0.90 on the Xeon.
* Rows bound by memory placement swing 20-30% with allocation order, and
  single-run verdict counts move by several rows. Read a row's verdict from
  repeated `--load-seed` runs, not from one table.
* Measurements against the former C++ backend (16-vCPU boxes) are in this file's
  history (`ad1f78c` and before).

## Benchmark it yourself

```bash
scripts/build_vllm_ref.sh             # vLLM reference -> ~/vllm_cpu_build/*.so (pinned commit)
uv run python benchmarks/bench.py --manifest full \
    --vllm-bf16 ~/vllm_cpu_build/vllm_cpu_attn_bench.so --vllm-fp32 ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --rounds 61 --paged-rounds 121 --json /path/outside/the/repo/run.json
```

On a fresh VM, `scripts/box_provision.sh` installs what the image lacks (a toolchain,
`python3-dev`, `uv`) and the repository venv; then `scripts/build_vllm_ref.sh`, and
`scripts/box_verify.sh BRANCH BASE "ISAS" <vLLM args>` runs every gate on BRANCH and the
full-manifest run against SDPA, vLLM and BASE as a previous build at threads = physical
cores (`scripts/ab_summary.py` summarizes the JSON). The tables above came from those
scripts.

* On an AMX box pass `--vllm-bf16 ~/vllm_cpu_build/vllm_cpu_attn_bench_amx.so
  --vllm-isa-bf16 amx`; on aarch64 pass `--vllm-isa-bf16 neon --vllm-isa-fp32
  neon`; always `--threads <physical cores>`. `--block-calls 40` is only for a desktop or laptop with
  frequency scaling; the GCP VMs above use one call per sample.
* `--only "<label substring>"` runs a subset, `--resume` continues a `--json`
  after an interruption, `--previous <saved package>` A/Bs two of your builds
  against an independent control, and `--load-seed N` changes the module-load and
  cache-allocation order (repeat memory-bound rows with several).
* Judge on the minimum and only inside one process; never compare two runs
  (these machines drift 5-15% between processes).

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16** where the model allows: on the Xeon, for sequences of 129 or
   more tokens, the AMX policy is 2.7-8.1x faster than the vector policy in
   bf16 and 1.1-3.4x in fp32 (fp32 runs as bf16 hi/lo products).
3. **Pass `out=`** to `self_attention` / `cross_attention` on the decode path.
   It saves the allocation, which is a sizeable share of a batch-1 call.
4. **Page size.** Use 32 on x86 and 16 on ARM (the engine defaults); 16, 32, 48
   and 64 are specialized at compile time.
5. **Leave AMX on** (the default on AMX CPUs). Few-query calls (fp32 up to ~65
   rows at head_dim 64, bf16 1-2 rows) automatically use the vector policy,
   which is faster there; `CPU_ATTENTION_AMX=0` forces it everywhere.

## Tests

```bash
uv run python tests/test_dense.py          # flash_attn_func vs a naive fp32 reference (82 checks)
uv run python tests/test_flash_api.py      # flash-shaped API and its rejections
uv run python tests/fuzz_dense.py --cases 80 --max-seq 900
uv run python tests/test_paged_sweep.py    # 504 paged cases, both dtypes
uv run python tests/test_regressions.py    # 457 checks: odd/unequal lengths, masked rows, poisoned caches, strides, scale, workers, concurrency
uv run python tests/test_amx.py --dtype both   # AMX policy (AMX CPUs; fails if AMX is present but unusable)
```

Prefix `CPU_ATTENTION_ISA=<tier>` to test a lower tier. The engine-level test
lives in fast-encdec: `PYTHONPATH=. .venv/bin/python tests/test_cpu_attention.py`
(it passes strided chunks of a fused projection; 45 checks).

Accuracy gates (all pass on all three boxes):

| case | limit |
|---|---|
| `flash_attn_func` fp32 | <= 2e-4 vs the naive fp32 reference |
| `flash_attn_func` bf16 | <= 3e-2 |
| paged fp32 | <= 1e-6 (fp64 arbitrates an fp32 reference that is itself out of the gate) |
| paged bf16 | <= 1e-3 *above the bf16 output-rounding floor* |

The paged bf16 "<= 1e-3" is not met literally: rounding the exact result to bf16 alone
exceeds 1e-3 on 77 of 252 sweep cases, so no bf16 output can satisfy an absolute
1e-3. The tests check the excess over that floor (`tests/reference.py`); see
`docs/Mojo.md` §5.

## Documentation

* [`docs/Mojo.md`](docs/Mojo.md): the kernel's design, ISA tiers,
  numerics, call path, defects found, falsification ledger and Mojo 1.2.0
  notes.
* [`AGENTS.md`](AGENTS.md): working rules for anyone changing a kernel.

## License

Apache-2.0. The kernel follows vLLM's `csrc/cpu` attention (Apache-2.0); the
Python API mirrors [flash-attention](https://github.com/Dao-AILab/flash-attention)
(BSD-3).
