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

**Setup** (Mojo 1.2.0 `2016d0cd`, torch 2.14.1+cpu, commit `7c898b8`, Oct 2026).
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
| decode, allocating the output (60) | vLLM | 32 / 22 / 2 (4 no ref.) | 47 / 3 / 10 | 37 / 10 / 9 (4 no ref.) |
| decode with `out=` (20) | vLLM | 12 / 8 / 0 | 18 / 1 / 1 | 14 / 2 / 4 |
| public `flash_attn_with_kvcache` decode (8) | vLLM | 4 / 4 / 0 | 5 / 1 / 2 | 5 / 3 / 0 |
| store-only (`reshape_and_cache`) (4) | vLLM | 3 / 1 / 0 | 3 / 0 / 1 | 4 / 0 / 0 |
| **all (170)** | | **129 / 35 / 2 (4 no ref.)** | **151 / 5 / 14** | **138 / 15 / 13 (4 no ref.)** |

Median ours / reference (bf16, fp32):

| rows | C4D | Axion | Xeon |
|---|---|---|---|
| `flash_attn_func` | 0.17, 0.30 | 0.46, 0.48 | 0.40, 0.32 |
| decode, allocating the output | 0.90, 1.00 | 0.85, 0.89 | 0.91, 0.91 |
| decode with `out=` | 0.90, 1.00 | 0.82, 0.89 | 0.85, 0.93 |
| public `flash_attn_with_kvcache` decode | 0.86, 1.01 | 0.95, 0.93 | 0.94, 0.74 |
| store-only (`reshape_and_cache`) | 0.44, 0.76 | 1.02, 0.80 | 0.64, 0.65 |

<details><summary>`flash_attn_func`: µs per call, ours / SDPA (78 rows)</summary>

| row | C4D ours | C4D SDPA | C4D ratio | Axion ours | Axion SDPA | Axion ratio | Xeon ours | Xeon SDPA | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `128:8:64 bf16` | 17.7 | 127.2 | 0.14 | 21.9 | 63.0 | 0.35 | 23.2 | 63.7 | 0.36 |
| `128:8:64 fp32` | 26.7 | 124.4 | 0.21 | 44.2 | 166.0 | 0.27 | 67.6 | 199.0 | 0.34 |
| `512:8:64 bf16` | 201.3 | 1607.6 | 0.13 | 211.2 | 612.2 | 0.34 | 155.5 | 443.9 | 0.35 |
| `512:8:64 fp32` | 335.0 | 1538.0 | 0.22 | 565.1 | 1772.8 | 0.32 | 340.7 | 1513.2 | 0.23 |
| `1024:8:64 bf16` | 774.8 | 4506.7 | 0.17 | 800.8 | 1646.9 | 0.49 | 504.3 | 1302.1 | 0.39 |
| `1024:8:64 fp32` | 1306.0 | 4361.3 | 0.30 | 2189.3 | 4577.3 | 0.48 | 1107.8 | 3585.4 | 0.31 |
| `2048:8:64 bf16` | 3051.0 | 14826.8 | 0.21 | 3096.0 | 5293.3 | 0.58 | 1810.2 | 4132.2 | 0.44 |
| `2048:8:64 fp32` | 5177.8 | 14306.5 | 0.36 | 8571.6 | 14357.4 | 0.60 | 4064.4 | 11654.0 | 0.35 |
| `4096:8:64 bf16` | 12017.6 | 53319.2 | 0.23 | 12167.4 | 19167.5 | 0.63 | 6546.7 | 14194.7 | 0.46 |
| `4096:8:64 fp32` | 20617.7 | 51567.7 | 0.40 | 34472.6 | 48138.6 | 0.72 | 15525.4 | 41325.5 | 0.38 |
| `256:16:128 bf16` | 211.4 | 1518.7 | 0.14 | 205.9 | 549.3 | 0.37 | 186.2 | 394.6 | 0.47 |
| `256:16:128 fp32` | 360.8 | 1497.5 | 0.24 | 540.5 | 1330.5 | 0.41 | 412.9 | 1485.9 | 0.28 |
| `1024:16:128 bf16` | 3017.5 | 16645.0 | 0.18 | 2636.9 | 5275.6 | 0.50 | 1692.8 | 3683.6 | 0.46 |
| `1024:16:128 fp32` | 5617.6 | 16433.1 | 0.34 | 7946.2 | 13552.9 | 0.59 | 4093.7 | 12564.6 | 0.33 |
| `2048:16:128 bf16` | 11933.5 | 55967.0 | 0.21 | 10287.0 | 17728.7 | 0.58 | 5839.4 | 12088.3 | 0.48 |
| `2048:16:128 fp32` | 22339.3 | 54120.8 | 0.41 | 31528.4 | 43454.3 | 0.73 | 14581.2 | 41822.7 | 0.35 |
| `4096:16:128 bf16` | 46721.1 | 204673.3 | 0.23 | 42219.6 | 62805.6 | 0.67 | 21989.7 | 42277.5 | 0.52 |
| `4096:16:128 fp32` | 86732.7 | 198404.8 | 0.44 | 125260.5 | 150051.1 | 0.83 | 72052.1 | 155991.0 | 0.46 |
| `512:32:128 bf16` | 1567.4 | 12021.2 | 0.13 | 1385.9 | 4059.5 | 0.34 | 1079.6 | 2570.4 | 0.42 |
| `512:32:128 fp32` | 2918.1 | 11807.8 | 0.25 | 4113.1 | 10276.4 | 0.40 | 2721.0 | 11876.5 | 0.23 |
| `1024:8:32 bf16` | 429.3 | 2575.5 | 0.17 | 537.2 | 1174.8 | 0.46 | 386.2 | 1084.4 | 0.36 |
| `1024:8:32 fp32` | 718.7 | 2453.9 | 0.29 | 1326.5 | 2749.0 | 0.48 | 715.6 | 2255.3 | 0.32 |
| `17:8:64 bf16` | 2.5 | 11.0 | 0.23 | 5.4 | 18.9 | 0.29 | 5.5 | 14.9 | 0.37 |
| `17:8:64 fp32` | 2.8 | 11.5 | 0.24 | 5.8 | 20.3 | 0.29 | 6.2 | 13.4 | 0.46 |
| `65:8:64 bf16` | 7.3 | 55.0 | 0.13 | 11.4 | 31.3 | 0.36 | 13.1 | 35.8 | 0.37 |
| `65:8:64 fp32` | 10.1 | 44.1 | 0.23 | 16.9 | 49.1 | 0.34 | 20.9 | 52.2 | 0.40 |
| `129:8:64 bf16` | 18.4 | 160.8 | 0.11 | 22.3 | 67.5 | 0.33 | 23.9 | 63.9 | 0.37 |
| `129:8:64 fp32` | 27.8 | 130.9 | 0.21 | 45.9 | 128.7 | 0.36 | 48.8 | 144.2 | 0.34 |
| `257:8:64 bf16` | 58.2 | 484.2 | 0.12 | 61.7 | 179.6 | 0.34 | 58.7 | 192.8 | 0.30 |
| `257:8:64 fp32` | 93.6 | 419.1 | 0.22 | 155.2 | 424.7 | 0.37 | 125.5 | 408.2 | 0.31 |
| `513:8:64 bf16` | 205.4 | 1715.8 | 0.12 | 214.7 | 623.8 | 0.34 | 166.5 | 449.8 | 0.37 |
| `513:8:64 fp32` | 340.0 | 1549.5 | 0.22 | 578.5 | 1589.8 | 0.36 | 359.1 | 1471.3 | 0.24 |
| `1025:8:64 bf16` | 783.0 | 4674.5 | 0.17 | 808.5 | 1661.9 | 0.49 | 532.5 | 1335.7 | 0.40 |
| `1025:8:64 fp32` | 1315.0 | 4342.2 | 0.30 | 2204.1 | 4093.4 | 0.54 | 1159.7 | 3646.1 | 0.32 |
| `2049:8:64 bf16` | 3065.2 | 15256.7 | 0.20 | 3106.8 | 5339.6 | 0.58 | 1825.7 | 4068.9 | 0.45 |
| `2049:8:64 fp32` | 5199.0 | 14410.2 | 0.36 | 8595.7 | 12592.8 | 0.68 | 4133.9 | 11880.2 | 0.35 |
| `4097:8:64 bf16` | 12086.8 | 54336.8 | 0.22 | 12227.3 | 19294.4 | 0.63 | 6777.9 | 14259.2 | 0.48 |
| `4097:8:64 fp32` | 22105.3 | 52114.8 | 0.42 | 34511.9 | 42532.9 | 0.81 | 15588.3 | 41349.9 | 0.38 |
| `129:16:128 bf16` | 69.0 | 632.2 | 0.11 | 65.4 | 191.1 | 0.34 | 82.6 | 174.1 | 0.47 |
| `129:16:128 fp32` | 110.1 | 461.5 | 0.24 | 162.0 | 404.4 | 0.40 | 185.4 | 530.9 | 0.35 |
| `257:16:128 bf16` | 220.6 | 1906.2 | 0.12 | 203.2 | 574.5 | 0.35 | 209.6 | 588.4 | 0.36 |
| `257:16:128 fp32` | 377.6 | 1557.9 | 0.24 | 558.9 | 1365.0 | 0.41 | 446.9 | 1550.7 | 0.29 |
| `1025:16:128 bf16` | 3129.7 | 18383.2 | 0.17 | 2666.3 | 5348.4 | 0.50 | 1735.3 | 3832.1 | 0.45 |
| `1025:16:128 fp32` | 6032.9 | 16626.4 | 0.36 | 8043.9 | 13851.8 | 0.58 | 4192.8 | 12638.9 | 0.33 |
| `2049:16:128 bf16` | 12001.5 | 58842.9 | 0.20 | 10312.3 | 17666.9 | 0.58 | 5990.3 | 12633.1 | 0.47 |
| `2049:16:128 fp32` | 20781.8 | 54630.6 | 0.38 | 31630.8 | 43390.1 | 0.73 | 15083.3 | 42406.2 | 0.36 |
| `1025:8:32 bf16` | 432.4 | 2732.9 | 0.16 | 540.7 | 1184.2 | 0.46 | 413.0 | 1096.8 | 0.38 |
| `1025:8:32 fp32` | 710.8 | 2464.6 | 0.29 | 1333.2 | 2760.1 | 0.48 | 736.8 | 2279.9 | 0.32 |
| `1x129:8:64 bf16` | 3.5 | 34.7 | 0.10 | 4.9 | 21.4 | 0.23 | 6.9 | 13.8 | 0.50 |
| `1x129:8:64 fp32` | 3.9 | 10.0 | 0.39 | 7.1 | 23.7 | 0.30 | 5.9 | 14.1 | 0.42 |
| `17x301:8:64 bf16` | 14.4 | 57.2 | 0.25 | 13.8 | 47.8 | 0.29 | 16.3 | 41.7 | 0.39 |
| `17x301:8:64 fp32` | 19.4 | 55.2 | 0.35 | 32.1 | 62.1 | 0.52 | 37.9 | 64.5 | 0.59 |
| `127x129:8:64 bf16` | 18.4 | 147.9 | 0.12 | 21.1 | 91.3 | 0.23 | 24.3 | 98.3 | 0.25 |
| `127x129:8:64 fp32` | 27.6 | 144.9 | 0.19 | 45.3 | 152.5 | 0.30 | 42.4 | 160.5 | 0.26 |
| `129x127:8:64 bf16` | 18.8 | 194.7 | 0.10 | 20.5 | 92.7 | 0.22 | 22.9 | 99.5 | 0.23 |
| `129x127:8:64 fp32` | 27.3 | 165.4 | 0.17 | 43.8 | 154.2 | 0.28 | 40.6 | 171.6 | 0.24 |
| `257x513:8:64 bf16` | 153.2 | 1015.7 | 0.15 | 155.5 | 387.3 | 0.40 | 112.2 | 323.0 | 0.35 |
| `257x513:8:64 fp32` | 262.9 | 861.1 | 0.31 | 424.1 | 798.0 | 0.53 | 249.4 | 860.8 | 0.29 |
| `513x257:8:64 bf16` | 58.7 | 924.2 | 0.06 | 60.9 | 368.6 | 0.17 | 73.6 | 363.3 | 0.20 |
| `513x257:8:64 fp32` | 95.2 | 871.4 | 0.11 | 152.1 | 775.0 | 0.20 | 137.8 | 845.5 | 0.16 |
| `1025x2049:8:64 bf16` | 2286.0 | 13114.0 | 0.17 | 2322.4 | 5015.4 | 0.46 | 1341.5 | 4402.3 | 0.30 |
| `1025x2049:8:64 fp32` | 3878.1 | 12509.4 | 0.31 | 6461.7 | 11365.8 | 0.57 | 3010.6 | 10838.3 | 0.28 |
| `2049x1025:8:64 bf16` | 790.6 | 12584.7 | 0.06 | 809.0 | 4658.0 | 0.17 | 605.3 | 3585.8 | 0.17 |
| `2049x1025:8:64 fp32` | 1400.8 | 12070.2 | 0.12 | 2218.7 | 10701.1 | 0.21 | 1334.0 | 10419.6 | 0.13 |
| `1024:8:64 noncausal bf16` | 1510.2 | 5913.3 | 0.26 | 1538.9 | 2167.0 | 0.71 | 853.4 | 1650.5 | 0.52 |
| `1024:8:64 noncausal fp32` | 2534.6 | 5707.8 | 0.44 | 4269.6 | 4459.6 | 0.96 | 1891.5 | 4631.1 | 0.41 |
| `1025:8:64 noncausal bf16` | 1538.2 | 6302.7 | 0.24 | 1572.1 | 2222.0 | 0.71 | 923.2 | 1728.9 | 0.53 |
| `1025:8:64 noncausal fp32` | 2606.9 | 5794.6 | 0.45 | 4327.5 | 4526.8 | 0.96 | 2043.9 | 4873.9 | 0.42 |
| `1024:16:128 noncausal bf16` | 5784.1 | 21979.5 | 0.26 | 5060.2 | 6975.2 | 0.73 | 2751.0 | 4753.8 | 0.58 |
| `1024:16:128 noncausal fp32` | 12562.7 | 21783.1 | 0.58 | 15502.9 | 16100.6 | 0.96 | 6482.4 | 16404.6 | 0.40 |
| `1024:32/8:128 bf16` | 5999.2 | 33084.4 | 0.18 | 5228.0 | 10357.6 | 0.50 | 3178.5 | 6881.7 | 0.46 |
| `1024:32/8:128 fp32` | 10980.2 | 32507.0 | 0.34 | 15937.7 | 27050.6 | 0.59 | 7563.1 | 24813.5 | 0.30 |
| `1024:16/1:64 bf16` | 1556.7 | 8853.9 | 0.18 | 1600.1 | 3220.3 | 0.50 | 982.2 | 2480.6 | 0.40 |
| `1024:16/1:64 fp32` | 2629.2 | 8571.6 | 0.31 | 4400.8 | 8030.4 | 0.55 | 1978.3 | 6887.7 | 0.29 |
| `1024:8:64 fused bf16` | 777.2 | 4458.2 | 0.17 | 797.2 | 1663.0 | 0.48 | 531.6 | 1295.6 | 0.41 |
| `1024:8:64 fused fp32` | 1311.5 | 4313.8 | 0.30 | 2191.7 | 4079.3 | 0.54 | 1056.7 | 3630.4 | 0.29 |
| `257:16:128 fused bf16` | 218.3 | 1904.9 | 0.11 | 202.0 | 573.8 | 0.35 | 212.7 | 624.4 | 0.34 |
| `257:16:128 fused fp32` | 385.0 | 1570.3 | 0.25 | 558.0 | 1365.1 | 0.41 | 460.9 | 1589.4 | 0.29 |

</details>

<details><summary>decode, allocating the output: µs per call, ours / vLLM (60 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `engine self b=1 ctx=256 bf16` | 3.8 | 7.1 | 0.53 | 6.6 | 10.1 | 0.65 | 6.1 | 12.2 | 0.50 |
| `engine self b=1 ctx=256 fp32` | 6.4 | 8.9 | 0.72 | 9.7 | 14.8 | 0.66 | 8.2 | 18.3 | 0.45 |
| `engine cross b=1 ctx=256 bf16` | 3.7 | 5.7 | 0.66 | 6.0 | 7.9 | 0.75 | 5.7 | 7.2 | 0.80 |
| `engine cross b=1 ctx=256 fp32` | 6.2 | 7.1 | 0.88 | 8.4 | 12.1 | 0.69 | 7.8 | 12.7 | 0.62 |
| `engine self b=8 ctx=256 bf16` | 29.1 | 47.4 | 0.61 | 32.8 | 79.0 | 0.41 | 79.6 | 111.0 | 0.72 |
| `engine self b=8 ctx=256 fp32` | 311.3ᶜ | 294.4 | 1.06 ▲ | 83.7 | 142.7 | 0.59 | 149.8 | 166.3 | 0.90 |
| `engine cross b=8 ctx=256 bf16` | 32.3 | 44.3 | 0.73 | 32.0 | 81.9 | 0.39 | 77.6 | 101.6 | 0.76 |
| `engine cross b=8 ctx=256 fp32` | 315.1ᶜ | 313.5 | 1.01 | 88.2 | 140.8 | 0.63 | 149.4 | 149.8 | 1.00 |
| `engine self b=8 ctx=1024 bf16` | 633.9ᶜ | 684.9 | 0.93 | 345.2ᶜ | 390.9 | 0.88 | 355.8 | 411.2 | 0.87 |
| `engine self b=8 ctx=1024 fp32` | 1258.0ᶜ | 1251.8 | 1.00 | 669.6ᶜ | 739.0 | 0.91 | 801.8 | 957.2 | 0.84 |
| `engine cross b=8 ctx=1024 bf16` | 636.3ᶜ | 682.6 | 0.93 | 354.3ᶜ | 390.0 | 0.91 | 384.8 | 416.1 | 0.92 |
| `engine cross b=8 ctx=1024 fp32` | 1259.2ᶜ | 1250.1 | 1.01 | 694.0ᶜ | 746.2 | 0.93 | 837.2 | 1029.0 | 0.81 |
| `engine self b=32 ctx=1024 bf16` | 2551.7ᶜ | 2883.0 | 0.89 | 1516.4ᶜ | 1765.4 | 0.86 | 3095.0ᶜ | 2932.7 | 1.06 ▲ |
| `engine self b=32 ctx=1024 fp32` | 5049.5ᶜ | 5129.6 | 0.98 | 3003.5ᶜ | 3277.3 | 0.92 | 5384.1ᶜ | 5286.4 | 1.02 |
| `engine cross b=32 ctx=1024 bf16` | 2561.9ᶜ | 2837.6 | 0.90 | 1485.3ᶜ | 1780.9 | 0.83 | 3155.5ᶜ | 2893.5 | 1.09 ▲ |
| `engine cross b=32 ctx=1024 fp32` | 5047.9ᶜ | 5063.6 | 1.00 | 2823.7ᶜ | 3259.0 | 0.87 | 5097.1ᶜ | 5041.2 | 1.01 |
| `engine self b=1 ctx=4096 bf16` | 314.0ᶜ | 336.7 | 0.93 | 128.5 | 149.1 | 0.86 | 178.7 | 197.8 | 0.90 |
| `engine self b=1 ctx=4096 fp32` | 629.7ᶜ | 612.1 | 1.03 | 342.7ᶜ | 322.5 | 1.06 ▲ | 345.1 | 361.3 | 0.96 |
| `engine cross b=1 ctx=4096 bf16` | 316.8ᶜ | 331.6 | 0.96 | 132.0 | 144.1 | 0.92 | 182.4 | 196.3 | 0.93 |
| `engine cross b=1 ctx=4096 fp32` | 633.3ᶜ | 615.1 | 1.03 | 339.6ᶜ | 330.0 | 1.03 | 347.2 | 350.6 | 0.99 |
| `engine self b=8 ctx=255 bf16` | 36.3 | 51.9 | 0.70 | 35.6 | 73.3 | 0.49 | 82.0 | 113.9 | 0.72 |
| `engine self b=8 ctx=255 fp32` | 312.7ᶜ | 303.8 | 1.03 | 79.3 | 134.2 | 0.59 | 152.0 | 164.8 | 0.92 |
| `engine cross b=8 ctx=255 bf16` | 32.1 | 44.9 | 0.72 | 33.7 | 70.4 | 0.48 | 81.7 | 104.1 | 0.78 |
| `engine cross b=8 ctx=255 fp32` | 313.7ᶜ | 294.9 | 1.06 ▲ | 81.0 | 136.9 | 0.59 | 149.1 | 151.5 | 0.98 |
| `engine self b=8 ctx=257 bf16` | 167.9ᶜ | 203.8 | 0.82 | 34.7 | 74.9 | 0.46 | 84.1 | 123.7 | 0.68 |
| `engine self b=8 ctx=257 fp32` | 326.9ᶜ | 341.7 | 0.96 | 87.0 | 132.4 | 0.66 | 158.5 | 183.6 | 0.86 |
| `engine cross b=8 ctx=257 bf16` | 167.3ᶜ | 197.0 | 0.85 | 33.3 | 78.0 | 0.43 | 85.9 | 112.5 | 0.76 |
| `engine cross b=8 ctx=257 fp32` | 327.5ᶜ | 351.8 | 0.93 | 87.1 | 136.2 | 0.64 | 156.7 | 170.0 | 0.92 |
| `engine self b=8 ctx=1023 bf16` | 635.3ᶜ | 698.3 | 0.91 | 339.1ᶜ | 398.8 | 0.85 | 351.2 | 399.4 | 0.88 |
| `engine self b=8 ctx=1023 fp32` | 1231.0ᶜ | 1219.9 | 1.01 | 688.0ᶜ | 756.4 | 0.91 | 768.5 | 970.1 | 0.79 |
| `engine cross b=8 ctx=1023 bf16` | 636.3ᶜ | 675.7 | 0.94 | 361.8ᶜ | 401.5 | 0.90 | 355.8 | 389.9 | 0.91 |
| `engine cross b=8 ctx=1023 fp32` | 1240.0ᶜ | 1216.0 | 1.02 | 674.4ᶜ | 746.7 | 0.90 | 765.9 | 962.6 | 0.80 |
| `engine self b=8 ctx=1025 bf16` | 640.5ᶜ | 713.6 | 0.90 | 373.2ᶜ | 408.4 | 0.91 | 362.1 | 415.0 | 0.87 |
| `engine self b=8 ctx=1025 fp32` | 1268.7ᶜ | 1284.5 | 0.99 | 675.2ᶜ | 765.7 | 0.88 | 791.5 | 931.8 | 0.85 |
| `engine cross b=8 ctx=1025 bf16` | 643.3ᶜ | 708.8 | 0.91 | 356.7ᶜ | 393.4 | 0.91 | 369.2 | 402.2 | 0.92 |
| `engine cross b=8 ctx=1025 fp32` | 1271.0ᶜ | 1288.0 | 0.99 | 666.1ᶜ | 750.5 | 0.89 | 807.1 | 893.9 | 0.90 |
| `engine self b=1 ctx=4097 bf16` | 313.6ᶜ | 333.4 | 0.94 | 132.8 | 149.9 | 0.89 | 173.6 | 204.9 | 0.85 |
| `engine self b=1 ctx=4097 fp32` | 635.8ᶜ | 623.2 | 1.02 | 338.8ᶜ | 320.0 | 1.06 ▲ | 330.5 | 337.3 | 0.98 |
| `engine cross b=1 ctx=4097 bf16` | 314.2ᶜ | 328.0 | 0.96 | 107.4 | 127.9 | 0.84 | 177.1 | 201.7 | 0.88 |
| `engine cross b=1 ctx=4097 fp32` | 629.5ᶜ | 616.8 | 1.02 | 333.0ᶜ | 325.4 | 1.02 | 326.3 | 333.5 | 0.98 |
| `engine self b=8 ctx=1024 page=16 bf16` | 588.2ᶜ | - | - | 347.8ᶜ | 416.4 | 0.84 | 338.0 | - | - |
| `engine self b=8 ctx=1024 page=16 fp32` | 1171.0ᶜ | - | - | 668.4ᶜ | 756.2 | 0.88 | 739.4 | - | - |
| `engine self b=8 ctx=1024 page=48 bf16` | 607.4ᶜ | - | - | 256.5ᶜ | 309.8 | 0.83 | 308.1 | - | - |
| `engine self b=8 ctx=1024 page=48 fp32` | 1165.2ᶜ | - | - | 477.0ᶜ | 617.2 | 0.77 | 746.6 | - | - |
| `engine self b=8 ctx=1024 page=64 bf16` | 636.5ᶜ | 684.2 | 0.93 | 270.4ᶜ | 316.5 | 0.85 | 334.5 | 352.4 | 0.95 |
| `engine self b=8 ctx=1024 page=64 fp32` | 1249.0ᶜ | 1256.1 | 0.99 | 493.4ᶜ | 613.1 | 0.80 | 821.1 | 1029.4 | 0.80 |
| `engine self b=8 ctx=1024 h=16/4 bf16` | 45.1 | 75.5 | 0.60 | 75.9 | 69.6 | 1.09 ▲ | 121.4 | 105.7 | 1.15 ▲ |
| `engine self b=8 ctx=1024 h=16/4 fp32` | 321.3ᶜ | 328.7 | 0.98 | 110.9 | 139.5 | 0.80 | 182.6 | 208.3 | 0.88 |
| `engine cross b=8 ctx=1024 h=16/4 bf16` | 46.4 | 68.8 | 0.67 | 77.7 | 64.5 | 1.20 ▲ | 117.4 | 96.8 | 1.21 ▲ |
| `engine cross b=8 ctx=1024 h=16/4 fp32` | 318.6ᶜ | 322.7 | 0.99 | 112.6 | 137.7 | 0.82 | 182.4 | 195.9 | 0.93 |
| `engine cross b=8 ctx=1024 h=16/1 bf16` | 34.4 | 108.2 | 0.32 | 67.3 | 37.0 | 1.82 ▲ | 74.6 | 34.9 | 2.14 ▲ |
| `engine cross b=8 ctx=1024 h=16/1 fp32` | 43.2 | 140.4 | 0.31 | 84.8 | 80.9 | 1.05 ▲ | 82.0 | 205.1 | 0.40 |
| `engine self b=8 ctx=het bf16` | 408.5ᶜ | 453.1 | 0.90 | 304.7 | 196.3 | 1.55 ▲ | 328.4 | 259.3 | 1.27 ▲ |
| `engine self b=8 ctx=het fp32` | 778.6ᶜ | 788.8 | 0.99 | 1221.1ᶜ | 456.9 | 2.67 ▲ | 664.8 | 466.1 | 1.43 ▲ |
| `engine cross b=8 ctx=het bf16` | 403.5ᶜ | 435.8 | 0.93 | 297.8 | 188.9 | 1.58 ▲ | 318.4 | 251.2 | 1.27 ▲ |
| `engine cross b=8 ctx=het fp32` | 777.6ᶜ | 780.7 | 1.00 | 1259.2ᶜ | 460.0 | 2.74 ▲ | 712.1 | 491.9 | 1.45 ▲ |
| `engine cross b=8 ctx=1024 d=32 bf16` | 323.0ᶜ | 346.2 | 0.93 | 82.9 | 111.4 | 0.74 | 174.2 | 183.0 | 0.95 |
| `engine cross b=8 ctx=1024 d=32 fp32` | 633.4ᶜ | 619.2 | 1.02 | 364.4ᶜ | 394.3 | 0.92 | 324.8 | 316.0 | 1.03 |
| `engine cross b=8 ctx=1024 d=128 bf16` | 627.0ᶜ | 651.7 | 0.96 | 251.9ᶜ | 356.8 | 0.71 | 333.9 | 354.0 | 0.94 |
| `engine cross b=8 ctx=1024 d=128 fp32` | 1249.9ᶜ | 1220.8 | 1.02 | 540.1ᶜ | 598.3 | 0.90 | 756.1 | 900.3 | 0.84 |

</details>

<details><summary>decode with `out=`: µs per call, ours / vLLM (20 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `engine_out self b=1 ctx=256 bf16` | 3.3 | 7.2 | 0.46 | 5.5 | 9.5 | 0.58 | 5.7 | 12.3 | 0.47 |
| `engine_out self b=1 ctx=256 fp32` | 6.0 | 8.7 | 0.70 | 8.3 | 15.2 | 0.54 | 7.2 | 17.2 | 0.42 |
| `engine_out cross b=1 ctx=256 bf16` | 3.4 | 5.6 | 0.60 | 5.1 | 6.6 | 0.76 | 5.1 | 6.7 | 0.77 |
| `engine_out cross b=1 ctx=256 fp32` | 5.8 | 7.4 | 0.79 | 9.0 | 10.8 | 0.83 | 6.9 | 12.8 | 0.54 |
| `engine_out self b=8 ctx=256 bf16` | 32.0 | 48.8 | 0.66 | 30.8 | 78.3 | 0.39 | 76.8 | 110.7 | 0.69 |
| `engine_out self b=8 ctx=256 fp32` | 313.5ᶜ | 308.3 | 1.02 | 85.8 | 148.5 | 0.58 | 147.7 | 163.6 | 0.90 |
| `engine_out cross b=8 ctx=256 bf16` | 31.3 | 44.1 | 0.71 | 31.9 | 75.7 | 0.42 | 76.7 | 100.6 | 0.76 |
| `engine_out cross b=8 ctx=256 fp32` | 312.6ᶜ | 312.3 | 1.00 | 82.0 | 134.4 | 0.61 | 146.4 | 152.3 | 0.96 |
| `engine_out self b=8 ctx=1024 bf16` | 633.3ᶜ | 677.2 | 0.94 | 348.7ᶜ | 394.8 | 0.88 | 346.0 | 405.3 | 0.85 |
| `engine_out self b=8 ctx=1024 fp32` | 1256.8ᶜ | 1255.8 | 1.00 | 674.4ᶜ | 764.5 | 0.88 | 787.2 | 905.9 | 0.87 |
| `engine_out cross b=8 ctx=1024 bf16` | 632.7ᶜ | 692.6 | 0.91 | 360.1ᶜ | 402.9 | 0.89 | 377.1 | 421.4 | 0.89 |
| `engine_out cross b=8 ctx=1024 fp32` | 1254.5ᶜ | 1242.9 | 1.01 | 680.9ᶜ | 746.0 | 0.91 | 780.2 | 938.3 | 0.83 |
| `engine_out self b=32 ctx=1024 bf16` | 2548.0ᶜ | 2869.2 | 0.89 | 1510.2ᶜ | 1769.3 | 0.85 | 3123.2ᶜ | 2901.0 | 1.08 ▲ |
| `engine_out self b=32 ctx=1024 fp32` | 5045.0ᶜ | 5115.4 | 0.99 | 2930.9ᶜ | 3171.3 | 0.92 | 5254.9ᶜ | 5076.8 | 1.04 ▲ |
| `engine_out cross b=32 ctx=1024 bf16` | 2554.5ᶜ | 2830.2 | 0.90 | 1437.0ᶜ | 1806.2 | 0.80 | 3033.2ᶜ | 2841.1 | 1.07 ▲ |
| `engine_out cross b=32 ctx=1024 fp32` | 5036.5ᶜ | 5070.2 | 0.99 | 2850.8ᶜ | 3166.3 | 0.90 | 5207.2ᶜ | 4958.4 | 1.05 ▲ |
| `engine_out self b=1 ctx=4096 bf16` | 312.7ᶜ | 336.0 | 0.93 | 130.7 | 148.3 | 0.88 | 175.4 | 205.3 | 0.85 |
| `engine_out self b=1 ctx=4096 fp32` | 629.9ᶜ | 614.9 | 1.02 | 329.5ᶜ | 318.0 | 1.04 ▲ | 327.7 | 337.8 | 0.97 |
| `engine_out cross b=1 ctx=4096 bf16` | 311.6ᶜ | 330.6 | 0.94 | 132.4 | 146.0 | 0.91 | 178.4 | 195.5 | 0.91 |
| `engine_out cross b=1 ctx=4096 fp32` | 630.1ᶜ | 607.8 | 1.04 | 329.4ᶜ | 325.1 | 1.01 | 327.5 | 334.3 | 0.98 |

</details>

<details><summary>public `flash_attn_with_kvcache` decode: µs per call, ours / vLLM (8 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `api self b=1 ctx=256 bf16` | 4.7 | 7.3 | 0.64 | 10.0 | 12.3 | 0.82 | 7.5 | 11.4 | 0.66 |
| `api self b=1 ctx=256 fp32` | 7.3 | 9.0 | 0.81 | 13.9 | 18.1 | 0.77 | 10.1 | 18.0 | 0.56 |
| `api cross b=1 ctx=256 bf16` | 4.6 | 5.8 | 0.79 | 9.4 | 7.7 | 1.22 ▲ | 6.9 | 6.9 | 1.00 |
| `api cross b=1 ctx=256 fp32` | 7.4 | 7.3 | 1.01 | 13.0 | 11.3 | 1.15 ▲ | 9.0 | 13.7 | 0.66 |
| `api self b=8 ctx=1024 bf16` | 642.8ᶜ | 697.9 | 0.92 | 365.6ᶜ | 389.2 | 0.94 | 349.9 | 395.3 | 0.89 |
| `api self b=8 ctx=1024 fp32` | 1265.2ᶜ | 1261.7 | 1.00 | 686.8ᶜ | 747.8 | 0.92 | 824.8 | 1006.7 | 0.82 |
| `api cross b=8 ctx=1024 bf16` | 640.2ᶜ | 658.6 | 0.97 | 378.5ᶜ | 392.0 | 0.97 | 355.3 | 359.3 | 0.99 |
| `api cross b=8 ctx=1024 fp32` | 1262.9ᶜ | 1253.1 | 1.01 | 686.3ᶜ | 729.3 | 0.94 | 807.1 | 956.7 | 0.84 |

</details>

<details><summary>store-only (`reshape_and_cache`): µs per call, ours / vLLM (4 rows)</summary>

| row | C4D ours | C4D vLLM | C4D ratio | Axion ours | Axion vLLM | Axion ratio | Xeon ours | Xeon vLLM | Xeon ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `store store b=32 ctx=1 bf16` | 3.4 | 5.5 | 0.61 | 4.5 | 4.1 | 1.12 ▲ | 8.7 | 9.2 | 0.95 |
| `store store b=32 ctx=1 fp32` | 11.4 | 10.8 | 1.05 | 6.5 | 6.8 | 0.96 | 18.3 | 20.5 | 0.89 |
| `store store b=1 ctx=1024 bf16` | 30.1 | 114.0 | 0.26 | 54.5 | 58.7 | 0.93 | 77.1 | 236.0 | 0.33 |
| `store store b=1 ctx=1024 fp32` | 94.1 | 203.4 | 0.46 | 72.7 | 113.3 | 0.64 | 170.1 | 420.9 | 0.40 |

</details>

**Where we do not win.**
* **GQA/MQA decode** (`h=16/4`, `h=16/1`, bf16) on the Axion and the Xeon: vLLM is
  up to 2.1x faster on MQA (`cross b=8 ctx=1024 h=16/1` bf16: 67.3 vs 37.0 us on the
  Axion, 74.6 vs 34.9 us on the Xeon). On the C4D these rows win or tie.
* **Heterogeneous contexts** (`ctx=het`: 8 sequences of 17 ... 2047 tokens): 1.27-1.45x
  slower on the Xeon and 1.55-2.74x on the Axion (fp32 `self` 1221 vs 457 us), and at
  parity on the C4D. On the Axion this row takes longer than `b=8 ctx=1024`, which has 1.6x
  its tokens, so the split of uneven work across threads is the suspect. vLLM's bf16
  output here is 1.8-2.3e-3 above the bf16 rounding, outside our 1e-3 gate but inside
  the reference sanity gate (3e-2), so these rows are scored.
* **Xeon `b=32 ctx=1024`** decode: 1.04-1.09. This VM reports its socket's 260 MB L3,
  so only 8 Xeon decode rows cross the cold threshold; the rest are timed warm.
* **Axion**, smaller ones: batch-1 context 4096 fp32 (1.04-1.06), the public API at
  `b=1 ctx=256` (1.15-1.22; 9.4 vs 7.7 us in bf16), and store-only `b=32 ctx=1` bf16
  (4.5 vs 4.1 us). **C4D**: two fp32 `b=8 ctx~256` rows at 1.06.
* **No reference**: vLLM on x86 takes only page sizes that are a multiple of 32, so
  the four `page=16` / `page=48` decode rows have no vLLM time on the C4D and the Xeon.

**Caveats.**
* `flash_attn_func` against vLLM's prefill path is in the JSON (`new/vllm`, 62
  batch-1 equal-length rows): 0.44-1.19 on the C4D, 0.60-0.93 on the Axion, and
  0.35-0.89 on the Xeon.
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
