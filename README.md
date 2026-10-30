# cpu-attention

Paged and dense attention kernels for CPUs, with flash-attn's API. One source
tree builds for x86-64 (AVX-512, AMX) and aarch64 (NEON + BF16). It is the
attention backend of [fast-encdec](https://github.com/ducviet00/fast-encdec)
and works as a standalone library.

* `flash_attn_with_kvcache`: paged decode (self- and cross-attention), one
  query token per sequence.
* `flash_attn_func`: dense attention over whole prompts, causal (bottom-right
  aligned) or not.
* fp32 and bf16, MHA, head_dim 32/64/128, page size a multiple of 16. GQA/MQA
  are not supported yet (`NotImplementedError`).
* The API is flash-attn's: `cpu_attention` exports flash-attn's functions (the
  packed and varlen ones raise `NotImplementedError`), plus `reshape_and_cache`
  with vLLM's signature to write the paged cache.
* One kernel, written in Mojo (`cpu_attention/_mojo/`,
  [`docs/Mojo.md`](docs/Mojo.md)): one driver per op for every CPU, a target
  differing only by compile-time constants and a trait-plugged inner kernel.
* Linux, PyTorch, Python >= 3.10.
* Not supported, and rejected with `NotImplementedError` rather than silently
  ignored: rotary, ALiBi, softcap, sliding windows, varlen/packed QKV,
  dropout, FP8, backward.

## Install and build

**From another project.** Installing the package builds the kernel (one
`_C_<tier>.so` per ISA tier, about a minute) against the project's own torch, and
bundles the three small Mojo runtime libraries it loads, so the project needs no
Mojo afterwards. The build needs Linux and a C toolchain (`build-essential`);
the Mojo compiler is fetched for the build only. With uv:

```toml
[project]
dependencies = ["torch", "cpu-attention"]

[tool.uv.sources]
cpu-attention = { git = "https://github.com/ducviet00/cpu-attention.git" }

# Build against the torch the project runs with: the extension links it.
[tool.uv.extra-build-dependencies]
cpu-attention = [{ requirement = "torch", match-runtime = true }]
```

With pip, in an environment that already has torch:

```bash
pip install mojo==1.2.0
pip install --no-build-isolation git+https://github.com/ducviet00/cpu-attention.git
```

**For development** (this repository):

```bash
uv sync --extra build --extra test                       # torch, the mojo wheel, pytest
uv run --extra build python scripts/build.py --tests     # one _C_<tier>.so per ISA tier
```

The build writes one shared object per ISA tier for the host architecture
(x86-64: AVX-512-BF16 + AMX, AVX2, baseline; aarch64: BF16, NEON), each compiled
for an explicit target rather than the build host. The loader picks the best
tier the CPU reports; `CPU_ATTENTION_ISA=amx|vec|avx2|scalar|neon` forces one
and is refused if the CPU lacks it.

## Use

```python
from cpu_attention import flash_attn_func, flash_attn_with_kvcache, reshape_and_cache

# Prefill: q/k/v [batch, seq, heads, head_dim]
out = flash_attn_func(q, k, v, causal=True)

# Stage prompts' K/V into the paged cache: token i goes to slot_mapping[i]
# (page slot // page_size, position slot % page_size; negative = skip)
reshape_and_cache(k_tokens, v_tokens, k_cache, v_cache, slot_mapping)  # [tokens, kv_heads, D]
# ... then decode, one query token per sequence
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,      # q [B, 1, H, D]; caches [blocks, kv_heads, page, D]
    k=k_new, v=v_new,         # appended at cache_seqlens (the length before the write)
    cache_seqlens=lens,       # int or int32 [B]
    page_table=pages,         # int32 [B, max_blocks]
)
```

Notes on the API:
* **Cache layout.** Head-major `[num_blocks, nheads_k, page, head_dim]` with K
  stored transposed inside the page, so it is written only by this library:
  `reshape_and_cache`, or `flash_attn_with_kvcache` with `k`/`v` (any number of
  new tokens, appended at `cache_seqlens`). A cache in another library's layout
  (vLLM's included) is not readable.
* **Strides.** q/k/v may be strided chunks of a fused projection: any token
  stride, with heads and dims contiguous.
* **`causal` on the paged path** is accepted and has no effect, because one
  query at the last position sees every key.
* **Engines.** [`examples/ops.py`](examples/ops.py) shows an inference engine's
  per-layer calls (fast-encdec's): its own slot mapping and a preallocated
  `out=`, through the same native entry.

## Performance

Ours against **torch SDPA** (prefill) and **vLLM's CPU attention** (paged
decode, `csrc/cpu` at the commit pinned in `scripts/build_vllm_ref.sh`), timed at
each library's Python API with L2/L3 flushed before every call: the median of 10
calls after 3 warmups. Each cell is ours / reference in µs and the **ratio**
(below 1: we are faster). Paged rows use each library's best page size; ᵖ¹⁶
marks rows where our page 16 beat page 32. Threads = physical cores; vLLM runs
AMX for bf16 on c4-standard-*, VEC for other x86 cases and NEON on c4a. GCP spot
VMs, `f074b2e`, Mojo 1.2.0 (2016d0cd), torch 2.14.1+cpu, Oct 2026. MHA shapes
(GQA/MQA are not supported yet).
[`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md) has the method and every
arm's median, min and max.

**Prefill, `flash_attn_func`, bf16**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| short prompt | 162 / 389 **0.42** | 63.5 / 159 **0.40** | 61.9 / 165 **0.38** | 38.3 / 98.1 **0.39** | 82.4 / 467 **0.18** | 52.2 / 218 **0.24** |
| prompt 1k | 1,422 / 3,997 **0.36** | 651 / 2,586 **0.25** | 1,660 / 3,353 **0.50** | 857 / 1,741 **0.49** | 1,954 / 9,385 **0.21** | 925 / 5,006 **0.18** |
| prompt 2k, d128 | 18,530 / 32,560 **0.57** | 7,016 / 14,032 **0.50** | 20,890 / 35,737 **0.58** | 10,480 / 17,740 **0.59** | 24,550 / 112,789 **0.22** | 15,497 / 61,039 **0.25** |
| long prompt 4k, d128 | 58,273 / 103,359 **0.56** | 38,669 / 58,327 **0.66** | 85,191 / 126,530 **0.67** | 42,543 / 63,312 **0.67** | 99,072 / 415,749 **0.24** | 55,910 / 218,106 **0.26** |
| encoder batch | 4,527 / 7,966 **0.57** | 2,295 / 4,114 **0.56** | 6,268 / 9,412 **0.67** | 3,195 / 4,780 **0.67** | 6,577 / 26,153 **0.25** | 3,451 / 13,266 **0.26** |
| encoder 1500 (Whisper) | 4,972 / 10,175 **0.49** | 2,539 / 4,341 **0.58** | 8,415 / 11,803 **0.71** | 4,382 / 6,234 **0.70** | 8,235 / 32,776 **0.25** | 4,480 / 17,282 **0.26** |
| encoder, d32 | 1,497 / 2,593 **0.58** | 619 / 1,089 **0.57** | 1,149 / 1,780 **0.65** | 579 / 924 **0.63** | 932 / 3,954 **0.24** | 508 / 2,065 **0.25** |

**Prefill, `flash_attn_func`, fp32**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| short prompt | 261 / 450 **0.58** | 121 / 258 **0.47** | 114 / 367 **0.31** | 63.9 / 209 **0.31** | 97.8 / 315 **0.31** | 86.3 / 209 **0.41** |
| prompt 1k | 3,054 / 8,080 **0.38** | 1,289 / 3,566 **0.36** | 4,624 / 9,299 **0.50** | 2,291 / 4,698 **0.49** | 3,205 / 9,181 **0.35** | 1,582 / 4,803 **0.33** |
| prompt 2k, d128 | 44,370 / 91,146 **0.49** | 28,208 / 42,717 **0.66** | 63,793 / 86,867 **0.73** | 31,974 / 43,571 **0.73** | 46,594 / 110,714 **0.42** | 28,609 / 59,271 **0.48** |
| long prompt 4k, d128 | 144,231 / 308,944 **0.47** | 75,628 / 137,846 **0.55** | 252,032 / 302,249 **0.83** | 127,509 / 156,954 **0.81** | 177,449 / 399,327 **0.44** | 93,138 / 206,086 **0.45** |
| encoder batch | 9,572 / 29,052 **0.33** | 5,860 / 13,307 **0.44** | 17,311 / 19,443 **0.89** | 8,844 / 9,796 **0.90** | 11,666 / 26,491 **0.44** | 6,267 / 13,574 **0.46** |
| encoder 1500 (Whisper) | 10,818 / 26,276 **0.41** | 5,541 / 11,655 **0.48** | 23,545 / 24,544 **0.96** | 12,194 / 12,991 **0.94** | 15,123 / 31,845 **0.47** | 7,365 / 16,281 **0.45** |
| encoder, d32 | 2,311 / 4,236 **0.55** | 1,155 / 1,952 **0.59** | 2,722 / 3,022 **0.90** | 1,382 / 1,565 **0.88** | 1,700 / 3,967 **0.43** | 866 / 2,044 **0.42** |

**Paged decode, `flash_attn_with_kvcache`, bf16**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| b1 short | 67.3 / 101 **0.67** | 50.1 / 71.0 **0.71** | 32.7 / 52.8 **0.62** | 27.5 / 42.7 **0.64** | 42.8 / 55.4 **0.77** | 42.4 / 54.7 **0.77** |
| b1 long 4k, d128 | 1,675ᵖ¹⁶ / 1,516 **1.10** | 938ᵖ¹⁶ / 852 **1.10** | 723ᵖ¹⁶ / 670 **1.08** | 445ᵖ¹⁶ / 440 **1.01** | 1,264 / 1,417 **0.89** | 1,275 / 1,301 **0.98** |
| b8 1k | 898ᵖ¹⁶ / 877 **1.02** | 515 / 490 **1.05** | 409ᵖ¹⁶ / 471 **0.87** | 256ᵖ¹⁶ / 334 **0.77** | 663ᵖ¹⁶ / 759 **0.87** | 664ᵖ¹⁶ / 698 **0.95** |
| b8 1k, d32 | 523 / 563 **0.93** | 311 / 342 **0.91** | 194ᵖ¹⁶ / 259 **0.75** | 133ᵖ¹⁶ / 174 **0.77** | 362 / 417 **0.87** | 359 / 397 **0.90** |
| b8 4k, d128 | 13,205 / 11,388 **1.16** | 6,634ᵖ¹⁶ / 5,954 **1.11** | 5,770ᵖ¹⁶ / 5,110 **1.13** | 3,766ᵖ¹⁶ / 2,982 **1.26** | 9,843 / 11,459 **0.86** | 9,756 / 10,496 **0.93** |
| b8 mixed 17-2047 | 624ᵖ¹⁶ / 648 **0.96** | 321 / 347 **0.92** | 265 / 327 **0.81** | 177ᵖ¹⁶ / 227 **0.78** | 425 / 513 **0.83** | 426ᵖ¹⁶ / 478 **0.89** |
| b8 cross 1500 (Whisper) | 1,579 / 1,501 **1.05** | 824 / 785 **1.05** | 639 / 762 **0.84** | 394ᵖ¹⁶ / 512 **0.77** | 1,175 / 1,366 **0.86** | 1,163 / 1,201 **0.97** |
| b32 512 | 1,660 / 1,729 **0.96** | 937 / 931 **1.01** | 693 / 897 **0.77** | 429ᵖ¹⁶ / 552 **0.78** | 1,290 / 1,539 **0.84** | 1,286ᵖ¹⁶ / 1,339 **0.96** |
| b32 1k, d128 | 13,348 / 11,536 **1.16** | 6,614ᵖ¹⁶ / 6,052 **1.09** | 5,621ᵖ¹⁶ / 5,301 **1.06** | 3,552 / 3,186 **1.11** | 9,981 / 11,782 **0.85** | 9,804 / 10,390 **0.94** |

**Paged decode, `flash_attn_with_kvcache`, fp32**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| b1 short | 85.9ᵖ¹⁶ / 115 **0.74** | 62.4ᵖ¹⁶ / 87.1 **0.72** | 45.1 / 66.2 **0.68** | 33.5 / 48.6 **0.69** | 64.5 / 75.5 **0.86** | 66.9 / 76.2 **0.88** |
| b1 long 4k, d128 | 2,633ᵖ¹⁶ / 2,686 **0.98** | 1,413ᵖ¹⁶ / 1,349 **1.05** | 1,292 / 1,475 **0.88** | 819 / 953 **0.86** | 2,458ᵖ¹⁶ / 2,491 **0.99** | 2,464 / 2,489 **0.99** |
| b8 1k | 1,418ᵖ¹⁶ / 1,471 **0.96** | 728ᵖ¹⁶ / 763 **0.95** | 667 / 812 **0.82** | 463 / 538 **0.86** | 1,256ᵖ¹⁶ / 1,288 **0.98** | 1,260 / 1,282 **0.98** |
| b8 1k, d32 | 747 / 814 **0.92** | 436 / 470 **0.93** | 398 / 490 **0.81** | 276ᵖ¹⁶ / 333 **0.83** | 660 / 679 **0.97** | 665 / 687 **0.97** |
| b8 4k, d128 | 20,656ᵖ¹⁶ / 21,406 **0.96** | 10,527ᵖ¹⁶ / 11,252 **0.94** | 10,131 / 10,087 **1.00** | 5,684 / 5,826 **0.98** | 19,349ᵖ¹⁶ / 20,054 **0.96** | 19,335ᵖ¹⁶ / 19,581 **0.99** |
| b8 mixed 17-2047 | 898ᵖ¹⁶ / 1,019 **0.88** | 511ᵖ¹⁶ / 562 **0.91** | 458 / 559 **0.82** | 332 / 378 **0.88** | 790ᵖ¹⁶ / 823 **0.96** | 801 / 839 **0.95** |
| b8 cross 1500 (Whisper) | 2,390ᵖ¹⁶ / 2,475 **0.97** | 1,284ᵖ¹⁶ / 1,325 **0.97** | 1,160 / 1,306 **0.89** | 765 / 784 **0.98** | 2,254ᵖ¹⁶ / 2,306 **0.98** | 2,267ᵖ¹⁶ / 2,268 **1.00** |
| b32 512 | 2,620ᵖ¹⁶ / 2,883 **0.91** | 1,418ᵖ¹⁶ / 1,528 **0.93** | 1,290 / 1,628 **0.79** | 823 / 989 **0.83** | 2,471ᵖ¹⁶ / 2,592 **0.95** | 2,475ᵖ¹⁶ / 2,540 **0.97** |
| b32 1k, d128 | 20,420ᵖ¹⁶ / 21,757 **0.94** | 10,707ᵖ¹⁶ / 11,443 **0.94** | 9,924 / 10,110 **0.98** | 5,948 / 6,173 **0.96** | 19,371 / 20,382 **0.95** | 19,334 / 19,740 **0.98** |

**Where it stands.**
* **Prefill**: faster than SDPA on every shape on every machine: 0.18-0.48 of
  its time on c4d-standard-*, 0.25-0.66 on c4-standard-*, 0.31-0.96 on
  c4a-standard-*.
* **Paged decode**: wins or ties vLLM (within 3%) on every row on
  c4d-standard-*, and on every fp32 row elsewhere except one (c4-standard-16,
  `b1 long 4k, d128`, 1.05).
* **The open problem is bf16 decode at head_dim 128** on c4-standard-* (vLLM on
  AMX, 1.09-1.16x faster on the long-context rows) and c4a-standard-* (1.06-1.26x);
  the Whisper cross-attention row on c4-standard-* (1.05) and `b8 1k` bf16 on
  c4-standard-16 (1.05) are the other losses beyond 3%.

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16** where the model allows; on AMX CPUs (c4-standard-*) prefill runs
   on AMX tiles (fp32 as bf16 hi/lo products).
3. **Page size.** 16 and 32 are both specialized at compile time (48 and 64
   too); the benchmark picks the faster one per shape. Page 16 won 5-11 of the 18
   decode rows per machine, by up to 1.16x (head_dim 32 bf16 on c4a-standard-*;
   head_dim 128 fp32 on c4-standard-*, 1.10-1.11x); elsewhere the two are within
   a few percent.
4. **Leave AMX on** (the default on AMX CPUs). Few-query calls automatically use
   the vector policy, which is faster there; `CPU_ATTENTION_AMX=0` forces it
   everywhere.

## Tests

```bash
uv run --extra test pytest            # ~360 cases, a few seconds
CPU_ATTENTION_ISA=avx2 uv run --extra test pytest    # a lower tier
```

Every case is checked against a slow, obviously correct reference
(`tests/reference.py`): eager attention in PyTorch float64 (matmul, additive
mask, softmax, matmul, as transformers' `eager_attention_forward`). The tests
cover prefill (tile-edge, single-row and unequal lengths, causal and not, 1-12
heads, head_dim 32/64/128, fused-projection strides, a 40-case fuzz) and paged decode (page sizes 16/32/48/64, empty and
uneven contexts, poisoned caches, `reshape_and_cache` and one- and multi-token
appends, 1-8 threads, concurrent callers, a 30-case fuzz), plus the rejected
arguments. Limits:

| case | max error vs float64 |
|---|---|
| `flash_attn_func` fp32 | 2e-4 |
| `flash_attn_func` bf16 | 3e-2 |
| paged decode fp32 | 1e-6 |
| paged decode bf16 | 1e-3 beyond the rounding of the exact result to bf16 |

Rounding the exact result to bf16 alone exceeds an absolute 1e-3 on many cases,
so the paged bf16 limit is taken above that floor (`docs/Mojo.md` §5). The fp32
limits hold at `softmax_scale` up to `1/sqrt(head_dim)` (the model scale); larger
scales grow the logits, and fp32 softmax error with them.

## Documentation

* [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md): how the benchmark
  measures, and every machine's full results.
* [`docs/Mojo.md`](docs/Mojo.md): the kernel's design, ISA tiers,
  numerics, call path, defects found, falsification ledger and Mojo 1.2.0
  notes.
* [`AGENTS.md`](AGENTS.md): working rules for anyone changing a kernel.

## License

Apache-2.0. The kernel follows vLLM's `csrc/cpu` attention (Apache-2.0); the
Python API mirrors [flash-attention](https://github.com/Dao-AILab/flash-attention)
(BSD-3).
