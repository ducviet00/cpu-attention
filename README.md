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

```bash
uv sync --extra build --extra test                       # torch, the mojo wheel, pytest + numpy
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
  per-layer calls (fast-encdec's): precomputed slot mappings, no per-call
  argument checks, a preallocated `out=`.

## Performance

Ours against **torch SDPA** (prefill) and **vLLM's CPU attention** (paged
decode, `csrc/cpu` at the commit pinned in `scripts/build_vllm_ref.sh`), timed at
each library's Python API with L2/L3 flushed before every call: the median of 10
calls after 3 warmups. Each cell is ours / reference in µs and the **ratio**
(below 1: we are faster). Paged rows use each library's best page size; ᵖ¹⁶
marks rows where our page 16 beat page 32. Threads = physical cores; vLLM runs
AMX for bf16 on c4-standard-*, VEC for other x86 cases and NEON on c4a. GCP spot
VMs, Mojo 1.2.0 (2016d0cd), torch 2.14.1+cpu, Oct 2026 (commits `b7b1796` to
`0081f7c`, the same kernel and benchmark; each machine's is in BENCHMARK.md).
[`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md) has the method and every
arm's median, min and max.

**Prefill, `flash_attn_func`, bf16**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| short prompt | 141 / 324 **0.43** | 77.5 / 193 **0.40** | 60.5 / 160 **0.38** | 41.9 / 103 **0.41** | 65.7 / 316 **0.21** | 53.5 / 212 **0.25** |
| prompt 1k | 1,260 / 3,704 **0.34** | 801 / 2,612 **0.31** | 1,656 / 3,336 **0.50** | 861 / 1,738 **0.50** | 1,660 / 9,376 **0.18** | 921 / 5,055 **0.18** |
| prompt 2k, d128 | 12,769 / 28,796 **0.44** | 7,691 / 14,814 **0.52** | 21,010 / 35,557 **0.59** | 10,519 / 17,953 **0.59** | 26,963 / 114,888 **0.23** | 14,533 / 60,856 **0.24** |
| long prompt, GQA | 44,959 / 86,126 **0.52** | 25,560 / 52,526 **0.49** | 84,388 / 123,294 **0.68** | 42,843 / 61,847 **0.69** | 95,709 / 406,710 **0.24** | 48,518 / 211,000 **0.23** |
| encoder batch | 4,304 / 7,794 **0.55** | 2,521 / 4,644 **0.54** | 6,275 / 9,395 **0.67** | 3,159 / 4,757 **0.66** | 6,531 / 26,279 **0.25** | 3,461 / 13,286 **0.26** |
| encoder 1500 (Whisper) | 4,644 / 9,310 **0.50** | 3,054 / 5,104 **0.60** | 8,757 / 12,213 **0.72** | 4,361 / 6,201 **0.70** | 8,267 / 32,986 **0.25** | 4,511 / 17,147 **0.26** |
| encoder, d32 | 1,250 / 2,122 **0.59** | 650 / 1,135 **0.57** | 1,149 / 1,761 **0.65** | 579 / 918 **0.63** | 942 / 4,011 **0.23** | 526 / 2,078 **0.25** |

**Prefill, `flash_attn_func`, fp32**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| short prompt | 254 / 461 **0.55** | 142 / 284 **0.50** | 112 / 372 **0.30** | 65.2 / 205 **0.32** | 95.7 / 324 **0.30** | 80.7 / 206 **0.39** |
| prompt 1k | 2,718 / 7,562 **0.36** | 1,590 / 3,946 **0.40** | 4,607 / 9,280 **0.50** | 2,290 / 4,738 **0.48** | 3,190 / 9,237 **0.35** | 1,606 / 4,774 **0.34** |
| prompt 2k, d128 | 43,207 / 85,819 **0.50** | 31,324 / 54,781 **0.57** | 63,906 / 86,632 **0.74** | 32,152 / 43,913 **0.73** | 48,094 / 112,772 **0.43** | 27,551 / 58,853 **0.47** |
| long prompt, GQA | 116,601 / 282,116 **0.41** | 61,435 / 142,877 **0.43** | 249,289 / 300,375 **0.83** | 126,660 / 151,429 **0.84** | 179,678 / 399,816 **0.45** | 92,971 / 203,651 **0.46** |
| encoder batch | 9,257 / 28,140 **0.33** | 6,098 / 13,927 **0.44** | 17,310 / 21,446 **0.81** | 8,738 / 9,652 **0.91** | 11,988 / 26,820 **0.45** | 6,237 / 13,480 **0.46** |
| encoder 1500 (Whisper) | 10,319 / 24,906 **0.41** | 6,117 / 12,914 **0.47** | 23,736 / 24,643 **0.96** | 12,158 / 12,951 **0.94** | 15,150 / 32,023 **0.47** | 7,403 / 16,303 **0.45** |
| encoder, d32 | 2,123 / 3,922 **0.54** | 1,350 / 2,045 **0.66** | 2,722 / 3,019 **0.90** | 1,371 / 1,550 **0.88** | 1,673 / 3,976 **0.42** | 889 / 2,077 **0.43** |

**Paged decode, `flash_attn_with_kvcache`, bf16**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| b1 short | 68.0 / 92.2 **0.74** | 57.9 / 79.4 **0.73** | 32.0 / 54.2 **0.59** | 27.4 / 42.2 **0.65** | 41.1 / 54.5 **0.75** | 40.9 / 53.3 **0.77** |
| b1 long, GQA | 777 / 504 **1.54** | 449 / 334 **1.34** | 267 / 235 **1.13** | 187 / 245 **0.76** | 401ᵖ¹⁶ / 481 **0.83** | 376ᵖ¹⁶ / 411 **0.91** |
| b8 1k | 994 / 972 **1.02** | 543 / 552 **0.98** | 438 / 472 **0.93** | 231ᵖ¹⁶ / 299 **0.77** | 672 / 767 **0.88** | 666 / 703 **0.95** |
| b8 1k, d32 | 548ᵖ¹⁶ / 568 **0.96** | 353 / 388 **0.91** | 204ᵖ¹⁶ / 256 **0.80** | 126ᵖ¹⁶ / 171 **0.74** | 358ᵖ¹⁶ / 400 **0.90** | 363 / 400 **0.91** |
| b8 1k, GQA 8:1 | 1,020ᵖ¹⁶ / 555 **1.84** | 566 / 351 **1.61** | 453ᵖ¹⁶ / 284 **1.59** | 284 / 201 **1.41** | 499ᵖ¹⁶ / 606 **0.82** | 419 / 440 **0.95** |
| b8 4k, GQA | 5,911ᵖ¹⁶ / 3,293 **1.80** | 2,978ᵖ¹⁶ / 1,831 **1.63** | 1,861 / 1,569 **1.19** | 1,168 / 976 **1.20** | 3,136 / 3,723 **0.84** | 2,573ᵖ¹⁶ / 2,809 **0.92** |
| b8 mixed 17-2047 | 642 / 656 **0.98** | 398 / 445 **0.89** | 254 / 327 **0.78** | 171ᵖ¹⁶ / 227 **0.75** | 422 / 505 **0.84** | 431ᵖ¹⁶ / 480 **0.90** |
| b8 cross 1500 (Whisper) | 1,666ᵖ¹⁶ / 1,595 **1.04** | 975 / 891 **1.09** | 669ᵖ¹⁶ / 749 **0.89** | 395ᵖ¹⁶ / 503 **0.79** | 1,188 / 1,352 **0.88** | 1,161 / 1,198 **0.97** |
| b32 512 | 1,760 / 1,804 **0.98** | 1,037 / 1,011 **1.03** | 710 / 901 **0.79** | 436ᵖ¹⁶ / 576 **0.76** | 1,307 / 1,555 **0.84** | 1,289ᵖ¹⁶ / 1,339 **0.96** |
| b32 2k, GQA | 11,367ᵖ¹⁶ / 6,123 **1.86** | 5,895ᵖ¹⁶ / 3,669 **1.61** | 3,649 / 3,068 **1.19** | 2,313 / 1,952 **1.19** | 5,795ᵖ¹⁶ / 7,604 **0.76** | 5,152ᵖ¹⁶ / 5,682 **0.91** |

**Paged decode, `flash_attn_with_kvcache`, fp32**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| b1 short | 86.6ᵖ¹⁶ / 115 **0.75** | 68.1 / 93.0 **0.73** | 44.2 / 64.8 **0.68** | 33.3ᵖ¹⁶ / 48.9 **0.68** | 65.9 / 76.0 **0.87** | 65.7 / 77.2 **0.85** |
| b1 long, GQA | 1,226ᵖ¹⁶ / 1,207 **1.02** | 647 / 649 **1.00** | 474 / 454 **1.04** | 336 / 435 **0.77** | 689ᵖ¹⁶ / 709 **0.97** | 676 / 692 **0.98** |
| b8 1k | 1,510 / 1,526 **0.99** | 890 / 868 **1.03** | 682 / 864 **0.79** | 438 / 509 **0.86** | 1,272 / 1,294 **0.98** | 1,267ᵖ¹⁶ / 1,291 **0.98** |
| b8 1k, d32 | 769ᵖ¹⁶ / 805 **0.96** | 481 / 508 **0.95** | 436 / 487 **0.90** | 270 / 316 **0.85** | 660 / 683 **0.97** | 667 / 685 **0.97** |
| b8 1k, GQA 8:1 | 1,709 / 1,556 **1.10** | 869 / 836 **1.04** | 907 / 573 **1.58** | 517 / 340 **1.52** | 897ᵖ¹⁶ / 879 **1.02** | 881ᵖ¹⁶ / 746 **1.18** |
| b8 4k, GQA | 8,734ᵖ¹⁶ / 8,406 **1.04** | 4,590ᵖ¹⁶ / 4,549 **1.01** | 3,990ᵖ¹⁶ / 3,357 **1.19** | 2,331 / 2,071 **1.13** | 5,369ᵖ¹⁶ / 5,670 **0.95** | 5,123ᵖ¹⁶ / 5,397 **0.95** |
| b8 mixed 17-2047 | 969ᵖ¹⁶ / 1,088 **0.89** | 603ᵖ¹⁶ / 652 **0.92** | 433 / 554 **0.78** | 318 / 380 **0.83** | 788 / 826 **0.95** | 806 / 845 **0.95** |
| b8 cross 1500 (Whisper) | 2,455ᵖ¹⁶ / 2,505 **0.98** | 1,409ᵖ¹⁶ / 1,438 **0.98** | 1,174 / 1,328 **0.88** | 780 / 810 **0.96** | 2,285ᵖ¹⁶ / 2,314 **0.99** | 2,266 / 2,285 **0.99** |
| b32 512 | 2,771ᵖ¹⁶ / 2,923 **0.95** | 1,607ᵖ¹⁶ / 1,690 **0.95** | 1,287 / 1,593 **0.81** | 893 / 1,015 **0.88** | 2,511ᵖ¹⁶ / 2,605 **0.96** | 2,481 / 2,546 **0.97** |
| b32 2k, GQA | 17,933ᵖ¹⁶ / 18,092 **0.99** | 9,083ᵖ¹⁶ / 8,906 **1.02** | 7,759ᵖ¹⁶ / 6,567 **1.18** | 4,299ᵖ¹⁶ / 3,907 **1.10** | 10,604ᵖ¹⁶ / 11,390 **0.93** | 10,086ᵖ¹⁶ / 10,706 **0.94** |

**Where it stands.**
* **Prefill**: faster than SDPA on every shape on every machine: 0.18-0.47 of
  its time on c4d-standard-*, 0.31-0.66 on c4-standard-*, 0.30-0.96 on
  c4a-standard-*.
* **Paged decode** wins or ties vLLM (within 3%) on the MHA, head_dim 32,
  mixed-length and batch-32 shapes on every machine, and on Whisper
  cross-attention everywhere except c4-standard-* bf16 (1.04-1.09, against
  vLLM's AMX path). On c4d-standard-* it loses only one row (`b8 1k, GQA 8:1`
  fp32 on -16, 1.18).
* **The open problem is GQA decode at head_dim 128**, the Llama-style shape
  (32 query heads over 8 or 4 KV heads). On c4-standard-* in bf16, vLLM runs it
  on AMX tiles and is 1.34-1.86x faster; on c4a-standard-* vLLM is 1.1-1.6x
  faster in both dtypes. Our decode keeps the group's query heads as rows of
  one vector register tile on every CPU (`docs/Mojo.md` §3).

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16** where the model allows; on AMX CPUs (c4-standard-*) prefill runs
   on AMX tiles (fp32 as bf16 hi/lo products).
3. **Page size.** 16 and 32 are both specialized at compile time (48 and 64
   too); the benchmark picks the faster one per shape. Page 16 won 5-13 of the 20
   decode rows per machine, most at head_dim 32 on c4a-standard-* (1.17-1.23x
   faster than page 32) and at batch 32, GQA on c4d-standard-8 (1.15x); elsewhere
   the two are within a few percent.
4. **Leave AMX on** (the default on AMX CPUs). Few-query calls automatically use
   the vector policy, which is faster there; `CPU_ATTENTION_AMX=0` forces it
   everywhere.

## Tests

```bash
uv run --extra test pytest            # ~490 cases, ~10 s
CPU_ATTENTION_ISA=avx2 uv run --extra test pytest    # a lower tier
```

Every case is checked against a slow, obviously correct reference
(`tests/reference.py`): numpy in float64, one (batch, head) at a time, with
explicit masking and softmax. The tests cover prefill (tile-edge, single-row and
unequal lengths, causal and not, MHA/GQA/MQA, head_dim 32/64/128, fused-projection
strides, a 40-case fuzz) and paged decode (page sizes 16/32/48/64, empty and
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
