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
VMs, `b3e10f8`, Mojo 1.2.0 (2016d0cd), torch 2.14.1+cpu, Oct 2026. MHA shapes
(GQA/MQA are not supported yet). Rows are labelled by shape, in SDPA's legend:
**N** batch, **L** query length, **S** key/value length (`17..2047`: one
length per sequence), **H** heads (query and key/value alike), **E** head
dimension. Prefill rows marked `causal` are bottom-right causal; decode rows
store the new token's K/V in the call unless marked `no append`
(cross-attention over a pre-staged cache).
[`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md) has the method and every
arm's median, min and max.

**Prefill, `flash_attn_func`, bf16**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | 151 / 378 **0.40** | 87.2 / 223 **0.39** | 63.0 / 158 **0.40** | 41.9 / 102 **0.41** | 72.2 / 321 **0.22** | 56.5 / 220 **0.26** |
| N=1 L=1024 S=1024 H=16 E=64 causal | 1,392 / 4,000 **0.35** | 771 / 2,681 **0.29** | 1,662 / 3,357 **0.49** | 862 / 1,740 **0.50** | 2,007 / 9,607 **0.21** | 933 / 5,018 **0.19** |
| N=1 L=2048 S=2048 H=32 E=128 causal | 12,235 / 30,091 **0.41** | 17,273 / 20,947 **0.82** | 20,991 / 35,406 **0.59** | 10,488 / 18,034 **0.58** | 26,193 / 118,363 **0.22** | 14,662 / 61,497 **0.24** |
| N=1 L=4096 S=4096 H=32 E=128 causal | 57,631 / 102,633 **0.56** | 43,000 / 63,228 **0.68** | 84,879 / 126,181 **0.67** | 42,724 / 64,006 **0.67** | 102,773 / 430,218 **0.24** | 56,447 / 220,596 **0.26** |
| N=8 L=512 S=512 H=16 E=64 | 4,471 / 7,985 **0.56** | 2,151 / 4,202 **0.51** | 6,410 / 9,729 **0.66** | 3,173 / 4,791 **0.66** | 6,723 / 26,553 **0.25** | 3,548 / 13,454 **0.26** |
| N=1 L=1500 S=1500 H=20 E=64 | 4,857 / 9,970 **0.49** | 2,741 / 5,033 **0.54** | 8,419 / 11,732 **0.72** | 4,376 / 6,267 **0.70** | 8,432 / 34,403 **0.25** | 4,541 / 17,416 **0.26** |
| N=8 L=256 S=256 H=16 E=32 | 1,432 / 2,579 **0.56** | 695 / 1,337 **0.52** | 1,138 / 1,757 **0.65** | 588 / 930 **0.63** | 1,057 / 4,067 **0.26** | 522 / 2,087 **0.25** |

**Prefill, `flash_attn_func`, fp32**: ours / SDPA µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | 266 / 470 **0.57** | 151 / 284 **0.53** | 114 / 365 **0.31** | 68.8 / 205 **0.34** | 107 / 324 **0.33** | 88.9 / 210 **0.42** |
| N=1 L=1024 S=1024 H=16 E=64 causal | 3,379 / 8,684 **0.39** | 1,490 / 3,883 **0.38** | 6,795 / 9,403 **0.72** | 2,287 / 4,704 **0.49** | 3,332 / 9,483 **0.35** | 1,681 / 5,282 **0.32** |
| N=1 L=2048 S=2048 H=32 E=128 causal | 44,286 / 89,705 **0.49** | 31,269 / 49,630 **0.63** | 64,038 / 86,484 **0.74** | 31,940 / 43,829 **0.73** | 50,079 / 114,460 **0.44** | 28,363 / 60,031 **0.47** |
| N=1 L=4096 S=4096 H=32 E=128 causal | 142,794 / 307,336 **0.46** | 84,263 / 160,105 **0.53** | 251,350 / 301,490 **0.83** | 127,116 / 152,891 **0.83** | 181,733 / 405,089 **0.45** | 94,258 / 208,321 **0.45** |
| N=8 L=512 S=512 H=16 E=64 | 12,331 / 30,277 **0.41** | 4,576 / 13,777 **0.33** | 17,309 / 19,461 **0.89** | 8,736 / 9,702 **0.90** | 12,214 / 26,862 **0.45** | 6,341 / 13,743 **0.46** |
| N=1 L=1500 S=1500 H=20 E=64 | 10,812 / 26,396 **0.41** | 5,826 / 13,056 **0.45** | 23,524 / 24,572 **0.96** | 12,329 / 13,777 **0.89** | 15,212 / 32,200 **0.47** | 7,481 / 16,460 **0.45** |
| N=8 L=256 S=256 H=16 E=32 | 2,374 / 4,178 **0.57** | 1,263 / 2,116 **0.60** | 2,736 / 3,013 **0.91** | 1,387 / 1,569 **0.88** | 1,704 / 4,055 **0.42** | 856 / 2,070 **0.41** |

**Paged decode, `flash_attn_with_kvcache`, bf16**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | 78.6ᵖ¹⁶ / 104 **0.76** | 56.9 / 78.5 **0.73** | 32.3 / 52.0 **0.62** | 28.8ᵖ¹⁶ / 44.1 **0.65** | 46.8 / 62.8 **0.75** | 41.6 / 54.0 **0.77** |
| N=1 L=1 S=4096 H=32 E=128 | 1,747ᵖ¹⁶ / 1,613 **1.08** | 986 / 929 **1.06** | 741ᵖ¹⁶ / 697 **1.06** | 455ᵖ¹⁶ / 434 **1.05** | 1,497 / 1,626 **0.92** | 1,262 / 1,298 **0.97** |
| N=8 L=1 S=1024 H=16 E=64 | 983 / 978 **1.01** | 608 / 573 **1.06** | 402 / 493 **0.82** | 234ᵖ¹⁶ / 298 **0.78** | 765 / 840 **0.91** | 664ᵖ¹⁶ / 704 **0.94** |
| N=8 L=1 S=1024 H=16 E=32 | 570 / 589 **0.97** | 383ᵖ¹⁶ / 403 **0.95** | 195ᵖ¹⁶ / 255 **0.76** | 129ᵖ¹⁶ / 175 **0.74** | 413 / 473 **0.87** | 361 / 387 **0.93** |
| N=8 L=1 S=4096 H=32 E=128 | 12,992 / 11,496 **1.13** | 7,077 / 6,298 **1.12** | 5,476ᵖ¹⁶ / 5,106 **1.07** | 3,618ᵖ¹⁶ / 2,960 **1.22** | 11,808 / 13,362 **0.88** | 9,834 / 10,413 **0.94** |
| N=8 L=1 S=17..2047 H=16 E=64 | 654 / 666 **0.98** | 406 / 457 **0.89** | 264 / 329 **0.80** | 187ᵖ¹⁶ / 250 **0.75** | 480ᵖ¹⁶ / 578 **0.83** | 423 / 471 **0.90** |
| N=8 L=1 S=1500 H=20 E=64 no append | 1,624 / 1,643 **0.99** | 940ᵖ¹⁶ / 875 **1.07** | 635 / 755 **0.84** | 410 / 460 **0.89** | 1,460 / 1,594 **0.92** | 1,166 / 1,194 **0.98** |
| N=32 L=1 S=512 H=16 E=64 | 1,812 / 1,848 **0.98** | 1,036 / 1,047 **0.99** | 710 / 958 **0.74** | 442 / 555 **0.80** | 1,460ᵖ¹⁶ / 1,718 **0.85** | 1,289ᵖ¹⁶ / 1,337 **0.96** |
| N=32 L=1 S=1024 H=32 E=128 | 12,926 / 11,572 **1.12** | 7,268 / 6,704 **1.08** | 5,496ᵖ¹⁶ / 5,263 **1.04** | 3,525ᵖ¹⁶ / 3,099 **1.14** | 12,182 / 14,390 **0.85** | 9,766 / 10,355 **0.94** |

**Paged decode, `flash_attn_with_kvcache`, fp32**: ours / vLLM µs, ratio

| shape | c4-standard-8 | c4-standard-16 | c4a-standard-8 | c4a-standard-16 | c4d-standard-8 | c4d-standard-16 |
|---|---:|---:|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | 95.4ᵖ¹⁶ / 129 **0.74** | 73.0ᵖ¹⁶ / 93.5 **0.78** | 44.0ᵖ¹⁶ / 65.4 **0.67** | 33.5 / 50.6 **0.66** | 76.1 / 87.0 **0.87** | 67.5ᵖ¹⁶ / 75.3 **0.90** |
| N=1 L=1 S=4096 H=32 E=128 | 2,755ᵖ¹⁶ / 2,781 **0.99** | 1,704ᵖ¹⁶ / 1,551 **1.10** | 1,257 / 1,437 **0.87** | 775 / 911 **0.85** | 2,812 / 2,880 **0.98** | 2,460 / 2,480 **0.99** |
| N=8 L=1 S=1024 H=16 E=64 | 1,518ᵖ¹⁶ / 1,529 **0.99** | 898ᵖ¹⁶ / 941 **0.95** | 675 / 836 **0.81** | 450 / 510 **0.88** | 1,417ᵖ¹⁶ / 1,450 **0.98** | 1,257 / 1,281 **0.98** |
| N=8 L=1 S=1024 H=16 E=32 | 840 / 879 **0.95** | 531 / 533 **1.00** | 409 / 496 **0.82** | 252ᵖ¹⁶ / 306 **0.83** | 772ᵖ¹⁶ / 800 **0.97** | 674ᵖ¹⁶ / 691 **0.98** |
| N=8 L=1 S=4096 H=32 E=128 | 21,957ᵖ¹⁶ / 22,451 **0.98** | 11,732ᵖ¹⁶ / 11,966 **0.98** | 10,112 / 10,151 **1.00** | 6,113 / 5,875 **1.04** | 22,112ᵖ¹⁶ / 23,472 **0.94** | 19,274ᵖ¹⁶ / 19,504 **0.99** |
| N=8 L=1 S=17..2047 H=16 E=64 | 973ᵖ¹⁶ / 1,037 **0.94** | 564ᵖ¹⁶ / 626 **0.90** | 440 / 570 **0.77** | 305 / 379 **0.81** | 883 / 963 **0.92** | 803 / 832 **0.96** |
| N=8 L=1 S=1500 H=20 E=64 no append | 2,553ᵖ¹⁶ / 2,627 **0.97** | 1,391ᵖ¹⁶ / 1,398 **1.00** | 1,159 / 1,313 **0.88** | 740 / 800 **0.92** | 2,623 / 2,761 **0.95** | 2,255ᵖ¹⁶ / 2,273 **0.99** |
| N=32 L=1 S=512 H=16 E=64 | 2,837ᵖ¹⁶ / 3,083 **0.92** | 1,585ᵖ¹⁶ / 1,627 **0.97** | 1,288 / 1,599 **0.81** | 849 / 959 **0.89** | 2,857 / 2,971 **0.96** | 2,516 / 2,623 **0.96** |
| N=32 L=1 S=1024 H=32 E=128 | 21,561ᵖ¹⁶ / 22,739 **0.95** | 11,281ᵖ¹⁶ / 12,072 **0.93** | 10,110 / 9,880 **1.02** | 5,502 / 5,940 **0.93** | 22,372 / 23,882 **0.94** | 19,302 / 19,717 **0.98** |

**Where it stands.**
* **Prefill**: faster than SDPA on every shape on every machine: 0.19-0.47 of
  its time on c4d-standard-*, 0.29-0.82 on c4-standard-*, 0.31-0.96 on
  c4a-standard-*.
* **Paged decode**: wins or ties vLLM (within 3%) on every row on
  c4d-standard-*, and on every E=32 and E=64 row elsewhere except two bf16 rows
  on c4-standard-16: `N=8 L=1 S=1024 H=16 E=64` (1.06) and
  `N=8 L=1 S=1500 H=20 E=64 no append` (1.07).
* **The open problem is decode at E=128** with long contexts (`S=4096`, and
  `N=32 S=1024`): in bf16 vLLM is 1.06-1.13x faster on c4-standard-* (its AMX
  path) and 1.04-1.22x on c4a-standard-*; fp32 is within 3% except
  `N=1 L=1 S=4096 H=32 E=128` on c4-standard-16 (1.10) and
  `N=8 L=1 S=4096 H=32 E=128` on c4a-standard-16 (1.04).

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16** where the model allows; on AMX CPUs (c4-standard-*) prefill runs
   on AMX tiles (fp32 as bf16 hi/lo products).
3. **Page size.** 16 and 32 are both specialized at compile time (48 and 64
   too); the benchmark picks the faster one per shape. Page 16 won 5-10 of the 18
   decode rows per machine, most on c4a-standard-* (up to 1.20x faster than page
   32 at `N=8 L=1 S=4096 H=32 E=128` bf16); elsewhere the two are within a few
   percent.
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
