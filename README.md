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
* The API is flash-attn's: `cpu_attention` exports exactly flash-attn's functions
  (the packed and varlen ones raise `NotImplementedError`).
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
from cpu_attention import flash_attn_func, flash_attn_with_kvcache

# Prefill: q/k/v [batch, seq, heads, head_dim]
out = flash_attn_func(q, k, v, causal=True)

# Stage a prompt's K/V into the paged cache (k/v [B, prompt_len, kv_heads, D]) ...
flash_attn_with_kvcache(q, k_cache, v_cache, k=k_prompt, v=v_prompt,
                        cache_seqlens=0, page_table=pages)
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
  stored transposed inside the page, so it is written only through
  `flash_attn_with_kvcache` (`k`/`v` of any number of new tokens); a cache in
  another library's layout (vLLM's included) is not readable.
* **Strides.** q/k/v may be strided chunks of a fused projection: any token
  stride, with heads and dims contiguous.
* **`causal` on the paged path** is accepted and has no effect, because one
  query at the last position sees every key.
* **Engines.** [`examples/ops.py`](examples/ops.py) shows an inference engine's
  per-layer calls (fast-encdec's): precomputed slot mappings, no per-call
  argument checks, a preallocated `out=`.

## Performance

PERFORMANCE

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16** where the model allows; on AMX CPUs (c4-standard-*) prefill runs
   on AMX tiles (fp32 as bf16 hi/lo products).
3. **Page size.** 16 and 32 are both specialized at compile time (48 and 64
   too); the benchmark picks the faster one per shape (PAGEADVICE).
4. **Leave AMX on** (the default on AMX CPUs). Few-query calls automatically use
   the vector policy, which is faster there; `CPU_ATTENTION_AMX=0` forces it
   everywhere.

## Tests

```bash
uv run --extra test pytest            # ~480 cases, ~10 s
CPU_ATTENTION_ISA=avx2 uv run --extra test pytest    # a lower tier
```

Every case is checked against a slow, obviously correct reference
(`tests/reference.py`): numpy in float64, one (batch, head) at a time, with
explicit masking and softmax. The tests cover prefill (tile-edge, single-row and
unequal lengths, causal and not, MHA/GQA/MQA, head_dim 32/64/128, fused-projection
strides, a 40-case fuzz) and paged decode (page sizes 16/32/48/64, empty and
uneven contexts, poisoned caches, one- and multi-token appends, 1-8 threads,
concurrent callers, a 30-case fuzz), plus the rejected arguments. Limits:

| case | max error vs float64 |
|---|---|
| `flash_attn_func` fp32 | 2e-4 |
| `flash_attn_func` bf16 | 3e-2 |
| paged decode fp32 | 1e-6 |
| paged decode bf16 | 1e-3 beyond the rounding of the exact result to bf16 |

Rounding the exact result to bf16 alone exceeds an absolute 1e-3 on many cases,
so the paged bf16 limit is taken above that floor (`docs/Mojo.md` §5).

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
