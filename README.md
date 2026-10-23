# cpu-attention

Paged and dense attention kernels for CPUs, with flash-attn's API. One source
builds for x86-64 (AVX-512 + AMX) and aarch64 (NEON + BF16). It is the attention
backend of [fast-encdec](https://github.com/ducviet00/fast-encdec), and it also
works as a standalone library.

* `flash_attn_with_kvcache` — paged decoder self-attention and cross-attention,
  one query token per sequence.
* `flash_attn_func` — dense attention over whole prompts, causal or not.
* fp32 and bf16, MHA / MQA / GQA, head_dim 32/64/128.
* Linux, PyTorch, Python ≥ 3.10. The build needs the `mojo` wheel (installed by
  the `build` extra), not a separate Mojo toolchain.

## Performance

`ours / vLLM csrc/cpu`, **min** over repeated rounds, `< 1` is faster. Ours
includes the K/V pack and the output allocation; vLLM's cache is staged before
the timer, so the ratios are conservative for us.

Boxes: **Ryzen 7 7840H** (8 threads), **Graviton 5 `m9g.2xlarge`** (8 cores,
8 threads), **Xeon 6975P-C `m8i.2xlarge`** (4 cores / 8 threads, 4 threads).
Xeon bf16 is against vLLM's **AMX** build (its honest bf16 bar); every other
column is against vLLM's VEC (x86) or NEON (ARM) build.

### Dense attention (`flash_attn_func`), 11 shapes × 2 dtypes

**bf16**

| shape | Ryzen 7840H | Graviton 5 | Xeon 6975P-C |
|---|---:|---:|---:|
| 128:8:64 | 0.88 | 0.97 | 0.85 |
| 512:8:64 | 0.75 | 0.88 | 0.80 |
| 1024:8:64 | 0.72 | 0.87 | 0.88 |
| 2048:8:64 | 0.72 | 0.94 | 0.86 |
| 4096:8:64 | 0.73 | 0.93 | 0.80 |
| 256:16:128 | 0.88 | 0.89 | 0.98 |
| 1024:16:128 | 0.89 | 0.94 | 1.01 |
| 2048:16:128 | 0.92 | 1.00 | 0.95 |
| 4096:16:128 | 1.07 | 0.99 | 0.98 |
| 512:32:128 | 0.91 | 0.95 | 1.02 |
| 1024:8:32 | 0.79 | 0.83 | 0.83 |

**fp32**

| shape | Ryzen 7840H | Graviton 5 | Xeon 6975P-C |
|---|---:|---:|---:|
| 128:8:64 | 1.12 | 0.92 | 0.98 |
| 512:8:64 | 1.04 | 0.88 | 0.62 |
| 1024:8:64 | 1.00 | 0.92 | 0.60 |
| 2048:8:64 | 1.03 | 0.92 | 0.56 |
| 4096:8:64 | 1.07 | 0.97 | 0.55 |
| 256:16:128 | 0.98 | 0.79 | 0.65 |
| 1024:16:128 | 1.20 | 0.82 | 0.50 |
| 2048:16:128 | 1.18 | 0.89 | 0.51 |
| 4096:16:128 | 1.68 | 0.98 | 0.79 |
| 512:32:128 | 1.31 | 0.83 | 0.56 |
| 1024:8:32 | 1.00 | 0.90 | 0.67 |

### Paged attention (one token per sequence)

**bf16**

| case | Ryzen 7840H | Graviton 5 | Xeon 6975P-C |
|---|---:|---:|---:|
| self b=1 ctx=256 | 1.09 | 1.09 | 0.77 |
| cross b=1 ctx=256 | 1.33 | 1.37 | 1.20 |
| self b=8 ctx=256 | 0.81 | 0.35 | 0.53 |
| cross b=8 ctx=256 | 0.85 | 0.44 | 0.59 |
| self b=8 ctx=1024 | 0.75 | 0.69 | 0.92 |
| cross b=8 ctx=1024 | 0.75 | 0.76 | 0.97 |
| self b=32 ctx=1024 | 0.92 | 0.89 | 0.66 |
| cross b=32 ctx=1024 | 0.96 | 0.94 | 0.85 |
| self b=1 ctx=4096 | 0.66 | 0.59 | 0.82 |
| cross b=1 ctx=4096 | 0.66 | 0.73 | 0.85 |

**fp32**

| case | Ryzen 7840H | Graviton 5 | Xeon 6975P-C |
|---|---:|---:|---:|
| self b=1 ctx=256 | 1.11 | 0.89 | 0.46 |
| cross b=1 ctx=256 | 1.50 | 1.02 | 0.86 |
| self b=8 ctx=256 | 0.75 | 0.34 | 1.04 |
| cross b=8 ctx=256 | 0.78 | 0.46 | 1.13 |
| self b=8 ctx=1024 | 0.95 | 0.86 | 1.20 |
| cross b=8 ctx=1024 | 0.94 | 0.86 | 1.15 |
| self b=32 ctx=1024 | 0.99 | 0.80 | 0.87 |
| cross b=32 ctx=1024 | 1.00 | 0.85 | 1.01 |
| self b=1 ctx=4096 | 0.90 | 0.87 | 1.08 |
| cross b=1 ctx=4096 | 0.90 | 0.86 | 1.11 |

Ryzen paged uses 40-call blocks; the server paged columns use 15 interleaved
rounds. See [`TECHNICAL.md`](TECHNICAL.md) for the protocol.

### The short version

* **Vs torch SDPA**, every dense row on Graviton and Xeon is faster. On Ryzen
  21 of 22 are; the exception is `4096:16:128` fp32 at 1.04× SDPA.
* **Dense vs vLLM**: bf16 wins 10/11 on Ryzen (4096:16:128 is 1.07), 10/11 on
  Graviton (2048:16:128 is 1.00), and 9/11 on Xeon (1024:16:128 and
  512:32:128 are 1.01/1.02). fp32 wins 11/11 on Graviton and Xeon (0.50-0.98);
  on Ryzen it is 0.98-1.68 — parity at short contexts, slower at D=128. So
  fp32 dense is the weak column on the laptop, not on the servers.
* **Paged**: bf16 wins the batch ≥ 8 rows on all three boxes. `b=1` is mixed:
  the bf16 cross rows lose on Ryzen and Graviton (1.33/1.37), the fp32 cross
  loses on Ryzen (1.50), while Xeon wins at b=1 (0.46-0.86). fp32 paged wins
  on Graviton except cross b=1 (1.02), is mixed on Ryzen (0.75-1.50), and loses
  at batch 8 on Xeon (1.04-1.20) and at b=1/4096 (1.08-1.11).

Ratios move between runs (duplicate-arm noise 0.92-1.04); a single 9-round run
is not enough near 1.00. Full per-row bands, protocols and raw data are in
[`docs/AUDIT.md`](docs/AUDIT.md) and [`docs/STATUS.md`](docs/STATUS.md).

## Quick start

### Install

```bash
uv sync --extra build                                # torch + mojo + this package
uv run --extra build python scripts/build.py --tests # build _C.so and run the kernel tests
uv run python tests/test_flash_api.py                # 35 checks vs SDPA
```

`CPU_ATTENTION_LIB` points the loader at another build. Without `_C.so`,
`cpu_attention.is_available()` is `False` and the entry points raise a clear
`RuntimeError`.

### Use

```python
import torch
from cpu_attention import flash_attn_func, flash_attn_with_kvcache

# Dense: q/k/v are [batch, seq, heads, head_dim]
out = flash_attn_func(q, k, v, causal=True)

# Paged decode: one query token per sequence
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,           # q [B,1,H,D], caches [blocks,H,page,D]
    k=k_new, v=v_new,              # written at cache_seqlens (length before write)
    cache_seqlens=lens,            # int32 [B]
    page_table=pages,              # int32 [B, max_blocks]
)
```

For an engine's per-layer path, call the thin aliases directly; they skip the
argument checks the runner already did:

```python
from cpu_attention import self_attention, cross_attention

out = self_attention(q, k, v, k_cache, v_cache, slot_mapping,
                     block_table, context_lens, scale)
out = cross_attention(q, k_cache, v_cache, block_table, cache_seqlens, scale)
```

### Get the best performance

1. **Set torch threads to the physical core count.** `torch.set_num_threads(n)`
   where `n` is sockets × cores, not SMT threads. The kernel runs on torch's
   OpenMP pool; oversubscribing makes it slower.
2. **Use bf16.** It is up to 3× faster than fp32 at the long dense rows on the
   servers (1.1-1.4× on the laptop); fp32 on x86 still goes through AMX (hi/lo
   split) and is fine at short contexts, but it is not the fast column on ARM
   or on the laptop.
3. **Leave the defaults alone on x86.** AMX is on by default and is worth
   2.8-9.8× over the AVX-512 path. Do not set `CPU_ATTENTION_AMX=0` unless you
   are A/B testing.
4. **Use page 32 on x86, page 16 on ARM** (the engine defaults), then measure
   before changing it.
5. **Stage cross-attention caches with the active backend.** `fp32/x86` caches
   are interchangeable with vLLM's; `bf16` caches are not. Use the engine's
   `stage_cross_kv` (it calls the active backend's `reshape_and_cache`) or
   `cpu_attention.reshape_and_cache` directly.
6. **Benchmark on the machine you will ship on**, with `--rounds 41` and a
   control arm, and judge on the minimum:
   `python tests/bench_vs_refs.py --vllm <ref.so> --rounds 41 --control`.
   The independent harness in [`audit/`](audit/) uses a naive fp32 reference.

## Documentation

* [`TECHNICAL.md`](TECHNICAL.md) — design, API details, page layout, numerics,
  tuning knobs, and historical measurements.
* [`docs/AUDIT.md`](docs/AUDIT.md) — independent 2026-10-23 audit of every
  performance claim on the three boxes, with raw data in `audit/logs/`.
* [`docs/STATUS.md`](docs/STATUS.md) — the scoreboard and the falsification
  ledger.
* [`AGENTS.md`](AGENTS.md) — handover notes and language-specific gotchas.

## License

Apache-2.0. The kernel is a port of vLLM's `csrc/cpu` attention implementation
(Apache-2.0); the Python API mirrors
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
(BSD-3).
