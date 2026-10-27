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
* Two interchangeable backends behind one API:
  * **C++** (`cpu_attention/csrc/`, [`docs/CPP.md`](docs/CPP.md));
  * **Mojo** (`cpu_attention/_mojo/`, [`docs/Mojo.md`](docs/Mojo.md)).

  `CPU_ATTENTION_BACKEND=mojo|cpp|auto` selects one. The default is `mojo`;
  `auto` picks `cpp` on x86-64 and `mojo` on aarch64. Both load in one
  process, so they can be A/B'd.
* Linux, PyTorch, Python >= 3.10.
* Not supported, and rejected with `NotImplementedError` rather than silently
  ignored: rotary, ALiBi, softcap, sliding windows, varlen/packed QKV,
  dropout, FP8, backward.

## Install and build

```bash
uv sync --extra build                                    # torch + the mojo wheel
uv run --extra build python scripts/build.py --tests     # Mojo backend -> cpu_attention/_C.so
uv run --extra build python scripts/build_cpp.py --test  # C++ backend  -> cpu_attention/_C_cpp.so
```

Both builds target the machine they run on, so rebuild per box. The C++
build links the installed torch, so rebuild it after a torch upgrade.

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
  transposed inside the page), and each backend defines its own. Stage a cache
  through the *active* backend's `reshape_and_cache`; a cache written by one
  backend is garbage to the other.
* **Strides.** q/k/v may be strided chunks of a fused projection: any token
  stride, with heads and dims contiguous.
* **`causal` on the paged path** is accepted and has no effect, because one
  query at the last position sees every key.

## Performance

**Setup.**
* **Boxes:** Ryzen 7 7840H (8 cores, AVX-512 + BF16, no AMX) at 8 threads;
  Google Axion Neoverse-V2 (GCP c4a, 8 cores) at 8 threads; Xeon Platinum 8581C
  (GCP C4, 4 cores / 8 threads, AMX) at 4 threads.
* **Baselines:** vLLM `csrc/cpu` at `db3b0f0`, built for the ISA the box
  uses. That is **AMX for Xeon bf16**, VEC for x86 fp32 and for the Ryzen, and
  NEON on the Axion.

**Method.**
* One process per box (Ryzen dense: one per row), all arms interleaved with
  rotated order, plus a duplicate C++ arm as the noise floor.
* Rounds: 61 dense / 121 paged on the servers; 41 dense / 41 x 40-call blocks
  paged on the Ryzen.
* Times are the **minimum**, in µs, listed as **C++ / Mojo / vLLM**.

**Ratios.**
* `c/best` is C++ divided by the faster of Mojo and vLLM; bold is a clear win
  (< 0.97).
* `c/SDPA` is C++ against `torch.nn.functional.scaled_dot_product_attention`
  in the same process, from a separate 21-round dense run.

**What is charged to whom.**
* Ours includes the K/V pack and the output allocation.
* vLLM's cache is staged, and its output preallocated, outside the timer.

Rows are `seq:heads:head_dim`, batch 1, causal. Paged rows use 16 heads,
head_dim 64, page 32 (x86) / 16 (ARM).

**Summary.**

| | Ryzen 7840H | Axion Neoverse-V2 | Xeon 8581C |
|---|---|---|---|
| C++ faster than vLLM | 42/42 | 42/42 | 42/42 |
| C++ faster than Mojo | 40/42 | 40/42 | 42/42 |
| clear wins vs the best baseline (< 0.97) | 38 | 35 | 39 |
| noise band (0.97-1.04) / losses | 3 / 1 | 7 / 0 | 3 / 0 |
| C++ / SDPA (dense) | 0.22-0.71 | 0.27-0.85 | 0.17-0.64 |

**Where C++ does not clearly win.**
* The one loss is Ryzen paged self b=32 ctx=1024 bf16, at 1.16 of Mojo.
* On the Axion, Mojo stays the faster baseline on seven dense rows, at
  0.97-1.01.

Details and open problems are in [`docs/CPP.md`](docs/CPP.md).

### Dense, bf16

| shape | Ryzen C++ / Mojo / vLLM µs | c/best | c/SDPA | Axion C++ / Mojo / vLLM µs | c/best | c/SDPA | Xeon C++ / Mojo / vLLM µs | c/best | c/SDPA |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| 128:8:64 | 28.2 / 42.6 / 40.2 | **0.70** | 0.25 | 23.2 / 24.7 / 24.7 | **0.94** | 0.41 | 24.7 / 28.5 / 31.2 | **0.87** | 0.43 |
| 512:8:64 | 167 / 264 / 359 | **0.63** | 0.22 | 216 / 233 / 264 | **0.93** | 0.36 | 181 / 242 / 266 | **0.75** | 0.32 |
| 1024:8:64 | 699 / 1,037 / 1,456 | **0.67** | 0.30 | 870 / 882 / 984 | 0.99 | 0.53 | 629 / 838 / 992 | **0.75** | 0.43 |
| 2048:8:64 | 2,854 / 4,124 / 5,899 | **0.69** | 0.38 | 3,323 / 3,446 / 3,742 | **0.96** | 0.62 | 2,307 / 3,069 / 3,686 | **0.75** | 0.50 |
| 4096:8:64 | 11,484 / 16,499 / 23,165 | **0.70** | 0.38 | 13,044 / 13,903 / 15,028 | **0.94** | 0.69 | 8,837 / 11,929 / 14,852 | **0.74** | 0.53 |
| 256:16:128 | 171 / 339 / 359 | **0.51** | 0.25 | 202 / 216 / 237 | **0.93** | 0.37 | 183 / 237 / 234 | **0.78** | 0.38 |
| 1024:16:128 | 2,652 / 5,109 / 5,697 | **0.52** | 0.29 | 2,870 / 2,936 / 3,100 | 0.98 | 0.55 | 1,916 / 2,386 / 2,574 | **0.80** | 0.52 |
| 2048:16:128 | 10,738 / 20,130 / 22,321 | **0.53** | 0.35 | 11,063 / 11,676 / 12,188 | **0.95** | 0.63 | 6,852 / 8,808 / 9,515 | **0.78** | 0.56 |
| 4096:16:128 | 43,430 / 85,527 / 89,784 | **0.51** | 0.37 | 43,720 / 46,636 / 48,168 | **0.94** | 0.70 | 27,684 / 34,363 / 38,023 | **0.81** | 0.64 |
| 512:32:128 | 1,374 / 2,728 / 3,061 | **0.50** | 0.22 | 1,427 / 1,553 / 1,618 | **0.92** | 0.35 | 1,160 / 1,491 / 1,525 | **0.78** | 0.41 |
| 1024:8:32 | 425 / 657 / 805 | **0.65** | 0.32 | 561 / 617 / 686 | **0.91** | 0.49 | 456 / 610 / 731 | **0.75** | 0.35 |

### Dense, fp32

| shape | Ryzen C++ / Mojo / vLLM µs | c/best | c/SDPA | Axion C++ / Mojo / vLLM µs | c/best | c/SDPA | Xeon C++ / Mojo / vLLM µs | c/best | c/SDPA |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| 128:8:64 | 24.9 / 30.8 / 28.6 ¹ | **0.87** | 0.36 | 43.8 / 48.6 / 51.3 | **0.90** | 0.27 | 46.4 / 53.3 / 62.5 | **0.87** | 0.34 |
| 512:8:64 | 306 / 349 / 344 | **0.89** | 0.39 | 557 / 574 / 638 | 0.97 | 0.32 | 330 / 426 / 626 | **0.77** | 0.22 |
| 1024:8:64 | 1,190 / 1,310 / 1,306 | **0.91** | 0.55 | 2,153 / 2,170 / 2,438 | 0.99 | 0.47 | 1,195 / 1,407 / 2,246 | **0.85** | 0.33 |
| 2048:8:64 | 4,871 / 5,376 / 5,313 | **0.92** | 0.65 | 8,493 / 8,473 / 9,498 | 1.00 | 0.60 | 4,438 / 5,191 / 8,661 | **0.85** | 0.35 |
| 4096:8:64 | 20,360 / 21,050 / 21,127 | **0.97** | 0.71 | 34,212 / 37,441 / 37,366 | **0.92** | 0.71 | 17,537 / 20,404 / 33,530 | **0.86** | 0.40 |
| 256:16:128 | 310 / 380 / 382 | **0.82** | 0.40 | 544 / 581 / 729 | **0.93** | 0.41 | 375 / 509 / 685 | **0.74** | 0.25 |
| 1024:16:128 | 5,179 / 6,495 / 5,377 | **0.96** | 0.56 | 8,202 / 8,142 / 10,314 | 1.01 | 0.66 | 3,518 / 4,570 / 8,366 | **0.77** | 0.28 |
| 2048:16:128 | 20,307 / 22,622 / 20,700 | 0.98 | 0.65 | 32,149 / 38,069 / 40,523 | **0.84** | 0.72 | 14,961 / 17,502 / 32,305 | **0.85** | 0.34 |
| 4096:16:128 | 78,747 / 133,861 / 83,781 | **0.94** | 0.67 | 126,285 / 159,610 / 159,846 | **0.79** | 0.85 | 77,076 / 87,543 / 123,415 | **0.88** | 0.49 |
| 512:32:128 | 2,871 / 3,634 / 2,889 | 0.99 | 0.42 | 4,073 / 4,617 / 5,304 | **0.88** | 0.40 | 2,135 / 2,860 / 4,610 | **0.75** | 0.17 |
| 1024:8:32 | 656 / 743 / 732 | **0.90** | 0.50 | 1,262 / 1,280 / 1,476 | 0.99 | 0.46 | 843 / 1,017 / 1,350 | **0.83** | 0.39 |

### Paged, bf16

| case | Ryzen C++ / Mojo / vLLM µs | c/best | Axion C++ / Mojo / vLLM µs | c/best | Xeon C++ / Mojo / vLLM µs | c/best |
|---|---|---:|---|---:|---|---:|
| self b=1 ctx=256 | 3.6 / 5.6 / 5.9 | **0.63** | 6.0 / 12.5 / 10.0 | **0.60** | 7.9 / 11.9 / 12.0 | **0.66** |
| cross b=1 ctx=256 | 5.3 / 8.2 / 6.4 | **0.83** | 5.5 / 11.2 / 7.3 | **0.75** | 7.2 / 10.5 / 7.5 | **0.96** |
| self b=8 ctx=256 | 17.1 / 22.1 / 32.0 | **0.78** | 29.9 / 51.2 / 75.2 | **0.58** | 59.5 / 86.1 / 104 | **0.69** |
| cross b=8 ctx=256 | 17.2 / 22.3 / 26.9 | **0.77** | 28.7 / 50.3 / 71.0 | **0.57** | 57.2 / 82.5 / 92.9 | **0.69** |
| self b=8 ctx=1024 | 356 / 381 / 427 | **0.93** | 254 / 290 / 300 | **0.87** | 311 / 368 / 385 | **0.85** |
| cross b=8 ctx=1024 | 355 / 378 / 414 | **0.94** | 269 / 315 / 307 | **0.88** | 306 / 357 / 368 | **0.86** |
| self b=32 ctx=1024 | 2,856 / 2,472 / 2,859 | 1.16 | 1,163 / 1,508 / 1,535 | **0.77** | 2,837 / 4,076 / 3,095 | **0.92** |
| cross b=32 ctx=1024 | 2,371 / 2,366 / 2,750 | 1.00 | 1,249 / 1,590 / 1,606 | **0.79** | 2,676 / 4,005 / 2,947 | **0.91** |
| self b=1 ctx=4096 | 37.8 / 77.0 / 82.9 | **0.49** | 83.5 / 112 / 119 | **0.74** | 164 / 185 / 187 | **0.89** |
| cross b=1 ctx=4096 | 38.9 / 76.1 / 81.1 | **0.51** | 82.1 / 109 / 114 | **0.75** | 163 / 185 / 183 | **0.89** |

### Paged, fp32

| case | Ryzen C++ / Mojo / vLLM µs | c/best | Axion C++ / Mojo / vLLM µs | c/best | Xeon C++ / Mojo / vLLM µs | c/best |
|---|---|---:|---|---:|---|---:|
| self b=1 ctx=256 | 4.6 / 6.9 / 7.2 | **0.66** | 9.5 / 15.2 / 15.9 | **0.63** | 8.7 / 12.0 / 16.1 | **0.73** |
| cross b=1 ctx=256 | 4.5 / 6.4 / 5.6 | **0.80** | 8.2 / 13.5 / 12.1 | **0.68** | 8.4 / 10.8 / 12.9 | **0.78** |
| self b=8 ctx=256 | 32.9 / 51.2 / 53.9 | **0.64** | 64.5 / 96.2 / 123 | **0.67** | 144 / 153 / 159 | **0.94** |
| cross b=8 ctx=256 | 31.8 / 49.4 / 49.1 | **0.65** | 63.1 / 92.1 / 121 | **0.69** | 142 / 146 / 145 | 0.98 |
| self b=8 ctx=1024 | 1,070 / 1,132 / 1,163 | **0.95** | 440 / 626 / 606 | **0.73** | 594 / 837 / 915 | **0.71** |
| cross b=8 ctx=1024 | 1,037 / 1,088 / 1,167 | **0.95** | 490 / 627 / 633 | **0.78** | 592 / 822 / 918 | **0.72** |
| self b=32 ctx=1024 | 4,788 / 5,226 / 5,082 | **0.94** | 2,667 / 2,775 / 2,930 | **0.96** | 4,340 / 4,898 / 4,760 | **0.91** |
| cross b=32 ctx=1024 | 4,637 / 4,788 / 4,922 | **0.97** | 2,670 / 2,791 / 2,997 | **0.96** | 4,419 / 4,900 / 4,685 | **0.94** |
| self b=1 ctx=4096 | 285 / 322 / 337 | **0.89** | 222 / 251 / 256 | **0.89** | 311 / 334 / 318 | 0.98 |
| cross b=1 ctx=4096 | 284 / 323 / 334 | **0.88** | 224 / 248 / 262 | **0.90** | 310 / 333 / 314 | 0.99 |

¹ Re-run in isolation (median of 3 runs). The matrix process read 1.33 for this
row because it ran straight after the test suites; see `docs/CPP.md` §7.

## Benchmark it yourself

```bash
scripts/build_vllm_ref.sh             # vLLM reference -> ~/vllm_cpu_build/*.so (pinned commit)
python benchmarks/bench.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa vec \
    --rounds 41 --json out.json        # add --block-calls 40 on a laptop
```

* Pass the AMX reference (`vllm_cpu_attn_bench_amx.so --isa amx --dtype bf16`)
  for bf16 on an AMX box, and `--isa neon` on aarch64.
* Judge on the minimum, and only against the duplicate arm's spread. Never
  compare two separate runs: these machines drift 5-15% between processes.

## Getting the best performance

1. **Set torch threads to the physical core count.** The kernels run on torch's
   OpenMP pool and cap themselves at physical cores; SMT threads only
   oversubscribe it.
2. **Use bf16.** It is up to 3x faster than fp32 on both dense and paged rows.
   The shortest Ryzen dense row (128:8:64) is the only case where fp32 is
   faster (24.9 vs 28.2 µs).
3. **Pass `out=`** to `self_attention` / `cross_attention` on the decode path.
   It saves the allocation, which is a sizeable share of a batch-1 call.
4. **Page size.** Use 32 on x86 and 16 on ARM (the engine defaults); measure
   before changing it.
5. **Leave AMX on** (the default on AMX CPUs). It is worth 2.8-9.8x over
   AVX-512 for dense bf16.

## Tests

```bash
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_dense.py        # dense vs a naive fp32 reference (89 checks)
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_flash_api.py    # flash-shaped API and its rejections
CPU_ATTENTION_BACKEND=cpp uv run python tests/fuzz_dense.py --cases 80 --max-seq 900
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_paged_sweep.py  # 504 paged cases, both dtypes
uv run python tests/test_amx.py                                    # Mojo AMX entries (AMX boxes)
```

Drop `CPU_ATTENTION_BACKEND=cpp` to test the Mojo backend.

The engine-level test lives in fast-encdec:
`CPU_ATTENTION_BACKEND=cpp PYTHONPATH=. .venv/bin/python tests/test_cpu_attention.py`.

Accuracy gates:

| case | limit |
|---|---|
| dense fp32 | <= 2e-4 vs the naive reference |
| dense bf16 | <= 3e-2 |
| paged fp32 | <= 1e-6 |
| paged bf16, x86 | 0-2 ulp between backends |
| paged bf16, ARM | <= 1e-3 |

## Documentation

* [`docs/CPP.md`](docs/CPP.md): the C++ backend's design, per-ISA techniques,
  numerics, measurement rules, falsification ledger and open problems.
* [`docs/Mojo.md`](docs/Mojo.md): the Mojo backend's kernels, ISA paths,
  language notes, failure modes, and the levers already exhausted.
* [`AGENTS.md`](AGENTS.md): working rules for anyone changing a kernel.

## License

Apache-2.0. Both backends follow vLLM's `csrc/cpu` attention (Apache-2.0); the
Python API mirrors [flash-attention](https://github.com/Dao-AILab/flash-attention)
(BSD-3).
