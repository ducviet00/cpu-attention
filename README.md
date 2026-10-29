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
uv run --extra build python scripts/build.py --tests     # Mojo: one _C_<tier>.so per ISA tier
uv run --extra build python scripts/build_cpp.py --test  # C++ backend  -> cpu_attention/_C_cpp.so
```

The Mojo build writes one shared object per ISA tier for the host architecture
(x86-64: AVX-512-BF16 + AMX, AVX2, baseline; aarch64: BF16, NEON), each compiled
for an explicit target rather than the build host. The loader picks the best
tier the CPU reports; `CPU_ATTENTION_ISA=amx|vec|avx2|scalar|neon` forces one
and is refused if the CPU lacks it. The C++ build links the installed torch, so
rebuild it after a torch upgrade.

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

**Status: the performance objective is not met.** The unified Mojo backend (the
default) is measured against the *fastest numerically valid C++ path* on each
row, on three machines. It wins a clear majority of the `flash_attn_func` rows
on the C4D and the Axion, and about half of them on the Xeon, but it loses
or cannot resolve many decode rows on all three. Numbers below are the single
source of truth; `docs/Mojo.md` explains the design and the experiments.

**Setup** (Mojo 1.2.0 `2016d0cd`, torch 2.14.1+cpu; the kernel sources are identical at the
measured commits, `bbe8c1b` for Axion and Xeon and `c9bf977` for the C4D).

| box | CPU | threads | C++ paths it is compared with |
|---|---|---:|---|
| C4D | Google C4D (c4d-standard-8): AMD EPYC 9B45 (Zen 5), 4 cores / 8 threads, 32 MB L3, AVX-512 + BF16, no AMX | 4 | vec, scalar |
| Axion | Google Axion Neoverse-V2 (GCP c4a-standard-8) | 8 | neon, scalar |
| Xeon | Intel Xeon Platinum 8581C (GCP c4-standard-8, 4 cores / 8 threads, AMX) | 4 | amx, vec, scalar |

vLLM `csrc/cpu` at `db3b0f0` (AMX for Xeon bf16, VEC for x86 fp32 and the
C4D, NEON on the Axion) and torch SDPA are secondary references.

**Method.**
* **The manifest** (`benchmarks/manifest.py`, fixed before optimizing) has 170
  rows, each in bf16 and fp32: the 42 historical rows plus odd lengths around
  tile and page boundaries (17 ... 4097), unequal query/key lengths, noncausal,
  GQA/MQA, head_dim 32/64/128, fused-QKV strided inputs, page sizes 16/48/64,
  heterogeneous contexts, store-only, a preallocated `out=` and the public
  `flash_attn_with_kvcache` call.
* **One process per box, independently loaded arms** (`benchmarks/arms.py`):
  the build under test, a second independent copy of it (the noise floor), the
  oracle build (`mojo-oracle-v1.1`), and one arm per applicable C++ ISA path;
  interleaved, order rotated every round, the **minimum** decides. Rounds: 61
  for `flash_attn_func`, 121 for decode, one call per sample. Every arm's output is validated before it can be a
  baseline. Ours includes the K/V pack and the output allocation (not on `out=`
  rows); each arm stages its own cache from the same data.
* **Verdict per row**: `ratio = Mojo min / fastest valid C++ min`; with
  `band = max(3%, |control / Mojo - 1|, twin)` a row is a **win** below
  `1 - band`, a **loss** above `1 + band`, and **unresolved** in between. `twin`
  is the spread between C++'s `amx` and `vec` decode arms, which run identical
  code, so it measures how far one process's minimum can sit from another's.

**Results** (win / unresolved / loss):

| rows | C4D | Axion | Xeon |
|---|---|---|---|
| `flash_attn_func` (78) | 71 / 7 / 0 | 77 / 1 / 0 | 47 / 23 / 8 |
| decode, allocating the output (60) | 10 / 32 / 18 | 32 / 27 / 1 | 19 / 32 / 9 |
| decode with `out=` (20) | 3 / 12 / 5 | 8 / 12 / 0 | 7 / 11 / 2 |
| public `flash_attn_with_kvcache` decode (8) | 4 / 3 / 1 | 7 / 1 / 0 | 4 / 3 / 1 |
| store-only (4) | 3 / 1 / 0 | 4 / 0 / 0 | 0 / 3 / 1 |
| **all (170)** | **91 / 55 / 24** | **128 / 41 / 1** | **77 / 72 / 21** |

**Where Mojo does not clearly win.**
* **C4D** (re-measured at `c9bf977`: one full run, plus two `--load-seed` reruns of
  every loss row): 24 losses, none in `flash_attn_func` (71 wins, 7 ties). 14 of the
  24 stop being losses under another allocation order (some become wins), so they are
  placement noise. 8 lose in all three runs: fp32 batch-1 decode at context 4096/4097
  (1.04-1.20), fp32 batch-8 context-1024 `self` (1.04-1.06), fp32 batch-1 context-256
  `self` (1.03-1.05) and bf16 batch-8 context-1024 page 48 `self` (1.05-1.07). Per
  thread Mojo matches or beats C++ on every shape isolated (0.92-0.98 on one thread);
  the gap appears only with 4 threads streaming a working set that spills out of the
  32 MB L3 (about 33 MB per call), where C++ sustains ~62 GB/s and Mojo ~56. The cause
  was not found: tile length, key-group width, thread pinning, wake-up policy and thread
  skew each made no difference, and this VM exposes no hardware performance counters.
  bf16 page 48 is 8% slower per core than C++ on one thread (parity without the
  compile-time page-48 body); 4 threads hide it.
* **Axion** (branch `axion-mojo`, re-measured; the other columns are from earlier
  commits): 1 loss, 41 unresolved, all decode except one `flash_attn_func` tie. The one loss,
  `engine cross b=8 ctx=1025 bf16` (1.03), re-measured 0.90 / 0.97 / 0.92 under
  `--load-seed` 1-3 (placement noise). The unresolved rows sit at 0.95-1.03 of C++ NEON,
  mostly batch 8-32 at context ~1024 and batch-1 context 4096, which stream 16-67 MB
  per call; the objective of a clear win on every row is **not met** there.
* **Xeon** (re-measured Oct 2026 after the bf16 Q-panel change; the other two boxes
  are as measured before it): 21 losses. `flash_attn_func`: 8, 1.04-1.12, seven of
  them fp32 head_dim 128 at 129-1025 tokens (and `513x257:8:64` fp32), plus bf16
  `129:16:128`. On four threads the fp32 head_dim-128 rows run 4-5% behind C++
  while Mojo is 3% *faster* on one thread; the cause is not found (not SMT
  placement, workspace alignment or offset). Decode: 13, most of them fp32 at
  batch 8, context ~1024, which move 0.82-1.02 with the allocation order (see the
  caveat). Store-only `b=32 ctx=1` fp32 lost once (1.07 against `cpp_scalar`).

**Caveats.**
* Rows bound by memory placement swing 20-30% with allocation order. The Xeon
  fp32 decode row `cross b=8 ctx=1024` measured 0.82-1.04 of C++ across four
  `--load-seed` values, while Mojo matched C++ in single-thread, four-thread
  and cold-cache controlled tests; treat those verdicts as unresolved until
  repeated across processes.
* Two Xeon runs of identical code differed by 5-10% on a few fp32 AMX
  `flash_attn_func` rows (the control arm moved with it, C++ did not); the
  cause is not found.
* Run-to-run: two Xeon runs of the same decode code gave 64 / 76 / 30 (earlier table) and 70 / 80 / 20 (Oct 2026, no oracle arm: the `mojo-oracle-v1.1` tag is absent from this clone), so single-run counts on this box move by about 6 rows, and the non-`flash_attn_func` Xeon rows above are within that noise of the earlier run. The consistent Xeon losses in both runs are the fp32 decode rows at batch 8, context ~1024 (1.06-1.31) and `flash_attn_func` head_dim 128 at 257-512 tokens (1.03-1.10).
* The Axion box was measured once. On the C4D the 32 MB L3 makes the verdict of
  decode rows with a 15-30 MB working set depend on which arm's cache happened
  to be resident; repeat them with `--load-seed` before believing a gap.
* The C4D run has no old-Mojo (`mojo-oracle-v1.1`) arm: that tag is not in this
  repository or on `origin`, so the harness's Python glue came from commit `290b0a7`
  (the commit that introduced the oracle setting). The Mojo and C++ arms are
  unaffected. The fast-encdec engine test (`tests/test_cpu_attention.py`) was not run
  on the C4D: that repository is not on the machine.

## Benchmark it yourself

```bash
scripts/build_vllm_ref.sh             # vLLM reference -> ~/vllm_cpu_build/*.so (pinned commit)
uv run python benchmarks/bench.py --manifest full \
    --vllm-bf16 ~/vllm_cpu_build/vllm_cpu_attn_bench.so --vllm-fp32 ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --rounds 61 --paged-rounds 121 --json /path/outside/the/repo/run.json
```

* On an AMX box pass `vllm_cpu_attn_bench_amx.so --vllm-isa-bf16 amx` for bf16
  and `--threads <physical cores>`; on aarch64 pass `--vllm-isa-bf16 neon
  --vllm-isa-fp32 neon`. `--block-calls 40` is only for a desktop or laptop with
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
uv run python tests/test_regressions.py    # 412 checks: odd/unequal lengths, masked rows, poisoned caches, strides, scale, workers, concurrency
uv run python tests/test_amx.py --dtype both   # AMX policy (AMX CPUs; fails if AMX is present but unusable)
```

The default backend is Mojo; prefix `CPU_ATTENTION_BACKEND=cpp` to test C++, and
`CPU_ATTENTION_ISA=<tier>` to test a lower Mojo tier. The engine-level test
lives in fast-encdec: `PYTHONPATH=. .venv/bin/python tests/test_cpu_attention.py`
(it passes strided chunks of a fused projection; 45 checks).

Accuracy gates (all pass on all three boxes):

| case | limit |
|---|---|
| `flash_attn_func` fp32 | <= 2e-4 vs the naive fp32 reference |
| `flash_attn_func` bf16 | <= 3e-2 |
| paged fp32 | <= 1e-6 (fp64 arbitrates an fp32 reference that is itself out of the gate) |
| paged bf16, x86 | <= 2 bf16 ulp at the row peak vs C++ |
| paged bf16, ARM | the same 2 ulp, and <= 1e-3 *above the bf16 output-rounding floor* |

The ARM "<= 1e-3" is not met literally: rounding the exact result to bf16 alone
exceeds 1e-3 on 77 of 252 sweep cases, so no bf16 output can satisfy an absolute
1e-3. The tests check the excess over that floor (`tests/reference.py`); see
`docs/Mojo.md` §5.

## Documentation

* [`docs/CPP.md`](docs/CPP.md): the C++ backend's design, per-ISA techniques,
  numerics, measurement rules, falsification ledger and open problems.
* [`docs/Mojo.md`](docs/Mojo.md): the Mojo backend's design, ISA tiers,
  numerics, call path, defects found, falsification ledger and Mojo 1.2.0
  notes.
* [`AGENTS.md`](AGENTS.md): working rules for anyone changing a kernel.

## License

Apache-2.0. Both backends follow vLLM's `csrc/cpu` attention (Apache-2.0); the
Python API mirrors [flash-attention](https://github.com/Dao-AILab/flash-attention)
(BSD-3).
