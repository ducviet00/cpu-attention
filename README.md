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
row, on three machines. It wins most `flash_attn_func` rows on all three and loses
none on the Axion and the C4D, but it ties or loses many decode rows, most of all on
the C4D. Numbers below are the single source of truth; `docs/Mojo.md` explains the
design and the experiments.

**Setup** (Mojo 1.2.0 `2016d0cd`, torch 2.14.1+cpu). Three 16-vCPU GCP spot VMs,
threads = physical cores, each measured `77889ea` (Oct 2026: one decode driver and
one vector flash policy for every CPU, `docs/Mojo.md` §3; large decode rows timed cold,
see Method) in one full-manifest run (`scripts/box_verify.sh`) against an independent
copy of `44035cc` as the previous build.

| box | CPU | threads | C++ paths it is compared with |
|---|---|---:|---|
| C4D | Google C4D (c4d-standard-16): AMD EPYC 9B45 (Zen 5), 8 cores / 16 threads, AVX-512 + BF16, no AMX | 8 | vec, scalar |
| Axion | Google Axion Neoverse-V2 (c4a-standard-16), 16 cores | 16 | neon, scalar |
| Xeon | Intel Xeon Platinum 8581C (c4-standard-16), 8 cores / 16 threads, AMX | 8 | amx, vec, scalar |

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
* **Cold decode rows**: a decode row whose per-arm K/V working set exceeds a quarter
  of the LLC is timed cold: before every sample, outside the timer and identically
  for every arm, a read-only sweep of twice the LLC evicts it. Interleaved warm, those
  rows were scored on whichever arm's lines the LLC replacement happened to keep: on
  the C4D (16-33 MB per arm, 32 MB L3) equal kernels read 0.9-1.7, while measured
  isolated, cold or warm, Mojo matched or beat C++ on every one of them. A serving KV
  cache is far larger than the LLC. The JSON records `cold` per row.
* **Verdict per row**: `ratio = Mojo min / fastest valid C++ min`; with
  `band = max(3%, |control / Mojo - 1|, twin)` a row is a **win** below
  `1 - band`, a **loss** above `1 + band`, and **unresolved** in between. `twin`
  is the spread between C++'s `amx` and `vec` decode arms, which run identical
  code, so it measures how far one process's minimum can sit from another's.

**Results** (win / unresolved / loss):

| rows | C4D | Axion | Xeon |
|---|---|---|---|
| `flash_attn_func` (78) | 73 / 5 / 0 | 60 / 18 / 0 | 64 / 14 / 0 |
| decode, allocating the output (60) | 12 / 48 / 0 | 35 / 24 / 1 | 14 / 43 / 3 |
| decode with `out=` (20) | 2 / 18 / 0 | 8 / 12 / 0 | 2 / 17 / 1 |
| public `flash_attn_with_kvcache` decode (8) | 7 / 1 / 0 | 8 / 0 / 0 | 7 / 1 / 0 |
| store-only (4) | 2 / 1 / 1 | 4 / 0 / 0 | 2 / 2 / 0 |
| **all (170)** | **96 / 73 / 1** | **115 / 54 / 1** | **89 / 77 / 4** |

**Where Mojo does not clearly win.**
* **C4D**: no decode or `flash_attn_func` losses. 66 of the 92 decode rows are timed cold
  (8 wins, 58 ties at the shared ~60-65 GB/s DRAM roof, where both backends stream the
  same bytes in the same order). The one loss here, store-only `b=1 ctx=1024` fp32
  (1.14), is fixed at `f1f3199`: the store-only path writes K through a register
  transpose (one vector store per K word row instead of a scalar store per dim and
  token), now 0.97 of C++ (bf16 0.21).
* **Axion**: 1 loss, decode bf16 batch 1 at context 4097 (1.03). Every bf16
  `flash_attn_func` row wins; fp32 ties or wins (no losses): its NEON policy was folded
  into the shared vector policy, now 2% slower than the dedicated one it replaced at
  head_dim 64 and ~13% at head_dim 32 (`docs/Mojo.md` §9), a cost taken for one
  implementation across CPUs.
* **Xeon**: 4 losses, all warm decode at 1.04-1.11: batch 1 at context 4096/4097 and
  fp32 `het`. This VM reports its socket's 260 MB L3, so only 8 decode rows cross the
  cold threshold; the rest are timed warm and interleaved as before.

**Caveats.**
* Rows bound by memory placement swing 20-30% with allocation order. The Xeon
  fp32 decode row `cross b=8 ctx=1024` measured 0.82-1.04 of C++ across four
  `--load-seed` values, while Mojo matched C++ in single-thread, four-thread
  and cold-cache controlled tests; treat those verdicts as unresolved until
  repeated across processes.
* Two Xeon runs of identical code differed by 5-10% on a few fp32 AMX
  `flash_attn_func` rows (the control arm moved with it, C++ did not); the
  cause is not found.
* Single-run counts move by 6-10 rows; read a row's verdict from repeated
  `--load-seed` runs, not from one table. Earlier 4-thread measurements on the
  8-vCPU boxes are in this file's history (`9a60079` and before).
* The `mojo-oracle-v1.1` tag is not on `origin`. These runs fetched it from a local
  clone; the C++ arms are built from it.

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
full-manifest A/B against BASE at threads = physical cores (`scripts/ab_summary.py`
summarizes the JSON). The tables above came from those scripts.

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
