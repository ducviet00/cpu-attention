# The C++ backend

`cpu_attention/csrc/` is a C++20 implementation of the library's two kernels
(paged decode and dense prefill), behind the same Python API as the Mojo
backend (`docs/Mojo.md`). One build per architecture, runtime ISA dispatch, and
it runs on torch's OpenMP pool. Its structure follows vLLM's `csrc/cpu`
(Apache-2.0); the kernels are rewritten.

```bash
uv run --extra build python scripts/build_cpp.py          # -> cpu_attention/_C_cpp.so
uv run --extra build python scripts/build_cpp.py --test   # + ISA check, dense + flash API tests
CPU_ATTENTION_BACKEND=cpp python -c "import cpu_attention"  # select it (default: mojo)
```

## 1. Status

C++ is now the *competitor* the unified Mojo backend is judged against: the
README's Performance section holds the current per-row verdicts (Mojo against
the fastest valid C++ ISA path on the C4D, Axion and Xeon) and is the single
source of truth for numbers. This section is the record of the C++ campaign's
own final verification (2026-10-28), when C++ was compared with the previous
Mojo build and vLLM `csrc/cpu`; the per-row tables of that comparison were
retired from the README because they describe a Mojo build that no longer exists.

| box | vLLM baseline | clear wins / 42 | noise band (0.97-1.04) | losses | acceptance (>= 90% clear, no loss) |
|---|---|---:|---:|---:|---|
| Xeon Platinum 8581C (GCP C4, 4C/8T) | AMX for bf16, VEC for fp32 | 39 | 3 | 0 | pass |
| Ryzen 7 7840H laptop (8 cores, AVX-512, no AMX), the campaign's box then | VEC | 38 | 3 | 1 | not met |
| Axion Neoverse-V2 (GCP c4a, 8 cores) | NEON | 35 | 7 | 0 | not met |

"Clear win" meant below 0.97 of the *faster* of the then-Mojo and vLLM on that
row. Against vLLM alone C++ was faster on all 126 rows; against SDPA, every dense
row on every box. `CPU_ATTENTION_BACKEND` defaults to `mojo`; an explicit `auto`
selects C++ on x86-64 and Mojo on aarch64.

The acceptance boxes named in the original plan were Graviton 5 and Xeon
6975P-C. They were unreachable, so the GCP machines above stand in and are
labelled as what they are.

## 2. Build and dispatch

| TU | flags |
|---|---|
| all | `-O3 -fPIC -std=c++20 -fopenmp` (no `-march=native`) |
| `vec.cpp` | `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2` |
| `amx.cpp` | the vec flags plus `-mamx-tile -mamx-bf16 -mamx-int8` |
| `neon.cpp` | `-march=armv8.2-a+bf16+i8mm` |
| `binding.cpp` | torch's include dirs; the module links `torch_python`, `torch_cpu`, `c10` |

- **Dispatch:** CPUID on x86 (BF16 is leaf 7 *subleaf 1*), HWCAP on aarch64.
- **AMX** additionally needs `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`,
  done once in `cpu_isa.cpp::amx_init()`.
- **Overrides:** `CPU_ATTENTION_ISA=scalar|vec|amx|neon` forces a path for A/B.
  A forced ISA the CPU lacks is refused.
- **Torch coupling:** because the binding links torch, the module belongs to
  the torch it was built with. Rebuild after upgrading torch.

```
csrc/cpu_attn.cpp / .hpp   entry points, ISA selection, workspace
csrc/cpu_attn_impl.hpp     shared core: jobs, per-thread workspace, dense/paged/store drivers
csrc/cpu_attn_vec.hpp      AVX-512 fp32/bf16 policies, dense + paged
csrc/cpu_attn_amx.hpp      AMX bf16 + fp32 hi/lo dense policy (paged forwards to vec)
csrc/cpu_attn_neon.hpp     NEON bfmmla/bfdot policies, dense + paged
csrc/cpu_attn_scalar.hpp   portable fallback
csrc/cpu_isa.cpp / .hpp    CPUID / getauxval / XTILEDATA permission
csrc/utils.hpp             aligned buffers, env knobs, cache sizes, bf16 helpers
csrc/binding.cpp           CPython C API module (no pybind)
csrc/{vec,amx,neon,scalar}.cpp   one TU per ISA, compiled with that ISA's flags
```

## 3. Design

**The driver/policy split.** `cpu_attn_impl.hpp` owns scheduling, the online
softmax bookkeeping, masks and the workspace. Each ISA file supplies a static
*policy*: the packed K/V layout, `pack`, `qk`, `softmax`, `pv` and, for the
paged path, `store_head` / `attend_head`. The optional hooks (`store_bf16`,
`store_f32`, `zeroes_acc`, `logit_aux_floats`) are detected with `requires`
rather than declared, so a policy pays only for what it uses.

**Dense work item = (batch, q head, `rq` consecutive query rows).**
- **Tiling.** The key range is walked in tiles of `kt` (256). Inside a tile the
  rows go in register tiles of `RQT` (8), so K/V are read once per `rq` rows
  (~16 KB of accumulator) instead of once per 8.
- **Causal bound.** Bottom-right aligned: row `i` keeps keys
  `j <= i + (seq_k - seq_q)`. The key loop stops at the last row's limit, and
  each row chunk clamps to its own, so no fully masked block is computed. Without
  this a causal call does up to 4.8x the useful work at rq=32.
- **Two dispatch orders.**
  - *Fused:* a thread packs a (batch, kv head) and then attends all of its query
    blocks. No barrier, and the slab stays in that core's L2.
  - *Split:* every head is packed, then a barrier, then a dynamic boustrophedon
    over query blocks, so concurrent threads share one head's slab through L3.

  Fused is chosen while `workers x slab <= 0.6 LLC`; it is 1.08-1.15x faster
  where both are legal.
- **Partial register tiles.** A partial tile dispatches a 2- or 4-row
  instantiation, rather than repeating the last query up to 8 times.

**Paged work item = (sequence, q head), end to end.** One scan of that head's
cache, with the accumulator in registers.
- **MHA:** the K/V store and the attend share one OpenMP region; a second
  fork-join is ~1.1 us, a sixth of a batch-1 call.
- **GQA/MQA:** a barrier separates the store from the attend, and the store runs
  once per *kv* head.

**Call path.**
- *Dense:* `dense_make_plan` + `dense_forward`. The geometry rides in one
  int64 plan buffer, so a call marshals five pointers.
- *Paged:* `paged_call(q, k, v, k_cache, v_cache, slots, block_table, lens,
  scale, out)`.
  - It reads pointers, sizes and strides from the tensors themselves.
  - It checks the layouts: heads/dims contiguous, any token stride (the engine's
    q/k/v are chunks of one fused projection).
  - It keeps a per-thread geometry -> workers cache, and allocates `out` with
    `at::empty`. A geometry the thread has not seen returns `NotImplemented`;
    Python then registers it with the worker policy (`_kernel._workers`) and
    calls again.
  - At batch 1 this removed 1.2-1.5 us of a 4.6 us call. It is what moved cross
    b=1 ctx=256 from 1.09-1.20 of vLLM to 0.75-0.96 on every box.

**Workspace.** Per thread and per geometry, 64-byte aligned, never returned to
the allocator. A large `torch.empty` is an mmap, so freeing it means faulting
the pages in again on the next call.

## 4. Layouts

This backend defines its own cache layouts and `reshape_and_cache` writes them,
so caches are not interchangeable with Mojo's or vLLM's.

| path | K | V |
|---|---|---|
| vec dense fp32 | `[tile of 64 keys][hd][64]` | in place (while V fits LLC/2), else `[pad][hd]` |
| vec dense bf16 | `[tile][hd/2][64][2]` (dim pair for `dpbf16ps`) | `[key/2][hd][2]` (key pair, for `dpbf16ps` P@V) |
| vec paged fp32 | `[hd][page]` | `[page][hd]` |
| vec paged bf16 | `[hd/2][page][2]` | `[page][hd]` |
| amx dense bf16 | `[dim pair][key]` | `[key pair][dim]` (both `tdpbf16ps` operands) |
| neon dense fp32 | `[hd][pad]` | `[pad][hd]` |
| neon dense bf16 | `[hd/4][pad/2][8]` | `[hd/2][pad/4][8]` (`bfmmla` operands) |
| neon paged | `[hd/2][page][2]` (fp32 `[hd][page]`) | `[page/2][hd][2]` |

K^T is blocked every 64 keys. A plain `[hd][seq_k]` transpose strides `seq_k*4`
bytes between dimensions and takes a TLB miss on every load; the blocked form is
1.17x faster at 4096:16:128 fp32.

## 5. Techniques, per ISA

### AVX-512 (`cpu_attn_vec.hpp`)

**Dense fp32.**
- **Q@K^T** is an 8-row x 16-key register tile: one K load plus eight
  broadcast-FMAs per dimension. Zen 4's `vfmadd zmm, zmm, m32bcst` is one
  macro-op whose broadcast does not contend for the 512-bit load port, so the
  "load:FMA ratio" model that predicts a bigger tile wins is wrong on this core
  (a 2x16-key variant was 1.10-1.15x slower).
- **Roof.** Both phases run at 72-85% of the 1 zmm-FMA/cycle roof (8.8 ms each
  at 2048:16:128 against 7.5 ms ideal). The remaining fp32 D=128 cost is the
  pack, which vLLM does not pay (its cache is staged outside the timer).
- **V is read where it lies** while all of V fits in half the LLC. The copy only
  buys a unit stride: 0.95 at 1024:16:128 and 512:32:128. Past that, a strided
  V is a DRAM stream the prefetchers lose (1.32x slower at 4096:16:128; gating
  at the full LLC is 1.17x slower at 2048:16:128).
- **Streaming stores** write the packed K^T and V when all slabs exceed the LLC.
  The slab goes to DRAM anyway, and an ordinary store would first read the line
  (0.96-0.97 on the long rows).
- **The softmax** folds the scale into the exponential, so there is no write-back
  pass. `exp2` is a degree-5 polynomial plus an exponent bit-insert (~10
  instructions against `std::exp`'s ~22).

**Dense bf16.** `dpbf16ps` does Q@K^T on dim pairs. **P@V is also
`dpbf16ps`**: the contraction runs on key pairs, with V packed as key pairs and
the probabilities written as bf16 pairs by the softmax. That is 2.05x on the
phase and 1.33-1.45x on the whole bf16 matrix. The same trick does not pay for
fp32: hi/lo needs three products.

**Paged.**
- bf16 uses a dim-outer Q@K^T with 4 independent 16-key groups and two P@V
  lanes. fp32 uses a dim-inner Q@K^T with four partial dot products, and one P@V
  lane.
- The bf16 exponential is degree 2: it sits inside the bf16 store's rounding
  two instructions later. fp32 uses degree-6 Taylor; the degree-5 minimax's
  3.4e-6 was the entire fp32 error budget.

### AMX (`cpu_attn_amx.hpp`)

- **Matmuls.** `tdpbf16ps` does both phases, with 16-row tiles of query rows;
  K and V are packed into the two operand layouts. fp32 is computed as bf16
  hi/lo (three products), accurate to 3-8e-5 against the 2e-4 gate.
- **Softmax and pack.** `kt=256` beat 128 and 512. The softmax writes P 32 keys
  at a time (one `cvtne2ps2bf16`, one full-line store). `exp2` uses
  magic-constant rounding and a minimax degree-4 fp32 polynomial (relative
  error 2.8e-6): fp32 0.90-0.98, bf16 0.95-0.99.
- **Output epilogue.** It uses non-temporal stores, through the driver's
  `store_f32` / `store_bf16` hooks. The output write was 15-17% of a short call,
  because every line paid a read-for-ownership on a buffer last written by
  another core. This took fp32 128:8:64 from 1.07 of Mojo to 0.87. A per-row
  `sfence` costs 1.25-2.4x; the OpenMP region's closing barrier (a locked
  operation) orders the stores.
- **Measured shares.** Both matmuls run at 55-87% of the `tdpbf16ps` roof. The
  softmax is the largest single phase at D=64 (about 45%). The pack is
  DRAM-bound.
- **The paged path forwards to vec.** With one query per sequence, AMX's 16-row
  tiles are idle for MHA.

### NEON (`cpu_attn_neon.hpp`)

- **bf16 dense** uses `bfmmla` for both phases in 2x2 fp32 tiles (Q@K^T: 2 rows
  x 4 dims against 2 keys x 4 dims; P@V: 2 rows x 4 keys against 2 dims x 4
  keys). K and V are packed into the paired layouts above.
- **Q@K^T epilogue (bf16).** The row maxima stay in registers, and a block with
  no masked key skips the mask tests: 196 -> 65 instructions per 8 keys.
  `exp2` builds `2^n` with an integer add into the exponent.
- **fp32 dense** uses broadcast-lane `fmla` with Q read from a contiguous
  per-chunk copy. It is at the scalar-pipe roof: 14.2 cycles/row-key = 32 FMAs
  at 2.25/cycle, and replacing a quarter of them with 3x-denser `bfmmla` hi/lo
  moved it by 1.2%.
- **Paged** uses `bfdot` Q@K^T and the `[page/2][hd][2]` V layout. bf16
  probabilities are rounded like vLLM's `prob_buffer_t` (<= 1e-3 against the
  fp32 reference).

## 6. Numerics and tests

| gate | limit | final result (all three boxes) |
|---|---|---|
| dense fp32 vs naive fp32 reference | 2e-4 (AMX hi/lo ~3-8e-5) | `test_dense.py` 89/89 |
| dense bf16 | 3e-2 | `fuzz_dense.py --cases 80 --max-seq 900`: 80/80, worst 0.52x tolerance, plus 24/24 stray-read probes |
| paged fp32 | 1e-6 | `test_paged_sweep.py` 504/504; worst 7.75e-7 to 8.34e-7 |
| paged bf16 (x86 vs the other backend) | 0-2 ulp | worst 1.92-1.96 ulp |
| paged bf16 (ARM) | 1e-3 | within |
| engine (`fast-encdec/tests/test_cpu_attention.py`) | 45 checks | 45/45 (development laptop) |

The sweep covers head_dim 32/64/128, pages 16/32/48/64, GQA 1/2/4, seven
shapes, both dtypes, strided q/k/v and permuted block tables.

Bugs it found, fixed in both backends:
- GQA stores indexed the new K/V by *query* head.
- The dense plan cache was not keyed on the backend.
- BF16 CPUID was read from subleaf 0.
- Dense plans missed the K/V strides of a contiguous Q.
- The AMX on/off/on toggle reused a freed workspace.

## 7. How to measure (the rules that survived)

1. **One process, interleaved arms, rotated order, the minimum, and a
   duplicate control arm.** `benchmarks/bench.py` does this.
   - Two separate runs on these boxes drift 5-10%, the large rows ~15%.
   - Never compare across processes.
2. **The control must be a separate module when comparing two builds.**
   Reusing the new build's closure as "control" hides module-level effects: two
   byte-identical `.so` files read 0.93 "new/old" at 1024:16:128. Load a copy of
   the old build as the third arm, in random load order. Its floor was then
   0.977-1.005.
3. **Allocation order can move a DRAM-bound row by 3-12%.** Paged b1/4096 and
   b32/1024 depend on which cache tensor was allocated first. Rotate it, or use
   several processes.
4. **Pin the governor to `performance`** on a laptop or desktop for a run and restore it after;
   use 40-call blocks for short paged calls (`--block-calls 40`).
   - The first row after a heavy test suite can be 1.7x off: one such row read
     1.33, and three isolated re-runs read 0.87.
5. **The right baseline per row.**
   - vLLM pinned at `db3b0f0`, built by `scripts/build_vllm_ref.sh`.
   - x86 bf16 on an AMX box is compared to vLLM's AMX build; every other x86
     row to VEC; aarch64 to NEON. Never C++ VEC against vLLM AMX.
   - vLLM's cache is staged and its output preallocated outside the timer.
     Ours includes the pack and the output allocation.
6. **Profile before changing a loop.**
   - In-situ phase ablation: `CPU_ATTENTION_VEC_MODE`, `_NEON_MODE`,
     `_AMX_MODE`, `_PAGED_MODE`, where each bit skips one phase.
   - Then `perf stat` / `perf record`.
   - A change that does not beat the old code by more than the control spread
     (3-5%) is reverted and recorded.

## 8. Falsification ledger

Every row was measured in-process and the change was reverted.

| idea | result | why |
|---|---|---|
| vec: two 16-key groups per Q@K^T step | 1.10-1.15x slower | Zen 4 memory-broadcast FMA is free of the load port |
| vec: 10 fp32 register rows (with partial tiles) | 1.03-1.21x slower at long D=128 | |
| vec: software-pipelined K load in `qk` | 1.48x slower | pointer walk + live register |
| vec: unblocked `[hd][seq_k]` K^T | 1.17x slower | TLB miss per dimension |
| vec: in-place V ungated / gated at full LLC | 1.32x / 1.17x slower on long rows | strided DRAM stream |
| vec: pack/attend overlap (per-head ready flags, no barrier) | 0.99-1.03 | the pack is a small share once streamed |
| vec: fused order at 2048:16:128 | 1.0-1.2% slower | slabs exceed the LLC share |
| vec: `rq` 32->64, `kt` 128/512, `prefetchnta` pack source | 0.98-1.01 | not the lever |
| paged: two OpenMP regions for store + attend | 2.4 vs 1.1 us fixed cost | a second fork-join |
| paged: 1/2/4 workers at b=1 | 19.8 / 12.9 / 8.8 us vs 7.3 at 8 | the pool wins above 32 KV positions |
| paged: software prefetch (vec, neon; 4 variants) | neutral to 1.10x slower | the cache streams are prefetched already |
| paged: `schedule(static, 1)` / head-chunking (neon) | big fp32 rows 0.85, bf16 b8/256 1.85x slower; footprint-gated version indistinguishable from its control | not adopted |
| amx: per-row `sfence` after NT stores | 1.25-2.4x slower | the closing barrier suffices |
| amx: prefetch output rows / `kt` 128 / pack 32-wide converts | mixed, 0.95-1.10 | |
| amx: fp32 via `dpbf16ps` hi/lo on vec | rejected on arithmetic | 3 products at 2x density |
| neon: fused register panel (logits never leave registers) | 1.52x slower | the output accumulator does not fit alongside |
| neon: Cauchy-Schwarz fixed max (no max pass, `qk` writes P) | 3-8% slower once correct | exposes the `exp2` chain in a register-tight loop |
| neon: fp32 through `bfmmla` hi/lo | 1.2% | scalar-pipe roof |
| neon: FA4 -8 rescale slack, pack prefetch, `rq` 64-512 | 0.99-1.02 | |

## 9. Open problems and what is worth trying

1. **Ryzen laptop (historical campaign) paged self b=32 ctx=1024 bf16 (1.16 of the then-Mojo).**
   - Store alone costs the same in both backends (54 us), but store followed by
     attend costs ~300 us more than attend alone for C++, against ~100 us for
     Mojo. Cross on the same row is at parity.
   - So the 17 K scattered store lines perturb the following DRAM stream.
   - Try streaming stores for the new token, or attending a head's early pages
     before storing into its last page.
   - Verify across several processes.
2. **Axion dense rows at parity with Mojo** (fp32 D=64; bf16 1024:8:64 and
   1024:16:128).
   - fp32 D=64: both FMA loops sit at exactly 16 accumulators; try an 8-row x
     12-key tile.
   - The bf16 softmax is ~27% of the call, and GCC adds `mov`s for the
     polynomial's constant accumulators.
3. **The dense entry in the `paged_call` style.** It should read tensors in C++
   and allocate the output there, worth 1-2 us on the 20-60 us short rows.
4. **fp32 D=128 K pack when DRAM-bound** (Ryzen laptop 2048:16:128 at 0.98, historical). Options:
   - a lazy per-tile pack done by the first query block that needs it, so the
     slab stays in L2;
   - 2 MB-aligned workspaces with `MADV_HUGEPAGE`.
5. **AMX.**
   - Cache the fp32 Q hi/lo panel per work item (114 us of 1352 at 1024:8:64).
   - Skip the max pass after a row's first tile on long rows.

## 10. Knobs

These are for A/B testing only; the defaults are what was measured.

| env | effect |
|---|---|
| `CPU_ATTENTION_BACKEND` | `mojo` (default), `cpp`, `auto` (cpp on x86-64) |
| `CPU_ATTENTION_ISA` | force `scalar` / `vec` / `amx` / `neon` |
| `CPU_ATTENTION_WORKERS` | worker count, capped at physical cores |
| `CPU_ATTENTION_KT`, `_RQ` | dense key tile (256) and rows per work item |
| `CPU_ATTENTION_FUSE` | force the fused (1) or split (0) dense order |
| `CPU_ATTENTION_PAGED_KT`, `_PAGED_QK`, `_PAGED_PV`, `_PAGED_PF` | paged tile, loop shapes, prefetch distance |
| `CPU_ATTENTION_AMX_KT` | AMX key super-tile |
| `CPU_ATTENTION_{VEC,NEON,AMX,PAGED}_MODE` | phase ablation bits (results are garbage) |
