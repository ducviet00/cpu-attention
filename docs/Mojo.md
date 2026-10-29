# The Mojo backend

`cpu_attention/_mojo/` is the default backend: one unified CPU attention for
both public APIs, `flash_attn_func` (FlashAttention over supplied Q/K/V) and
`flash_attn_with_kvcache` (one-token paged decode, PagedAttention), built with
Mojo 1.2.0 (pinned in `pyproject.toml`). The C++ backend (`docs/CPP.md`) is the
performance competitor; the previous Mojo build is kept as the oracle at tag
`mojo-oracle-v1.1`.

```bash
uv sync --extra build                                   # torch + mojo==1.2.0
uv run --extra build python scripts/build.py --tests    # every tier + the standalone kernel test
```

## 1. Status

**Integration checkpoint (2026-10-28).** The primary tree at `c7f1c28` already
contains the AMX policy merge (`6d2cf34`) and the fast-plan AMX-toggle fix. The
in-progress Axion merge (`afbea06`) has been resolved retaining both policy IDs
(AMX 1, ARM 2); only AMX uses the AMX-specific 64-row work-item rule. All three
x86 tiers build with Mojo 1.2.0. On Ryzen, both backends pass `test_dense`,
`test_flash_api`, 80-case fuzz, the 504-case paged sweep, 412 regressions, and
fast-encdec's engine test. AMX tests skip on this non-AMX CPU. The engine test uses fast-encdec's own
environment (torch 2.14.1+cpu, transformers 5.19.0). This is a correctness
checkpoint, not a claim that the performance objective is met.

The integration baseline (raw JSON kept outside the repo),
41 rounds / 41 x 40-call paged blocks, measured all 170 rows with valid outputs:

| rows | win | unresolved | loss |
|---|---:|---:|---:|
| flash_attn_func bf16 (39) | 36 | 3 | 0 |
| flash_attn_func fp32 (39) | 13 | 25 | 1 |
| decode / store / public API (92) | 39 | 37 | 16 |
| **all (170)** | **88** | **65** | **17** |

These are observations from that run, not an across-process speedup claim over
the older checkpoint below. A fresh C++ build reproduced
the baseline library's SHA256 exactly. The baseline's in-process Mojo-version
subprocess failed after loading the extensions; `compiler.txt` records the
compiler separately. The harness now probes it before loading any extension.

Ryzen 7 7840H, full 170-row manifest (`benchmarks/manifest.py`), against the
fastest numerically valid C++ path per row (`cpp_vec` or `cpp_scalar`), one
process, independently loaded arms, 41 rounds (paged: 41 x 40-call blocks),
governor pinned:

| rows | win | unresolved | loss |
|---|---:|---:|---:|
| flash_attn_func bf16 (39) | 38 | 1 | 0 |
| flash_attn_func fp32 (39) | 12 | 27 | 0 |
| decode / store / public API (92) | 33 | 47 | 12 |
| **all (170)** | **83** | **75** | **12** |

A win is below `1 - max(0.03, control spread)`, a loss above `1 + band`;
anything else is unresolved, not a win. The starting point (the oracle build,
 was 7 / 21 / 142.

**Where it does not win on the Ryzen.**
- The 12 losses (1.03-1.12) are decode rows that stream 16-32 MB of cache per
  call (b=8 ctx~1024, b=1 ctx~4096, page 64): both backends run at the box's
  DRAM bandwidth there (b=8 ctx=1024 bf16 is ~360 us for 32 MB). They are at a
  shared roof rather than behind a missing technique; neither a page-64
  specialization nor extra dot chains moved them (§9).
- fp32 flash_attn_func is FMA-roof bound in both backends (~80% of one zmm FMA
  per cycle, single-thread parity at every shape measured); 27 of 39 rows are
  unresolved.

**Axion.** The ARM hi/lo-probability `bfdot` P@V policy (`8b6521a`) improved all
44 bf16 decode/API rows by 8-41% beyond their saved-build controls, but the full
92-row paged/store run still had 12 wins / 27 unresolved / 53 losses against
valid C++.  A globally larger work item regressed short calls; a guarded long-square
causal fp32 D=128 rule is under integration validation. Default C++ NEON still
fails poisoned-tail regressions; the scalar path passes. Invalid C++ arms are
excluded per row, not silently accepted as references.

**Xeon.** The AMX policy is integrated, but full primary-tree validation and
the missing paged/store measurements remain pending. No cross-machine
dominance claim is made.

## 2. Layout

```
_mojo/src/isa.mojo        tier flags, LANES, ptr, one exp2[degree], dot_bf16, AMX permission
_mojo/src/parallel.mojo   one GOMP_parallel region on torch's OpenMP pool, barrier
_mojo/src/kvcache.mojo    flash_attn_with_kvcache decode + reshape_and_cache (store)
_mojo/src/flash_attn.mojo flash_attn_func: FlashJob, FlashPolicy trait, the generic driver
_mojo/src/flash_vec.mojo  the vector policy (AVX-512 / AVX2 / NEON-fp32, LANES-generic)
_mojo/src/dispatch.mojo   runtime dtype x head_dim x ISA-policy selection
_mojo/src/pyview.mojo     reads torch tensors natively (the fast call paths, §6)
_mojo/src/amx.mojo        AMX tile primitives (inline asm, x86 only)
_mojo/src/_C.mojo         the Python extension
_mojo/tests/test_attention.mojo   standalone decode parity test vs a scalar fp32 reference
```

## 3. One design for both APIs

**The page layout is shared.** flash_attn_func packs each (batch, kv head) into a
private slab whose K^T is blocked every 64 keys -- exactly a KV-cache page of 64
tokens: `[tile][hd][64]` fp32, `[tile][hd/2][64][2]` bf16 (the dim pair a dot
instruction contracts). The decode cache uses the same dim-pair K layout per
page. So Q@K^T, the online softmax and P@V are the same register-tile
algorithm over pages of keys:
- decode: rows = the query heads of one kv group;
- flash_attn_func: rows = consecutive query positions.

**flash_attn_func driver / policy split.** `flash_attn.mojo` owns scheduling,
the bottom-right causal bound (row i sees keys `j <= i + seq_k - seq_q`, exact
per register tile, fully masked tiles skipped), the online-softmax state, the
accumulator and the epilogue (a row with no visible key is exactly 0). An ISA
supplies a `FlashPolicy` struct (pack, slab/tile sizes, one register tile of
rows). `dispatch.mojo` picks the policy at runtime; the abstraction is zero-cost
(1024:8:64 bf16 single-thread 4.61 -> 4.56 ms across the refactor). Scheduling:
*fused* (a thread packs a head, then attends all of its query blocks, no
barrier) while the workers' slabs fit 0.6 of the LLC, else *split* (pack all,
barrier, dynamic boustrophedon over query blocks through an atomic counter).

**Vector policy.** 8-row x 16-key Q@K^T tile with the query broadcast folded into
the instruction (`vdpbf16ps ... {1to16}` / `vfmadd ... {1to16}`), the tile
maximum tracked in registers during Q@K^T (no max pass), degree-2 `exp2` for bf16
probabilities (they are rounded to bf16 right after), degree 5 for fp32; bf16
P@V by `dpbf16ps` on key pairs (V packed `[key/2][hd][2]`); fp32 V read in place
while all of V fits LLC/2. Partial tiles instantiate 2- and 4-row variants.

**Decode.** Work item = (sequence, kv head, chunk of that head's query heads):
every query head of a GQA/MQA group is a row of one register tile, so the group
shares one scan of the cache (0.52-0.73 of C++ on the GQA/MQA rows, where C++
rescans per query head). Groups are split into chunks only when there are fewer
(sequence, kv head) pairs than workers; stores then precede a barrier.
x86 keeps C++'s numerics class (dpbf16ps Q@K^T, fp32 P@V, degree-2 exp2 for
bf16, degree-6 fp32) so the 2-ulp cross-backend bf16 gate is meaningful; bf16 V
is widened even/odd split (`w << 16`, `w & 0xffff0000`: 2 instructions per 32
values instead of 4), bit-identical. The page size is a compile-time constant
for 16 and 32 (the engine defaults). P@V never reads past `lens`, a negative slot
skips its store, a zero-length context returns 0.

## 4. Tiers and runtime ISA selection

`scripts/build.py` builds one shared object per tier for an explicit target --
never the build host -- and `cpu_attention._kernel` loads the best tier the CPU
reports in `/proc/cpuinfo` (a listed AVX-512 flag is also an OS-state check):

| tier | target | selected when |
|---|---|---|
| `x86-avx512bf16` | `x86-64-v4 +avx512bf16,+avx512vnni` (+ AMX kernels) | AVX-512 F/BW/VL/DQ/CD + BF16 + VNNI |
| `x86-avx2` | `x86-64-v3` | AVX2, FMA, BMI1/2, F16C, MOVBE |
| `x86-64` | `x86-64` | always |
| `aarch64-bf16` | `neoverse-n1 +bf16,+i8mm` | asimd + bf16 + i8mm |
| `aarch64-neon` | `generic` | always |

`CPU_ATTENTION_ISA=amx|vec|avx2|scalar|neon` forces a tier (and turns AMX off
except for `amx`); a tier the CPU lacks is refused with an error. AMX code sits in
the AVX-512-BF16 tier (inline asm assembles for any x86 target) and runs only
when the CPU reports `amx_tile`/`amx_bf16` and `arch_prctl` grants tile state.
`_C.so` is a copy of the host's best tier for older callers. Every tier the CPU
supports passes the gates (Ryzen: avx512bf16, avx2, x86-64; Xeon: vec, avx2,
scalar). bf16 K is dim pairs on every tier, so a cache written under one tier
is read correctly under another.

## 5. Numerics and gates

| gate | limit | Ryzen result |
|---|---|---|
| flash_attn_func fp32 vs naive fp32 reference | 2e-4 | test_dense 82/82, fuzz 80/80 (worst 0.52x tol) |
| flash_attn_func bf16 | 3e-2 | same |
| paged fp32 | 1e-6 (fp64 arbitrates an out-of-gate fp32 reference) | sweep 504/504, worst 8.9e-7 |
| paged bf16 x86 | <= 2 bf16 ulp at the row peak vs C++ | worst 1.89 |
| regressions (`tests/test_regressions.py`) | as above | 412/412 |
| fast-encdec `tests/test_cpu_attention.py` (strided fused q/k/v) | 45 checks | 45/45 |

**ARM bf16 "<= 1e-3 against the reference".** Taken literally (absolute error to
the real-valued result) no bf16 output can meet it: at short contexts both
backends sit 3-4e-3 from the fp32 reference while agreeing to 0.45-1.5 ulp, and
the earlier campaign measured 77/252 Axion sweep cases where rounding the exact
result to bf16 alone exceeds 1e-3 (worst 6.6e-3). The tests therefore check the
error *above the bf16 output-rounding floor*, `max(|out - ref| - |ref -
bf16(ref)|) <= 1e-3` (`tests/reference.bf16_excess`), on top of the 2-ulp
cross-backend gate. This is a stated interpretation of an unattainable literal
gate, not a passed literal gate. The scale used by the changed-scale regression
stays where the 1e-6 fp32 gate is defined: at scale 0.9 the fp32 *reference* is
6.4e-6 from fp64.

## 6. Call path and threading

- **Torch's OpenMP pool.** One `GOMP_parallel` region per call (~2.4 us warm);
  MAX's `parallelize` and private pools were slower (`docs/CPP.md` §8). Slices
  use `omp_get_num_threads()`; workers are capped at physical cores.
- **Native tensor reads (`pyview.mojo`).** `data_ptr()`/`.shape` cost ~50 ns each
  from Python, and a decode call needs a dozen. At import, `_kernel._calibrate_layout`
  finds the TensorImpl/StorageImpl field offsets by matching what torch reports
  (`tensor._cdata` is the TensorImpl, `untyped_storage()._cdata` the StorageImpl),
  verifies them on probe tensors (views with offsets, transposes, 1-4 dims, five
  dtypes, `nn.Parameter`) through ctypes and again through the module's reader,
  and disables the path on any disagreement (`CPU_ATTENTION_KV_FAST=0` turns it
  off). `kv_fast` / `flash_fast` accept only exact `torch.Tensor`/`Parameter`
  objects whose dtype, dims, sizes, strides, causal flag, scale and thread count
  match a per-thread plan registered after a fully validated call; anything else
  returns to the checked Python path. A grown workspace, `set_num_workers` and
  `_drop_workspace` invalidate every table.
  - b=1 ctx=256 decode with `out=`: 4.4 -> 2.75 us (C++ 3.06).
  - flash_attn_func 17:8:64 bf16: 5.26 -> 3.05 us (C++ 5.28).
- **Workspace** is per thread, grown on demand, never shrunk (a freed large
  tensor is an mmap that faults in again).

## 7. Measuring

`benchmarks/bench.py` is the harness (docs/CPP.md §7 rules): every arm is an
independently loaded copy of the package (`benchmarks/arms.py`) -- `mojo_new`,
`mojo_ctrl` (a second copy: the noise floor), `mojo_old` (the oracle tag), one
`cpp_<isa>` per applicable C++ path -- plus pinned vLLM and SDPA, interleaved
with rotated order, minimum as the statistic, raw samples kept, every arm's
output validated, the environment recorded in the JSON. Each arm stages its own
cache through its own `reshape_and_cache`; repeated appends rewrite the same
slot. `scripts/pin_governor.sh` pins `performance` and restores each CPU's
original setting. Paged rows on the Ryzen use 40-call blocks, which keep a
16 MB working set warm; that is the documented protocol, not a cold-cache
number.

For an optimization A/B, save the previous package's Python files and shared
libraries, then pass `--previous /path/to/saved/package`. This loads the previous
build twice independently (`mojo_previous`, `mojo_ctrl`); their spread defines
the control uncertainty, not the candidate's improvement. Reports fingerprint
the Python files, shared libraries, kernel sources, manifest, lockfile and build
scripts. Screening uses only numerically valid C++ arms to establish its cutoff.
Store-only rows start with poisoned caches, and self-attention rows stage a
poisoned append slot, so missing writes cannot pass merely because setup already
wrote the expected data.
Store validation reads the logical K/V elements directly and requires exact
copies; it does not reject a valid store because that arm's decoder reads
unused poisoned padding incorrectly.
Use `--load-seed` to repeat with different reproducible module-load and cache
allocation orders. A bf16 C++ peer must itself pass the reference check before
it can arbitrate the cross-backend gate; its name is recorded per row.

## 8. Defects found and fixed (each has a regression test)

1. **Old Mojo: negative slot** stored at a negative offset (segfault). C++ skips
   the store; so does the new kernel.
2. **Old Mojo: zero-length context** returned NaN; now exactly 0.
3. **AMX plan key** left out `softmax_scale` (and the worker override): a second
   call with a different scale reused the first one (Xeon, errors 2-4).
4. **aarch64 misaligned atomic**: the split order's Int64 work counter sat after
   an odd-sized scratch block; ARM atomics fault (SIGBUS) where x86 does not.
   Scratch is 64-byte aligned.
5. **Mojo 1.2.0 assembles inline asm in dead instantiations**: the AMX asm
   reached the ARM assembler. The primitives are `comptime if is_x86()`.
6. **LLVM ISel crash** (`Do not know how to split this operator's operand`) in
   the old bf16 dense kernel for non-AVX-512-BF16 targets; the old kernel is gone.
7. **Lower tiers read bf16 K wrongly** when the dot path was gated on the
   instruction rather than the layout; bf16 always goes through `dot_bf16`
   (one instruction, or widened).
8. **C++ (not fixed, reported):** on Axion the C++ NEON bf16 decode returns NaN
   when unused cache slots/padding hold NaN, and a non-finite row for a
   zero-length context (`test_regressions.py` on the cpp backend).

## 9. Falsification ledger (this campaign)

Measured in one process against an independent control; reverted unless it beat
the control by more than 3-5%.

| idea | result | decision |
|---|---|---|
| page size as a compile-time constant (16/32) | decode bf16 1.015 -> 0.956 us/head, fp32 2.79 -> 2.46 | kept |
| even/odd split bf16 V widen | 0.956 -> 0.86 us/head, bit-identical | kept |
| runtime head-dim loop in Q@K^T (no hoisted, spilled broadcasts) | decode bf16 0.86 -> 0.72 us/head (C++ 0.83) | kept |
| query/probability as the last `vdpbf16ps` source (broadcast folds) | dense bf16 5.72 -> 5.06 ms | kept |
| degree-2 exp2 for bf16 dense probabilities | 5.08 -> 4.92 ms | kept |
| tile maximum tracked in Q@K^T (no max pass) | 4.92 -> 4.61 ms | kept |
| native tensor reads (decode / flash) | 4.4 -> 2.75 us; 5.26 -> 3.05 us | kept |
| MHA decode fast path (`R=1` separated from GQA `_rows_tile`, incremental page walk, `PAGE=16/32/48/64` compile-time constants, store-only + pre-append `slots=None` in `kv_fast`) | Ryzen paged bf16 0.60-0.96 of C++ (`b=1 ctx=256` 3.61 -> 3.50 us, `b=1 ctx=4096` 0.53-0.67 of `mojo_old`); ARM `bfdot` P@V single-bf16 for `len >= 128` + hi/lo residual for `len < 128` | kept |
| single-lane (`LN=1`) fp32 decode P@V on x86 | `b=8 ctx=256 fp32` 39.7 -> 43.6 us (+9.5% slower; one FMA chain per dim vector is latency-bound at `R=1`) | reverted |
| interleaved (`i = tid; i += threads`) MHA decode scheduling on x86 | `b=8 ctx=256 bf16` 19.2 -> 21.5 us (+12% slower due to cross-core page-table/cache contention vs static contiguous ranges) | reverted |
| x86 decode P@V via `dpbf16ps` key pairs | rounding P to bf16 alone adds up to 1.81 peak ulp; with the existing 1.89 it breaks the 2-ulp gate | rejected on arithmetic |
| two dot chains per key group in decode Q@K^T | 0.86 -> 0.86 us/head (L3-bound) | reverted |
| one dim vector per P@V pass (fold the probability broadcast) | fp32 hd=128 1.28x slower (V re-read), others neutral | reverted |
| page 64 as a compile-time constant | 0.972 bf16 / 0.999 fp32 vs the previous build | reverted |
| fp32 scratch without the bf16 probability buffer | no change (104.6 vs 104.6 us) | kept for sizing only |
| fp32 flash multi-thread gap on short shapes (before the fast path) | single-thread parity at every shape; 3-8% multi-thread, more L1/LLC misses, cause not found | open |

**Axion decode ledger (c4a-standard-8, one process, independent control, vs `2fc4652`).**

| idea | result | decision |
|---|---|---|
| NEON decode key tile `KT` 256 -> 64 (hd 64), 128 -> 64 (hd 128) | b=8 ctx=1024 bf16 0.986; d=128 cross 0.992 (within control) | reverted |
| software prefetch of the next-but-one K and V page at each page crossing | b=8 ctx=1024 1.02, b=1 ctx=4096 bf16 1.06 / fp32 1.08 (slower) | reverted |
| bf16 Q@K^T key groups per step G 8 -> 16 (ARM, R=1) | b=32 ctx=1024 1.001, b=8 ctx=256 1.047 | reverted |
| recycle allocated decode outputs (ring checked by raw refcounts) | `engine` rows b=1 ctx=256 bf16 8.0 -> 6.5 us (C++ 8.1) | rejected: reads c10 refcounts from raw memory, unsafe |

Findings. Plain `engine` rows lose to `engine_out` rows by the cost of `torch.empty_like`
(0.7 us hot, 2-3 us in the interleaved harness; C++ allocates inside its extension, about
1.4 us). Large-batch bf16 decode (b=8..32, ctx~1024, d=128, page 64, MQA h=16/1) streams
16-67 MB per call at 60-150 GB/s and sits at 1.04-1.17 of C++ NEON; no loop-shape change
tried moved it. Open: why C++'s 64-key K/V interleave streams better than the
Mojo 256-key tile at b=32, and why MQA h=16/1 bf16 (92 us vs C++ 83, vLLM 39) is not
compute-bound-efficient. Requests: none for the shared driver.

Earlier campaigns' ledgers (`docs/CPP.md` §8 and the oracle's notes at
`mojo-oracle-v1.1`) still apply; nothing on them was re-run without a new
mechanism.

## 10. Mojo 1.2.0 notes (the ones that cost time)

- `--fp-mode contract=fast` is the default: `a + b*c` contracts into FMA across
  statements.
- `Int` has no `bit_length`; `Atomic[Int64].fetch_add(ptr, 1)`; there is no
  `memset_zero`; positional pointer indexing is deprecated (`unsafe_load`).
- A fully `comptime for`-unrolled head-dim loop lets LLVM hoist every query
  broadcast out of the key loop and spill it: keep a runtime loop with a small
  unroll.
- `llvm_intrinsic[..., has_side_effect=False]` for dot instructions, or the calls
  stay ordered (~4x slower).
- `out` is a keyword (not an argument name); `PythonObject._obj_ptr` is the raw
  object; stack arrays of megabytes overflow the 8 MB stack.
