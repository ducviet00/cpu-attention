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

Current per-box results, the verdict rule and the caveats are in the
[README](../README.md#performance), which is the single source of truth for
numbers; this file does not repeat them. The objective (a clear per-row win over
the fastest valid C++ path on all three machines) is **not met**. What limits it,
by row family:

- **Decode that streams more cache than fits in L2** (batch 8-32, context ~1024
  and up): both backends run at the same memory roof, so a few percent either way
  is placement and noise, and the same row moves 20-30% with allocation order.
  Neither a compile-time page 64, extra dot chains, key-tile changes, software
  prefetch nor wider key groups moved these rows (§9).
- **Decode that allocates its output**, short calls: the extension now
  allocates the output natively, as C++ does (`aoti_torch_empty_strided` +
  `THPVariable_Wrap`, §9); `torch.empty_like` cost several microseconds inside an
  interleaved run.
- **fp32 `flash_attn_func`** is FMA-bound in both backends (parity single-thread
  at every shape measured).
- **Xeon AMX, head_dim 128, 256-512 tokens:** the K/V pack is slower than C++'s
  (§9); the rest of the gap there is not explained.

## 2. Layout

```
_mojo/src/isa.mojo        tier flags, LANES, ptr, one exp2[degree], dot_bf16, AMX permission
_mojo/src/parallel.mojo   one GOMP_parallel region on torch's OpenMP pool, barrier
_mojo/src/paged_attn.mojo flash_attn_with_kvcache decode (one driver, `Decode` + `PVKernel`) + reshape_and_cache
_mojo/src/flash_attn.mojo flash_attn_func: FlashJob, FlashPolicy trait, the generic driver
_mojo/src/flash_vec.mojo  the vector policy (AVX-512 / AVX2 / NEON fp32, LANES-generic; target constants)
_mojo/src/flash_amx.mojo  the AMX policy (bf16, fp32 as bf16 hi/lo), x86 only
_mojo/src/flash_neon.mojo the aarch64 bfmmla policy (bf16; fp32 is the vector policy)
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
probabilities (they are rounded to bf16 right after), degree 4 (minimax) for fp32;
bf16 P@V by `dpbf16ps` on key pairs (V packed `[key/2][hd][2]`); fp32 V read in
place while all of V fits LLC/2. Partial tiles instantiate 2- and 4-row variants.
The same source is the fp32 policy on aarch64: per target only compile-time
constants differ (`KV32` key vectors per Q@K^T pass, `NS` partial sums, and where
a broadcast does not fold into the FMA, `isa.BCAST_FOLD` false, query dims and
probabilities as vectors used by lane, 4 keys per P@V step, and a contiguous Q
panel). The bf16 matrix engines (AMX, `bfmmla`) keep their own policies.

**Decode.** Work item = (sequence, kv head, chunk of that head's query heads):
every query head of a GQA/MQA group is a row of one register tile, so the group
shares one scan of the cache (0.52-0.73 of C++ on the GQA/MQA rows, where C++
rescans per query head). Groups are split into chunks only when there are fewer
(sequence, kv head) pairs than workers; stores then precede a barrier. P@V never
reads past `lens`, a negative slot skips its store, a zero-length context returns
0. The page size is a compile-time constant for 16/32/48/64.

**One decode driver for every tier** (`paged_attn._attend_rows`). A softmax step of
`KT` keys is resolved into 16-key units (page % 16 == 0, so a unit never
straddles a page): one key vector on AVX-512, two on AVX2, four on NEON. Q@K^T
scores `G` key vectors per call (independent dot chains) with a masked tail that
stays inside one unit, tracks the step maximum as a tree, and the softmax keeps
`NS` partial sums. What differs per target is compile-time only:
- `Decode[dtype, hd, R]` holds the shape as `comptime` members computed from
  `simd_width_of` and `CompilationTarget` (`KT`, `G`, `G_TAIL`, `DEG`, `NS`, `NC`,
  `FULL`); `comptime assert` pins the tiling invariants.
- `Decode.PV` is the one per-target kernel, a type conforming to the `PVKernel`
  trait (as `InnerMatmulKernel` in Modular's unified matmul): `PVRows` keeps
  fp32 P against V rows, the x86 choice (C++'s numerics class, so the 2-ulp
  cross-backend bf16 gate is meaningful; bf16 V is widened even/odd split,
  `w << 16` / `w & 0xffff0000`, 2 instructions per 32 values); `PVPairs` keeps
  bf16 P against bf16 V key pairs with `dot_bf16` (`bfdot`, plus P's low residual
  below 128 keys), the aarch64 choice (widening V was 70% of the kernel there).
  The V layout follows (`PageLayout.V_PAIRS`).

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
supports passes the gates (C4D: avx512bf16, avx2, x86-64; Xeon: vec, avx2,
scalar). bf16 K is dim pairs on every tier, so a cache written under one tier
is read correctly under another.

## 5. Numerics and gates

| gate | limit | C4D result |
|---|---|---|
| flash_attn_func fp32 vs naive fp32 reference | 2e-4 | test_dense 82/82, fuzz 80/80 (worst 0.52x tol) |
| flash_attn_func bf16 | 3e-2 | same |
| paged fp32 | 1e-6 (fp64 arbitrates an arm over the gate) | sweep 504/504, worst 8.9e-7 |
| paged bf16 x86 | <= 2 bf16 ulp at the row peak vs C++ | worst 1.89 |
| regressions (`tests/test_regressions.py`) | as above | 412/412 |
| fast-encdec `tests/test_cpu_attention.py` (strided fused q/k/v) | 45 checks | 45/45 (development laptop; fast-encdec is not on the VMs) |

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
original setting. Paged rows on a laptop or desktop can use 40-call blocks (`--block-calls 40`), which keep a
single power state; the reported runs are on GCP VMs and use one call per sample.

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

9. **Harness (fixed, not a kernel defect):** the paged fp32 gate measured arms only against the
   fp32 reference and consulted fp64 only when the reference was itself over 1e-6 from it. On
   heterogeneous contexts the reference is 6-9e-7 from fp64 while Mojo and C++ vec are 2-5e-7 from
   it, yet 1.0e-6 / 1.07e-6 from the reference, so valid arms (both backends) read INVALID on some
   data draws. Any arm over 1e-6 is now judged against fp64.

## 9. Falsification ledger (this campaign)

Measured in one process against an independent control; reverted unless it beat
the control by more than 3-5%. Per-head and per-call timings in this table were
taken on the development laptop (AMD Ryzen 7 7840H) or on the Xeon/Axion VMs as
noted; the results that count are the three GCP VMs in the README.

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
| per-call choice of the vector or the AMX flash policy (Xeon, 4 threads) | fp32 one query 19.3 -> 7.1 us (8:64) and 74 -> 26 us (16:128); 129.5 -> 53.4 us at 1025 keys; bf16 one query 10.2 -> 8.3 us; AMX stays 2.7-8.1x ahead (bf16) elsewhere. Rule: fp32 up to 65 rows at hd 64, 24 at hd 128, or <= 64 keys; bf16 1-2 rows | kept |
| fp32 AMX with a large softmax scale (`scale*sqrt(hd) > 6`) | hi/lo products reach 2.03e-4 vs the 2e-4 gate at scale 1.7; the exact-fp32 vector policy has no such limit | kept (routes to vector) |
| AMX K/V pack vs C++ (Xeon, bf16 hd 128, S=129, one thread) | pack 9.6% of samples vs 7.4% (attention loops equal, 35.7% vs 36.4%); the K block reloads 16 row pointers from the stack, 27% of the function's samples are stack moves, shuffles 40%, `vpermw` 20% | open |
| cheaper tensor constructor than `empty_like` for the decode output | cold 2.6 us; `empty(shape, dtype)` 5.7, `new_empty` 4.7, `empty_strided` 3.0 | none; `empty_like` is already the cheapest |
| Xeon fp32 decode b=8 ctx~1024 "losses" | Mojo == C++ single-thread (17.2 vs 17.1 us/head), 4 threads (562 vs 562 us), cold cache (~600 us both) and 7-cache rotation (977 vs 930); `--load-seed` 1-4 gave 0.82-1.04 with control spreads of 17-31% | placement noise, not code |
| harness verdict band includes the spread of C++'s identical-code `amx` and `vec` paged arms | Xeon 67/61/42 -> 64/76/30 (win/unresolved/loss); no other box has such twins | kept |
| `bench.py` memory growth | closures kept every arm's K/V cache alive until a GC pass: 8.1 GB RSS and an OOM kill at row 156 of 170 on an 11.6 GB machine; `gc.collect()` per row and `--resume` | fixed |
| AMX partial query tile: configure only the live A/C rows (Xeon) | the hardware is cheaper with fewer rows (isolated tile-dot microbenchmark: 1-4 rows 3.0 ns, 8 rows 3.2, 16 rows 5.2 ns), but building the configuration from runtime row counts changed the full-row code: fp32 1024:8:32 1.125, 1025:8:32 1.080, fused 1024:8:64 1.104 against an independent previous build; one win (fp32 17x301, 0.94, where the vector policy is faster anyway) | rejected; next try would keep the constant full-tile configuration and patch only the row bytes on a tail |
| x86 decode softmax tile `KT` 512 for MHA (`R=1`) at head_dim <= 64 (C++ vec uses 512), Xeon, vs an independent previous build | b=8 ctx~1024 fp32 rows new/prev 0.92-1.01 (page 16/48/64, d=32/128, `out=`, public API), control spread up to 2.6%; GQA rows (tile unchanged) moved 1.09-1.12 by placement alone | reverted: inside the 3-5% bar |
| Xeon AMX K/V pack: non-inlined `_pack_k_block` / `_pack_v_block` (the inlined pack reloaded 16 row addresses from the stack, 23% of its samples) | spills gone from the K block; 257:16:128 bf16 single thread 826.2 -> 826.8 us, 129:16:128 281.8 -> 279.5 us. Sampled profile: pack 13.6% of the call (C++ 11.6%) accounts for ~19 of the 25 us single-thread gap; the V block's hot spots are source-load latency behind `vinserti64x4 + vpermw`, not shuffle throughput | reverted; open |
| Xeon fp32 decode `engine_out cross b=8 ctx=1024` under `--load-seed` 1-4 | ratio 0.99, 0.82, 1.02, 1.01 with control spreads 12-21% | placement noise, confirms the earlier row |
| Xeon AMX pack: software prefetch of the next key block's rows (K 16 rows, V 32 rows, every line of the head's slice) | 257:16:128 bf16 single thread: L1 prefetch 826 -> 912 us (+10%), L2 prefetch 826 -> 834; 129:16:128 282 -> 285 / 283; 513:8:64 ~709 -> 708 / 714. The 4 KB inter-key stride (heads interleaved) does not make the pack miss-bound | reverted |
| Xeon AMX pack upper bounds (wrong output, timing only): V interleave replaced by a plain join; then also the K 16x16 transpose removed | 257:16:128 bf16 single thread 826 -> 818 us (V shuffles gone), 819 (K transpose gone too); 129:16:128 282 -> 280. Removing every pack shuffle buys < 1%, so the dimension-permuted V layout (unpack lo/hi, un-permute in the driver epilogue, which runs per query row) cannot pay for itself, and the pack's remaining cost is load/store traffic, not shuffles. Slab, workspace and scratch are 64-byte aligned | not implemented; the Xeon head_dim-128 gap (3-6%) stays unexplained |
| Xeon AMX bf16: Q through a contiguous panel when rows are >= 2 KB apart (heads * hd * 2) and seq_k >= 256 (C++ reads Q in place) | 257:16:128 bf16, 4 threads, 6 processes: in place 208-226 us (bimodal, per process), panel 197-200 (C++ 200-205); against an independent previous build in alternating processes: 1024:16:128 0.92, 512:32:128 0.90, 257:16:128 0.97, 1024:16:64 0.99; the panel for every block cost 129:16:128 +6%, 2048:8:64 +2%, 1025:8:32 +2%, hence the threshold. Full manifest 70/80/20 -> 77/72/21, `flash_attn_func` 43/26/9 -> 47/23/8 | kept |
| Xeon fp32 head_dim 128, 4 threads, 257:16:128 | Mojo 1552 us on one thread against C++ 1604, but 414-422 us on four against 399-403 (C++ scales perfectly, Mojo 8% below linear); not bimodal, unchanged by pinning one thread per CPU, `taskset 0-3`, workspace alignment 4 KB, workspace offsets 0-512 B | open |
| x86 decode key tile 256/512 -> 2048 keys (R=1), C4D | single-process cache-rotation A/B: bf16 b=1 ctx=4096 0.96 of the previous build, others within 2%; full-manifest harness with an independent previous-build control: median 0.994 over 92 paged rows, b=1 ctx=4096 bf16 1.06-1.09 (control spread up to 1.07-1.28 on memory-bound rows) | reverted: not reproducible beyond control spread |
| x86 fp32 decode Q@K^T with 4 key groups per step (`G=4`, R=1), C4D | per core 0.90-0.95 of the previous build on one thread (b=1 ctx 4096, b=8 ctx 256-1024), 0.96 on two, 0.98-1.00 on four in a single-process A/B; but the full 170-row harness with an independent previous-build control (4 threads): fp32 paged median 0.995, and 7 fp32 rows 4-7% slower, among them b=8 ctx 255-257, the rows it was meant to help (the control itself moved 1-5% there) | reverted: per-core gain does not carry to the 4-thread manifest condition |
| compile-time `PAGE=48` decode body removed (runtime page), C4D | bf16 b=8 ctx=1008 page 48 1 thread: 0.949 of the specialized build and 0.99 of C++ (the specialized body is 8% slower than C++ per core, while pages 16/32/64 are 2-6% faster than C++); 4 threads 1.010 (DRAM-bound, the manifest condition); fp32 1.01; b=1 ctx=240 bf16 4 threads 1.03 slower | reverted: no gain at the 4-thread manifest condition. Open: why the 48-key body loses 7% per core in bf16 only |

**Unification campaign (branch `mojo-unified-decode`, all three boxes, full manifest
against an independent copy of `5b3d1ba`; paged median new/previous Axion / C4D / Xeon).**
Goal: one implementation across CPUs, then latency (AGENTS.md).

| idea | result | decision |
|---|---|---|
| one decode driver: `Decode[dtype, hd, R]` compile-time shape, 16-key units, masked tail inside a unit | 1.014 / 1.013 / 1.008; gates pass on every tier | kept, cost won back below |
| Q@K^T call max as a tree (a 16-deep `vmax` chain sat on the softmax path), NEON by-element fp32 Q, P@V over contiguous unit runs | 1.010 / 1.015 / 1.011 | kept |
| P@V as a `PVKernel` trait (`PVRows` x86, `PVPairs` aarch64) | 1.009 / 1.011 / 1.001 (codegen-neutral) | kept |
| unit bases resolved per page; Q@K^T unit bases + compile-time offsets | 1.015 / 1.009 / 1.012 | superseded |
| per-page bases instead of per-unit bookkeeping (`perf stat`, Zen 4, one thread: per-unit +5-11% instructions per call against the old x86 page walk, per-page +1-4% with cycles within noise) | 0.999 / 1.001 / 1.002 | kept |
| NEON by-lane probabilities in `PVRows`, two-vector bf16 P stores | ARM decode bf16 1.017, fp32 0.985 | kept |
| `NeonF32Policy` folded into `VecPolicy` (`KV32`, `NS`, `KU32`, `BCAST_FOLD`) | Axion fp32 flash 1.082 of the dedicated policy, vs C++ 15 / 10 / 14 | kept with the fixes below |
| `isa.exp2` on NEON: exponent from the magic-added value's bits (`fcvt*` issues on one pipe at 1/cycle) | fp32 flash 1.065, C++ losses 14 -> 3; ARM fp32 decode faster | kept |
| `exp2` degree 4 as a Remez minimax (2.6e-6, degree 5's class) for fp32 flash | 1.049, losses 2 | kept |
| NEON fp32 Q@K^T: K for 4 dims first, one query vector per row; P@V chunk loop at runtime on NEON | 1.051 / 1.049 (no change) | kept (the old policy's shape, smaller code) |
| NEON fp32 Q through a contiguous panel (32 heads x 128 dims: rows 16 KB apart, one L1 set) | GQA rows 1.09 -> ~1.0; C++ losses 2 -> 0 | kept |
| lane max through `isa.vmax` (one `fmaxnm` / `vmaxps` against compare + select) | round 6 (all above): Axion fp32 flash 1.050 (19 / 20 / 0 vs C++), flash elsewhere 0.99-1.00 | kept |
| `kvcache.mojo` -> `paged_attn.mojo`; one `isa.exp2` (AVX-512: the AMX policy's `rndscale` + `scalef`, 7 uops against 9; NEON: integer exponent; one coefficient set per degree), `isa.vmax` and `isa.bf16_lo` replace the AMX and NEON policies' copies; one `_with_policy` table for running and sizing | flash Axion / C4D / Xeon 0.998 / 0.985 (C4D bf16 0.973) / 0.989, paged 0.995 / 0.999 / 1.000; `test_amx` 102/102 | kept |

Result: decode unification at no measured cost; Axion fp32 `flash_attn_func` is ~5% slower
than the dedicated NEON policy (wins became ties, no losses), accepted for the unification
(2026-10-29) and open: the separate accumulator-rescale pass (the old policy fused alpha into
P@V's accumulator load) and the first-tile write (`ZEROES_ACC`) are the remaining differences.

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

**Axion campaign 2 (branch `axion-mojo`, vs `/tmp` copy of the previous build as control).**

| idea | result | decision |
|---|---|---|
| allocate decode/flash outputs inside the extension (`aoti_torch_empty_strided` + `THPVariable_Wrap`, links libtorch_python) | removes the `empty_like` cost: engine rows 49 win / 35 unresolved / 8 loss on the 92 paged rows (was 23 losses) | kept |
| ARM 16-key page unit (PT=16, PG=4, KT=64) paged decode, bfdot/ldp-shaped P@V, vectorized K/V store | b=8 ctx=1024 bf16 0.80 of the previous build | kept |
| Q@K^T tail skip when a page group has <= 4 live keys | ctx 2^k+1 rows (257/1025/4097) turn into wins | kept |
| `@no_inline` PV helpers taking `vbase` by pointer | `d=32` segfault: the pointer array's stores were dead-eliminated in the caller (tail call + stack slot), so the callee read garbage | rejected; helpers are `@always_inline` |
| PV helpers `@no_inline` with `vbase: Array[Int, 4]` by value | 40/92 rows slower than the inline version (register pressure across Q@K^T/softmax) | rejected |

Open: batch 8-32 context ~1024 and batch-1 context 4096 decode sit at 0.95-1.03 of C++ NEON
(bandwidth-bound); fp32 `engine_out` rows at batch 8 still tie or lose by up to 9% in single
runs. C++'s NEON bf16 paged kernel returns NaN on poisoned/padded cache pages (26 of 412
`test_regressions` fail with `CPU_ATTENTION_BACKEND=cpp`, identically on the untouched tree;
`CPU_ATTENTION_ISA=scalar` passes), so the scalar path is the valid peer there.

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
