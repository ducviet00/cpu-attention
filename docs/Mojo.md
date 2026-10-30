# The Mojo kernel

`cpu_attention/_mojo/` is the library's kernel: one unified CPU attention for
both public APIs, `flash_attn_func` (FlashAttention over supplied Q/K/V) and
`flash_attn_with_kvcache` (one-token paged decode, PagedAttention), built with
Mojo 1.2.0 (pinned in `pyproject.toml`). It is measured against torch SDPA
(`flash_attn_func`) and vLLM's CPU attention (paged decode). A C++ backend served
as the competitor until it was removed (Oct 2026; its design and ledger are in
git history, `docs/CPP.md` before the removal).

```bash
uv sync --extra build                                   # torch + mojo==1.2.0
uv run --extra build python scripts/build.py --tests    # every tier + the standalone kernel test
```

## 1. Status

Current per-box results, the verdict rule and the caveats are in the
[README](../README.md#performance), which is the single source of truth for
numbers; this file does not repeat them. The objective is a clear per-row win
over SDPA and vLLM on all three machines. Where rows tie, the limit is:

- **Decode that streams more cache than fits in L2** (batch 8-32, context ~1024
  and up): every kernel runs at the same memory roof, so a few percent either
  way is placement and noise, and the same row moves 20-30% with allocation
  order. Neither a compile-time page 64, extra dot chains, key-tile changes,
  software prefetch nor wider key groups moved these rows (§9).
- **bf16 decode at head_dim 128** (MHA, long contexts): vLLM is 1.06-1.13x
  faster on c4-standard-* (its AMX path) and 1.04-1.22x on c4a-standard-*
  (README); fp32 and c4d-standard-* tie or win.
- **GQA/MQA are out of scope for now**: the API raises `NotImplementedError`
  for `nheads_k < nheads`, and the group paths are removed from both kernels
  (in git history before the removal: a group's query heads as rows of one
  decode register tile, `R` up to 8). When last measured (`0081f7c`),
  GQA decode at head_dim 128 (32 query heads over 8 or 4 KV heads) ran 1.34-1.86x
  behind vLLM in bf16 on c4-standard-*, where vLLM uses AMX tiles for the group,
  and 1.1-1.6x behind on c4a-standard-*; an AMX (and a `bfmmla`) `PVKernel` for
  groups is the candidate.
- **fp32 `flash_attn_func`** is FMA-bound (parity with the former C++ backend
  single-thread at every shape measured).

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
_mojo/src/amx.mojo        AMX tile primitives (inline asm, x86 only)
_mojo/src/_C.mojo         the Python extension: one native entry per op (decode, store, flash)
_mojo/src/torch_tensor.mojo  torch tensors read in place, outputs allocated (libtorch C shim)
_mojo/tests/test_attention.mojo   standalone decode parity test vs a scalar fp32 reference
```

## 3. One design for both APIs

**The page layout is shared.** flash_attn_func packs each (batch, head) into a
private slab whose K^T is blocked every 64 keys -- exactly a KV-cache page of 64
tokens: `[tile][hd][64]` fp32, `[tile][hd/2][64][2]` bf16 (the dim pair a dot
instruction contracts). The decode cache uses the same dim-pair K layout per
page. So Q@K^T, the online softmax and P@V are the same algorithm over pages of
keys: decode scores one query row per (sequence, head), flash_attn_func a
register tile of consecutive query positions. MHA only (GQA/MQA raise in the API;
the group paths were removed from both kernels, §9).

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

**Decode.** Work item = (sequence, head), one query row; workers take contiguous
ranges balanced on context length (`_share`). P@V never reads past `lens`, a
negative slot skips its store, a zero-length context returns 0. The page size is
a compile-time constant for 16/32/48/64.

**One decode driver for every tier** (`paged_attn._attend`). A softmax step of
`KT` keys is resolved into 16-key units (page % 16 == 0, so a unit never
straddles a page): one key vector on AVX-512, two on AVX2, four on NEON. Q@K^T
scores `G` key vectors per call (independent dot chains) with a masked tail that
stays inside one unit, tracks the step maximum as a tree, and the softmax keeps
`NS` partial sums. What differs per target is compile-time only:
- `Decode[dtype, hd]` holds the shape as `comptime` members computed from
  `simd_width_of` and `CompilationTarget` (`KT`, `G`, `G_TAIL`, `DEG`, `NS`, `NC`,
  `FULL`); `comptime assert` pins the tiling invariants.
- `Decode.PV` is the one per-target kernel, a type conforming to the `PVKernel`
  trait (as `InnerMatmulKernel` in Modular's unified matmul): `PVRows` keeps
  fp32 P against V rows, the x86 choice (bf16 V is widened even/odd split,
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
supports passes the tests (c4d-standard-*: avx512bf16, avx2, x86-64;
c4-standard-*: vec, avx2, scalar; c4a-standard-*: bf16, neon). bf16 K is dim pairs on every tier, so a cache written under one tier
is read correctly under another.

## 5. Numerics and gates

The tests (`tests/`, pytest) compare every case against `tests/reference.py`:
eager attention in PyTorch float64. Limits:

| case | limit vs float64 |
|---|---|
| flash_attn_func fp32 | 2e-4 |
| flash_attn_func bf16 | 3e-2 |
| paged fp32 | 1e-6 |
| paged bf16 | 1e-3 above the bf16 output-rounding floor |

**Paged bf16 "<= 1e-3 against the reference".** Taken literally (absolute error
to the real-valued result) no bf16 output can meet it: an earlier campaign
measured 77/252 Axion sweep cases where rounding the exact result to bf16 alone
exceeds 1e-3 (worst 6.6e-3). The tests therefore check the error *above the bf16
output-rounding floor*, `max(|out - ref| - |ref - bf16(ref)|) <= 1e-3`
(`tests/reference.check`), on every CPU. This is a stated interpretation
of an unattainable literal gate, not a passed literal gate. (Until the C++
backend was removed, x86 bf16 was judged instead as <= 2 bf16 ulp against it,
which its degree-2 `exp2` met; the reference gate needs degree 5, §9.) The fp32
limits hold at `softmax_scale <= 1/sqrt(head_dim)`, and the tests stay there: fp32
softmax error grows with the logits (c4a-standard-8, d=128, MQA, ctx 708: scale 0.2,
2.3x the model scale, up to 2.2e-6 from fp64; at the model scale 1.4-2.2e-7), and at
scale 0.9 even an fp32 reference is 6.4e-6 from fp64.

## 6. Call path and threading

- **Torch's OpenMP pool.** One `GOMP_parallel` region per call (~2.4 us warm);
  MAX's `parallelize` and private pools were slower. Slices
  use `omp_get_num_threads()`; workers are capped at physical cores.
- **One native call route.** Each public function checks its unsupported options
  in Python and passes the tensor objects to one Mojo entry per op (`_C.decode`,
  `_C.store`, `_C.flash`). The entry reads each tensor's TensorImpl fields in
  place (`torch_tensor.mojo`, at offsets `_kernel.runtime()` finds at import by
  matching what torch reports and verifies on probe tensors), validates them,
  computes the launch parameters, allocates the output through libtorch's C shim
  and runs the kernel. Nothing is cached per geometry. It returns None for
  anything it does not take; Python then raises the precise error, or copies a
  layout it cannot read in place and calls once more. A torch whose tensor layout
  the calibration cannot match fails on the first call (no silent fallback).
  Laptop, decode b=1 ctx=256 bf16: warm 3.5 us, cold 43.1 us against a 39.9 us
  bare kernel (§9 for the routes it replaced).
- **Workspace** is per thread, grown on demand, never shrunk (a freed large
  tensor is an mmap that faults in again).

## 7. Measuring

`benchmarks/BENCHMARK.md` describes the benchmark: the Python API of each
library, cold L2/L3 before every call, 3 warmup + 10 samples interleaved across
arms, the median with min and max. `--compare <package dir>` adds another build
of the package to the same process for an A/B (a private copy, imported under its
own name, so the two share no module state or `dlopen` handle).
`scripts/pin_governor.sh` pins `performance` on a laptop or desktop and restores
each CPU's original setting. Ledger entries below that cite "full manifest",
`mojo_ctrl`, verdict counts or cold/warm rows were measured with the previous
harness (170 rows, an independent control copy, the minimum of 41-121 samples);
it is in git history before the restructure.

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
| vector flash policy: the rescale rides P@V's accumulator load and a row tile's first key tile starts from zero (`ZEROES_ACC`: no rescale pass, no clear) | Axion fp32 flash 0.992 (16 threads 0.990); x86 neutral (Zen 4 one thread 0.99-1.00) | kept |
| decode `PVRows`: two key pairs per loop iteration (the loop's adds and compares were 11% of a cache-resident bf16 decode) | Zen 4 one thread: 6% fewer instructions, cycles 0.85-0.93; boxes paged 0.987-1.000 | kept |
| bf16 vector P@V one dim vector per pass, so the probability broadcast folds into `vdpbf16ps` (a separate `vbroadcastss` was 16% of the call) | Zen 4: 4-6% fewer instructions, cycles ~0.98; Zen 5 (C4D-16): head_dim 64 1.17, 128 1.15 | reverted: not a gain on every AVX-512 core |
| flash plan: at least ~8 work items per worker (halve `rq`, clamped to `seq_q`, down to one register-tile pair) | Axion-16 1024:8:32 fp32 802 -> 712 us (C++ 704), bf16 329 -> 296 (C++ 323); 8 threads unchanged. The first version halved before clamping: 17-row calls split into a 1-row block and ran 15x slower | kept (fixed) |
| Axion fp32 flash vs the old dedicated NEON policy, with hardware counters (c4a-standard-16, `PMU=standard`; C4D offers no PMU on any size): the FMA phases already ran fewer instructions; all of the +7-16% was outside them | 1024:8:32: 123.6M -> 136.7M instructions, FMA phases 94.2 -> 91.8M, other 7.6 -> 19.7M | diagnosis |
| causal mask from one `rel = key - limit0` vector and per-row constants (8 runtime per-row limits spilled and were reloaded on every masked pass) | Axion hd 32 cycles 1.126 -> 1.049 of the old policy (one run) | kept |
| one K^T base per Q@K^T pass, constant key/dim offsets (per-load block arithmetic kept NEON from pairing loads: `ldp` 10.9% of samples in the old policy, 1.2% here) | instructions vs the old policy hd 64 1.109 -> 1.072, 512:32:128 1.074 -> 1.038; cycles 1.06 -> 1.02 | kept |
| key tile `max(256, 16384 / hd)` (512 at hd 32) | -1.3% instructions, cycles within the VM's +-5% run-to-run spread | reverted |
| `isa.online_max`: the one running-max step (vector, NEON and AMX flash, decode; AMX keeps its slack and skip) | final round (`cf9688d`, vs `200e6e2`): flash new/previous Axion-16 0.992-0.997, Xeon-16 0.995, C4D-16 0.991-0.999; paged 0.978-1.004; `test_amx` 102/102 | kept |
| C4D decode "losses" (40 rows at 1.03-1.38) | isolated, cold-cache (8 cache copies, 8 threads): Mojo = C++ at ~60-65 GB/s (bf16 b=1 ctx=4096 261 vs 264 us, b=8 ctx=1024 512 vs 515); warm: Mojo = or faster (fp32 b=8 ctx=256 30 vs 63-69 us). Same layout, partition, page order and OpenMP pool (one `libgomp`); a 512-key tile changed nothing. Two arms interleaved in one process: C++'s warm call 186 us vs its own isolated 333 (more of its lines survive the other arm's stream). A first "previous call slows the next" result was an artifact (the same cache copy reused) and was redone | diagnosis: a measurement artifact |
| bench: decode rows above LLC/4 per arm timed cold (read-only sweep of 2x LLC before every sample, every arm) | C4D 93/36/41 -> 96/73/1 (decode losses 40 -> 0); Axion 106/60/4 -> 115/54/1; Xeon (reports 260 MB L3, 8 cold rows) 87/78/5 -> 89/77/4 | kept (user decision) |
| store-only K through a LANES x LANES word transpose for prefill-style staging (consecutive slots); scattered appends keep token order | C4D store `b=1 ctx=1024` fp32 1.14 -> 0.97 of C++, bf16 42.6 -> 20.9 us (0.21 of C++); `b=32 ctx=1` unchanged (blocks for scattered tokens had cost it 0.4 us) | kept |
| C++ backend removed; x86 paged bf16 judged like aarch64 (<= 1e-3 above the bf16 output rounding vs the fp32 reference) instead of <= 2 ulp vs C++; `exp2` degree 5 for bf16 decode on every CPU (x86 was 2) | degree 2 failed the reference gate on 91/252 sweep cases (worst 3.9e-3); degree 5 worst 7.0e-7 on every x86 tier; +1.2% instructions on the Ryzen, cycles within noise | kept: one numerics rule and one `DEG` for every CPU |
| decode tasks split across workers by cost (context + 16 keys per task, contiguous ranges) instead of by count (`_share`) | equal counts gave each Axion thread one whole sequence of `ctx=het` (17-2047 keys x 16 heads on 8 threads): the 2047-key thread did 3.3x the mean. Against an independent previous build: Axion `ctx=het` 0.34-0.41 (fp32 1221 -> 430 us, bf16 305 -> 154; vs vLLM 2.7x slower -> 0.71-0.89), C4D (4 threads) 0.90-0.97, Xeon (4 threads) 0.63-0.80; the rest of the manifest median 0.995-1.016 | kept |
| GQA/MQA paths removed from both kernels (decode: no `R`-row register tile, `_rows_tile`, group chunks or store barrier; prefill: no `kv_heads`/group mapping), `157f6c6` vs `56bd38c` in one process, 3 warmup + 10 cold samples | median new/old: c4-standard-8 prefill 0.985, decode 0.999; c4d-standard-8 0.992, 0.999; c4a-standard-8 1.001, 1.013 (every row within the sample spread, both directions); 535 lines -> 384 | kept |
| the native call route removed (`pyview.mojo`, `kv_fast`/`flash_fast`, the import-time TensorImpl calibration and the per-thread geometry tables, ~750 lines); the remaining route validates and plans once per geometry | warm calls on the Ryzen laptop, decode b=1 ctx=256 bf16/fp32: old checked route 10.7/11.9 us, native route 4.1/3.7, now 5.5/6.4; prefill 17:8:64: 5.4/5.8, 3.2/2.5, now 4.0/4.4 | kept: one route |
| the cold-cache cost of that removal, and a supported native route (vLLM's: a dispatcher custom op reading tensors through the official API; the stable C ABI `aoti_torch_library_impl` + `aoti_torch_get_*` would give Mojo the same), prototyped as a C++ dispatcher op with the decode signature that reads every tensor's metadata and runs no kernel | Ryzen laptop, decode b=1 ctx=256 bf16, median, caches flushed before each call (warm in brackets): bare kernel 39.9 (2.5) us; removed native route +3.1 (+0.6); single Python route +12.2 (+3.1: key + lookup 6.7, output allocation 4.0, launch 2.4); dispatcher op alone, no kernel, 9.9 (2.7). On the VMs the single route was 15-35 us slower than the native route on short calls (c4-standard-8 `b1 short` 63 -> 96 us); larger rows unchanged | dispatcher route rejected: `torch.ops` runs as much cold Python as our route; only reading tensors in Mojo skips it |
| the native route rebuilt as the only route: one Mojo entry per op validating and planning every call from TensorImpl fields (no geometry tables, no registration, Python only for options and error messages) | laptop, decode b=1 ctx=256 bf16 cold / warm: 43.1 / 3.5 us (old native route 42.4 / 3.0, single Python route 52.1 / 5.5); prefill 17:8:64 15.4 / 2.7 (14.5 / 2.5); Florence-2-large decode simulation, attention per step at b=1: 772-792 us (old 770, Python route 1046); +338 lines over the Python route. In-process A/B against the old native route on c4-, c4a-, c4d-standard-8 (`7026f86` vs `efc28ee`): median new/old 0.99-1.03 everywhere; `b1 short` decode 1.08-1.16 on c4a/c4 (~5 us; ranges overlap), parity on c4d (the single Python route had read 1.41-1.70) | kept |
| the first version of that removal: an `R = 1` epilogue with only compile-time offsets into the decode accumulator buffer | LLVM promoted the buffer to vector registers live across the key loop and spilled the Q@K^T accumulators: aarch64 bf16 d=128 hot loop 155 stack accesses against 57 (the old runtime row loop had blocked the promotion), c4a-standard-8 decode 4-14% slower (confirmed with the arms swapped). Fixed by runtime loops in the epilogue (50). An out-of-line epilogue also fixed the spills but returned wrong results: the caller's stores into the stack buffer were dead-eliminated, as with the PV helpers above | fixed |
