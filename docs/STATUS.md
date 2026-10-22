# Where the campaign stands

`docs/PROTOCOL.md` says how numbers are taken; `docs/CEILINGS.md` says how much
headroom exists at all (it is the file to read before proposing another
optimisation); `docs/BENCHMARKS.md` holds older tables. This file is the summary
and the log of what actually moved.

## The one number

`ours / baseline < 1`, for both baselines (`torch SDPA` and vLLM `csrc/cpu`), on
both platforms, for all 22 rows of `tests/bench_vs_refs.py`. Below 1 is a win.

## Where it stands, 2026-10-22 (`main` at `cacc785`, m9g.xlarge, 4 cores)

Verified by the integrator: clean rebuild from scratch, `ROUNDS=21`, one
process, all variants interleaved. **22/22 against SDPA, 8/22 against vLLM.**

| shape | bf16 o/sdpa | bf16 **o/vllm** | fp32 o/sdpa | fp32 **o/vllm** |
|---|---:|---:|---:|---:|
| S=128 H=8 D=64 | 0.58 | **1.26** | 0.35 | **1.07** |
| S=512 H=8 D=64 | 0.46 | **1.09** | 0.33 | 1.00 |
| S=1024 H=8 D=64 | 0.62 | 1.00 | 0.48 | 0.99 |
| S=2048 H=8 D=64 | 0.73 | **1.02** | 0.60 | 0.98 |
| S=4096 H=8 D=64 | 0.79 | **1.01** | 0.71 | **1.01** |
| S=256 H=16 D=128 | 0.51 | **1.20** | 0.45 | 0.93 |
| S=1024 H=16 D=128 | 0.64 | **1.06** | 0.59 | 0.90 |
| S=2048 H=16 D=128 | 0.73 | **1.09** | 0.72 | 0.92 |
| S=4096 H=16 D=128 | 0.82 | **1.09** | 0.92 | **1.03** |
| S=512 H=32 D=128 | 0.46 | **1.15** | 0.43 | 0.96 |
| S=1024 H=8 D=32 | 0.62 | 0.99 | 0.48 | **1.02** |

The tail that has to move, in the order it is worth attacking:

| group | rows | needs |
|---|---|---|
| 1.00-1.03 | 1024:8:64 bf16, 2048/4096:8:64 bf16, 512:8:64 fp32, 4096:8:64 fp32, 4096:16:128 fp32, 1024:8:32 fp32 | 2-4% |
| 1.06-1.09 | 1024/2048/4096:16:128 bf16, 512:8:64 bf16 | 6-9% |
| 1.15-1.26 | 128:8:64 bf16+fp32, 256:16:128 bf16, 512:32:128 bf16 | 13-26% |

### x86 without AMX (laptop Ryzen, VEC path) — a current data point, not the target

The Xeon table above is the one that counts and it is stale, so here is the only
x86 number that is current: `main`, two runs, `scripts/band.py`, VEC reference,
8 threads. It exercises the same shared call path, packs and dispatch, so it
catches a shared-code regression even though the ISA is not the target's.

| | wins vs SDPA | wins vs vLLM-VEC | notes |
|---|---|---|---|
| bf16 | 11/11 | **8/11** (0.74-0.97) | loses 128:8:64 (1.07-1.45) and 4096:16:128 (1.01-1.03) |
| fp32 | 11/11 | **1/11** | 1.09-1.90; the AVX-512 fp32 path with no AMX is the weak one, and on the Xeon fp32 runs on AMX |

So the VEC fallback is not competitive for fp32. That is not a regression — the
AVX-512 fp32 path has always been the weakest of the three — but it means a
machine with AVX-512 and no AMX should be expected to lose the fp32 rows, and
`docs/NEXT.md` item 1 is what settles the Xeon question.

## What moved, and what it cost

**Second ARM campaign (merged `cacc785`).** `o/vllm` 4/22 -> 8/22.
fp32 4/11 -> 7/11 (K/V pack vectorised — `zip1/zip2` instead of per-lane
inserts plus a runtime-`kt_blk` division in the inner loop; Q@K^T unrolled over
head_dim). bf16 long-S went 1.03-1.13 -> 1.00-1.03 with three precision-neutral
fixes: the exp pass with four key groups in flight (+2.3%, the phase is
latency-bound at 1.75 cycles/row-key for 2.5 instructions), `KS_STRIDE` instead
of the runtime `job.kt_blk` in the sub-block address (it was an `sdiv`+`msub`
per 8 keys, +1%), and FA4's -8 rescale slack (+1.3% long S, +2% at 512).

**The host path (merged `c769df3`).** The Python side of one call went 5.0 ->
3.4 us: a per-geometry call plan (kernel path, `qt`, worker count, scratch
pointer, and the ~30 constant extension arguments, cached per thread) plus one
`.shape`/`.stride()`/`.dtype` read per tensor instead of nineteen `.stride(i)`
calls. It is charged to us by the scorecard and it is 7% of a 51 us call at
S=128. `tests/test_dense.py` covers the cache: stale entries, new data pointers,
a causal flip, `clear_scratch`, and two threads (64 checks).

**Falsified, with the number** (each of these is a day not to spend again):

* **int8 Q@K^T** (`smmla`, 2x the MACs of `bfmmla`): 19% faster and it misses the
  fixed 3e-2 bf16 gate. `tests/int8_qk_error.py` simulates the instruction
  exactly — per-key K scales give 2.0-3.5e-2 on the benchmark shapes (fails at
  head_dim 128), and every scheme is 2.5-3x worse than the shipped bf16 kernel,
  which is itself at 2.6-7.4e-2 on adversarial families. The hybrid (bf16 only
  for the first 256 rows) passes on `torch.randn` and fails on all four
  adversarial families. Closed, not gated off.
* **fp16 MMA (`fmmla`)**: does not exist on this part. `f32mm` is unshipped;
  LLVM has no instruction-selection pattern for the intrinsic. `bfmmla` at
  207.6 GFLOP/s/core is the bf16 ceiling and vLLM spends 95% of it at
  S=4096/D=128.
* **8 key groups in flight** in the exp pass: 5.75 cycles/row-key against 5.36
  for four. **Estrin** for the same polynomial: 5.80. **`dcp=8` at head_dim 128**
  (halves the P@V panel reloads): 12.39 against 9.58, `RP*dcp = 32` accumulators
  spill.


## The falsification ledger, with the number that killed each idea

Every one of these cost a session somewhere, so they are written down with the
measurement that closed them. `docs/CEILINGS.md` has the ceiling arithmetic
behind the last two.

| idea | why it looked good | what killed it |
|---|---|---|
| int8 Q@K^T (`smmla`, 2x the MACs) | 19% faster, and the ARM agent measured it | misses the fixed 3e-2 bf16 gate at every scaling that fits: 2.0-3.5e-2 on the benchmark shapes, per-key K scales (1.4x better than per-tile) still fail at D=128, and all four adversarial input families fail — including for the *shipped* bf16 kernel, which is itself at 2.6-7.4e-2 on them (`tests/int8_qk_error.py`) |
| fp16 matrix multiply (`fmmla`) | same 16 MACs as bfmmla, 3 more mantissa bits | the extension does not exist on any shipping core: LLVM has no instruction-selection pattern for the intrinsic (`Cannot select: intrinsic %llvm.aarch64.neon.fmmla`) |
| wider vectors (SVE) | Neoverse-V3 has SVE2 | `sve_default_vector_length` = 16 bytes: 128-bit, same as NEON |
| K^T layout, then V layout | `l1d_cache_lmiss_rd` 0.348 vs vLLM's 0.086 per row-key | both reorderings leave the miss count unchanged: the misses are *compulsory* stream misses, 55-62% of them already covered by the prefetcher |
| more rows per work item (`qt`) at short S | the per-work-item cost is ~780 cycles | qt=8 is already optimal (interleaved sweep 4/6/8/10/12) and forcing qt=4 on the tile kernel is +22% cycles/row-key, because rows are the only K/V reuse axis |
| the K/V pack | `bench_phases` says 6 us at S=128 | `bench_phases` runs `NO_FUSE`, which adds a fork-join to that column; called alone the pack is 2.90 us = 88% of a plain memcpy, at every size up to 16 MB |
| a fused pack+attention pass | would skip a write+read of the packed copy | there is nothing to skip: the pack is 6% of the call at S=128 and 0.3% at S=4096 |
| three-phase split (pack / QK+softmax / V pack + P@V) | hides the V pack behind the softmax | deferring P@V makes every item's probabilities survive; the buffer costs 2-8x the phase it hides |

## Two rules that keep paying for themselves

**Show spare *issue* slots, not spare matmul cycles.** A microbenchmark of pure
`bfmmla`s against pure `fmla`s does not mean a softmax hides inside a matmul
loop: the Q@K^T body is 26 instructions per 16 `bfmmla` (3.25 IPC against a
machine that sustains 4.0), so it has nothing to give. Count the instructions
*inside* the loop that is supposed to host the work.

**Measure counters by subtracting two iteration counts, and only trust
interleaved timings.** `perf stat` over a python driver is mostly torch import;
these boxes also throttle 2-2.5x, and absolute times and ratios move 30-50%
between sessions. `tests/bench_vs_refs.py` and `tests/bench_scale.py` interleave
every variant in one process; `tests/ab_two_trees.py` alternates short processes
for tree-vs-tree.

## What the remaining rows need (from `docs/CEILINGS.md`)

Short S is **per-work-item cost**, not the matmul: the kernel's own rate at a
128-key range is 44% of the bfmmla ceiling however many heads are in flight
(4.55 cycles/row-key against 2.70 at 2048 keys), and the fit puts ~1000 cycles
of prologue/epilogue on every work item — 3% of a long-S item, ~40% of a
128-key one. vLLM sits at 50% of ceiling at S=128 *including* its per-call
costs, so that row needs the per-item cost cut.

The D=128 rows at 1.06-1.09 are 80-85% of ceiling against vLLM's 89-93%, and the
measured difference is L1 read misses (0.348 vs 0.086 per row-key) from the K^T
pack layout being read 16-strided.
