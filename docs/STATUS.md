# Where the campaign stands

`ours / baseline` (below 1 wins) on every row of `tests/bench_vs_refs.py`,
min-based, interleaved in one process, box idle.

**Current numbers: `docs/AUDIT.md` (2026-10-23)** re-measured both ISAs with an
independent harness and a naive fp32 reference on fresh boxes: an
`m9g.2xlarge` (Graviton 5 / Neoverse-V3, 8 cores, 8 threads) and an
`m8i.2xlarge` (Xeon 6975P-C, 4 cores / 8 threads, 4 threads), against vLLM
`db3b0f0` built as NEON / VEC / AMX. Read this file with that report; several
per-row numbers that used to live here moved 5-20% when the box or the process
changed. The **older x86 scoreboard was measured on an Emerald Rapids Xeon
8581C (GCP)**, a different CPU from the 6975P-C; those values are kept below as
history and do not carry over row for row.

## The scoreboard (2026-10-23 audit)

| | x86 bf16 vs vLLM AMX | x86 fp32 vs VEC | ARM bf16 | ARM fp32 |
|---|---|---|---|---|
| rows below 1 | **9/11 min, 6/11 median** | **11/11** (one 9-round process: 10/11) | **10/11 median, 11/11 min** | **11/11** |
| worst row | 1.02 min (`512:32:128`); 1.09 median (`256:16:128`) | 1.04 median / 0.95 min (`128:8:64`) | 1.02 median (`2048:16:128`); 0.997 min | 0.982 (`4096:16:128`) |

The duplicate `ours` arm (`ctrl`) read **0.97-1.04 on m9g** and **0.92-1.04 on
m8i**, so a count that differs by one row and a per-row value within 4% are not
resolvable on these boxes.

## ARM per row (m9g, 8 threads, 41 rounds, min / median `ours/vllm`)

128:8:64 **0.974 / 0.958**, 512:8:64 **0.881 / 0.890**, 1024:8:64
**0.873 / 0.883**, 2048:8:64 **0.936 / 0.938**, 4096:8:64 **0.927 / 0.943**,
256:16:128 **0.886 / 0.952**, 1024:16:128 **0.945 / 0.959**, 2048:16:128
**0.997 / 1.017**, 4096:16:128 **0.994 / 0.994**, 512:32:128
**0.946 / 0.957**, 1024:8:32 **0.826 / 0.842**. fp32 is **0.788-0.982 min**
(0.790-0.984 median), 11/11.

The two rows at the line are the bf16 D=128 long rows; 2048:16:128 is 0.99-1.01
across repeat runs and 1.02 on the median, so "10/11" is the honest count. The
older text's `256:16:128 = 0.99` was one instance's reading of a row that moves
between 0.89 and 1.01 on m9g; the `fp32 0.79-0.95` band ends at **0.982**, not
0.95. **Every row beats SDPA**: 22/22, `o/sdpa` **0.277-0.874** (1.14-3.61x),
not 0.29-0.88.

## x86 per row (m8i, 4 threads, 41 rounds, min / median `ours/vllm`)

bf16 vs vLLM's **AMX** build (the honest bf16 bar on an AMX box):
128:8:64 **0.854 / 0.930**, 512:8:64 **0.803 / 0.907**, 1024:8:64
**0.885 / 0.906**, 2048:8:64 **0.864 / 0.823**, 4096:8:64 **0.802 / 0.822**,
256:16:128 **0.981 / 1.090**, 1024:16:128 **1.010 / 1.019**, 2048:16:128
**0.953 / 1.031**, 4096:16:128 **0.976 / 1.020**, 512:32:128
**1.018 / 1.023**, 1024:8:32 **0.827 / 0.869**. That is 9/11 by min, 6/11 by
median: the kernel is at parity on the D=128 long rows, **not 15-22% ahead** as
the 8581C table below recorded.

fp32 vs VEC: 11/11 in the 41-round run; 128:8:64 **0.946 min / 1.044 median**
(ctrl 0.93), 1024:8:64 0.563 / 0.708, 4096:16:128 0.731 / 0.732. One 9-round
process read 128:8:64 at 1.13, so quote that row as **parity (0.95-1.13)**, not
as a win. bf16 vs VEC is 0.20-0.55, so "22/22 beat VEC" holds but VEC is the
weak bar for bf16.

**Every row beats SDPA**: 22/22 on m8i, `o/sdpa` **0.21-0.79** (1.3-4.7x). The
older "x86 1.6-3.7x (0.27-0.63)" range is too narrow.

## The 8581C scoreboard, as originally recorded (history)

Per row, x86 bf16 (three clean runs, min-based, `ours/vllm`): 128:8:64
**0.87-0.93**, 512:8:64 **0.80-0.88**, 1024:8:64 **0.88-0.89**, 2048:8:64
**0.83-0.84**, 4096:8:64 **0.70-0.72**, 256:16:128 **0.99-1.05**, 1024:16:128
**0.98-0.99**, 2048:16:128 **0.78-0.83**, 4096:16:128 **0.83-0.85**, 512:32:128
**0.99-1.01**, 1024:8:32 **0.84-0.85**: 10/11 below 1, worst 1.02. On m8i the
short D=64 rows reproduce (0.80-0.89) but 4096:8:64 reads 0.80 instead of
0.70-0.72 and the two D=128 long rows read 0.95-0.98 instead of 0.78-0.85. The
older ARM per-row list (128 0.99, 512 0.87, 1024 0.88, 2048 0.93, 4096 0.92,
256 0.99, 1024:16:128 0.97, 2048:16:128 1.01, 4096:16:128 1.01, 512:32:128
0.97, 1024:8:32 0.84) is within 3% of the audit except 256:16:128 (0.89 here).

Where the 8581C numbers came from, same box, same harness (x86 bf16,
`ours/vllm`; kept as the campaign's history):

| row | start of session | end |
|---|---:|---:|
| 128:8:64 | 1.55 | **0.93** |
| 256:16:128 | 1.33 | 1.05 |
| 512:8:64 | 1.16 | **0.88** |
| 512:32:128 | 1.17 | 1.03 |
| 1024:16:128 | 1.02 | **0.99** |
| 4096:16:128 | 0.87 | **0.83** |
| 128:8:64 fp32 | 1.43 | 1.04 |

## What moved it, in payoff order (x86)

1. **One work item = one (batch, kv head)**: a thread packs that head's K/V and
   then attends every query block of it, instead of a pack phase, a barrier and
   a separate attention phase. Re-measured on m8i: **1.16x at 128:8:64,
   1.17x at 512:8:64, 1.14x at 256:16:128, 1.00-1.08x on the long rows**
   (`mode & 512` selects the old order for an in-process A/B). The original
   session recorded 1.47/1.29/1.16 on the 8581C; that overstates the short rows
   by 20-30%. Two effects are real: the head's packed K/V is 32 KB at
   D=64/128 keys and stays in L1 for its own query blocks, and a whole head is
   a fixed amount of work, so the causal imbalance goes away.
2. **A 5-argument call entry** (`dense_amx_plan`): the geometry (strides, tile
   counts, workspace addresses, kt, workers) rides in an int64 buffer instead
   of 36 Python arguments, and the per-call rebuild is cached per thread.
   `pub` 43.1 -> 37.0 us at 128:8:64 on the 8581C; m8i measures 23-40 us for
   the whole call depending on the process, so the call path is not the
   1.28x. Keep the entry (it also makes the A/B legible), not the headline.
3. **comptime `head_dim`**: every tile offset and loop bound folds to a constant
   (1.00-1.03x on its own; it is what makes the rest legible).
4. Merged from the agents: the ARM short-sequence kernel's cheaper exponential
   and `_SMALL_MAXK` 256 -> 1024, and a second AMX kernel
   (`dense_amx_v2.mojo`, `CPU_ATTENTION_AMX2=1`) kept opt-in: v1 is **1.51x**
   faster at 128:8:64, at parity on the long rows.

## What the machine actually costs (measured, x86)

`_mojo/tests/bench_amx_mma.mojo`, `bench_amx_item.mojo`, `prof_amx_callpath.py`,
`prof_amx_pack.py`; agent A's `bench_amx_ceiling.mojo` independently agrees.

| | value |
|---|---|
| one `tdpbf16ps`, operands in tiles | 5.5 ns = 2.98 TFLOP/s/core (4.86 ns / 3.37 by agent A's probe) |
| one 1 KB tile load | 6.4 cycles (160 B/cycle) |
| tile loads behind dots | free |
| AMX + vector FMAs, interleaved | same time as AMX alone -- **the pipes do overlap** |
| the item, per phase, D=128 | Q@K^T 0.32, softmax 0.54, P@V 0.30 cycles/logit |
| the item, total | 1.25-1.54 cycles/logit (L2-resident / 4 MB working set) |
| softmax, isolated vs in situ | 0.176 vs 0.265 ns/logit -- **the remaining 1.3x** |
| the pack | 0.2-1.7% of a call for S>=512, ~10% at 128:8:64; 243-695 GB/s depending on working set (`docs/REMEASURE.md`) |

**Correction (2026-10-23 audit).** The older "2.0-2.8 TFLOP/s against a 3.0
TFLOP/s dot roof" compared an aggregate (4 threads) with a per-core roof. The
whole dense call runs at **1.9-2.5 TFLOP/s aggregate on 4 cores = 0.5-0.6
TFLOP/s/core**, i.e. **17-21% of the 2.98 TFLOP/s/core `tdpbf16ps` roof**; vLLM's
AMX attention measures **0.65 TFLOP/s/core**, so the two are at parity, not 3-4x
apart. The item's matmul phases alone are the higher number in the table above.
oneDNN's bf16 GEMM on this box was recorded at 1.11 TFLOP/s/core; it has not
been re-measured.

## Falsifications (the number that killed each)

| idea | result |
|---|---|
| pipeline the softmax under the Q@K^T dots (chunk-lagging max) | probe 1.65x, **kernel 0.73-0.91x**. Agent A reproduced the sign independently (1217 vs 1185 us). The overlap is real; the per-chunk horizontal reduce costs more |
| cheaper exponential: degree-3 for bf16, `round()`, one-sided clamp, `llvm.ldexp`/`vscalefps` | 3 instructions fewer per 16 logits, **10-27% slower** end to end while a single-threaded probe of the same loop read 8% faster |
| interleave the K and V packs for memory-level parallelism | 1.00-1.02x |
| a running V pointer + hoisted P pointers in P@V | 1.00x (LLVM already strength-reduces it) |
| kt 128 / 384 / 512 vs 256 | **0.90 / 0.94-1.07 / 0.94-1.07**, not flat: kt=128 is 10% *faster* at 128:8:64 and 7% slower at 4096:16:128; 256 is a compromise, not a corner |
| 8 SMT workers | 27620 vs 24314 us at 4096:16:128 (8581C) |
| v1's `mode & 64` P@V ablation bit | measures nothing (`nk // 32 == (nk + 31) // 32` for a full super-tile); P@V is 7-14%, not 1% |
| v2 (second kernel) as the default | **1.51x slower** than v1 at 128:8:64; parity or a 3-5% *win* on 4096, so the "0.93-0.99 elsewhere" claim did not reproduce |
| per-head item worth 1.47/1.29/1.16x | **1.16/1.17/1.14x** on m8i; the short-row gains were overstated by 20-30% |
| AMX is 4-9.6x over the AVX-512 path | **2.8-9.8x**: 2.80x at 128:8:64, 3.18x at 1024:8:32, 9.79x at 4096:16:128 |
| "Xeon 4/22, Graviton 17/22 beat SDPA" | **22/22 on both** boxes, `o/sdpa` 0.21-0.79 and 0.28-0.87 |
| "still 1.44-1.8x behind vLLM" (ARM) | **0.79-1.02**, i.e. parity; the 1.44-1.8x was the pre-`bfmmla` tree |

## ARM: the bfmmla ceiling is not settled

Agent B's three probes disagree by 2.4x: 128-134 MAC/ns/core for a 16-accumulator
no-load loop, 168 for a variant at 3.3-3.5 `bfmmla`/cycle, and 61 for the same
loop with 4 `ldp` feeding the next instruction pair. So "the matmul is at the
wall" (94% of a measured ceiling) is **not established in either direction** --
the number is a property of the loop shape, not of the core. What does hold is
the per-row verdicts and the 68-85% range for the matmul phases depending on
which probe you believe. `docs/CEILINGS.md` and `docs/FINDINGS.md` carry all
three probes; the honest next step is to instrument the kernel's own Q@K^T body
rather than another microbenchmark. The audit measured the whole call at
**1.34-1.59 TFLOP/s** (8 threads) / 0.71-0.85 (4 threads), which is the number
to compare against those ceilings.

## Beyond these boxes' resolution

1. **The boxes move more than the old text says.** The m8i short rows read
   0.85-1.25 (128:8:64) and 0.80-1.33 (512:8:64) across four processes of the
   same binary; m9g rows move 5-10% between instances. The duplicate-arm floor
   is 0.92-1.04 (m8i) / 0.97-1.04 (m9g). A verdict needs 41 rounds (or
   `scripts/band.py` over several processes), not 9.
2. **The first timed variant in a fresh process reads up to 2.7x slow** (agent B)
   even after a burn-in; the harness survives it (2 warmups per arm, min over
   rotated rounds), a one-shot probe does not.
3. **Interleaving with the SDPA arm costs us ~9% at 256:16:128 and vLLM nothing**
   (agent B's `bench_pressure.py`, 8581C): the scorecard's own environment
   charges us for cold K/V because our pack writes 2 MB per call. That is a fair
   reading of "the whole call", not an artefact.
4. **The x86 paged b8/256 row is 1.7x min-to-median** on m8i, and the paged
   fp32 rows lose (1.04-1.20) while bf16 wins; do not quote one number for that
   shape without the pair.
