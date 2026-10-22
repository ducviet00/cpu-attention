# Where the campaign stands

`ours / baseline` (below 1 wins) on every row of `tests/bench_vs_refs.py`,
min-based, interleaved in one process, box idle. Two boxes: an Emerald Rapids
Xeon 8581C (4 physical cores / 8 threads, AMX, GCP) and a Graviton5 m9g
(8 cores, AWS). On an AMX box the honest bf16 bar is **vLLM's AMX build**; fp32
has no AMX kernel in vLLM, so its **VEC build** is that column's bar.

## The scoreboard

| | x86 bf16 vs vLLM AMX | x86 fp32 vs VEC | ARM bf16 | ARM fp32 |
|---|---|---|---|---|
| rows below 1 | **9/11** | **10/11** | 8/11 | 11/11 |
| worst row | 1.05 (`256:16:128`) | 1.04 (`128:8:64`) | 1.03 (`256:16:128`) | 0.99 |

**Every row beats torch SDPA on both boxes**: x86 by 1.6-3.7x (`o/sdpa`
0.27-0.63), ARM by 1.1-3.4x (0.29-0.88). Full x86 table in the commit message
of `c29fc5a`; the four rows still at or just above 1 are 0.99-1.05, i.e. inside
the 3% band this box moves between runs (a duplicate arm reads 0.99-1.01).

Where they came from, same box, same harness (x86 bf16, `ours/vllm`):

| row | start of session | now |
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
   a separate attention phase. Worth **1.47x at 128:8:64**, 1.29x at 512:8:64,
   1.16x at 256:16:128, 1.03-1.07x on the long rows (`mode & 512` selects the old
   order for an in-process A/B). Two effects: the head's packed K/V is 32 KB at
   D=64/128 keys and stays in L1 for its own query blocks, and a whole head is a
   fixed amount of work, so the causal imbalance goes away.
2. **A 5-argument call entry** (`dense_amx_plan`): the geometry (strides, tile
   counts, workspace addresses, kt, workers) rides in an int64 buffer instead of
   36 Python arguments, and the per-call rebuild is cached per thread.
   `pub` 43.1 -> 37.0 us at 128:8:64, 253 -> 237 at 256:16:128. The row where we
   lose is the row where the *kernel* is already faster than vLLM's (21 vs 25 us
   by ablation), so the call path was the whole margin.
3. **comptime `head_dim`**: every tile offset and loop bound folds to a constant
   (1.00-1.03x on its own; it is what makes the rest legible).
4. Merged from the agents: the ARM short-sequence kernel's cheaper exponential
   and `_SMALL_MAXK` 256 -> 1024 (ARM 18/22 -> 21/22), and a second AMX kernel
   (`dense_amx_v2.mojo`, `CPU_ATTENTION_AMX2=1`) kept opt-in because v1 with the
   per-head item is 1.46x faster than it at 128:8:64.

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
| the pack | 3-8% of a call, 40-83 GB/s at 4 threads, latency-bound not bandwidth-bound |

So the kernel runs at 2.0-2.8 TFLOP/s against a 3.0 TFLOP/s dot roof; oneDNN's
bf16 GEMM on this box is 1.11 TFLOP/s/core and vLLM's attention 0.66-0.74.

## Falsifications (the number that killed each)

| idea | result |
|---|---|
| pipeline the softmax under the Q@K^T dots (chunk-lagging max) | probe 1.65x, **kernel 0.73-0.91x**. Agent A reproduced the sign independently (1217 vs 1185 us). The overlap is real; the per-chunk horizontal reduce costs more |
| cheaper exponential: degree-3 for bf16, `round()`, one-sided clamp, `llvm.ldexp`/`vscalefps` | 3 instructions fewer per 16 logits, **10-27% slower** end to end while a single-threaded probe of the same loop read 8% faster |
| interleave the K and V packs for memory-level parallelism | 1.00-1.02x |
| a running V pointer + hoisted P pointers in P@V | 1.00x (LLVM already strength-reduces it) |
| kt 128 / 384 / 512 vs 256 | 0.99 / 0.97 / 0.96 and flat -- 256 is the corner |
| 8 SMT workers | 27620 vs 24314 us at 4096:16:128 |
| v1's `mode & 64` P@V ablation bit | measures nothing (`nk // 32 == (nk + 31) // 32` for a full super-tile); P@V is 7-14%, not 1% |
| v2 (second kernel) as the default | 1.46x slower than v1 at 128:8:64 once v1 has the per-head item |

## ARM: the bfmmla ceiling is not settled

Agent B's three probes disagree by 2.4x: 128-134 MAC/ns/core for a 16-accumulator
no-load loop, 168 for a variant at 3.3-3.5 `bfmmla`/cycle, and 61 for the same
loop with 4 `ldp` feeding the next instruction pair. So "the matmul is at the
wall" (94% of a measured ceiling) is **not established in either direction** --
the number is a property of the loop shape, not of the core. What does hold is
the per-row verdicts and the 68-85% range for the matmul phases depending on
which probe you believe. `docs/CEILINGS.md` and `docs/FINDINGS.md` carry all
three probes; the honest next step is to instrument the kernel's own Q@K^T body
rather than another microbenchmark.

## Beyond this box's resolution

1. **The box drifts 1.4-3.5% between runs and 2x on a bad day.** `bench_vs_refs`
   with `--rounds 13-21` and the min is the only stable reading; a 5-round median
   moved a row from 1.06 to 1.33 for the same binary.
2. **The first timed variant in a fresh process reads up to 2.7x slow** (agent B)
   even after a burn-in; the harness survives it (2 warmups per arm, min over
   rotated rounds), a one-shot probe does not.
3. **Interleaving with the SDPA arm costs us ~9% at 256:16:128 and vLLM nothing**
   (agent B's `bench_pressure.py`): the scorecard's own environment charges us
   for cold K/V because our pack writes 2 MB per call. That is a fair reading of
   "the whole call", not an artefact, and it is the largest remaining structural
   difference -- vLLM stages its cache before the timer.
4. The four rows at 0.99-1.05 are at parity inside (1) and (3). A verdict on
   them needs a quieter box, not more code.
