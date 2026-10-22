# ARM dense kernel, second campaign: what was measured, what died

Branch `arm-v2`, worktree `arm-dense-v2`, rebased onto `main` at `507b41d`.
Box: m9g.2xlarge (Graviton 5 / Neoverse-V3, 8 cores, 3.28 GHz, L1d 64 KiB 4-way,
L2 2 MiB/core, L3 48 MiB). Shared with other agents: every number below was
taken under `~/bench.lock` after waiting for a 1-minute load average < 0.8
(`scripts/box_idle.sh`, added here), and the load is printed on both sides of
each run. A `mojo build` running beside a sweep turns a 0.97 row into a 1.7 one
-- measured, twice.

## Where the rows stand

`tests/bench_vs_refs.py --vllm ... --isa neon --rounds 9`, medians of 9,
interleaved in one process (ratios inside a run are trustworthy; absolute times
are not). `o/vllm` < 1 = we win.

| shape | bf16 | fp32 |
|---|---:|---:|
| S=128 H=8 D=64 | 1.13 | **0.94** |
| S=512 H=8 D=64 | **0.94** | **0.89** |
| S=1024 H=8 D=64 | **0.96** | **0.93** |
| S=2048 H=8 D=64 | **0.98** | **0.93** |
| S=4096 H=8 D=64 | **0.97** | **0.98** |
| S=256 H=16 D=128 | **0.99** | **0.80** |
| S=1024 H=16 D=128 | 1.01 | **0.85** |
| S=2048 H=16 D=128 | 1.04 | **0.87** |
| S=4096 H=16 D=128 | 1.04 | **0.98** |
| S=512 H=32 D=128 | 1.03 | **0.84** |
| S=1024 H=8 D=32 | **0.96** | **0.90** |

(`min o/vllm` from the same run, which is what `main` now judges on: every row is
within 1% of the median-based ratio above.)

**17/22** -- all 11 fp32 rows and 6 of the 11 bf16 ones. Four of the five
remaining bf16 rows are inside 1.01-1.04, i.e. inside this box's noise band; only
`128:8:64` (1.13) is reproducibly above 1.05.

## What this branch actually changed (and what it is worth)

Measured with `tests/ab_two_trees.py` (one short process per tree, alternating
rounds), current tree vs `main` at `2cb0e82`, 8 threads, 16 shapes x 2 dtypes:
**1.00x on thirteen of them, 1.01-1.02x on the rest.** In other words the kernel
work in this branch is worth about a percent on top of `main`; the table below
is what each piece is, and the only one with a large number attached is the
workspace bound, which that harness cannot see (it runs one shape per process).

| change | worth |
|---|---|
| K^T packed as `[tile][key pair][dim group][8]` instead of `[tile][dim group][key pair][8]` | **1.01x**, and it cuts the L1 refill rate from 3.5x vLLM's to **1.7x** (0.878 -> 0.764 refills/row-key against vLLM's 0.25 at 1024:16:128) |
| 128-key tiles at head_dim 128 (was 256) | 1.00-1.01x |
| packed-K/V workspace bounded to 48 MB (was one buffer per geometry, unbounded to 64) | **5-15% on the rows a long sweep measures late** -- see below |
| qt=12/16 row blocks | swept and **not selected** (see falsifications) |

The workspace bound is the one with a visible payoff, and it is a memory-hygiene
bug, not a kernel one: `_scratch` cached one K^T/V buffer per (kind, shape) and
the plans held references, so a 22-row `bench_vs_refs` sweep kept every
geometry's scratch alive -- several hundred MB by the last row. The fingerprint
is positional, not shape-dependent: 512:32:128 fp32 measured 4579-5389 us as row
20 of the sweep and 4094 us when run on its own; 2048:16:128 fp32 32505-33504 vs
31807; 1024:16:128 fp32 8953-9276 vs 8527. With the byte bound the same rows
land at their isolated values within 3% every run. It is a byte budget rather
than a count so that a workload with many *small* geometries (a server with
many prompt lengths) keeps all of its workspace and only a second *large* one
evicts the first -- a count of 8 would thrash that workload. An engine that only ever sees one
geometry is unaffected, which is why this was invisible until the scorecard was
run end to end.

## The wall, with numbers

Per core the kernel is now **at vLLM's efficiency**: 1 thread, cycles per
row-key (`tests/ab_two_trees.py --threads 1` and `prof_one.py`),

| shape | ours | vLLM | note |
|---|---:|---:|---|
| 1024:16:128 bf16 | 8.46 | 8.28 | bfmmla ceiling is 8.0 (16 `bfmmla`/row-key at 2/cycle) |
| 2048:16:128 bf16 | 8.17 | 8.21 | 99% of ceiling |
| 4096:16:128 bf16 | 8.13 | 8.09 | 99% of ceiling |
| 4096:8:64 bf16 | 4.94 | 5.14 | we win |

So there is no per-core headroom left at D=128: 16 `bfmmla` per row-key at the
measured 2.0/cycle is 8.0 cycles, and both kernels are at 99% of it. At 8
threads we are 1.03-1.05 on those rows against vLLM's 1.00, and the counters say
the difference is instruction efficiency, not traffic:

1024:16:128 bf16, 8 threads, `perf stat` (differencing two iteration counts,
counters at 74-75% multiplexing so only the ratios mean anything):

| | ours | vLLM |
|---|---:|---:|
| instructions / row-key | 32.0 | 32.8 |
| IPC | 4.06 | 4.21 |
| L1 refills / row-key | 0.39 | 0.23 |
| L2 refills / row-key | 0.047 | 0.037 |
| stalled cycles | 18.3% | 12.6% |

and the phase budget that reproduces 9.0 cycles/row-key: Q@K^T 4 (matmul-bound,
24 instructions per 16 `bfmmla` = IPC 3), P@V 4 (same), exp pass ~1 (4
instructions/row-key, and it is a ~10-deep dependency chain so it is *latency*,
not issue, bound). The three phases are serial, so the exp cannot overlap the
matmul, and that one cycle is the whole gap. Making it overlap needs the exp's
working set (2 logit loads + 2 polynomials + 4 sums) live beside P@V's 24
registers, i.e. 34-36 against a 32-register file; five previous attempts to get
there lost 8-11% each (`docs/CEILINGS.md`).

## Falsified this session, with the number

**Mixed `bfmmla` + fp32 `fma` contraction** (split the head dims across both
pipes, which would be up to 1.38x if they were independent). Measured with a new
probe, `cpu_attention/_mojo/tests/bfmmla_fma_mix.mojo`, on the box:

```
8 bfmmla/iter      0.1538 ns/instr   2.00/cycle
8 fma/iter         0.3029 ns/instr   0.98/cycle   (fmul+fadd, not contracted)
8 bfmmla + 8 fma   0.3661 ns per pair  (shared-pipe predicts 3.65 ns,
                                        independent predicts 2.42 ns)
```

so they *partially* overlap but the arithmetic kills the idea: the machine
issues ~4 instructions/cycle and a `bfmmla` already delivers 16 MACs per issue
slot against an fma's 4, so a split raises the instruction count above the
32/row-key we are already at and the issue floor (32/4 = 8 cycles) rises faster
than the pipe relief. The kernel is issue-limited, not pipe-limited. Do not
spend a day here.

**A tile-level max pass instead of the per-sub-block tree** (move ~28
instructions of serial `fmaxnm`/`rev64` out of the matmul-saturated Q@K^T loop
into the issue-bound exp pass). 1.6% *slower*: 1024:16:128 bf16 o/vllm
1.125 -> 1.143, 4096:16:128 1.090 -> 1.107, 4096:8:64 1.025 -> 1.053. The Q@K^T
epilogue is not draining the `bfmmla` pipeline.

**16 query rows per work item** (`qt=16` with `nsq=2`/`dcp=2` to fit the
register file, and `ks=128`). Halves the compulsory K/V traffic per row-key.
Neutral-to-2% at long S, and after the per-head dispatch (`ca0a85e`) it is
**9-12% slower at every shape**, including the short ones it was meant to fix:

```
shape            qt6   qt8   qt12  qt16     (us, 8 threads, medians of 11)
128:8:64          25    23    26    24
256:16:128       231   207   229   214
512:8:64         249   226   239   235
512:32:128      1682  1524  1615  1561
1024:16:128     3145  2876  3014  2944
```

The code for it stays (`CPU_ATTENTION_QT=12|16`, parity clean) because it is the
cheapest way to re-run that sweep, but nothing selects it.

**128-key vs 256-key tiles at head_dim 128 after the layout change**: 1.00-1.01x
either way, where before the layout change 128 keys was 9% better at 256:16:128.
The layout change removed the reason the tile size mattered.

## The one row outside the noise band: 128:8:64 bf16

`bench_vs_refs` calls it 1.10-1.13 (24-25 us against vLLM's 22), but in a tight
loop with no interleaved variant in between the same call measures **22.8 us**
against vLLM's 22.0 -- i.e. the row is borderline, and what is left of it is
fixed cost, not kernel rate. The H-scaling fit at S=128/D=64 (`fixed_cost.py`,
8 threads) separates the two terms:

| heads | ours us | vLLM us | ratio |
|---|---:|---:|---:|
| 4 | 16.9 | 13.1 | 1.28 |
| 8 | 24.1 | 22.0 | 1.09 |
| 16 | 42.0 | 40.0 | 1.05 |
| 32 | 79.0 | 78.8 | 1.00 |

so our *marginal* cost per head is better than theirs (2.29 vs 2.37 us) and the
whole difference is the intercept: **5.8 us of per-call cost against their 3.0**.
Of our 5.8, about 4.1 is the Python path (8 tensor attribute reads in
`flash_attn_func`, the plan key + dict get, the 42-tuple unpack, `torch.empty`,
four `data_ptr`, and 0.88 us of marshalling 42 arguments into the extension) and
~1.7 is the OpenMP region. Every item was measured separately; the largest is
the output allocation at 0.87 us, which cannot go. Cutting the extension's
argument list from 42 to ~6 (hand the geometry to the kernel once and keep the
struct's address on the Python side) is the only remaining route, it is worth
~1.5 us, and it would move this row to ~1.04.

At one thread the same row is at parity (7.09 vs 7.14 cycles/row-key), so this is
not a kernel problem: at S=128 both kernels are at 57% of the 4-cycle `bfmmla`
floor at D=64 and the per-call costs are what is left.

---

# Addendum: the floor of the D=128 rows, and the 384-key tile (arm-tile)

Written by the bf16 agent after the merge of `arm-v2` + `d13dafe` (main
`078c79d`), answering "how much is left on the four remaining bf16 rows". All
numbers measured on the same box under `scripts/box_run.sh` after the 1-minute
load average dropped below 0.8 (`scripts/quiet.sh`, added here); the load is
printed on both sides of every run.

## 1. The floor, measured rather than argued

Three measurements, then arithmetic that closes to within 1%.

**(a) `bfmmla` throughput** (`cpu_attention/_mojo/tests/bfmmla_peak.mojo`,
extended here: 8 copies pinned to the 8 cores at once, so it is the all-core
sustained rate, not a single-core burst):

| loop shape | ns/instr | MAC/ns/core | implied |
|---|---:|---:|---|
| 16 accumulators, **no loads** | 0.0769 | **208** | 4 instr/cycle |
| 16 accumulators, 8 loads/iter (the kernel's Q@K^T body) | 0.1527 | **105** | 2 instr/cycle |
| 16 accumulators, 8 loads from a rotating address | 0.1521 | **105** | same |

So the MMA itself is not the limit -- the *operand loads* halve it, and 105
MAC/ns/core = 840 MAC/ns over 8 cores is the ceiling for the loop shape both we
and vLLM write. (The kernel actually runs slightly *above* this, because it
emits `ldp` pairs: 4 load instructions per 16 `bfmmla`, where the probe issues
8.)

**(b) The phase split**, `tests/bench_ablate.py` (a runtime phase-ablation
selector added for this, on the `arm-floor` branch only -- the guards cost 2-3%
of a D=128 call so nothing below is judged on that build), `qt=8`, two runs,
medians agreeing within 1%, `oh_only` = pack + work-item prologue/epilogue:

| shape | full | Q@K^T | exp | P@V | pack + prologue |
|---|---:|---:|---:|---:|---:|
| 512:32:128 | 1485 µs | 538 (36%) | 220 (15%) | 556 (37%) | 177 (12%) |
| 1024:16:128 | 2800 | 1064 (38%) | 425 (15%) | 1093 (39%) | 214 (8%) |
| 2048:16:128 | 10981 | 4234 (39%) | 1749 (16%) | 5086 (46%) | 734 (6.7%) |
| 4096:16:128 | 44224 | 16887 (38%) | 6776 (15%) | 18617 (42%) | 2086 (4.7%) |

**(c) The pack alone**: 180 GB/s, so 4·S·H·D·2 bytes of K/V traffic costs
93 µs at 512:32:128 (6.3% of the call), 190 µs at 1024:16:128 (3.7%), 370 µs at
2048:16:128 (1.8%) and 745 µs at 4096:16:128 (0.8%). The rest of `oh_only` is
the per-work-item cost, ~690 cycles at 4096:16:128 over 8192 items.

**The arithmetic, at 4096:16:128** (256 MACs/row-key, 134.2 M row-keys = 34.36 G
MACs, floor 40.9 ms): matmul 92.5% + exp *not* hidden 2.9% + pack 0.8% +
per-item 3.9% = **100.1%**. At 2048:16:128: 93.2 + 1.8 + 4.9 + ~0 = 99.9%. At
1024:16:128: 91.3 + 3.7 + 4.3 + ~1 = 100.3%. At 512:32:128: 86.3 + 6.3 + 5.7 +
1.7 = 100.0%.

So on long-S D=128 **there is ≤1% not accounted for**, and vLLM's own 1579
GFLOP/s against the same 1680 GFLOP/s ceiling (94%) is the same story from the
other side: they carry no pack and a smaller per-item cost, we carry 0.8% pack
and ~3% per-item, and that difference *is* the 1-2% the scorecard shows. At
512:32:128 the matmul share drops to 86% because the per-tile work per row-key
is 2.5x higher and the pack is 6.3% -- that row is fixed-cost-bound, not
matmul-bound.

## 2. The 384-key tile (shipped here, +1-2%)

`TILE_KS` was 128 with a comment saying 256 lost because "the two logit buffers
start competing with the K/V streams for L1". That measurement predates the K^T
scratch becoming `[tile][key pair][dim group][8]`; with one Q@K^T sub-block now
a linear run, the tile no longer has to be L1-resident to stream. Re-measured
with a temporary in-process `ks` selector (9 rounds, two runs, min of the two,
parity against SDPA checked per variant, all ok):

| shape | ks128 | ks192 | ks256 | ks384 | ks512 |
|---|---:|---:|---:|---:|---:|
| 512:8:64 | 221 | 216 | 216 | **215** | 221 |
| 2048:8:64 | 3316 | 3264 | 3239 | **3220** | 3301 |
| 4096:8:64 | 13076 | 12672 | 12856 | **12563** | 13021 |
| 512:32:128 | 1498 | 1471 | 1463 | 1464 | **1459** |
| 1024:16:128 | 2823 | 2772 | 2770 | 2764 | **2750** |
| 2048:16:128 | 11135 | 11027 | 10984 | **10903** | 10966 |
| 4096:16:128 | 44776 | 43973 | 43953 | **43744** | 43686 |

384 is the best or within 0.6% of it on six of seven shapes and 2.1-4.1% below
128 on all of them; the mechanism is amortising the per-tile row max/sum/
rescale, the P@V accumulator round trip and the tile prologue over 3x more keys.
On a **clean build** (no ablation guards) against main, `ab_two_trees`, 7
rounds, two runs: 512:32:128 1.01x, 1024:16:128 1.01x, 2048:16:128 1.01x,
4096:16:128 1.01x, 2048:8:64 1.01-1.02x, 4096:8:64 1.01-1.02x. Parity:
`test_dense` 64/64, `test_flash_api` green, `fuzz_dense` 200/200 on the box
(the odd `seq_k` / partial-last-tile cases are the ones this kernel has broken
on before, so they are the gate).

## 3. Under 1% each, left alone

* **Q-panel build**: RP·DG = 128 iterations of two 8-byte loads + a join + a
  16-byte store (~640 instructions per work item, ~0.3-0.6% of a call). Halving
  it with 16-byte loads + `zip1/zip2` is worth <0.5%.
* **Skipping the P@V accumulator init and the first tile's reload** (vLLM's
  `accum_c`): worth ~1000 memory ops per item, but the `if t0 > 0` branch turns
  `accv` into a phi and LLVM spills all 16 accumulators -- measured **26%
  slower**, reverted. Do not retry without restructuring the loop.
* **Contiguous K^T layout** (the pre-arm-v2 "k_seq" probe): 2-6% when probed,
  but arm-v2 already took it via the layout change; the L1-resident probes
  (`k_hot`/`v_hot`) put *all* K/V memory cost at 2-3%, so there is nothing left
  in the access pattern.
