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
| S=128 H=8 D=64 | 1.13 | **0.95** |
| S=512 H=8 D=64 | **0.99** | **0.89** |
| S=1024 H=8 D=64 | **0.96** | **0.90** |
| S=2048 H=8 D=64 | **0.99** | **0.94** |
| S=4096 H=8 D=64 | **0.98** | **0.99** |
| S=256 H=16 D=128 | 1.04 | **0.80** |
| S=1024 H=16 D=128 | 1.02 | **0.84** |
| S=2048 H=16 D=128 | 1.05 | **0.88** |
| S=4096 H=16 D=128 | 1.06 | **0.99** |
| S=512 H=32 D=128 | 1.03 | **0.84** |
| S=1024 H=8 D=32 | **0.97** | **0.91** |

(`min o/vllm`, which is what `main` now judges on.) **16/22**: all 11 fp32 rows
(0.80-0.99) and 5 of the 11 bf16 ones. The four bf16 D=128 rows move 1.02-1.06
between runs -- they are the wall described below, worth calling parity-or-1.05
rather than a number. Only `128:8:64` bf16 (1.13) is reproducibly outside the
noise band, and it is per-call cost, not kernel rate.

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
