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
| S=128 H=8 D=64 | 1.16 | **0.94** |
| S=512 H=8 D=64 | **0.97** | **0.91** |
| S=1024 H=8 D=64 | **0.96** | **0.93** |
| S=2048 H=8 D=64 | **0.99** | **0.97** |
| S=4096 H=8 D=64 | **0.98** | **0.96** |
| S=256 H=16 D=128 | 1.05 | **0.84** |
| S=1024 H=16 D=128 | 1.03 | **0.90** |
| S=2048 H=16 D=128 | 1.05 | **0.88** |
| S=4096 H=16 D=128 | 1.04 | **0.98** |
| S=512 H=32 D=128 | 1.03 | **0.84** |
| S=1024 H=8 D=32 | **0.96** | **0.91** |

**16/22**, all 11 fp32 rows and 5 of the 11 bf16 ones; five of the six
remaining bf16 rows are within 5%, one (128:8:64) is 16%.

## What this branch actually changed (and what it is worth)

Measured with `tests/ab_two_trees.py` (one short process per tree, alternating
rounds), current tree vs `main`:

| change | worth |
|---|---|
| K^T packed as `[tile][key pair][dim group][8]` instead of `[tile][dim group][key pair][8]` | **1.01x**, and it cuts the L1 refill rate from 3.5x vLLM's to **1.7x** (0.878 -> 0.764 refills/row-key against vLLM's 0.25 at 1024:16:128) |
| 128-key tiles at head_dim 128 (was 256) | 1.00-1.01x |
| packed-K/V workspace bounded to 8 geometries (was unbounded to 64) | **5-15% on the rows a long sweep measures late** -- see below |
| qt=12/16 row blocks | swept and **not selected** (see falsifications) |

The workspace bound is the one with a visible payoff, and it is a memory-hygiene
bug, not a kernel one: `_scratch` cached one K^T/V buffer per (kind, shape) and
the plans held references, so a 22-row `bench_vs_refs` sweep kept every
geometry's scratch alive -- several hundred MB by the last row. The fingerprint
is positional, not shape-dependent: 512:32:128 fp32 measured 4579-5389 us as row
20 of the sweep and 4094 us when run on its own; 2048:16:128 fp32 32505-33504 vs
31807; 1024:16:128 fp32 8953-9276 vs 8527. With the bound at 8 the same rows
land at their isolated values every run. An engine that only ever sees one
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

## The one row that is still 16%: 128:8:64 bf16

26 us against vLLM's 22-23. Split measured in-process at 8 threads:

| term | us |
|---|---:|
| kernel (the raw `dense_forward` call) | 21.0 |
| Python: 8 tensor attribute reads, plan key + dict get, 42-tuple unpack, `torch.empty`, 4 `data_ptr`, 42-argument marshal | 4.1 |
| vLLM's whole call | 22.0 |

and at *one* thread we are at parity (7.09 vs 7.14 cycles/row-key), so the row
is fixed cost, not kernel rate: the kernel is 0.9 us of it and the Python path
2.5. `_dense_plan` already caches everything that is a pure function of the
geometry; what is left is ~1.1 us of `.shape`/`.stride()`/`.dtype` reads in
`flash_attn_func`, 0.87 us of output allocation and 0.88 us of marshalling 42
arguments into the extension. Cutting the argument count to ~6 (hand the
geometry to the kernel once, keep the struct's address on the Python side) is
the only remaining route and it is worth ~2 us on this row and ~0 on every other.
