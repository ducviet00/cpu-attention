# FINDINGS — independently measured, 2026-10-22 (integrator)

> **Historical (third campaign, `03-3a`).** The merged-tree numbers are in
> `docs/STATUS.md` and the independent re-measurement in `docs/AUDIT.md`; the
> harness rules and phase anatomy here are still useful.


Everything here was measured today on the shared m9g.2xlarge (Neoverse-V3, 8
cores, 3.0-3.3 GHz) by the campaign integrator, not inherited. Commands are
given so any of it can be re-run. If a number here disagrees with your
measurement, your measurement wins — but say so and re-measure both ways.

## 1. The core's peaks (hand-written asm, `perf stat`, no compiler noise)

| unit | rate | MAC/ns/core |
|---|---|---|
| `bfmmla` (bf16 2x2x4) | 2.0 /cycle | **105** |
| `fmla` f32 (4 lanes) | 3.3-4.2 /cycle | **39-52** |
| `smmla` i8 (2x2x8) | ~3 /cycle | **~300** |
| `bfdot` (4 MACs) | 3.3 /cycle | 39 |
| 4 `bfmmla` + 4 `fmla` interleaved | same 8-instruction slot | no port win |

Consequences: bf16 matmul is ~2-2.7x fp32 per core (not 8x); `bfmmla` is the
only fast MAC unit; **fusing fmla into the bfmmla loop does not add throughput**.
int8 `smmla` is 3x bfmmla but the accuracy gate is fixed at bf16 3e-2 / fp32
2e-4 and int8 Q@K^T was already measured at 2.0-3.5e-2 by the previous campaign
(reproduce `tests/int8_qk_error.py` before re-opening it).

So the bf16 matmul floor at head_dim 64 is 8 `bfmmla` per row-key = 4 cycles /
row-key at 2/cycle, and that is *the whole headroom argument*: at 4096:8:64 we
measure 5.4 cycles/row-key, i.e. 74% of the floor. A perfect kernel would be
o/vllm 0.74; there is no 2x lying around, but 10-20% is real.

## 2. Our kernel vs vLLM, single thread, `perf stat` (NT=1, taskset -c 3)

| shape | variant | time | instr (PMU) | cycles (PMU) | IPC | L1 refills |
|---|---|---|---|---|---|---|
| 2048:8:64 bf16 | ours | 27.20 ms | 12.374 G | 3.995 G | 3.10 | 183.7 M |
| 2048:8:64 bf16 | vllm | 27.77 ms | 12.733 G | 3.975 G | 3.20 | 101.5 M |
| 1024:16:128 bf16 | ours | 23.08 ms | 11.414 G | 3.800 G | 3.00 | 220.2 M |
| 1024:16:128 bf16 | vllm | 22.23 ms | 11.861 G | 3.602 G | 3.29 | 88.1 M |

(PMU cycle/instr counts on this box are inflated ~2.2x in absolute terms —
4 GHz over a 27 ms call is impossible — but the *ratios* are consistent with
wall time, so IPC and instructions-per-row-key are usable.)

Reading: at long S we and vLLM run the **same instruction count** (we are 3-4%
lower) at the **same IPC** (3.0-3.1 vs 3.2-3.3). The long-S gap is 3-6% and it is
IPC, not instructions. The one counter with a big spread is **L1 refill: we take
1.8-2.5x vLLM's**. Whoever chases the long-S rows should explain that with a
layout/stride experiment, not with an algorithm change.

## 3. Where a short-S call goes (128:8:64 bf16, 8 threads)

`tests/bench_phases.py` (fused path in the scorecard: 34 us total):

| phase | us | share |
|---|---|---|
| K/V pack | 6 | 18% |
| row kernel | 28 | 82% |

The row kernel runs at **302 GFLOP/s**, against 1236 GFLOP/s at 2048:8:64 —
a 4x efficiency collapse. `perf record` on the same call:

| symbol | share |
|---|---|
| `dense::_dense_task_pair[bf16,64,qt=8]` | 33% |
| `gomp_team_barrier_wait_end` (spin) | 20% |
| `gomp_barrier_wait_end` (spin) | 9% |

**~29% of a 128:8:64 call is threads spinning at an OpenMP barrier.**

Thread scaling of the whole call (medians, `--threads t`):

| | 1t | 2t | 4t | 8t | 8t/1t |
|---|---|---|---|---|---|
| ours 128:8:64 | 146 us | 89 | 50 | 34 | **4.3x** |
| vllm 128:8:64 | 146 us | 75 | 40 | 23 | **6.3x** |

At *one thread* we are exactly at parity with vLLM (146 vs 146 us). The whole
1.53x gap on that row is parallel efficiency, not per-core efficiency. Same
pattern at 512:8:64 (ours 6.6x, vllm 7.7x) and it closes by 2048 (7.1x vs 7.7x).

Hypothesis worth testing (not yet measured): at S<=512 one head's whole packed
K/V (32 KB at 128:8:64) fits in L1, but the dispatcher hands out (head, q-pair)
tasks so every core re-reads the same K/V. Making **one thread own a whole head
(all its row blocks)** would cut that traffic ~8x and is balanced by
construction when `H % threads == 0` (H=8,16,32 against 8 threads). The current
`_dense_task_pair` pairing is the right thing at long S; short S wants
head-major affinity.

## 4. Commands

```bash
scripts/box_run.sh ubuntu@52.77.215.13 'cd ~/cpu-attention && .venv/bin/python tests/bench_phases.py --vsllm ... --shapes 128:8:64 --rounds 9'
scripts/box_run.sh ubuntu@52.77.215.13 'NT=1 perf stat -x, -e cycles,instructions,inst_retired,l1d_cache_refill taskset -c 3 ~/cpu-attention/.venv/bin/python ~/prof_one.py 2048 8 64 bf16 ours 20'
scripts/box_run.sh ubuntu@52.77.215.13 'perf record -q -g -o /tmp/p.data -- .venv/bin/python /tmp/prof128.py; perf report -i /tmp/p.data --sort symbol'
```

## 5. The harness rules that came out of this session

1. **Judge on the minimum, not the median.** On a shared box a burst of foreign
   load lands inside one variant's sample and moves a median 15%: bf16
   2048:16:128 read median 1.15 / min 1.04 in the *same* matrix run, and fp32
   512:32:128 read 0.85 twice then 1.01 twice while an in-process probe of the
   same binary read 4046 us both times. `tests/bench_vs_refs.py` now prints
   `min o/vllm` beside the median, and `scripts/band.py` / `scripts/scorecard.py`
   judge on it (interference can only make a sample slower).
2. **Wait for the box.** `scripts/verify.sh` blocks until `/proc/loadavg` < 0.8
   before timing. Every build, test and timing run takes `~/bench.lock`
   (`scripts/box_run.sh`, `scripts/box_sync.sh`, `scripts/verify.sh`).
3. **Toggle plan-level knobs in one process.** A knob that invalidates the call
   plan inside a timing loop costs up to 19% by itself — that artefact produced
   one false "0.22x regression" and one false "12% instability" this session.
   `tests/ab_two_trees.py` (two trees, alternating processes) is for code that
   cannot be toggled at runtime; everything else should be interleaved in one
   process.
4. **Rebuild clean for a verdict.** `verify.sh` deletes `_mojo/build` and
   `_C.so` first, so a stale `.so` cannot pass for a fresh kernel.

## 6. Call-path anatomy at 128:8:64 bf16 (25 us total)

Measured by wrapping the extension entry with a no-op:

| item | us |
|---|---|
| `flash_attn_func` total | 4.47 (now **2.6** with the seven-argument entry, `10c18a7`) |
| Python side without the extension call | 3.25 |
| of which the extension call itself (42 args) | 1.22 |
| `torch.empty` (the output) | 0.56 |
| geometry + plan key + dict lookup | 0.5 |
| the 40 argument checks | ~0.3 |

> **Re-measured on a fresh instance (2026-10-23, `REMEASURE.md` §5).** The
> column above is from the previous box and two of its entries do not transfer.
> Same method, 400-800 rounds, box idle: `flash_attn_func` **21.4 us** (kernel
> included), the Python path with the seven-argument entry no-op'd **3.40 us**
> (min 3.25), with the 42-argument entry no-op'd **3.80 us** — so the plan
> buffer is worth **0.40 us**, not 1.87 — `torch.empty(q.shape)` **1.36 us**
> (not 0.56), and the plan key + `dict.get` **0.027 us**. The Python path is
> therefore 15.9% of the call, not 11%, and the output allocation is its largest
> single item. The direction of the doc's conclusion (the marshal is not the
> lever; leaving Python is) stands; the sizes do not.

vLLM reaches its kernel through pybind for ~1.5 us, so ~3 us of a 25 us call is
our Python. Two attempts: caching the validation saves 0.26 us (not worth a
cache), and a plan-buffer ABI (5 args instead of 42) saves ~1 us but needs 49
hand-mapped fields — its two bugs were silent (raw instead of packed V strides)
until they segfaulted, and it was abandoned. The route to that row is to move the
call path out of Python, not to tune it.

### 6a. Update: the marshal was collected, the rest was not

The seven-argument entry (geometry in an int64 plan buffer the call plan owns)
took the call path from 4.47 us to ~2.6 us, worth 5% of a 24 us call at
128:8:64. What is left is the output allocation (0.56 us), six `.shape`/
`.stride()` reads (0.9 us, unavoidable -- they are the cache key) and the
`torch` call overhead. Caching the *validation* was worth 0.26 us and was not
kept. The remaining route would be to leave Python entirely, which is an engine
change, not a kernel change.

**Corrected 2026-10-23 (fresh instance, same method):** the seven-argument
entry is worth **0.40 us**, not 1.87, and the call path is **3.40 us of a
21.4 us call**, not 2.6 of 24. The output allocation measures **1.36 us** on
this box (0.56 on the old one) and is the largest item; the plan key and the
dict lookup are 0.027 us together. See `docs/REMEASURE.md` §5 — quote the new
numbers, not this paragraph's.

## 7. Where the *rest* of the 128:8:64 budget goes

25 us = kernel 19.6 + pack 2.4 + Python 4.5 (overlapping a little). vLLM's 22 =
attention 21 + ~1.5 Python, with its cache staged **outside** the timer. So on
that row we win the kernel by 6% and lose 2.4 us of pack (charged to us by the
scorecard's contract) and ~3 us of Python.

## 8. Re-measured on a fresh instance, 2026-10-23 (`docs/REMEASURE.md`)

Same box type (m9g.2xlarge, Neoverse-V3), new physical host, `main` @ `973cce9`.
The §1 peaks are the part of this file that does **not** transfer, and they are
the part the campaign's ceiling argument used:

| quantity | this file §1 | re-measured |
|---|---:|---:|
| `bfmmla`, no loads, 16 accumulators | 208 MAC/ns/core (0.0769 ns/instr) | **128-134 MAC/ns** (0.119-0.126 ns/instr) |
| `bfmmla`, 8 loads per 16 | 105 MAC/ns/core (0.1527 ns/instr) | **114-115** (0.139-0.141) |
| `bfmmla`, 4 `ldp` per 16 (the kernel's shape) | not measured | **144-145** (0.110-0.111) |
| `bfmmla`, 8 accumulators (the shipped probes) | 105 | **105** — reproduces exactly (`peak.c` 209-210 GFLOP/s/core) |
| `fmla` f32 | 39-52 MAC/ns | not re-measured |
| `bfdot` | 39 | 140 GFLOP/s/core = 35 MAC/ns (`peak.c`, reproduces) |

**Two independent probes of the no-load ceiling disagree by 25%** (128-134 above,
168 in the integrator's `peak5.s`: 16 accumulators, 2 operands, 16 `bfmmla` per
iteration, 4.55 cycles per 16 at a 3.1 GHz clock = 3.3-3.5 `bfmmla`/cycle), and
the *with-loads* figure is worse: the same 16-accumulator loop with 4 `ldp` per
16 measures **61 MAC/ns** when the loads feed the next instruction pair and
114-145 when they do not. So a microbenchmark does not settle this: the number
moves by 2.4x with the loop shape. What is settled is that **105 was
latency-bound and is not the ceiling**; what is not settled is what the ceiling
is for *the kernel's* loop. The way to settle it is to instrument the kernel's own
Q@K^T body (count `bfmmla` against cycles in a `perf stat` window at one shape),
not to write another microbenchmark.

The hand-written asm probe is `/tmp/opencode/remeasure/bfmmla_probe.c`
(16 accumulators, 8 threads pinned one per core, per-core min). The finding: the
documented no-load figure is 1.55x faster than this core delivers, and the
"loads halve it" ratio is a property of an 8-accumulator latency-bound loop, not
of the operand loads. What *does* reproduce is 105 MAC/ns for that
8-accumulator loop, which is why every shipped probe agrees with the old number.

Also re-measured, and different from this file: the call path (§6/§6a above,
3.40 us not 2.6), the pack rate and share (§7's 2.4 us at 128:8:64 is 2.16 us,
but it is 10% of a 21.4 us call), and the row counts (18/22 vs vLLM on the min,
not 19/22). The harness rules in §5 all held: min-based ratios, a control arm
(0.97-1.05), and a box idle at load < 0.3.
