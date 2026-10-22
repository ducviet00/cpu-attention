# FINDINGS — independently measured, 2026-10-22 (integrator)

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
