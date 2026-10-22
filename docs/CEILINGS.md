# How much headroom is actually left on ARM (arithmetic + measured ceilings)

Written 2026-10-22 by the integrator, after the second ARM campaign. Every
number here is either (a) a hardware ceiling measured in-process on the box, or
(b) a ratio from `tests/bench_vs_refs.py` interleaved in one process. The point
of the file is to answer, per row, *what would have to be true* for us to beat
vLLM — before anyone spends another session on a schedule tweak.

## The ceiling on bf16

`cpu_attention/_mojo/tests/bfmmla_peak.mojo` and `scripts/prof/arm/peak.c` both
measure, with 8 independent accumulators (throughput, not latency):

| instruction | MACs/instr | instr/cycle | MACs/cycle | GFLOP/s/core @3.26 GHz |
|---|---:|---:|---:|---:|
| `bfmmla` (bf16 2x2x4) | 16 | **2.0** | 32 | **208** |
| `fmla v.4s` (fp32) | 4 | 3.3 | 13.2 | 86 |
| `bfdot` (bf16 4-lane pairwise) | 8 | 2.7 | 21.6 | 140 |

SVE is **128-bit** on Neoverse-V3 (`/proc/sys/abi/sve_default_vector_length` =
16), i.e. no wider vector than NEON, and there is no wider MMA. So 208 GFLOP/s
per core is the ceiling for any bf16 dense attention on this part, and 834
GFLOP/s for the 4-core m9g.xlarge we measure on.

## Efficiency per row, ours vs vLLM (bf16, causal, 4 cores)

Efficiency = measured GFLOP/s (exact causal MACs) / 835. Ours and theirs from the
same interleaved run of the verified 22-row matrix (branch `arm-e2e`, i.e. the
tree merged as `main` at 02b5c0c). `o/vllm` is the time ratio the scorecard
reports, i.e. the reciprocal of the efficiency ratio.

| shape | ours | vLLM | `o/vllm` |
|---|---:|---:|---:|
| S=128 H=8 D=64 | 39% | 50% | 1.30 |
| S=512 H=8 D=64 | 62% | 70% | 1.13 |
| S=1024 H=8 D=64 | 68% | 70% | 1.03 |
| S=2048 H=8 D=64 | 70% | 74% | 1.07 |
| S=4096 H=8 D=64 | 71% | 77% | 1.09 |
| S=256 H=16 D=128 | 65% | 78% | 1.21 |
| S=1024 H=16 D=128 | 80% | 89% | 1.11 |
| S=2048 H=16 D=128 | 84% | 93% | 1.12 |
| S=4096 H=16 D=128 | 85% | **95%** | 1.12 |
| S=512 H=32 D=128 | 73% | 85% | 1.17 |
| S=1024 H=8 D=32 | 49% | 51% | 1.04 |

fp32 against the `fmla` ceiling (344 GFLOP/s over 4 cores): we are at 67-75% and
vLLM at 61-71%, i.e. **fp32 is not efficiency-limited, it is fixed-cost and
schedule limited**, and 7 of its 11 rows already win:

| shape | ours | vLLM | `o/vllm` |
|---|---:|---:|---:|
| S=128 H=8 D=64 | 49% | 54% | 1.10 |
| S=512 H=8 D=64 | 67% | 68% | 1.01 |
| S=1024 H=8 D=64 | 70% | 70% | 0.99 |
| S=2048 H=8 D=64 | 72% | 71% | 0.98 |
| S=4096 H=8 D=64 | 72% | 71% | 0.99 |
| S=256 H=16 D=128 | 65% | 61% | 0.93 |
| S=1024 H=16 D=128 | 75% | 64% | 0.85 |
| S=2048 H=16 D=128 | 75% | 69% | 0.92 |
| S=4096 H=16 D=128 | 68% | 70% | 1.03 |
| S=512 H=32 D=128 | 68% | 65% | 0.95 |
| S=1024 H=8 D=32 | 57% | 59% | 1.03 |

**The consequence, stated plainly.** At `S=4096 H=16 D=128` vLLM runs at 95% of
the bfmmla ceiling. There is no bf16 schedule that wins that row: matching them
is 1.00 and beating them needs >95% of a ceiling they already sit on. Rows at
D=128 and long S are therefore **tied at best** with a precision-neutral bf16
kernel, and the rows that are genuinely winnable are the ones where *both*
kernels are inefficient — short S (41% vs 53%) and D=32/64 medium S — plus the
fixed costs we pay and they do not (the K/V pack, the Python call path).

## So what can actually win

1. **Short/medium S (S<=512, all dtypes): both kernels are at 41-79% of the
   ceiling, so there is real headroom**, and on top of that we pay ~6 us of pack
   and ~3.5 us of host per call out of a 40-52 us call. That is the largest
   addressable gap in the matrix and it is *not* matmul-bound.
2. **fp32 rows: the ceiling is `fmla` at 13.2 MACs/cycle, and we are at 66-68%
   of it** (vLLM 68-70%). A 5% schedule win flips the 4 fp32 rows that sit at
   1.01-1.10. fp32 has no faster instruction on ARM (a 3-term bf16 split is
   10.7 effective MACs/cycle, *worse* than 13.2; a 2-term one is 21 but loses
   the 2e-4 fp32 gate).
3. **int8 Q@K^T (`smmla`, 2x the MACs of bfmmla) is not shippable.**
   `tests/int8_qk_error.py` simulates the instruction exactly (int8 x int8,
   exact int32 accumulate, one fp32 rescale per logit, per-row Q scale, per-key
   K scale) and leaves the rest of the pipeline as the shipped bf16 kernel runs
   it. Against the fixed 3e-2 bf16 gate:

   | scheme | benchmark shapes | worst adversarial family |
   |---|---:|---:|
   | shipped bf16 | 9.5e-3 .. 1.6e-2 | 7.4e-2 |
   | int8 QK, per-key K scale | **2.0e-2 .. 3.5e-2 (fails at D=128)** | 1.3e-1 |
   | int8 QK, per-64-key K scale | 2.3e-2 .. 5.5e-2 | 1.5e-1 |
   | int8 QK on rows >=256, bf16 below | passes at 1.0e-2 .. 1.6e-2 | 1.3e-1 |

   The mechanism: 8-bit quantization of q and k adds ~0.1 absolute logit error
   at D=128, which is ~10% of the logit's own spread, and the softmax turns that
   into a 1-3% output error *on top of* the bf16 rounding that already consumes
   half the gate. Per-key scales are 1.4x better than per-tile and still not
   enough; the hybrid only passes on well-conditioned random inputs, and every
   realistic family (per-channel outliers, repeated tokens, query-magnitude
   spread, heavy tails) breaks it — including for the *shipped bf16 kernel*,
   which is itself at 2.6-7.4e-2 on those families. An 8-bit quantizer 2.5-3x
   coarser than bf16 cannot be made safe when bf16 is already at half the gate.
4. **The pack and the call path, at short S, is where the objective actually
   lives.** At S=128/8/64 bf16 we are 52 us against vLLM's 40 with ~6 us of pack
   and ~3.5 us of host inside our number and nothing inside theirs; the row
   kernel alone is already at parity there (41 vs 40 us).

## Caveat: these numbers are the 4-core dev slice

Everything above is measured on `m9g.xlarge` (4 cores) because that is the box
that is reachable. The scorecard's target is `m9g.2xlarge` (8 cores), where
bandwidth per core is lower: both kernels would lose efficiency, and the *ratio*
can move 5-10% either way. Ours is the one that pays a K/V repack per call and
reads V twice at long S, so if anything the 8-core box penalises us more than
vLLM — every conclusion pointing at "parity at best" should be re-checked there
before it is treated as final.

## What this rules out (do not spend a day on it)

* A `bf16` schedule rewrite for the long-S D=128 rows: capped at parity.
* `int8` Q@K^T at any scaling that fits the fixed gate.
* Wider SIMD: SVE is 128-bit; there is no 256-bit path on this part.
* `AMX` (x86 only) and `bfdot`-instead-of-`bfmmla` (21.6 vs 32 MACs/cycle).

## Open questions that a 10-minute microbenchmark answers

`cpu_attention/_mojo/tests/mma_peak.mojo` measures these in-process on the box.

**Answered on the box (2026-10-22):**

* `fmmla` (FP16 MMA) **does not exist**. It is the `f32mm` extension, which no
  shipping core implements; `mojo run` dies with `Cannot select: intrinsic
  %llvm.aarch64.neon.fmmla` with and without `--target-cpu neoverse-v3`. So
  there is no fp16 matrix path: an fp16 operand pays `bfmmla`'s rate at best.
* `bfmmla` re-measured at **207.6 GFLOP/s/core** (0.154 ns/instr), i.e. exactly
  2.0/cycle at 3.25 GHz. The ceiling in the table above stands.
* `smmla` (int8, 32 MACs/instr) measured at 0.043 ns/instr, which would be
  7 instructions/cycle — impossible, so the probe is being strength-reduced even
  with memory-derived operands. **The int8 rate is unmeasured**; do not quote it.
  Its only remaining use would be a 2-term split for the fp32 rows, which needs
  3 products (16-bit operands, ~1e-4 relative error against a 2e-4 gate) and
  comes out at 21 effective MACs/cycle against `fmla`'s 13.2 — worth ~1.5x on the
  matmul in the best case, with an accuracy margin too thin to ship.
* co-issue: the loop reported 1223962 "cycles/iter" because the clock probe was
  folded; re-measure with a fixed clock before drawing any conclusion.

## Where short S actually goes (measured, both boxes)

`tests/bench_scale.py` raises the head count at fixed seq: work scales, the key
range and every per-call cost do not, so the split falls out. m9g, 4 threads,
bf16, round-robin medians:

| S | H | us | %ceil | cycles/row-key |
|---|---|---|---|---|
| 128 | 8 | 75.3 | 27% | 7.44 |
| 128 | 32 | 190.9 | 42% | 4.71 |
| 128 | 64 | 368.9 | 44% | 4.55 |
| 512 | 8 | 495.0 | 65% | 3.07 |
| 2048 | 8 | 6945.5 | 74% | 2.70 |

Two separate things, and only one of them is dispatch:

1. **A 128-key range caps the kernel at 44%** however many heads are in flight
   (`4.55` cycles/row-key against `2.70` at 2048 keys). Fitting
   `cost = a*rowkeys + b` gives `a = 2.58` cycles/row-key and **b ~ 1000 cycles
   per work item** — a prologue/epilogue cost that is 3% of a long-S work item
   and ~40% of a 128-key one. That is the thing to attack for S<=512.
2. The rest is per-call: at S=128/H=8 the call is 75 us of which ~10 us is the
   per-work-item term above, ~6 us the pack and ~3.5 us the host path.

At S=128 vLLM is at 50% of the ceiling *including* their per-call costs, i.e.
above our best measured short-range rate. So that row needs the per-work-item
cost cut, not a faster matmul.

