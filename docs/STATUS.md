# Where the campaign stands

`docs/PROTOCOL.md` says how numbers are taken; `docs/FINDINGS.md` holds this
session's independent machine measurements and the harness rules that came out of
them (`§8` is the 2026-10-23 re-measurement that corrects `§1`); `docs/CEILINGS.md`
is the older headroom analysis — its per-row bounds are superseded, and its
`bfmmla` ceiling paragraph was mis-measured and carries a correction. The
re-measurement log with commands, per-run tables and the control readings is
`docs/REMEASURE.md`.

## The one number

`ours / baseline < 1`, for both baselines (`torch SDPA` and vLLM `csrc/cpu`), on
all 22 rows of `tests/bench_vs_refs.py`. Below 1 is a win.

## Current state — m9g.2xlarge (8 cores, Neoverse-V3), re-measured 2026-10-23

Fresh spot instance (`ubuntu@54.255.175.68`), `main` @ `973cce9`, clean rebuild,
two `scripts/verify.sh` runs (`--rounds 9`) on a box idle at load 0.28 and 0.29,
plus one full-matrix run at `--rounds 21 --control` whose duplicate `ours` arm
read 0.97-1.05 (the run's noise floor). Ratios are `ours/baseline` and
**min-based** (`scripts/band.py`; the minimum cannot be inflated by foreign
load, the median can — see `FINDINGS.md` §5). `o/sdpa` is 0.282-0.877 over the
22 rows. Absolute µs, per-run tables and commands are in `docs/REMEASURE.md`.

| shape | bf16 o/vllm | fp32 o/vllm |
|---|---:|---:|
| S=128 H=8 D=64 | **1.01-1.05** | 0.92-0.96 |
| S=256 H=16 D=128 | **1.05** | 0.80 |
| S=512 H=8 D=64 | 0.90-0.92 | 0.90 |
| S=512 H=32 D=128 | 0.99-1.00 | 0.84-0.85 |
| S=1024 H=8 D=32 | 0.87 | 0.91 |
| S=1024 H=8 D=64 | 0.89-0.92 | 0.91-0.93 |
| S=1024 H=16 D=128 | 0.98 | 0.83-0.85 |
| S=2048 H=8 D=64 | 0.92-0.93 | 0.94-0.95 |
| S=2048 H=16 D=128 | **1.00** | 0.87-0.88 |
| S=4096 H=8 D=64 | 0.93 | 0.95 |
| S=4096 H=16 D=128 | **1.00-1.01** | 0.97-0.99 |

**18/22 rows beat vLLM in both runs on the min** (run 1: 20/22, run 2: 18/22;
by median 19/22 and 17/22; **20/22** in the 21-round + control run). One row is
on the line (`4096:16:128` bf16, 0.995 then 1.010) and three lose in both runs,
all bf16: `128:8:64` (1.01-1.05), `256:16:128` (1.05) and `2048:16:128`
(0.999-1.002 min, 1.017 median). **22/22 beat SDPA** by 1.1-3.5x (`o/sdpa`
0.28-0.88). **fp32 is 11/11** in both runs; the closest is `4096:16:128` at
0.967/0.994.

The rows near the line need more than a 9-round median, because their
within-run spread is 3-8%. Three of them were re-measured in five separate
processes at `--rounds 21 --control` (the protocol in `NEXT.md` §4; control arm
0.97-1.03):

| shape (bf16) | isolated, 5 runs (min) | 9-round matrix (min) | verdict |
|---|---:|---:|---|
| S=128 H=8 D=64 | 0.987-1.020 | 1.014, 1.045 | parity, not resolved |
| S=256 H=16 D=128 | **0.914-0.957** | 1.049, 1.050 | parity or better; see below |
| S=4096 H=16 D=128 | 1.001-1.009 | 0.995, 1.010 | a real 0-1% loss |
| S=2048 H=16 D=128 | not run | 0.999, 1.002 | at the line, not below it |

`256:16:128` bf16 is the one row whose value depends on the harness: 0.91-0.96
alone, 0.973 in a 21-round full matrix, 1.03-1.07 in the 9-round full matrix.
Its per-sample spread is ±8% (40 samples, 199-239 µs, median 214) against 1-2%
for the other rows, and a 9-round median lands anywhere in that band. A 12 s
8-core burn and preceding it with the whole 4096:8:64 fp32 row do not move it,
so it is a property of the row, not drift. Quote it as **parity, 0.91-1.07**.

Two rows are *better* than the previous campaign recorded, by more than this
box's variation and in three independent measurements: `512:32:128` fp32
**0.84-0.85** (was 0.99-1.00) and `1024:16:128` fp32 **0.83-0.85** (was
0.89-0.92). `2048:16:128` bf16 moved the other way: 1.00-1.02, not 0.97-0.99.

## What moved this session, in order of payoff

| change | where | effect |
|---|---|---|
| **per-head dispatch**: one thread packs a head's K/V and attends every one of its query blocks, no barrier between phases | `dense.mojo`, `_kernel._dense_plan` | 0.94-0.97 on every shape in-process; it is what removed the short-S parallel-efficiency collapse |
| **work-item order head-first below an L2-sized working set** (the non-MHA fallback) | `_dense_task_pair` | 1.14-1.20x at 128/256, 1.07-1.12x at 512 |
| **K^T packed as `[tile][key pair][dim group][8]`**, 128→384-key tiles at D=128, qt=8 everywhere | `dense.mojo`, pack | bf16 2048:16:128 1.15 → 1.02, 256:16:128 1.07 → 1.01 |
| **short-sequence kernel** (`dense_small.mojo`): key coverage rounded to 8 keys, its own fused region | `dense_small.mojo` | 1.02-1.03x at seq_k ≤ 256 |
| **bf16-precision exponential**: degree-2 minimax + exponent bit-add | `dense.mojo` | 3-7% on every bf16 row; max\|diff\| 7.8e-3 → 1.6e-2 against a 3e-2 gate |
| **fp32 exp/max instruction diet**, V-copy gate, qt validation | `dense.mojo`, `_kernel` | fp32 4/11 → 11/11 wins |
| **workspace budget charged up front** (24 MB, at plan build) | `_kernel` | late rows in a sweep read their isolated value instead of 5-15% slow |
| **min-based ratios + box-quiet wait** in the harness | `tests/bench_vs_refs.py`, `scripts/band.py`, `verify.sh` | the numbers stopped lying; see `FINDINGS.md` §5 |

The "effect" column above is the previous box's measurement and is kept as the
record of what each change was worth *there*. On the 2026-10-23 instance the
same tree lands where the table at the top of this file says — in particular
`256:16:128` bf16 does not hold the 1.01 the K^T-layout + short-sequence rows
claim here (it is 0.91-1.07 depending on the harness), and `2048:16:128` bf16 is
1.00-1.02 rather than the 0.97-0.99 recorded above.

Precision, re-measured: bf16 rows are at 7.8e-3 (D=64) to 1.6e-2 (D=128)
against the fixed 3e-2 gate, fp32 at 6.3e-7-2.1e-6 against 2e-4. Everything is
inside tolerance with 1.9x margin in the worst case (the degree-2 exponential is
a one-line revert worth 2-5% of bf16 if that margin is ever needed). Gates:
`tests/test_dense.py` 65/65 including "plan-buffer entry is bit-identical on 24
cases", `tests/test_flash_api.py` green, `tests/fuzz_dense.py --cases 80
--max-seq 900` 80/80 with the worst case at 0.52x its tolerance and the
stray-read probe 24/24 clean.

## Why the last four rows are not a schedule tweak

* **2048/4096:16:128 bf16 (1.00-1.01)** are matmul-bound. Phase ablation, two
  runs: Q@K^T 38-39%, P@V 42-46%, exp 15-16% *not overlapped*, pack 0.8-1.8%,
  per-item 3.9-4.7% — 99.9-100.3% of the call is accounted for. The `bfmmla`
  ceiling has been re-measured on a fresh instance and **the old number was
  wrong**: with the kernel's own load shape (4 `ldp` per 16 `bfmmla`) a
  hand-written asm loop on 8 cores does **144-145 MAC/ns/core**, not 105, and
  the no-load figure is 128-134, not 208; 105 is exactly what an
  **8-accumulator** loop measures (0.152 ns/instr) and it is latency-bound, not
  the issue limit. So the Q@K^T + P@V phases (118-124 MAC/ns/core) run at
  **82-85%** of the measured ceiling, and the whole call at 68% — i.e. the
  matmul phases have ~15-18% headroom, not 6%. What has not changed is the
  empirical answer: five restructurings of that block lost (ledger below), and
  the 1-2% to vLLM is their smaller per-item cost plus the fact that **we are
  charged for the K/V pack and they are not** (their cache is staged before the
  timer). Full numbers in `docs/REMEASURE.md` §4a.
* **128:8:64 bf16 (1.01-1.05)** is not a kernel-rate loss. The call is 21.4 us
  of which the pack is 2.16 (measured alone, `pack_only`; charged to us, staged
  outside their timer) and the Python path **3.4 us** (theirs ~1.0-1.5 through
  pybind). A H-scaling fit on the row puts the whole difference in the per-call
  cost. Attempts to cut the call path this session: caching the validation
  (0.26 us, not worth a cache), a 5-argument plan-buffer ABI for the extension
  (abandoned — 49 hand-mapped fields, and the two bugs it produced were silent
  until they crashed; see `FINDINGS.md` §6) and the seven-argument plan buffer
  that shipped, which is worth **0.40 us**, not the 1.9 us claimed when it
  landed. The row needs the Python path under ~2 us, which means moving the
  call path out of Python, not tuning it.
* **256:16:128 bf16 (0.91-1.07, quote it as parity)** is fixed-cost-bound: pack
  3.5% + per-item 5.7% against a matmul share of 86%. The short-sequence kernel
  already takes 3% of it; the residue is the same per-call cost as the row
  above, plus this row's own ±8% sample spread (the widest in the matrix).

## The falsification ledger (this session, with the number that killed each idea)

| idea | why it looked good | what killed it |
|---|---|---|
| no-pack kernel for short S (read k/v natively) | the pack is 2.4 us of a 25 us call | assembling a bfmmla operand from `k[token][head][dim]` costs +206 instructions per 256 `bfmmla` and the P@V operand +29 per 16: ~10 instructions per row-key against 2.4 us of pack (`dense_small.mojo` header) |
| per-head item to collect the per-block drain (Q panels for 8 blocks, first tile reading a zero buffer) | 28% of the call is per-query-block cost | 23.2 vs 23.4 us — hoisting the panel rebuild costs what it saves, the 8 KB panel shares L1 with the K/V tile |
| 16 query rows per work item (v3, two 8-row panels) | halves compulsory K/V traffic | 1.00x against main; the 5.6% it first showed was against its own qt=8 geometry |
| fusing the exp into the Q@K^T loop | the exp pass is 15-29% of the call | 12-25% slower on every shape: the per-sub-block max/rescale state spills the 16 accumulators |
| `accum_c` (skip the P@V init and first reload, vLLM's trick) | one round trip fewer per tile | **26% slower** — the `t0 > 0` test makes the accumulator a phi and LLVM spills all 16 |
| contiguous K^T layout / L1-resident K/V probes | vLLM's pages are contiguous | 2-3% and 2.8%: the kernel is not memory-bound, and arm-v2's layout change already took the contiguous sweep |
| 128 vs 256 vs 512-key tiles at D=128, `dcp` 2/8, `nsq` 8 | register/L1 balance | 384 is best (2.1-4.1% under 128 on every shape, shipped); dcp=8 15-27% slower, nsq=8 12-23%, 512 loses at D=64 |
| int8 `smmla` Q@K^T | 3x bfmmla's MAC rate on this core | misses the fixed 3e-2 bf16 gate at every scaling (2.0-3.5e-2 on the benchmark shapes; 4 adversarial families fail) |
| mixed `bfmmla` + fp32 `fma` in one loop | fmla runs on other ports | 8+8 takes 0.366 ns/pair against 0.154/0.303 separately: the core issues ~4 instructions/cycle and a `bfmmla` already yields 16 MACs per slot |
| caching the call-path validation | 4.5 us of Python on a 25 us call | 0.26 us saved; the six `.shape`/`.stride()` reads and the output allocation are the cost, not the 40 comparisons |
| plan-buffer extension ABI (5 args instead of 42) | the marshal is 1.2 us of a 25 us call | 49 hand-mapped fields; two silent mismatches (packed-vs-raw V strides) before it segfaulted. Re-measured on a fresh box as the 7-argument entry that shipped: it saves **0.40 us** (3.80 -> 3.40 us for the Python path), not 1.2-1.9, so the entry was worth keeping but it is not the row's lever |

**Corrected by the 2026-10-23 re-measurement** (details and the probe source in
`docs/REMEASURE.md` §4): the `bfmmla` ceiling this campaign reasoned from —
"208 MAC/ns/core without loads, 105 with the kernel's 8 loads per 16, the loads
halve it" — does not reproduce. A hand-written asm loop, 16 accumulators, all 8
cores: **128-134 MAC/ns** with no loads, **114-115** with 8 loads per 16, and
**144-145** with the 4 `ldp` per 16 that the kernel actually emits. 105 is what
an 8-accumulator loop measures (0.152 ns/instr, reproduced on `peak.c` and
`bfmmla_peak.mojo`); that loop is latency-bound, so the "loads halve the MMA"
mechanism was a loop-shape artefact. The conclusion it was used for ("the
matmul is at the hardware wall, 94% of it") is therefore not established: the
Q@K^T + P@V phases are at 82-85% of the measured 4-load ceiling and the whole
call at 68%. The pack's documented "422-495 GB/s, 0.8-6.3% of a call" is also
narrower than what this box measures (243-695 GB/s; 0.2-1.7% for S>=512 but
10.1% at 128:8:64). Both numbers are quoted, both boxes' figures are in
`REMEASURE.md`. The ledger above is left as it stood except for the one row the
new box re-measured (the plan-buffer ABI, amended in place with both numbers).

## Beyond this campaign's measurement resolution

Three things are true and worth stating plainly:

1. **The box is shared and it drifts.** Foreign load moved medians 15% and turned
   a 0.85 row into 1.01 twice. Every number here is min-based over ≥9 interleaved
   rounds on a box that was quiet at the start; `scripts/verify.sh` waits for a
   1-minute load average under 0.3. Even on an idle box a few rows carry 3-8%
   sample spread (`256:16:128` bf16 is the worst: ±8%), so a row near the line
   needs the 21-round + control protocol, not one 9-round run.
2. **The scorecard charges us for the K/V pack and not vLLM.** That is the
   contract (vLLM's cache is staged before its timer). Re-measured, the pack is
   0.2-1.7% of a call for S>=512 but **3.5% at 256:16:128 and 10.1% at
   128:8:64**, so it is the single largest structural difference left exactly on
   the rows that are above 1.
3. **x86 must keep building.** Merging `dense_small.mojo` broke it with
   `LLVM ERROR: Cannot select: intrinsic %llvm.aarch64.neon.bfmmla` because Mojo
   instantiates the ARM kernel even where nothing calls it. Fixed with
   `comptime if BFMMLA`; build locally before every merge.
