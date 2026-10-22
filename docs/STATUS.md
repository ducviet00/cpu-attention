# Where the campaign stands

`docs/PROTOCOL.md` says how numbers are taken; `docs/FINDINGS.md` holds this
session's independent machine measurements and the harness rules that came out of
them; `docs/CEILINGS.md` is the older headroom analysis (its arithmetic is still
right, its conclusions about *where* the headroom is are not — see below).

## The one number

`ours / baseline < 1`, for both baselines (`torch SDPA` and vLLM `csrc/cpu`), on
all 22 rows of `tests/bench_vs_refs.py`. Below 1 is a win.

## Current state — m9g.2xlarge (8 cores, Neoverse-V3)

> **The table below is from `19f10b0`.** Three commits since then move the three
> contested rows and are not in it: the seven-argument extension entry
> (`10c18a7`, 128:8:64 1.11 → 1.04 in the full matrix), the last-tile store out
> of the P@V registers (`ca26a10`, 2–2.5% at 128:8:64 and ~3% at 256:16:128 in
> isolated A/B), and the short-sequence path back on (`_SMALL_MAXK=256`, 1–2%
> on the two rows it covers with the arm order rotated). With those, the matrix
> reads **19/22 vs vLLM** with `128:8:64` 1.00-1.02, `256:16:128` 1.01-1.02 and
> `4096:16:128` 1.01-1.02 — i.e. parity inside this box's 1-3% drift. These rows
> are being re-measured on a fresh instance; replace this note with the number
> you measure.

Two `scripts/verify.sh` runs, `--rounds 9`, clean rebuild, **min-based** ratios
(`scripts/band.py`; the minimum cannot be inflated by foreign load, the median
can — see `FINDINGS.md` §5). `o/sdpa` is 0.29-0.88 on every row.

| shape | bf16 o/vllm | fp32 o/vllm |
|---|---:|---:|
| S=128 H=8 D=64 | **1.12-1.13** | 0.95-1.02 |
| S=256 H=16 D=128 | **1.01-1.07** | 0.80 |
| S=512 H=8 D=64 | 0.90-0.93 | 0.90-0.91 |
| S=512 H=32 D=128 | 1.00 | 0.99-1.00 |
| S=1024 H=8 D=32 | 0.87 | 0.91 |
| S=1024 H=8 D=64 | 0.90-0.91 | 0.93 |
| S=1024 H=16 D=128 | 0.98 | 0.89-0.92 |
| S=2048 H=8 D=64 | 0.92-0.93 | 0.96-0.98 |
| S=2048 H=16 D=128 | 0.97-0.99 | 0.86-0.87 |
| S=4096 H=8 D=64 | 0.91-0.92 | 0.97-0.98 |
| S=4096 H=16 D=128 | **1.01-1.02** | 0.98-0.99 |

**19/22 rows beat vLLM, none is on the line, 3 are above 1; 22/22 beat SDPA**
by 1.3-3.4x (`o/sdpa` 0.30-0.88). fp32 is 11/11. The three rows still above 1 are
all bf16: `128:8:64` (1.12-1.13), `256:16:128` (1.01-1.07) and `4096:16:128`
(1.01-1.02). Two of them are inside a 2% band and flip verdict between runs;
`128:8:64` is a real 12% and its budget is below.

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

Precision: bf16 rows are at 7.8e-3-1.6e-2 against the fixed 3e-2 gate, fp32 at
8e-7-3e-6 against 2e-4. Everything is inside tolerance with 2x margin in the
worst case (the degree-3 exponential is a one-line revert worth 2-5% if that
margin is ever needed).

## Why the last four rows are not a schedule tweak

* **2048/4096:16:128 bf16 (1.00-1.01)** are matmul-bound and at the floor. Phase
  ablation, two runs: Q@K^T 38-39%, P@V 42-46%, exp 15-16% *not overlapped*, pack
  0.8-1.8%, per-item 3.9-4.7% — 99.9-100.3% of the call is accounted for. The
  `bfmmla` ceiling with the kernel's own load shape is **105 MAC/ns/core** (208
  without loads, i.e. the operand loads halve the MMA), so 840 MAC/ns over 8
  cores; we measure 1582 GFLOP/s = 94% of it, which is where vLLM sits too. The
  1-2% is their smaller per-item cost and the fact that **we are charged for the
  K/V pack and they are not** (their cache is staged before the timer).
* **128:8:64 bf16 (1.10)** is not a kernel-rate loss. At one thread we are at
  parity with vLLM (146 us each); the call is 25 us of which the kernel is ~19.6
  (against their 21 — we win the kernel), the pack 2.4 (charged to us, staged
  outside their timer), and the Python path ~4.5 (theirs ~1.5 through pybind). A
  H-scaling fit on the row puts the whole difference in the per-call cost.
  Attempts to cut the call path this session: caching the validation (0.26 us,
  not worth a cache), a 5-argument plan-buffer ABI for the extension (abandoned —
  49 hand-mapped fields, and the two bugs it produced were silent until they
  crashed; see `FINDINGS.md` §6). The row needs the Python path under ~2 us,
  which means moving the call path out of Python, not tuning it.
* **256:16:128 bf16 (1.03-1.05)** is fixed-cost-bound: pack 6.3% + per-item 5.7%
  against a matmul share of 86%. The short-sequence kernel already takes 3% of
  it; the residue is the same per-call cost as the row above.

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
| plan-buffer extension ABI (5 args instead of 42) | the marshal is 1.2 us of a 25 us call | 49 hand-mapped fields; two silent mismatches (packed-vs-raw V strides) before it segfaulted. Worth ~1 us on one row; not worth the correctness surface |

## Beyond this campaign's measurement resolution

Three things are true and worth stating plainly:

1. **The box is shared and it drifts.** Foreign load moved medians 15% and turned
   a 0.85 row into 1.01 twice. Every number here is min-based over ≥9 interleaved
   rounds on a box that was quiet at the start; `scripts/verify.sh` now refuses to
   time until the load average is under 0.8.
2. **The scorecard charges us for the K/V pack and not vLLM.** That is the
   contract (vLLM's cache is staged before its timer), and it is 0.8-6.3% of the
   rows above 1 us — the single largest structural difference left.
3. **x86 must keep building.** Merging `dense_small.mojo` broke it with
   `LLVM ERROR: Cannot select: intrinsic %llvm.aarch64.neon.bfmmla` because Mojo
   instantiates the ARM kernel even where nothing calls it. Fixed with
   `comptime if BFMMLA`; build locally before every merge.
