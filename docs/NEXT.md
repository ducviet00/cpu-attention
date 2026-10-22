# What is left, and how to pick it up

Written at the end of the third campaign (ARM/Graviton, 2026-10-23) as the
handover. `docs/STATUS.md` is the scoreboard and the falsification ledger,
`docs/FINDINGS.md` is the machine data and the harness rules, this file is the
*plan*. **Every number below was measured on one spot box and must be
re-measured on the box you are given before you trust it** — that is the first
task, not a formality (see "Re-measurement" at the end).

## The state you inherit

`ours / baseline` (below 1 wins), min-based, two clean-rebuild matrix runs,
interleaved, arm order rotated, box idle at load < 0.3:

| | value |
|---|---|
| rows beating **torch SDPA** | 22/22, by 1.1–3.5x (`o/sdpa` 0.28–0.88) |
| rows beating **vLLM csrc/cpu** | **18/22 in both runs on the min** (20/22 in run 1, 18/22 in run 2; 20/22 in a 21-round run with a control arm) |
| remaining rows | `128:8:64` bf16 1.01–1.05, `256:16:128` bf16 0.91–1.07 (harness-dependent: 0.91–0.96 isolated), `2048:16:128` bf16 1.00–1.02, `4096:16:128` bf16 1.00–1.01 |
| parity | bf16 `max\|diff\|` 7.8e-3–1.6e-2 against a 3e-2 gate, fp32 6.3e-7–2.1e-6 against 2e-4, green on every row (test_dense 65/65, fuzz 80/80) |

Re-measured 2026-10-23 on a fresh instance; `docs/REMEASURE.md` has the per-run
tables, the commands and the control readings. The count is 18/22 rather than
19/22 because `2048:16:128` bf16 now reads 1.00–1.02 (a 3% miss) and
`4096:16:128` is on the line at 0.995/1.010; `256:16:128` reads 0.91–0.96 when
it is measured alone but 1.05 as row 6 of a 9-round sweep (its sample spread is
±8%), so it is not resolved either way. fp32 is 11/11 with `4096:16:128` at
0.97–0.99 the closest to the line; `512:32:128` fp32 and `1024:16:128` fp32 are
both 6–15% better than the table this file inherited (0.84–0.85).

## Why they are at parity (measured, not argued)

Per-row accounting from phase ablation, two runs each, closing to ±0.3%:

| shape (bf16) | matmul | exp (not overlapped) | pack | per-item | unexplained |
|---|---:|---:|---:|---:|---:|
| 512:32:128 | 86% | 15% | 6.3% | 5.7% | 0.0% |
| 1024:16:128 | 77% | 15% | 3.7% | 8% | 0.3% |
| 2048:16:128 | 85% | 16% | 1.8% | 1.5–6.7% | 0.1% |
| 4096:16:128 | 80% | 15% | 0.8% | 1.5–3.9% | 0.1% |

The matmul was thought to be at a hardware wall. **That ceiling was
mis-measured and has been re-derived** (`REMEASURE.md` §4a): `bfmmla` is
**128–134 MAC/ns/core with no loads** and **144–145 with the four `ldp` per 16
that the kernel emits** (hand-written asm, 8 cores). The old pair "208 no
loads / 105 with 8 loads, the loads halve it" is the 8-accumulator
latency-bound loop, not the issue limit — 105 reproduces exactly for that loop
on `peak.c` and `bfmmla_peak.mojo`, which is why it survived.

**How much headroom that leaves is unresolved, and it is worth one hour to
settle before anything else.** Three probes of the same core now disagree:
128–134 (no loads, 16 acc), **168** (no loads, 16 acc, 2 operands, 16 `bfmmla` per
iteration = 3.3–3.5 `bfmmla`/cycle), and **61–145** with 4 `ldp` per 16 depending
on whether the loads feed the next instruction pair. The number moves 2.4x with
the loop shape, so a microbenchmark cannot decide it. The Q@K^T + P@V phases
measure 118–124 MAC/ns/core at 4096:16:128, so the matmul is at **68–85%** of the
candidate ceilings and the whole call at 68%: somewhere between 0 and 40% of
headroom, not the 6% this campaign assumed, and not the 15–18% a single probe
suggests either.

Settle it from inside the kernel: instrument the shipped Q@K^T body (count
`bfmmla` retired against cycles in a `perf stat` window at one shape, both
cached-hot and streaming) and compare against a hand-written loop with *that
exact* load pattern. If the phases are within a few percent of it, the ranked
list below is the whole plan; if they are 15% off, the matmul is the plan. Two things are charged to us and not to vLLM:

* **the K/V pack** (0.6-1.7% for S>=512, 3.5% at 256:16:128, 10% at 128:8:64
  on the re-measured instance): their cache is staged before the scorecard's timer
  and ours is inside it. That is the contract, not a bug.
* **the Python call path** (~3.4 µs after the seven-argument entry; theirs is
  ~1.0 through pybind). At 128:8:64 that is 16% of the call. The plan buffer is
  worth 0.40 µs of it, not the 1.9 µs originally recorded.

## Ranked next steps

### 0. Settle the matmul ceiling from inside the kernel (1 hour, decides the ranking)
See above. It is a measurement, not a rewrite: `perf stat` around one shape's
Q@K^T phase versus a hand-written loop with the same load pattern. Publish the
ratio. Everything below is ranked on the assumption that the answer is "near the
ceiling"; if it is not, swap item 0's answer in for items 1–3.

### 1. The exp block, hidden instead of shrunk (biggest prize, three failures so far)
The exp is 15% of a D=128 call and 29% of a D=64 one, it does not overlap the
matmul, and the polynomial is already degree-2. What is left is to overlap it:
`bfmmla` takes 2 issue slots/cycle and the core issues ~4, so the other two slots
are free in principle. Three attempts lost — fusing the exp into the Q@K^T body
(12–25% slower, the per-sub-block max/rescale state spills the 16 accumulators),
software pipelining the tile (11% slower), and interleaving P@V with the next
tile's exp (register wall at 34–36 against 32). The untried version is vLLM's own
shape: keep the logits in memory and run **one straight-line softmax pass** over
the whole 384-key tile, so the max/exp/sum has a long dependence-free body. Our
`s=384` tile is now big enough for that to be worth more than it was when this
was rejected.

### 2. The pack, made to disappear from the timed call for the mid rows
6.3% at `512:32:128`, 3.7% at `1024:16:128`. Fusion of the K^T and V passes is
worth only 0.2–0.5% (measured); the pack is already at 422–495 GB/s ≈ 90% of
memcpy. The only real escape is to stop repacking: agent B's operand probe says
reading K/V in their native layout costs ~10 instructions per row-key (against
2.4 µs of pack for a whole 128-key call), which is why the pack stays. A
*partial* escape is untried: keep the V pack (which the P@V operand layout really
needs) and read K natively through a K-side minipack per tile — K's native layout
costs +0.8 instructions per bfmmla against V's +1.8.

### 3. `qt=16` with a *smaller* key tile (the register wall may be movable)
Every `qt` sweep this campaign lost, but they all held `nsq`/`dcp` fixed. At
`qt=16/nsq=2/dcp=2` the QK tile is 16 accumulators — the same count as `qt=8/nsq=4`
— so the failure was something else (measured 9–12% slower). Re-run it as a
2x2 sweep of (qt, ks) with `RP*nsq = 16` held, on the D=128 rows only.

### 4. The three rows inside the drift band
Do not spend a day here. Run five separate processes with `--control` and publish
the band. If a row reads 1.01 ± 0.01 it is parity and the scorecard should say so.

## x86 / Xeon: what to leverage

Nothing on x86 has been re-measured since this campaign's ISA-independent changes
landed (per-head dispatch, work-item ordering, the plan-buffer entry, the
workspace budget, the min-based harness). **That is the first x86 task** — the
tree is very likely faster there for free, and the recorded Xeon baselines are
stale.

Then, in order:

* **AMX for bf16 is the one big x86-only lever** and it already exists:
  `dense_amx.mojo` (+ `_csrc/dense_amx.c` as the reference), gated on
  `amxbf16`, worth 4–9.6x on bf16 in the second campaign. The scorecard's honest
  bar for bf16 on an AMX machine is vLLM's AMX build (`verify.sh x86_64` builds
  it), not VEC. fp32 has no AMX kernel in vLLM either, so VEC is that column's
  bar — and our fp32 AMX path is the hi/lo bf16 split, three products, measured
  5.5–7.6e-5 against the 2e-4 gate.
* **The fp32 rows are the weak column on x86**: the AVX-512 fp32 path with no
  AMX ran 1.09–1.90 against VEC. The ARM answer (a per-head dispatch so a thread
  reuses one head's K/V, plus the plan-buffer call path) should port directly and
  is worth measuring before anything new is written.
* **`_qt_for`'s x86 branch is from the second campaign** (`qt=8` at head_dim
  64/128 fp32, 4 at 32; bf16 8 at 128, 4 at 64). It was swept on a 4-physical-core
  m8i; re-sweep it on whatever box you get, with the rotated-order harness.
* **Do not port the ARM tile kernel to x86.** AVX-512 has 32 zmm registers and a
  different load:store ratio; the second campaign's attempts to transplant
  structure between ISAs (and `_dense_rows_gemm`) both lost 20–26%.
* **The pack is cheaper per byte on x86** (fewer, faster cores) but there are
  fewer cores to hide it: measure `pack_only.py` before assuming the shares in
  `STATUS.md` transfer (on the 2026-10-23 instance the pack is 243-695 GB/s,
  depending on whether the working set fits in L3).

## Measurement protocol (non-negotiable, this cost the campaign two sessions)

1. **The box must be idle.** `scripts/verify.sh` waits for a 1-minute load
   average under 0.3 before it times anything and prints the load; a foreign
   `mojo build` inflated a row by 1.7x and a foreign python by 25%.
2. **Rotate the arm order.** Whichever arm runs first in a round is 1–3% slower —
   two arms running *identical* code measured 3.4% apart. `bench_vs_refs.py` and
   `ab_two_trees.py` rotate; `--control` adds a duplicate arm and its ratio is the
   run's noise floor (should read 0.99–1.01). Any claim under 3% needs a control.
3. **Judge on the minimum**, not the median: interference can only slow a sample
   down. `scripts/band.py` and `scorecard.py` do.
4. **Plan-level knobs must be toggled in one process**; a knob that invalidates
   the call plan inside a timing loop costs up to 19% by itself.
5. **Build on x86 before every handover.** Merging an ARM-only file without
   `comptime if BFMMLA` broke the x86 build with
   `LLVM ERROR: Cannot select: intrinsic %llvm.aarch64.neon.bfmmla`.

## Re-measurement checklist for the fresh box

Provisioning: `scripts/box_sync.sh arm64 ubuntu@<ip> . --full` (venv + kernel +
the pinned vLLM reference `.so`, ~12 min), then `ln -sfn ~/cpu-attention/.venv
~/<dir>/.venv` for any extra tree.

Then reproduce, in this order, and correct the docs where they disagree:

1. `scripts/verify.sh arm64 ubuntu@<ip> . r1` with `RUNS=2` → the 22-row table.
   Measured on the 2026-10-23 instance: **18/22 vs vLLM on the min in both runs**
   (20/22 in run 1, 20/22 in a 21-round + control run), all rows `o/sdpa`
   0.28-0.88. **Absolute µs will differ from the numbers in `STATUS.md`**
   (different physical host, and these boxes vary 5-10%); the *ratios* are what
   the docs claim, and the remaining rows are all within ±7% of 1.
2. `tests/bench_phases.py --shapes 128:8:64,256:16:128,512:8:64` → the pack
   share. Its `pack%` column carries an extra fork-join; the honest number is
   `tests/pack_only.py`, which on 2026-10-23 did not run on `main` (see
   `REMEASURE.md` §4b for the two-line patch).
3. `perf stat -e cycles,instructions taskset -c 3 python tests/prof_one.py …`:
   the PMU on this instance inflates absolute cycles ~2.2x (a 27 ms call cannot
   retire 4 GHz-cycles), so use ratios only, or drop the counter entirely.
4. `python scripts/band.py` on the two JSONs; note the worst-run min per row.
5. `tests/fuzz_dense.py --cases 80 --max-seq 900` (parity, odd lengths, non-causal).
6. The ceiling probes if you plan to touch the matmul: a hand-written asm loop is
   the only trustworthy way to measure `bfmmla`/`fmla`/`smmla` throughput here;
   `tests/bench_amx_ablate.py`-style in-process probes are fine for phases. The
   shipped `bfmmla_peak.mojo` and `peak.c` probes both measure an
   **8-accumulator** loop, which is latency-bound at 105 MAC/ns; use 16
   accumulators (see `REMEASURE.md` §4a for the probe).

Anything in `docs/` that does not reproduce: fix the doc, not the number, and
write the discrepancy down with the measurement that killed it — that ledger is
what stops the next campaign repeating a dead end.
