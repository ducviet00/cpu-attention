# Dense attention vs SDPA and vLLM: campaign report (2026-10-22)

> **Superseded (2026-10-23).** This is the closing report of the third campaign;
> its verdict table (ARM 5 wins / 11 losses, x86 unverified) predates the merge
> of `bfmmla`, the per-head dispatch and the AMX work. The current numbers are
> in `docs/STATUS.md` and the independent re-measurement in `docs/AUDIT.md`:
> ARM is 21-22/22 vs vLLM, x86 is 22/22 vs VEC, and bf16 vs vLLM's AMX build is
> at parity (9/11 by min). The mechanism analysis below is still the right
> historical account of why the old residue existed.

This is the closing report for the campaign whose objective was: `ours/baseline
< 1` in both columns (`torch SDPA`, vLLM `csrc/cpu`) on both platforms, for all
22 rows of `tests/bench_vs_refs.py`, parity green. Ratios are `ours/baseline`;
below 1 is a win; ours includes the K/V pack, vLLM's timed region does not.

Read `docs/CEILINGS.md` for the headroom arithmetic and `docs/STATUS.md` for the
per-row tables and the ledger of what was falsified.

## Verdict

| platform | vs SDPA | vs vLLM | status |
|---|---|---|---|
| ARM (Graviton 5, m9g.xlarge, 4 cores) | **22 / 22** | **5 wins, 6 within ±3%, 11 losses** | measured, bands over repeat runs |
| x86 (laptop Ryzen, AVX-512 **no AMX**, VEC path) | **22 / 22** | **9 / 22** | measured, current tree |
| x86 (Xeon, AMX path — the x86 target) | 21-22 / 22 (last reading) | 21-22 / 22 (last reading) | **unverified since `164e8e6`**: AWS auth expired and cannot be renewed without an interactive `aws login` |

The objective is **not met**: 11 ARM rows still lose to vLLM, and the x86 target
has not been re-verified after this campaign's shared-code changes.

## Why the ARM residue is not a schedule tweak

Three independent, measured reasons. All are in `docs/CEILINGS.md` with the
arithmetic.

1. **The instruction ceiling.** `bfmmla` issues at 2.0/cycle on this core
   (207.6 GFLOP/s/core, re-measured). For each row, the ratio a *perfect* kernel
   would achieve is `instruction floor / vLLM's measured cycles per row-key`:
   0.51-0.76 at D=64/D=32, **0.95-0.96 at D=128 long S**, 0.54-0.73 fp32. So the
   D=128 rows at 2048/4096 need 96%+ of the hardware ceiling — vLLM already runs
   at 94-96% there, so those rows are parity at best.
2. **The IPC gap, not the instruction count.** vLLM runs IPC 4.07-4.16 where we
   run 3.50, at the same or a *higher* instruction count (35.7 vs 32.4 per
   row-key at 1024:16:128). Closing that gap alone would be worth 13-16%.
3. **Every way to close it is measured dead.** The exp is 23-24% of a D=64 call
   (ablation: patching it out is 1.23-1.24x), but hiding it needs every P@V
   d-chunk to have every key group's probability, i.e. `RP*DP` = 128 resident
   accumulators against 32 registers. Every way to pay for that measured worse
   than the 23% it would buy: `qt=4` +22%, `TILE_NSQ=2` 0.90x, `dcp=8` +29%,
   P@V unroll 0.99x. Both K and V layout reorderings leave the L1 miss counters
   unchanged — the misses are compulsory stream misses, 55-62% prefetched — so
   the memory fourth of the stall is not addressable.

Profiling vLLM's own kernel closes the loop: their hottest instructions are the
same softmax block (a serial row-sum `fadd` at 5.75%, mask `bsl`/`bic`, `bfcvtn`,
`fcvtzs`+integer-add exp) and at 2048:8:64 they are 4% ahead of us. There is no
structure to copy; both kernels are softmax-bound.

## What did move (this campaign, verified interleaved)

* **fp32 4/11 -> 7/11 vs vLLM.** K/V pack vectorised (`zip1/zip2` instead of
  per-lane inserts plus a runtime `kt_blk` division in the inner loop), the
  output epilogue de-interleaved with `zip1.2d/zip2.2d`, Q@K^T unrolled over
  head_dim.
* **bf16 long-S D=64 rows 1.03-1.13 -> 1.00-1.03.** The exp pass with four key
  groups in flight (+2.3%, it is latency-bound), `KS_STRIDE` instead of the
  runtime `job.kt_blk` in the sub-block address (`sdiv`+`msub` per 8 keys), and
  FA4's -8 rescale slack.
* **256-key tiles at head_dim 128** (+1-2% on the four D=128 rows).
* **The Python call path 5.0 -> 3.4 us** via a per-geometry call plan; the
  extension marshal alone is 0.76 us of that, so the path is at its floor.
* **A real correctness bug found and fixed**: the fp32 P@V loop read up to three
  V rows past the caller's tensor when `seq_q != seq_k` (NaN out of a
  neighbouring allocation, a fault away from crashing). `tests/fuzz_dense.py`'s
  stray-read probe makes it deterministic; the fix pads and zeros the scratch and
  costs nothing at the benchmark shapes.

## What the next session should do

`docs/NEXT.md`, in order: re-verify x86 (needs AWS), re-run ARM on the 8-core
`m9g.2xlarge` target the scorecard is defined on, and only then decide whether
the D=64 rows are worth another attempt at the register-lean softmax — the one
structural route left, whose tax (+2 instructions/row-key of accumulator
staging against a 23% prize) is now measured rather than guessed.
