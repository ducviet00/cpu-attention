# Where the campaign stands

This file is the honest, measured state. `docs/PROTOCOL.md` says how numbers are
taken; `docs/BENCHMARKS.md` holds the full tables; this is the summary and the
log of what actually moved.

## The one number

`ours / baseline < 1`, for both baselines, on both platforms, for all 22 rows of
`tests/bench_vs_refs.py`. Below 1 is a win.

## Two rules for optimization claims

**Show spare *issue* slots, not spare matmul cycles.** A microbenchmark of pure
`bfmmla`s against pure `fmla`s says the machine co-issues them (8 + 8 in 3.99
cycles). It does not follow that a softmax can be hidden inside a matmul loop:
the Q@K^T body is 26 instructions per 16 `bfmmla`, i.e. 3.25 IPC for 8 cycles of
matmul against a machine that sustains ~3.5, so that loop is already issue-bound
and has nothing to give. Three co-issue builds (all four row pairs, two, and the
half-exp variant) measured +5%, +3.7% and +0.7% cycles/row-key respectively —
monotone in the interleaved fraction, converging on the baseline, which is the
signature of an overhead with no upside. Before proposing any fusion, count the
instructions *inside* the loop that is supposed to host the work.

**Measure the kernel's own counters by subtracting two iteration counts, never
from one process run.** `perf stat` over a python driver is mostly torch import
(82 M instructions, 40 M cycles at any iteration count you care to name); run the
same driver at `N` and `M` iterations and divide the difference by `N - M`. On
the box, also beware that `P=~/prof/one.py` inside a loop leaves a literal `~`
(unquoted `~/prof/one.py` is what expands) and silently measures a 12 ms
"failed to open file" instead — the numbers look plausible and are garbage.

## Verified tables, 2026-10-22

`scripts/verify.sh` on each box: clean rebuild from scratch, both parity suites
(`test_dense` 49/49, `test_flash_api`, and on x86 `test_amx` for both dtypes),
then the full 22-row matrix with the reference kernel in one process. Ratios are
`ours / baseline`; below 1 wins. Ours includes the K/V pack, vLLM's timed region
excludes theirs, which is the campaign's stated asymmetry.

### x86 — Xeon 6975P (m8i, 2 physical cores), bf16 via AMX, fp32 via 3-term AMX

| | beats SDPA | beats vLLM |
|---|---|---|
| **VEC reference** (bf16 + fp32) | **22 / 22** | **21 / 22** |
| AMX reference (bf16 only; vLLM's AMX path has no fp32) | **11 / 11** | 4 / 11 |

fp32 is 0.33-0.54 of SDPA and 0.54-1.18 of VEC; bf16 is 0.49-0.80 of SDPA and
0.21-0.36 of VEC. The single VEC loss is `128:8:64 fp32` at 1.18. Against
vLLM's **AMX** build the bf16 losses are all short/medium S (1.03-1.62) and all
of them are the pack: 10-25% of our call is K/V packing that their timed region
does not contain, and at S=128 the whole call is 60 us.

### ARM — Graviton 5 (m9g, 4 cores), NEON + `bfmmla`

| | beats SDPA | beats vLLM |
|---|---|---|
| NEON reference | **22 / 22** | 4 / 22 |

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 0.64 | 1.40 | 0.35 | 1.09 |
| S=512  H=8  D=64  | 0.50 | 1.17 | 0.35 | 1.05 |
| S=1024 H=8  D=64  | 0.66 | 1.05 | 0.50 | 1.03 |
| S=2048 H=8  D=64  | 0.77 | 1.06 | 0.63 | 1.01 |
| S=4096 H=8  D=64  | 0.83 | 1.08 | 0.74 | 1.05 |
| S=256  H=16 D=128 | 0.55 | 1.26 | 0.46 | 0.95 |
| S=1024 H=16 D=128 | 0.67 | 1.12 | 0.60 | 0.90 |
| S=2048 H=16 D=128 | 0.77 | 1.13 | 0.75 | 0.97 |
| S=4096 H=16 D=128 | 0.85 | 1.14 | 0.91 | 1.03 |
| S=512  H=32 D=128 | 0.49 | 1.21 | 0.44 | 0.99 |
| S=1024 H=8  D=32  | 0.65 | 1.03 | 0.48 | 1.02 |

Every row is at or below 1.40 of vLLM's kernel, sixteen of 22 are within 15%,
and the first four have crossed. The kernel is now at vLLM's instruction count
for bf16 (20.99 vs 20.88 per row-key) and behind on IPC (3.62 vs 4.07), against
measured ceilings of 4.0 (bf16) and 9.4 (fp32) cycles per row-key.

### Where the remaining work is

1. **ARM bf16** (1.06-1.42) and **ARM fp32 at short S** (1.12-1.18). The ARM
   kernel just moved to vLLM's *structure* (L1 logits tile, one softmax per
   64-key tile, 8 rows per work item), which took bf16 from 1.39-2.35 to
   1.07-1.32 and fp32 from 1.19-1.60 to 0.96-1.19. Perf puts us at vLLM's
   instruction count for bf16 and behind on IPC — and the 2026-10-22 IPC study
   (NOTES-ARM.md, "The 0.5 IPC", counters taken by subtracting two iteration
   counts rather than from one process run) says that residue is **not a
   removable instruction**: measured at 2048:8:64 we execute *fewer* instructions
   than vLLM (20.10 vs 20.96 per row-key) and take more cycles (5.645 vs 5.319,
   IPC 3.56 vs 3.94), the inner loops are identical in shape (1.44 vs 1.47
   instructions per `bfmmla`, both spill-free), and the one instruction class we
   have and they do not (the two constant-seed `mov`s per exp) is *cheaper* than
   the arithmetic that would replace it: removing them measured 2.5% slower,
   because the softmax binds on FP-pipe slots, not on instruction count. What is
   left is a diffuse FP-pipe/scheduling residual; the two remaining structural
   levers (fp16 exp like their `fast_exp_f16`, and hiding the softmax in the
   matmul) have both been built and both lost. The next measured items are the
   two fork-joins (18% of an S=128 call) and the pack at short S.
2. **x86 bf16 at short/medium S against vLLM's AMX build** (1.03-1.62). The pack
   is memory-bound at 88% of memcpy, so this needs the kernel, not the pack.
3. **x86 `128:8:64 fp32`** (1.18 against VEC), the only VEC row left.

## The laptop (Ryzen 7840H, 8 cores, AVX-512 + avx512_bf16, no AMX), 2026-10-21
