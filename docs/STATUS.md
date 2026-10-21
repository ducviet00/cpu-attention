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

## Verified tables, 2026-10-22 (final for this session)

Produced by `scripts/verify.sh <arch> <host> <tree> <label>`: sync, **clean
rebuild from scratch** (the `.so` is deleted first, so a stale build cannot pass
for a fresh one), both parity suites, then the full 22-row matrix with the
reference kernel in one process. `ROUNDS=21` and a 1.2 s warmup: these boxes idle
at a fraction of their clock and row 1 otherwise reports the idle number for
*every* variant, ours and the baselines alike.

Ratios are `ours / baseline`; below 1 wins. Ours includes the K/V pack; vLLM's
timed region does not include their staging. That asymmetry is the campaign's
stated contract.

### x86 — Xeon 6975P (m8i, 2 physical cores)

bf16 runs on AMX (`tdpbf16ps`, 32 query rows per work item, 256-key softmax
super-tile); fp32 runs on AMX as well, with every operand split into bf16 hi/lo
halves and three products per phase. Last verified at `164e8e6`; `main` since
then changed only the shared Python dispatch path (one `.stride()` per tensor,
laptop A/B 1.04x at S=128 and 1.01x above it, parity green) and the ARM kernel,
so the x86 kernel and its dispatch are untouched by it. **Not re-verified after
`164e8e6`**: the spot box was reclaimed at the end of the session and the AWS CLI
session had expired, so `docs/NEXT.md` starts there.

| | beats SDPA | beats vLLM-VEC |
|---|---|---|
| bf16 (11 rows) | **11 / 11** | **11 / 11** |
| fp32 (11 rows) | **11 / 11** | **11 / 11** |

bf16 is 0.19-0.51 of vLLM's VEC kernel and 0.49-0.80 of SDPA; fp32 is 0.51-0.96
of VEC and 0.33-0.96 of SDPA. Against vLLM's **own AMX build** (bf16 only: their
AMX path has no fp32) it is 6/11, and the five losses are all short/medium S
where our call pays a cold read of K/V plus the pack inside a 52-60 µs call while
their timed region pays neither.

### ARM — Graviton 5 (m9g, 4 cores), NEON + `bfmmla`

Verified on `main` at `25cb62e`, `ROUNDS=21`.

| | beats SDPA | beats vLLM-NEON |
|---|---|---|
| NEON reference | **22 / 22** | **2-5 / 22** (a sample, see below) |

**The `N/22` count against vLLM should never be quoted without this caveat.** Four
runs of the same binary gave 5, 2, 3 and 4 fp32 rows below 1.00, with fp32
`o/vllm` at 0.89-1.11 every time: those rows sit on the parity line and a 3%
shift moves them across it. bf16 is stable across all four runs at 1.02-1.32 and
never below 1.00. So the honest summary is **22/22 against SDPA, bf16 always
above 1 (1.02-1.32), fp32 at parity (0.89-1.11)**.

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 0.59 | 1.32 | 0.35 | 1.09 |
| S=512  H=8  D=64  | 0.49 | 1.13 | 0.35 | 1.04 |
| S=1024 H=8  D=64  | 0.66 | 1.06 | 0.50 | 0.99 |
| S=2048 H=8  D=64  | 0.77 | 1.08 | 0.63 | 1.02 |
| S=4096 H=8  D=64  | 0.83 | 1.07 | 0.74 | 1.05 |
| S=256  H=16 D=128 | 0.55 | 1.28 | 0.45 | 0.95 |
| S=1024 H=16 D=128 | 0.67 | 1.11 | 0.60 | 0.90 |
| S=2048 H=16 D=128 | 0.77 | 1.13 | 0.75 | 0.96 |
| S=4096 H=16 D=128 | 0.85 | 1.14 | 0.91 | 1.03 |
| S=512  H=32 D=128 | 0.48 | 1.20 | 0.44 | 0.98 |
| S=1024 H=8  D=32  | 0.65 | 1.02 | 0.48 | 1.03 |

Sixteen rows sit between 1.00 and 1.32. The gap is now explained rather than
mysterious, and the explanation is narrow: **at matched instruction counts we run
a slightly worse schedule.** Per row-key, counters taken by subtracting two
iteration counts give ours 20.10 instructions / 5.645 cycles / 3.56 IPC against
vLLM's 20.96 / 5.319 / 3.94 — 4% fewer instructions, 6% more cycles. A static
diff of both inner loops shows identical density (1.44 bfmmla per instruction
against 1.47) and no spills on either side; stall attribution finds no single
site. The one asymmetry (two constant-seed `mov` per exp) is *cheap*: replacing
it with a multiply made the kernel 2.5% slower, because the softmax binds on
FP-pipe slots, not instruction count. That is also why the fp16 softmax lost
(10% fewer instructions, 1.7% slower — the conversions land on the same pipes)
and why co-issuing the exp with the next tile's Q@K^T lost (that body is 26
instructions per 16 bfmmla, i.e. 3.25 IPC against a machine that sustains 3.5:
there are no spare issue slots to hide it in).

