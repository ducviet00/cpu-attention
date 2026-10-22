# NOTES-FAST — per-head dispatch, fp32 instruction diet, bf16 short S

Session scope: **fp32 at every shape** (make the 1.00-1.05 rows comfortable wins)
and **bf16 short S** (128/256/512). Worktree `arm-fast`, remote dir `~/arm-b` on
the shared m9g.2xlarge (8 cores, 48 MB L3). Every number below is measured on
that box, interleaved in one process or through `tests/ab_two_trees.py`; the
commands are in each section.

## 0. What shipped

| change | where | effect |
|---|---|---|
| **per-head dispatch**: one thread packs a kv head and then sweeps *all* of that head's query blocks, one region, **no internal barrier** | `dense.mojo` (`HeadDispatch`, `_gomp_worker_head`), `_kernel.py` (`mode += 4`) | **1.00-1.18x on every row of the matrix**, both dtypes |
| fp32 exp pass: two vectors per iteration, running sums in registers | `_dense_rows_tile_f32` | part of 1.01-1.06x fp32 |
| fp32 Q@K^T running max: `max()` (fmaxnm) instead of `lt`+`select` (fcmgt+bsl) | same | same |
| qt guard: reject a `qt` the compiled kernel cannot render | `_run_shape`, `_fused_shape` | correctness (see §3.1) |
| fp32 V copy gate: `V >= LLC/3` instead of `V >= LLC/2` | `_kernel._dense_plan` | 1.02-1.07x at 2048:16:128 fp32 |
| LLC size in the plan key | `_kernel.py` | a patched/odd box cannot reuse the other layout's plan |

## 1. The per-head dispatch is the whole short-S story

The integrator's `docs/FINDINGS.md` §3: at **one thread** we are exactly at vLLM
parity on 128:8:64 bf16 (146 us each) while at 8 threads we get 4.3x against
their 6.3x, and `perf record` puts **29% of the call in
`gomp_team_barrier_wait_end` + `gomp_barrier_wait_end`**. So the 1.53x on that row
was parallel efficiency, not the kernel.

The old dispatcher handed out (head, q-pair) items: every thread touched every
head's K/V, and the pack and the attention were separated by a `GOMP_barrier`.
The fix is one item per (batch, head): thread `t` packs head `t`'s K/V and then
runs head `t`'s query blocks. The pack and the attention then touch the same
bytes, so **nothing crosses threads and the barrier disappears**; balancing is by
construction because every head's causal work is identical.

Gated on three things the caller checks (`_kernel._dense_plan`): MHA
(`num_heads == num_kv_heads`, so the pack item and the attention item are the
same index), `batch*heads % workers == 0`, and `batch*heads >= workers`.
Otherwise the old round robin still runs, so nothing regresses on a geometry the
scheme cannot balance.

`tests/bench_head.py` (in-process interleaved, 8 threads, 5 rounds), head/base:

| shape | bf16 | fp32 |
|---|---:|---:|
| 128:8:64 | 1.043 | 1.002 |
| 512:8:64 | 1.039 | 1.080 |
| 1024:8:64 | 1.054 | 1.053 |
| 2048:8:64 | 1.028 | 1.000 |
| 4096:8:64 | 0.994 | 1.037 |
| 256:16:128 | 1.126 | 1.181 |
| 1024:16:128 | 1.096 | 1.093 |
| 2048:16:128 | 1.087 | 1.044 |
| 4096:16:128 | 1.045 | 1.058 |
| 512:32:128 | 1.166 | 1.129 |
| 1024:8:32 | 1.026 | 1.027 |

Output is **bit-identical** between the two dispatches (`max|diff|` vs SDPA equal
to the last bit on every row), which is what a pure schedule change must be.

```bash
REMOTE_DIR=arm-b scripts/box_sync.sh arm64 ubuntu@52.77.215.13 .
scripts/box_run.sh ubuntu@52.77.215.13 'cd ~/arm-b && .venv/bin/python tests/bench_head.py \
  --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa neon --dtype both --rounds 5'
```

### 1.1 Falsified: "long S wants the old sharing"

I expected per-head to lose at 4096, because the round robin keeps all 8 threads
on one head at a time and their K/V prefixes overlap in L3, while per-head gives
each thread its own 1-2 MB stream. Measured: 4096:8:64 fp32 1.037x, 4096:16:128
fp32 1.058x, 4096:16:128 bf16 1.045x, 4096:8:64 bf16 0.994x. Sharing does not
buy back the barrier and the balance. `_HEAD_MAXK` exists anyway, so a box where
one head's K/V no longer fits can be swept (`0` = off).

## 2. fp32: where the instructions were, and where they are not

Per row-key the fp32 tile kernel issues ~50 instructions at D=64: Q@K^T 21.5
(16 fma), P@V 22 (16 fma), exp 6-8, ~1 of tile bookkeeping. The `fmla` port
floor is 32/3.3 = 9.7 cycles/row-key; we measure ~13.1.

Found by reading the emitted asm (`mojo build --emit asm --target-triple
aarch64-unknown-linux-gnu --target-cpu neoverse-v3`), not by guessing:

* the exp loop kept its running sum in **memory**: `sumv[(k // LANES) % 2]` is a
  runtime index, so each 4-key iteration had `and`/`lsl` to rebuild it plus an
  `ldr`/`str` of the sum, and two `mov`s to save the polynomial constants it
  clobbered -- **32 instructions per 4 keys for 17 of arithmetic**. Two vectors
  per iteration with pointer addresses and register sums: **24 per 8 keys**.
* the running max used `lt()`+`select()` (fcmgt + bsl) where `max()` is one
  fmaxnm: 1429 -> 1414 instructions per 8-key sub-block.

Together 1.01-1.06x on the fp32 rows (`tests/ab_two_trees.py`, 5 rounds,
`~/arm-b-base` = `3fe12f2` vs this tree), bf16 unchanged (the change is fp32-only).

### 2.1 Falsified: 32-byte operand loads in the Q@K^T loop

The 8-key K pair is already one `ldp`; only the q pair is two `ldr`. Making the
accumulator 8 lanes wide (one 32-byte load for the K sub-block *and* one for the
8 rows) needs 16 registers for the accumulators plus 16 for the 8-row running
max, and the running max goes to the stack: the loop **grew to 1424-1432
instructions** from 1414 with 12 `ldr/str` pairs per sub-block on `[sp]`, and the
32-byte loads were not merged into `ldp` (they sit in a loop with stores that the
compiler cannot prove disjoint). `mojo build --emit asm ... | grep -c ldp` on the
sub-block loop: 4 before and after. Reverted.

A micro-benchmark (`/tmp/opencode/ld8.mojo`) does show LLVM emitting
`ldp q, q` for `unsafe_load[width=8]` and keeping the lane-indexed
`fmla v.4s, v.4s, vN.s[k]` form -- so the instruction is available, the *register
file* is what refuses it.

### 2.2 Falsified: wider row blocks (qt) for fp32

In-process interleaved (`tests/bench_qt.py --dtype fp32 --qts 8,12`), medians of
5: qt=12 is **1.05-1.12x slower** than qt=8 at 2048:8:64 (9188 vs 8183 us),
4096:8:64 (37522 vs 34564) and 4096:16:128 (147865 vs 140872). qt=8 stands. qt=4
takes a different (older) kernel and is not comparable.

## 3. Two correctness landmines, both closed

### 3.1 `CPU_ATTENTION_QT` could silently drop rows

The ARM fp32 tile kernel walks query rows `LANES` (=4) at a time and the bf16 one
walks row *pairs*, so qt=6 (which the old online-softmax kernel wanted, and which
`_qt_for`'s docstring still calls a free 1.09-1.11x) leaves rows 4-5 of every
block at whatever was in the scratch. It measures **1.22x "faster"** than qt=8 and
comes out with max|diff| **1.5-2.1** vs SDPA -- a sweep would have shipped it.
`_run_shape`/`_fused_shape` now raise. Verified on the box:

```
qt=4 ok d=3.6e-07 | qt=6 raised: ... must be a multiple of the vector width ...
qt=8 ok d=1.6e-06 | qt=12 ok d=1.6e-06
```

### 3.2 The V-copy gate was calibrated on another box

`2 * V_bytes >= LLC` (half) comes from a 32 MB-L3 laptop. On this 48 MB box it
leaves 2048:16:128 fp32 (V = 16.8 MB) reading V in place; forcing the copy there
measured **1.067x, 1.021x, 1.043x** in three interleaved runs
(`tests/bench_f32_vpack.py`), and every smaller shape stayed inside noise. A
third of the LLC fires at exactly the shapes the half rule did on the laptop,
plus that one.

### 3.3 Measurement trap: clearing the plan cache inside a timing loop

`bench_f32_vpack.py`'s first version called `clear_scratch()` per timed call to
re-evaluate the gate. That re-`torch.empty`s multi-megabyte scratch, so the page
faults land in the measurement: two arms that must be identical (auto and
"never copy", both of which take the no-copy path) differed by **19%** at
512:32:128. Now the cache is cleared before the warmup only, and the gate is part
of the plan key so the arms cannot cross. Every other row is then within 1.7%
between those two arms, which is the noise floor to quote.

## 4. bf16 short S: what the 34 us call is made of, and what is left

128:8:64 bf16, after the dispatch fix: **27 us against vLLM's 22** (o/vllm
1.22-1.25, was 1.53). The composition, from the integrator's phase split plus my
own measurement of the region entry:

| term | us | share |
|---|---:|---:|
| K/V pack (inside the region, per head) | 2.9 | 11% |
| host path (Python + output alloc + 42-arg marshal) | 1.5 | 6% |
| one `GOMP_parallel` region entry (~2.4 us warm, AGENTS §7) | 2.4 | 9% |
| row kernel | ~20 | 74% |

vLLM's 22 us is attention only -- its `reshape_and_cache` is staged before the
timer, which is the scorecard's contract -- and at one thread the two row kernels
are equal. So the remaining 5 us is exactly the pack + host + region entry that
we are charged for and they are not, plus a kernel that is at parity.
**Conclusion: this row is structural at ~1.2 with this contract.** The levers
that would move it are all falsified elsewhere (`docs/CEILINGS.md`: the pack is
at 88% of a memcpy; the region entry is the cheapest pool wake available; the
host path is the marshal floor).

What *did* move on the short-S rows, all from the dispatch:

| row | before | after |
|---|---:|---:|
| 128:8:64 bf16 | 1.53 | **1.22-1.25** |
| 256:16:128 bf16 | 1.25-1.30 | **1.05-1.08** |
| 512:8:64 bf16 | 1.08-1.13 | **0.98-0.99** |
| 512:32:128 bf16 | 1.22-1.23 | **1.04-1.05** |

## 5. Flag for whoever owns bf16 long S

2048:16:128 bf16 measured **1.16 in two consecutive runs** while 1024:16:128 is
1.02 and 4096:16:128 is 1.04-1.07. That non-monotonicity is not the dispatch
(per-head is 1.087x there, the largest long-S gain); it looks like the
`dcp`/tile choice at that exact shape. Not mine to fix, but it is 12 points on
one row.

## 5b. The pack is 2% of a 512-key call, not 18% — and the 512:32:128 row

`tests/pack_only.py` (8 threads, this box, after the pack was re-measured):

| shape (bf16) | pack us | bytes moved | GB/s | share of the call |
|---|---:|---:|---:|---:|
| 128:8:64 | 2.41 | 0.52 MB | 218 | 9% of 26 us |
| 512:8:64 | **4.97** | 2.10 MB | **422** | **2.1% of 235 us** |
| 256:16:128 | 9.20 | 4.19 MB | 456 | 4% of 229 us |
| 512:32:128 | 34.6 | 16.8 MB | 485 | 2.2% of 1570 us |
| 2048:8:64 | 17.5 | 8.39 MB | 478 | 0.5% |
| 4096:8:64 | 33.9 | 16.8 MB | 495 | 0.25% |

So the pack is **not** why 512:8:64 bf16 sits at 0.98-1.02: at 422-495 GB/s it is
2% of the call, and `bench_phases.py`'s larger number carries the `NO_FUSE`
fork-join (docs/NEXT §3.3). That row is kernel parity at D=64/512 keys.

**512:32:128 fp32** (the one band that crossed 1.0) is a *layout/noise* row, not
a dispatch row. In one process, 15 interleaved rounds, spread under 1% per arm:

| arm | median | min..max |
|---|---:|---|
| V in place, qt=8 | 4543 us | 4518..4562 |
| V copied (`vpack` forced), qt=8 | 4585 | 4551..4624 |
| V in place, qt=12 | 5137 | 5116..5170 |

i.e. 0.93 against vLLM *in isolation*, the V copy is neutral (0.9% worse) and
qt=12 is 13% worse. Inside the scorecard (where SDPA and vLLM also run) the same
row measures 4785-4888 = 0.98-1.01, and across processes it has ranged
4530..5423. We touch ~25 MB of K/V per call (pack write + pack read + V read)
against vLLM's staged 16.8 MB, so the scorecard's cache churn costs us more than
it costs them; that is the contract, not the kernel.

What I did do: the per-head order is capped at **2 heads per thread for fp32**
(`_HEAD_RATIO`, `CPU_ATTENTION_HEAD_RATIO`). At 4 heads per thread (512:32:128 is
the only such row) fp32 loses to the paired order *and* is unstable (4065..5350
within one process, 4802 median vs 4659), while bf16 at the same shape wants it
(1.17x, 1547..1585) and keeps it. So that row now takes the paired order:
0.98-1.01 instead of 0.94-1.12. Still not comfortable, and I do not think this
kernel shape can make it so (see §7.1 for the one structural idea left).

## 6. State of the matrix after this tree (2 runs, `tests/bench_vs_refs.py`)

`scripts/band.py` over two runs of the final tree (with the 2-heads/thread cap):
**22/22 vs SDPA, 14/22 vs vLLM, 1 on the line** (was 5 wins / 6 on the line / 11
losses at the start of the session). The single "on the line" row is
4096:8:64 bf16 (0.99-1.00), and 6 of the 7 losses are bf16 D=128 rows owned by
another agent.

**fp32 is 11/11 below 1 in both runs**, with 5 rows at ≤0.92:

| shape | before | after |
|---|---:|---:|
| 128:8:64 | 1.02-1.18 | **0.93-0.96** |
| 512:8:64 | 1.02-1.04 | **0.91-0.92** |
| 1024:8:64 | 1.00-1.01 | **0.92-0.94** |
| 2048:8:64 | 1.02-1.03 | **0.97-0.98** |
| 4096:8:64 | 0.98-1.00 | **0.97-0.98** |
| 256:16:128 | 0.96-0.97 | **0.80** |
| 1024:16:128 | 0.93-0.94 | **0.95** |
| 2048:16:128 | 0.95-1.01 | **0.88-0.89** |
| 4096:16:128 | 1.05-1.06 | **0.99** |
| 512:32:128 | 0.95-0.96 then 0.94-1.12 | **0.96-0.97** (paired order, §5b) |
| 1024:8:32 | 1.00-1.01 | **0.91** |

bf16 rows in my scope: 128:8:64 1.53 -> **1.13-1.15**; 256:16:128 1.25-1.30 ->
**1.09**; 512:8:64 1.08-1.13 -> **1.00-1.01**; 512:32:128 1.22-1.23 -> **1.05**;
1024:8:32 1.00 -> **0.97-0.98**; 1024:8:64 1.02 -> **0.96-0.97**; 2048:8:64 1.03
-> **0.99**; 4096:8:64 1.02 -> **0.99-1.00**.

Parity: on the box, `test_dense` 64/64 at 1/2/4/8 workers (and with the head
dispatch off, and with the ratio cap off), `test_flash_api` all passed,
`fuzz_dense` 200/200 with the stray-read probe 24/24, twice. Every row's `d=`
column is inside tolerance.

### 6b. x86 is not regressed (laptop, `ab_two_trees`, 5 rounds)

| shape | fp32 | bf16 |
|---|---|---|
| 512:8:64 | 1.05x faster | 1.00x |
| 256:16:128 | 1.07x faster | 1.02x |
| 1024:16:128 | 0.99x | 1.02x |

The dispatch change is ISA-agnostic and helps or ties there too.

**One local caveat**: `tests/test_dense.py` segfaults on this laptop (exit 139,
before it prints anything, at 1 and at 8 workers) for **both** the base tree and
this one, after the shared dev venv was recreated today; `ab_two_trees` runs fine
on the same build. Not a regression from this work, but the x86 verification in
`docs/NEXT.md` §1 has to run on a server box, not here.

## 8. The S<=256 kernel (`dense_small.mojo`), and why the pack stays

The integrator's brief: 128:8:64 bf16 is 1.17-1.19, our whole call is 26.5 us =
host 4.5 + region 22, and the region contains a **pack of 2.4 us** that vLLM does
not pay inside its timer. Attention-only we already win (19.6 vs their 21).

### 8.1 Falsified: dropping the pack and reading k/v natively

Measured with an asm probe (`/tmp/opencode/operands.mojo`, one loop with one
operand source swapped):

| loop | instructions | bfmmla | per bfmmla |
|---|---:|---:|---:|
| P@V, packed V (`ldr q` per (4 keys x 2 dims)) | 41 | 16 | 2.6 |
| P@V, native V (4 keys x 2 dims, rows `v_st` apart) | 70 | 16 | **4.4** |
| Q@K^T, packed K^T (one 16-byte load per operand) | 588 | 256 | 2.3 |
| Q@K^T, native K (two 8-byte loads + join) | 794 | 256 | **3.1** |

So a native-layout operand costs +0.8 (K) and +1.8 (V) instructions per bfmmla,
i.e. ~10 instructions per row-key against a 2.4 us pack for the whole 128-key
call. The layouts the pack produces are exactly what makes every bfmmla operand a
single load; that is the trade, and the pack wins. (A `bfdot` kernel reads both
k/v natively, but bfdot is 21.6 MAC/cycle against bfmmla's 32 and the matmul is
already ~10 us of the 19.6 us kernel at this size, so it loses by more.)

### 8.2 What the per-query-block cost actually is (ablation, 128:8:64 bf16)

| removed | us | delta |
|---|---:|---:|
| nothing (shipped) | 24.05 | - |
| Q panel build | 22.47 | **-6.6%** |
| accumulator zeroing | 23.42 | -2.6% |
| output epilogue | 12.07 | degenerate (kills the accumulator's live-out) |

A non-causal run needs no work rounding -- every block scores exactly 128 keys --
and it costs 35.0 us for 131072 row-keys against the causal 24.4 us for 66048, so
fitting `F + keys * 8 * m` gives **m = 5.6 cycles/row-key (the long-S rate) and
F = 1391 cycles of fixed cost per query block**: 28% of a 128-key call. Of that,
the Q panel is ~6.6% of the call and the accumulator zeroing 2.6%; the rest is
drain/latency at 16 blocks per head, not instructions.

### 8.3 Shipped: round the key coverage to 8 keys, not 16

`nkp` was rounded to 8 key *pairs* because the exp pass consumes four key groups
per iteration. At S=128 the average block only has 68 keys, so 12 of the 80 it
scored were keys no row may see. A two-group tail loop in the exp pass lets the
tile cover `ceil(jn/8)*8` instead. `tests/bench_small.py`, in-process, 21 rounds,
min of the per-round times, two sessions:

| shape | shipped | small | ratio | o/vllm shipped -> small |
|---|---:|---:|---:|---|
| 128:8:64 bf16 | 24.6, 24.7 | 23.3, 24.5 | **1.051, 1.008** | 1.12 -> 1.07 |
| 256:16:128 bf16 (path off by default) | 208.7 | 201.5 | **1.068** | 1.01 -> 0.95 |
| 512:8:64 bf16 (path off) | 226.4 | — | 1.000 | — |
| 1024:16:128 bf16 | 2897.8 | 2849.8 | 1.017 | — |

Same-code control arms in the same harness differ by 0.4-1.7% (min-based), so
1.05 is a real win and the 1.017 at 1024 is at the noise floor. Output is
bit-identical (`bench_small.py` asserts equality, not tolerance). Caveat from the
shared box: one session's identical arms differed by 6.6%, so quote the min and
re-run.

It is in a new file because the bf16 D=128 tile is being rewritten in
`dense.mojo`; the path is `seq_k <= 256`, MHA, bf16 `bfmmla`, **head_dim < 128**
(the D=128 rows are the other agent's -- the 256:16:128 gain above is measured
but deliberately not enabled). `CPU_ATTENTION_SMALL_MAXK` / `set_small_maxk()`
raise or lower the threshold; `0` disables the path.

**The 16-key rounding is not the row's problem, though**: at 128 keys the fixed
cost per query block is 28% of the kernel (§8.2), and the only way to collect
that is a kernel that keeps a *head* resident and overlaps block g+1's prologue
with block g's epilogue -- a restructure, not a bound. That is what I would do
next, together with the Q panel's 6.6%.

Skipping the first tile's accumulator zeroing (the other obvious per-block win)
measured **4% slower**: the `t0 == 0` test lands inside the unrolled P@V chunk
loop. Not done.



1. **The V d-chunk layout** (`[d-chunk][key][dc]`): the P@V sweep makes
   `NCH = head_dim/dc` passes over the tile and touches 32 B of every 256 B (D=64)
   or 512 B (D=128) key row, so each 64-byte line is fetched twice and its other
   half is used `NCH-1` passes later, long after the line left L1. This is also
   the only candidate left for 512:32:128 fp32: we touch ~25 MB of K/V per call
   (K^T write + K^T read + V read) against vLLM's staged 16.8 MB, and the copy
   that would fix the stride is neutral because it pays the same traffic. A
   *blocked* P@V (one pass over V, all chunks accumulated) is the version that
   would pay. `docs/NEXT.md` §3.2 describes the same thing for bf16 -- whoever
   does it should do both kernels in one go, or split the file.
2. **The D=128 bf16 `dcp`** (see §5): 12 points sitting on one row.
3. Not worth it, measured: `qt` beyond 8 (1.05-1.13x slower at four shapes), 32-byte
   operand loads (register wall), the per-head scheme at long S (already won), the
   pack (2% of a 512-key call, 422-495 GB/s), the V copy at 512:32:128 (neutral),
   and the pack/host/region terms at 128 keys (all at their floors).
