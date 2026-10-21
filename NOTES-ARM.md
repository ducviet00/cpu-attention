# NOTES-ARM.md — ARM (Graviton 5 / Neoverse-V3) work log

Ratios are `ours / baseline` (below 1 = we win). Kernel numbers are *thread*-
cycles per row-key from `scripts/prof/arm/pk.py` (differential `perf stat`), on
the m9g.xlarge dev box (4 cores). **Check the box is idle first** (`uptime`,
`ps aux`): a stray process halved a whole session's measurements once and
produced a fake 2x.

## Hardware ceilings (measured, `scripts/prof/arm/one_fma.c`, `scripts/prof/arm/peak.c`)

3.26 GHz. bf16 `bfmmla` = **2.0/cycle** (207 GFLOP/s/core); fp32 `fmla v.4s` =
**3.3/cycle** (88 GFLOP/s/core); `bfdot` = 2.7/cycle. So at head_dim 64:
* bf16 floor: 8 bfmmla/row-key / 2 = **4.0 cycles/row-key**
* fp32 floor: 16+16 FMA/row-key / 3.3 = **9.4 cycles/row-key**

## The reference point

`perf stat` on vLLM's `csrc/cpu` NEON kernel, S=2048 H=8 D=64 causal:

| kernel | cycles/rk | instr/rk | IPC | L1 refill/rk | backend stall/rk |
|---|---:|---:|---:|---:|---:|
| vLLM bf16 | **5.13** | 20.88 | 4.07 | 0.15 | 1.00 |
| ours bf16 v1 (16-key online softmax, 4 rows) | 7.31 | 23.15 | 3.17 | 0.86 | 2.25 |
| ours bf16 v2 (tile logits in L1, 8 rows) | **6.07** | 21.24 | 3.50 | 0.33 | 1.46 |
| vLLM fp32 | **13.76** | 51.63 | 3.75 | 0.63 | 3.37 |
| ours fp32 v1 (`_dense_rows_mem`, 6 rows) | 18.25 | 72.70 | 3.98 | 1.57 | 4.86 |
| ours fp32 v2 (tile logits in L1, 8 rows) | **15.50** | 57.15 | 3.69 | 1.34 | 3.91 |

Both dtypes spend 8 (bf16) / 32 (fp32) instructions on matmul, so the gap was
never algorithmic: it was instructions and IPC. **The old "ARM softmax is 40% of
the call and vLLM's is 9%" story does not survive the instruction counts**:
vLLM's whole non-matmul budget is 12.9 (bf16) / 19.6 (fp32) instructions per
row-key, and they also pay a mask (fcmgt+bsl+bic = 7% of their samples) and
address adds (9%).

## What v2 is (`_dense_rows_tile` bf16, `_dense_rows_tile_f32` fp32)

vLLM's structure instead of flash-decoding's:

* the key tile's logits are staged in **L1** (`stack_allocation`, runtime
  indices, so LLVM's SROA cannot forward them back into registers — the reason
  the previous campaign's "L1 logits" probe measured 0%);
* **one** max/sum/accumulator-rescale per 64-key tile instead of one per 16 keys;
* **8 rows** per work item instead of 4 — rows are the only thing the K/V read
  is amortised over, and they only became affordable once the logits left the
  register file;
* K^T **blocked by key tile** (`[tile][dim group][key pair][8]`, 8 KB) so the
  Q@K^T sweep walks it linearly (it was 16 scattered lines per sub-block, a 12%
  branch bubble at S=2048), and V packed `[key group][dim pair][8]` so P@V's
  inner loop has no address arithmetic (8 of 35 instructions);
* for fp32, Q transposed to `[head_dim][qt]` once per work item so one 4-lane
  load broadcasts four rows into a lane-indexed `fmla`, and P@V walks 4-key
  groups reading each row's probability *vector* once per head-dim chunk (v1
  re-read a scalar probability per row per key per chunk = 8 scalar loads per
  row-key, 16% of its cycles);
* **only the keys that exist**: the Q@K^T sub-block loop, the exp loop and P@V
  all stop at the causal limit of the block's last row, so masked keys cost
  nothing. Worth 1.5x at S=128, 1.3x at S=256, 1.2x at S=512 (noise at 2048).

### Measured increments, S=2048 H=8 D=64 bf16, thread-cycles per row-key

| change | cycles/rk | instr/rk |
|---|---:|---:|
| v1 (baseline) | 7.31 | 23.15 |
| v2 skeleton (tile logits + 8 rows) | 6.68 | 22.61 |
| + K blocked by tile, runtime exp loop | 6.73 | 22.87 |
| + panel build as 2 converts + join, shared zero, exp loop unrolled x2 | 6.22 | 22.33 |
| + `roundeven` + degree-3 minimax exp | **6.07** | 21.24 |

fp32: 18.25 -> 15.50 cycles, 72.70 -> 57.15 instructions (same restructure).
The causal-limit skip then added 1.5x/1.3x/1.2x at S=128/256/512 on both dtypes.

## Ablations (the tool that decided where to spend effort)

* **exp -> `y*0.5`** (keeps the loads and the panel stores): 6.25 -> 4.37
  cycles/row-key, i.e. the polynomial was **30% of the call** and 5.1
  instructions/row-key. That is what motivated `roundeven` and the degree-3
  minimax fit.
* Old note that "the exp is 20% of the call" (from `_vexp` costing 22
  instructions) is consistent with this on ARM once the *chain latency* is
  counted, not just the instruction count.

* **Compile-time K strides + contiguous Q panels** (kt_blk is always 64, so the
  stride is folded; panels laid out `[dim group][row pair][8]` so one group's
  four load as two `ldp`s): neutral (6.07 -> 6.17 cycles, inside the noise).
  Kept because it removes the last runtime stride from the sweep, but it is not
  where the time is.
* **Key tile 128/256** (`TILE_KS`): the per-tile P@V accumulator round trip and
  the tile epilogue amortise with the tile size. bf16 D=64 cycles/row-key: 64
  keys 5.86, **128 keys 5.33**, 256 keys 5.66; fp32 D=64: 14.16 / 14.07 /
  **13.68**. So bf16 takes 128 and fp32 takes 256 (its logit buffer is half the
  size per key). Parity is fine because the K^T base is now recomputed per
  sub-block.
* **`TILE_KS = 128` with the old per-tile K base**: broke parity. A tile wider than the K block steps out of
  its key-pair run into the next head-dim group; the addressing is per sub-block
  now so it would work, but the expected gain (~1.5%) did not justify chasing
  the residual flakiness.

## The fp32 exp rewrite, and a trap worth remembering

Replacing the Cephes `_vexp` in the fp32 tile kernel with a width-generic
`_exp2_g` (magic-constant rounding + degree-5 minimax + exponent insert, log2(e)
folded into Q) is **15.14 -> 14.16 cycles/row-key, IPC 3.78 -> 3.93**.

It first looked like it cost accuracy: the fp32 output error went 1.2e-6 ->
1.1e-4 against a 2e-4 tolerance, and *raising the polynomial degree did nothing*
(degree 5, 7.5e-8 relative, still 1.1e-4). The polynomial was never the problem:

> **an error in the softmax `alpha` rescale does not cancel between the
> accumulator and the row sum**, because the terms added *after* the rescale are
> not scaled by it. It therefore accumulates over tiles (which is why the error
> grew with seq_k and looked like a polynomial problem). `alpha` was still using
> the degree-3 `_exp2_fast`; at degree 5 the error is back to 3.2e-6.

Corollary: `alpha` must stay a **4-lane** exp in the bf16 kernel -- the two rows
of a row pair have their own max, so collapsing it to a scalar silently gives
row 1 row 0's rescale factor (caught by parity: 33/49).

## The pack is NOT cold-read-bound on ARM (flush measurement)

`scripts/prof/arm/flush.py` walks a 64 MB buffer (L3 is 48 MB) between timed calls
and times the pack from *inside* the call, so hot and evicted are directly
comparable -- the x86 lane found 6.7x there and it changes the plan if it holds:

| 128:8:64 bf16 | hot | evicted | ratio |
|---|---:|---:|---:|
| pack | 8.27 us | 10.13 us | **1.23x** |
| full call | 55.5 us | 59.4 us | 1.07x |

512:8:64 pack 26.4 -> 28.6 (1.08x); 2048:8:64 107 -> 111 (1.04x). So the extra is
just the DRAM transfer of K/V (256 KB at S=128 ~= 1.9 us at 135 GB/s), and ~8 of
the 10 us at S=128 is **hot** work: the 4-byte gathers, at ~2 loads/cycle.

Consequences:
* "make the pack faster" is still the right lever on ARM (it is instruction
  bound, not bandwidth bound), unlike x86 where it is one cold read;
* a one-pass (fused pack) kernel would save the *write+read of the packed copy*
  (~4 us at S=128, 7%) but not the gathers, so it is not the 5x the x86 story
  would suggest;
* and the real matrix is only 7-8% worse than a hot loop at S=128, so the
  `bench_phases` numbers are not the lie they are on x86.

**Warning on tooling**: `pk.py`'s cycle deltas are unreliable at small S (a 44%
"improvement" at S=128 that wall-clock says is 2%). Use `flush.py` or
`bench_vs_refs.py` for short shapes; `perf stat` remains fine for instruction
counts at S>=1024.

## fp16 softmax: the arithmetic is faster, the kernel is not

NEON fp16 is 8 lanes per 128-bit vector, so the softmax's exp/sum/max should cost
half the instructions. Measured per-instruction throughput
(`scripts/prof/arm/f16perf.c`, 8 accumulators):

| | cycles / 8 instructions | per instruction |
|---|---:|---:|
| `fmla v.4s` (fp32) | 3.97 | 2.0/cycle |
| `fmla v.8h` (fp16) | 2.90 | 2.8/cycle |
| `fmul v.8h` | 1.98 | 4.0/cycle |
| `fadd v.8h` | 2.27 | 3.5/cycle |
| `fmaxnm v.8h` | 2.43 | 3.3/cycle |

So 1.4x per instruction and ~2.8x per element. Implemented (`_exp2_f16`: clamp,
`roundeven.v8f16`, degree-3 poly, `fcvtzs`+`shl`+`fmul`, all 8-lane) with the
panel order built in fp32 first, one `fcvtn`+`fcvtn2` to get the 8 lanes, a `+4`
headroom offset folded into the per-row `off` (so the fp16 denormal cliff sits at
2^-18 of the row max and the exponent field stays in [1,15]; the +4 cancels in
acc/sum), and the row sum accumulated in one fp16 vector, converted once per
tile.

Result: **instructions 20.87 -> 18.75 per row-key (-10%) and wall time +1.7%
(7248 -> 7344 us at S=2048; +3.8% at 1024:16:128)**. IPC 3.56 -> 3.28 and
`stall_backend` up: the fp32->fp16 conversion of the two argument halves and the
fp16->bf16 conversion of the panel sit on the critical path, and they cost more
than the halved polynomial saves. Parity was fine (7.8e-3 / 1.6e-2). Reverted.

If this is worth revisiting it needs the *conversions* removed, not the exp made
faster: i.e. keep the tiles in fp16 end to end (fp16 K/V panels via the matmul),
which is blocked because `bfmmla` takes bf16 operands and fp16 K/V would break
the range contract for general inputs.

## The short-S "tile waste" hypothesis: falsified, and what the cost actually is

The reasoning was that a 128-key tile makes every causal block at S=128 compute a
full tile, so ~2x the matmul and softmax. It is not true of this code: the
causal-limit skip (commit 659e66a) already bounds the sub-block loop *and* the
exp loop by `jn = loop_end - t0`, so the work is proportional to the keys that
exist. Tested anyway by rebuilding with `TILE_KS = 32`, which would have been a
large win at short S if the waste were real:

| | TILE_KS=128 | TILE_KS=32 |
|---|---:|---:|
| S=64:8:64 | 26.5 us | 27.2 |
| S=128:8:64 | 55.0 | **59.6** |
| S=512:8:64 | 506.7 | 588.0 |
| S=2048:8:64 | 7238 | 8475 |

Smaller tiles lose at *every* size (the per-tile P@V accumulator round trip and
epilogue are paid more often), so no adaptive-tile dispatch can help.

What the cost actually is: fitting kernel time = F + G*blocks + 0.4248*row_keys
over S=64..512 gives **F = 10.0 us per call** and G = 0.07 us per q-block, and
the model reproduces S=256 to 0.4 us. The per-call part is the whole story at
short S (47% of an S=64 call, 21% of S=128, 4% of S=512, 0.1% of S=2048), and it
is host-side: the extension call alone (40 args, one GOMP region, real work) is
**3.1 us**, `torch.empty_like` 0.66, and the rest was `_dense`'s Python -- 21
`Tensor.stride()` calls and rebuilding a 40-argument call. Caching the per-shape
call plan (keyed on `id(q), id(k), id(v)`, holding the tensors so the ids cannot
be reused) removes 0.6-0.8 us of it: S=8 10.1 -> 9.5, S=128 55.9 -> 55.1.

Note for anyone measuring: `flush.py` sets `CPU_ATTENTION_NO_FUSE` (it needs the
unfused path to split pack from kernel), so a change to the *fused* call path
does not show up there at all. Use `flash_attn_func` directly or the matrix.

## The 0.5 IPC: measured, attributed, and it is not a removable instruction

The last open question was why we run at vLLM's instruction count and below its
IPC. Answered on m9g at 2048:8:64 bf16, one run at a time, counters taken by
subtracting 10- and 40-iteration runs of the same driver (`perf stat` over a
python process is otherwise 99% torch import):

| | instructions/row-key | cycles/row-key | IPC |
|---|---:|---:|---:|
| ours | **20.10** | **5.645** | **3.56** |
| vLLM (`vllm_cpu_attn_bench.so`) | 20.96 | 5.319 | 3.94 |

**We execute 4.1% fewer instructions and take 6.1% more cycles.** `bench_phases`
puts the K/V pack at 1.5% of that call, so both sides are kernel numbers. That
kills the "we waste instructions somewhere" family of explanations before any
code reading: there is nothing to delete.

Static diff of the inner loops (`analyze_asm.py`, their symbols resolve):

| loop | instructions | bfmmla | per bfmmla | stack refs |
|---|---:|---:|---:|---:|
| ours, Q@K^T steady state | 23 | 16 | 1.44 | 0 |
| ours, Q@K^T epilogue (max + store) | 132 | 32 | 4.13 | 2 |
| ours, Q@K^T epilogue, masked | 180 | 32 | 5.63 | 6 |
| theirs, Q@K^T steady state | 94 | 64 | 1.47 | 0 |
| theirs, whole tile loop | 166-230 | 32-64 | 5.2-3.6 | 6-17 |

Same shape, same density, both spill-free in the steady state. Theirs fuses more
per iteration and carries `zip1/zip2/stp` output packing inside the loop; ours
carries a mask epilogue (`cmgt` 16 + `bif` 16 + `fmaxnm` 20 per 32 bfmmla) that
only runs on boundary tiles.

Stall attribution, same event (`armv8_pmuv3_0/stall_backend/`) on both kernels:

| | ours | theirs |
|---|---|---|
| bfmmla | 22% | **27%** |
| loads (ldp+ldr) | 24% (ldp 24) | 24% (ldr 15, ldp 9) |
| scalar/vector add | **18%** | 9% |
| mov | **11%** | ~0 |
| fadd (row sums) | 3% | 10% |
| mask (fcmgt/bsl/bic) | ~0 | 7% |

The one class we have that they do not is `mov`: two constant-seed register
copies per exp evaluation (`mov v23.16b, v4.16b`), because LLVM splits the
degree-3 Horner into two chains and each needs a copy of a constant vector to
start from. That is 0.5 of our 20.1 instructions per row-key (2.5%) and 11% of
our stalls, so it looked like the find.

It is not. Rewriting the polynomial to seed with a *multiply* into a fresh
register (0 `mov`, +1 `fmul`, same dependent depth) measured **2.5% slower**
(7422 vs 7240 us at 2048:8:64; 1926 vs 1879 at 1024:8:64; 6435 vs 6251-6439 at
1024:16:128). The experiment is the explanation: **the softmax binds on FP-pipe
slots, not on instruction count**, so trading an integer-pipe `mov` for an
`fmul` is a loss, and our exp's `mov`s are the cheap encoding of the polynomial
rather than waste. It also explains why vLLM's `fast_exp_f16` uses fp16 (half the
FP-pipe pressure at the same lane count) and why our fp16 version measured +1.7%
anyway: the conversions land on the same pipes.

So the residue is a diffuse FP-pipe/scheduling difference, not an instruction:
we are leaner, they are issuing slightly more matmul per cycle (27% of their
stalls are on `bfmmla` against 22% of ours), and the 0.33 cycles/row-key of gap
is spread across the exp, the loads and the address arithmetic with no single
site carrying it. Per the campaign's stop rule this is the "parity in code,
residual is scheduling" branch, and the two structural levers that could still
touch it — fp16 exp, hiding the softmax inside the matmul — are both measured
and both lost.

## The co-issue restructure: built twice, loses twice, and the reason is decisive

`mixed.c` says the machine co-issues (8 bfmmla 3.98 cycles, 8 fmla 2.99, 8+8
**3.99**), so the fp32 softmax ought to hide inside a tile's matmul. Built the
full version (all 4 row pairs' exp interleaved with the next tile's Q@K^T) and
the half version (row pairs 0,1 only, so the fused body carries 16 + 2 + 2 + 4
registers instead of 16 + 4 + 4 + 4), each with double-buffered logits:

| | cycles/row-key | instr/row-key | stack refs per fused iteration |
|---|---:|---:|---:|
| baseline | 5.68 | 20.87 | - |
| full co-issue | 6.07 | 22.4 | 80 |
| half co-issue | 5.84 | 21.13 | 40 |

Both lose. The register pressure explains the spill traffic but **not** the
verdict, because the trend is monotone in the interleaved fraction: less
interleaving is less bad, converging to the baseline at zero.

The real reason is in the loop shape. The Q@K^T body is 26 instructions per 16
bfmmla, i.e. **3.25 IPC for 8 cycles of matmul**, against a machine that sustains
~3.5. It is already issue-bound, so there are no spare issue slots for the exp to
hide in; interleaving adds branches, spill traffic and register pressure on top.
`mixed.c`'s 8+8-in-3.99 is 8 FMAs and nothing else -- no loads, no stores, no
address arithmetic. **Any future "co-issue the softmax" proposal has to first
show the matmul loop has spare *issue* slots, not spare matmul cycles.**

## The bf16 D=128 probability panels: worth 0%, not 2/row-key

The 2-per-row-key panel reload looked like the D=128 story (3 rows at 1.12-1.13).
Ablated it (constant panel, timing only):

| | baseline | panel ablated |
|---|---:|---:|
| 1024:16:128 bf16 | 6251-6439 us | 6374 |
| 512:32:128 bf16 | ~3520 | 3523 |
| 2048:8:64 bf16 (control) | 7240 | 7241 |

Zero. So the D=128 second kernel has no headroom from that direction, and the
alternative structures are register-infeasible anyway: panels-resident wants a
16-key tile (4 kg x RP = 16 registers) but the P@V accumulator round trip is
`DP*RP*2 = 512` ops per tile whatever `dcp` is, i.e. 4/row-key at 16 keys against
0.5 at 128; and halving the rows to afford `dcp=8` halves the panels but doubles
the V loads.

## Dead ends (measured, do not repeat)

* **Q@K^T loop unrolled x2** (two head-dim groups per iteration, to halve the
  `subs` loop-back that is the hottest single instruction in the profile):
  neutral at S=2048 (7220 vs 7226 us) and at S=128. LLVM's scheduling of the
  16-bfmmla body was already fine.
* **`dcp=2` in P@V** (16 -> 8 accumulator registers, hoping to kill register
  copies): 6.95 cycles, 24.35 instructions. The panel loads double and cost more
  than the copies. dcp=4 with 4 row pairs stays.
* **Unrolling the exp loop over all 16 key groups** (comptime): 4448-instruction
  tile body, no dynamic change — it was code size, not spills.
* **Rows > 8**: `RP*nsq` accumulators hit 32 registers and spill; nsq=2 to
  compensate doubles the Q-panel loads, which is a worse trade than the K loads
  it saves (`_qt_for` keeps 8).
* **Key-group-outer / row-pair-inner exp pass** (8 chains in flight instead of 4):
  6.25 vs 6.22, i.e. neutral — the phase is not latency-bound the way the
  `fmov`-at-1.5% share suggested.

## Traps

* `scripts/box_sync.sh` piped the remote build through `tail`, so a **failed
  mojo build reported success** and left the previous `.so` in place; one
  measurement session (and a 27/49 parity run) came from a stale binary. Fixed
  with `set -o pipefail`.
* `perf stat` needs `sudo` on the box, and `sudo` resets the environment, so a
  kernel knob (`CPU_ATTENTION_QT`) has to be re-exported through it
  (`sudo -n env VAR=... perf ...`).
* `llvm.roundeven.v4f32` is fine, but a *textual* edit that drops the `yc`
  declaration compiles into nothing useful — the failure mode was a silent
  "everything folded away" kernel that measured 1.4x faster and failed parity.
* A non-square causal shape cannot be compared against torch SDPA: torch aligns
  top-left, flash-attn bottom-right. `scripts/prof/arm/dbg.py` therefore only
  uses Sq == Sk.

## The fused pack+attention region

`dense_forward()` runs the pack tasks, `GOMP_barrier`, then the attention tasks
in **one** `GOMP_parallel` (the AMX kernel's structure). Worth exactly one
fork-join, measured interleaved on the same box: S=8 12.6 -> 10.5 us (-17%),
S=128 55.2 -> 54.4 (-1.5%), S=256 -0.7%, S=512 -0.4%, S=2048 -0.05%. An earlier
estimate of "14-18% of an S=128 call" came from `gomp_barrier_wait_end` *sample
share* at a trivial shape, which overstates it: the barrier is most of the
*CPU* time there only because the work is ~1 us.

## Scoreboard (dev box, 4 cores, pack included, rounds 21, clean process)

o/vllm bf16 **1.03-1.34**, fp32 **0.91-1.10**; 22/22 vs SDPA, 5/22 vs vLLM.
Rows above the line: bf16 1.03, 1.06, 1.07, 1.07, 1.11, 1.13, 1.13, 1.14, 1.21,
1.26, 1.34 and fp32 1.03, 1.03, 1.03, 1.05, 1.10.

### Row-1 caveat, and my own mistake

`bench_vs_refs.py` now burns 1.2 s before the first shape, and row 1 reads 54 us
/ 0.61 / 1.33 correctly -- *provided nothing else is running*. A `verify.sh`
launched while a previous run was still alive reported the S=128 bf16 row as
1.47 / 2.83 (ours 117 us, SDPA 132 us, vLLM 116 us: every variant 2.5x slow).
That was my own overlap, not the harness. One run at a time.

### Earlier table (rounds 9, before the output rewrite)

`tests/bench_vs_refs.py`, o/vllm: bf16 **1.03-1.36**, fp32 **0.91-1.12**;
**22/22** rows beat SDPA (was 17/22 at the start of the campaign, 22/22 after the
tile kernels) and **4/22** beat vLLM (was 0/22 at the start, 1/22 after them).

Kernel-level, thread-cycles per row-key at S=2048: bf16 D=64 **5.33** (vLLM
5.13, 3.9% behind), fp32 D=64 **13.68** (vLLM 13.76, parity), bf16 D=128 9.76
(~13% behind), fp32 D=128 ~1-3% behind.

What is still above the line, and why:

| rows | o/vllm | blocker |
|---|---|---|
| bf16 D=128 (4 rows) | 1.13-1.14 | 36 instructions/row-key against ~32; the P@V probability panels are reloaded per 4-dim-pair chunk (2 per row-key) because `dcp=8` needs 32 accumulator registers and spills (measured: 45.7 instructions, 12.5 cycles) |
| bf16 D=64 long (6 rows) | 1.03-1.08 | 3.9% of kernel IPC; the exp pass runs *beside* the matmul instead of overlapping it (see the co-issue note below) |
| bf16 short (3 rows) | 1.13-1.36 | the K/V pack is 15% of an S=128 call (28 GB/s, 4-byte gathers), plus one extra fork-join's worth of overhead |
| fp32 (5 rows) | 1.02-1.12 | same pack cost at short lengths; long rows are at parity |

**The first row of a `bench_vs_refs` sweep is not real**: S=128 bf16 measures
~500 us there against 58 us in `bench_phases`, and even *SDPA* reads 357 us
against 88 us -- the process is still at its idle clock. Report from the second
row on, or warm the process up.

## Open questions (nothing here is a knob)

* **The bf16 residue is scheduling, not code.** The wrap-up below has the
  counters, the static diff and the two failed manipulations; do not re-derive
  it from the stall profile.
* **fp32 D=64 is at vLLM's kernel**: 14.16 vs 13.76 cycles/row-key, 56.0 vs 51.6
  instructions, IPC 3.93 vs 3.75. Its remaining rows are at parity and move with
  box drift; its short-S rows are the pack.
* **Short S is per-work-item fixed cost**, measured rather than guessed: at S=128
  bf16 the kernel executes 56.5 instructions/row-key against 21.2 at S=2048 *at
  the same IPC* (3.52), and the pack is 15% of the call. The two fork-joins are
  already one GOMP region (`GOMP_barrier` in the worker), so what is left there
  is the P@V accumulator round trip at small `jn` (2*head_dim/keys per row-key)
  and the fact that vLLM's timed region does not contain a pack at all.

## Campaign wrap-up (2026-10-22): final state, mechanisms, and everything falsified

### Final numbers

`scripts/verify.sh arm64`, `ROUNDS=21`, clean rebuild from scratch, both parity
suites green, ratios `ours/baseline` (<1 = a win), ours includes the K/V pack:

| | vs torch SDPA | vs vLLM `csrc/cpu` |
|---|---|---|
| bf16, 11 shapes | 0.48-0.85 (11/11) | 1.02-1.32, never below 1.00 |
| fp32, 11 shapes | 0.34-0.91 (11/11) | 0.89-1.07, i.e. parity |

**22/22 beat SDPA and 2-5 of 22 beat vLLM, from 17/22 and 0/22 at the start of
the ARM campaign.** The range on that second number is not sloppiness: three
runs of the *same binary* on this box gave 5, 2 and 3 fp32 rows below 1.00
(0.89-1.07 all three times), because those rows sit on the parity line and a
row lands on either side of it with a 3% shift. Quote bf16 as
"1.02-1.32, always above 1" and fp32 as "at parity"; the "N/22 vs vLLM" count is
±2 and should never be quoted without the band.

What the remainder is, by family:

| family | vs vLLM | what it actually is |
|---|---|---|
| bf16 long (S >= 1024) | 1.03-1.13 | the scheduling residue: 4% *fewer* instructions at 6% more cycles |
| bf16 short (S <= 512) | 1.12-1.32 | the K/V pack, 15% of an S=128 call, which vLLM's timed region does not contain |
| fp32 | 0.91-1.07 | at parity in the kernel (14.16 vs 13.76 cycles/row-key); the rest is the pack at 0.8-1.4% |

### The mechanisms that produced the wins, in order of payoff

1. **vLLM's structure instead of flash-decoding's** (`_dense_rows_tile` /
   `_dense_rows_tile_f32`): logits staged in L1 with runtime indices (so LLVM's
   SROA cannot forward them back into registers -- the reason the previous
   campaign's "L1 logits" probe measured 0%), one max/sum/rescale per 64-key
   tile, 8 rows per work item, K^T blocked by key tile, V packed
   `[key group][dim pair][8]`. bf16 7.31 -> 6.07 cycles/row-key (23.15 -> 21.24
   instructions), fp32 18.25 -> 15.50 (72.7 -> 57.2).
2. **`TILE_KS` = 128 (bf16) / 256 (fp32)**, the biggest knob after the structure.
3. **Only the keys that exist**: the Q@K^T sub-block loop, the exp loop and P@V
   all stop at the causal limit of the block's last row. 1.5x at S=128, 1.3x at
   S=256, 1.2x at S=512.
4. **`_exp2_fast` / `_exp2_g`**: `roundeven` + degree-3 minimax + exponent
   insert, with `log2(e)` folded into Q on fp32. 15.14 -> 14.16 cycles/row-key
   fp32; the polynomial had been 30% of the call (5.1 instructions/row-key).
5. **One GOMP region for pack + attention** (`GOMP_barrier` inside the worker):
   two fork-joins per call had been 18% of an S=128 call.
6. Output written de-interleaved (zip1/zip2 + converts + one join).

### Falsified on ARM, with the number (do not repeat)

| lever | measured | why it cannot work |
|---|---|---|
| `TILE_KS` = 32 / adaptive tile | loses at every size, S=128 included | the causal-limit skip already bounds work by the block's limit, so a smaller tile only multiplies per-tile fixed cost |
| fp16 softmax (`_exp2_f16`) | **-10% instructions, +1.7% time** | the conversions sit on the critical path; `fmul v.8h` is 4.0/cycle but `fcvt` is not free |
| co-issue, all 4 row pairs | 6.07 vs 5.68 cycles/row-key (**+5%**), 22.4 vs 20.9 instructions, **80** stack refs per iteration | 16 Q@K^T accumulators plus 4 row pairs of exp state do not fit in 32 registers |
| co-issue, 2 of 4 row pairs | **+3.7%** (5.84 cycles/row-key, 21.13 instructions, 40 stack refs, 7518 vs 7240 us at 2048:8:64) | and less interleaving is less bad, converging on the baseline: the Q@K^T body needs **3.25 IPC** (26 instructions per 16 bfmmla) on a machine sustaining ~3.5, so there are no spare *issue* slots for the exp. `mixed.c`'s 8+8-in-3.99 contains no loads, stores or address arithmetic |
| bf16 D=128 probability panels | **0%** by ablation (6374 vs 6251-6439 us at 1024:16:128, 3523 vs ~3520 at 512:32:128, control unchanged) | the panels are L1-resident and free, and every alternative structure is register-infeasible: panels-resident needs a 16-key tile, whose P@V accumulator round trip is `DP*RP*2` = 4/row-key against 0.5 at 128 keys; halving the rows to afford `dcp=8` halves the panels and doubles the V loads |
| int8 P@V (`i8mm`, 2x `bfmmla`) | max abs error **2.9e-2 - 4.6e-2** with a per-output-dim V scale (3.2e-2 - 4.1e-2 global) against a **3e-2** gate, where bf16 is already 1.0e-2 - 1.3e-2 | `tests/int8_pv_error.py`: it is accuracy, not speed, and it is not close. A per-key V scale would be finer but sits inside the sum and cannot be factored out afterwards |
| K/V pack treated as a cold-read problem | **1.23x slower**, not faster, when the cache is evicted at S=128 | the pack is instruction-bound: 92% of it is hot gather work (4-byte gathers, 28 GB/s). Prefetch/layout work on it is worth 0.7-7% in the best case |
| Python per call | F = **10.0 us/call** (3.1 extension call + 0.66 `empty_like` + ~6 Python), ~1.5% of S=128 | real but small. The integrator then took the `stride()` marshalling from 19 per call to 1 (`fa0062c`), worth 5% at S=128 |
| pack `alignment=64` | 0% | |
| Q@K^T loop unrolled x2 | neutral (7220 vs 7226 us) | LLVM's scheduling of the 16-bfmmla body was already fine |
| P@V pointer hoists | 0% | |
| `dcp=2` in P@V | 6.95 cycles, 24.35 instructions | halves the accumulator registers but doubles the panel loads |
| more than 8 rows per work item | register wall; `nsq=2` to compensate doubles the Q-panel loads | `_qt_for` keeps 8 |
| key-group-outer / row-pair-inner exp pass | 6.25 vs 6.22 cycles/row-key | the phase is not latency-bound |

### The residue, stated as a measurement

At bf16 2048:8:64, counters subtracted across 10- and 40-iteration runs (one
process run is 99% torch import): **ours 20.10 instructions and 5.645 cycles per
row-key (IPC 3.56); vLLM 20.96 and 5.319 (IPC 3.94).** We are 4% leaner and 6%
slower. The inner loops are identical in shape (1.44 vs 1.47 instructions per
`bfmmla`, both spill-free), and the only instruction class we have that they do
not -- two constant-seed `mov`s per exp evaluation, 11% of our stalls -- is
*cheaper* than the arithmetic that replaces it: seeding the Horner with a
multiply instead measured **2.5% slower** at three shapes. That is the finding:
the softmax binds on **FP-pipe slots**, not on instruction count, which is also
why their `fast_exp_f16` is fp16 and why ours is not (fp16 conversions land on
the same pipes: +1.7%). Their 27%-vs-22% `bfmmla` stall share is the difference
-- they issue slightly more matmul per cycle. There is no site carrying the 0.5
(actually 0.38) IPC.

### The two routes not taken, with the evidence for each

* **A from-scratch kernel shaped like theirs** (blocked GEMM micro-kernel, 8-row
  x N-key accumulator panel, logits staged per tile). Two prototypes in the
  previous campaign lost **20-26%**, and nothing found since weakens that. The
  cheap version of it: port only their Q@K^T `tile_gemm` -- they reach 1.49-1.86
  `bfmmla`/cycle where we reach 1.02-1.30 -- and keep every other part of v2.
* **An instruction-by-instruction study of `tile_gemm`** to find the 0.38 IPC.
  What it would have to explain is now written down above (equal instructions,
  equal loads, their excess stalls on `bfmmla`, ours on `add`/`mov`, 0.33
  cycles/row-key in total). Their source is
  `~/vllm/csrc/cpu/cpu_attn_neon_bfmmla.hpp` on the box; `llvm-mca` is **not**
  installed there. The ceiling if it succeeds is ~6% on the bf16 rows, i.e.
  1.06 -> 1.00 -- parity, not a crossing.

### For a successor: the tools and the traps

`tests/bench_vs_refs.py` is the authoritative matrix (`--rounds`, `--json`);
`tests/bench_phases.py` splits pack from kernel (sets `CPU_ATTENTION_NO_FUSE=1`);
`scripts/prof/arm/` holds `flush.py` (eviction harness -- note it measures the
NO_FUSE path, so fused changes are invisible there), `pk.py` (instruction
counts; cycle deltas unreliable at small S), `analyze_asm.py` (per-loop
histograms from an `objdump` dump), `hot.py` (per-instruction stall shares from
`perf annotate`), `ab_qt.py`, `mixed.c`, `f16perf.c`, `one_fma.c`, `peak.c`.
`scripts/box_sync.sh` needs `set -o pipefail` (a failed remote build used to
report success and leave the old `.so`). Never run two measurements at once: a
second process doubles every number and the first row of a sweep is still at
idle clock.
