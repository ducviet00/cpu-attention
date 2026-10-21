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
* **`TILE_KS = 128`**: broke parity. A tile wider than the K block steps out of
  its key-pair run into the next head-dim group; the addressing is per sub-block
  now so it would work, but the expected gain (~1.5%) did not justify chasing
  the residual flakiness.

## Dead ends (measured, do not repeat)

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

## Scoreboard after the two tile kernels (dev box, 4 cores, pack included)

`tests/bench_vs_refs.py --rounds 7`, o/vllm: bf16 **1.06-1.30**, fp32 **0.96-1.15**;
21/22 rows beat SDPA (was 17/22), 2/22 beat vLLM (was 0/22). The warst bf16 rows
are S=256 D=128 (1.30) and S=512 H=32 D=128 (1.24); fp32's are S=1024 D=32 (1.15)
and S=4096 D=64 (1.14).

**The first row of a `bench_vs_refs` sweep is not real**: S=128 bf16 measures
~500 us there against 58 us in `bench_phases`, and even *SDPA* reads 357 us
against 88 us -- the process is still at its idle clock. Report from the second
row on, or warm the process up.

## Open

* bf16 is 21.2 instructions vs vLLM's 20.9 — the remaining 18% is **IPC** (3.50
  vs 4.07, backend stalls 1.46 vs 1.00), not instruction count.
* fp32 is 57.2 vs 51.6 instructions. Next candidates: K^T's dim stride is a
  compile-time 64 (`kt_blk`) but is read from the job at runtime, costing an
  address add per dim per sub-block; and the P@V probability/`V` addressing.
* **Short sequences are per-work-item overhead, measured**: at S=128 bf16 the
  kernel executes 56.5 instructions/row-key against 21.2 at S=2048, at the same
  IPC (3.52) -- so it is instruction count, not stalls. vLLM goes 20.9 -> 24.2.
  The per-tile P@V accumulator round trip is 2*head_dim/keys per row-key (4.0 at
  jn=8, 0.5 at jn=64), the boundary mask runs on every sub-block, and ~18% of an
  S=128 call is still OpenMP barrier (two fork-joins per call: pack + kernel).
  Collapsing the two GOMP regions into one (`GOMP_barrier` inside the worker) is
  the next thing to try.
* bf16 D=128 P@V pays 2 panel loads/row-key because the 64 dim pairs are walked
  in 16 chunks of 4 (`RP*dcp` = 16 registers); dcp=8 would halve it but needs 32.
* The `-e armv8_pmuv3_0/stall_backend/` sampling profile (works on this box)
  attributes the bf16 backend stalls to `ldp` (18%) and integer `add` (17%,
  addresses), with bfmmla at 23% -- not to the exp.
