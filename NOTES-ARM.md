# NOTES-ARM.md — ARM (Graviton 5 / Neoverse-V3) work log

Ratios are `ours / baseline` (below 1 = we win). All numbers are *thread*-cycles
per row-key from `scripts/prof/arm/pk.py` (differential `perf stat`), measured on
the m9g.xlarge dev box (4 cores, 3.26 GHz measured, bf16 `bfmmla` = 2/cycle,
fp32 FMA = 3.3/cycle, so bf16's hard floor at head_dim 64 is 8 bfmmla / 2 =
**4.0 thread-cycles per row-key**).

**Check the box is idle before trusting a number** (`uptime`, `ps aux`). A stray
process halved every measurement in one session and produced a fake 2x.

## The reference point (measured, not inferred)

`perf stat` on vLLM's `csrc/cpu` NEON kernel at S=2048 H=8 D=64 bf16 causal:

| kernel | cycles/rk | instructions/rk | IPC | L1 refill/rk | backend stall/rk |
|---|---:|---:|---:|---:|---:|
| vLLM csrc/cpu | **5.13** | 20.88 | 4.07 | 0.15 | 1.00 |
| ours v1 (per-16-key online softmax) | 7.31 | 23.15 | 3.17 | 0.86 | 2.25 |
| ours v2 (tile logits in L1) | **6.22** | 22.33 | 3.59 | 0.33 | 1.48 |

Both spend 8 instructions on bfmmla. So the gap is 2.1 non-matmul instructions
and 0.5 of IPC — it is *not* an algorithmic difference, and the old "softmax is
40% of the call and vLLM's is 9%" story does not survive contact with the
instruction counts: vLLM's whole non-matmul budget is 12.9 instructions/row-key
and they also pay a mask (fcmgt+bsl+bic = 7% of their samples) and address adds
(9%).

## What v2 is (`_dense_rows_tile`)

vLLM's structure instead of flash-decoding's: the key tile's logits are staged in
L1 (runtime-indexed `stack_allocation`, so LLVM's SROA cannot forward them back
into registers - the reason the previous campaign's "L1 logits" probe measured
0%), one max/sum per 64-key tile instead of one per 16 keys, 8 rows per work item
instead of 4. K is packed blocked by key tile (`[tile][dim group][key pair][8]`)
and V as `[key group][dim pair][8]` so P@V's inner loop needs no address
arithmetic.

Measured increments, each an interleaved or same-session `perf stat` delta at
S=2048 H=8 D=64 bf16 causal:

| change | cycles/rk | instr/rk |
|---|---:|---:|
| v1 (baseline for this table) | 7.31 | 23.15 |
| v2 skeleton (tile logits + 8 rows) | 6.68 | 22.61 |
| + K blocked by tile, runtime exp loop | 6.73 | 22.87 |
| + panel build as 2 converts + join, shared zero, exp loop unrolled x2 | **6.22** | 22.33 |

## Dead ends

* **`dcp=2` in P@V** (16 -> 8 accumulator registers, hoping to kill the register
  copies): 6.95 cycles, 24.35 instructions - the panel loads double and that
  costs more than the copies. dcp=4 with 4 row pairs stays.
* **Unrolled exp loop (16 comptime copies)**: 4448-instruction tile body, no
  dynamic change (it was code size, not spills).
