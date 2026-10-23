# ARM (Graviton 5, m9g.2xlarge, Neoverse-V3) dense kernel — round 1

`cpu_attention/csrc/cpu_attn_neon.hpp` only. Branch `cpp-arm-dense`, base
`b199f3a`. Harness: `tests/bench_cpp.py --isa neon --rounds 41` (one process,
four arms rotated, min of 41, duplicate `cpp` control arm = noise floor),
vLLM = `~/vllm_cpu_build/vllm_cpu_attn_bench.so` (NEON build). Ratios are
`ours / baseline`, **below 1 is faster**. Raw JSON: `bench_arm_final.json`.

## What changed

Two hot loops in both dense policies (bf16 `DenseBF`, fp32 `DenseF32`), nothing
else — the paged policies, the layouts and the call path are untouched.

1. **`pv` walks K/V with pointers instead of recomputing the address.** The
   bf16 body was 42 instructions for 16 `bfmmla`; 16 of those 42 were integer
   address arithmetic recomputed per 4-key group (`(dim pair, key group)` ->
   index -> byte offset, with the head-dim stride a runtime value). Now the P
   and V panels advance by 16 bytes per key group and the dim-pair bases are
   hoisted. 26.4 -> 21.7 instructions per row-key.
2. **Both loops step 16 keys per iteration instead of 4-8.** This is what
   actually bought the time: the kernel is *latency* bound, not issue bound.
   Cutting 18% of the instructions without widening the step moved 2048:8:64
   bf16 by 0.1% (3827.9 -> 3824.8 us) because the freed slots just filled with
   stall; going from 4 to 16 keys in flight took the same call to 3445.1 us
   (-10.0%). The bf16 softmax keeps eight independent `exp2` chains live
   instead of four, and `pv` issues 64 `bfmmla` per step so its eight loads run
   ahead of the matmul.
3. **The `-inf` row test left the inner loop** in both softmaxes. It was
   loop-invariant (`mn` is the row's running max) but the compiler kept a
   branch and re-splatted the polynomial constants every 4 keys (54
   instructions where 40 are the polynomial). A row with no visible key anywhere
   in the call is now a separate, rare path.

## Result

Measured effect, alternating-process A/B (`min` of 9, one process per sample,
variants alternated round-robin), 2048:8:64:

| variant | bf16 us | fp32 us |
|---|---:|---:|
| base `b199f3a` | 3827.9 | 9466.0 |
| after 1 (addresses only) | 3824.8 | — |
| after 2+3 (this patch) | **3445.1** (-10.0%) | **9185.6** (-3.0%) |

The 41-round three-arm matrix (control arm `ctrl/mojo` reads 0.965-1.222, i.e.
it tracks the `cpp` arm to within 0.005 on every row — the numbers below are
resolvable):

| dense row | dtype | cpp us | mojo us | vllm us | cpp/mojo | cpp/vllm |
|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 25.6 | 21.0 | 22.3 | 1.217 | 1.147 |
| 128:8:64 | fp32 | 43.7 | 41.9 | 45.4 | 1.045 | 0.964 |
| 512:8:64 | bf16 | 242.4 | 202.6 | 232.4 | 1.197 | 1.043 |
| 512:8:64 | fp32 | 584.8 | 518.0 | 588.8 | 1.129 | 0.993 |
| 1024:8:64 | bf16 | 889.6 | 775.2 | 932.1 | 1.148 | 0.954 |
| 1024:8:64 | fp32 | 2306.2 | 2073.2 | 2263.3 | 1.112 | 1.019 |
| 2048:8:64 | bf16 | 3438.3 | 3133.0 | 3569.0 | 1.097 | 0.963 |
| 2048:8:64 | fp32 | 9174.3 | 8220.6 | 8859.0 | 1.116 | 1.036 |
| 4096:8:64 | bf16 | 13479.5 | 12601.3 | 14091.9 | 1.070 | 0.957 |
| 4096:8:64 | fp32 | 36584.3 | 33442.6 | 34520.5 | 1.094 | 1.060 |
| 256:16:128 | bf16 | 224.3 | 191.3 | 211.8 | 1.172 | 1.059 |
| 256:16:128 | fp32 | 584.7 | 523.8 | 658.2 | 1.116 | 0.888 |
| 1024:16:128 | bf16 | 2935.2 | 2754.1 | 2954.5 | 1.066 | 0.993 |
| 1024:16:128 | fp32 | 8673.8 | 8011.8 | 9519.2 | 1.083 | 0.911 |
| 2048:16:128 | bf16 | 11284.3 | 11006.5 | 11446.8 | 1.025 | 0.986 |
| 2048:16:128 | fp32 | 34037.4 | 31366.2 | 36043.7 | 1.085 | 0.944 |
| 4096:16:128 | bf16 | 44301.3 | 43012.9 | 45422.6 | 1.030 | 0.975 |
| 4096:16:128 | fp32 | 135357.8 | 140234.7 | 146406.1 | **0.965** | 0.925 |
| 512:32:128 | bf16 | 1578.0 | 1461.0 | 1552.3 | 1.080 | 1.017 |
| 512:32:128 | fp32 | 4674.7 | 4177.8 | 4812.9 | 1.119 | 0.971 |
| 1024:8:32 | bf16 | 601.2 | 524.0 | 639.8 | 1.147 | 0.940 |
| 1024:8:32 | fp32 | 1289.7 | 1191.5 | 1322.9 | 1.082 | 0.975 |

| | rows | `cpp/vllm` <= 0.97 | `cpp/mojo` <= 0.97 | min-max cpp/mojo | min-max cpp/vllm |
|---|---:|---:|---:|---|---|
| dense bf16 | 11 | 5 | 0 | 1.025-1.217 | 0.940-1.147 |
| dense fp32 | 11 | 4 | 1 | 1.045-1.129 | 0.888-1.060 |
| dense all | 22 | 9 | 1 | 0.965-1.217 | 0.888-1.147 |
| paged (bf16+fp32) | 20 | 18 | 14 | 0.512-1.044 | 0.260-1.360 |

Progress against the starting state of this branch (same harness, 41 rounds):
bf16 `cpp/mojo` was **1.20-1.29** on every row and is now **1.03-1.22**; fp32
was 1.06-1.15 and is now 1.05-1.13. `cpp/vllm` went from 0.90-1.25 (4/22 <= 1)
to 0.89-1.15 (9/22 <= 0.97).

**Paged is untouched** (`git diff` is only the two dense policy structs):
`cpp/vllm` 0.26-1.36, 18/20 <= 1.0, `cpp/mojo` 0.51-1.04. The two rows above 1.0
are `b=1 ctx=256 bf16` (1.19 self, 1.36 cross) where the Mojo arm reads the same
1.20/1.34 — it is the harness asymmetry the CPP report already documents (vLLM's
output tensor is preallocated outside the timed region, `torch.empty` is 1.2 us
of an 8 us call), not a kernel difference. The cache layout is byte-identical
(Mojo and vLLM staging both still read it: `d_cpp` 1.2e-3 bf16 / 3.4e-7 fp32).

## Parity

Every gate passes on the box with `CPU_ATTENTION_BACKEND=cpp` after the change:

| gate | result |
|---|---|
| `tests/test_dense.py` | 65/65 checks |
| `tests/test_flash_api.py` | all checks |
| `tests/fuzz_dense.py --cases 80 --max-seq 900` | 80/80, worst case 0.52x its tolerance |
| `audit/audit_dense.py` (naive fp32 reference) | bf16 <= 9.4e-3 (gate 3e-2), fp32 <= 2.8e-6 (gate 2e-4) |
| `audit/audit_paged.py` | bf16 <= 1.2e-3, fp32 <= 4.2e-7 |
| Mojo backend unchanged | `tests/test_dense.py`, `test_flash_api.py` with `BACKEND=mojo` still pass |

## Falsification ledger

Each measured in-process with the duplicate control arm (or with the
alternating-process protocol for variants that need their own `.so`), one change
at a time on top of the row above.

| idea | number | verdict |
|---|---|---|
| `pv` address strength reduction alone (bf16) | 26.37 -> 22.06 instructions/row-key, **3827.9 -> 3824.8 us (0.1%)** | kept as a prerequisite of the unroll, but it buys no time by itself: the kernel is not issue bound |
| softmax + `pv` at 16 keys per step (bf16) | **3827.9 -> 3445.1 us (0.900x)** | kept |
| softmax `-inf` branch hoisted + 8-key step + `pv` pointer walk (fp32) | **9466.0 -> 9185.6 us (0.970x)** | kept |
| `RQT = 4` for bf16 dense (2 row pairs, Mojo's D=64 choice) | 3827.9 -> 4982.4 us (**1.302x slower**) | reverted |
| `kt = 128` for D<128 (ff. L1-resident K/V tile: 32 KB vs 64 KB) | 3868.9 us (1.011x slower) | reverted |
| `rq = 32` (rows per work item) at D=64 | 3856.7 us (1.008x slower) | reverted |
| `rq` capped at 64 for `seq_q <= 512` | 512:8:64 244.6 vs 244.9 (1.000x), 128:8:64 26.0 vs 26.5 (noise) | not kept |
| 8/4-key `pv` step with 8 loads + 32 `bfmmla` (before widening further) | part of the 0.900x above | superseded by 16 keys |

## Where the remaining gap is, with numbers

At 2048:8:64 bf16 (cycles/row-key from `perf stat` differentials, 5 vs 205
calls, so process overhead is subtracted; 3.11 GHz measured from
cycles/elapsed):

| | cycles/row-key | instructions/row-key | IPC | L1D accesses | L2 accesses | stall_backend |
|---|---:|---:|---:|---:|---:|---:|
| ours, before this patch | 5.99 | 26.4 | 4.40 | 5.57 | 0.78 | 1.56 |
| ours, after | **5.09** | 18.6 | 3.65 | — | — | — |
| Mojo (the target) | 4.67 | 18.3 | 3.78 | 5.08 | 0.56 | 0.94 |
| vLLM NEON | 5.30 | — | — | — | — | — |

Cycle-sampled attribution (`perf record -e cycles:u`, 2048:8:64 bf16): `pv`
36%, `qk` 27%, softmax 27%, `qk` store epilogue 5%. `qk` runs at 2.5
`bfmmla`/cycle and `pv` at 1.88, so **the eight `bfmmla` per row-key are 3.7 of
the 5.1 cycles**; the bf16 matmul floor is 3.2 cycles/row-key at the measured
2.5/cycle issue rate. Instruction count is now at parity with Mojo (18.6 vs
18.3), which means the residue is **latency, not instructions** — the earlier
26.4 -> 21.7 cut moved nothing until the step width changed.

What that leaves, per row:

* **bf16 D=64/32 long rows (`cpp/mojo` 1.07-1.15, worst at 512:8:64 1.197):**
  5.1 vs 4.67 cycles/row-key. Mojo holds the logits *in registers* (16-key
  tile, `qt=4`, `bfmmla` accumulators are the logits) so it has no logits
  store/load round trip and no separate softmax pass; our policy contract
  (`qk` -> `softmax` -> `pv` through the `float* logits` buffer) forces one.
  Closing this needs the fused row kernel, i.e. one function that keeps a
  16-key x 2-row-pair logit panel in the 32-register file, exps it in place and
  feeds P@V from it — a rewrite of the policy, not a knob. Estimated ceiling:
  Mojo's 4.67 x (18.3/18.6) ~ 4.6, i.e. parity, not the 0.97 the goal asks for.
* **fp32 (`cpp/mojo` 1.05-1.13):** 32 scalar-pipe FMAs per row-key is the whole
  budget (16 for Q@K^T, 16 for P@V) and NEON has no fp32 matmul instruction.
  At the measured 2.16 FMA/cycle that is 14.8 cycles/row-key of arithmetic
  against 12.7 for Mojo, whose fp32 kernel uses a larger `qt` and an
  L1-resident accumulator. The only way past 2.2 FMA/cycle is bf16 `bfmmla`
  with a hi/lo split (`q_hi*k_hi + q_hi*k_lo + q_lo*k_hi` = 3 `bfmmla` per 16
  MACs against 4 fp32 FMAs), which is ~25% fewer instructions in both phases at
  a ~1.5e-5 relative error — inside the 2e-4 gate. Not attempted here.
* **Short rows (128:8:64 1.217, 256:16:128 1.172):** per-call fixed cost. The
  V pack for bf16 still moves 4 bytes per store (four `memcpy` of 4 bytes per
  16-byte group); at 128 keys it is ~2.6 us of a 25.6 us call. A two-dim-pair
  pack would halve it (~5% of that row) and nothing more.
