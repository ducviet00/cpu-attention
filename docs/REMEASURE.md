# Re-measurement on a fresh instance, 2026-10-23

> **Superseded by `docs/AUDIT.md` (2026-10-23, fourth campaign).** The counts
> below are the pre-merge `03-3a` tree (18/22 vs vLLM); the merged tree is
> 21/22 on m9g and the x86 numbers were re-measured on m8i. Keep this file for
> its falsifications and the harness protocol, not for the row counts.


Handover measurement pass: re-measure every claim in `docs/` on a brand-new
Graviton box and correct what does not reproduce. **No kernel, dispatch or
threshold was changed** — this file and the corrections it points at are the
only edits.

## The box

| | |
|---|---|
| host | `ubuntu@54.255.175.68`, `ip-172-31-35-179` |
| instance | m9g.2xlarge, 8 cores, Neoverse-V3, spot, started minutes before the run |
| tree | `main` @ `973cce9`, synced by `scripts/verify.sh`, clean rebuild (`_mojo/build` and `_C.so` deleted) |
| torch | 2.14.1+cpu, 8 threads (the box's default) |
| vLLM reference | `~/vllm_cpu_build/vllm_cpu_attn_bench.so`, pinned `db3b0f0`, `--isa neon` |
| provisioning check | `tests/test_dense.py` 65/65, `tests/test_flash_api.py` green before anything was timed |

Every timing run below waited for a 1-minute load average under 0.3 first (the
printed load is in the command log), took `~/bench.lock` through
`scripts/box_run.sh`, and judged on the **minimum** of interleaved samples.

## 1. The 22-row matrix, twice

```
REMOTE_DIR=cpu-attention RUNS=2 scripts/verify.sh arm64 ubuntu@54.255.175.68 . fresh
```

Clean rebuild, then `tests/bench_vs_refs.py --vllm … --isa neon --rounds 9`
twice in the same process image. Load at the start of run 1: **0.28**, run 2:
**0.29**. `scripts/band.py` output is in `/tmp/opencode/remeasure/verify-fresh.log`
on the driving box; the numbers:

| shape | dtype | run 1 min | run 2 min | run 1 med | run 2 med | 21-round med |
|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 1.014 | 1.045 | 1.047 | 1.031 | 0.987 |
| 128:8:64 | fp32 | 0.919 | 0.925 | 0.934 | 0.962 | 0.918 |
| 512:8:64 | bf16 | 0.898 | 0.917 | 0.930 | 0.910 | 0.934 |
| 512:8:64 | fp32 | 0.896 | 0.896 | 0.914 | 0.913 | 0.901 |
| 1024:8:64 | bf16 | 0.893 | 0.917 | 0.900 | 0.921 | 0.901 |
| 1024:8:64 | fp32 | 0.933 | 0.910 | 0.931 | 0.908 | 0.930 |
| 2048:8:64 | bf16 | 0.922 | 0.932 | 0.930 | 0.935 | 0.920 |
| 2048:8:64 | fp32 | 0.953 | 0.935 | 0.963 | 0.944 | 0.930 |
| 4096:8:64 | bf16 | 0.933 | 0.930 | 0.940 | 0.938 | 0.940 |
| 4096:8:64 | fp32 | 0.951 | 0.947 | 0.952 | 0.948 | 0.978 |
| 256:16:128 | bf16 | 1.049 | 1.050 | 1.034 | 1.070 | 0.973 |
| 256:16:128 | fp32 | 0.798 | 0.802 | 0.803 | 0.805 | 0.793 |
| 1024:16:128 | bf16 | 0.979 | 0.977 | 0.988 | 0.999 | 0.968 |
| 1024:16:128 | fp32 | 0.834 | 0.846 | 0.836 | 0.846 | 0.849 |
| 2048:16:128 | bf16 | 0.999 | 1.002 | 1.017 | 1.017 | 1.016 |
| 2048:16:128 | fp32 | 0.876 | 0.870 | 0.879 | 0.873 | 0.888 |
| 4096:16:128 | bf16 | 0.995 | 1.010 | 0.995 | 1.011 | 1.005 |
| 4096:16:128 | fp32 | 0.967 | 0.994 | 0.968 | 0.995 | 0.993 |
| 512:32:128 | bf16 | 0.986 | 0.995 | 0.997 | 1.000 | 0.975 |
| 512:32:128 | fp32 | 0.842 | 0.854 | 0.983 | 0.856 | 0.848 |
| 1024:8:32 | bf16 | 0.865 | 0.868 | 0.875 | 0.887 | 0.862 |
| 1024:8:32 | fp32 | 0.909 | 0.911 | 0.915 | 0.910 | 0.907 |

`min` = per-run minimum, the project's decision statistic; `med` = per-run
median (what the scorecard's table used to quote); the last column is the extra
full-matrix run at `--rounds 21 --control` described below.

Counts: **18/22 beat vLLM in both runs on the min** (run 1 alone: 20/22; run 2:
18/22; medians: 19/22 and 17/22). One row is on the line (4096:16:128 bf16,
0.995 then 1.010) and three lose in both runs: 128:8:64 bf16, 256:16:128 bf16
(1.05 in the matrix, 0.91-0.96 isolated — section 2) and 2048:16:128 bf16
(0.999/1.002 min, 1.017/1.017 median). **22/22 beat SDPA**, `o/sdpa`
0.282-0.877. **fp32 is 11/11 in both runs**, the closest being 4096:16:128 at
0.967/0.994.

## 2. The contested rows, five processes with a control arm

```
scripts/box_run.sh ubuntu@54.255.175.68 'cd ~/cpu-attention && \
  .venv/bin/python tests/bench_vs_refs.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
  --isa neon --shapes 128:8:64,256:16:128,4096:16:128 --dtype bf16 --rounds 21 --control'
```

Five times, one process each (`~/contested-{1..5}.json` on the box). `ctrl` is a
second `ours` arm: its ratio is the run's noise floor.

| shape | o/vllm median, 5 runs | o/vllm min, 5 runs | `ctrl` |
|---|---|---|---|
| 128:8:64 | 0.978-1.011 | 0.987-1.020 | 1.00-1.03 |
| 256:16:128 | 0.956-0.990 | 0.914-0.957 | 0.97-0.99 |
| 4096:16:128 | 1.001-1.010 | 1.001-1.009 | 1.00 |

The control is inside 0.97-1.03, i.e. the noise floor on these rows is ~3%, and
the two rows that disagree with the docs disagree by more than that except
`128:8:64` (see the verdict list).

`256:16:128` bf16 is the one row that reads very differently in the two
harnesses: **0.914-0.957 isolated** (and 0.973 in the 21-round full matrix, and
0.96-0.99 median in the isolated runs) but **1.034-1.070 in the 9-round full
matrix**. Its per-sample spread is ±8% (40 samples: 199-239 µs, median 214) —
the largest of any row — and a 9-round median lands anywhere in that band. The
same process, same shape, 40-round trace:

```
236 226 219 211 217 223 211 221 237 208 219 204 201 239 199 203 209 203 214
204 220 209 205 212 214 216 214 222 210 215 220 214 224 212 217 216 209 228 213 211
```

Preceding it with the whole 4096:8:64 fp32 row in one process does not move it
(0.938 min), and 12 s of 8-core burn does not either (0.943 min); SDPA and vLLM
are flat across all of it (491-497 and 205-219 µs). So this is a property of the
row, not drift in the box.

## 3. Parity and the gates

```
tests/test_dense.py                     65/65 checks passed
  (incl. "plan-buffer entry is bit-identical on 24 cases")
tests/test_flash_api.py                 all checks passed
tests/fuzz_dense.py --cases 80 --max-seq 900
                                        80/80 within tolerance, worst 0.52x
                                        its tolerance; stray-read probe 24/24 clean
```

Matrix parity column: bf16 `max|diff|` 7.8e-3 at D=64 and 1.6e-2 at D=128,
fp32 6.3e-7..2.1e-6. Gates 3e-2 / 2e-4. **Reproduces.**

## 4. The two claims the campaign's conclusion rests on

### 4a. `bfmmla` throughput (hand-written asm, 8 cores at once)

`/tmp/opencode/remeasure/bfmmla_probe.c` (on the box as `/tmp/bfmmla_probe.c`):
16 independent accumulators, v16-v31, and 16 `bfmmla` per iteration; the load
variants add `ldp`s from one 32 KB L1-resident buffer. Each thread is pinned to
one core; eight threads run at once; the reported ns/instr is per instruction per
core, min over 2-3 runs.

| loop shape | ns/instr | MAC/ns/core | GFLOP/s/core |
|---|---:|---:|---:|
| 16 accs, no loads, 16 distinct operand regs | 0.119-0.126 | **128-134** | 256-268 |
| 16 accs, no loads, one shared operand pair | 0.140-0.146 | 110-114 | 220-229 |
| 16 accs, **8 `ldp`** per 16 bfmmla (256 B/iter) | 0.139-0.141 | **114-115** | 228-229 |
| 16 accs, **4 `ldp`** per 16 bfmmla (128 B/iter) | 0.110-0.111 | **144-145** | 288-291 |

The shipped probes, for comparison, on this box:

* `tests/bfmmla_peak.mojo` (8 chains): 0.1519 ns/instr = **105.4 MAC/ns**
  (its clock probe is broken here and prints `1000000.0` GHz — do not use it).
* `scripts/prof/arm/peak.c` (8 accumulators, intrinsics): 209-210 GFLOP/s/core
  at every thread count = 0.153 ns/instr = **104.7 MAC/ns**.

**Verdict: does not reproduce.** The docs' "208 MAC/ns/core with no loads" and
"the operand loads halve it, 208 → 105" are not what this core does. 105 is
exactly what an **8-accumulator** loop measures (0.152 ns/instr) and it
reproduces on this box, but that loop is latency-bound, not issue-bound: with 16
accumulators the no-load rate is 128-134, and with the load pattern the kernel
actually emits (4 `ldp` per 16, per `ARM_V2_NOTES`) it is **144-145 MAC/ns**.
Adding 4 loads to 16 bfmmla costs nothing measurable; adding 8 costs 14%. The
"208" is not reproducible in any of the four shapes above (0.0769 ns/instr would
be 13 instructions/ns, i.e. ~4 bfmmla/cycle plus loop control — above what a
4-wide core can issue).

Consequence for the accounting: at 4096:16:128 the call does 34.36 GMAC in
43.2 ms = 99 MAC/ns/core = **68% of the 4-ldp ceiling**. The phase ablation's
Q@K^T + P@V = 80-84% of the call is 118-124 MAC/ns/core = **82-85% of it**. So
the matmul phases have ~15-18% headroom, not 6%; the campaign's five failed
restructurings are still the empirical answer to "can it be collected", but
"we are at 94% of the hardware" is not the right reason.

### 4b. The K/V pack

`tests/pack_only.py` does not run on `main` (it unpacks 42 plan fields, the plan
has 43 since the plan-buffer entry; and it computes `mode - 4` while the
short-sequence path now adds 8). Patched copy at
`/tmp/opencode/remeasure/pack_only_fixed.py` (`_pbuf` added, `pmode = mode & 3`),
run twice, medians of 21:

| shape (bf16) | pack µs | GB/s | bytes moved | share of the scorecard's call |
|---|---:|---:|---:|---:|
| 128:8:64 | 2.16 | 243 | 0.52 MB | 10.1% of 21.4 µs |
| 256:16:128 | 7.25 | 578 | 4.19 MB | 3.5% of 209 µs |
| 512:32:128 | 26.2 | 641 | 16.8 MB | 1.7% of 1503 µs |
| 1024:16:128 | 26.7 | 628 | 16.8 MB | 0.9% of 2822 µs |
| 2048:16:128 | 75.5 | 444 | 33.6 MB | 0.7% of 11175 µs |
| 4096:16:128 | 246.6 | 272 | 67.1 MB | 0.6% of 43285 µs |
| 4096:8:64 | 25.6 | 656 | 16.8 MB | 0.2% of 12771 µs |
| 128:8:64 fp32 | 3.38 | 310 | 1.05 MB | 7.5% of 45 µs |

**Verdict: partially reproduces.** 438-462 GB/s at 2048:16:128 sits inside the
documented "422-495 GB/s on 8 threads", but the rate is one number only over a
narrow size range: it is 578-695 GB/s from 256 to 1024 keys (L3-resident) and
265-272 GB/s at 4096:16:128 (67 MB, DRAM). The documented **share** band
(0.8-6.3%) is right for S>=1024 (0.6-0.9%) and wrong at both ends: it is 1.7% at
512:32:128 (not 6.3%) and **10% at 128:8:64**, the row where it matters. Those
shares are pack-alone; `tests/bench_phases.py` reports 1.4-9.7% because
`CPU_ATTENTION_NO_FUSE=1` adds a fork-join inside the measured pack.

## 5. The call path at 128:8:64 bf16

The FINDINGS §6/§6a method: wrap the extension entry with a no-op and time
`flash_attn_func` (400-800 rounds, median and min).

| item | measured | FINDINGS §6/§6a claim |
|---|---:|---:|
| `flash_attn_func`, kernel included | 21.4 µs (min 21.1) | 24-25 |
| Python path, 7-argument entry (no-op) | **3.40 µs** (min 3.25) | **2.6** |
| Python path, 42-argument entry (no-op) | 3.80 µs (min 3.57) | 4.47 |
| saved by the plan buffer | **0.40 µs** | 1.87 (4.47 → 2.6) |
| `torch.empty(q.shape)` | 1.36-1.38 µs | 0.56 |
| plan key + `dict.get` | 0.027 µs | 0.5 |

**Verdict: does not reproduce.** The seven-argument entry saves **0.40 µs**, not
1.87, and the call path is 3.40 µs of a 21.4 µs call (15.9%), not 2.6 of 24
(11%). Two of the components move with the box: `torch.empty` measures 1.36 µs
here against 0.56 documented (it is the largest single item), and the remaining
Python is ~2.0 µs. The direction of the doc's claim is right (the plan buffer is
worth something, the marshal is not the lever); the size is 4x smaller.

## 6. fp32 is 11/11

Reproduces: all 11 fp32 rows are below 1 in both runs. Closest to the line:
**4096:16:128 fp32 at 0.967/0.994 min (0.968/0.995 median)** — one row is within
1% of parity. Next: 2048:8:64 0.935-0.953, 4096:8:64 0.947-0.951. Two rows are
much better than documented: **512:32:128 fp32 0.842-0.854** (docs 0.99-1.00;
three independent measurements agree: r1 min 0.842, r2 median 0.856 = min 0.854,
21-round run median 0.848) and **1024:16:128 fp32 0.834-0.846** (docs 0.89-0.92).

## Verdict list, one line per claim checked

| doc claim | verdict | the number |
|---|---|---|
| 19/22 rows beat vLLM | **does not** as a single number | 18/22 on the min in both runs; 20/22 run 1, 18/22 run 2; 17-19/22 by median; 20/22 in the 21-round+control run |
| 22/22 beat SDPA, `o/sdpa` 0.29-0.89 | **reproduces** | 22/22, `o/sdpa` 0.282-0.877 |
| the three contested bf16 rows are 1.00-1.02 | **partly** | 128: 0.99-1.02 (isolated 5x), 1.01-1.05 (matrix); 256: 0.91-0.96 (isolated 5x), 1.03-1.07 (9-round matrix), 0.97 (21-round matrix); 4096: 1.001-1.010 — real 0-1% loss |
| 2048:16:128 bf16 is 0.97-0.99 | **does not** | 0.999/1.002 min, 1.017 median — at the line, not below it (3% worse) |
| 4096:8:64 bf16 is 0.91-0.92 | **does not (2-3%)** | 0.930-0.940 |
| fp32 11/11 | **reproduces** | 11/11 both runs; closest 4096:16:128 0.967/0.994 |
| 512:32:128 fp32 is 0.99-1.00 | **does not (better)** | 0.842-0.854 in three measurements |
| 1024:16:128 fp32 is 0.89-0.92 | **does not (better)** | 0.834-0.846 |
| bf16 `max\|diff\|` <= 1.6e-2 / fp32 <= 3e-6, gates 3e-2 / 2e-4 | **reproduces** | 7.8e-3-1.6e-2 and 6.3e-7-2.1e-6; test_dense 65/65, fuzz 80/80 worst 0.52x |
| `bfmmla` 208 MAC/ns no loads, 105 with loads, loads halve it | **does not** | 128-134 no loads; 114-115 with 8 loads/16; **144-145 with the kernel's 4 loads/16**; 105 is the 8-accumulator latency-bound rate |
| the matmul is at 94% of the ceiling | **does not follow** | Q@K^T+P@V is 82-85% of the 4-load ceiling (118-124 of 144-145 MAC/ns); the whole call is 68% |
| the pack is 422-495 GB/s and 0.8-6.3% of a call | **partly** | 243-695 GB/s depending on working set; share 0.2-1.7% for S>=512, 3.5% at 256:16:128, 10.1% at 128:8:64 |
| call path is ~2.6 µs of 24 µs after the 7-arg entry | **does not** | 3.40 µs of 21.4 µs; the plan buffer is worth 0.40 µs |
| `tests/pack_only.py` reproduces the pack numbers | **cannot run** | stale against the 43-field plan and `mode + 8`; patched copy used, fix is outside `docs/` |
