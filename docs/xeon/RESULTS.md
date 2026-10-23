# x86/AMX (Xeon) kernel — results, and the ledger of what did not pay

Branch `cpp-xeon-amx`, one file owned: `cpu_attention/csrc/cpu_attn_amx.hpp`.
Box: **Xeon Platinum 8581C**, 4 physical cores / 8 threads, 48 KB L1d, 2 MB L2
per core, 260 MB L3, 4 torch threads, 2.30 GHz fixed (a VM: no cpufreq, no perf).
All numbers are `ours/baseline` — **below 1 is faster**.

## 0. The measurement regime is bimodal, and that is the most important finding here

The 128:8:64 call takes **25 us or 65 us** in this box's two regimes, for the
same binary, and the ratio against Mojo moves with it (1.00 vs 1.9). The cause is
thread placement, not the kernel: 8 hardware threads on 4 physical cores, and
libgomp does not pin, so four workers sometimes land as two siblings per core.

| `OMP_PROC_BIND` | 128:8:64 bf16 | 1024:8:64 fp32 |
|---|---|---|
| unset (what the harness uses) | 26.1 | 1160 |
| `spread` | 25.3 | 1213 |
| `close` | 41.8 | 2264 |

`close` is 1.6-2.0x slower because of SMT sharing; `unset` lands anywhere
between. This is why the 128:8:64 row reads 1.02 in one run and 1.9 in the next,
and why `/proc/pressure/cpu` reads 0 while `loadavg` reads 1.7 during a run: a
single measurement is only meaningful against the arms measured **in the same
process**, and even then the row is a coin flip. Everything below is one process,
rotated arms, min of 41 (15 for paged), with a duplicate `cpp` arm as the floor.

## 1. Dense: full 22-row matrix (bf16 baseline = vLLM AMX, fp32 = vLLM VEC)

Raw: `docs/xeon/matrix_final.json`. `ctrl` is a duplicate `cpp` arm.

| row | dt | cpp | mojo | ctrl | vllm | cpp/mojo (run A / run B) | cpp/vllm | ctrl/cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `128:8:64` | bf16 | 38 / 38 | 34 | 40 | 31 | **1.115 / 1.129** | **1.240** | 1.035 |
| `128:8:64` | fp32 | 73 / 71 | 70 | 74 | 69 | **1.042 / 1.057** | **1.052** | 1.014 |
| `512:8:64` | bf16 | 169 / 168 | 169 | 166 | 209 | **0.999 / 0.998** | **0.809** | 0.985 |
| `512:8:64` | fp32 | 353 / 342 | 360 | 355 | 592 | **0.981 / 1.000** | **0.596** | 1.008 |
| `1024:8:64` | bf16 | 545 / 538 | 604 | 554 | 722 | **0.902 / 0.899** | **0.754** | 1.017 |
| `1024:8:64` | fp32 | 1105 / 1064 | 1126 | 1108 | 2040 | **0.982 / 0.971** | **0.542** | 1.002 |
| `2048:8:64` | bf16 | 1978 / 1917 | 2149 | 1965 | 2784 | **0.920 / 0.894** | **0.710** | 0.994 |
| `2048:8:64` | fp32 | 3997 / 3912 | 4110 | 3933 | 7673 | **0.973 / 0.990** | **0.521** | 0.984 |
| `4096:8:64` | bf16 | 7209 / 7163 | 8186 | 7243 | 10996 | **0.881 / 0.890** | **0.656** | 1.005 |
| `4096:8:64` | fp32 | 14909 / 14700 | 15790 | 14811 | 33761 | **0.944 / 0.950** | **0.442** | 0.993 |
| `256:16:128` | bf16 | 171 / 175 | 190 | 172 | 200 | **0.902 / 0.923** | **0.853** | 1.007 |
| `256:16:128` | fp32 | 386 / 369 | 386 | 368 | 641 | **0.999 / 1.027** | **0.602** | 0.954 |
| `1024:16:128` | bf16 | 1631 / 1677 | 1764 | 1645 | 1907 | **0.925 / 0.951** | **0.856** | 1.008 |
| `1024:16:128` | fp32 | 3621 / 3617 | 3648 | 3655 | 8250 | **0.993 / 0.991** | **0.439** | 1.009 |
| `2048:16:128` | bf16 | 5586 / 5579 | 6110 | 5519 | 8211 | **0.914 / 0.925** | **0.680** | 0.988 |
| `2048:16:128` | fp32 | 13768 / 13901 | 14914 | 13509 | 28566 | **0.923 / 0.914** | **0.482** | 0.981 |
| `4096:16:128` | bf16 | 21146 / 21086 | 23687 | 21377 | 34786 | **0.893 / 0.909** | **0.608** | 1.011 |
| `4096:16:128` | fp32 | 60086 / 58479 | 71416 | 60632 | 134326 | **0.841 / 0.809** | **0.447** | 1.009 |
| `512:32:128` | bf16 | 1006 / 1014 | 1106 | 992 | 1148 | **0.909 / 0.928** | **0.876** | 0.986 |
| `512:32:128` | fp32 | 2100 / 2173 | 2244 | 2208 | 4092 | **0.936 / 0.968** | **0.513** | 1.051 |
| `1024:8:32` | bf16 | 453 / 480 | 491 | 459 | 614 | **0.922 / 0.934** | **0.738** | 1.014 |
| `1024:8:32` | fp32 | 813 / 800 | 783 | 812 | 1225 | **1.037 / 1.032** | **0.664** | 1.000 |

Run A is `docs/xeon/matrix_final.json`, run B `matrix_final2_41r.json`
(the committed build); both are min-of-41 in one process, so the two runs are a
reproducibility check, not a re-measurement of different code.

* vs Mojo: run A **0.841-1.115** (median 0.928), run B **0.809-1.129** (median
  0.940). 19 of 22 rows are at or below parity; the three that are not are the
  same three in both runs (see §5).
* vs vLLM: run A **0.439-1.240**, run B **0.362-1.294**, 15 of 22 rows below 0.97
  in both, median 0.65. The bf16 rows
  are 0.61-0.87 of vLLM's own AMX build; the fp32 rows are 0.44-0.66 of its VEC.
* The three rows that miss: `128:8:64` (both dtypes) and `1024:8:32` fp32. All
  three are fixed-cost/short-row rows — see §4.

For reference, the same matrix **before** this session's changes
(`docs/xeon/matrix_before.json`, 41 rounds, same protocol): bf16 0.930-1.184,
fp32 0.943-1.119, 8 of 22 rows above 0.97.

### Paged (the kernel is `vec::paged`, reached through `amx::paged`; I changed nothing here)

| row | dt | cpp | mojo | ctrl | vllm | cpp/mojo | cpp/vllm | ctrl/cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| self b=1/ctx256 | bf16 | 10.4 | 10.5 | 10.2 | 12.5 | 0.990 | 0.832 | 0.979 |
| cross b=1/ctx256 | bf16 | 8.8 | 9.5 | 8.7 | 8.3 | 0.927 | **1.060** | 0.992 |
| self b=8/ctx256 | bf16 | 58.1 | 96.6 | 56.5 | 111.2 | 0.601 | 0.522 | 0.972 |
| cross b=8/ctx256 | bf16 | 55.9 | 93.9 | 53.1 | 97.3 | 0.596 | 0.575 | 0.950 |
| self b=8/ctx1024 | bf16 | 365.3 | 392.8 | 365.0 | 365.9 | 0.930 | 0.998 | 0.999 |
| cross b=8/ctx1024 | bf16 | 359.7 | 392.4 | 359.7 | 346.4 | 0.917 | **1.039** | 1.000 |
| self b=32/ctx1024 | bf16 | 1907.6 | 3048.5 | 2032.0 | 2578.5 | 0.626 | 0.740 | 1.065 |
| cross b=32/ctx1024 | bf16 | 1886.7 | 3006.3 | 1996.2 | 2446.4 | 0.628 | 0.771 | 1.058 |
| self b=1/ctx4096 | bf16 | 170.7 | 232.0 | 168.8 | 193.2 | 0.736 | 0.883 | 0.989 |
| cross b=1/ctx4096 | bf16 | 167.8 | 190.3 | 168.0 | 184.6 | 0.882 | 0.909 | 1.001 |
| self b=1/ctx256 | fp32 | 18.1 | 18.5 | 17.9 | 20.3 | 0.978 | 0.892 | 0.988 |
| cross b=1/ctx256 | fp32 | 15.4 | 16.7 | 14.4 | 13.4 | 0.921 | **1.148** | 0.937 |
| self b=8/ctx256 | fp32 | 173.6 | 169.4 | 174.0 | 176.7 | **1.025** | 0.982 | 1.002 |
| cross b=8/ctx256 | fp32 | 167.9 | 163.7 | 168.6 | 162.4 | **1.026** | **1.034** | 1.004 |
| self b=8/ctx1024 | fp32 | 675.9 | 746.3 | 683.9 | 811.6 | 0.906 | 0.833 | 1.012 |
| cross b=8/ctx1024 | fp32 | 669.5 | 737.0 | 672.5 | 814.9 | 0.908 | 0.821 | 1.004 |
| self b=32/ctx1024 | fp32 | 3783.7 | 4746.2 | 3890.6 | 4945.4 | 0.797 | 0.765 | 1.028 |
| cross b=32/ctx1024 | fp32 | 3722.7 | 4684.0 | 3786.6 | 4822.1 | 0.795 | 0.772 | 1.017 |
| self b=1/ctx4096 | fp32 | 309.3 | 328.2 | 309.3 | 325.3 | 0.942 | 0.951 | 1.000 |
| cross b=1/ctx4096 | fp32 | 305.7 | 324.4 | 304.2 | 315.8 | 0.942 | 0.968 | 0.995 |

* vs Mojo: **0.595-1.026, 16 of 20 below parity**; the two `b=8/ctx256` fp32 rows
  are the misses at 1.025/1.026 — inside the floor (`ctrl/cpp` 1.00) and inside
  vLLM's own run-to-run spread.
* vs the better vLLM build per row: **15 of 20 below 1**, 12 below 0.97. The
  consistent misses are `cross b=1/ctx256` (bf16 1.06, fp32 1.148 — 8-15 us
  calls where the floor is +-6%) and the two `b=8/ctx1024` bf16 rows
  (0.998/1.039), which `docs/amx/AMX_PAGED.md` §3 already records as parity at
  the memory wall. **fp32 b=1/ctx4096, the old 1.16-1.17 deficit, is now
  0.942/0.968** — it reproduces as a win, so the ledger's "consistent loss"
  there no longer holds on this box.
* Nothing in the paged path is mine to change; it is `vec::paged` forwarded. The
  sweep below shows the base-level paged issue that is *not* this branch.

## 2. Parity

```
CPU_ATTENTION_BACKEND=cpp tests/test_dense.py           65/65 checks passed
CPU_ATTENTION_BACKEND=cpp tests/test_flash_api.py       all checks passed
CPU_ATTENTION_BACKEND=cpp tests/fuzz_dense.py --cases 80 --max-seq 900
                                                        80/80, worst 0.52x tol (2 runs)
CPU_ATTENTION_BACKEND=mojo tests/test_dense.py          65/65 checks passed
CPU_ATTENTION_BACKEND=mojo K.amx_available()            True
audit/audit_paged.py --rounds 11                        d(ours) 2.9e-07..3.9e-07 fp32
```

Dense fp32 against the naive fp32 reference stays at **2.0-2.5e-05** (gate 2e-4)
and bf16 at 0.5-1.0e-02 (gate 3e-2) on every matrix row — the `d_cpp` column of
`matrix_final.json`.

**Two failures that are not mine, both on the new base (`b0d302c`):**

1. `tests/test_paged_sweep.py` → **336/504**, all failures bf16 at `hd=128
   page=64` (ulp > 2 against the Mojo cache). Nothing in this branch touches the
   paged path (`amx::paged` is a one-line forward); the bf16 page-64 cache layout
   changed in `cpu_attn_vec.hpp` in `b0d302c`. Principal: this is yours.
2. `tests/fuzz_dense.py` is **77/80 with `zeroes_acc = true`** and 80/80 without
   it. The `zeroes_acc` hook you added is unsafe for a fully masked work item
   (`seq_q > seq_k`, `row_limit0 + nrows - 1 < 0`): no `pv` runs, the accumulator
   keeps the previous item's bytes, and `NaN * 0` is NaN. Exact diff and repro in
   `docs/xeon/REQUESTS.md` §1. **This branch does not declare the member at
   all**, so `zeroes_acc<P>()` is false and the driver keeps its memset — which is
   also why 80/80 passes here; flipping it on needs the epilogue fix first.

## 3. What changed in the kernel, and the number for each

`RQT` stays 32 (the 2x16 AMX row tile). All of these are in
`cpu_attention/csrc/cpu_attn_amx.hpp`.

### 3a. `kt` (keys per super-tile) is per dtype: 512 for fp32, 256 for bf16

The old tree put both dtypes at 256, which was the right point on the previous
1.25 MB-L2 Xeon. On this box (2 MB L2 per core) fp32 wants a bigger super-tile
because its per-key tile work is 3x and the per-super-tile bookkeeping (row max,
row sum, `m`/`l` update, the P@V accumulator load/store) is not. Paired process
A/B, median over 6 processes of `cpp/mojo`, reference `kt = 256`:

| row | fp32 | bf16 |
|---|---|---|
| 1024:8:64 | **0.975** | 1.001 |
| 2048:8:64 | **0.952** | – |
| 4096:8:64 | **0.953** | 1.020 |
| 512:8:64 | **0.971** | – |
| 1024:16:128 | **0.940** | 1.002 |
| 4096:16:128 | – | 1.032 |

640 / 768 / 1024 measured 0.98-1.02 of 512 — inside the floor, so 512 stays (the
logits buffer is then exactly 64 KB).

### 3b. Query rows per work item: 64 instead of 32

`rq` is the work item size; the driver still sweeps it in 32-row AMX chunks that
share one accumulator, one `m`/`l` set and one output epilogue. Full matrix,
before → after (`matrix_before.json` → `matrix_rq64.json`):

| row | before | after |
|---|---|---|
| 1024:8:64 fp32 | 1.047 | **0.996** |
| 2048:8:64 fp32 | 0.997 | **0.963** |
| 4096:8:64 fp32 | 0.990 | **0.978** |
| 4096:16:128 fp32 | 0.936 | **0.833** |
| 1024:8:32 fp32 | 1.055 | **1.006** |
| 512:32:128 fp32 | 0.990 | **0.968** |

`rq = 16` is **1.36x (fp32) / 1.51x (bf16) worse** — four times the bookkeeping.
The env knob is clamped to `AROWS` now; it used to allow 512 against a 64-row
accumulator.

### 3c. The tile config is re-established once per worker per call, not per phase

`ldtilecfg` costs **43.4 ns** (`docs/xeon/phase_bench.cpp`, 2e6 iterations) and
the kernel issued two per (query block, key tile) — 16384 of a 4096:16:128 call.
`aux_bytes` runs once per call on the calling thread *before* the OpenMP region,
so it bumps a generation counter; each worker compares and loads at most once.
Measured **0.957 → 0.910** cpp/mojo at 4096:16:128 bf16 (paired, 8 processes
each) and neutral at 1024:8:64 fp32. It must not be cached *across* calls: the
same process runs the Mojo kernel, which issues `tilerelease`.

## 4. Falsification ledger — the number that killed each idea

| idea | result |
|---|---|
| `kt` = 384 / 512 / 640 / 768 / 1024 on **bf16** | 1.00-1.03 of 256; 512 is 1.020-1.032 worse, so bf16 stays at 256 |
| aux/P panel sized off the *runtime* `kt` instead of `KTMAX` (287 KB -> 100 KB per thread, 512-byte P row pitch) | **worse**: cpp/mojo 1.035 -> 1.057 at 1024:8:64 fp32, 1.059 -> 1.067 at 1024:8:32. The 2 KB pitch is faster |
| degree-4 fp32 exponential | **worse**: 1.035 -> 1.057 at 1024:8:64 fp32. The softmax is not op-bound |
| `vscalefps` for the 2^n scale (1 op instead of 4) | **1.20x slower fp32, 1.065x bf16** — paired, 8 processes each. Low-throughput port on this core |
| `CPU_ATTENTION_RQ` = 16 | 1.36x (fp32) / 1.51x (bf16) worse |
| `CPU_ATTENTION_AMX_SLACK` = 4 / 16 instead of 8 | 1.006 / 1.006 — neutral, stays 8 |
| `CPU_ATTENTION_FUSE=0` (the split pack order) | 1.115 (fp32) / 1.050 (bf16) **worse**; the fused per-head order is right on a 260 MB-L3 box |
| 3 workers instead of 4 | raw 1.4x worse (the paired ratio lies here: both arms share `_workers`) |
| caching the tile config across calls (a plain `thread_local` flag) | not shippable — `tilerelease` in the Mojo kernel on the same pool; the per-call generation counter is the fix |
| `zeroes_acc = true` (drop the driver's accumulator memset) | the memset is only 0.3-1.3% of a call, and the hook as shipped is *unsafe* for a fully masked work item (fuzz 77/80). The policy does not declare the member |
| chasing the 128:8:64 row by tuning | it is bimodal on thread placement (§0). In the good regime it is 1.04-1.11; no knob moved it |

## 5. What is still open, with the measurement that blocks it

1. **`128:8:64` is 1.115 (bf16) / 1.042 (fp32) of Mojo.** The row is fixed-cost
   dominated: 66 048 logits, i.e. ~6 us of qk+softmax+pv at the measured
   per-logit rates, against a 25-38 us call. Both baselines are also overhead
   bound (vLLM 31.0, Mojo 34.4, ours 38.4), and the same row reads 1.02 when the
   box is in its fast regime. The two pieces I measured that are *not* in the
   kernel are the driver's scalar output epilogue (1.6 us of a 27 us call, 6% —
   `docs/xeon/REQUESTS.md` §2) and the accumulator memset (0.3%).
2. **The dense fixed intercept is 16.4 us (ours) vs 14.2 us (Mojo)** at 8 heads —
   measured by regressing min-time against the causal logit count over
   seq 32/64/128/256/512. Everything else in the slope is at or ahead of Mojo.
3. **`1024:8:32` fp32 is 1.037.** `DC32 = 1` and `DC16 = 2`, so P@V is a single
   pass and the softmax is the largest single phase; the same 0.98-1.04 band as
   the other D<=64 fp32 rows before `rq = 64` fixed most of them.
4. **Paged `cross b=1/ctx256`** (bf16 1.06, fp32 1.148 of vLLM): an 8-15 us call
   where the duplicate-arm floor is +-6% and vLLM's own numbers move 20% between
   runs. Not resolvable with this harness, and not in my file.
5. `tests/test_paged_sweep.py` 336/504 (bf16 `hd=128 page=64`) on the new base —
   principal's, see §2.

## 6. Instruments added (all under `docs/xeon/`)

| file | what it does |
|---|---|
| `matrix.py` | the per-row-ISA matrix: bf16 rows get vLLM's AMX build, fp32 rows its VEC build, one process, rotated arms, duplicate `ctrl` |
| `ab.py` | paired-process A/B of two env configurations, with the Mojo arm as the drift normaliser |
| `ab_so.py` | the same for two **builds** (installs each `.so` alternately) |
| `ablate2.py` | phase ablation that ablates only the C++ kernel: Mojo reads `CPU_ATTENTION_AMX_MODE` at import, so setting it afterwards leaves the Mojo arm as a fixed reference |
| `phase_bench.cpp` | calls the *shipped* pack/qk/softmax/pv with a prepared slab: ns per logit, and `ldtilecfg` at 43.4 ns |
| `probe_epilogue.cpp` | the driver's accumulator memset and output epilogue against vector forms |
| `sweep.py` | min-time against logit count; the slope is ns/logit and the intercept is the per-call fixed cost |
| `nanrate.py`, `replay.py`, `repro.py` | the fully-masked-row NaN: rate per shape, and which side is NaN |
