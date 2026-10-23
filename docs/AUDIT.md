# Independent performance audit — 2026-10-23

This is an auditor's report, not a project document: every number below was
produced on a fresh EC2 box from the tree at `ff2bfdd` with an **independent
harness** (`audit/`, described in `audit/README.md`) whose reference is a naive
blockwise fp32 attention, not torch SDPA and not the repo's own benchmark code.
Raw JSON/logs are in `audit/logs/{ryzen,arm,x86}/`. File:line citations refer to
the docs at `ff2bfdd` (what was audited); the corrections from this report
have since been applied to the current tree, so the quoted text may have moved.

The repo's own harness was also run, on the same boxes and trees, so the two
can be compared. Where they disagree the disagreement is reported; where they
agree only the independent number is quoted.

## Boxes, protocol, and what "measured" means

| | ARM | x86 |
|---|---|---|
| instance | `m9g.2xlarge` (spot), `ap-southeast-1a` | `m8i.2xlarge` (on-demand), `ap-southeast-1a` |
| CPU | Graviton 5 / Neoverse-V3, 8 cores, 1 thread/core | Intel Xeon 6975P-C, 4 cores / 8 threads |
| caches | 64 KiB L1d/core, 2 MiB L2/core, 48 MiB L3 (shared) | L3 shared; per-core sizes not recorded |
| threads | 8 (`torch` default) and 4 (a 4-core slice) | 4 (`torch` default = physical cores) |
| tree | `ff2bfdd`, clean `scripts/box_sync.sh --full` (kernel rebuilt on the box) | same |
| reference | vLLM `db3b0f0`, `-DARM_BF16_SUPPORT -march=armv8.2-a+bf16+i8mm`, `--isa neon` | vLLM `db3b0f0`, `--isa vec`; plus the AMX build (`-DCPU_CAPABILITY_AMXBF16 -mamx-tile -mamx-bf16`, 862 `tdpbf16ps`) for bf16 |
| OS / torch | Ubuntu 26.04, torch 2.14.1+cpu | Ubuntu 26.04, torch 2.14.1+cpu |
| parity before timing | `test_dense.py` 65/65, `test_flash_api.py` green | same |

Both instances were created from the stock Ubuntu 26.04 AMI and synced with
`scripts/box_sync.sh`. Benchmarks were run one box at a time; both instances
were terminated when the audit finished. The x86 box reports exactly the
CPU model (`Xeon 6975P-C`) that `AGENTS.md` §12 calls "the Xeon"; note that
the STATUS x86 scoreboard names a *different* x86 box (an Emerald Rapids
8581C on GCP), which is part of why some x86 rows below do not carry over.

Protocol: one process per run; a 1.2 s matmul burn before the first row; two
warmups per arm; round-robin with rotated arm order; `min` is the decision
statistic (the project's convention), `median` is printed to expose drift; a
duplicate `ours` arm (`ctrl`) is the noise floor. Contested rows were run with
41 rounds; bands come from repeats (3 repo runs + 2 independent runs + the
41-round run for ARM; 2 repo + 1 independent + the 41-round run for x86).

**Noise floor measured by the duplicate arm: 0.97–1.04 on m9g, 0.92–1.04 on
m8i.** Anything inside that cannot be resolved by this box, and that includes
several rows the docs quote to two decimals.

---

## TL;DR

* **Reproduced** (within the duplicate-arm noise floor 0.97–1.04 on m9g /
  0.92–1.04 on m8i): the ARM dense scoreboard (10/11 bf16, 11/11 fp32;
  o/sdpa 0.28–0.87), the ARM paged bf16 b=8 rows, the x86 fp32-vs-VEC
  scoreboard (11/11, worst ≈1.00), 22/22 vs SDPA on both boxes, 22/22 vs
  VEC, AMX being 2.8–9.8× over the vector path, the v1-vs-v2 1.5× at 128:8:64, fp32-through-AMX
  (1.4–3.4×), and the fp32 `qt=8` choice (1.45–1.61× over qt4).
* **Not reproduced / stale**: `AGENTS` §4b's "still 1.44–1.8× behind vLLM"
  and its 853/1093 GFLOP/s; `AGENTS` §10c's "Xeon 4/22 / Graviton 17/22 beat
  SDPA, 1.2–2.3× behind vLLM"; `README`'s "8 of 22 beat vLLM NEON" on a
  4-core slice; the x86 D=128 long-row per-row values in `STATUS` (measured
  15–20% worse than documented on m8i); the AMX per-head short-row gains
  (1.16× not 1.47×); the AMX "4–9.6×" range (2.80× at 128:8:64 and 3.18× at 1024:8:32); "kt is flat"; the x86 "2.0–2.8 TFLOP/s against a 3.0
  roof" (aggregate vs per-core); and `README`'s "page layout = vLLM's,
  unchanged" (bf16 caches are incompatible on both ISAs).
* **Harness**: its numbers reproduce under an independent implementation, but
  the bf16 parity gate compares two approximations and fired spuriously once
  (3.12e-2 at 1024:8:64 on x86 while both kernels were <1e-2 from truth), the
  9-round default is below the project's own 13–21 recommendation, the
  console's headline ratio is the median while the verdicts are the min, and
  there is no check that the requested vLLM ISA actually ran.
* **Ryzen dev box** (same harness, governor pinned): dense bf16 is 10/11 vs
  vLLM, fp32 is 1/11 — the fp32 D=128 long rows are 1.18-1.68x slower; vs SDPA
  21/22 (the exception is 4096:16:128 fp32 at 1.04). Paged bf16 wins the batch
  rows and loses b=1/cross, as on the servers.
* **Latent bug**: toggling `CPU_ATTENTION_AMX` off/on in one process can
  SIGSEGV because `_drop_workspace()` frees the AMX workspace without clearing
  `_AMX_PLANS` (`_kernel.py:844`); the v1↔v2 and `mode & 512` A/Bs are clean.

Two boxes were used sequentially (m9g.2xlarge, then m8i.2xlarge), each freshly
provisioned, both terminated afterwards. Every number below is from this tree
and these boxes unless the row says otherwise.

---

## 1. Headline verdicts

| # | claim (source) | measured | verdict |
|---|---|---|---|
| 1 | ARM dense, bf16: **10/11 rows beat vLLM NEON**, worst 1.02 (`docs/STATUS.md:13`) | 10/11 on the 41-round median, 11/11 on the min of that run, 9/11 in the two independent 11-round runs; worst min 0.997, worst median 1.017 | **✓ inside the band** (count is protocol-sensitive) |
| 2 | ARM dense, fp32: **11/11**, worst 0.95 (`STATUS:14`) | 11/11 in all 6 artifacts; worst 0.982 (4096:16:128) | **✓ count; band is 3% low** |
| 3 | ARM dense: **every row beats SDPA, 1.1–3.4×, o/sdpa 0.29–0.88** (`STATUS:32-33`) | 22/22, o/sdpa **0.277–0.874** (1.14–3.61×) | **✓** |
| 4 | ARM bf16 is **1.44–1.8× behind vLLM**, 853/1093 GFLOP/s (`AGENTS:298`, §4b) | 0.79–1.02 vs vLLM; **1.34–1.59 TFLOP/s** at the long rows; 0.44–0.85 TFLOP/s at 4 threads | **✗ stale** (pre-bfmmla text) |
| 5 | "Graviton (4-core slice): 22/22 SDPA, **8/22** beat vLLM NEON; fp32 wins 7/11, bf16 1.00–1.26" (`README:345`) | at 4 threads on m9g: **22/22 beat vLLM**; fp32 11/11 (0.76–0.93), bf16 11/11 (0.82–0.99) | **✗ on every audited config** (a true m9g.xlarge was not audited) |
| 6 | x86 dense bf16: **10/11 beat vLLM AMX**, worst 1.02 (`STATUS:13`) | 9/11 by min (worst 1.018), 6/11 by median (worst 1.090), ctrl 0.92–1.04 | **≈ count; per-row values do not carry over** |
| 7 | x86 dense fp32 vs VEC: 10–11/11, worst **1.00** at 128:8:64 (`STATUS:14`) | 0.946 min / 1.044 med at 128:8:64 (ctrl 0.93); all other rows 0.55–0.79 | **✓ within noise** |
| 8 | x86: **21–22/22 beat vLLM's VEC** (`README:340`) | 22/22 in the 41-round run (bf16 0.20–0.55, fp32 0.51–0.79); one 9-round process read fp32 128:8:64 at 1.13 → 21/22. Band 0.20–1.13 | **✓** |
| 9 | "Xeon **4/22** [and Graviton 17/22] beat SDPA, 1.2–2.3× behind vLLM" (`AGENTS:758-759`) | **22/22 beat SDPA on both boxes** (x86 0.21–0.79, ARM 0.28–0.87); bf16 at 0.80–1.02 vs AMX | **✗ stale** |
| 10 | AMX is **4–9.6× on bf16** (`AGENTS:755`, `README:146`) | 2.80–9.79× (128:8:64 2.80×, 1024:8:32 3.18×; 4096:16:128 9.79×) | **✗ at the short rows** |
| 11 | per-head AMX item is **1.47/1.29/1.16×** at 128/512/256 (`AGENTS` header) | **1.16 / 1.17 / 1.14×**; long rows 1.00–1.08 (claim 1.03–1.07) | **✗ short rows overstated** |
| 12 | v2 is **1.46× slower** at 128:8:64 (`STATUS:101`) | **1.51×** slower; v2 is up to 5% *faster* at 4096:8:64 / 4096:16:128 | **✓ headline; "0.93–0.99 elsewhere" does not hold** |
| 13 | fp32 through AMX (hi/lo) (`README:341-343`) | AMX on/off: **1.37–3.38×** (D=64 128:8:64 1.37×, 1024:8:64 1.99×; D=128 2.6–3.4×) | **✓** |
| 14 | x86 AMX kernel runs **2.0–2.8 TFLOP/s against a 3.0 TFLOP/s dot roof** (`STATUS:87`) | whole call 0.65–2.54 TFLOP/s **aggregate on 4 cores** = 0.16–0.64/core vs 2.98/core | **✗ dimensional error** (aggregate vs per-core; 17–21% of roof, not 67–93%) |
| 15 | ARM paged bf16 b8/256: self 0.43–0.48, cross 0.43–0.51; b32/1024 self 0.86–0.95 (`AGENTS` §10) | block 16: self **0.35**, cross **0.44**; block 32: **0.35 / 0.42**; b32/1024 self **0.77–0.89** | **✓ / better** |
| 16 | "parity-or-better elsewhere" paged (`AGENTS:113`) | b1/256 **cross** bf16 **1.37–1.51× slower**; fp32 b8/256 1.04–1.13, b8/1024 1.15–1.20, b1/4096 1.08–1.11 | **✗** |
| 17 | "Page layout: **vLLM's, unchanged**" (`README:195`) | bf16 caches are **not interchangeable**: cross-reading gives max abs diff 0.26–0.43 (x86) / 1.28 (ARM); fp32 x86 cache is identical | **✗ for bf16, both ISAs** |
| 18 | "bf16 output is **bit-identical** to the C++ kernel" (`README:246`, `AGENTS:111`) | paged bf16 ours-vs-vLLM: **0.0** at b1/256, **2.4e-4–4.9e-4** at b8/b32 (1–2 bf16 ulp); fp32 1.5–2.6e-7 | **≈** (exact only at the small case) |
| 19 | "dense ... **~1 bf16 ulp**" (`AGENTS:112`) | vs naive fp32: **6.5e-3–1.6e-2** max abs, on max abs output of 3–4 where one bf16 ulp is 1.6e-2–3.1e-2 → **0.4–1 ulp** | **✓** |
| 20 | dense fp32 **≤4e-6 vs SDPA** (`AGENTS:112`); paged fp32 **~4e-6** (`README:247`) | dense ARM 0.9–3.6e-6 ✓, dense x86 AMX **3.1e-5–7.7e-5** ✗; paged fp32 1.5–2.6e-7 ✓ | **✗ for dense on x86 AMX; ✓ for paged** |
| 21 | `kt=256` is "the corner", 128/384/512 are 0.99/0.97/0.96 "and flat" (`STATUS:98`) | kt128 **0.90** at 128:8:64 (10% *faster*), 1.07 at 1024/4096:16:128; kt384/512 0.94–1.07 | **✗ not flat** |
| 22 | bf16 `qt=8` best at every shape; table shows 8–13% spreads (`_kernel._qt_for`) | qt 6/8/12/16 within **1–3%** at every tested shape; qt8 is still a fine choice | **✗ magnitudes; choice OK** |
| 23 | fp32 `qt=8` beats 4 (`_qt_for`) | **1.45–1.61×** over qt4 at every tested shape (legal qt are multiples of 4) | **✓ (stronger than documented at D=64)** |

`README`'s paged *laptop* table and the end-to-end tok/s tables were not
audited: they are single-box Ryzen numbers and neither box here can reproduce
the engine (no model weights / fast-encdec env). The server rows of the
three-box table (`README:287-291`) and the paged table in `AGENTS` §10 were
audited and are covered above.

---

## 2. ARM dense (m9g, 8 threads)

`o/vllm` = ours / vLLM, min-based unless marked; the doc column is
`docs/STATUS.md:25-28`. Six artifacts were used for the band: three repo-harness
runs, two independent runs, and the 41-round control run.

| row | doc | measured min (41r) | measured med (41r) | band, all artifacts | verdict |
|---|---:|---:|---:|---|---|
| 128:8:64 bf16 | 0.99 | 0.974 | 0.958 | 0.95–0.99 | ✓ |
| 512:8:64 bf16 | 0.87 | 0.881 | 0.890 | 0.88–0.92 | ✓ |
| 1024:8:64 bf16 | 0.88 | 0.873 | 0.883 | 0.87–0.89 | ✓ |
| 2048:8:64 bf16 | 0.93 | 0.936 | 0.938 | 0.91–0.94 | ✓ |
| 4096:8:64 bf16 | 0.92 | 0.927 | 0.943 | 0.92–0.93 | ✓ |
| 256:16:128 bf16 | 0.99 | **0.886** | 0.952 | **0.89–1.01** | ✗ (−10%) |
| 1024:16:128 bf16 | 0.97 | 0.945 | 0.959 | 0.94–0.98 | ✓ |
| 2048:16:128 bf16 | 1.01 | 0.997 | **1.017** | 0.99–1.01 | ✓ (the 1.01 is the median) |
| 4096:16:128 bf16 | 1.01 | 0.994 | 0.994 | 0.99–1.03 | ✓ |
| 512:32:128 bf16 | 0.97 | 0.946 | 0.957 | 0.94–0.98 | ✓ |
| 1024:8:32 bf16 | 0.84 | 0.826 | 0.842 | 0.83–0.85 | ✓ |
| fp32 (range) | 0.79–0.95 | 0.788–0.982 | 0.790–0.984 | 0.79–0.99 | ⚠ 4096:16:128 is 0.982, outside the band |

Counts: bf16 **10/11 by median** (the 2048:16:128 row is 1.017), **11/11 by
min** in the 41-round run, **9/11 by min** in the two independent 11-round runs. fp32
**11/11** in all six artifacts. The doc's "10/11" is therefore a fair summary,
but "worst row 1.02 at **256:16:128**" is not: 256:16:128 is the *best* of the
D=128 rows here (0.886 min / 0.952 med), and the 1.02 is 2048:16:128's median.
The per-row table in STATUS was measured on a different m9g run (the docs do
not name the instance) and one row (256:16:128) differs by 10%; the other ten
agree to ±3%.

**Absolute throughput.** 41-round mins, causal:

| | doc (superseded table, `README:378-387`; `AGENTS:395`) | measured 8t | measured 4t |
|---|---:|---:|---:|
| bf16 S=2048 H=8 D=64 | 853 GFLOP/s (5038 µs) | **1339** | 717 |
| bf16 S=2048 H=16 D=128 | 1093 GFLOP/s | **1540** | 835 |
| bf16 S=4096 H=16 D=128 | — | **1586** | 846 |
| fp32 S=2048 H=8 D=64 | 209–255 | **523** | 284 |
| fp32 S=4096 H=16 D=128 | — | **485** | 270 |

The "previous campaign" table is labelled superseded in README, but `AGENTS`
§4b presents the same 853/1093 and "still 1.44–1.8× behind vLLM" as current
("Result: ... 331 → 853 ... at D=64"). On this tree the kernel is 1.4–1.6×
faster than those numbers and **at parity with vLLM** (0.79–1.02), not behind
it. `AGENTS` §4b is pre-`bfmmla`/pre-per-head text.

**4-thread slice** (not a true 4-vCPU instance — the memory system is still
m9g's — but the configuration `README:345` describes as a "Graviton 4-core
slice"): 22/22 beat vLLM, bf16 0.82–0.99, fp32 0.76–0.93, every row also beats
SDPA. The README's "8 of 22" and "fp32 wins 7 of 11, bf16 1.00–1.26" are from
an older tree and do not reproduce on a 4-thread slice; a true 4-vCPU box was
not audited.

---

## 3. ARM paged (m9g, 8 threads, heads=16, dim=64)

Independent harness, each kernel on its own cache layout (the layouts differ —
see §6), 15 rounds, block = 16 (the engine's aarch64 default) and 32.
`o/vllm` min (median).

| case | doc (`AGENTS` §10) | block 16 | block 32 |
|---|---:|---:|---:|
| self b=8 ctx=256 | 0.43–0.48 | **0.35** (0.38) | **0.35** (0.36) |
| cross b=8 ctx=256 | 0.43–0.51 | **0.44** (0.47) | **0.42** (0.50) |
| self b=8 ctx=1024 | 0.60–0.79 | 0.69 (0.74) | 0.71 (0.80) |
| cross b=8 ctx=1024 | 0.65–0.77 | 0.76 (0.80) | **0.82** (0.83) |
| self b=32 ctx=1024 | 0.86–0.95 | 0.89 (0.92) | **0.77** (0.84) |
| cross b=32 ctx=1024 | — | 0.94 (0.89) | 0.79 (0.84) |
| self b=1 ctx=4096 | 0.86–1.09 | **0.59** (0.73) | **0.74** (0.85) |
| cross b=1 ctx=4096 | 0.77–0.99 | **0.73** (0.81) | **0.80** (0.87) |
| cross b=1 ctx=256 | — | **1.37** (1.30) | **1.51** (1.35) |

The claimed bf16 rows reproduce or are conservative, except `cross b=8/1024`
at block 32 (0.82 vs the 0.77 upper bound) and the unlisted `b=1/256 cross`
rows, which lose by 37–51%. `AGENTS` §1's "parity-or-better elsewhere" is
therefore not true at b=1.

fp32 paged (independent harness): block 32 gives self/cross b=8/256
**0.37/0.52**, b=8/1024 **0.79/0.75**, b=32/1024 **0.76/0.76**, b=1/4096
**0.72/0.76** — i.e. fp32 is fine on ARM (unlike x86, §4).

Absolute `cross b=8/enc256` is **34.2 µs** (block 16) / **32.6 µs** (block 32)
against `AGENTS` §10's "65.9/65.6 → 33.6/33.3" — the documented libgomp gain
reproduces.

---

## 4. x86 dense (m8i, 4 threads, Xeon 6975P-C)

### 4a. bf16 vs vLLM's AMX build

41 rounds with a control arm, min-based; doc column from `STATUS:16-23`.

| row | doc | measured min | measured med | band over 4 artifacts | verdict |
|---|---:|---:|---:|---|---|
| 128:8:64 | 0.87–0.93 | 0.854 | 0.930 | **0.85–1.25** | ✓ min / box unstable |
| 512:8:64 | 0.80–0.88 | 0.803 | 0.907 | **0.80–1.33** | ✓ min / box unstable |
| 1024:8:64 | 0.88–0.89 | 0.885 | 0.906 | 0.85–0.94 | ✓ |
| 2048:8:64 | 0.83–0.84 | 0.864 | 0.823 | 0.86–0.87 | ⚠ 3% high |
| 4096:8:64 | 0.70–0.72 | 0.802 | 0.822 | 0.78–0.80 | **✗ 11–15% high** |
| 256:16:128 | 0.99–1.05 | 0.981 | **1.090** | 0.96–1.09 | ⚠ median is the doc's "worst" |
| 1024:16:128 | 0.98–0.99 | **1.010** | 1.019 | 1.00–1.01 | ✗ 2–3% high |
| 2048:16:128 | **0.78–0.83** | **0.953** | **1.031** | 0.92–0.95 | **✗ 15–20% high** |
| 4096:16:128 | **0.83–0.85** | **0.976** | **1.020** | 0.97–1.00 | **✗ 15% high** |
| 512:32:128 | 0.99–1.01 | 1.018 | 1.023 | 1.01–1.02 | ✓ (loses) |
| 1024:8:32 | 0.84–0.85 | 0.827 | 0.869 | 0.83–0.88 | ✓ |

Count: doc 10/11, measured **9/11** by min (ctrl 0.92–1.04), 6/11 by median.
The two rows that lose are 1024:16:128 (1.010) and 512:32:128 (1.018), not the
doc's 256:16:128 (0.98 min, but 1.09 median). The largest disagreements are the
D=128 long rows: the docs claim a 15–22% *win*, this box measures 0–5% (min) or
a 2–3% *loss* (median). The 128:8:64 and 512:8:64 bands (0.85–1.25, 0.80–1.33)
show why a single 9-round process is not enough on this box: the same binary
moves 25–30% between processes there.

Caveat in the docs' favour: the STATUS x86 scoreboard says it was measured on
an Emerald Rapids 8581C (GCP), not on this 6975P-C. `AGENTS` §12 nevertheless
calls the 6975P-C "the Xeon", so the docs cite two different x86 boxes
interchangeably. The scoreboard count survives; the per-row values should be
read as 8581C-specific and the doc should say so.

### 4b. fp32 and bf16 vs vLLM's VEC build

| dtype | doc | measured (41r where available) | verdict |
|---|---|---|---|
| fp32 128:8:64 | worst ~1.00 | 0.946 min / 1.044 med (ctrl 0.93); 0.83–1.13 across four runs | ✓ at parity |
| fp32 all other rows | <1 | **0.55–0.79 min** (1024:8:64 0.563, 4096:16:128 0.731) | ✓ better than documented |
| bf16 all rows | — | **0.20–0.55 min** | ✓ |
| fp32 11/11 | 10–11/11 | 11/11 in 3 of 4 runs; the fourth read 128:8:64 at 1.13 | ✓ / box noise |

Absolute throughput on 4 cores: bf16 **0.65–2.54 TFLOP/s**, fp32
**0.35–1.00 TFLOP/s** (aggregate), i.e. 0.16–0.64 bf16 TFLOP/s/core against a
2.98 TFLOP/s/core `tdpbf16ps` roof. `STATUS:87`'s "kernel runs at 2.0–2.8
TFLOP/s against a 3.0 TFLOP/s dot roof" compares an aggregate (4 threads) with
a per-core roof; as written it implies 67–93% of the roof, while the measured
fraction is **17–21%**. vLLM's AMX attention is 0.66–0.74 TFLOP/s/core, i.e.
the two kernels are at parity, not 3–4× apart.

### 4c. vs torch SDPA

22/22 rows beat SDPA: bf16 o/sdpa **0.41–0.67**, fp32 **0.21–0.79** (1.5–4.7×
faster). `STATUS:32-33`'s "x86 by 1.6–3.7× (0.27–0.63)" is the wrong range for
this box (0.21 at 512:32:128 fp32 is outside it); "every row beats SDPA" holds.

---

## 5. x86 paged (m8i, 4 threads, block 32)

`o/vllm` min (median), independent harness, 15 rounds.

| case | bf16 | fp32 |
|---|---:|---:|
| self b=8 ctx=256 | 0.53 (0.79) | **1.04** (1.09) |
| cross b=8 ctx=256 | 0.59 (0.87) | **1.13** (1.12) |
| self b=8 ctx=1024 | 0.92 (0.90) | **1.20** (1.10) |
| cross b=8 ctx=1024 | 0.97 (0.94) | **1.15** (1.15) |
| self b=32 ctx=1024 | 0.66 (0.76) | 0.87 (0.90) |
| cross b=32 ctx=1024 | 0.85 (0.84) | 1.01 (1.00) |
| self b=1 ctx=4096 | 0.82 (0.86) | **1.08** (1.08) |
| cross b=1 ctx=4096 | 0.85 (0.94) | **1.11** (1.10) |
| self b=1 ctx=256 | 0.77 (0.80) | 0.46 (0.54) |
| cross b=1 ctx=256 | **1.20** (1.22) | 0.86 (0.73) |

The `AGENTS` §10 bf16 rows (self b8/256 0.67–0.70, cross 0.71–0.84, self
b8/1024 0.89–0.90, cross 1.00–1.02, self b32/1024 0.85–0.90) bracket these
numbers; two rows are better (b8/256, b32/1024), one is marginally worse
(self b8/1024 0.92), and the b8/256 min/median spread is 1.7×, so that row is
not resolvable beyond "1.3–1.9× faster". **fp32 loses on 7 of 10 cases**, so
the "parity-or-better elsewhere" claim (`AGENTS:113`) is false for the paged
kernel's fp32 row. The doc's absolute `cross b8/enc256` 82.5/85.5 µs is not
comparable (it was measured with the old private pool at 4 workers); this box
measures 67.3 µs bf16.

## 5b. Ryzen 7840H (the dev box, 8 threads)

Measured last, with the same harness after setting the cpufreq governor to
`performance` (and restoring `powersave` afterwards). Dense: 11 rounds. Paged:
40 consecutive calls per arm after 8 warmups, alternating blocks — the laptop
method, because a single-call median is dominated by the power state here (the
interleaved file is `audit/logs/ryzen/paged-perf.json`).

| shape | bf16 | fp32 |
|---|---:|---:|
| 128:8:64 | 0.88 | 1.12 |
| 512:8:64 | 0.75 | 1.04 |
| 1024:8:64 | 0.72 | 1.00 |
| 2048:8:64 | 0.72 | 1.03 |
| 4096:8:64 | 0.73 | 1.07 |
| 256:16:128 | 0.88 | 0.98 |
| 1024:16:128 | 0.89 | 1.20 |
| 2048:16:128 | 0.92 | 1.18 |
| 4096:16:128 | 1.07 | 1.68 |
| 512:32:128 | 0.91 | 1.31 |
| 1024:8:32 | 0.79 | 1.00 |

bf16 is 10/11 below 1; fp32 is 1/11. The fp32 D=128 long rows are **1.18-1.68x
slower** than vLLM VEC — this is the register-wall shape, and it is the laptop's
weak column (the servers win all 11 fp32 rows). Against SDPA, **21/22** are
below 1 (o/sdpa 0.36-1.04); the exception is `4096:16:128` fp32 at 1.04.

Paged `ours/vllm` (min, block method), self/cross: b1/256 bf16 **1.09 / 1.33**,
b8/256 **0.81 / 0.85**, b8/1024 **0.75 / 0.75**, b32/1024 **0.92 / 0.96**,
b1/4096 **0.66 / 0.66**; fp32 b1/256 **1.11 / 1.50**, b8/256 **0.75 / 0.78**,
b8/1024 **0.95 / 0.94**, b32/1024 **0.99 / 1.00**, b1/4096 **0.90 / 0.90**.
So the laptop paged kernel wins the batch rows and loses b1/cross, as on the
servers. The interleaved harness reads 0.67-1.10 min / 1.10-1.59 median on
b8/256 bf16; that spread is the power state, not the kernel.

---

## 6. Numerics and the cache-layout claim

| claim | measured | verdict |
|---|---|---|
| dense bf16 within 3e-2 of SDPA | vs naive fp32: **6.5e-3–1.6e-2**; vs SDPA the harness saw **3.12e-2** once (1024:8:64, x86) | gate holds vs truth; the SDPA gate can fail by chance |
| "bf16 is ~1 bf16 ulp" | max abs 6.5e-3–1.6e-2 on max abs output 3–4 → 0.4–1 ulp | ✓ |
| dense fp32 ≤4e-6 vs SDPA (`AGENTS:112`) | ARM 0.9–3.6e-6 ✓; x86 AMX **3.1e-5–7.7e-5** (README:341-343 itself says 5.5–7.6e-5) | ✗ on x86 / self-contradictory |
| paged bf16 bit-identical to csrc on x86 | 0.0 at b1/256; **2.4e-4–4.9e-4** at b8/b32 (1–2 ulp); fp32 1.5–2.6e-7 | ≈ (not literal at b>1) |
| "Page layout: vLLM's, unchanged" | bf16 caches differ on both ISAs: x86 ours `[d/2][t][2]` vs vLLM `[d/4][t][4]`; ARM ours `[d/2][t][2]` / `[t/2][d][2]` vs vLLM `[d/4][t][4]` / `[d][t][4]`. Cross-reading a cache gives max abs diff **0.26–0.43** (x86) and **1.28** (ARM). fp32 x86 caches are byte-identical. | ✗ for bf16 |

The layout difference is not a correctness bug — each backend stages its own
cache (`stage_cross_kv` does) and each is within 1e-3 of the naive reference —
but the README sentence is wrong and would break a user who hands a vLLM-staged
bf16 cache to this kernel (or the reverse). The engine is safe only because it
stages through the active backend.

`AGENTS` §8's "the cache layout stays ours (head-major, K transposed), not
flash's page-major one" is the correct statement and contradicts README's
"vLLM's, unchanged".

---

## 7. The AMX levers (m8i, 4 threads, bf16 unless stated)

Independent A/B, same process, toggling the documented switches between calls.
`on/off` = vector-kernel time / AMX time; `v1/v2` = v2 time / v1 time (above 1
means v1 is faster); `per-head` = split time / per-head time.

| row | AMX speedup | v1/v2 | per-head / split |
|---|---:|---:|---:|
| 128:8:64 | **2.80×** | 1.51 | 1.16 |
| 512:8:64 | 4.30× | 1.19 | 1.17 |
| 1024:8:64 | 4.27× | 1.07 | 1.08 |
| 2048:8:64 | 5.30× | 0.97 | 1.06 |
| 4096:8:64 | 8.28× | 0.95 | 1.05 |
| 256:16:128 | 4.85× | 1.38 | 1.14 |
| 1024:16:128 | 8.01× | 1.05 | 1.08 |
| 2048:16:128 | 9.24× | 1.02 | 1.07 |
| 4096:16:128 | 9.79× | 0.97 | 1.00 |
| 512:32:128 | 6.87× | 1.06 | 1.06 |
| 1024:8:32 | **3.18×** | 1.05 | 1.08 |

* **AMX 4–9.6×** (`AGENTS:755`): 8 of 11 rows are in that range, but
  128:8:64 is 2.80× and 1024:8:32 is 3.18×; 4096:16:128 is 9.79×. Quote
  2.8–9.8× or say "≥4× from 512 keys up".
* **Per-head item 1.47×/1.29×/1.16×**: measured 1.16/1.17/1.14 — the short-row
  gains are overstated by 20–30%; the long-row 1.03–1.07 claim holds.
* **v2**: 1.51× slower at 128:8:64 (claim 1.46×) but 5% faster at 4096:8:64
  and 3% faster at 4096:16:128; "0.93–0.99 elsewhere" is not what this box
  measures (1.19–1.38 slower at 512:8:64 and 256:16:128).
* **fp32 AMX hi/lo** is a real and large win: on/off 1.37× (128:8:64), 1.99×
  (1024:8:64), 2.61× (512:32:128), 3.38× (2048:16:128), 2.85× (4096:16:128).
* **kt**: not flat. kt=128 is **0.90** of kt=256 at 128:8:64 (i.e. 10%
  faster), 1.07 at 1024/4096:16:128; kt=384/512 are 0.94–1.07. The shipped
  256 is a reasonable compromise, but the STATUS "0.99/0.97/0.96 and flat"
  is not reproducible.
* **qt (bf16)**: 6/8/12/16 are within **1–3%** at 128:8:64, 256:16:128,
  512:8:64, 512:32:128, 1024:16:128 — the docstring table's 8–13% spread does
  not reproduce. qt=8 is still at/near the best.
* **qt (fp32)**: legal values are multiples of 4; **qt=8 is best everywhere**
  (1.45–1.61× over qt4: 128:8:64 0.69, 512:8:64 0.62, 1024:8:64 0.62,
  1024:16:128 0.64, 2048:16:128 0.66), and qt=12 is 12–18% slower than qt8.
  This supports the shipped 8.

### Latent bug found by the on/off A/B

Toggling `CPU_ATTENTION_AMX` from 1 to 0 and back in one process **segfaults**
after a few shapes (`_kernel.py:844`, `_AMX_PLAN(plan.data_ptr(), ...)`); the
v1↔v2 and `mode & 512` A/Bs are clean over all 11 shapes. Cause: when the
non-AMX path builds a plan and the workspace exceeds 24 MB, `_drop_workspace()`
frees the buffers but does **not** pop `_AMX_PLANS`, so the cached AMX plan
points at freed memory; `clear_scratch()` does pop it, `_drop_workspace()`
does not. Only reachable through the runtime toggle (the shipped default never
runs the non-AMX dense path once AMX is on), but the docstring advertises the
toggle for A/B, so a test harness can hit it.

---

## 8. Audit of the benchmark harness itself

`tests/bench_vs_refs.py` is honest about what it measures and its numbers
reproduce under an independent implementation within the noise floor. These
are the places where it can still mislead, in rough order of importance:

1. **The parity gate's reference is SDPA, not truth.** `parity()` compares our
   bf16 output to torch's bf16 SDPA with a 3e-2 gate. On m8i the gate fired
   (`3.12e-2`) at 1024:8:64 while *both* kernels were within 8.6e-3/6.7e-3 of
   the naive fp32 reference — the failure is the sum of two bf16 kernels'
   errors, not a wrong kernel. A naive blockwise reference (as in
   `audit/audit_common.py`) costs a second per large row and removes the false
   positive; alternatively raise the bf16-to-bf16 gate to ~5e-2 and report the
   naive-ref error beside it.
2. **Default `--rounds 9` is below the project's own recommendation.** On x86
   bf16 the same binary reads 6/11, 7/11 or 9/11 below 1 depending on the
   process and the 9/21/41-round protocol; on the short rows a process reads
   0.85–1.33. `STATUS`
   says min-of-13-21; the default should be 21, or `band.py` should run several
   processes.
3. **The printed headline ratio is the median; the verdicts are min.** Line
   228 of `bench_vs_refs.py` prints `o/vllm` from the median and appends "min
   o/vllm" separately; `scorecard.py` uses the min. The docs quote min, but a
   reader of the console output sees a different number (up to 10% apart on the
   contested rows).
4. **No assertion that the requested vLLM ISA was selected.** `--isa amx` on a
   non-AMX build, or a bad `--isa`, silently falls through to another kernel.
   A one-line sanity check (the AMX build should be ≥2× VEC at 1024:8:64) would
   prevent a flattering baseline.
5. **The AMX reference is built only by `scripts/verify.sh`, not by
   `box_sync.sh --full`.** A reader who follows the build section gets the VEC
   baseline only; the scoreboard's bf16 column is against a hand-built artifact
   whose flags are documented in one script and applied in another.
6. **`box_sync.sh x86_64 host --full` mis-parses `--full` as the local tree**
   (`$3` is the tree, `$4` is the flag). I hit this; the documented usage line
   is order-sensitive. Use `scripts/box_sync.sh <arch> <host> <dir> --full`.
7. **The baseline asymmetry is documented but easy to miss.** Ours pays the
   K/V pack and a fresh output allocation; vLLM's cache is staged and its
   output is preallocated before the timer; vLLM's `get_scheduler_metadata` is
   hoisted out of the loop. All three favour vLLM, so the ratios are
   conservative for ours — good, but the table is not "kernel vs kernel".
8. **No frequency/load record in the JSON.** On the Xeon box `/proc/cpuinfo`
   moved 2.7–3.9 GHz across cores within one run; the ratio between processes
   on the short rows moved 25–30%. The JSON records the thread count but no `cpu MHz`
   and no load average, so a future reader cannot tell a 1.25 from a 0.85.
9. **The SDPA arm is fair**: passing transposed views costs SDPA 0–1% (tested
   directly), so `o/sdpa` is not flattered by layout.
10. **The vLLM page layout is per-ISA**, so `--isa` choices also change the
    cache the reference reads; the harness stages with `args.isa`, which is
    correct, but it means the dense and paged comparisons are only valid
    against a reference built with the same `--isa` (and a different build
    flags set produces a different vLLM kernel).

---

## 9. Documentation contradictions to fix

These are pairs of statements that cannot both be true; they are not new
measurements, just consistency:

1. **ARM, vLLM gap.** `AGENTS` §4b (line 298) "still 1.44–1.8× behind vLLM"
   vs the top-of-file update and `STATUS` (0.87–1.01). Measured 0.79–1.02.
2. **SDPA counts.** `AGENTS` §10c (line 758) "Graviton 17/22 ... Xeon 4/22"
   vs `STATUS:32` "every row beats SDPA on both boxes". Measured 22/22 on both.
3. **x86, vLLM gap.** `AGENTS` §10c (line 759) "1.2–2.3× behind vLLM" vs the
   top-of-file bf16 10/11 and fp32 11/11. Measured: 9/11 bf16, 11/11 fp32.
4. **ARM, scoreboard.** `README:345` "8 of 22 beat vLLM NEON" vs `STATUS`
   "10/11 + 11/11". Measured 21–22/22 (8 threads) and 22/22 (4 threads).
5. **Page layout.** `README:195` "vLLM's, unchanged" vs `AGENTS` §8 "the cache
   layout stays ours". The latter is correct; the bf16 layouts differ.
6. **dense fp32 accuracy.** `AGENTS:112` "dense is ≤4e-6 vs SDPA in fp32" vs
   `README:341-343` "error 5.5–7.6e-5" for fp32 through AMX hi/lo. The first
   holds on ARM (0.9–3.6e-6) and not on x86 AMX (3.1e-5–7.7e-5).
   (`README:247`'s "~4e-6" is the paged kernel, which measures 1.5–2.6e-7.)
7. **`qt`.** `AGENTS` §4a "qt = 4 for every head dim" is already listed as
   retracted at the top, but the body line 236 still reads it; `_qt_for`
   returns 4–10 depending on ISA/dtype/shape/seq. The body should point to the
   table, not restate 4.
8. **x86 box identity.** `STATUS` scoreboard = Emerald Rapids 8581C (GCP);
   `AGENTS` §10/§12 = Xeon 6975P-C (AWS). The tables do not say which box each
   came from. On m8i the 8581C per-row values for the D=128 long rows do not
   carry over (15–20% off); the scoreboard count does.
9. **`2.0–2.8 TFLOP/s against a 3.0 roof`** (`STATUS:87`) mixes aggregate and
   per-core. Write "1.9–2.5 TFLOP/s aggregate on 4 cores = 0.5–0.6/core, i.e.
   17–21% of the 2.98 TFLOP/s/core `tdpbf16ps` roof, at parity with vLLM's
   0.66–0.74/core".
10. **AMX range.** `AGENTS:755` "4–9.6×" vs measured 2.8–9.8×. Either widen
    the range or say "≥4× from 512 keys".
11. **ARM box identity.** `STATUS` = m9g 8-core; `README:345`/`CEILINGS.md` =
    "4-core slice" (m9g.xlarge per CEILINGS). The two are not the same memory
    system, and `AGENTS` §10's 8/8 rows are the 8-core box. Say which box each
    ARM table came from. The 4-thread run in this report is a slice of the
    8-core box, not an m9g.xlarge.

---

## 10. What was not audited

* **End-to-end engine numbers** (README's tok/s tables, dispatch µs). The
  engine (fast-encdec + model weights) is not installed on the audit boxes, and
  the Ryzen end-to-end numbers were not re-run. The Ryzen kernel-level numbers
  *were* audited last — see §5b.
* **GCP Emerald Rapids 8581C rows** (the STATUS x86 scoreboard). m8i is a
  6975P-C; the two are not the same CPU, and the doc never says which table
  belongs to which.
* **Historical "before" numbers** (the "start of session → now" table, the
  old ARM table, v2-vs-v1 pre-per-head). The old trees are not present.
* **Profiling counters** in `PROFILE_X86.md` / `PROFILE_ARM.md`, and the
  `bfmmla` / `tdpbf16ps` microbenchmarks. Only whole-call throughput was
  re-measured.
* **The pack rate** (422–495 GB/s, 0.8–6.3% of a call): already retracted at
  the top of AGENTS and re-measured in `docs/REMEASURE.md`; I did not repeat it.
* **Numerics beyond max-abs**: no ulp distributions, no GQA/MQA, no causal
  boundary fuzz beyond what `tests/test_dense.py` does.

## 11. Reproducing

On a fresh box (after `scripts/box_sync.sh <arch> <host> <dir> --full`):

```bash
# independent dense matrix (naive reference, own timing loop)
.venv/bin/python audit/audit_dense.py \
    --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa neon \
    --rounds 41 --json /tmp/audit.json
# independent paged matrix (heads=16, dim=64, the engine's shapes)
.venv/bin/python audit/audit_paged.py --vllm ... --isa neon --block 16 \
    --rounds 15 --json /tmp/paged.json
# laptop paged (power-state-robust): 40-call blocks per arm
.venv/bin/python audit/audit_paged_block.py --vllm ... --isa vec --block 32 \
    --json /tmp/paged-laptop.json
# AMX levers (x86; --which onoff, v1v2, phase)
.venv/bin/python audit/audit_amx_ab.py --dtype bf16 --rounds 9
# qt sweeps
.venv/bin/python audit/audit_qt.py          # bf16 6/8/12/16
.venv/bin/python audit/audit_qt_fp32.py     # fp32 4/8/12 (ARM)
```

The repo harness, for comparison:

```bash
V=~/vllm_cpu_build/vllm_cpu_attn_bench.so
.venv/bin/python tests/bench_vs_refs.py --vllm $V --isa neon --rounds 41 \
    --control --json /tmp/repo.json
```
