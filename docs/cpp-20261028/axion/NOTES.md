# Axion (Neoverse-V2) NEON campaign, 2026-10-28

Box: GCP c4a spot `fe-arm-20261026-173549` (8 Neoverse-V2 cores, 2 MB L2/core,
80 MB L3). Remote tree `~/cpu-attention-arm-agent`, `.venv` symlinked. Only
`cpu_attn_neon.hpp` is committed; the paged scheduling change needs a driver
hook in the primary's `cpu_attn_impl.hpp`, proposed as `impl-paged-chunk.patch`.

Commits (on top of `cpp` @ d5b0667):

* `394ce24` bf16 Q@K^T epilogue, integer-add `exp2v`, contiguous fp32 Q panel.
* `85110ac` `paged_chunk` hook in both paged policies (inert without the patch).

Wrapped up early on the coordinator's request: **no final 61/121-round full
matrix was run**. The evidence is the interleaved A/Bs below plus one 41-round
`tests/bench_cpp.py --only dense` run of the committed tree with the patch
applied (`raw/dense-pgf.json`).

## Harness and its caveats

`ab.py` loads saved `_C_cpp.so` builds in one process, rotates arms, and adds a
`ctrl` arm. Paged arms each get their own staged cache (sharing the cpp cache
with ctrl gave a same-binary arm 15% in the first try), and the cache
allocation order rotates across epochs: the first-allocated 32 MB cache was
3-12% faster on the b=1/4096 rows irrespective of binary.

**Caveat:** `ctrl` reuses the *new* module object, the same structure the
primary found biased in `docs/cpp-20261027/ab_dense_variants.py` (identical
builds up to 0.93 new/old). So every dense A/B below that is **under ~5% is
unconfirmed**. Mitigating evidence: a same-file A/B
(`base` vs `base`) read 1.003 at 1024:8:64 bf16, and the bench_cpp run (which
does not have this structure, Mojo/vLLM baselines) shows the same direction.

`push.sh` syncs and builds on the box; `tab.py` renders the dense table.
Every timing was under `flock ~/bench.lock`, `OMP_NUM_THREADS=8
OMP_PROC_BIND=spread OMP_PLACES=cores`, block 16.

## Before / after

Dense, C++ / best(Mojo, vLLM), min (median). Before = the 2026-10-27 Axion
matrix (61 rounds), after = `raw/dense-pgf.json` (41 rounds, committed tree);
ratios are within-process. ctrl/cpp is the after run's duplicate arm.

| row | dtype | before µs | before c/best | after µs | Mojo | vLLM | after c/best | ctrl/cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 25.6 (28.4) | 1.005 | 23.1 (25.4) | 24.2 | 24.8 | **0.955** | 0.993 |
| 1024:8:64 | bf16 | 880.3 (907.2) | 0.994 | 863.2 (897.0) | 883.1 | 988.8 | 0.977 | 0.996 |
| 2048:8:64 | bf16 | 3416.8 (3449.1) | 0.991 | 3313.6 (3342.2) | 3447.2 | 3870.0 | **0.961** | 1.001 |
| 4096:8:64 | bf16 | 13430.6 (13916.6) | 0.981 | 13059.9 (13207.0) | 13625.1 | 15057.8 | **0.959** | 1.000 |
| 1024:16:128 | bf16 | 2904.2 (2946.4) | 0.990 | 2855.7 (2893.8) | 2945.8 | 3096.1 | **0.969** | 0.999 |
| 2048:16:128 | bf16 | 11235.8 (11702.7) | 0.970 | 11080.1 (11416.3) | 11546.3 | 12116.9 | **0.960** | 1.000 |
| 4096:16:128 | bf16 | 44885.1 (45088.5) | 0.963 | 43700.6 (43869.4) | 46707.3 | 48091.4 | **0.936** | 0.998 |
| 512:8:64 | fp32 | 552.8 (587.6) | 0.974 | 556.3 (587.5) | 569.6 | 639.1 | 0.977 | 0.999 |
| 1024:8:64 | fp32 | 2168.1 (2206.2) | 0.996 | 2154.6 (2194.1) | 2167.0 | 2449.2 | 0.994 | 1.000 |
| 2048:8:64 | fp32 | 8554.6 (8627.6) | 1.004 | 8504.3 (8541.8) | 8543.6 | 9673.1 | 0.995 | 0.999 |
| 1024:16:128 | fp32 | 8036.7 (8161.4) | 0.985 | 8100.9 (8195.8) | 8291.4 | 10487.5 | 0.977 | 1.000 |
| 512:32:128 | fp32 | 4279.3 (4467.0) | 0.998 | 4093.5 (4377.2) | 4294.2 | 5314.4 | **0.953** | 1.001 |
| 1024:8:32 | fp32 | 1253.4 (1281.7) | 0.976 | 1258.5 (1290.9) | 1279.1 | 1446.9 | 0.984 | 1.004 |

Non-target dense rows did not regress (all 22 in `raw/dense-pgf.json`, worst
c/best 0.942, 512:8:64 bf16). Status: 6 of 7 bf16 target rows now below 0.97 (1024:8:64 at
0.977); of the fp32 rows only 512:32:128 crossed. fp32 D=64 is still at parity
with Mojo.

Paged, old C++ (`c1r`, = commit 394ce24) vs new (`pgf`, 85110ac + patch), 63
rounds in 3 allocation epochs, `raw/pgf.json`, min (median):

| row | old | new | new/old | ctrl/old |
|---|---:|---:|---:|---:|
| self b1/4096 fp32 | 228.3 (242.5) | 201.3 (212.1) | 0.882 | 0.886 |
| cross b1/4096 fp32 | 221.5 (237.6) | 197.5 (209.2) | 0.892 | 0.880 |
| self b8/1024 fp32 | 486.5 (561.5) | 439.7 (522.5) | 0.904 | 0.965 |
| cross b8/1024 fp32 | 491.4 (559.5) | 436.8 (511.0) | 0.889 | 0.909 |
| self b32/1024 fp32 | 2838.3 (2879.5) | 2382.8 (2472.0) | 0.840 | 0.848 |
| cross b32/1024 fp32 | 2830.2 (2873.1) | 2351.8 (2452.4) | 0.831 | 0.852 |
| self b32/1024 bf16 | 1367.0 (1442.5) | 1125.6 (1193.5) | 0.823 | 0.796 |
| cross b32/1024 bf16 | 1364.3 (1424.4) | 1113.2 (1155.9) | 0.816 | 0.815 |
| others (b1/256, b8/256, b8/1024 bf16, b1/4096 bf16) | | | 0.96-1.04 | same path |

The b=1/4096 fp32 target rows were 260-268 µs against Mojo/vLLM 237-243 in the
old matrix; with the patch the C++ kernel is ~200 µs in this harness, but a
Mojo/vLLM rerun of those rows has **not** been done. The b8/1024 bf16 target
rows (0.967/0.996) are on the unchanged paged path; the `exp2v` change measured
0.92-0.94 there (`raw/exp1.json`, cross only, unconfirmed by a matrix).

Parity of the committed kernel (`raw/c1-parity.log`): dense 88/88, flash API,
fuzz 80/80 + 24/24 stray-read probes, paged sweep 504/504 (worst fp32 8.05e-7,
bf16 1.92 ulp). The scheduling change does not touch arithmetic.

## What each change measured (new/old, min; `raw/`)

| change | result | file |
|---|---|---|
| bf16 qk: 5 row-pair max registers + mask-free path (~196 -> ~65 epilogue instructions per 8 keys) | 0.96 at 128:8:64, 0.973-0.997 elsewhere, ctrl within 0.2% | `qkepi-bf16.json` |
| `exp2v`: 2^n as integer add into p's exponent (fcvtns+scvtf) | bf16 0.96-0.987, fp32 0.988-1.004 (0.945 at 1024:16:128), paged b8/1024 bf16 0.92 | `exp1.json` |
| fp32 qk: contiguous Q panel + single K tile pointer (bit-identical) | 0.985-0.999, 0.947 at 512:32:128 | `f1.json` |
| paged heads interleaved over threads when K/V > LLC/4 | 0.82-0.90 on the DRAM-sized rows | `pgf.json` |

Most single changes are under the 3% acceptance bar individually; they were
kept because they are consistent across rows and stack (bench_cpp above).

## Falsification ledger

| idea | result | mechanism / note |
|---|---|---|
| software prefetch of the next paged step's K/V (L1, L2, every line / every other line) | 1.03-1.10, no better than ctrl (`pf-screen`, `pf-rescreen`) | not the binding term; DRAM concurrency across threads is (next row) |
| `schedule(static, 1)` for every paged call | fp32 big rows 0.85-0.89, but b8/256 bf16 **1.85x slower**, b1/4096 bf16 1.06-1.11 (`s1-screen`, `s1r`, `pg`) | interleaving only pays when the call is DRAM-bound; gated on footprint instead |
| chunk = heads per 4 KB OS page (bf16 pairs) | still 1.5x slower at b8/256 bf16 (`pg`) | the OS-page hypothesis does not explain the small-row loss |

## Still worth trying

* fp32 dense D=64 (512/1024/2048:8:64, 1024:8:32) is at parity with Mojo. The
  perf profile puts P@V at ~35%, qk 40%, softmax ~15%; both FMA loops have
  exactly 16 accumulators (one 4-cycle-latency FMA in flight per pipe), so a
  24-accumulator qk (8 rows x 12 keys with ld1r Q) is the next experiment.
* bf16 1024:8:64 (0.977): softmax is still ~27% of the call; the remaining
  per-vector cost is the `mov`s GCC adds for constant accumulators in the
  polynomial (3 per vector).
* Validate the paged hook on the vec/AMX boxes before generalising it, and
  rerun the full 61/121-round matrix on the integrated tree.
