# Xeon 8581C (GCP C4, 4C/8T, AMX) — AMX dense + VEC paged, 2026-10-28

Box `fe-x86-20261027-112615`; OMP_NUM_THREADS=4, spread/cores, torch 4 threads.
All A/Bs are in-process, rotated permutations of five arms (old, new, a duplicate
of each, Mojo; vLLM VEC for paged). Times in µs, **min (median)**. Ratios new/old.

**Harness caveat.** `ab_so.py` / `ab_paged.py` duplicate arms reuse the *same*
loaded module as `new`/`old` (like the biased `cpp-20261027/ab_dense_variants.py`).
Treat every result below ~5% as **unconfirmed** until rerun with the fixed harness
(`docs/cpp-20261028/ab_dense_variants.py` on `cpp`, control = third module).

Pitfall found: the shared venv's editable install points at `~/cpu-attention`;
`tests/*.py` must be run with `PYTHONPATH=$PWD` (see `parity.sh`) or they test
another tree (and segfault on its stale `_C_cpp.so`).

## Profile (before)

Phase ablation (`abl.py`, `CPU_ATTENTION_AMX_MODE`), fp32, µs removed by skipping:

| row | total | pack | QK | softmax | PV | Q panel | mode 15 (driver only) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128:8:64 | 51.6 | ~5.5 | 6 | 11.6 | 5 | 1.5 | 16.2 |
| 1024:8:64 | 1352 | 100 | 418 | 560 | 300 | 114 | 64 |
| 1024:8:32 | 977 | 47 | 250 | 500 | 136 | 35 | 26 |
| 256:16:128 | 442 | 86 | 117 | 90 | 86 | 45 | 58 |

`perf` (cpu-clock, sudo): softmax 44% at 1024:8:32 fp32, hot on the P stores;
at 128:8:64 the fp32 output epilogue's stores were 18.5% of the attend lambda.
Removing the fp32 output store entirely: **0.852 / 0.832** of the call at
128:8:64 / 256:16:128 — the epilogue pays a read-for-ownership per line.

## Accepted (commits on `worktree-agent-a4d0ad07a460ec916`, rebased on `cpp` d5b0667)

| commit | change | new/old min (median), ctrl/new | json |
|---|---|---|---|
| 09f2631 | exp2: magic-constant rounding, minimax deg-4 fp32 (2.8e-6 rel, was 3.4e-6) | fp32 128:8:64 0.978 (0.989), 1024:8:64 0.934 (0.942), 1024:8:32 0.899 (0.941), 2048:8:64 0.938 (0.939), 256:16:128 0.974 (0.978); bf16 0.947-0.986; ctrl 0.99-1.01 | ab-exp1.json |
| 5cab199 | softmax emits P 32 keys/step (cvtne2ps + one zmm store), tail loops dead | long D<=64 rows 0.948-0.983 both dtypes (**unconfirmed**, <5%); short rows neutral | ab-cvt2.json |
| f1c4d8d | bf16 output epilogue: non-temporal stores (closing barrier drains WC) | 128:8:64 0.934 (0.899), 256:16:128 0.880 (0.876), 1024:8:64 0.935 (0.965), 2048:8:64 0.963, 1024:8:32 0.981 | ab-ntbf.json |
| d3af1f1 | fp32 `store_f32` NT epilogue in the AMX policy — **inert until** `proposed-impl-store_f32.patch` (shared `cpu_attn_impl.hpp`, optional `P::store_f32` hook) is applied | with patch: 128:8:64 0.852 (0.795), 256:16:128 0.842 (0.834), 1024:8:64 0.929, 1024:8:32 0.932, 2048:8:64 0.945; bit-identical | ab-ntf.json |

Parity after every accepted step (box, C++): test_dense 88-89/88-89, flash API
green, fuzz 80/80 (worst 0.52-0.57x tol), stray-read 24/24. Paged code untouched
(paged sweep not rerun).

## Target rows, combined tree (HEAD + store_f32 patch), bench_cpp 31 rounds

Not the requested 61/121-round final matrix (stopped on the usage limit).
c/best = C++ / min(Mojo, vLLM-ISA).

| row | before c/best | after C++ µs | Mojo | vLLM | c/best |
|---|---:|---:|---:|---:|---:|
| fp32 128:8:64 | 1.071 | 46.8 (50.6) | 52.2 | 60.9 | **0.896** |
| fp32 1024:8:64 | 1.012 | 1179.6 (1370.3) | 1404.4 | 2241.3 | **0.840** |
| fp32 1024:8:32 | 1.003 | 852.2 (943.1) | 1005.1 | 1348.3 | **0.848** |
| fp32 2048:8:64 | 0.972 | 4423.3 (5930.8) | 5077.8 | 8602.2 | **0.871** |
| fp32 256:16:128 | 0.978 | 372.5 (401.3) | 490.8 | 692.8 | **0.759** |
| bf16 128:8:64 | 0.967 | 24.7 (28.1) | 29.5 | 31.8 | **0.838** |

All 22 dense rows are 0.73-0.90 of the best baseline in that run (ctrl within
±3%). **Without the impl patch** fp32 128:8:64 stays ~1.0 of Mojo (ab-cvt2:
new/mojo 1.04; ab-ntf old/mojo 1.01).

## Paged fp32 (VEC) — not won, at the memory floor

`read_floor.py`: a plain 4-thread `torch.sum` over the same bytes is 139.6 µs
(b8/256, 16 MB) and 275.9 µs (b1/4096, 32 MB). On the rebased tree (primary's
`paged_call`): b8/256 cross C++ 139-143 vs vLLM 142-147 — **C++ is at the read
floor**, so <0.97 of vLLM is not reachable. b1/4096: C++ 295-303 vs vLLM 302-317
(0.95-1.00, vLLM is noisy), 8% above the floor.

## Ledger (measured, reverted)

| trial | result | json |
|---|---|---|
| fp32 kt 128/256 vs 512 (env, cross-process vs Mojo) | 256 neutral, 128 +6-10% slower at long rows | (log only) |
| prefetchw the item's output rows at item start | 256:16:128 0.954, but 128:8:64 1.029-1.069 | ab-pfw.json |
| NT bf16 stores + sfence per row | 1.25-2.4x slower | (log only) |
| pack + Q panel via cvtne2ps (`split_hilo32`, V via one vpermw) | 128:8:64 0.965, 1024:8:64 1.051, 256:16:128 1.015: mixed | ab-pack2.json |
| paged fp32 dim-outer QK (`CPU_ATTENTION_PAGED_QK=0` as default) | b1/4096 0.972-0.980, b8/256 0.992-1.009; error 3.6e-7 -> 6.6e-7 | abp-qk0.json |
| paged fp32 prefetch V lines during QK (T1) | 1.07-1.10 slower | abp-vpf.json |
| paged `CPU_ATTENTION_PAGED_PF` 4/8/16, `PAGED_KT` 256/1024/2048 | all within ±2% (cross-process) | (log only) |

## Still worth trying

* Cache the fp32 Q hi/lo panel per (item, chunk) instead of rebuilding it per
  key super-tile (114 µs of 1352 at 1024:8:64; grows with seq_k).
* Skip the softmax max pass after a row's first super-tile (stale max with a
  +64 overflow fixup); only pays at seq_k >> kt.
* Fork/finish skew: `gomp_team_barrier_wait_end` is 11% of samples at 128:8:64.
* Rerun the <5% wins (5cab199) with the fixed control-module harness, then the
  full 61/121-round matrix (`tests/bench_cpp.py ... --json`) and scorecard.
