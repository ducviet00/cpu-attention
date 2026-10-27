# C++ backend, 2026-10-28 continuation: findings and what is worth trying

Branch `cpp`. Primary box: Ryzen 7 7840H (8 cores, AVX-512 + BF16, no AMX),
8 threads, governor pinned to `performance` for every run and restored after.
Ratios are `ours / baseline`; below 1 is faster. All numbers below are minima,
with medians and raw samples in the JSON beside this file.

**Acceptance is still not met and Mojo is still the default.** The Ryzen now has
no row over the 1.04 ceiling, but the exact Graviton 5 / Xeon 6975P-C matrix
has not been run, and the GCP boxes (Axion Neoverse-V2, C4 Xeon 8581C) report
separately (`axion/`, `xeon/`).

## 1. What landed

| commit | change | measured |
|---|---|---|
| `5310012` | 10-row fp32 register tile **not adopted** (it was in the working tree) | 1.034-1.035 slower at 2048/4096:16:128 even with the partial-tile specialisation |
| `12c3ccd` | torch-aware paged entry `paged_call` (reads tensors, C++ geometry cache, `at::empty` output) replaces `paged_make_plan`/`paged_forward` | bit-identical; cross b=1 ctx=256 bf16 4.43 -> 3.24 us, fp32 5.32 -> 4.05 us; b=8 ctx=256 0.90-0.995 (`ryzen-paged-call-ab.json`) |
| `d5b0667` | vec fp32 dense: read V in place while all of V fits in LLC/2; streaming stores in the pack when all slabs exceed the LLC | see §2 (re-confirmed on the fixed harness) |

The binding now compiles against torch's headers and links `torch_python`,
`torch_cpu` and `c10` (`scripts/build_cpp.py`). Rebuild after a torch upgrade.

Validation on the final tree: dense 89/89, flash API green, fuzz 80/80 plus the
24/24 stray-read probes, paged sweep 504/504 (worst fp32 8.34e-7, bf16 1.96 ulp),
engine (`fast-encdec/tests/test_cpu_attention.py`) 45/45.

## 2. A/B evidence (fixed three-module harness, 41 rounds)

| change | row | new/old | ctrl/old |
|---|---|---:|---:|
| V in place | 256:16:128 fp32 | 0.973 | 0.986 |
| | 1024:16:128 fp32 | **0.951** | 0.987 |
| | 512:32:128 fp32 | **0.946** | 0.986 |
| | 2048:8:64 fp32 | 0.979 | 1.012 |
| | 1024:8:64 fp32 | 1.022 | 0.990 |
| streaming-store pack | 2048:16:128 fp32 | **0.962** | 0.997 |
| | 4096:16:128 fp32 | **0.973** | 1.000 |

Files: `ryzen-f32-vdirect-ab3.json`, `ryzen-f32-ntpack-ab3.json`. The
1024:8:64 reading (1.022, above its 0.990 floor) is the one row that may lose
1-2% from the in-place V. It is still 0.88 against the best baseline.

## 3. Ryzen scorecard on the final tree

Dense (`ryzen-dense.json`, one process per row, 41 rounds): **22/22 faster
than Mojo, 21/22 faster than vLLM VEC.** Against the best baseline, 18 rows are
clear wins and 4 are in the noise band:

| row | C++ us | best baseline | c/best |
|---|---:|---|---:|
| 2048:16:128 fp32 | 20 334 | vLLM 19 966 | 1.018 |
| 512:32:128 fp32 | 2 770 | vLLM 2 834 | 0.977 |
| 1024:16:128 fp32 | 5 197 | vLLM 5 343 | 0.973 |
| 4096:8:64 fp32 | 20 425 | Mojo 21 106 | 0.968 |

The previous checkpoint had these four D=128 fp32 rows at 1.031-1.155. bf16 is
0.49-0.81 everywhere.

Paged (`ryzen-paged.json`, 41 rounds x 40-call blocks, both backends
allocating, vLLM preallocated): **every row beats vLLM VEC** (0.41-0.99). Against
Mojo the b=32 ctx=1024 rows are at parity (cross 1.006 for both dtypes, self fp32
0.990) and **self b=32 ctx=1024 bf16 read 1.163**. That last row is not settled
(§4.3).

## 4. Findings

1. **Batch-1 paged calls were 40% Python.** At b=1 ctx=256 the whole call was
   4.6 us. The raw C++ entry was 2.6 us, of which 1.1 us is the OpenMP
   fork-join and the empty kernel. The other 1.25 us was the wrapper: hashing
   the plan key (0.35-0.5 us), six `data_ptr()` (0.26), `stride(0)` (0.11) and
   the call layers. `empty_like` added 0.55 us more. Doing all of it in C++
   fixed the b=1 cross row on every box at once. That row had been the only
   loss shared by all three boxes (1.09-1.20).
2. **fp32 dense D=128 was never a compute problem.** The QK and PV phases run
   at about 85% of the Zen 4 zmm-FMA peak (8.8 ms each at 2048:16:128 against a
   7.5 ms bound). Our compute alone was already faster than vLLM's whole call.
   The gap was the K/V pack, which vLLM does not pay inside its timer: 3.8 ms of
   20.7 ms at first, about 1.9 ms after `d5b0667`.
3. **Two A/B harness defects.**
   - `docs/cpp-20261027/ab_dense_variants.py` reused the new module's closure
     as its "control", so the control could not see module-level effects. Two
     byte-identical builds read **0.929** new/old at 1024:16:128 bf16. Every
     A/B under ~5% made with it is unconfirmed. The fixed harness
     (`ab_dense_variants.py` here) loads the control as a third module, a copy
     of the old build, in random order. Its identical-binary floor is
     0.977-1.005.
   - The large rows also move ~15% between *processes*. Self b=32 ctx=1024 bf16
     read 0.95-1.18 against Mojo across four processes while its in-process
     control stayed within 1-3%. Four separate cache allocations in one process
     agree within 1.4%, so tensor placement is ruled out. A one-process verdict
     on this row needs several processes, alternated.
4. **The store costs this row more than it should.** Store alone takes 54 us in
   both backends. But store followed by cross costs ~300 us more than cross
   alone for C++, against ~100 us for Mojo. That gap holds within one process,
   so the store perturbs the following DRAM stream. The mechanism is not
   identified.

## 5. Falsification ledger (this session)

| idea | result | status |
|---|---|---|
| 10 fp32 register rows (with partial-tile specialisation) | 1.034-1.035 at 2048/4096:16:128 | not adopted |
| V in place, ungated (slab halved, which also flips 2048:16:128 to fused) | 1.32x slower at 4096:16:128, 1.03 at 2048:16:128 | replaced by the LLC/2 gate |
| V-in-place gate at the full LLC instead of LLC/2 | **1.166x slower** at 2048:16:128 | rejected |
| fused order at 2048:16:128 (`CPU_ATTENTION_FUSE=1`) | 1.0-1.2% slower | rejected |
| pack/attend overlap: one dynamic list, per-head ready flags instead of the split barrier | 0.988-1.026 across 8 rows and both dtypes, ctrl 0.972-1.031 | rejected; `rejected-pack-attend-overlap.patch` |
| paged software prefetch at b=32 (`CPU_ATTENTION_PAGED_PF` 4/16) | inside process-to-process noise | not pursued |

## 6. Worth trying next, most promising first

1. **The dense entry in the same style as `paged_call`.** `flash_attn_func` still
   builds a Python plan key and makes `data_ptr()` calls. The short dense rows
   are 20-60 us, so 1-2 us is 3-5%. That is about the size of the remaining
   Xeon fp32 128:8:64 loss against Mojo (1.071) and the Axion bf16 128:8:64
   loss (1.005).
2. **Cheaper K pack for fp32 D=128 when it is DRAM-bound.** At 2048:16:128 the
   pack is ~1.9 ms of 19.6 ms and the row sits at 1.018. Options:
   - a bf16 K^T for the fp32 path with hi/lo split. This is the AMX idea on
     AVX-512; its accuracy budget needs checking first.
   - packing K per (head, 256-key tile) inside the first query block that needs
     it, i.e. a lazy pack. That keeps the slab in L2 rather than writing 32 MB
     to DRAM and reading it back.
3. **The self b=32 ctx=1024 bf16 store perturbation (§4.4).** First check
   whether it is the RFO of 17K scattered lines interleaved with the attend
   stream. Candidates:
   - order the stores so each task stores its token *after* attending its first
     pages, since the stored token is in the last page;
   - write the K token with streaming stores.

   Then A/B across several alternated processes, not one.
4. **2 MB-aligned workspaces with `madvise(MADV_HUGEPAGE)`.** The module-level
   bias in §4.3 suggests where a workspace lands matters by up to 7%. Aligning
   the dense slabs to huge pages could make that consistent, and possibly
   faster. It is cheap to test with the fixed harness.
5. **Re-run every pre-2026-10-28 dense A/B under 5% on the fixed harness**
   before building on it. That includes the partial-tile specialisation in
   `86e9ec6` and the 2026-10-27 tail experiments.
6. **Run the matrix on the exact boxes.** Graviton 5 / Xeon 6975P-C acceptance
   needs those boxes (one endpoint timed out; the other changed its host key).
   The GCP boxes are stand-ins and must stay labelled as such.

## 7. Box agents (merged into `cpp`)

Full write-ups: [`xeon/NOTES.md`](xeon/NOTES.md), [`axion/NOTES.md`](axion/NOTES.md).
Neither box ran the final 61/121-round matrix: both stopped early for the usage
limit. The rows below are therefore **not acceptance evidence**.

**Xeon 8581C (C4, AMX).** Merged `09f2631`, `5cab199`, `f1c4d8d` and
`d3af1f1` (all `cpu_attn_amx.hpp`), plus the shared `P::store_f32` hook
(`038bd69`).

- **Exp2.** Magic-constant rounding with a minimax degree-4 fp32 polynomial
  (rel. err 2.8e-6): fp32 0.899-0.978, bf16 0.947-0.986.
- **Non-temporal output epilogues.** The fp32 output store was 15-17% of a
  short call, because every line paid a read-for-ownership. With the hook,
  fp32 is 0.85-0.95 and bf16 0.88-0.98.
- **Dense target rows**, C++ / best, 31 rounds:

  | row | before | after |
  |---|---:|---:|
  | fp32 128:8:64 | 1.071 | 0.896 |
  | fp32 1024:8:64 | 1.012 | 0.840 |
  | fp32 1024:8:32 | 1.003 | 0.848 |
  | fp32 2048:8:64 | 0.972 | 0.871 |
  | fp32 256:16:128 | 0.978 | 0.759 |
  | bf16 128:8:64 | 0.967 | 0.838 |

  All 22 dense rows read 0.73-0.90 in that run.
- **Paged fp32.** b8/256 is already at the box's 4-thread read floor (139.6 us),
  so it cannot beat vLLM by 3% there. b1/4096 is 0.95-1.00, about 8% above its
  floor.
- `5cab199` (<5%) was measured with a biased harness and is **unconfirmed**.

**Axion Neoverse-V2 (c4a, NEON).** Merged `394ce24` (the dense NEON changes,
re-committed as `ef51486`) and the docs.

- **The dense change:** a leaner bf16 Q@K^T epilogue (196 -> 65 instructions
  per 8 keys), an integer-add exp2, and a contiguous fp32 Q panel.
- **Dense, 41-round `bench_cpp.py`.** 6 of 7 bf16 target rows are below 0.97
  of the best baseline (1024:8:64 at 0.977). fp32 512:32:128 moved 0.998 ->
  0.953. The fp32 D=64 rows are still level with Mojo (0.977-0.995).
- **Not merged: the `paged_chunk` hook and its driver patch**
  (`axion/impl-paged-chunk.patch`). Its own control does not separate it from
  the baseline: new/old 0.82-0.90 against ctrl/old 0.80-0.97, so the *old* arm
  is the outlier. The agent also found the paged b1/4096 timing depends on
  which cache tensor was allocated first (3-12% for the same binary). Re-test
  it on the fixed three-module harness with rotated allocation order before
  adopting it.

**Still worth trying, from the agents:**
- **AMX fp32.** Cache the Q hi/lo panel per work item instead of rebuilding it
  per key tile: 114 us of 1352 at 1024:8:64.
- **AMX.** Skip the softmax max pass after a row's first tile, on long rows.
- **NEON fp32 D=64.** Both FMA loops sit at exactly 16 accumulators; try an
  8-row x 12-key Q@K^T tile (24 accumulators).
- **NEON bf16 softmax.** It is ~27% of the call, and GCC adds `mov`s for the
  polynomial's constant accumulators.
- **Before any acceptance claim:**
  - re-run every <5% A/B on the fixed harness;
  - run the full 61/121-round three-box matrix on the integrated tree and
    render it with `scripts/cpp_scorecard.py --check`.

## 8. Final verification, all three boxes (integrated tree `4f07d2b`)

Same tree on every box, built there. Parity first, then the full 42-row
matrix against Mojo and the vLLM build for the ISA that box actually uses.
The rendered scorecard is [`SCORECARD.md`](SCORECARD.md); per-box raw JSON and
logs are in `ryzen-final/`, `axion-final/` and `xeon-final/`. The remote driver
is `remote_verify.sh`.

| box | baseline ISA | parity | clear wins | noise band | losses | acceptance |
|---|---|---|---:|---:|---:|---|
| Xeon 8581C (GCP C4, 4C/8T) | AMX bf16, VEC fp32 | dense 89/89, fuzz 80/80, paged 504/504 (7.75e-7, 1.96 ulp) | **39/42** | 3 | 0 | **PASS** |
| Axion Neoverse-V2 (GCP c4a) | NEON | dense 89/89, fuzz 80/80, paged 504/504 (8.05e-7, 1.92 ulp) | 35/42 | 7 | 0 | not met (83% < 90%) |
| Ryzen 7840H | VEC | dense 89/89, fuzz 80/80, paged 504/504, engine 45/45 | 37/42 (38/42*) | 3 | 2 (1*) | not met |

Rounds: servers 61 dense / 121 paged; Ryzen 41 dense (one process per row)
and 41 x 40-call paged blocks. Mojo's own suites pass on all three boxes.

\* **The Ryzen dense 128:8:64 fp32 matrix row (1.326) is a transient.** It ran
first, straight after the parity suites. Mojo and C++ both slowed about 1.7x
in that process while vLLM moved 14%. Three isolated re-runs read
**0.870 / 0.870 / 0.873** (control 0.997-1.007, `ryzen-final/rerun-*.json`),
matching this morning's 0.866. The rendered scorecard keeps the matrix value
unedited; with the re-run the Ryzen is 38/42 clear (90.5%) and one loss.

**What is still open:**

- **Ryzen paged self b=32 ctx=1024 bf16: 1.155 against Mojo** (ctrl 1.028;
  cross on the same row 1.002). This is the third session `bench_cpp.py` has
  read 1.155-1.163 here. Ad-hoc probes in other processes read 0.95-1.18, so
  the size is process-dependent, but the direction is consistent in this
  harness. It is the only real loss on any box, and §4.4 / §6.3 have the lead
  (store traffic perturbing the following DRAM stream).
- **Axion's 7 noise-band rows are all dense rows where Mojo is the faster
  baseline**, at 0.970-1.007:
  - fp32 512:8:64, 1024:8:64, 2048:8:64, 1024:16:128, 1024:8:32
  - bf16 1024:8:64, 1024:16:128

  vLLM NEON is beaten on all 42 rows. The 90% bar needs 3 of these 7 below
  0.97. The fp32 D=64 tile experiment in §7 is the lead.
- **Ryzen noise band:** dense fp32 2048:16:128 0.981 and 512:32:128 0.994 (vs
  vLLM VEC), and paged cross b=32 ctx=1024 bf16 1.002 (vs Mojo).
- **The exact Graviton 5 / Xeon 6975P-C acceptance boxes were not run** (one
  endpoint unreachable, one with a changed host key). These results are from
  the GCP stand-ins and are labelled as such.

**Default backend:** this does not meet the flip criterion on two of the three
boxes, so `CPU_ATTENTION_BACKEND` still defaults to `mojo`. Explicit `auto`
already selects C++ on x86-64, where the Xeon passes outright and the Ryzen's
only loss is the one paged row.
