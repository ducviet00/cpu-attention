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
