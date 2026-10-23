# AGENTS.md — Mojo paged-attention port (handover)

Everything learned building `_mojo/src/attention.mojo`: a Mojo port of the hot
path of `csrc/` (the vendored vLLM CPU attention kernel), wired into the engine
as `attn_backend="mojo"`. Read this before touching the kernel; the root
`AGENTS.md` covers the engine side.

> **Status of this file (2026-10-23, fourth campaign: Intel Xeon).**
> `docs/STATUS.md` is the scoreboard and this session's falsification ledger.
> **`docs/AUDIT.md` (2026-10-23) is an independent re-measurement** on fresh
> m9g (Graviton 5) and m8i (Xeon 6975P-C) boxes, with a naive fp32 reference
> and its own harness in `audit/`. It corrects the numbers below where marked.
> What the x86 fourth campaign changed, and where the kernel now stands:
>
> * **The AMX work item is one (batch, kv head)**: a thread packs that head's
>   K/V and then attends every query block of it, so there is no phase barrier
>   and the head's 32 KB of packed operands stay in L1 (`dense_amx.mojo`,
>   `per_head`). Re-measured 2026-10-23: **1.16x at 128:8:64, 1.17x at
>   512:8:64, 1.14x at 256:16:128** (the older 1.47/1.29/1.16 was measured on
>   the previous x86 box and overstates the short rows by 20-30%; long rows are
>   1.00-1.08x). `mode & 512` forces the old phase-split order for an in-process
>   A/B -- use it before touching dispatch.
> * **The AMX call entry takes five arguments**, not 36: `dense_amx_plan` reads
>   the geometry out of an int64 buffer that `_kernel._amx_plan_buffer` builds
>   once per (shapes, strides, dtype, causal, kt, workers) and caches per thread.
>   43.1 -> 37.0 us at 128:8:64. `tests/test_amx.py` requires the plan entry and
>   the 36-argument one to agree bit for bit.
> * **AMX machine facts**, measured three ways (`bench_amx_mma.mojo`,
>   `bench_amx_ceiling.mojo`, `prof_amx_pack.py`): a `tdpbf16ps` is 4.9-5.5 ns
>   (3.0-3.4 TFLOP/s/core, i.e. ~11 cycles), tile loads are free behind dots, and
>   **AMX and the vector pipes do overlap** -- a loop of 4 dots + 96 FMAs takes
>   the same time as the 4 dots. The item is 0.32 (Q@K^T) + 0.54 (softmax) +
>   0.30 (P@V) cycles/logit at D=128. The softmax runs 0.176 ns/logit isolated
>   and 0.265 in situ; that gap is the remaining 1.3x.
> * **Two restructuring attempts lost** and are recorded in the kernel and in
>   `docs/STATUS.md`: pipelining the softmax under the dots (0.73-0.91x) and a
>   cheaper exponential (`llvm.ldexp`/`vscalefps`, `round()`, degree-3) which was
>   8% faster in a single-threaded probe and 10-27% slower with four threads.
>   A probe that runs one thread cannot settle an inner-loop change here.
> * **vLLM's AMX build is the bf16 bar** on an AMX box; VEC is the fp32 bar.
>   `CPU_ATTENTION_AMX2=1` selects a second kernel (`dense_amx_v2.mojo`, agent A)
>   that is **1.51x slower** than v1 at 128:8:64 (older 1.46x) and reaches
>   parity or a 3-5% win on the long rows; the older "0.93-0.99 elsewhere" did
>   not reproduce on m8i. Re-measured AMX speedup over the AVX-512 path is
>   **2.8-9.8x**, not 4-9.6x: 2.80x at 128:8:64 and 3.18x at 1024:8:32.
>
> **Status of this file (2026-10-22, after the third dense-attention
> campaign, ARM/Graviton).** `docs/STATUS.md` is the current scoreboard and the
> falsification ledger; `docs/FINDINGS.md` is this session's independent machine
> measurements, the harness rules, and the call-path anatomy; `docs/ARM_V2_NOTES.md`
> and `NOTES-FAST.md`/`NOTES-BF16.md` in the agent worktrees have the per-row
> work. Read `docs/STATUS.md` before trusting any performance claim below.
>
> What the third campaign changed in the kernel (details and numbers in
> `docs/STATUS.md`):
>
> * **The work item is a whole (batch, head)** — one thread packs that head's
>   K/V and then attends every one of its query blocks, with no barrier between
>   the phases (`mode += 4`, `_kernel._dense_plan`). This, not a kernel change,
>   was the short-S answer: `perf` had put 29% of a 128:8:64 call in the OpenMP
>   barrier.
> * **K^T is packed `[tile][key pair][dim group][8]`** so one Q@K^T sub-block is
>   a linear run, and the tile is 384 keys at head_dim 128 (128 below it).
> * **A separate short-sequence kernel** (`dense_small.mojo`, `seq_k <= 256`)
>   rounds key coverage to 8 keys instead of 16 and packs in the same region.
> * **The bf16 exponential is degree-2** (2.3e-3 relative, against a bf16 store
>   two instructions later at ~4e-3), which moves bf16 `max|diff|` from 7.8e-3
>   to 1.6e-2 against the fixed 3e-2 gate. fp32 keeps the degree-4 form.
> * **Measure with the minimum, not the median.** On a shared box foreign load
>   moves medians 15%. `tests/bench_vs_refs.py` prints both, and
>   `scripts/band.py`/`scorecard.py` judge on the min. `scripts/verify.sh` waits
>   for the load average to drop below 0.8 before it times anything.
>
> Claims in this file that no longer survive measurement, with the number that
> kills each one in `docs/STATUS.md`:
>
> * §4a "qt = 4 for every head dim": ARM ships `qt = 8` for both dtypes
>   (`_kernel._qt_for`); 12/16 exist and do not ship.
> * §7/§9.3 "the Python path is 3.6 us and a plan cache measured worse": the
>   call path is 3.4 us, of which the extension marshal alone is 1.2 us (42
>   arguments); a per-geometry plan took the rest from 5.0 to 3.4.
> * §9.6 "the K/V pack is cold-read-bound, 85 GB/s and cannot be made faster":
>   the pack runs at 422-495 GB/s on 8 threads and is 0.8-6.3% of a call.
> * §13.1 "bf16 P@V / int8 is a maybe": int8 `smmla` is 3x `bfmmla`'s MAC rate on
>   Neoverse-V3 but misses the fixed 3e-2 gate at every scaling;
>   `tests/int8_qk_error.py` has the numbers and four adversarial families.
> * §14 "AMX from Mojo is the only route" stands, but fp16 matrix multiply
>   (`fmmla`) does not exist on any shipping core, so there is no fp16 path.
> * §4b/§9.10's ARM "the remainder is the softmax block": the exp is 15% of a
>   D=128 call and 29% of a D=64 one, and after the degree-2 change that block is
>   close to its floor. The residue on the D=128 long rows is the operand loads
>   that halve `bfmmla` (208 -> 105 MAC/ns/core) and the pack we are charged for.


## 0. Conventions (read this first)

**Performance ratios are `ours / baseline`.** A ratio **below 1 means we are
faster** than that baseline; above 1 means slower. Two baselines are reported,
in this order: `torch SDPA` and `vLLM csrc/cpu`. So `o/sdpa 0.62, o/vllm 1.20`
reads "1.6x faster than SDPA, 1.2x slower than vLLM". Every table in this file,
in `README.md` and in `tests/bench_vs_refs.py` uses this direction — the older
tables in this file were written as `baseline/ours` (a "speedup") and are being
converted as they are touched; where a number could be ambiguous the column name
says `o/<baseline>`.

Times are microseconds of the whole call. Ours includes the K/V pack and its
allocations; vLLM's is attention only (its cache is staged before timing), so we
are charged for work vLLM does once per request. Measure with
`tests/bench_vs_refs.py`, which interleaves the variants in one process — never
compare two separate runs on these boxes (they throttle 2-2.5x).

## 1. Status in one line

Two kernels, compile-time per-ISA paths (AVX512-BF16 `vdpbf16ps` + fp32 P@V on
x86; NEON `bfdot` QK **and** P@V on ARM): the paged decoder step
(`attention.mojo`) and dense attention (`dense.mojo`, `flash_attn_func`, §4a —
vLLM has no cache-free kernel to port, so that one is built from vLLM's
structure). Paged: bf16 output matches `csrc/` to **0-2 bf16 ulp** on x86
(bit-identical at b1/ctx256, one ulp at b8/ctx1024) and to ≤1e-3 on ARM;
fp32 agrees to 1.5-2.6e-7. Dense: fp32 is ≤4e-6 vs SDPA on ARM but
**3.1-7.7e-5 on x86 AMX** (hi/lo splitting), and bf16 is ~1 ulp;
**every row beats SDPA by 1.1-4.7x** across the 22-row matrix. At `cross
b8/ctx256` bf16 it beats `csrc/` by **1.4-1.7x on the 4-core Xeon and
2.3-2.4x on Graviton 5**; b1 traffic and x86 fp32 paged are not always wins
(`docs/AUDIT.md` §3, §5). It runs on
torch's own OpenMP pool (`GOMP_parallel`), which makes the engine **faster than
the vendored C++ backend end to end** on the laptop: 392-396 vs 364-377 tok/s at
8x128 (§9.3), 29.8 vs 90.7 µs per attention call at b8/ctx256 (§7).

## 2. Layout

```
cpu_attention/
  __init__.py               public API (flash-attn names + low-level ops)
  flash_attn_interface.py   flash_attn_with_kvcache / flash_attn_func / rejects
  ops.py                    self_attention / cross_attention / reshape_and_cache
  _kernel.py                .so loader, paged + dense calls, worker policy
  _C.so                     built kernel (gitignored; `scripts/build.py`)
  _mojo/
    src/attention.mojo      paged kernel (Job, store, generic + dot attention, dispatch)
    src/dense.mojo          dense kernel (DenseJob, q-block sweep, K transpose pass)
    src/_C.mojo             Python extension: PyInit__C, `attention`, `dense_attention`,
                            `dense_transpose`, `dense_bf16_dot`
    tests/test_attention.mojo  standalone parity test vs a scalar fp32 reference
    tests/bench_qk.mojo     isolated QK micro-benchmark (fp32 vs dpbf16ps)
tests/test_flash_api.py     the flash-shaped API vs an SDPA reference
tests/test_dense.py         dense kernel vs SDPA (41 checks: shapes, GQA, masks)
tests/bench_dense.py        dense kernel vs SDPA, the table in README §Dense
README.md                   user-facing: design, per-box tables, "what not to redo"
AGENTS.md                   this file
```

How fast-encdec consumes it: `import cpu_attention` (installed editable with
`uv pip install -e ../cpu-attention --no-deps`), its `layers/mojo_ops.py`
stages the per-step Context and calls `self_attention`/`cross_attention`;
dispatch lives in `layers/attention.py` and compile gating in
`engine/model_runner.py` (`compiled_backend`).  The engine test is
`../fast-encdec/tests/test_cpu_attention.py`.

## 3. Build, run, test

```bash
uv sync --extra build                                  # torch + the mojo wheel
uv run --extra build python scripts/build.py --tests   # _C.so + 11 kernel cases
```

`mojo` comes from the PyPI wheel (`[project.optional-dependencies] build`), so
uv is the only tool; there is no pixi env any more.

The extension leaves the libgomp symbols undefined (they resolve at load time;
`cpu_attention` imports torch first, which loads `libgomp.so.1`), while the
standalone test binary links them — `scripts/build.py --tests` uses the system
`libgomp.so` or torch's bundled `libgomp.so.1`. `mojo run` **cannot** run the
kernel (its JIT cannot resolve `GOMP_parallel` / `omp_get_thread_num` and fails
with `Symbols not found`), which is why the build script produces a binary. The
backend is Linux-only: no Windows/macOS libgomp, and the mojo wheel's target is
the host CPU.

From this repo's root (needs the `.so` above):

```bash
python tests/test_flash_api.py          # flash-shaped API vs SDPA (35 checks)
python tests/test_dense.py              # dense kernel vs SDPA (41 checks)
python tests/bench_dense.py             # dense kernel vs SDPA, two dtypes
```

Engine-level parity and the kernel A/B live in fast-encdec:

```bash
cd ../fast-encdec
PYTHONPATH=. .venv/bin/python tests/test_cpu_attention.py   # 45 checks, engine parity
PYTHONPATH=. .venv/bin/python benchmarks/kernel_ab.py       # drift-controlled A/B
```

`mojo build` targets the **host CPU** (`--print-effective-target` shows
`znver4 +avx512bf16` here), so rebuild per machine. `CPU_ATTENTION_LIB`
overrides the `.so` path; `CPU_ATTENTION_WORKERS` overrides the worker count.

## 4. The kernel in one page

Contract (enforced by the engine): **exactly one query token per sequence**,
attending to every cached position. That makes the causal mask the cache tail.

- Work item = one `(sequence, head)` pair, dispatched with `GOMP_parallel` on
  **torch's own OpenMP pool**, `num_workers = min(torch threads, physical cores,
  tasks)`; one worker (inline, no region) when `num_tasks * pages <= 32`. Each
  task owns one head's cache slice end to end → no KV split, no scratch pad, no
  cross-thread reduction.
- KV is scored in **tiles** of `tile` positions (`32` when `block_size % 32 == 0`,
  else `16`) and `GROUP = 4` tiles per online-softmax step.
- A **page** holds `block_size / tile` tiles. The group loop resolves
  `(page, offset)` per tile into `tile_page[]`/`tile_off[]` and shares it between
  Q@K^T and P@V. **Walking pages instead of tiles is the bug from §11.2.**
- Page layouts (all inside one flat page of `block_size * head_dim` elements):
  - V, portable/x86: `[block_size, head_dim]` row-major.
  - V, ARM dot path: `[block_size/2, head_dim, 2]` — `(v[j][d], v[j+1][d])`
    adjacent, because `bfdot` contracts two adjacent values.
  - K: `[head_dim/2, block_size, 2]` (dot paths) or `[head_dim, block_size]`
    (fp32 fallback). One token is a column.
- Online softmax: `m' = max(m, max(group))`, `l = l·exp(m-m') + Σexp(group-m')`,
  `acc = acc·exp(m-m') + P@V`. Probabilities stay **in registers**; the C++
  kernel stores and reloads them (a 4-byte reload out of a 64-byte store = a
  store-forwarding stall per position).
- `scale` is folded into Q on the fp32 path; on the dot path Q stays bf16 and
  `scale` multiplies the logits (never round Q through a scaled multiply).
- **Strides:** `q`/`k`/`v` are chunks of the fused `qkv_proj` output, so the
  token stride is `3 * d_model`, not `num_heads * head_dim`. The kernel takes
  `q_stride`/`kv_stride`/`dst_stride` explicitly. Assuming contiguity silently
  gives every sequence sequence 0's output (§11.1).
- Supported: `head_dim` 32/64/128 (compile-time specialisations), fp32/bf16,
  `block_size` a multiple of 16. fp16 is rejected (no `avx512_fp16` on these
  boxes).


## 4a. The dense kernel (`dense.mojo`, `flash_attn_func`)

vLLM has **no** cache-free attention kernel: the only attention entry points in
`csrc/cpu` are `cpu_attention_with_kv_cache` and `mla_decode`, both paged, and
its "dense" prefill is the same kernel fed `query_start_loc` with many query
tokens per request (Q-tile `MaxQHeadNumPerIteration` = 8 on `VEC`, 32 on AMX).
Its `use_sdpa_prefill` is deprecated and the zentorch encoder path is a
delegation to torch. So the dense kernel is built from vLLM's *structure*, not
ported file-for-file.

- **Work item = `qt` consecutive query rows of one (batch, head), swept over the
  whole key range**, so the `head_dim`-wide output accumulator stays in
  registers and K/V are read once per row block. `qt` is chosen by
  `_kernel._qt_for` per (ISA, dtype, head_dim, seq_k) — 4-10 on x86, 8 on ARM
  for the L1-tile kernels; `CPU_ATTENTION_QT` overrides it. `qt * head_dim`
  above ~512 elements spills the register accumulator, which is why the
  register path caps near 4-8 rows and the rest use the L1-accumulator
  kernels. Re-measured 2026-10-23: fp32 `qt=8` is 1.45-1.61x over qt4 on ARM;
  bf16 6/8/12/16 are within 1-3% at every tested shape.
- **K is transposed first** (`dense_transpose`, blocked 16 tokens = one 64-byte
  store, 40-70 GB/s against 11-15 GB/s for `torch.permute(...).contiguous()`;
  the Python side just allocates the scratch). That is what lets one
  broadcast-FMA cover `KT_TILE = 64` positions — 32 was 6% slower on bf16 and
  16% on fp32. V stays row-major.
- **Causal mask is bottom-right aligned**: row `i` keeps keys `j <= i + (seq_k -
  seq_q)`. The k-loop stops at `q0 + nrows + diag`, and only the boundary tile
  builds a per-row lane mask; rows whose limit is negative write zeros.
- **P@V only walks keys that exist** (`min(KT_TILE, seq_k - t0)`): the masked
  lanes have probability exactly 0, but reading their V would run past the
  tensor and `0 * garbage` is not 0 (§11.6).
- **Dispatch is round-robin** (`task += threads`), not a contiguous slice, for
  the same reason: causal work grows with `q0`, so a contiguous slice leaves
  half the cores idle at the end of a prefill.
- **Three row kernels, chosen at compile time**: `_dense_rows` (acc + logits in
  registers, used when `qt * head_dim <= 384`), `_dense_rows_mem` (acc in L1 and
  the head dim tiled by 32, used above that -- head_dim 128 -- so nothing
  spills; within 1% of the register version, kept because it is the no-spill
  path and it is what makes `qt = 8` usable there) and `_dense_rows_dot` (bf16).
- bf16 uses the dot instruction for Q@K^T and fp32 for P@V, like the paged
  kernel, which is why **bf16 is faster than fp32** here (936 vs 700 GFLOP/s at
  S=2048, H=8, D=64) — a bf16-native kernel on AVX-512 is slower than fp32.

## 4b. The dense kernel on ARM (measured on a Graviton 5 / Neoverse-V3)

The dense kernel was written and tuned on AVX-512 and it showed: against SDPA it
was 0.24-0.50x on bf16 and 0.65-1.24x on fp32, and 2-4x behind vLLM's NEON
kernel. Three fixes, in order of payoff:

1. **Tile sizes must be lane-relative, not x86-relative.** `KT_TILE` was a flat
   64, i.e. 4 registers of positions on AVX-512 but 16 on NEON; with `qt = 4` a
   block wanted 64 logit + 64 accumulator registers on a 32-register machine and
   spilled everything. `KT_TILE = 4 * LANES` (64 x86 / 16 NEON) plus `qt = 2` on
   ARM (`_lanes() < 16` in `_kernel.py`) fixed that.
2. **On ARM the accumulator goes to L1 for every head dim** (`_dense_rows_mem`
   with `dc = 2 * LANES`): `head_dim / LANES` floats per row is 16 registers at
   head_dim 64 whatever `qt` is, so the register kernel cannot hold it.
3. **bf16 keeps its dot instruction inside that kernel** (`pairs: Bool` on
   `_dense_rows_mem`): the paired K layout and `bfdot` for Q@K^T, which is what
   the Q-outer register kernel already did. Without it bf16 read the unpaired
   layout Python never wrote (18/41 parity checks failed).

Together: 16-32% on ARM (S=1024/H=16/D=128 bf16 18.6 -> 12.6 ms; fp32 S=512
1121 vs 1410 us).

Then the real fix landed: **`_dense_rows_bfmmla`**, `bfmmla` on both phases
(Q@K^T 2 rows x 4 dims vs 2 keys x 4 dims, P@V 2 rows x 4 keys vs 2 dims x 4
keys), so a task lives in 2x2 fp32 tiles, the logits *are* the Q@K^T
accumulators and the softmax handles a row pair at a time. K gets a
position-paired layout `[head_dim/4][seq/2][8]` and V a position-major one
`[head_dim/2][seq/4][8]` (P@V contracts over keys), both packed in the one
transpose pass for 1-2% of the call. Plus `_exp2_fast` (magic-constant rounding
+ degree-4 polynomial + exponent insert, ~10 instructions against
`std.math.exp`'s ~22, which was **20% of the call**; the clamp at -127 makes
masked lanes exactly 0 via exponent underflow, and one `fmaxnm` removes the
"no visible key" special case) and `_vmax` as a single `fmaxnm`.
The campaign's result was **2.2-3.2x**, 331 -> 853 GFLOP/s (D=64) and 348 ->
1093 (D=128) at the time; it left the kernel **1.44-1.8x behind vLLM**. That is
history: re-measured 2026-10-23 on m9g (8 threads) the merged tree is
**0.79-1.02 vs vLLM** and **1.34-1.59 TFLOP/s** at the long rows, with bf16
0.87-1.01 and fp32 0.79-0.98 (`docs/AUDIT.md` §2). A 41-round run puts 9-10 of
11 bf16 rows and 11 of 11 fp32 rows below 1, the bf16 D=128 long rows at
0.99-1.01. The residue analysis below is from the superseded campaign; the
numbers are kept because the mechanism (operand loads, spill traffic) is still
the right place to look if the remaining 0-5% on those rows matters.

At the time the explanation was: we are 5-13% behind SDPA at D=64/S>=2048,
their `.so` runs 2 `bfmmla`/cycle where we run
1.2-1.35, and they issue ~16 instructions per row-key against our ~21 at IPC
3.7 vs 2.6. About 300 instructions per 16-key tile are **spill traffic**:
`qt/2 * sub/2` 2x2 tiles plus the softmax working set exactly fill the
32-register file, and the static asm still shows `str/ldr q, [sp, ...]` around
every tile. The lever is a register-lean (vLLM-style) softmax with the logits in
L1 and one row pair live at a time — §13.4. fp32 on ARM has no matrix-multiply
instruction, so it stays at ~250 GFLOP/s against vLLM's 480 and needs §13.3.

**Rows per work item is the one lever that moved ARM fp32**: the default was
`qt=2` (the worst measured point). Measured on m9g, fractions of `qt=4`,
interleaved, medians of 5: `qt=2` 0.77-0.80x, `qt=6` **1.09-1.11x**, `qt=8`
0.76-0.78x (register wall). `_kernel._qt_for` now picks 6 for fp32 at head_dim
<= 64 and `seq_k >= 1024`, and the drift-controlled A/B of the *whole call* is
**1.37-1.39x** (S=1024/2048, H=8, D=64) — that row went from 0.79x SDPA to
**1.15x**. head_dim 128 deliberately stays at 2 (32 registers per row already).
For bf16 it returns 8 rows at head_dim 128 and 4 at 64, where the `bfmmla` kernel
was already optimal.

**Decoupling `qt` from the tile geometry on ARM was tried and rejected** (the
worker's own abandon criterion fired): at fixed `qt=8`, `sub=16/dcp=8` measures
0.862x against the merged `qt=4/8,16` and 0.940x against the old coupled
`qt=8/4,8` — `RP=4 x NS=8` is 32 logit registers before any operand, so the
allocator spills more than the 2x key-tile amortisation returns. With `qt` at its
per-dtype optimum (4 bf16, 6 fp32), the residual 3.76 cycles/row-key at D=64 are
**not** recoverable through row count or tile shape: they are spill and latency
inside the 23% softmax block.

Also: the ARM build needs **libgomp linked into the extension**. Leaving `omp_*`
undefined for load time works on x86 because torch brings a global copy, but on
aarch64 torch's bundled `libgomp.so.1` is `RTLD_LOCAL` and `import` fails with
`undefined symbol: omp_get_thread_num`. `scripts/build.py` now links it (same
soname, so the loader still reuses torch's copy when it is already loaded).

## 4c. What vLLM actually does (read from `csrc/cpu`, not inferred)

Six things, with file:line. Only the first is structural; the rest are cheap and
we do none of them.

1. **The key tile is L2-sized, 480-1024 keys** — ours is 16 (NEON) / 64
   (AVX-512), so we pay every per-tile fixed cost **8-64x more often**.
   `calcu_default_tile_size` (`cpu_attn_impl.hpp:962`) is
   `L2 / (head_dim * (8 + 2*elem) + 32)`, capped at `MaxQTileIterNum(128) *
   MaxQHeadNumPerIteration(8)` = 1024, rounded to 32: **992 keys at D=64 fp32,
   480 at D=128 fp32, 1024 / 640 for bf16**. One `tile_gemm<QK>` call writes the
   whole tile's logits to a buffer, then `apply_softmax` runs **once per tile**,
   then `tile_gemm<PV>` consumes it. The per-tile cost we repeat 16-64x per key
   block: the row-wise `reduce_max`, `reduce_add`, the accumulator rescale, and
   the mask.
2. **Mask by range, not per lane** (`apply_mask`, `cpu_attn_impl.hpp:1360`):
   the causal/window mask is contiguous, so they write `-inf` into
   `[0, left_invalid)` and `[right_invalid, n)` with scalar stores. We do
   `iota()` + compare + `select` over **every lane of every tile** — on NEON
   that is 4x more often per key than on x86 and it was counted in the ~770
   instructions/16-key-tile the ARM worker measured.
3. **Rescale only when the max moves by more than 8**: `need_rescale =
   (init_max - new_max) < -8.0` (`cpu_attn_impl.hpp:1468`, "same rescale
   threshold with FA4"), and when it does not fire they do not even update the
   running max. We rescale on *any* increase.
4. **Probabilities alias the logits buffer** (`curr_prob_buffer =
   reinterpret_cast<...>(logits_buffer)`, `prob_buffer_stride`,
   `cpu_attn_impl.hpp:1318-1330`): one buffer, no copy, and P@V reads it through
   `probability_token_store_t::store_probabilities`. We keep a separate `probs`
   stack buffer and re-read it scalar by scalar in P@V.
5. **Work items are packed by KV length, with KV splits and a reduce pass**
   (`AttentionScheduler::schedule`, `cpu_attn_impl.hpp:500-700`):
   `kv_len_per_thread = total_kv_len / thread_num`, one work item accumulates
   *many* q-tiles until that budget is spent, a single q-tile longer than the
   budget is split (`min_split_kv_len`, `local_split_id`) and a
   `PartitionReduction` work item combines the partials. We have no split-KV path
   at all, and we balance causally with round-robin instead of by measured KV
   length. P@V is also d-chunked by `headdim_alignment = 32` with `accum_c`
   ("accumulate onto the first iteration's output"), i.e. their accumulator
   lives in `partial_q_buffer` and is never separately zeroed.
6. **Their `fast_exp` is exactly our `_vexp`** (`cpu_arch_macros.h:20-60`:
   degree-5 polynomial, exponent bit-insert, clamp at the float min/max), and on
   aarch64 bf16 they use a cheaper `fast_exp_f16`. So the transcendental is
   settled — do not spend more time there.

## 5. One source, two ISAs

```mojo
comptime X86_DOT  = CompilationTarget.is_x86() and CompilationTarget._has_feature["avx512bf16"]()
comptime NEON_DOT = CompilationTarget.has_neon() and CompilationTarget._has_feature["bf16"]()
comptime BF16_DOT = X86_DOT or NEON_DOT
comptime LANES = simd_width_of[DType.float32]()   # 16 AVX-512, 4 NEON
```

One dot instruction consumes `2 * LANES` bf16 (= two head dims of `LANES`
positions), so `tile / LANES` accumulators per page — 2 on x86, 8 on NEON for a
32-token page. The intrinsic is the only per-ISA code:

| ISA | intrinsic | signature |
|---|---|---|
| x86 | `llvm.x86.avx512bf16.dpbf16ps.512` | `<16 x f32>, <32 x bf16>, <32 x bf16>` |
| ARM | `llvm.aarch64.neon.bfdot` | `<4 x f32>, <8 x bf16>, <8 x bf16>` |

Both via `std.sys.intrinsics.llvm_intrinsic[..., has_side_effect=False]`.
**Without `has_side_effect=False` the calls are ordered and the phase is ~4×
slower.** The fp32 broadcast-FMA path stays as the fallback (AVX2, pre-BF16
ARM).

Verify both targets without the hardware (this is the cheap CI check):

```bash
cd mojo
uv run --extra build mojo build --emit llvm _mojo/src/_C.mojo -o /tmp/x86.ll
grep -c dpbf16ps /tmp/x86.ll                     # 73 on the current tree
uv run --extra build mojo build --emit llvm --target-triple aarch64-unknown-linux-gnu \
    --target-cpu neoverse-v3 _mojo/src/_C.mojo -o /tmp/arm.ll
grep -c neon.bfdot /tmp/arm.ll                   # 5665 on the current tree
```

The `--target-cpu` is **not optional**: the default aarch64 target has no
`+bf16`, so `_has_feature["bf16"]` is false and the build silently uses the
portable path while still compiling. Same trap on x86 (`--target-cpu
x86-64-v2` → fp32 path, `0` dots).

Checking a built `.so`: `objdump -d build/fastencdec_mojo_attn.so | grep -c
vdpbf16ps` (72 here) or `grep -c bfdot` (5664 on Graviton).

## 6. Mojo language notes (things that cost time here)

- `fn` is gone → `def`; `@parameter` → `@__parameter`, and **do not** put
  `@__parameter` on nested closures: use unified closures with capture lists
  (`{imm}`, `{mut x, imm}`, `{var x}`). A nested `def` with no capture list
  fails with "Could not infer capture convention".
- **No mutable globals.** `var X = ...` at module scope is an error. Module-level
  `comptime` constants are fine. This is what blocks a hand-rolled thread pool
  without a C shim (state can be smuggled as a Python-held `Int` address, §9.3).
- **`comptime` ternaries do not fold into comptime integers.**
  `comptime LANES = 16 if X86 else 4` stays a runtime expression and every
  `SIMD[.., LANES]` fails to match. `simd_width_of[DType.float32]()` folds.
- **`comptime if` arms are type-checked even when dead.** A 32-wide value cannot
  appear in a `tile`-generic function, so the dot kernel is a separate
  `_attention_head_dot` with `tile` fixed per instantiation, not a branch inside
  the generic kernel. (Module-level comptime booleans *are* fine as conditions
  when both arms typecheck for every instantiation.)
- Runtime index into a `SIMD` (`v[i]` with runtime `i`) spills the vector to the
  stack — use `comptime for i in range(N)` so indices are constants.
- `SIMD` comparison operators are **reducing** (`a < b` → `Bool`). For per-lane
  masks use `.lt/.le/.gt/.ge/.eq/.ne` (→ `SIMD[bool, N]`) and then
  `mask.select(true_vec, false_vec)`. `iota[DType.int32, N](offset)` builds an
  index vector.
- SIMD helpers used here: `.cast[DType]()` (widening bf16→f32), `.join` (concat),
  `.interleave` (`a0,b0,a1,b1,...` — how the broadcast pair operand is built),
  `SIMD[dtype, N](scalar)` (splat), `.reduce_max()`, `.reduce_add()`.
- Memory: `stack_allocation[count, dtype]()`; `Array[T, N](uninitialized=True)`
  (promoted to registers when every index is comptime);
  `ptr.unsafe_load[width=W](offset)` / `unsafe_store[width=W](offset, val)`
  (`width` defaults to 1; a value-only store overload exists).
  `Pointer` is non-nullable; build one with `unsafe_from_address=`, origin
  `MutUntrackedOrigin`; `Int(ptr)` gives the address.
- FFI: `external_call["symbol", RetType](args...)`; `@export def f() abi("C")`;
  a thin function value can be taken (`comptime entry = worker_entry`) and passed
  to C — that is enough for `pthread_create`/`pthread_join` and for the whole
  libgomp dispatch (`GOMP_parallel(entry, Int(state), Int32(n), Int32(0))`).
- **`fn` is a reserved word**: a field, parameter or `comptime` binding named
  `fn` is a parse error (`var fn: ...` → "expected name for 'var' declaration";
  `comptime fn = f` → "'comptime' cannot be used with 'fn'"). Name them
  `task_fn` / `entry`.
- A struct that holds a **thin function value** (`comptime TaskFn = def(Job, Int,
  Bool, Bool) thin -> None`) is assignable and callable through the field —
  including when the struct is reached through a raw pointer — but it must
  conform to `ImplicitlyCopyable` (as must the `Job` it carries), otherwise
  `state[] = Dispatch(...)` fails with "cannot be implicitly copied".
- **libgomp slices must use `omp_get_num_threads()`**, not the requested worker
  count: `GOMP_parallel` clamps the request to `OMP_NUM_THREADS`, which torch
  sets via `set_num_threads`, and a slice computed from the request leaves work
  unexecuted when the clamp bites. The region is synchronous, so the call
  state can live on the caller's stack.
- Python extension: `PythonModuleBuilder`, `def_function`, args are
  `PythonObject`, convert with `Int(py=...)` / `Float64(py=...)`. 23 int args
  cost 0.68 µs total, so marshalling is not worth micro-optimising.
  `std.runtime.initialize_runtime()` (MAX's runtime) is **gone**: nothing uses a
  Mojo thread pool now. A shared library must not define `main`.
- `mojo format src tests` exists and is expected to be clean; `mojo build
  --emit asm|llvm`, `--print-effective-target`, `--target-triple/--target-cpu`.
- Prefer `stack_allocation` or torch tensors over `List` for buffers a kernel
  reads through raw pointers; the tests do that.

## 6b. Mojo std-lib inventory (what the kernels use, what to reach for)

Both kernels are already written on stdlib primitives rather than hand-rolled
ones; this is the list, so nobody re-implements them:

| need | stdlib |
|---|---|
| vector width, ISA detection | `simd_width_of[DType.float32]()`, `CompilationTarget.is_x86()/has_neon()/_has_feature["avx512bf16"\|"bf16"]()` |
| buffers a pointer-based kernel reads | `stack_allocation[N, dtype]()` (`std.memory`) |
| typed pointer arithmetic | `MutPointer[T, MutUntrackedOrigin](unsafe_from_address=...)`, `.unsafe_offset`, `.unsafe_load[width=N](off)`, `.unsafe_store[width=N](off, v)` |
| lane index vector for masks | `iota[DType.int32, N](start)` (`std.math`) |
| per-lane compare + blend | `.lt/.le/.gt/.ge/.eq/.ne` then `mask.select(a, b)` (SIMD ops are *reducing*, so the mask form is required) |
| horizontal max/sum | `SIMD.reduce_max()`, `SIMD.reduce_add()` |
| pack (lo,hi) pairs | `SIMD.interleave` (this is how the bf16 broadcast pair operand is built) |
| transcendental | `exp` (`std.math`) — already lowers to a vector polynomial (`vrndscaleps`/`vscalefps` + FMAs) |
| ceil in tile math | `std.math.ceildiv` (`-(-a // b)` only remains in Python, where it is cheaper than importing) |
| thread dispatch | `external_call["GOMP_parallel"/"omp_get_*"]` (`std.ffi`) on torch's pool |
| ISA intrinsics | `llvm_intrinsic[name, RetType, has_side_effect=False]` (`std.sys`) — **always pass `has_side_effect=False`**, or the calls are ordered and the phase runs ~4x slower |

Verified while reviewing the ARM work: **`bfmmla` is reachable from Mojo**

```mojo
from std.sys import llvm_intrinsic
def bfmmla(a: SIMD[DType.float32, 4], x: SIMD[DType.bfloat16, 8],
           y: SIMD[DType.bfloat16, 8]) -> SIMD[DType.float32, 4]:
    return llvm_intrinsic["llvm.aarch64.neon.bfmmla", SIMD[DType.float32, 4],
                          has_side_effect=False](a, x, y)
```
`mojo build --emit asm --target-triple aarch64-unknown-linux-gnu --target-cpu
neoverse-v3` emits `bfmmla v0.4s, v1.8h, v2.8h`. Operands: acc = 2x2 fp32 tile
`(r0c0, r0c1, r1c0, r1c1)`, `x` = 2x4 bf16 row-major (rows = 2 queries, 4 k),
`y` = 4x2 bf16 row-major (k-major: `(k0c0, k0c1, k1c0, k1c1, ...)`). For Q@K^T that
means K needs a *position*-paired layout (`(k[d][t], k[d][t+1])` adjacent); for
P@V plain row-major V already has the dim pair adjacent.

What the stdlib does **not** have (do not go looking for it): a partial/tail
load or a gather (`iota` + `select` is the idiom here, §13), a non-NaN-propagating
float `max`/`reduce_max` (ours emits `vcmpordps` + `vmaxps` per step because our
values are finite, but there is no cheaper entry point), and any batched
matmul — that is what the per-ISA intrinsics are for.

## 7. Python-extension / host-side cost (measured on the Ryzen, b8/ctx256)

| item | µs |
|---|---|
| empty call, `num_seqs=0` (23 args + dispatch) | 0.68 |
| full Python wrapper (`mojo_attn._call`) | 2.8-3.6 |
| `torch.empty_like(query)` | 0.54 |
| 9 × `data_ptr()` | 0.51 |
| `_attention()` (extension callable, bound once) | 0.03 |
| `_workers()` (env override cached at import) | 0.16 |
| dtype lookup | 0.08 |
| **`GOMP_parallel`, 8 workers, warm** | **2.4** |
| **`GOMP_parallel`, 8 workers, 1 ms idle gap** | **2.7** |
| inline (1 worker, no region) | 0.36 |
| `max.algorithm.parallelize`, warm (removed) | 6.3 |
| same, 1 ms gap (removed) | 103-140 |
| attention call in-engine, ours / C++ backend (0.8 ms gap) | **29.8** / 90.7 |

Consequences: the dispatch is no longer the story, so `_workers()` inlines only
`num_tasks * pages <= 32` (24.5 vs 31.0 µs at b1/ctx64) — from b1/ctx128 up the
pool wins (1.3x at b1/ctx128, 8x at b8/ctx256). The cap at the **physical** core
count stays (8 workers on a 4-core Xeon: 1.5x slower at ctx 1024 and unstable at
batch 8), though torch now already sizes its pool that way, so it only bites when
a parent sets a larger `OMP_NUM_THREADS`. The old MAX-pool workarounds are gone
with the pool: no spin knob, no `initialize_runtime()`. Interleaved with GEMMs at
the engine's cadence the totals are now 88.8 µs per call (ours) vs 116.6 (C++),
i.e. attention 29.8-49.6 + interference +39.2 against 90.7 + 25.9.

## 8. Engine integration

- Backend name `mojo`, opt-in; `auto` still resolves to the vendored C++ kernel
  so all recorded baselines keep their meaning.
- **Public API mirrors flash-attn**: `flash_attn_with_kvcache(...)` and
  `flash_attn_func(...)` take flash's full argument list; anything the kernel
  cannot do (`qv`, `rotary_*`, `window_size`, `softcap`, `cache_leftpad`,
  `cu_seqlens_*`, `*_descale`, `return_softmax_lse`, `seqlen > 1`, dense
  multi-token) raises `NotImplementedError` rather than being ignored, and the
  tuning hints are no-ops.  The cache layout stays ours (head-major, K
  transposed in the page), not flash's page-major one, and `causal` has no
  effect with one query token.  The engine's per-layer path calls the thin
  aliases `self_attention`/`cross_attention` instead: the checked wrapper costs
  ~4% end to end at 24 calls/step (~2% with `check_inputs=False`), measured.
- `attention.uses_custom_ops(backend)` gates the frozen-compile path (the ops are
  `torch.library.custom_op` with `mutates_args={"k_cache","v_cache"}` so Dynamo
  never traces the kernel or the `Context` reads).
- `ModelRunner.compiled_backend` records which backend the graph was built with,
  so a later `attention.set_backend(...)` override falls back to eager.
- `attention.stage_cross(layers, request_ids, backend)` stages the paged encoder
  cache for whichever CPU kernel is active; the page layout is ours, so
  `stage_cross_kv` must stage through `mojo_attn.reshape_and_cache` (not the
  torch scatter) for the Mojo backend.

## 9. Performance: where the time goes

1. **x86 dot path** is worth 1.87× on the QK phase (33.8 vs 63.4 ns per 32-token
   page, isolated). P@V stays fp32 on x86 on purpose (bit-exactness).
2. **GROUP = 4** amortizes the softmax bookkeeping; per-page softmax cost was
   4× more max/exp/rescale work at short contexts. Measured neutral 2/4/8 on
   both servers afterwards.
3. **Per-call fixed cost dominated short contexts — solved by running on torch's
   pool.** With a private pool it was ~12.6 µs of a 75 µs call on the 4-core Xeon
   at b8/ctx256 and ~6 µs of a 33 µs call on the Ryzen, and every attempt to
   shrink it failed: MAX's `parallelize` 6.3 warm / 103-140 cold, a hand-rolled
   pthread pool 9.4 / 103, a 200 µs worker spin that cost 4.5% end to end.
   `GOMP_parallel` on the libgomp pool torch keeps busy between calls is 2.4 µs
   warm / 2.7 µs cold, and it removed the problem instead of shaving it: the
   in-engine call at b8/ctx256 went 180-196 → **29.8 µs** against the C++
   backend's 90.7-167, and 8x128 end to end 344 → **392-396 tok/s** (C++:
   364-377). The backend is now faster end to end on the laptop.
4. **Do not re-try**: a private thread pool in any form, including a spin
   (`GOMP_parallel` is the answer, §9.3) — and note `max.algorithm.parallelize`
   never had an OpenMP backend to select: it enqueues on AsyncRT's own
   `CPUDevice` work queue (`std::thread` + semaphore, `WorkQueueType` is only
   `kSingleThread`/`kThreadPool`), `libAsyncRTMojoBindings.so`/`libmax.so` contain
   no `GOMP_*`/`omp_*`/`__kmpc` symbols, and the only knob is
   `MODULAR_THREAD_BUSY_WAIT_US`; software prefetch of the next page's K/V (3 strides,
   neutral-to-worse — the pages are 4 KB sequential streams), `GROUP` 2/8,
   6/8 workers (`GROUP` beyond 4 registers-spills; >physical threads oversubscribes
   the fork-join), AMX.
5. **AMX is useless for this workload**: Q@K^T and P@V are M=1 per head for MHA
   (each head owns its own K *and* V page), and AMX tiles are 16 rows;
   `csrc`'s AMX path (`MaxQHeadNumPerIteration = 32`) only pays with GQA. On the
   Xeon the built `csrc` extension contains **0** AMX instructions and our
   wrapper cannot select `isa="amx"` anyway.
6. **Bandwidth is the wall on the servers**: Xeon 133 GB/s from L3 at
   b8/ctx256 (C++ runs 81 GB/s, compute-bound) and 62-70 GB/s DRAM at ctx 1024;
   Graviton 298 GB/s at 8.4 MB, 231 at 33.5 MB, 199 at 134 MB. Measured by
   pointing every sequence at one shared page set (1 MB unique vs 8.4 MB): the
   shared set is 1.6× faster at ctx 256 and 4× at ctx 1024.
7. **MAX's own CPU attention** (`max.kernels.nn.attention.cpu.mha`, measured with
   a script that went away with the `max` dependency) is the wrong yardstick: not paged, no block table, and its bf16 path is **5-9× slower than
   fp32** on AVX-512 (355 vs 39 µs at b8/ctx256) because AVX-512 has no native
   bf16 add/mul/fma — a bf16-native kernel scalarizes. Widening to fp32 and using
   bf16 only for the dot instruction is what makes this port fast.

8. **A throttled baseline invalidates an optimisation.** A worker rewrote the
   fp32 inner loop, measured 2.2-4.7x against a baseline taken while the box sat
   at ~2 GHz, and the result was **1.63x slower at head_dim 64** than the kernel
   it replaced (and 1.03-1.17x slower at 128). The fix is process-level
   interleaving: `tests/ab_two_trees.py --old <tree> --new <tree>` runs one short
   process per tree in alternating rounds and takes the median of the per-round
   medians, so drift cannot favour either side. Run it before believing any
   cross-run speedup.
9. **The laptop throttles to ~2.0 GHz under sustained AVX-512** (`scaling_cur_freq`
   reads 2004 MHz mid-sweep; a second session on the box pushes the load average
   past 2). Absolute times and even *ratios* move 30-50% between sessions — the
   fp32/D=128 causal ratio measured 0.63 to 1.09 for the same binary. Only
   interleaved A/B inside one process is trustworthy (`/tmp`-style script: run
   variant A, B, A, B... and take per-variant medians); `tests/bench_dense.py`
   blocks per variant and is therefore a coarse number only.
10. **Profiled properly, the dense kernel is *stall*-bound on the servers, and
   the lever is rows-per-K/V-sweep.** Two independent profiles
   (`PROFILE_X86.md`, `PROFILE_ARM.md`) with different tools agree, and both
   falsified the theories in this file:

   * x86 (`perf stat`, hardware counters): we retire **19.07 instructions per
     row-key against vLLM's 18.67 — a wash** — yet take **260 M cycles where they
     take 109 M, IPC 1.23 vs 2.87**. The separating counters are
     `l2_rqsts.demand_data_rd_miss/loads` **5.97% vs 0.07%**,
     `l1d.replacement/loads` 42% vs 21%, `exe_activity.bound_on_loads/cycles`
     **48% vs 15%**. At S=512 (128 KB of K^T, cache-resident) we are 1.28x behind
     with 0.42% misses; at S=2048, 2.25x with 5.97%. `llvm-mca` puts our Q@K^T
     loop's port floor at 1.0 FMA/cycle (6 zmm loads per 8 FMAs) and we run 0.42.
   * ARM: the `bfmmla` floor is 4.0 cycles/row-key (8 `bfmmla`/row-key at the
     measured 2.0/cycle ceiling); we measure 7.76 and vLLM 5.40, so **100% of the
     2.36-cycle gap is non-matmul cycles** — 18 instructions/row-key of which 8
     are `bfmmla`, 21-26% of the super-tile loop is `[sp]` spill, and 0.63 loads
     per `bfmmla`. The operand port does not bind; what is missing is **reuse
     across rows**: vLLM's K/V panel feeds 8-16 rows, ours 4.

   Consequences, measured: `CPU_ATTENTION_QT=8` is **1.20-1.34x** on x86 fp32
   (L2 miss 6.34 -> 2.62%) at 2048:8:64, 1024:16:128 and 4096:8:128; on ARM
   `qt=2` is 0.74-0.80x and **fp32 `qt=6` is a free 1.09-1.11x** at every long S.
   The shipped rule (`_kernel._qt_for`) is a per-(ISA, dtype, head_dim, seq_k)
   table: x86 fp32 `qt=8` for head_dim 64/128 and 4 at 32; x86 bf16 8 at 128 and
   4 at 64 (8 rows x 64 dims = 32 zmm of accumulator, the spills eat the gain);
   ARM fp32 6 at <=64 / 2 at 128; ARM bf16 4 (D=64) / 8 (D=128). The register
   kernel's threshold moved 384 -> 512 so 8 rows x 64 dims stays on it (6% faster
   than the L1-acc kernel where both are legal). **In-process A/B on the Ryzen
   measures qt 4 vs 8 as 1.00x at every shape**, i.e. this is a server-only win
   (4 physical cores); the laptop number moves 20-30% between runs on that box
   and cannot settle it.

   `qt=8` does *not* transplant to ARM because `qt > 4` flips the tile to
   `[BKS, 4, 8]` (half the key tile, double the row pairs) and blows the register
   file — that is a config coupling, not a row-count limit.

   **But the *layout* half of it was right, and the probe that "falsified" it was
   invalid.** vLLM's page is per-KV-head contiguous, so one head's K and V are a
   single stream; our dense call read V across `[seq, heads, dim]` (tokens 8 KB
   apart) and K^T across `[dim, seq_k]`. Repacking V per (batch, kv head) plus
   blocking the transposed K every 64 keys is **1.9-2.0x**: S=4096/H=16/D=128 fp32
   went 188 207 -> 93 231 us, **0.61x -> 1.21x of SDPA**, and the mechanism was
   isolated first (handing the same values head-major in pure Python, no code
   change, was 1.95x — the stride, not the copy). The earlier "strides do not
   matter" probe overrode only the strides the kernel is handed, i.e. it read
   wrong values out of a shrunken region; that cannot test a layout.

   **The bandwidth reading of the counters, though, is dead** (the x86
   worker retracted its own round-1 mechanism with a stride-override probe:
   starving K and V to an 88 KB, L1-resident working set buys **1-2%**, and the
   `l2_rqsts` counter turns out to be a vPMU artefact that reports the same
   absolute miss count for 88 KB and 1 MB). What binds is (a) the load:FMA ratio
   as a *floor* -- 6 zmm loads per 8 FMAs at `qt=4/kt=32` gives 1.0 FMA/cycle even
   with all loads in L1 -- and (b) the serial
   `reduce_max -> _vexp -> reduce_add -> rescale` chain draining the pipeline
   every 32-64 keys; vLLM reaches IPC 2.94 where we reach 1.41 *with fewer*
   instructions per row-key. `qt` is the lever for both, and the row effect is
   1.18-1.19x independent of which kernel runs (register or L1-acc).

   Why the old "blocking does not matter" note was wrong: it was measured on the
   **laptop**, 8 threads, 32 MB L3 and a private 1 MB L2 per core, where K/V
   traffic per row was not the binding term. On a 4-physical-core server slice
   with 2 MB L2 it is. Both regimes are real; the tuning must be per box, and
   `CPU_ATTENTION_QT` exists for exactly that.

   Also falsified, keep them dead: the L2-sized-tile theory (per-tile fixed cost
   is 0.47 of 7.76 cycles/row-key at D=64; a 1024-key tile buys **5.7%**, and
   ≤1.18x even if driven to zero), mask-by-range (~1%), the −8 rescale threshold
   (~4%), probabilities-aliasing (~3%), store-forwarding stalls (~1%,
   `ld_blocks.store_forward`), spills in the Q@K^T/P@V loops (none outside the
   23% softmax block), KV-split for long S (8192 tasks are already balanced),
   and "short S is our fixed cost" — charged fairly, **we are 1.32x faster than
   vLLM at S=128** because their `reshape_and_cache` costs more than our
   transpose, and the GOMP pool is a 2.5x win, not a cost.

10b. **The remainder is the softmax block, and that is now measured by ablation,
   not inferred.** Interleaved phase ablations of the ARM `bfmmla` kernel
   (`PROFILE`-style: patch out one phase, run through `ab_two_trees.py`, with an
   identity patch as control): Q@K^T **29%**, P@V **31%**, softmax +
   bookkeeping **40%** of the call. Q@K^T and P@V are *at parity with vLLM*
   (bf16 `bfmmla` both phases, 2560 instructions in the `.so`), so
   `1.45 = 1/(0.60 + s)` puts vLLM's softmax at **~9%**. Two independent workers
   on two boxes reached the same conclusion from different probes, and the
   MQA-vs-MHA probe (identical flops, 8x less K/V: **1.00-1.14x**) independently
   kills the traffic theory the x86 stride probe killed.

   Byte-level evidence for the difference: vLLM reduces max/sum once per **row
   per tile** (480-1024 keys) in `apply_softmax`; we do it per **row per 16-key
   step** (7 `fmax` + a horizontal reduce, twice). A per-64-key instead of
   per-16-key max/alpha is worth **1.18-1.22x** (`nomax` ablation) — but it can
   only be collected by moving the logits out of the register file, because a
   32-key softmax step alone measures **0.84x** (8 more live accumulators spill).
   That is why this is a rewrite of the row kernel rather than a knob.

   **And the cheap version of that rewrite does nothing**: deferring the row-sum
   horizontal reduce to one per row per call (a per-row vector accumulator folded
   by `n/LANES - 1` adds per tile) plus FA4's -8 threshold on the running max
   measured **1.00-1.01x** in a process-interleaved A/B (bf16, the untouched
   control, 0.99-1.00x). So the `reduce_add` is not the drain either, and the
   `nomax` ablation's 1.18-1.22x cannot be collected this way — the reduce that
   costs is `reduce_max`, and removing it needs an overflow-safe formulation
   (a stale max with a group-level fixup) that has not been tried. Fifth
   softmax-restructure attempt that lost or tied; do not redo the first four.

## 10c. Levers exhausted — read before proposing another

Everything in `csrc/cpu` that differs from us has now been ported, falsified, or
shown to be already done by the compiler. In order of what actually paid:

| lever | result | status |
|---|---|---|
| per-head-contiguous K/V (vLLM's page layout) | **2.0x** (S=4096/D=128 fp32 0.61x -> 1.24x SDPA) | ported |
| rows per K/V sweep (`qt`) | 1.14-1.19x x86, 1.42-1.73x ARM fp32 D=128 | ported (`_qt_for`) |
| worker count for the K/V pack | 1.34x at S<=128 | ported |
| ARM bf16 softmax state in vector lanes | 1.01-1.06x | ported |
| fast exp (deg-5 poly + bit insert) | +18% x86 bf16 D=128 | ported (`_vexp`, `_exp2_fast`) |
| register-tiled GEMM micro-kernel | 1.63x **slower** | falsified |
| L2-sized logits tile / memory logits | 0.74-0.81x, twice, independently | falsified |
| 32-key softmax step | 0.84x | falsified |
| deferred row-sum + FA4 -8 threshold | 1.00x | falsified |
| one-row-pair / L1-staged logits / pipelining | -11% / 0% / -11% | falsified |
| KV splits, GQA packing, mask-by-range, probs aliasing | <=4%, not needed for MHA | rejected |
| lane-indexed FMA (vLLM's `vfmaq_laneq_f32`) | **already emitted**: 100 of 360 `fmla` in the ARM fp32 kernel are `fmla v.4s, v.4s, vN.s[k]` | nothing to do |
| AMX (16x16 bf16 tiles) | **2.8-9.8x on bf16** (`cpu_attention/_csrc/dense_amx.c`); 4-9.6x from 512 keys up | done for bf16; see below on Mojo vs C |

So: on x86 and ARM the kernel is **at or ahead of SDPA everywhere we measured**
(22/22 on both boxes; o/sdpa 0.28-0.87 on ARM, 0.21-0.79 on x86) and roughly
**at parity with vLLM** (ARM 0.79-1.02, x86 bf16 0.80-1.02, x86 fp32 0.51-1.13)
— the "1.2-2.3x behind vLLM" and the "Graviton 17/22, Xeon 4/22" counts belong
to the pre-AMX/pre-`bfmmla` tree and were superseded by the fourth campaign
(`docs/AUDIT.md` §1-§5). The remainder is fused-loop IPC (1.4 vs 2.9), and the
two from-scratch vLLM-shaped prototypes both lost 20-26%; "beat vLLM" now means
winning the last few percent on the D=128 long rows, not a structural gap.


11. **Dense kernel (laptop, 8 threads, causal, `tests/bench_dense.py`)**: bf16 is
   1.26-2.5× torch SDPA, fp32 1.0-2.0× — except `H=16/D=128` at S=1024 (0.83×,
   the register wall above) and S=128 (0.93×, where the call is 80 µs and SDPA
   is 78). Absolute: bf16 936 GFLOP/s (H=8/D=64) and 645 (H=16/D=128); fp32 700
   and 444. vLLM's `VEC` prefill on the same box is 915 GFLOP/s fp32 (its own
   cache is already staged, so it pays no transpose), SDPA 570 — i.e. bf16 here
   matches vLLM's fp32. Phase probes say the split is roughly Q@K^T 40%, P@V
   40%, softmax/mask 20% at `qt = 4`; a `qt`-block big enough to shrink the K/V
   re-reads further (vLLM keeps the partial output in memory and runs 8-32
   rows) is the next structural step, not a knob.

## 10. Final three-box numbers (ratio `mojo/vllm`, <1 = faster)

The Ryzen row is the original `benchmarks/attn_backend.py --mode kernel` +
`benchmarks/kernel_ab.py` run (two runs, old private pool). The two server rows
were re-measured on 2026-10-23 with the independent harness (`docs/AUDIT.md`
§3, §5): 15 rounds, bf16, **min (median)**, page 32 on x86 / 16 on ARM, each
backend staging its own cache layout:

| box (threads/workers) | self b8/256 | cross b8/256 | self b8/1024 | cross b8/1024 | self b32/1024 |
|---|---:|---:|---:|---:|---:|
| Ryzen 7840H (8/8) | 0.76-0.88 | 1.02-1.18 | 0.84-0.92 | 0.92-0.96 | 0.86 |
| Xeon 6975P-C (4/4) | 0.53 (0.79) | 0.59 (0.87) | 0.92 (0.90) | 0.97 (0.94) | 0.66 (0.76) |
| Graviton 5 (8/8) | 0.35 (0.38) | 0.44 (0.47) | 0.69 (0.74) | 0.76 (0.80) | 0.89 (0.92) |

The Xeon `b8/256` row has a 1.7x min-to-median spread; treat it as "1.3-1.9x
faster", not a two-decimal band. The table is bf16; **fp32 paged loses on x86**
at b8/256 (1.04-1.13), b8/1024 (1.15-1.20) and b1/4096 (1.08-1.11), and wins on
ARM (0.37-0.84). At b1 nothing is guaranteed: x86 is 0.77-0.82 (self) and
0.85-1.20 (cross), ARM 0.59-1.12 (self) and 0.73-1.51 (cross).

Absolute µs at `cross b8/enc256` bf16: Xeon 82.5/85.5 with the old pool →
**67.3 min / 106 median** now; Graviton 33.6/33.3 → **34.2 min (block 16) /
32.6 (block 32)**. The laptop is *faster in absolute terms* than the Xeon
because it has 8 full-speed cores and 32 MB L3 vs a 4-core cloud slice; the
repo already records that ordering for decode GEMMs. What matters is the ratio
on the same box.

Config findings worth remembering: on the previous Xeon box `block_size=128`
measured ~9% faster at ctx 1024 (a `Config`/CLI knob, not a kernel change); on
Graviton the aarch64 default of 16 is the C++ kernel's best point (67 vs 76 µs
at block 32), while for our kernel block 32 is mixed — better at b32/1024
(0.79 vs 0.94) and worse at b8/1024 (0.82 vs 0.76).

## 11. Failure modes we hit (each has a regression test now)

1. **Token strides.** `q/k/v` come from `torch.chunk(3, dim=-1)` + `reshape`, so
   `stride(0) == 3 * d_model` and the tensor is *not* contiguous. The first
   in-engine run gave every sequence sequence 0's output. Passing strides
   explicitly fixed it; `../fast-encdec/tests/test_cpu_attention.py` covers it via the engine.
2. **Page-vs-tile walk.** The group loop advanced by whole pages while scoring
   only `tile` positions of each page: correct only when `block_size == tile`.
   For `block_size ∈ {48,64,128,256}` (accepted by `supported()`) it scored the
   first 32 positions of every page and skipped the rest — silent garbage
   (2046/2048 elements wrong at block 64). The engine test missed it because the
   tiny random model decodes the same token everywhere and the logits check only
   ran at `lens == 1`. `tests/test_attention.mojo` now covers block 48/64 with
   `lens` 190-200 (multi-tile pages, groups straddling pages).
3. **The K-page/V-page layout must match the selected path.** Store and load both
   read the same `comptime` predicate (`BF16_DOT`, `NEON_DOT`, `dtype == bf16`,
   `tile % LANES == 0`). A mismatch shows up as bit-exact-but-wrong results for
   one dtype or block size only.
4. **bf16 probabilities on ARM** (only place parity with `csrc` is given up):
   ≤1e-3 vs the fp32 reference, matching `csrc`'s ARM `prob_buffer_t`. x86
   matches `csrc` to 0-2 ulp (exact at b1/ctx256, one ulp at b8/ctx1024 from a
   different summation order; re-measured 2026-10-23, `docs/AUDIT.md` §6).
5. **Engine tests can be blind.** `../fast-encdec/tests/test_cpu_attention.py`'s token comparison is
   degenerate on the tiny model; the standalone test against a scalar fp32
   reference is the real gate. Keep both.

6. **Dense: the causal limit is inclusive.** `j <= i + (seq_k - seq_q)`, so the
   `select` needs `t < limit + 1`. Shipped with `t < limit`, which dropped the
   diagonal key: correct for non-causal (fp32 matched SDPA to 3e-7) and wrong
   only under `causal`, i.e. exactly the case the engine would use first.
7. **Dense: P@V must not read past `seq_k`.** The tile is a fixed 64 keys, so the
   last tile of a short `seq_k` has lanes whose probability is exactly 0 — but
   reading their V walks past the tensor and `0 * NaN = NaN`. It looked like a
   dtype bug (bf16 "failed", fp32 "passed") because it depended on what the
   allocator left there: the same case gave 4e-7 or NaN on consecutive runs.
   Loop bound is now `min(KT_TILE, seq_k - t0)`.
8. **Dense: query rows past `seq_q` must not be loaded.** A `qt`-row block reads
   all `qt` rows unconditionally, so the last block of `seq_q = 1` read three
   rows of garbage (and NaN'd through `exp`). Guarded with `if r < nrows`; the
   masked-out rows are computed but never stored.
9. **Dense: `torch.permute(...).contiguous()` for the K transpose costs more
   than the kernel sometimes.** 11-15 GB/s (element-wise, because the innermost
   dim after the permute has stride 1) — 1.5 ms for the 16 MB K of
   `S=1024/H=16/D=128`, which was the whole reason fp32 lost that shape. The
   blocked Mojo pass (16 tokens per 64-byte store) does it at 40-70 GB/s.

10. **AMX: `_drop_workspace()` does not clear the AMX call plans.** The plan
    buffer holds the packed K/V addresses; `clear_scratch()` pops
    `_AMX_PLANS`, but `_drop_workspace()` (called by `_dense_plan` when the
    non-AMX workspace exceeds 24 MB) frees the buffers without popping the
    plans. Toggling `CPU_ATTENTION_AMX` off and on in one process then
    SIGSEGVs at `_kernel.py:844` (reproduced by `audit/audit_amx_ab.py
    --which onoff`, 7 shapes in). The shipped path never runs the non-AMX
    dense kernel once AMX is on, so only the A/B toggler is exposed; run
    `onoff` one shape per process until the pop is added to
    `_drop_workspace()`. The `v1v2` and `mode & 512` A/Bs are clean.

11b. **`exp` on a SIMD is not a polynomial in this stdlib.** `std.math.exp`
   costs ~22 instructions per vector here (`~36 cycles/lane` in the x86 worker's
   measurement, which made it 58% of *their* kernel). `_vexp` in `dense.mojo`
   (Cephes degree-8 + magic-constant rounding + exponent insert, ~10 FMAs) is
   neutral on AVX-512 fp32 but worth **+18% on x86 bf16 head_dim 128** (647 ->
   782 GFLOP/s, i.e. vLLM parity) and 20% on NEON. Clamping at -87 keeps masked
   lanes at ~0, and `fma()` from `std.math` is what guarantees contraction -- the
   multiply/add pairs in this kernel did *not* fuse on their own.

## 12. Box access and environment

| box | IP | notes |
|---|---|---|
| Ryzen 7840H (dev) | local | 8 physical/16 SMT, avx512_bf16, 32 MB L3; repo `.venv` or `~/.venvs/torch-cpu` |
| Xeon 6975P-C | last campaign: `ubuntu@13.210.181.255` | 8 vCPU = **4 physical cores**, `avx512_bf16` + `amx_bf16`; `~/venv`; torch default 4 threads |
| Graviton 5 | last campaign: `ubuntu@13.236.70.191` | 8 cores, ASIMD + `bf16`; `~/venv`; torch default 8 threads |
| `m8i.2xlarge` (audit, 2026-10-23) | terminated | same 6975P-C, 4C/8T, on-demand, `ap-southeast-1a`; the x86 numbers in `docs/AUDIT.md` |
| `m9g.2xlarge` (audit, 2026-10-23) | terminated | Neoverse-V3, 8 cores, spot, `ap-southeast-1a`; the ARM numbers in `docs/AUDIT.md` |

The first three are the campaign's long-lived boxes; the two audit boxes were
built from the stock Ubuntu 26.04 AMI and terminated. `scripts/setup_box.sh`
provisions a fresh one and `scripts/box_sync.sh <arch> <host> <dir> --full`
syncs + builds; expect ~6-10 min from launch to parity. The audit's exact
commands are in `docs/AUDIT.md` §11 and `audit/README.md`.

- Both checkouts live on the boxes: `~/fast-encdec` and `~/cpu-attention`,
  with the library installed editable into the engine env
  (`uv pip install -e ~/cpu-attention --no-deps`) and `_C.so` rebuilt there
  (`uv run --extra build python scripts/build.py`), so a sync of the Python
  side alone is enough unless the
  kernel changed. **Never `rm -rf ~/fast-encdec` on the servers** — their
  `~/venv` is a symlink into `~/fast-encdec/.venv`.
- Both servers need `uv` and `gcc/g++/ninja` (the latter for fast-encdec's
  `csrc` JIT); rebuild the kernel after a sync with
  `uv run --extra build python scripts/build.py`. `scripts/build.py --tests`
  links torch's bundled `libgomp.so.1`, so no system libgomp package is needed.
- `benchmarks/kernel_ab.py --threads N --workers N` pins the knobs; `0` means
  "torch default / the kernel's physical-core rule". These boxes drift ±10% and
  Graviton's 1024-token rows ±25%, so always run it twice and report the band.

## 13. Ideas not tried (in rough priority order)

1. **bf16 P@V on x86** (probabilities quantized like MAX/`csrc`-AMX): halves the
   P@V multiply count, ~30% at short contexts; costs bit-parity with `csrc/`.
   Only worth it if the engine, not the kernel table, is the target.
2. **Chunked prefill on the paged path**: still the paged kernel's one gap
   (the dense kernel already does multi-token queries). It needs the paged
   mask generalised to `j <= q0 + r + (len - q)` and the group loop to start at
   the row's window instead of position 0 — the dense kernel's boundary-tile
   mask (§4a) is the worked-out version of exactly that logic.
3. **Dense: ~~a GEMM micro-kernel for both phases~~ — falsified, do not retry.**
   `_dense_rows_gemm` was built and measured (§9.10): 1.63x *slower* at fp32
   head_dim 64, 1.03-1.17x slower at 128. The premise ("1 load per 8 FMAs beats
   1 per 2") does not hold because these cores are not load-bound (see §9.10).
   What the same worker did find is worth keeping: **multiply/add pairs did not
   contract on their own** in the fp32 loops (`vmulps` + `vaddps`), so `fma()`
   from `std.math` should be explicit wherever a loop is written fresh.

4. **ARM: the softmax, not the tile.** Round 2 built four variants and reverted
   all of them (numbers in `4c7ca5d`), which fixes the remaining 1.45x
   precisely. Measured on the box: `bfmmla` issues at **2.0/cycle** (0.152 ns,
   1640 GFLOP/s ceiling) and we run at 1.02 (D=64) / 1.30 (D=128) per cycle
   while vLLM runs 1.49/1.86. A two-head-dim marginal fit gives **0.5 cycle per
   `bfmmla`** — the ceiling — so the matmul has no headroom, and an intercept of
   **~3.0 cycles/row-key of softmax that does not overlap it** (37% of the D=64
   runtime). The softmax block is ~770 instructions per 16-key tile (12 per
   row-key) against ~300 of real work; the rest is `ldr/str q,[sp]` spill and
   remat. Do NOT retry: one row pair at a time (-11%/-9%, the duplicated K loads
   cost more than the register relief even when they are L1 hits), logits staged
   through L1 (-0%, LLVM SROA forwards the stores so the allocator still keeps
   them in registers), software-pipelining the softmax into the matmul (-11%/-8%
   plus a parity bug). qt=8 at D=64 loses too. The only untried route is a
   Vec16-style *single-loop* softmax over a memory-resident logits buffer
   (vLLM's actual structure, 200-300 lines, softmax only — Q@K^T, P@V and all
   layouts stay). Expected ~1.2-1.3x at the time. The merged tree now measures
   0.99-1.01 on those rows (`docs/AUDIT.md` §2), so there is no 1.45x left to
   collect; this is no longer the priority.
5. **Dense: `bfmmla` on ARM for both phases** — **done** (§4b): at the time
   2.2-3.2x, 331 -> 853 GFLOP/s (D=64), 348 -> 1093 (D=128). The merged tree
   measures **1.34-1.59 TFLOP/s** at the long rows and **0.79-1.02 vs vLLM**
   (`docs/AUDIT.md` §2), so this item is closed; what is left is item 4 above
   on the bf16 D=128 long rows only.

6. **GQA/MQA**: one KV page shared by `q_heads_per_kv` heads would let one task
   amortise the K/V loads over several query heads — and only then does AMX
   (16-row tiles) become interesting. Not needed for BART/Florence-2.
7. ~~Own thread pool~~ **Done, differently**: not a pool of our own (no win,
   §9.3) but `GOMP_parallel` on torch's pool. A hand-rolled pool cannot beat it
   because the win is *reusing* the pool that is already awake.
8. **`partial_simd_load`** for tails instead of `iota`+`select` (cosmetic).
9. **T5** (root `AGENTS.md`): same paged path, but its per-layer pointwise chains
   are what the frozen compile graphs fuse, so the CPU gains would be larger.

## 14. AMX from Mojo: the "C shim is the only route" claim was wrong

LLVM's AMX *intrinsics* (`llvm.x86.tileloadd64`, `tdpbf16ps`, ...) really are
unreachable from Mojo, because they take and return the `x86_amx` type and Mojo
cannot name it. But that type exists only so a compiler can track tile
registers, and a tile register can equally be spelled *inside the instruction
text*. `std.sys.inlined_assembly` does that, and MAX itself reaches Apple's AMX
this way (`max/kernels/src/linalg/arch/cpu/apple_amx_intrinsics.mojo` emits raw
`.word` encodings). Verified in this repo:

```bash
mojo build -I cpu_attention/_mojo/src --target-triple x86_64-unknown-linux-gnu \
    --target-cpu sapphirerapids --emit asm \
    cpu_attention/_mojo/tests/amx_smoke.mojo -o /tmp/amx.s
grep tdpbf16ps /tmp/amx.s     # tdpbf16ps %tmm2, %tmm0, %tmm1
```

`cpu_attention/_mojo/src/amx.mojo` has the primitives (`ldtilecfg`,
`tileloadd`, `tdpbf16ps`, `tilestored`, `tilezero`, `tilerelease`) with the tile
number as a *compile-time* parameter interpolated into the asm string, so one
call site serves every tile. `tests/amx_smoke.mojo` is a 16x16x32 bf16 GEMM
against a scalar reference; it needs an AMX CPU to run.

What the C shim still buys, and why it is what ships today: LLVM does not model
tile registers, so a `tileloadd` can be hoisted above a `tdpbf16ps` that still
reads the tile (a WAR hazard with silently wrong results), which forces a
`~{memory}` clobber on every primitive and gives up some software pipelining
that gcc's intrinsics keep. So the Mojo port must be A/B'd, not assumed. It is
still the right destination: one build artifact instead of two, no ctypes
marshalling, no GIL-released mutex, and the AMX path can be gated at compile
time with `CompilationTarget._has_feature["amxbf16"]()` exactly like the AVX-512
and NEON paths already are.
