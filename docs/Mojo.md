# The Mojo backend

`cpu_attention/_mojo/` is the original kernel: a Mojo port of the hot path of
vLLM's `csrc/cpu` attention, plus a dense kernel built from vLLM's structure
(vLLM has no cache-free kernel to port). It is the default backend and the
oracle the C++ backend (`docs/CPP.md`) is checked against.

```bash
uv sync --extra build                                   # torch + the mojo wheel
uv run --extra build python scripts/build.py --tests    # -> cpu_attention/_C.so + kernel tests
```

`mojo` comes from the PyPI wheel; there is no separate toolchain. `mojo build`
targets the **host CPU**, so rebuild per machine. `mojo run` cannot run the
kernel: its JIT cannot resolve `GOMP_parallel`, which is why the tests are a
built binary.

## 1. Status

Against vLLM, from the final three-box matrices (2026-10-28, same harness as
the README), as `mojo / vllm` (below 1 is faster):

| box | dense bf16 | dense fp32 | paged bf16 | paged fp32 |
|---|---|---|---|---|
| Ryzen 7840H (vLLM VEC) | 0.70-1.06, 10/11 faster | 0.99-1.60, 2/11 | 0.69-1.27, 9/10 | 0.93-1.14, 7/10 |
| Axion Neoverse-V2 (vLLM NEON) | 0.88-1.00, 11/11 | 0.79-1.00, 10/11 | 0.68-1.53, 7/10 | 0.76-1.11, 8/10 |
| Xeon 8581C (vLLM AMX bf16 / VEC fp32) | 0.80-1.01, 10/11 | 0.54-0.85, 11/11 | 0.83-1.39, 6/10 | 0.75-1.06, 5/10 |

- **Against C++:** C++ beats this backend on 122 of 126 rows.
- **Where Mojo is still the faster baseline:** Axion dense, where C++ is at
  parity on 7 rows, and the Ryzen paged b=32 ctx=1024 bf16 rows.
- **Against SDPA:** every dense row is faster.
- **Numerics:**
  - x86 paged bf16 matches vLLM `csrc` to 0-2 bf16 ulp; fp32 agrees to
    1.5-2.6e-7.
  - ARM rounds probabilities to bf16 like `csrc`, so it is within 1e-3.
  - Dense fp32 is <= 4e-6 against the reference on ARM, and 3-8e-5 on AMX
    (hi/lo).

## 2. Layout

```
_mojo/src/attention.mojo    paged kernel: Job, store, generic + dot-product attention, dispatch
_mojo/src/dense.mojo        dense kernels: row kernels, K transpose/pack, bfmmla kernel
_mojo/src/dense_small.mojo  short-sequence dense kernel (seq_k <= 256)
_mojo/src/flash_amx.mojo    AMX policy of the unified flash kernel (default on AMX CPUs)
_mojo/src/amx.mojo          AMX primitives as inline assembly
_mojo/src/_C.mojo           Python extension (PyInit__C) and the call entries
_mojo/tests/test_attention.mojo   standalone parity test vs a scalar fp32 reference
_mojo/tests/amx_smoke.mojo        16x16x32 AMX GEMM vs a scalar reference
_mojo/tests/bfmmla_layout.mojo    bfmmla operand-layout check
```

## 3. Paged kernel (`attention.mojo`)

**Contract.** Exactly one query token per sequence, attending every cached
position, so the causal mask is the cache tail.

**Work item = one (sequence, head).** It owns that head's cache slice end to
end: no KV split, no scratch, no cross-thread reduction.
- **Dispatch:** `GOMP_parallel` on torch's own OpenMP pool.
- **Workers:** `min(torch threads, physical cores, tasks)`. A single inline
  worker is used when `tasks * pages <= 32`.

**Tiles.** KV is scored in tiles of 32 (or 16) positions, with `GROUP = 4`
tiles per online-softmax step. The group loop resolves `(page, offset)` per
tile and shares it between Q@K^T and P@V. Walking pages instead of tiles was a
bug: it silently scored the first tile of each multi-tile page (§8).

**Page layouts** (one page = `block_size * head_dim`):

| | layout | why |
|---|---|---|
| K, dot paths | `[hd/2][block][2]` | one token per column |
| K, fp32 fallback | `[hd][block]` | |
| V, x86 | `[block][hd]` | |
| V, ARM | `[block/2][hd][2]` | `bfdot` contracts two adjacent keys |

**Online softmax.** `m' = max(m, max(group))`, `l = l*exp(m-m') +
sum(exp(group-m'))`, `acc = acc*exp(m-m') + P@V`. The probabilities stay in
registers.

**Arithmetic.**
- On the dot path, Q stays bf16 and the scale multiplies the logits; never
  round Q through a scaled multiply.
- x86 bf16 does Q@K^T with `vdpbf16ps` and keeps P@V in fp32. That keeps
  bit-parity with `csrc` and is 1.87x on the QK phase.
- ARM bf16 uses `bfdot` for both phases.

**Strides.** q/k/v are chunks of the fused `qkv_proj` output (token stride
`3*d_model`), so `q_stride` / `kv_stride` / `dst_stride` are explicit.
Assuming contiguity gave every sequence sequence 0's output (§8).

## 4. Dense kernels

**Generic row kernels (`dense.mojo`).**
- **Work item.** `qt` consecutive query rows of one (batch, head), swept over
  the whole key range, so the accumulator stays in registers. `qt` comes from
  `_kernel._qt_for` per (ISA, dtype, head_dim, seq_k): x86 fp32 8, ARM 8, x86
  bf16 4-8. `CPU_ATTENTION_QT` overrides it.
- **Three row kernels:**
  - register accumulator, while `qt*hd <= 512`;
  - L1 accumulator with the head dim tiled, for head_dim 128 without spills;
  - the bf16 dot kernel.
- **K/V packing.** Per (batch, kv head) contiguous K/V (vLLM's page layout)
  was worth 2.0x. K^T is blocked every 64 keys; one blocked transpose pass runs
  at 40-70 GB/s against 11-15 GB/s for `torch.permute().contiguous()`.
- **Masking and bounds.**
  - The causal limit is bottom-right aligned and *inclusive*
    (`j <= i + seq_k - seq_q`). Only the boundary tile builds a lane mask.
  - P@V walks only keys that exist (`0 * garbage` is not 0).
  - Query rows past `seq_q` are never loaded.
- **The (batch, head) work item.** A thread packs that head and then attends all
  of its query blocks, with no phase barrier. `perf` had put 29% of a short call
  in the OpenMP barrier.

**ARM: `bfmmla` for both phases.**
- **Tiles.** Q@K^T is 2 rows x 4 dims against 2 keys x 4 dims; P@V is 2 rows x
  4 keys against 2 dims x 4 keys. A task lives in 2x2 fp32 tiles, and the logits
  *are* the Q@K^T accumulators.
- **Layouts.** K is `[hd/4][seq/2][8]` and V `[hd/2][seq/4][8]`, packed in one
  pass.
- **Exponential.** `_exp2_fast` uses magic-constant rounding, a polynomial and
  an exponent insert: degree 4 for fp32, degree 2 for bf16, ~10 instructions
  against `std.math.exp`'s ~22. Clamping at -127 makes masked lanes exactly 0.
- **Result.** 2.2-3.2x over the previous kernel.

**Short sequences (`dense_small.mojo`, `seq_k <= 256`).** Rounds key coverage
to 8 keys instead of 16 and packs in the same region.

**AMX (`flash_amx.mojo`, a `FlashPolicy`; default on AMX CPUs).**
- **Tiles.** A register tile is 32 query rows (two 16-row groups) x 32 keys: four
  fp32 accumulators, `tdpbf16ps` for both phases, fp32 as bf16 hi/lo (3 products).
  Keys are walked in super-tiles of 256 (bf16) / 512 (fp32): Q@K^T into an fp32
  logits buffer, one softmax pass writing bf16 P, then P@V into the driver's
  accumulator. `rq = 64` rows per work item (`_kernel._flash_plan`).
- **Selection.** `isa = 1` in the plan when the CPU reports AMX, `arch_prctl` grants
  tile data and `CPU_ATTENTION_AMX` / `_AMX_FP32` / `CPU_ATTENTION_ISA` allow it;
  the toggles are read per call and are part of the plan key.
- **What made it faster than the old standalone kernel (Xeon, 4 threads):**
  - `vscalefps` exponential with finite masks (-1e30): 4-9% (the old ledger entry
    "cheaper exponential" lost because it kept the clamp and `-inf`);
  - `max()` on SIMD floats compiles to compare + masked max + move (3 uops per
    vector, also in the exponent clamp): `a.gt(b).select(a, b)` is one `vmaxps`;
  - row max / row sum / running-max state reduced 16 rows at a time by a shuffle
    network instead of per-row horizontal reductions and a branch-free scalar
    update (which evaluated `exp2` for every row);
  - one `vcvtne2ps2bf16` per 32 probabilities, logit/P strides padded by 64 B,
    `ldtilecfg` once per query sweep, no accumulator clear (`ZEROES_ACC`).
- **Machine facts (Xeon 8581C, core ~3.08 GHz):**
  - a 32x32 Q@K^T step is 8 `tdpbf16ps` = 128 cycles of TMUL; with the four
    `tilestored` into a 35 KB logits buffer it takes ~340 cycles, ~210 when the
    logits fit L1 (KT <= 128): the stores are L1<->L2 bound, not TMUL bound;
  - shrinking the logits to L1 in the kernel (64-key chunks, deferred P@V) did not
    pay: the extra per-chunk statistics and the P@V flush cost more than the QK
    saving (0.87-1.13x);
  - `CPU_ATTENTION_AMX=0` falls back to AVX-512.

**AMX from Mojo.** LLVM's AMX intrinsics take the `x86_amx` type, which Mojo
cannot name. `amx.mojo` therefore spells the tile number inside inline assembly
(`ldtilecfg`, `tileloadd`, `tdpbf16ps`, `tilestored`, `tilezero`), with the tile
as a compile-time parameter.
- LLVM does not model tile registers, so every primitive carries a `~{memory}`
  clobber against a `tileloadd` being hoisted above a `tdpbf16ps` that still
  reads it.
- In AT&T operand order the destination tile is last.

## 5. One source, several ISAs

```mojo
comptime X86_DOT  = CompilationTarget.is_x86() and CompilationTarget._has_feature["avx512bf16"]()
comptime NEON_DOT = CompilationTarget.has_neon() and CompilationTarget._has_feature["bf16"]()
comptime LANES = simd_width_of[DType.float32]()     # 16 AVX-512, 4 NEON
```

Intrinsics go through `llvm_intrinsic[..., has_side_effect=False]`:
`llvm.x86.avx512bf16.dpbf16ps.512`, `llvm.aarch64.neon.bfdot` and
`llvm.aarch64.neon.bfmmla`. **Without `has_side_effect=False` the calls are
ordered and the phase is ~4x slower.**

Tile sizes must be lane-relative (`KT_TILE = 4 * LANES`), not x86-relative.
Verify both targets without the hardware:

```bash
mojo build --emit llvm cpu_attention/_mojo/src/_C.mojo -o /tmp/x86.ll && grep -c dpbf16ps /tmp/x86.ll
mojo build --emit llvm --target-triple aarch64-unknown-linux-gnu --target-cpu neoverse-v3 \
    cpu_attention/_mojo/src/_C.mojo -o /tmp/arm.ll && grep -c neon.bfdot /tmp/arm.ll
```

`--target-cpu` is not optional. Without it the build has no `+bf16`, so it
silently compiles the portable path.

## 6. Call path and threading

- **Run on torch's pool.** `GOMP_parallel` on torch's libgomp pool costs 2.4 us
  warm and 2.7 us after a 1 ms gap. MAX's `parallelize` was 6.3 / 103-140 us
  and a hand-rolled pthread pool 9.4 / 103 us. In the engine that took an
  attention call from 180-196 us to 29.8 us. Do not build a private pool.
- **Use the team size.** libgomp slices must use `omp_get_num_threads()`, not
  the requested count: `GOMP_parallel` clamps to `OMP_NUM_THREADS`.
- **Linking libgomp.**
  - On x86 the extension leaves the libgomp symbols undefined; torch loads
    `libgomp.so.1` first.
  - On aarch64 torch's copy is `RTLD_LOCAL`, so `scripts/build.py` links it into
    the extension.
- **Plans.** The paged path keeps a per-thread plan in Python that patches only
  the data pointers, and the dense path uses plan-buffer entries. The paged call
  path is ~3.4 us, of which the extension marshal is 1.2 us.

## 7. Mojo language notes (the ones that cost time)

- **Syntax.**
  - `def`, not `fn` (which is also reserved as a name: use `task_fn`).
  - `@__parameter`, and no `@__parameter` on nested closures: use capture lists
    (`{imm}`, `{mut x}`).
- **No mutable globals.** Module-level `comptime` constants are fine.
- **Compile time.**
  - `comptime` ternaries do not fold to comptime integers;
    `simd_width_of[...]()` does.
  - `comptime if` arms are type-checked even when dead, hence the separate
    per-tile-width functions.
- **SIMD.**
  - A runtime index into a SIMD spills it to the stack; use `comptime for`.
  - SIMD comparisons (`a < b`) reduce to `Bool`. Use `.lt/.le/.gt/.eq` and
    `mask.select(a, b)` for lanes.
- **Arithmetic.**
  - `std.math.exp` on SIMD is ~22 instructions; the hand-rolled `_vexp` /
    `_exp2_fast` are 18-20% on bf16.
  - Multiply/add pairs do not contract on their own; use `fma()` explicitly.
- **FFI.** `external_call["GOMP_parallel", ...]`, and `@export def f() abi("C")`.
  A struct holding a thin function value must be `ImplicitlyCopyable`.
- **Buffers.** Prefer `stack_allocation` or torch tensors over `List` for
  anything read through raw pointers.

## 8. Failure modes we hit (each has a regression test)

1. **Token strides.** The fused-projection chunks are not contiguous.
   Explicit strides fixed it; the engine test covers it.
2. **Page-vs-tile walk.** At block 64 it was silently 2046/2048 wrong.
   `test_attention.mojo` covers blocks 48/64 with multi-tile pages.
3. **K/V page layout per path.** Store and load must read the same `comptime`
   predicate, or one dtype or block size is bit-exact but wrong.
4. **The inclusive causal limit.** `t < limit` dropped the diagonal, and only
   under `causal`.
5. **P@V past `seq_k`.** It read past the tensor: NaN on some runs and 4e-7 on
   others.
6. **Loading rows past `seq_q`.** The garbage NaN'd through `exp`.
7. **Blind engine tests.** The engine's token comparison is degenerate on the
   tiny model, so the standalone scalar-reference test is the real gate.
8. **Freed workspace under AMX.** `_drop_workspace()` must also drop the AMX
   plans that hold the workspace's addresses (fixed).

## 9. Levers exhausted (do not redo)

| lever | result |
|---|---|
| per-head-contiguous K/V | **2.0x**, ported |
| rows per K/V sweep (`qt`) | 1.14-1.19x x86, 1.42-1.73x ARM fp32 D=128, ported |
| `bfmmla` both phases on ARM | 2.2-3.2x, ported |
| AMX via inline assembly | 2.8-9.8x bf16 on AMX boxes, ported |
| fast exp | +18% x86 bf16 D=128, ported |
| register-tiled GEMM micro-kernel | 1.63x slower |
| L2-sized logits tile / memory-resident logits | 0.74-0.81x, twice |
| 32-key softmax step | 0.84x (spills) |
| deferred row sum + FA4 -8 threshold | 1.00x |
| one row pair at a time / L1-staged logits / softmax pipelining | -11% / 0% / -11% |
| AMX: pipelining the softmax under the dots | 0.73-0.91x |
| AMX: cheaper exponential (`ldexp`, degree 3, clamp kept) | +8% single-thread, -10..-27% on four threads (the unified kernel's clamp-free `vscalefps` form with finite masks is +4-9%, kept) |
| AMX: software prefetch of the K/V source rows in the pack | 0.92-1.02x |
| AMX: 64-key chunks with L1-resident logits, deferred P@V | 0.87-1.13x |
| AMX: rows per work item 32 / 96 / 128 vs 64 | within noise |
| KV splits, mask-by-range, probability aliasing | <= 4%, not needed for MHA |
| private thread pool, any form | slower than torch's pool |
| software prefetch of the next page | neutral to worse |
| int8 `smmla` for Q@K^T on ARM | 3x MAC rate, but fails the 3e-2 gate |

**What is left** is fused-loop IPC on the D=128 long rows. The C++ backend's
results show which of these levers paid off (`docs/CPP.md` §5):
- bf16 P@V via `dpbf16ps` on key pairs (2.05x on the phase);
- non-temporal output stores;
- in-place V while it fits the cache.
