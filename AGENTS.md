# AGENTS.md — Mojo paged-attention port (handover)

Everything learned building `_mojo/src/attention.mojo`: a Mojo port of the hot
path of `csrc/` (the vendored vLLM CPU attention kernel), wired into the engine
as `attn_backend="mojo"`. Read this before touching the kernel; the root
`AGENTS.md` covers the engine side.

## 1. Status in one line

Two kernels, compile-time per-ISA paths (AVX512-BF16 `vdpbf16ps` + fp32 P@V on
x86; NEON `bfdot` QK **and** P@V on ARM): the paged decoder step
(`attention.mojo`) and dense attention (`dense.mojo`, `flash_attn_func`, §4a —
vLLM has no cache-free kernel to port, so that one is built from vLLM's
structure). Paged is bit-identical to `csrc/` on x86, ≤1e-3 on ARM (bf16
probabilities); dense is ≤4e-6 vs SDPA in fp32 and ~1 bf16 ulp in bf16, and is
1.3-2.5x SDPA on bf16 / 1.0-2.0x on fp32 (§9.4). Beats `csrc/` at `cross b8/ctx256` by 1.4× on
the 4-core Xeon and 2.0× on Graviton 5, parity-or-better elsewhere. It runs on
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
  registers and K/V are read once per row block. `qt = 4` for every head dim
  (`CPU_ATTENTION_QT` overrides): `qt * head_dim` above ~512 elements spills the
  accumulator, and 8 was 20-40% slower than 4, 2 was worse still.
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
Result: **2.2-3.2x**, 331 -> 853 GFLOP/s (D=64) and 348 -> 1093 (D=128); bf16
now beats SDPA at every D=128 size and up to S=1024 at D=64.

Still open, measured, not guessed: we are **1.44-1.8x behind vLLM** (and 5-13%
behind SDPA at D=64/S>=2048). Their `.so` runs 2 `bfmmla`/cycle where we run
1.2-1.35, and they issue ~16 instructions per row-key against our ~21 at IPC
3.7 vs 2.6. About 300 instructions per 16-key tile are **spill traffic**:
`qt/2 * sub/2` 2x2 tiles plus the softmax working set exactly fill the
32-register file, and the static asm still shows `str/ldr q, [sp, ...]` around
every tile. The lever is a register-lean (vLLM-style) softmax with the logits in
L1 and one row pair live at a time — §13.4. fp32 on ARM has no matrix-multiply
instruction, so it stays at ~250 GFLOP/s against vLLM's 480 and needs §13.3.

Also: the ARM build needs **libgomp linked into the extension**. Leaving `omp_*`
undefined for load time works on x86 because torch brings a global copy, but on
aarch64 torch's bundled `libgomp.so.1` is `RTLD_LOCAL` and `import` fails with
`undefined symbol: omp_get_thread_num`. `scripts/build.py` now links it (same
soname, so the loader still reuses torch's copy when it is already loaded).

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
| vector width, ISA detection | `simd_width_of[DType.float32]()`, `CompilationTarget.is_x86()/has_neon()/_has_feature["avx512bf16"|"bf16"]()` |
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
10. **Nothing about the dense kernel's blocking matters**: with drift-controlled
   A/B, `qt` = 2/4/6/8 sits within 1% at head_dim 128 (register spilling is not
   the limiter), and the KV-outer loop order (vLLM's: one (batch, head) per work
   item, L2-sized key chunks, `acc`/`m`/`l` in memory) is also within 1% there
   (+5-9% at head_dim 64, where it is not needed). An 8x smaller KV working set
   (GQA `Hk=1`) does run **1.72x** faster, but that is K/V *shared across heads*,
   which neither loop order reproduces per head — so the D=128 gap is not
   bandwidth. What is left is instruction density: vLLM is 1.2x ahead at
   head_dim 64 and 1.7x at 128, the ratio growing with head_dim, i.e. their
   `TileGemm82` micro-kernel does more per FMA than the broadcast-FMA inner
   loops here. Closing it is a GEMM-tile rewrite (§13.3), not a knob.

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

`benchmarks/attn_backend.py --mode kernel` + `benchmarks/kernel_ab.py`, two runs
each, threads = physical cores, workers = `_workers()`. Measured with the old
private pool; the libgomp switch removes ~4 µs of dispatch per call (a few % of
the short-context rows, nothing at 1024), so treat these as slightly
conservative for the current tree:

| box (threads/workers) | self b8/256 | cross b8/256 | self b8/1024 | cross b8/1024 | self b32/1024 |
|---|---:|---:|---:|---:|---:|
| Ryzen 7840H (8/8) | 0.76-0.88 | 1.02-1.18 | 0.84-0.92 | 0.92-0.96 | 0.86 |
| Xeon 6975P-C (4/4) | 0.67-0.70 | 0.71-0.84 | 0.89-0.90 | 1.00-1.02 | 0.85-0.90 |
| Graviton 5 (8/8) | 0.43-0.48 | 0.43-0.51 | 0.60-0.79 | 0.65-0.77 | 0.86-0.95 |

Absolute µs at `cross b8/enc256`: Xeon 116.7/114.4 → 82.5/85.5; Graviton
65.9/65.6 → 33.6/33.3; Ryzen 29.6/32.5 → 32.8/38.3 (parity). The laptop is
*faster in absolute terms* than the Xeon because it has 8 full-speed cores and
32 MB L3 vs a 4-core cloud slice; the repo already records that ordering for
decode GEMMs. What matters is the ratio on the same box.

Config findings worth remembering: on the Xeon `block_size=128` is ~9% faster at
ctx 1024 (a `Config`/CLI knob, not a kernel change); on Graviton the aarch64
default of 16 is also the C++ kernel's best point (67 vs 76 µs at block 32).

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
   ≤1e-3 vs the fp32 reference, matching `csrc`'s ARM `prob_buffer_t`. x86 stays
   bit-identical (`max|diff| 0.0`).
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
| Xeon 6975P-C | `ubuntu@13.210.181.255` | 8 vCPU = **4 physical cores**, `avx512_bf16` + `amx_bf16`; `~/venv`; torch default 4 threads |
| Graviton 5 | `ubuntu@13.236.70.191` | 8 cores, ASIMD + `bf16`; `~/venv`; torch default 8 threads |

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
3. **Dense: a GEMM micro-kernel for both phases** (vLLM's `TileGemm82`).
   Measured facts (§9.9-9.10): blocking does not matter, bandwidth does not
   matter, and the gap to vLLM grows with head_dim (1.2x at 64, 1.7x at 128), so
   the remaining cost is FMA *density*: 8 rows x 8-column register tiles for
   Q@K^T (one reload of K^T feeds 8 accumulator updates in registers instead of
   1) and the same for P@V, with the accumulator in memory. Both variants tried
   this round — head-dim-tiled acc in L1, and KV-outer chunks — were within 1%
   of the current kernel, so a rewrite must change the *inner* GEMM, not the
   loop order or the buffers.
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
   layouts stay). Expected ~1.2-1.3x, i.e. ~1.15x from vLLM rather than 1.45x.
5. **Dense: `bfmmla` on ARM for both phases** (done, see §4b). `bfdot` gives 2x the fp32-lane
   MACs, `bfmmla` gives 4x and amortizes the loads over a 2x2 output tile -- that
   is where vLLM's 1234 GFLOP/s bf16 on Graviton comes from (`cpu_attn_neon_bfmmla.hpp`
   is the file to read). It needs V in a *position*-paired layout
   (`[seq_k/2][head_dim][2]`), i.e. a second transpose pass, plus 2x4 / 4x2
   operand staging for `llvm.aarch64.neon.bfmmla`. Expect ~2x over the current
   ARM bf16 path, which is still 3.7x behind vLLM.
6. **GQA/MQA**: one KV page shared by `q_heads_per_kv` heads would let one task
   amortise the K/V loads over several query heads — and only then does AMX
   (16-row tiles) become interesting. Not needed for BART/Florence-2.
7. ~~Own thread pool~~ **Done, differently**: not a pool of our own (no win,
   §9.3) but `GOMP_parallel` on torch's pool. A hand-rolled pool cannot beat it
   because the win is *reusing* the pool that is already awake.
8. **`partial_simd_load`** for tails instead of `iota`+`select` (cosmetic).
9. **T5** (root `AGENTS.md`): same paged path, but its per-layer pointwise chains
   are what the frozen compile graphs fuse, so the CPU gains would be larger.
