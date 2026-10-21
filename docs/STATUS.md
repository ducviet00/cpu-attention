# Where the campaign stands

This file is the honest, measured state. `docs/PROTOCOL.md` says how numbers are
taken; `docs/BENCHMARKS.md` holds the full tables; this is the summary and the
log of what actually moved.

## The one number

`ours / baseline < 1`, for both baselines, on both platforms, for all 22 rows of
`tests/bench_vs_refs.py`. Below 1 is a win.

## The laptop (Ryzen 7840H, 8 cores, AVX-512 + avx512_bf16, no AMX), 2026-10-21

`tests/bench_vs_refs.py --vllm <csrc/cpu @ db3b0f0> --isa vec --rounds 7
--threads 8`, parity asserted per row (max|diff| 2.4e-7..1.6e-2, all inside
tolerance). vLLM's number is attention only; ours includes the K/V pack.

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | **0.52** | 1.19 | **0.58** | 1.52 |
| S=512  H=8  D=64  | **0.37** | **0.77** | **0.43** | 1.07 |
| S=1024 H=8  D=64  | **0.47** | **0.77** | **0.63** | 1.02 |
| S=2048 H=8  D=64  | **0.55** | **0.76** | **0.72** | 1.03 |
| S=4096 H=8  D=64  | **0.57** | **0.74** | **0.70** | 1.02 |
| S=256  H=16 D=128 | **0.49** | **0.93** | **0.59** | 1.15 |
| S=1024 H=16 D=128 | **0.56** | **0.90** | **0.61** | 1.27 |
| S=2048 H=16 D=128 | **0.63** | **0.92** | **0.68** | 1.21 |
| S=4096 H=16 D=128 | **0.72** | **0.97** | **0.79** | 1.20 |
| S=512  H=32 D=128 | **0.47** | **0.97** | **0.58** | 1.42 |
| S=1024 H=8  D=32  | **0.50** | **0.86** | **0.62** | 1.10 |

**22 of 22 rows beat SDPA. 10 of 22 beat vLLM.** At the start of this session it
was 4 of 22 against SDPA and 0 of 22 against vLLM on this box.

## What moved it (all measured interleaved, same box, same session)

| change | effect | where |
|---|---|---|
| reuse the pack workspace instead of `torch.empty` per call | **1.41-1.74x** | `_kernel.py`, 917285e |
| 16x16 register transpose for the K pack | 1.5-3x on the pack itself | `dense.mojo`, a038f8d |
| 8x64 fp32 dims to the L1-accumulator kernel | 1.05-1.10x | `dense.mojo`, c9f8e3d |
| fp32 6 rows for seq >= 256 (was 4) | 1.13-1.25x | `_kernel.py`, 8faf85e |
| bf16 head_dim 128 to the L1-accumulator dot kernel | 1.19-1.39x | `dense.mojo`, 9f5010a |
| P@V stops at `loop_end` | neutral, less work | `dense.mojo`, c5c0b55 |

The workspace one is the biggest single lever found so far and it was invisible
without `perf`: 16.8% of a call was `clear_page_erms`, the kernel zeroing
freshly faulted pages, and it landed inside the pack, which made the pack look
six times its real cost.

## What does *not* explain the rest (measured, not guessed)

Ablations on x86 fp32, interleaved A/B against a patched tree:

* the whole online-softmax block (no `reduce_max`, no `exp`, no `reduce_add`) is
  **8-20%** of the call, not the 40% the previous campaign's ARM ablation
  reported; it matters most at short S (1.40x at 128x8x64);
* the probability buffer round trip (P@V reading the buffer it was just written
  to) is **1-5%**;
* explicit `fma()` instead of `*`/`+` is **0%** -- the multiply/add pairs already
  contract on this toolchain, contrary to the note in `AGENTS.md`;
* dropping the per-tile `reduce_max` entirely (keep the first tile's max as a
  reference, with a `sum > 1e25` overflow fallback that recomputes the tile
  exactly) is **net zero**: 1.10x at 512x32x128, 1.00x at 256, 0.98x at 1024,
  0.96x at 4096, parity green. Preferring the shape that wins is not a result;
  this is the sixth softmax restructure to lose or tie.
* non-causal we are at **parity** with vLLM's VEC kernel (1024x16x128: ours
  618M instructions / 382M cycles, vLLM 572M / 364M, IPC 1.62 vs 1.57). Causal
  costs us 9-22% relative and vLLM nothing, and that is where the remaining
  fp32 gap lives at short and medium S.

## The open routes

1. **AMX for x86 bf16** (x86 agent). SDPA's bf16 is AMX-backed and 1.7-6.7x
   ahead of any AVX-512 kernel; nothing else closes that. A C shim is the only
   route, since Mojo cannot express `x86_amx` tiles and needs `arch_prctl`
   permission for XTILEDATA anyway.
2. **fp32 on AMX by splitting each fp32 operand into two bf16 halves** (3-term /
   3xTF32-style): `hi*hi + hi*lo + lo*hi` misses only a ~2^-16 term, i.e. ~1e-5
   absolute against a 2e-4 gate. Three `tdpbf16ps` per k-block against AMX's
   8-16x throughput advantage over AVX-512 fp32 FMA. fp32 has no instruction-set
   win of its own, so this is its only structural one.
3. **The causal 9-22%**: our work items are small at short S and pay a per-task
   prologue, two fork-joins and a Python wrapper (~3 us) that vLLM does not.
4. **ARM** (arm agent): the row kernel/softmax structure.
