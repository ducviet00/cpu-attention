# vLLM `csrc/cpu` techniques vs this port

An inventory of what upstream does, whether we do it, and what is left. Read
against upstream `main` (`cpu_attn_impl.hpp` 2291 lines, `cpu_attn_vec.hpp`
327, `cpu_types_x86.hpp` 1164). Three people derived this independently (the
author plus two workers who were deliberately not shown these conclusions); the
rows marked **MISSING** are the ones that survived.

## What the micro-kernel actually is

`TileGemm82` (`cpu_attn_vec.hpp:37-130`) is not a magic GEMM — it is the same
broadcast-FMA loop this port uses:

- `M` query rows (1/2/4/6/8 chosen by `m_size`), `N = 2 x FP32Vec16 = 32 keys`.
- `c_regs[M * 2]`: 16 zmm of accumulator at `M = 8`.
- inner loop over `k` = head dim: **2 vector loads of B** (32 keys of one head
  dim from the transposed K page), then per row one broadcast of `a` and **2
  FMAs** (one per key vector). So `2 FMA per broadcast`, `8 FMA per B load` at
  `M = 8`.
- the accumulator lives in registers for the k-loop and is written to `c_tile`
  (memory) once per call; `accum_c` decides load-vs-zero for the P@V phase, so
  the output buffer is never separately zeroed.

We run the same loop with `M = 4` (`qt`) and 64-key tiles. The FMA-per-load ratio
at `M = 8` is 8 vs our 4 — but per-row broadcasts dominate the instruction
count either way, and the `qt` sweeps that change exactly this ratio measured
flat on the laptop and 1.2-1.34x on the servers, so it is a *traffic* effect, not
an issue-slot one.

## The inventory

| # | technique | upstream | ours | verdict |
|---|---|---|---|---|
| 1 | 8-row x 32-key register-tiled micro-kernel, acc in registers, k-loop over head_dim | `cpu_attn_vec.hpp:37-130` | `_dense_rows` (4 rows x 64 keys) | same shape, different constant; `qt` is the knob |
| 2 | **K^T stored block-major inside a page: `[head_dim, block_size]`, `k_cache_token_group_stride = BlockSizeAlignment = 32`** | `cpu_attn_vec.hpp:204-208`, page walk `cpu_attn_impl.hpp:1222-1276` | **no**: `_transpose_head` writes `[head_dim, seq_k]` | **MISSING — the one structural difference.** A Q@K^T tile then reads 128 rows at `block_size` stride inside a 64 KB page that the next 3 sub-tiles re-touch; ours reads 128 rows at `seq_k` stride (8 KB at S=2048/D=128 fp32) spread over 1 MB, so per tile we open 128 streams into a 1 MB stripe instead of streaming one 64 KB block. Same bytes, different locality: matches the measured L2 demand-miss 5.97% vs 0.07% and `bound_on_loads` 48% vs 15% |
| 3 | V row-major per page (`[block_size, head_dim]`), so P@V's B operand is 2 vectors of 32 contiguous dims per key | `cpu_attn_vec.hpp:216` | same (V is `[seq_k, head_dim]`) | have it |
| 4 | P@V accumulator in memory, `accum_c` = "add onto first iteration's output" so it is never zeroed | `cpu_attn_impl.hpp:1330-1360` | `_dense_rows_mem` does the L1 acc, but zeroes it explicitly | partial |
| 5 | P@V tiled over the head dim by `headdim_alignment = 32` | `:1330` | `dc = 2*LANES` (32 x86 / 8 NEON) | have it |
| 6 | logits written to a buffer by a whole-tile GEMM, then softmax, then P@V (one softmax per L2-sized tile of 480-1024 keys) | `calcu_default_tile_size` `:962`, loop `:1194-1360` | logits in registers, softmax per 16/64 keys | have it, smaller tile — **falsified as a cause**: per-tile fixed cost is 0.47 of 7.76 cycles/row-key, a 1024-key tile buys 5.7% |
| 7 | mask by range: write `-inf` into `[0,left)` and `[right,n)` (the mask is contiguous per row) | `apply_mask` `:1360` | per-lane `iota`+compare+`select` | missing, worth ~1% |
| 8 | rescale the output only when the tile max exceeds the running max by more than 8 (`< -8.0`, "FA4 threshold"), and then do not even update `m` | `:1468-1483` | rescale on any increase | missing, worth ~4% |
| 9 | probabilities alias the logits buffer (`curr_prob_buffer = reinterpret_cast<...>(logits_buffer)`), read back with a stride via `probability_token_store_t` | `:1318-1330` | separate `probs` array | missing, worth ~3% |
| 10 | fast exp: degree-5 poly + exponent bit-insert + clamp | `cpu_arch_macros.h:20-60` | `_vexp` (degree-8 Cephes) / `_exp2_fast` (ARM) | have it (this was worth +18% on x86 bf16 D=128) |
| 11 | bf16 P@V on `bfmmla` with a packed probability buffer (`TokenStride=2`, `row_in_pair`) | `cpu_attn_neon_bfmmla.hpp:330,371-390` | `_dense_rows_bfmmla` (both phases `bfmmla`, packed bf16 panels, linear K + position-major V) | have it |
| 12 | work items packed by **KV length**, one item accumulating many q-tiles, KV split across threads with a `PartitionReduction` combine | `AttentionScheduler::schedule` `:500-700` | one work item per q-block, round-robin | missing; **not needed** — 8192 tasks are already balanced, and a split buys only locality that a bigger tile gets for free |
| 13 | GQA: pack all query heads of one KV head into the 8-row tile (`request_q_head_group`) | `:456-470` | no | missing; only pays for GQA models (BART/Florence-2 are MHA) |
| 14 | AMX: 16x16 tiles, `MaxQHeadNumPerIteration = 32`, `TileGemm224`/`122` | `cpu_attn_amx.hpp:29,215` | no | missing; Xeon-only, bf16-only, 4-8x headroom there |
| 15 | `vec16` variant for `head_dim % 32 != 0` | `cpu_attn_vec16.hpp` | only 32/64/128 | missing, irrelevant to the shapes we run |
| 16 | no software prefetch in the VEC loop (`vec_op::prefetch` exists at `cpu_types_x86.hpp:1125` but `cpu_attn_*.hpp` never calls it) | — | we do not either | equal — upstream relies on the layout being streamable (#2) |

## The short list

1. **#2, block-major K^T.** The only row where the *storage layout*, not the
   arithmetic, differs, and it is exactly the shape of the measured stall
   profile. Our transpose already walks 16-token blocks, so writing
   `[seq_k/BS][head_dim][BS]` instead of `[head_dim][seq_k]` is a small change on
   the Python/writer side plus a stride in the kernel. Test: the same
   `ab_two_trees.py` A/B with `l2_rqsts` counted per arm.
2. **#14, AMX** for bf16 on Xeon, if Xeon is a deployment target — the only way to
   be competitive with SDPA there (it uses AMX: 5x our bf16).
3. **#8 + #9 + #7** as a group (~8% combined) once #2 is in — cheap, and they
   only make sense on top of a bigger tile.
4. **#13, GQA head packing** if the target models are GQA rather than MHA.

## Deliberately not ported

`#12` KV splits (no load-balance benefit at our task counts, and the combine
costs an `exp(m_local - m_global)` pass), `#6`'s L2-sized tile (falsified: 5.7%
even if the per-tile cost went to zero), `#15` vec16 (no such shapes).
