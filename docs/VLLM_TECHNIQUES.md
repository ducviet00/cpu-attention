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
| 2 | K^T stored block-major inside a page (`[head_dim, block_size]`, `k_cache_token_group_stride = 32`) | `cpu_attn_vec.hpp:204-208`, walk `cpu_attn_impl.hpp:1222-1276` | we write `[head_dim, seq_k]` | **FALSIFIED as a cause — do not port.** It correlates with the L2 miss rate (5.97% vs 0.07%) but the correlation is not causal: overriding only the strides the kernel is handed, so that the K and V panels shrink to **88 KB total (L1-resident)** and the strided walk becomes dense, buys **1-2%** (`PROFILE_X86.md`, stride-override probe). The `l2_rqsts.demand_data_rd_miss` counter is a vPMU artefact on that box — it reports the same absolute count (5.7 M/call) for an 88 KB set as for a 1 MB one. Keeping our layout also protects the paged kernel, which already beats vLLM |
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

0. **What actually binds (measured, after #2 died).** Two deficits, and neither
   is bytes: (a) the **load:FMA ratio sets a hard floor** — at `qt=4/kt=32` the
   Q@K^T loop needs 6 zmm loads per 8 FMAs, so `llvm-mca` gives 1.0 FMA/cycle even
   with every load in L1, and we run 0.42; (b) the **serial softmax chain**
   (`reduce_max -> exp -> reduce_add -> rescale`) drains the pipeline every 32-64
   keys — a tile is ~5 000 instructions, the reorder buffer cannot span two, and
   vLLM reaches IPC 2.94 where we reach 1.41 *with fewer instructions per
   row-key* (32.9 vs 35.3). Both point at the same lever: **more rows per K/V
   sweep** (`qt`), which raises the FMA-per-load ratio and shortens the softmax
   share. Measured on x86: `qt=8` is 1.19x (D=64 fp32) / 1.14x (D=128) at 1-8
   threads, and the row effect is 1.18-1.19x *independently of which kernel runs*
   (register 1.19x, L1-acc 1.18x — the L1-acc kernel is 6% slower wherever both
   are legal, so raise `qt`, do not port the mem path).
2. **Softmax off the inner loop** at `qt=8` with the vLLM structure (logits and
   probabilities in a buffer, `kt` up to 512): predicted to lift IPC 1.57 -> >=2.0
   for <=15% more instructions. Kill switch: abandon if IPC stays below 1.8 or
   `in/pair` rises above 15%.
3. **#14, AMX** for bf16 on Xeon, if Xeon is a deployment target — the only way to
   be competitive with SDPA there (it uses AMX: 5x our bf16).
4. **#8 + #9 + #7** as a group (~8% combined) once #2 is in — cheap, and they
   only make sense on top of a bigger tile.
5. **#13, GQA head packing** if the target models are GQA rather than MHA.

## Deliberately not ported

`#12` KV splits (no load-balance benefit at our task counts, and the combine
costs an `exp(m_local - m_global)` pass), `#6`'s L2-sized tile (falsified: 5.7%
even if the per-tile cost went to zero), `#15` vec16 (no such shapes).
