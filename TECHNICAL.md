# Technical details

Design, API, numerics, tuning, and historical measurements for
`cpu-attention`. The user-facing overview and the current per-shape tables are
in [README.md](README.md); the scoreboard is `docs/STATUS.md`, the independent
re-measurement is `docs/AUDIT.md`, and the handover notes are `AGENTS.md`.

## Reading the numbers

Ratios are `ours / baseline`; below 1 is faster. Every published number is the
**minimum** of an interleaved run with rotated arm order, because these boxes
drift more than the effects being measured: the duplicate-arm floor is
0.92-1.04 on the audit boxes. Ours includes the K/V pack and the output
allocation; vLLM's cache is staged before the timer, so the ratios are
conservative for us.

Timing the paged kernel on a laptop needs one extra caveat: with `powersave`
EPP the clock changes between single calls, so a round-robin median is dominated
by the power state. The Ryzen paged numbers in the README use 40 consecutive
calls per arm after 8 warmups (the `benchmarks/kernel_ab.py` method), with the
arms alternated in blocks; the interleaved numbers are in
`audit/logs/ryzen/paged-perf.json` for comparison. At ctx 1024 the pages are
memory-roofline bound on both servers (231-298 GB/s on Graviton, 133 GB/s from
L3 on the Xeon), so those rows are insensitive to the kernel.

## What is (and is not) ported

Ported, from the x86 `VEC` implementation (`csrc/cpu/cpu_attn_vec.hpp`) with
its `TileGemm82` micro-kernel (`csrc/cpu/cpu_attn_impl.hpp`):

* the paged decoder **self-attention** (scatter K/V, then attend) and
  **cross-attention** over a staged encoder cache,
* the vLLM page *model* (head-major pages, K transposed inside a page, V
  row-major) and the fp32-accumulate numerics,
* `reshape_and_cache` in that layout, with the per-ISA operand packing the
  chosen dot instruction needs. fp32 on x86 is byte-identical to vLLM's cache;
  bf16 is not — vLLM packs 4 dims/positions for AMX/`bfmmla`, this kernel packs
  2 for `dpbf16ps`/`bfdot` — so a bf16 cache is not interchangeable between the
  two backends. `stage_cross_kv` stages through whichever backend is active.

There is a second kernel for the other half of flash-attention — **dense
attention** (`flash_attn_func`): whole `seqlen_q x seqlen_k` attention on
contiguous `q`/`k`/`v`, causal (bottom-right aligned) or not, MHA/MQA/GQA,
fp32/bf16. vLLM has no cache-free attention kernel (its only entry point is the
paged `cpu_attention_with_kv_cache`), so this one borrows vLLM's *structure*
(Q-tile of rows held in registers across the K/V sweep, transposed K, online
softmax) rather than a specific file. See "Dense attention" below.

Not ported (deliberately — the engine never calls them):

* multi-token queries on the *paged* path (chunked prefill), sliding windows,
  ALiBi, softcap, attention sinks, FP8 KV cache (the paged kernel keeps the
  `q_heads_per_kv` mapping but BART/Florence-2 are MHA),
* vLLM's per-ISA `VEC16` / `NEON` / `AMX` / `RVV` / `VSX` / `VXE`
  implementations and its work-item scheduler. (The dense kernel has its own
  AMX path; see below.) One task owns one `(sequence, head)` pair end to end,
  so there is no KV split, no scratch buffer and no cross-thread reduction.

The contract is the engine's (see `AGENTS.md` invariant 1): **one query token
per sequence, attending to every cached position**. The query is always the
last position, so the causal mask is exactly the cache tail.

## Build details

The kernel dispatches on libgomp (`GOMP_parallel`). The extension itself does
not link it — the symbols resolve when the library is loaded, and `cpu_attention`
imports torch first, which brings `libgomp.so.1` in — but the standalone test is
an executable and does need the library at link time, hence `scripts/build.py --tests`.
`mojo run` cannot run the kernel either (its JIT cannot resolve the OpenMP
symbols), which is why the build script produces a binary.

`mojo build` targets the **host CPU** (`--print-effective-target` shows
`znver4 +avx512bf16` on the dev box), so rebuild per machine rather than copying
the `.so`. In fast-encdec the engine-level checks (`tests/test_cpu_attention.py`)
and the A/B scripts (`benchmarks/kernel_ab.py`, `benchmarks/attn_backend.py`)
run against the installed package.

## API

The public entry points mirror `flash_attn`:

```python
flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    block_table=None,
    slot_mapping=None,
    check_inputs=True,
)
flash_attn_func(q, k, v, softmax_scale=None, causal=False)
```

`flash_attn_with_kvcache` accepts every flash-attn argument; what the kernel
cannot do raises `NotImplementedError` instead of being silently ignored: `qv`,
`rotary_*`, `cache_leftpad`, `cu_seqlens_*`, `max_seqlen_q`, `*_descale`,
`window_size`, `softcap`, `attention_chunk`, `return_softmax_lse`, and
`seqlen > 1` (the paged kernel decodes one query token per sequence — use
`flash_attn_func` for whole prompts). The tuning hints (`num_splits`,
`pack_gqa`, `sm_margin`, `scheduler_metadata`, `rotary_interleaved`) are
accepted and ignored.

On x86 CPUs with **AMX** the dense path is a different kernel:
`cpu_attention/_mojo/src/dense_amx.mojo`, 32 query rows per work item, K and V
packed into the AMX operand layouts, one softmax pass per 256-key super-tile, and
`tdpbf16ps` for both phases — which is worth 2.8-9.8x over the AVX-512 path
(4-9.6x from 512 keys up; the short rows are 2.8-3.2x) and takes bf16 from
1.7-6.8x behind torch SDPA on a Xeon to 1.4-2.4x ahead of it.
It is selected at compile time (`CompilationTarget._has_feature["amx-tile"]`) and
gated at run time on `arch_prctl(XFEATURE_XTILEDATA)`, so every other CPU, and
`CPU_ATTENTION_AMX=0`, falls back to the AVX-512/NEON kernels below. The AMX tile
instructions are emitted from Mojo inline assembly (`amx.mojo`) — a C shim is not
needed; `AGENTS.md` §14 has the details and the one trap (AT&T operand order puts
the destination tile last).

`flash_attn_func` keeps flash-attn's dense shape — `q [B, Sq, H, D]`,
`k`/`v [B, Sk, Hk, D]`, MQA/GQA, bottom-right-aligned `causal`, `softmax_scale` —
and *drops* the arguments the kernel cannot honour (`dropout_p`, `window_size`,
`softcap`, `alibi_slopes`, `deterministic`, `return_attn_probs`) instead of
taking them and raising. `flash_attn_varlen_func` and the packed variants raise
`NotImplementedError`.

Deliberate differences from flash-attn:

* the cache is this backend's head-major
  `[num_blocks, nheads_k, page_block_size, headdim]` (`attention.self_cache_shape`),
  not flash's page-major layout — K is stored transposed inside the page, which
  is exactly what the bf16 dot products read, so the two are not
  interchangeable;
* `causal` is accepted and has no effect: one query at the last cached position
  keeps every key under the bottom-right-aligned mask;
* with `k`/`v`, they are written at `cache_seqlens` (the length *before* the
  write, flash's convention); passing `slot_mapping` writes them at those slots
  and reads `cache_seqlens` as the length *after* the write, which is what the
  engine's per-step Context holds.

`check_inputs=False` skips the argument validation. The engine's per-layer path
calls the thin aliases `self_attention(...)` / `cross_attention(...)` (the same
kernel call, geometry already checked by the runner): the checked entry point
costs ~4% end to end at the engine's 24 calls per decode step, ~2% with
`check_inputs=False`, so the aliases stay on the hot path. `flash_attn_func`
runs the dense kernel below for any `seqlen_q` (one query token included).

## Design

* **Parallelism: one task per (sequence, head), on torch's own pool.** The work
  is embarrassingly parallel across heads because every head owns a disjoint
  slice of the paged cache, so no reduction is needed. 128 tasks at batch 8 /
  16 heads, and a thread only ever streams its own head's pages, which keeps the
  access pattern prefetch-friendly. The dispatch is `GOMP_parallel` (libgomp,
  linked with `-Xlinker -lgomp`), i.e. **the OpenMP pool torch already keeps
  warm** rather than a second pool of our own: 2.4 µs per region warm and 2.7 µs
  with a 1 ms gap, against 6.3 µs / 103-140 µs for a private pool that has to be
  woken (see "Where the time goes"). The kernel is memory-bandwidth-bound, like
  the C++ one.
* **Page model: vLLM's; operand packing: per-ISA.** A page is
  `[num_blocks, num_kv_heads, block_size * head_dim]` with K stored
  *transposed* inside the page (`[head_dim, block_size]`, row-major) and V
  row-major (`[block_size, head_dim]`). Q@K^T then reads one contiguous
  `block_size`-wide row of K per head dim and accumulates a TILE-wide vector,
  and P@V reads one contiguous `head_dim`-wide row of V per position. That is
  also why `reshape_and_cache` writes K column-wise, like the C++ kernel. The
  bf16 dot paths re-pack the page for the instruction (`[head_dim/2,
  block_size, 2]` for K, position-paired V on ARM); vLLM's AMX/`bfmmla` paths
  use 4-wide groups instead, so the **bf16 caches are not interchangeable**
  even though fp32/x86 is byte-identical.
* **Tiles, groups and masks.** A page is one 32-token tile (vLLM's
  `BlockSizeAlignment`, hence `block_size % 32 == 0`; 16 on aarch64 pages). Four
  pages are scored into one **group** before the online-softmax step, which is
  how the C++ kernel amortizes its softmax over an L2-sized tile — per *page*
  softmax bookkeeping is 4x more max/exp/rescale work at short contexts. The
  last page of a sequence masks positions past `context_lens` with `-3.4e38`
  (not `-inf`: `exp()` underflows it to 0 and it cannot produce a NaN).
* **Online softmax per group.** `m_new = max(m, max(group))`,
  `l = l * exp(m - m_new) + sum(exp(group - m_new))`,
  `acc = acc * exp(m - m_new) + P @ V`. Probabilities stay in registers; the
  C++ kernel writes them to a buffer and reloads them as broadcast operands
  (a 4-byte reload out of a 64-byte store), which costs a store-forwarding
  stall per position. The C++ kernel also skips the rescale unless the max
  moves by more than 8 (an FA4 trick); rescaling unconditionally costs
  `head_dim / 16` multiplies per group.
* **Q@K^T as AVX512-BF16 dot products** (`llvm.x86.avx512bf16.dpbf16ps.512`,
  gated on `CompilationTarget` at compile time, fp32 broadcast-FMA otherwise).
  One instruction folds two head dims into each of 16 position lanes, so the
  query-key phase takes half the instructions of the widened fp32 form
  (measured 1.87x on this box: 34 vs 63 ns per 32-position tile). Two bf16
  values multiplied into fp32 are exact, so this is the same math as widening
  first — bf16 output matches the C++ kernel to 0-2 ulp (exactly at batch 1 /
  ctx 256). It needs K stored
  as `[head_dim/2, block_size, 2]` so a `(k[2d], k[2d+1])` pair of one position
  is contiguous; the portable path keeps the plain `[head_dim, block_size]`
  layout. Q is not pre-scaled on this path (it would round through bf16);
  `scale` is applied to the logits instead.
* **P@V: fp32 on x86, bf16 dot on NEON.** A dot instruction needs the two
  contracted positions adjacent, so the ARM path stores V position-paired
  (`[block_size/2, head_dim, 2]`) and rounds the probabilities to bf16 — exactly
  what `csrc` does on ARM (`prob_buffer_t`). One `bfdot` then retires 8 MACs
  instead of 4, and the tile's contribution is folded into the fp32 accumulator
  once per tile. This is the only place the port gives up bit-exactness, and
  only on ARM (x86 keeps the fp32 P@V and agrees with `csrc/` to ~2e-7).
* **Strides.** `q`/`k`/`v` arrive as chunks of the fused `qkv_proj` output, so
  a token sits `3 * d_model` floats apart, not `num_heads * head_dim`. The
  kernel takes the token strides explicitly, like the C++ entry point does;
  without that a batch of 2+ sequences reads the wrong rows.
* **Dispatch.** `head_dim` is a comptime parameter (32/64/128 specializations,
  anything else raises), `dtype` is fp32 or bf16 (fp16 is rejected — this
  project's CPUs have no `avx512_fp16`).

## Numerics

On x86, bf16 output matches the vendored C++ kernel to 0-2 bf16 ulp on the
engine's shapes (bit-identical at batch 1 / ctx 256; one ulp at batch 8 /
ctx 1024 from a different summation order), and fp32 agrees to 1.5-2.6e-7. On
ARM the probabilities are rounded to bf16 (as in `csrc`'s `prob_buffer_t`), so
bf16 agrees with the fp32 reference to ≤1e-3 rather than bit-exactly. These
were re-measured on m9g and m8i in `docs/AUDIT.md` §6.

## Dense attention

`flash_attn_func(q, k, v, softmax_scale=None, causal=False)` — one work item owns
`qt` consecutive query rows of one (batch, head) for the whole key sweep, so its
online-softmax accumulator lives in registers and K/V are read once per row
block. Keys are scored `KT_TILE` at a time; for that, K is handed over
*transposed* (one `reshape/permute` produces it, and the kernel has a blocked
parallel transpose pass for it, 40-70 GB/s against 11-15 GB/s for
`torch.permute(...).contiguous()`), which is what lets one broadcast-FMA cover
64 positions. V stays row-major, so `acc[d] += p[j] * v[j][d]` is one FMA per
head-dim chunk shared by every row of the block. bf16 runs both phases on the
2x2 matrix-multiply where one exists (`tdpbf16ps` on AMX, `bfmmla` on NEON) and
on the bf16 dot instruction (`dpbf16ps` / `bfdot`) with fp32 P@V elsewhere, which
is why bf16 is **faster than fp32** here — the opposite of a bf16-native kernel
on AVX-512, which has no native bf16 add/multiply.

Work items are handed out round-robin over the same libgomp pool: causal masking
makes the last query rows ~2x more expensive than the first, and a contiguous
slice per thread leaves half the cores idle at the end of a prefill.

Rows per work item is the only axis K and V are reused across, so traffic per
row-key is `head_dim * sizeof(T) / qt` bytes for Q@K^T and again for P@V, while
the register file caps `qt * head_dim` near 512 (above that the accumulator has
to live in L1: `_dense_rows_mem`). The shipped rule is `_kernel._qt_for`, swept
per (ISA, dtype, head_dim, seqlen) and documented there: on ARM 8 rows for both
dtypes (the tile kernels keep the logits in L1, so rows are free); on AVX-512
fp32 6 rows at head_dim 64 and 8-10 at 128, bf16 4 at 64 and 6-8 at 128. Measured,
interleaved, on the Ryzen: 1.13x at 256x16x128 and 1.25x at 512x32x128 for 6 rows
over 4, and 1.10x at 1024x8:64 for 6 over 8. `CPU_ATTENTION_QT` overrides it for
experiments, and it is part of the per-geometry call plan's key so a one-process
sweep still measures different kernels.

### Against torch SDPA and vLLM

The current table, the measurement protocol and the list of what moved it live in
`docs/STATUS.md`; it is regenerated from `tests/bench_vs_refs.py --json` with
`scripts/scorecard.py` (or `scripts/band.py` for a per-row band over repeat runs),
never typed by hand. The campaign's closing report is `docs/CAMPAIGN.md`, and
`docs/CEILINGS.md` says how much headroom exists per row before anyone spends
another session on a schedule tweak. Headline, Ryzen 7 7840H, 8 threads,
causal, ratios `ours/baseline` (`<1` wins). The matrix was re-measured
independently on 2026-10-23 (`docs/AUDIT.md` §1-§5). On the Xeon 6975P-C
(m8i, 4 threads): **22/22 rows beat vLLM's VEC kernel** (bf16 and fp32 both go
through AMX; fp32 by splitting each operand into bf16 hi/lo halves, error
3.1-7.7e-5 against the 2e-4 gate), with `128:8:64 fp32` at parity (0.95 min /
1.04 median). The honest bf16 bar on an AMX box is vLLM's **AMX** build, not
VEC: against that, **9/11 rows win on the min** (worst 1.02) and 6/11 on the
median (worst 1.09) -- parity, not a win, with the D=128 long rows at 0.95-0.98
on the min. On Graviton 5 / m9g (8 threads): **21-22 of 22 beat vLLM's NEON
kernel** (10/11 bf16, 11/11 fp32; bf16 worst 1.02 on the median / 0.997 on the
min, fp32 worst 0.98) and 22/22 beat SDPA; on a 4-thread slice of the same box
it is 22/22 vs vLLM. `o/sdpa` is 0.28-0.87 on ARM and 0.21-0.79 on x86, so
every row beats SDPA on both. The old "8 of 22 on a 4-core slice" predates
`bfmmla` and the per-head dispatch; `docs/CEILINGS.md` is the historical
ceiling analysis of the 4-vCPU m9g.xlarge.

The three moves that mattered most, in order, were: **reusing the K/V pack
workspace** instead of `torch.empty` per call (1.4-1.7x -- the pack was writing
into freshly faulted pages, and `clear_page_erms` was 16.8% of a whole call);
**rows per work item** (above); and sending 8x64 fp32 and 8x128 bf16 to the
L1-accumulator kernels instead of spilling the register accumulator (1.05-1.10x
and 1.19-1.39x). The older claims in this section -- that per-head-contiguous K/V
was worth 2.0x, or that the softmax block is 40% of an x86 call -- did not
survive re-measurement: the ablation in `docs/STATUS.md` puts the softmax at
8-20% on x86 and 1-5% for the probability round trip.

### On Graviton 5 (Neoverse-V3, 8 cores) -- previous campaign, superseded

The numbers below are from an older tree and are kept only as history; the
current ARM table is re-measured per session (see `docs/STATUS.md`).

The dense kernel was first tuned on AVX-512 and it showed on ARM. bf16 now runs
both phases on `bfmmla` (the 2x2 bf16 matrix-multiply, which vLLM's NEON kernel
uses and which `bfdot` does not reach): Q@K^T as 2 rows x 4 dims against 2 keys x
4 dims, P@V as 2 rows x 4 keys against 2 dims x 4 keys, so the logits *are* the
Q@K^T accumulators and the softmax runs a row pair at a time. K gets a
position-paired layout and V a position-major one, both built in the same
transpose pass; the softmax uses a magic-constant `exp2` (~10 instructions
instead of `std.math.exp`'s ~22, which was 20% of the call).

Causal, bf16, ours before -> after, against vLLM's `csrc/cpu` NEON kernel and
torch SDPA (same session, interleaved):

| case | before | after | vLLM | SDPA |
|---|---:|---:|---:|---:|
| S=512 H=8 D=64 | 857 | **390** | 236 | 542 |
| S=1024 H=8 D=64 | 3277 | **1376** | 911 | 1451 |
| S=2048 H=8 D=64 | 12972 | **5038** | 3417 | 4719 |
| S=4096 H=8 D=64 | 51467 | **19901** | 13691 | 17397 |
| S=256 H=16 D=128 | 858 | **386** | 217 | 487 |
| S=1024 H=16 D=128 | 12824 | **4164** | 2868 | 4829 |
| S=2048 H=16 D=128 | 49432 | **15715** | 10931 | 17027 |
| S=512 H=32 D=128 | 6687 | **2371** | 1501 | 3722 |

Rows per work item is the lever that binds on ARM as it does on x86: the fp32
default used to be 2 rows (the worst measured point), and `qt=6` is
**1.37-1.39x** on the whole call at D=64 (S=1024/2048) -- that row went from
0.79x SDPA to **1.15x**. Decoupling `qt` from the bf16 tile geometry was tried and
rejected (0.86x: 32 logit registers at `RP=4 x NS=8` spill).

That campaign's result was 2.2-3.2x, and 331 -> 853 GFLOP/s at D=64 / 348 ->
1093 at D=128 (both S=2048), **still 1.44-1.8x behind vLLM at the time**. It is
not the current state: the merged tree measures **0.79-1.02 vs vLLM** and
**1.34-1.59 TFLOP/s** at the long rows on m9g (8 threads; `docs/AUDIT.md` §2).
fp32 on ARM was 209-255 GFLOP/s then and is 0.49-0.55 TFLOP/s now; there is no
matrix-multiply instruction for it, so it stays on the GEMM-tiled path
(`AGENTS.md` §13.3-13.4).

**Read these as bands, not points.** This laptop throttles to ~2.0 GHz under
sustained AVX-512 (and a second session on the box makes it worse), and the
fp32 / `D=128` causal ratio measured anywhere from 0.63 to 1.09 for the same
binary depending on when it ran. Interleaved A/B in one process is the only
number to trust; `tests/bench_dense.py` blocks per variant, so use it for
orders of magnitude. Blocking the kernel further is not where the remaining gap
is: `qt` 2/4/6/8 and a KV-outer (L2-chunked) loop order both measured within 1%
at `D=128`, and coming level with vLLM there needs a GEMM-tile inner loop
(`AGENTS.md` §9.9-9.10, §13.3).

Absolute throughput on the Ryzen 7 7840H (causal, S=2048): bf16 936 GFLOP/s at
H=8/D=64 and 645 at H=16/D=128; fp32 700 and 444. For reference, vLLM's
`csrc/cpu` `VEC` kernel in its chunked-prefill mode reached 915 GFLOP/s fp32 on
this box (staged cache, no transpose), and torch SDPA 570 — so the bf16 path
matches vLLM's fp32 and beats it outright once the key transpose is counted in.
The servers are faster: at the same shapes m9g measures 1.34/1.54 TFLOP/s bf16
and 0.52/0.53 fp32, m8i 1.93/2.44 bf16 and 0.96/1.00 fp32 (`docs/AUDIT.md`
§2, §4b). The one shape where fp32 still loses on this laptop (`H=16 D=128`,
mid-size) is the register wall: `qt * head_dim` above ~512 spills the
accumulator, and `qt` cannot grow, so K^T/V are re-read once per 4 rows.

## End to end (engine)

`benchmarks/benchmark.py --scenarios '256->256'` and
`benchmarks/benchmark_florence2.py`, dev Ryzen 7840H, 8 threads, one engine per
process, Mojo and the vendored C++ backend measured in the same session:

BART-large-CNN, batch x 256 generated tokens:

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 3324 | 77.0 | 3611 | 70.9 | **0.92** | 1.66x |
| batch=8, beams=1, bf16 | 5060 | 404.7 | 5157 | 397.2 | 0.98 | 1.75x |
| batch=8, beams=4, bf16 | 7661 | 267.3 | 8054 | 254.3 | 0.95 | 4.16x |
| batch=8, beams=1, fp32 | 8527 | 240.2 | 8977 | 228.1 | 0.95 | 2.95x |

Florence-2-base on a real 640x480 photo, four tasks (caption / detection /
referring-expression segmentation / caption-to-phrase grounding), EOS-stopped so
the output lengths are the tasks' own (13/22/256/22):

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 398 | 32.7 | 427 | 30.5 | 0.93 | 1.51x |
| batch=8, beams=1, bf16 | 4344 | 144.1 | 4454 | 140.6 | 0.98 | 2.02x |
| batch=8, beams=4, bf16 | 4873 | 128.5 | 4953 | 126.4 | 0.98 | 4.99x |
| batch=8, beams=1, fp32 | 10070 | 62.2 | 10170 | 61.6 | 0.99 | 2.28x |

So end to end the backend is now **ahead of the C++ one on every row**: 2-8% on
BART (widest at batch 1), 1-7% on Florence-2. Before the move onto torch's
OpenMP pool it was 3-9% behind on BART and 2-6% on Florence-2; the difference is
the per-call dispatch (2.4 µs vs 6.3 warm, and no cold wake), which is most of a
short attention call. Both engines beat HF by 1.5-5x on these points.

## Where the time goes (and what not to redo)

Measured on the Ryzen box with the engine's call pattern, not the microbenchmark.

**The dispatch is the whole story, and it is solved.** An attention call is a
fine-grained parallel region with ~0.8-1.5 ms of torch GEMMs between calls, so a
pool that sleeps pays a wake every time. Measured per dispatch:

| dispatch | µs |
|---|---:|
| inline, 1 worker (no region) | 0.36 |
| **`GOMP_parallel` on torch's pool, warm** | **2.4** |
| **`GOMP_parallel`, 1 ms idle gap, no torch work** | **2.7** |
| `max.algorithm.parallelize`, private pool, warm | 6.3 |
| same, 1 ms gap (the engine's cadence) | 103-140 |
| a hand-rolled spin-then-condvar pool, 1 ms gap | 103 |

The last two rows are why the backend no longer uses MAX's runtime: a private
pool either sleeps (103-140 µs wake, 24 times per step) or spins (which steals
GEMM cycles — MAX's default 200 µs spin cost 4.5% end to end and was worse than
useless). Running the kernel on the pool that *torch* keeps busy between calls
removes the wake entirely, and it also removes the need for the `max` package:
the build needs only the `mojo` wheel (uv's `build` extra).

What that is worth, measured in one harness (8 matmuls ~ 0.8 ms between calls,
i.e. the engine's cadence; per-call numbers include the call itself):

| b8/ctx256 | attention call | GEMM-phase interference |
|---|---:|---:|
| C++ backend | 90.7 µs | +25.9 µs |
| this backend | **29.8-49.6 µs** | +39.2 µs |

And end to end on bart-large-cnn, 8x128 tokens, same session:

| | tok/s |
|---|---:|
| this backend, before the dispatch change | 344 |
| this backend now | **392-396** |
| C++ backend | 364-377 |

So the backend is now **faster than the C++ one end to end on this laptop**
(+7.5% at 8x128, +4.8% at batch 1 x 128), and its attention call is ~3x cheaper
than vLLM's at `cross b8/256`.

Remaining host-side cost, all measured: `torch.empty_like` 0.53 µs, 9
`data_ptr()` 0.53 µs, pybind with 23 args 0.68 µs, the Python wrapper ~3.1 µs
(callable and worker override resolved once, one `stride()` per tensor). Tiny
calls run **inline** (`_workers()` returns 1 when `num_tasks * pages <= 32`:
24.5 µs vs 31.0 µs at batch 1 / ctx 64); from batch 1 / ctx 128 up the pool wins
(1.3x), and 8x at batch 8 / ctx 256.

**On a 4-physical-core Xeon 6975P-C (m8i) the picture inverts.** The kernel is
bandwidth-bound (133 GB/s from L3 at this shape, 62-70 GB/s from DRAM, both
measured by pointing every sequence at one shared page set), the C++ kernel is
compute-bound at 81 GB/s, and bf16 beats it on every row at 4 threads —
**self 0.53-0.92, cross 0.59-0.97** (min; medians 0.76-0.94; `docs/AUDIT.md`
§5). fp32 does not: it is 1.04-1.20x slower at b8/256 and b8/1024. There,
prefetching K/V (3 strides), `GROUP` 2/8 and 6/8 workers were all
neutral-to-worse; `_workers()` is capped at the physical core count (sysfs,
cached) — torch already sizes its pool that way, so the cap only matters when
a parent sets a larger `OMP_NUM_THREADS`. `block_size=128` measured ~9% at
ctx=1024 on the previous 4-core box (config-level:
`attn_backend.py --block-size 128`).

## Tuning and knobs

The defaults are what the scoreboard measured; change these only with an
in-process A/B (the boxes drift more than the effects).

* **Threads.** `torch.set_num_threads(physical cores)` is the right setting.
  The kernel runs on torch's libgomp pool and `_workers()` caps itself at the
  physical-core count, so setting a larger `OMP_NUM_THREADS` only oversubscribes
  the fork-join. `CPU_ATTENTION_WORKERS` pins the worker count for A/B.
* **dtype.** bf16 is faster than fp32 on both ISAs (2-3x on the dense kernel at
  the long rows) and is the engine default. fp32 on x86 goes through the AMX
  hi/lo path and is accurate to 3.1-7.7e-5; on ARM it has no matrix-multiply
  instruction and is the slow column.
* **Page size.** 32 on x86 and 16 on ARM are the engine defaults; the paged
  kernel accepts any multiple of 16. `block_size=128` measured ~9% at ctx 1024
  on the previous 4-core Xeon; on m9g block 32 is better for our kernel at
  b32/1024 and worse at b8/1024, so keep 16 unless you measure.
* **AMX (x86).** On by default (`CPU_ATTENTION_AMX=1`); `0` falls back to the
  AVX-512 path. `CPU_ATTENTION_AMX_FP32=0` disables the fp32 hi/lo path;
  `CPU_ATTENTION_AMX2=1` selects the opt-in v2 kernel (1.51x slower at
  128:8:64, parity on the long rows). `CPU_ATTENTION_AMX_KT` (default 256) is
  the key-super-tile; 128 is 10% faster at 128:8:64 and 7% slower at
  4096:16:128. `CPU_ATTENTION_AMX_MODE=512` forces the old phase-split order
  for an in-process A/B of the per-head work item.
* **Rows per work item.** `CPU_ATTENTION_QT` overrides `_kernel._qt_for`
  (4-10 on x86, 8 on ARM). Measured: fp32 `qt=8` is 1.45-1.61x over qt4 on
  ARM; bf16 6/8/12/16 are within 1-3%. fp32 values must be multiples of 4 on
  ARM.
* **Cache staging.** `stage_cross_kv` must stage through the active backend's
  `reshape_and_cache`: fp32/x86 caches are byte-identical to vLLM's, bf16
  caches are not (2-wide vs 4-wide packing). A cache written for one backend
  and read by the other returns garbage.
* **Benchmark protocol.** One process, interleaved arms with rotated order,
  41 rounds for rows near 1.00, and the **min**, with a duplicate `ours` arm as
  the noise floor (0.92-1.04 on these boxes). `tests/bench_vs_refs.py
  --rounds 41 --control`; `audit/` is the independent harness with a naive
  fp32 reference and a separate cache layout per backend.

## Layout

```
cpu_attention/
  __init__.py                   public API (flash-attn names + low-level ops)
  flash_attn_interface.py       flash_attn_with_kvcache / flash_attn_func / rejects
  ops.py                        self_attention / cross_attention / reshape_and_cache
  _kernel.py                    .so loader, 23-arg call, worker policy
  _C.so                         built kernel (gitignored)
  _mojo/
audit/                          independent re-measurement harness (naive reference)
docs/AUDIT.md                   the 2026-10-23 audit, raw JSON in audit/logs/
scripts/build.py              uv-driven build (mojo wheel from PyPI)
    src/attention.mojo          the kernel (Job, store, attention, dispatch)
    src/_C.mojo                 Python extension module (PyInit__C)
    tests/test_attention.mojo   standalone parity test against a scalar reference
    tests/bench_qk.mojo         isolated QK micro-benchmark
tests/test_flash_api.py         flash-shaped API vs an SDPA reference
```
