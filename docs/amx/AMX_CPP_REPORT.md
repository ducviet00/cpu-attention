# C++ AMX dense kernel — `cpu_attn_amx.hpp`

Branch `cpp-amx` (based on `09ec7e3`), own file only: `cpu_attention/csrc/cpu_attn_amx.hpp`.
Everything below was measured on a dedicated `m8i.2xlarge` (Xeon 6975P-C, 8 vCPU =
**4 physical cores**, `amx_tile`/`amx_bf16`/`amx_int8`, 48 KB L1d, 2 MB L2,
480 MB L3 reported, us-west-2 spot), 4 torch threads, all arms interleaved and
rotated in **one process**, `min` of 41 rounds.

## 1. What it is

One `tdpbf16ps` accumulates a 16x16 fp32 tile from two bf16 operands, and the
tile is a 16x16 matrix of **32-bit units** (two bf16 each):

```
C[i][j] += sum_r dot2( A[i][2r..2r+1], B[r][2j..2j+1] )
```

so **A's rows are M and B's rows are the contraction index**, N along B's
columns. For Q@K^T that means Q is read straight out of the caller's tensor
(row = query, 32 head dims contiguous, any stride) and K must be packed as
`[dim pair][key]`; for P@V, P is row-major bf16 (row = query, 32 keys) and V is
packed as `[key pair][output dim]`. `tests/amx_smoke.mojo` derives the same two
layouts from 512 single-bit probes and vLLM's `reshape_and_cache` states them
("head elements packed as quadwords"), so the kernel and both references agree
structurally.

Work item = (batch, q head, 32 query rows) — the driver's `rq`/`RQT`. Inside it
the key range is walked in super-tiles of `kt` = 256 (128/512 legal, worse):

* Q@K^T: 32 keys per step (2 K tiles x 2 Q row groups = 4 C tiles) into an fp32
  logits buffer, 32 rows x kt.
* softmax: one pass over the super-tile, in place, emitting bf16 P (plus its
  hi/lo halves for fp32) into a per-thread P buffer.
* P@V: 32 keys x 32 output dims per step, accumulating into the output buffer,
  so the output is loaded/stored once per super-tile instead of once per 32 keys.

Eight tile registers is the budget: 2 A + 2 B + 4 C. head_dim 128 therefore needs
four P@V passes over the super-tile (the C tiles span 32 dims); P is re-read from
L1 each pass, V is read once. That is vLLM's `TileGemm224` shape and the Mojo
tree's.

**fp32** rides the same kernel with every operand split into bf16 hi + bf16 lo
and three products per phase (hi*hi + hi*lo + lo*hi, lo*lo dropped at 2^-16):

```
x = hi + lo,  hi = bf16(x),  lo = bf16(x - fp32(hi)),  |lo| <= 2^-8 |x|
```

Products are exact in bf16xbf16->fp32 and the accumulation is fp32, so the result
is fp32-accurate at 3x the tile ops.

## 2. Parity (gates: fp32 <= 2e-4, bf16 <= 3e-2 vs a naive fp32 reference)

```
$ CPU_ATTENTION_BACKEND=cpp .venv/bin/python tests/test_dense.py
cpu_attention dense: 65/65 checks passed
$ CPU_ATTENTION_BACKEND=cpp .venv/bin/python tests/test_flash_api.py
cpu_attention 0.1.0: all checks passed
$ CPU_ATTENTION_BACKEND=cpp .venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900
stray-read probe: 24/24 clean
fuzz dense: 80/80 cases within tolerance, worst was 0.52x its tolerance
$ CPU_ATTENTION_BACKEND=cpp .venv/bin/python -c "import cpu_attention._kernel as K;                                 K.set_backend('cpp'); print(K._entry('isa')())"
amx
```

`tests/test_amx.py` is Mojo-only (`K.amx_available()`, `K._AMX_PLAN`) and prints
"AMX kernel unavailable here: skipped" against the cpp backend, so the same shape
grid was run as a standalone sweep (64 cases: the 12 `test_amx` shapes plus GQA,
`seq_q != seq_k`, odd lengths, 32/64/128 head dims, both dtypes, causal and not):

```
worst bfloat16 causal=0: 3.91e-03      worst float32 causal=0: 1.01e-05
worst bfloat16 causal=1: 1.56e-02      worst float32 causal=1: 3.00e-05
cases: 64 failures: 0
```

fp32 against the **naive fp32 reference** (`audit_common.ref_attention`, fp32
blockwise, no SDPA), 22 rows, both masks:

```
fp32 worst vs naive fp32 reference: 2.146e-05 (gate 2e-4)     [22 rows, 2.1e-6 .. 2.1e-5]
```

That is 3x better than the Mojo tree's 3.1-7.7e-5: the softmax keeps the exact
max (no slack-induced extra rescales) for fp32, so P is the only bf16 rounding.

## 3. Performance — one process, rotated, min of 41, duplicate control arm

`ours/baseline`, `< 1` is faster. `amx` = the shipped dispatch on this box,
`fuse` = the same kernel with `CPU_ATTENTION_FUSE=1` (see §5 — the shipped fuse
rule rejects everything on this box because of a shared-file bug), `vec` = the
AVX-512 policy in the same build, `mojo` = the Mojo AMX kernel in the same
process, `ctrl` = a duplicate `amx` arm (noise floor), `vllm_amx`/`vllm_vec` =
vLLM's own AMX / VEC builds. Ours includes the K/V pack and the output
allocation; the vLLM arms are handed a pre-staged cache and a preallocated out.
Raw JSON: `docs/amx/matrix_fuse_41r.json`.

| row | dt | amx | fuse | vec | mojo | ctrl | vllm_amx | vllm_vec | fuse/vec | fuse/mojo | fuse/vllm_amx | fuse/vllm_vec | amx/ctrl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 27.4 | 27.6 | 56.7 | 25.2 | 26.9 | 28.0 | 53.8 | 0.485 | 1.093 | 0.986 | 0.512 | 1.021 |
| 128:8:64 | fp32 | 46.8 | 46.6 | 53.2 | 45.2 | 46.0 | – | 56.4 | 0.875 | 1.032 | – | 0.826 | 1.017 |
| 512:8:64 | bf16 | 250.1 | 193.6 | 733.8 | 193.2 | 222.2 | 220.1 | 596.1 | 0.264 | 1.002 | 0.880 | 0.325 | 1.126 |
| 512:8:64 | fp32 | 477.5 | 403.9 | 741.8 | 355.2 | 457.6 | – | 566.7 | 0.544 | 1.137 | – | 0.713 | 1.044 |
| 1024:8:64 | bf16 | 692.3 | 616.0 | 2679.8 | 622.8 | 678.4 | 717.9 | 2207.2 | 0.230 | 0.989 | 0.858 | 0.279 | 1.021 |
| 1024:8:64 | fp32 | 1408.2 | 1296.6 | 2384.5 | 1166.0 | 1436.2 | – | 2073.6 | 0.544 | 1.112 | – | 0.625 | 0.980 |
| 2048:8:64 | bf16 | 2311.9 | 2179.8 | 9908.5 | 2190.1 | 2287.1 | 2815.7 | 8430.6 | 0.220 | 0.995 | 0.774 | 0.259 | 1.011 |
| 2048:8:64 | fp32 | 4680.8 | 4314.2 | 8469.4 | 4151.4 | 4570.5 | – | 7658.8 | 0.509 | 1.039 | – | 0.563 | 1.024 |
| 4096:8:64 | bf16 | 8312.5 | 8030.9 | 38354.4 | 8178.4 | 8252.9 | 10865.7 | 33362.6 | 0.209 | 0.982 | 0.739 | 0.241 | 1.007 |
| 4096:8:64 | fp32 | 17872.4 | 17430.4 | 34184.0 | 16649.8 | 17737.4 | – | 30848.4 | 0.510 | 1.047 | – | 0.565 | 1.008 |
| 256:16:128 | bf16 | 306.5 | 203.0 | 839.5 | 203.5 | 280.8 | 204.1 | 642.3 | 0.242 | 0.998 | 0.995 | 0.316 | 1.091 |
| 256:16:128 | fp32 | 606.9 | 414.0 | 914.1 | 439.8 | 602.8 | – | 673.9 | 0.453 | 0.941 | – | 0.614 | 1.007 |
| 1024:16:128 | bf16 | 2037.9 | 1741.8 | 9907.8 | 1854.5 | 2082.1 | 2041.1 | 8198.3 | 0.176 | 0.939 | 0.853 | 0.212 | 0.979 |
| 1024:16:128 | fp32 | 4734.7 | 3820.6 | 8914.8 | 3832.7 | 4746.2 | – | 7824.5 | 0.429 | 0.997 | – | 0.488 | 0.998 |
| 2048:16:128 | bf16 | 6692.5 | 5908.4 | 37381.9 | 6239.8 | 6633.4 | 7113.5 | 33228.4 | 0.158 | 0.947 | 0.831 | 0.178 | 1.009 |
| 2048:16:128 | fp32 | 16484.9 | 14011.8 | 33023.3 | 14554.9 | 15863.1 | – | 28324.2 | 0.424 | 0.963 | – | 0.495 | 1.039 |
| 4096:16:128 | bf16 | 24851.0 | 23337.1 | 148687.6 | 24289.5 | 24167.1 | 27771.6 | 122810.5 | 0.157 | 0.961 | 0.840 | 0.190 | 1.028 |
| 4096:16:128 | fp32 | 75378.3 | 75458.9 | 137215.4 | 76035.9 | 72957.4 | – | 106269.0 | 0.550 | 0.992 | – | 0.710 | 1.033 |
| 512:32:128 | bf16 | 1467.4 | 1131.9 | 5616.4 | 1172.8 | 1432.4 | 1206.1 | 4458.5 | 0.202 | 0.965 | 0.938 | 0.254 | 1.024 |
| 512:32:128 | fp32 | 3237.2 | 2346.6 | 5177.4 | 2332.0 | 3194.9 | – | 4249.4 | 0.453 | 1.006 | – | 0.552 | 1.013 |
| 1024:8:32 | bf16 | 551.9 | 514.7 | 1516.7 | 509.6 | 524.2 | 614.0 | 1292.3 | 0.339 | 1.010 | 0.838 | 0.398 | 1.053 |
| 1024:8:32 | fp32 | 967.6 | 900.8 | 1398.1 | 796.2 | 972.4 | – | 1186.4 | 0.644 | 1.131 | – | 0.759 | 0.995 |

Summary (min / max of the ratios, wins = ratio < 1):

| | band | rows |
|---|---|---|
| bf16 fuse/vec (**the AMX win**) | **0.157-0.485 (2.1-6.4x)** | 11/11 |
| fp32 fuse/vec | **0.424-0.875 (1.14-2.36x)** | 11/11 |
| bf16 fuse/mojo | 0.939-1.093, median 0.989 | 8/11 |
| fp32 fuse/mojo | 0.941-1.137, median 1.032 | 4/11 |
| bf16 fuse/vllm_amx | **0.739-0.995** | 11/11 |
| fp32 fuse/vllm_vec | 0.488-0.826 | 11/11 |
| bf16 amx/mojo (shipped fuse rule) | 1.016-1.506 | 0/11 |
| control (duplicate arm) | 0.974-1.141 | – |

Read this with the control: the noise floor on this box is ±7-14% per round-min
(`ctrl` duplicates of the same arm read 0.974-1.141 in one process), so
"bf16 = parity with Mojo, fp32 = 3% behind, both within the floor, and 11/11
against vLLM's own AMX build" is the honest claim. The 10-40% earlier numbers in
`docs/CPP_REPORT.md` for the vec path are the ones AMX replaces.

## 4. Where the time goes (isolated per-phase, one thread, 32 rows)

`phase_bench` (packed slab built once, each phase timed alone), ns per logit
(= one (row, key) pair), and the same as a fraction of the 2.98 TFLOP/s/core
`tdpbf16ps` roof:

| dtype | d | nk | Q@K^T | softmax | P@V |
|---|---|---|---|---|---|
| bf16 | 64 | 256 | 0.09 (1365 GF/s) | 0.188 | 0.03-0.07 |
| bf16 | 128 | 256 | 0.12-0.16 (1.6-2.0 TF/s) | 0.156-0.188 | 0.12-0.14 |
| fp32 | 64 | 256 | 0.19 (2.0 TF/s) | 0.281 | 0.19 |
| fp32 | 128 | 256 | 0.34 (2.2 TF/s) | 0.250-0.305 | 0.38 (2.0 TF/s) |

So the two matmul phases run at 55-87% of the `tdpbf16ps` roof and the softmax is
the largest single phase at D=64 (46% of the per-logit cost). Whole-call
attribution (phase-ablation bits, `CPU_ATTENTION_AMX_MODE`, 1024:8:64 bf16, 763 us
before the last three fixes): softmax 45%, Q@K^T 23%, P@V 21%, pack 5%, driver +
python + region 8%.

The **pack is memory-bound, not shuffle-bound**: 21 GB/s on one thread, 60-111
GB/s on four, against the 62-70 GB/s DRAM ceiling `docs/AUDIT.md` measures for
this box. It is 2-11% of a call and has no headroom.

## 5. Required shared-file change (not applied — I own one file)

### 5a. `utils.hpp`: the L3 probe parses "16384K\n" as 16384 bytes

```diff
--- a/cpu_attention/csrc/utils.hpp
+++ b/cpu_attention/csrc/utils.hpp
@@ -..., size_t n = std::strlen(buf);
     size_t n = std::strlen(buf);
+    // sysfs always ends the line with '\n', so the K/M suffix test below never
+    // fired and the LLC was read as a few thousand bytes.
+    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
     if (n == 0) continue;
     if (buf[n - 1] == 'K') {
       mult = 1024;
     } else if (buf[n - 1] == 'M') {
       mult = 1024 * 1024;
     }
```

Effect: `last_level_cache_bytes()` returns 16384 (bytes) instead of 16 MB here,
so `dense_run`'s fuse rule (`per_thread_set <= 0.6 * LLC`) rejects the per-head
pack-then-attend order for **every** shape and always takes the split path
(pack every head, barrier, dynamic dispatch). Forcing the fused order is worth:

| row | split (`amx`) | fused | speedup |
|---|---:|---:|---:|
| 256:16:128 bf16 | 306.5 | 203.0 | **1.51x** |
| 256:16:128 fp32 | 606.9 | 414.0 | **1.47x** |
| 512:32:128 fp32 | 3237.2 | 2346.6 | **1.38x** |
| 512:32:128 bf16 | 1467.4 | 1131.9 | 1.30x |
| 512:8:64 bf16 | 250.1 | 193.6 | 1.29x |
| 1024:16:128 bf16 | 2037.9 | 1741.8 | 1.17x |
| 2048:16:128 fp32 | 16484.9 | 14011.8 | 1.18x |
| 4096:16:128 bf16 | 24851.0 | 23337.1 | 1.06x |
| 4096:16:128 fp32 | 75378.3 | 75458.9 | 1.00x |

Reproduced twice per row, interleaved (`192/192`, `280/281` us at
256:16:128 bf16). The fused path is only legal for MHA (`group == 1`), which the
rule already enforces: with `CPU_ATTENTION_FUSE=1` forced on GQA shapes, 12/65
`test_dense` checks fail because the fused loop attends only the first query head
of each KV group — that is why the gate exists, and it is untouched by this fix.
All MHA shapes pass with the fused order (65/65 MHA-only, 64-case sweep clean).

### 5b. Optional: the driver's per-item `memset` of the accumulator

`dense_run` memsets `arows * hd` floats per work item. This kernel zeroes its
16x16 C tiles on the first key tile of every item (`k0 == 0`), so that memset is
dead work for it (16 KB per item at head_dim 128; the whole residual with all
four phases skipped is 57 us of a 203 us call at 256:16:128 bf16, and the memset
is the largest piece). Suggested shape, policy-owned:

```diff
+  // in the policy
+  static constexpr bool zeroes_acc = true;   // AMX: tile_zero on k0 == 0
-      std::memset(acc, 0, sizeof(float) * static_cast<size_t>(arows) * hd);
+      if constexpr (!P::zeroes_acc)
+        std::memset(acc, 0, sizeof(float) * static_cast<size_t>(arows) * hd);
```

`static constexpr bool zeroes_acc = false;` then has to be added to the vec,
scalar and neon policies. Not measured (it needs the shared file), so treat the
size as an upper bound: ~5-8% of the short rows, ~1% of the long ones.

## 6. Falsification ledger (number that killed each idea)

| idea | result |
|---|---|
| transpose as a standalone function (arrays `t[16]`+`u[16]` live) | GCC emits it out of line and round-trips 1 KB through the stack per tile; `always_inline` + a 3-stage in-place form removes it, worth **0-3%** of the call (the pack is DRAM-bound, so the spilling was hidden) |
| store the scaled logits in the max pass (the vec policy's shape) | bf16 softmax 0.219 -> 0.219 ns/logit (**no change**), fp32 0.312 -> 0.281 (**-10%**). Kept, but not for the reason it was done in `cpu_attn_vec.hpp` |
| degree-2 exponential for bf16 (vs degree-5) | softmax 0.219 -> **0.188 ns/logit** (-14%), full call -1 to -5%; bf16 `max|diff|` unchanged at 1.56e-2 (the P rounding dominates), fuzz still 0.52x tolerance |
| FA4 running-max slack of 8 log2 units (vs the exact max) | **1.04-1.12x** at 256:16:128 (bf16 333 -> 292 us, fp32 691 -> 609), 1.02-1.04x at 1024:*, 1.01-1.02x at 4096:*. It removes the driver's scalar 32*head_dim rescale plus `l`'s multiply on most super-tiles. Error unchanged (P is relative-precision bf16, up to 2^8 = 256) |
| `a = 0` on the first tile instead of `a = 1` | the driver's rescale loop then ran a full 16 KB scalar multiply over an accumulator that was zero anyway: 1.10x at 256:16:128 |
| `kt` = 128 / 512 instead of 256 | 128: 1024:8:64 bf16 787 vs 778, 1024:16:128 bf16 2464 vs 2347 (**-5%**); 512: 1024:8:64 bf16 903 vs 778 (**-16%**), 4096:16:128 fp32 76570 vs 80449 (+5%). 256 stays |
| per-head pack-then-attend (the fused order) | **1.0-1.5x** — but it is the *shared* fuse rule that rejects it (§5a), not a kernel choice |
| `always_inline` on every helper | no effect beyond the transpose; the qk/pv inner loops were already clean (verified in the `.s`: 4 `tileloadd` + 4 `tdpbf16ps` per dim chunk, no stack traffic) |
| a cheaper `tile_stream_loadd` for K/V | not tried past the codegen check — the tiles are re-read per query block, so an L1-bypassing load hint is the wrong direction |
| 16-row work items (1 AMX row group) | not built: the driver's `RQT` is 32 (vLLM's `MaxQHeadNumPerIteration`), and 16 rows would double the number of key sweeps for the same flops |

What is left, in order: (1) the LLC fix above, which is the whole 1.0-1.5x on the
short rows; (2) the accumulator memset (§5b); (3) the softmax, which is 45% of a
D=64 call at 0.188 ns/logit and instruction-bound at ~16 vector ops per 16
logits — the only remaining lever there is to stop materialising bf16 P in
memory and feed the P@V A tiles straight from registers, which AMX cannot do
(it needs a memory operand).

## 7. Instruments

* `docs/amx/matrix_fuse_41r.json` — the 7-arm matrix above (raw).
* Phase ablation: `CPU_ATTENTION_AMX_MODE` bits 1/2/4/8 skip pack / Q@K^T /
  softmax / P@V (results are garbage; timings only).
* `phase_bench.cpp`, `pack_bench.cpp`, `amx_layout_test.cpp`,
  `amx_e2e_test.cpp` (scratch, in the session's `/tmp`): per-phase ns/logit,
  pack GB/s, pack+Q@K^T against a scalar reference, and the whole
  `attend()` loop per stage — the e2e probe is what localised both correctness
  bugs (the K tile of key block `kb+1` is `DC32` slots on, not one; and the
  alpha rescale must not re-apply `c`, which is already folded into `m`).
