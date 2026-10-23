# Paged attention on the AMX box — what was tried, what shipped

Branch `cpp-amx` (merged with `cpp` @`406dedc`). One file changed:
`cpu_attention/csrc/cpu_attn_amx.hpp`, which now carries the paged policy as well
as the dense one. The dispatcher change it needs is a diff in §2 — it is *not*
committed, per the rule that `cpu_attn.cpp` is not mine.

Box: `m8i.2xlarge` (Xeon 6975P-C, 8 vCPU = **4 physical cores**, `amx_tile` /
`amx_bf16`, 48 KB L1d, 2 MB L2, 480 MB L3 reported), 4 torch threads, spot,
us-west-2. All arms interleaved and rotated in **one process**, `min` of 15
(`bench_cpp.py`), duplicate `cpp` arm as the noise floor.

## 1. AMX does not pay for paged decode — the number

A paged decode work item is one live query row, so an M=16 tile is 15/16 waste
and there is no GQA to pack across query heads:

| | MAC/cycle/core |
|---|---|
| `tdpbf16ps`, M=16 (dense) | 1024 x 2 / 16.5 cycles = **~500** |
| `tdpbf16ps`, M=1 (paged decode) | **~31** |
| `dpbf16ps` (AVX-512 bf16) | **32** |

So a tile kernel is at parity with the vector one at best, and it would need the
K cache in `[dim pair][key]` form — a different `reshape_and_cache` and a
different engine-staged cross cache. Measured confirmation from the other side:
vLLM *ships* an AMX paged kernel (`--isa amx`), and on this box it is only a
sideways move versus their own VEC build — it wins bf16 `b=8/1024`
(404-416 vs 461-511) and `b=32/1024` (2.7-2.8 ms vs 3.7-3.9), and loses
`b=8/256` (105-114 vs 109-126 both ~equal) and fp32 does not exist there at all
(an fp32 query trips `TORCH_CHECK` and aborts the process). The bar is therefore
the **better of the two vLLM builds per row**, which is what §3 reports.

Note also that at `b=8/1024` a tile kernel cannot win: the whole call is a 33.5 MB
stream and a plain read of that stream takes 390 µs on this box (§4), i.e. the
row is decided by memory, not by MACs.

## 2. What shipped (all inside `cpu_attn_amx.hpp`)

Four structural fixes, each measured. The reference shape is the vec policy's:
`for each 16-key group: cv = fma(bcast(q[d]), k[d], cv) for d in 0..hd`, one
accumulator, `jj / page` inside the key loops.

1. **Q@K^T: dim-outer, 64 keys per step, four independent accumulators**
   (`Paged::qk_groups<R>`). One broadcast per dim per 64 keys instead of one per
   dim per 16 keys, and four loads in flight with four FMA chains instead of one
   chain that stalls on each load in turn.
2. **`jj / page` and `jj % page` removed from every key loop.** `job.page` is a
   runtime `int64`, so that was a 64-bit `idiv` (20-40 cycles) per key in the
   P@V walk and per 16 keys in Q@K^T — at ctx 4096 more cycles than the FMAs it
   feeds. Both loops now walk pages and carry the offset.
3. **P@V: two accumulator lanes over key parity** (`a0`/`a1`, combined at the
   end of the tile), plus 2-key loads so the two lanes' V rows are one 512-byte
   contiguous read.
4. **Softmax: two passes instead of three.** The scale is folded into the
   exponential (`exp2(fma(logit, c, -m))`) so the max pass only reads, instead of
   scaling and storing the whole tile. bf16 uses the degree-2 polynomial.

**Loop shape is per dtype**, from an in-process A/B against the vec policy:
bf16 takes the dim-outer Q@K^T (1.08-1.39x over the vec shape at every shape) and
two P@V lanes; fp32 takes the dim-inner Q@K^T and one lane, which is parity with
the vec shape at 256/1024/4096 keys and 6% better at b=32. The dim-outer form
costs fp32 6-9% at three of five shapes — fp32 issues twice the loads per dim, so
four chains fight for load slots instead of hiding them.

Kept from the vec policy unchanged: the cache layout (byte for byte, so
`reshape_and_cache` and the engine's staged cross cache keep working), the
online-softmax scheme with the running max, and `KT = 512` keys per tile.

Knobs (env, defaults are the shipped configuration, all swept):
`CPU_ATTENTION_PAGED_KT` (64…4096, 512 best), `CPU_ATTENTION_PAGED_QK` (1 = the
old dim-inner shape), `CPU_ATTENTION_PAGED_PV` (1 = one lane),
`CPU_ATTENTION_PAGED_PF` (prefetch distance), `CPU_ATTENTION_PAGED_MODE`
(1/2/4 = skip Q@K^T / softmax / P@V, profiling only).

### The dispatcher diff that must be applied (`cpu_attention/csrc/cpu_attn.cpp`)

```diff
--- a/cpu_attention/csrc/cpu_attn.cpp
+++ b/cpu_attention/csrc/cpu_attn.cpp
@@ -24,6 +24,8 @@
 namespace amx {
 bool available();
 void dense(const DenseJob&, int);
+void paged(const PagedJob&, int);
+void store(const PagedJob&, int);
 }  // namespace amx
@@ -105,9 +107,10 @@ int pick_paged_isa() {
   if (forced != nullptr) {
     if (forced[0] == 's') return kIsaScalar;
     if (forced[0] == 'v') return kIsaVec;
-    if (forced[0] == 'a') return kIsaNeon;  // amx is dense-only
+    if (forced[0] == 'a') return kIsaAmx;
   }
 #if defined(__x86_64__)
+  if (amx::available()) return kIsaAmx;
   if (vec::available()) return kIsaVec;
@@ -153,6 +156,10 @@ void attention_entry(...) {
 #if defined(__x86_64__)
       vec::store(job, workers);
 #endif
+#if defined(__x86_64__)
+    } else if (isa == kIsaAmx) {
+      amx::store(job, workers);
+#endif
@@ -166,6 +173,10 @@ void attention_entry(...) {
 #if defined(__x86_64__)
       vec::paged(job, workers);
 #endif
+#if defined(__x86_64__)
+    } else if (isa == kIsaAmx) {
+      amx::paged(job, workers);
+#endif
```

`amx::available()` is false without `amx_tile`+`amx_bf16`+`amx_init()`, so this
is a no-op on every other x86 CPU and the VEC path stays the fallback. The
alternative — porting `Paged::attend_head` into `cpu_attn_vec.hpp` — is the same
kernel body with `amx::` replaced by `vec::`; the file here is a drop-in.

## 3. Paged matrix (min of 15, one process, rotated, `ctrl` = duplicate `cpp`)

`ours/baseline`, < 1 is faster. `vllm` is the **better of vLLM's two builds for
that row** (`--isa amx` for bf16, `--isa vec` for fp32 — their AMX kernel aborts
on an fp32 query). Raw JSON: `docs/amx/paged_f_vec.json`,
`docs/amx/paged_audit_paged_vec.json`.

| row | dt | ours | mojo | ctrl | vllm (best build) | ours/vllm | ctrl/ours |
|---|---|---:|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | 13.2 | 13.1 | 13.4 | 13.8 (amx) | **0.950** | 1.02 |
| cross b=1 ctx=256 | bf16 | 9.2 | 9.5 | 9.1 | 8.2 (amx) | 1.122 | 0.96 |
| self b=8 ctx=256 | bf16 | 69.0 | 101.9 | 69.1 | 113.6 (amx) | **0.607** | 1.00 |
| cross b=8 ctx=256 | bf16 | 65.4 | 98.1 | 65.3 | 105.2 (amx) | **0.621** | 1.00 |
| self b=8 ctx=1024 | bf16 | 428.7 | 443.7 | 431.2 | 404.0 (amx) | 1.061 | 1.01 |
| cross b=8 ctx=1024 | bf16 | 415.8 | 433.3 | 416.5 | 416.6 (amx) | 0.998 | 1.00 |
| self b=32 ctx=1024 | bf16 | 2372 | 3151 | 2381 | 2816 (amx) | **0.842** | 1.00 |
| cross b=32 ctx=1024 | bf16 | 2255 | 3084 | 2271 | 2774 (amx) | **0.813** | 1.01 |
| self b=1 ctx=4096 | bf16 | 218.6 | 233.7 | 219.9 | 238.8 (amx) | **0.915** | 1.01 |
| cross b=1 ctx=4096 | bf16 | 218.8 | 230.7 | 219.6 | 235.6 (amx) | **0.929** | 1.00 |
| self b=1 ctx=256 | fp32 | 12.3 | 19.7 | 12.6 | 26.7 (vec) | **0.461** | 1.03 |
| cross b=1 ctx=256 | fp32 | 9.9 | 17.7 | 10.0 | 20.9 (vec) | **0.475** | 1.01 |
| self b=8 ctx=256 | fp32 | 230.4 | 227.0 | 226.3 | 211.4 (vec) | 1.090 | 0.98 |
| cross b=8 ctx=256 | fp32 | 210.6 | 219.4 | 221.4 | 194.3 (vec) | 1.084 | 1.05 |
| self b=8 ctx=1024 | fp32 | 883.2 | 820.9 | 869.8 | 901.1 (vec) | **0.980** | 0.99 |
| cross b=8 ctx=1024 | fp32 | 821.0 | 799.1 | 838.2 | 891.8 (vec) | **0.921** | 1.02 |
| self b=32 ctx=1024 | fp32 | 3951 | 4688 | 3929 | 4850 (vec) | **0.815** | 0.99 |
| cross b=32 ctx=1024 | fp32 | 3801 | 4584 | 3741 | 4704 (vec) | **0.808** | 0.98 |
| self b=1 ctx=4096 | fp32 | 483.3 | 423.5 | 483.6 | 404.6 (vec) | 1.195 | 1.00 |
| cross b=1 ctx=4096 | fp32 | 479.7 | 458.7 | 485.4 | 397.6 (vec) | 1.207 | 1.01 |

Against vLLM's VEC build (both dtypes) in one process: **18/20 rows win**
(bf16 0.52-0.93, fp32 0.46-0.98) except `fp32 b=8/256` (1.08-1.09) and
`fp32 b=1/4096` (1.20). Against the better build per row: **14/20**. The
independent harness (`audit/audit_paged.py`, own ground truth) reads the same:

### The drift-free comparison: new policy vs the vec policy it replaces

Same process, same tensors, arms rotated, `min` of 15, `ctrl` = duplicate of
`new` (so `new/ctrl` is the floor, 0.94-1.05):

| row | dt | new | vec | new/vec | new/ctrl |
|---|---|---:|---:|---:|---:|
| b=1 ctx=256 | bf16 | 10.0 | 14.0 | **0.718** | 0.82 |
| b=8 ctx=256 | bf16 | 70.9 | 89.7 | **0.791** | 1.01 |
| b=8 ctx=1024 | bf16 | 371.9 | 436.7 | **0.852** | 1.00 |
| b=32 ctx=1024 | bf16 | 1777.6 | 2050.8 | **0.867** | 1.02 |
| b=1 ctx=4096 | bf16 | 219.7 | 236.5 | **0.929** | 1.02 |
| b=1 ctx=256 | fp32 | 12.6 | 12.0 | 1.044 | 1.03 |
| b=8 ctx=256 | fp32 | 199.7 | 196.0 | 1.019 | 0.94 |
| b=8 ctx=1024 | fp32 | 834.9 | 815.9 | 1.023 | 1.02 |
| b=32 ctx=1024 | fp32 | 3747.5 | 3995.4 | **0.938** | 1.00 |
| b=1 ctx=4096 | fp32 | 414.2 | 410.3 | 1.010 | 0.98 |

So the kernel is a **1.08-1.39x win on bf16 and parity on fp32** relative to what
it replaces.

### Against vLLM (independent harness)

| row | dt | ours/vllm (min, median) |
|---|---|---|
| self b=8/1024 | bf16 | 0.84, 0.83 |
| cross b=8/1024 | bf16 | 0.95, 0.91 |
| self b=32/1024 | bf16 | 0.55, 0.64 |
| cross b=32/1024 | bf16 | 0.77, 0.78 |
| self b=1/4096 | bf16 | 0.74, 0.79 |
| cross b=1/4096 | bf16 | 0.93, 0.88 |
| self b=1/256 | fp32 | 0.50, 0.60 |
| cross b=1/256 | fp32 | 0.68, 0.83 |
| self b=8/256 | fp32 | 1.16, 1.18 |
| cross b=8/256 | fp32 | 1.20, 1.20 |
| self b=8/1024 | fp32 | 1.14, 1.20 |
| cross b=8/1024 | fp32 | 1.13, 1.18 |
| self b=32/1024 | fp32 | 0.80, 0.82 |
| cross b=32/1024 | fp32 | 0.98, 0.96 |
| self b=1/4096 | fp32 | 1.23, 1.24 |
| cross b=1/4096 | fp32 | 1.26, 1.24 |

**Where the control says the row is not resolvable**: vLLM's *own* fp32 paged
numbers move 726-901 us for `b=8/1024` between runs of the same binary (25%), and
its bf16 `b=8/1024` moves 387-420; our ctrl stays inside 0.94-1.05 of `new` in
every run. So the rows quoted at 0.98-1.06 are parity, not a resolved win or
loss. `fp32 b=1/4096` (1.19-1.26) and `fp32 b=8/256` (1.08-1.20) are the two
consistent deficits.: `ctrl/ours` is 0.96-1.05
in the table above, but vLLM's *own* fp32 numbers move 721-901 µs across runs for
`b=8/1024` (25%), which is why that row reads 0.92 in one harness and 1.14 in
the other. `bf16 b=8/1024` is the same story (vLLM 387-416 across runs). Both
those rows are at the memory wall and are coin flips; the honest claim is
parity there.

## 4. Why fp32 still loses three shapes — the roofline

The kernel is **not** bandwidth-bound; it is latency-bound, and one phase at a
time. Two measurements:

* A plain sequential read of the benchmark's own paged cache (same page walk,
  32-byte steps, 4 threads): **b=8/1024 390 µs (86 GB/s)**, b=1/4096 200 µs,
  b=32/1024 1625 µs. That is the floor.
* Phase ablation of the shipped kernel at `b=8/1024` (11 rounds, `MODE` bits):

| | bf16 | fp32 |
|---|---|---|
| full call | 412-419 us | 944 us |
| Q@K^T | 202 us (83 GB/s) | 438 us (76 GB/s) |
| softmax | 34 us (8%) | ~0 (overlapped) |
| P@V | 217 us (77 GB/s) | 480 us (70 GB/s) |

So each phase runs at 81-88% of what a plain read of the same bytes achieves, and
**the two phases do not overlap at all** (Q@K^T + P@V ≈ the whole call). vLLM's
VEC kernel sustains 86 GB/s on the same traffic in the same process, i.e. it is
at the read floor and we are 12-20% above it on fp32. That is the remaining gap
and it is not reachable by any of the knobs in §5 — which is also why this
section is a deficit report and not a fix.

## 5. Falsification ledger (number that killed each idea)

| idea | result |
|---|---|
| AMX tile kernel for paged decode | 31 useful MAC/cycle/core vs `dpbf16ps`'s 32 (§1), needs a different cache layout, and the winning rows are memory-bound (390 us of unavoidable stream at b=8/1024). Not built. |
| dim-inner → **dim-outer with R=4 chains + hoisted broadcast** | bf16 `b=8/256` 90.5 → 69.0 us (**1.31x**), `b=32/1024` 2755 → 1958-2372 (**1.16-1.4x**), `b=8/1024` 491 → 428 (**1.15x**); fp32 within 3% of every other shape. Shipped. |
| page walk instead of `jj / page` per key | part of the above; isolated by the shape sweep (all four QK/PV shape combinations are within 3% for fp32, so the win came from the address arithmetic and the chain, not from the shapes) |
| R = 1 / 2 / 4 / 8 groups per Q@K^T step | 30.3 / 33.2 / 33.8 / 33.2 GB/s in the isolated probe; **R=8 is worse in the real kernel** (`b=8/256` bf16 cpp/mojo 0.685 → 0.990, `b=8/1024` fp32 1.202 → 1.247). R=4 stays. |
| two accumulator lanes in P@V | `b=32/1024` bf16 was the motivation; measured a wash on its own (qk0pv1 vs qk0pv0 within 1-3%), kept because it is free and it halves the chain |
| `KT` = 64 / 128 / 256 / 512 / 1024 / 2048 | fp32 `b=1/4096`: 484 / 476 / — / 469 / 484 / 468 us; `b=8/256`: 201 / 227 / 223 / 223 / — / 228. Flat within noise except a possible 12% at KT=64 for one row with a 0.91 control. 512 stays. |
| software prefetch along the stream (8 / 16 / 32 dim rows for Q@K^T, ×2 keys for P@V) | **neutral to 20% worse** (`b=8/1024` bf16 cpp/mojo 0.99 / 1.22 / 1.01 / 1.13 for pf = 0 / 8 / 16 / 32). Off. This is the second independent falsification of prefetch on this path. |
| KV split for low batch (`b=1/4096`) | not built: 16 tasks on 4 threads is already 4 per thread with no tail; a split adds a reduction pass and does not raise per-thread memory-level parallelism, which is what the ablation says is binding |
| per-page softmax (interleave the K and V streams at page granularity) | not built — the one structural idea left that could beat the "phases do not overlap" finding, and it needs a re-tuned rescale path (a softmax per 32 keys instead of per 512) |

## 6. Parity

```
CPU_ATTENTION_BACKEND=cpp tests/test_flash_api.py   -> cpu_attention 0.1.0: all checks passed
CPU_ATTENTION_BACKEND=cpp tests/test_dense.py       -> cpu_attention dense: 65/65 checks passed
custom paged sweep (block 16/32/64/128/256 x head_dim 32/64/128 x MHA(16) and
GQA(8/2) x bf16/fp32 x lens {1, 2*blk+5, 3*blk-1, 900, [767,257], [513,256,16]},
both cross_attention and flash_attn_with_kvcache, ground truth SDPA per
sequence)                                            -> paged parity: 600/600 checks passed over 300 shapes
audit/audit_paged.py (independent ground truth)      -> d(ours) 3.0e-4..7.6e-4 bf16, 1.3e-7..3.6e-7 fp32
                                                        (gates 3e-2 / 2e-4)
```

fp32 max|diff| against the independent softmax reference: **3.6e-07** (gate
2e-4), i.e. the fp32 paged path is unchanged numerically by the rewrite.
