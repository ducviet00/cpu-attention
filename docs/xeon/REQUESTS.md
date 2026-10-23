# Shared-file requests from the x86/AMX agent (branch `cpp-xeon-amx`)

I own `cpu_attention/csrc/cpu_attn_amx.hpp` only. Everything below lives in a
shared file, with the exact diff and the measurement that justifies it.

Both items are in the **dense driver's per-work-item path**
(`cpu_attention/csrc/cpu_attn_impl.hpp`, `dense_run`'s `attend` lambda), which
runs once per (batch, q head, `rq`-row block). `rq` is 64 now, `RQT` is 32.

---

## 1. BUG in the `zeroes_acc` hook as shipped: a fully masked work item reads a stale accumulator

You have already applied the `zeroes_acc` hook (`cpu_attn_impl.hpp:114-122`,
`188`). It is correct for the AMX policy's *coverage* — `pv` writes every
accumulator row whenever it runs at all — but it is **wrong for a work item that
never runs `pv`**, and such items are reachable on every causal call with
`seq_q > seq_k`:

```
empty = job.causal && (row_limit0 + nrows - 1) < 0
```

For an empty item no `qk`/`softmax`/`pv` runs, so the accumulator keeps whatever
the *previous* work item on that thread left in the same workspace: possibly
`inf`/`NaN`. The epilogue then computes

```cpp
const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
o[d] = src[d] * inv;          // NaN * 0 == NaN, not 0
```

and the row that should be all zeros comes out all NaN.

**Repro, deterministic on this box** (`CPU_ATTENTION_BACKEND=cpp`):

```
tests/fuzz_dense.py --cases 80 --max-seq 900
  with zeroes_acc = true  : 77/80, case 46 q=(1,129,4,128) k=(1,43,2,128)
                                   case 53 q=(1,129,8,64)  k=(1,43,1,64)
                                   case 55 q=(2,65,2,64)   k=(2,21,1,64)
                            all NaN, all causal, all seq_q > seq_k
  with zeroes_acc = false : 80/80   (4 consecutive runs)
```

`cuda_attention/csrc`'s `with_kv_cache` path cannot hit it (one query token, the
cache is never empty), which is why nothing has failed before: the benchmark
matrix is all `seq_q == seq_k`.

**Fix** — zero the row instead of scaling it, which makes the hook safe for every
policy:

```diff
--- a/cpu_attention/csrc/cpu_attn_impl.hpp
+++ b/cpu_attention/csrc/cpu_attn_impl.hpp
@@ -213,6 +213,18 @@
       for (int r = 0; r < nrows; ++r) {
         char* op = static_cast<char*>(job.out) +
                    (b * job.ob + (q0 + r) * job.ot + h * job.oh) * esz;
+        // A row whose softmax saw no key at all is exactly zero -- and with
+        // `zeroes_acc` the accumulator may not have been written at all, so it
+        // must not be multiplied by 0 (NaN * 0 is NaN).
+        if (l[r] <= 0.0f) {
+          if (job.dtype == kF32) {
+            std::memset(op, 0, sizeof(float) * hd);
+          } else {
+            std::memset(op, 0, sizeof(bf16) * hd);
+          }
+          continue;
+        }
         const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
         const float* src = acc + r * hd;
         if (job.dtype == kF32) {
```

With that in place I will set `static constexpr bool zeroes_acc = true;` in the
AMX policy again; **today it is `false`** on my branch, because a shared file I
do not own must not be left with a gate that is red.

**Size**: measured with `docs/xeon/probe_epilogue.cpp`, the memset is 1.07 ns/row
at head_dim 64 and 2.05 at 128, i.e. 34-66 ns per work item (8-16 KB) — 0.3-1.3%
of a call. Take it as hygiene, not as a win.

---

## 2. The output epilogue is a scalar loop; 2.5x of it is free

(Independent of item 1; the two touch the same loop and can land together.)

```diff
--- a/cpu_attention/csrc/cpu_attn_impl.hpp
+++ b/cpu_attention/csrc/cpu_attn_impl.hpp
@@ -213,13 +213,27 @@
       for (int r = 0; r < nrows; ++r) {
         char* op = static_cast<char*>(job.out) +
                    (b * job.ob + (q0 + r) * job.ot + h * job.oh) * esz;
         const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
         const float* src = acc + r * hd;
         if (job.dtype == kF32) {
           float* o = reinterpret_cast<float*>(op);
           for (int d = 0; d < hd; ++d) o[d] = src[d] * inv;
         } else {
           bf16* o = reinterpret_cast<bf16*>(op);
-          for (int d = 0; d < hd; ++d) o[d] = f32_to_bf16(src[d] * inv);
+          int d = 0;
+#if defined(__AVX512BF16__)
+          const __m512 vinv = _mm512_set1_ps(inv);
+          for (; d + 16 <= hd; d += 16) {
+            const __m256bh h = _mm512_cvtneps_pbh(
+                _mm512_mul_ps(_mm512_loadu_ps(src + d), vinv));
+            __m256i raw;
+            std::memcpy(&raw, &h, sizeof(raw));
+            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + d), raw);
+          }
+#endif
+          for (; d < hd; ++d) o[d] = f32_to_bf16(src[d] * inv);
         }
       }
```

`f32_to_bf16` carries a NaN branch, so GCC does not vectorise the loop; the
16-wide `vcvtneps2bf16` form is bit-identical (both round to nearest even) and
measurably cheaper.

**Measured** (`docs/xeon/probe_epilogue.cpp`, min of 7, one thread, exact copy of
the driver's loop against the vector form):

| head_dim | scalar | vector | elements in a 128:8:64 call | scalar / 4 threads | vector |
|---|---|---|---|---|---|
| 64 | 0.10 ns/elem | 0.04 | 65 536 | 1.6 us (6% of 27 us) | 0.65 us |
| 128 | 0.11 ns/elem | 0.05 | 8 388 608 (4096:16:128) | 210 us (0.9% of 22.5 ms) | 95 us |

So ~1 us at the short rows and ~0.5% of the long ones. Together with item 1 the
short-row saving is ~4% of the call, which is the size of the whole gap at
`128:8:64` (see `docs/xeon/RESULTS.md`).

---

## 3. Not requested, just reported: two things I found in the shared code

* `last_level_cache_bytes()` reads `index3` (L3) and the fuse rule accepts the
  per-head pack-then-attend order on this box (260 MB L3). Confirmed 1.05-1.12x
  better than the split order here (`CPU_ATTENTION_FUSE=0` A/B).
* `AROWS` in the AMX policy bounds `CPU_ATTENTION_RQ`; the old code let the env
  knob go to 512 against a 64-row accumulator. I clamp it now.
