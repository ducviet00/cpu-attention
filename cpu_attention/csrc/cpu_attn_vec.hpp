// AVX-512 kernel: fp32 and bf16, dense and paged.
//
// Layouts (this backend defines them; reshape_and_cache writes them):
//   fp32 page/tile:  K [head_dim][tokens]        V [tokens][head_dim]
//   bf16 page/tile:  K [head_dim/2][tokens][2]   V [tokens][head_dim]
// K is dim-major because Q@K^T wants one dimension's tokens contiguous: one
// 64-byte load feeds 16 positions. The bf16 form pairs the two dimensions
// `dpbf16ps` contracts, so one instruction retires 16 positions x 2 dims.
//
// Q@K^T is a register tile of RQT query rows against 16 positions: one load of
// K plus RQT broadcast-FMAs. P@V keeps the whole head-dim accumulator in
// registers (paged: one row, 4-8 zmm) or an 8x32 tile (dense), so both phases
// are FMA-bound rather than load-bound.
//
// Apache-2.0; structure after vLLM's csrc/cpu/cpu_attn_vec.hpp.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "cpu_attn_impl.hpp"
#include "cpu_isa.hpp"

namespace attn {
namespace vec {
namespace {

constexpr int LANES = 16;

inline __m512 ld16(const void* p) { return _mm512_loadu_ps(p); }
inline void st16(void* p, __m512 v) { _mm512_storeu_ps(p, v); }

// bf16 -> fp32, 16 values: the exponent lands in the high half.
inline __m512 widen_bf16(const void* p) {
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

inline __m512 bcast_f32(const void* p) {
  return _mm512_set1_ps(*static_cast<const float*>(p));
}

// One bf16 pair (two adjacent head dims of one query row) broadcast to all 16
// position lanes: the A operand of dpbf16ps.
inline __m512bh bcast_pair(const void* p) {
  const __m512i i = _mm512_set1_epi32(*static_cast<const int*>(p));
  __m512bh r;
  std::memcpy(&r, &i, 64);
  return r;
}

// The B operand: 32 bf16 = 16 positions x 2 adjacent head dims.
inline __m512bh load_pair_bf16(const void* p) {
  const __m512 v = _mm512_loadu_ps(p);
  __m512bh r;
  std::memcpy(&r, &v, 64);
  return r;
}

// Two 16-lane bf16 rows (dim-major within a row) interleaved into the key-pair
// form `dpbf16ps` contracts for P@V: (a0,b0,a1,b1,...,a15,b15), one 64-byte
// store. The two source rows are 32 bytes each, so they are joined into one
// zmm and permuted with a 16-bit index vector.
inline void store_key_pair(char* dst, const void* row_a, const void* row_b) {
  const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_a));
  const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_b));
  __m512i ab = _mm512_castsi256_si512(a);
  ab = _mm512_inserti64x4(ab, b, 1);
  // (a0,b0,a1,b1,...): 0,16,1,17,... as 16-bit lane indices.
  const __m512i idx = _mm512_set_epi16(
      31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22,
      6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(idx, ab));
}

// One 32-bit word (a bf16 pair) broadcast to all 32 half-lanes: the A operand of
// `dpbf16ps` for P@V, where the contraction runs over two adjacent keys.
inline __m512bh bcast_word(const void* p) {
  const __m512i i = _mm512_set1_epi32(*static_cast<const int*>(p));
  __m512bh r;
  std::memcpy(&r, &i, 64);
  return r;
}

// fp32 -> bf16, 16 lanes, as a bit pattern the stores can take.
inline __m256i pack_bf16_16(__m512 v) {
  const __m256bh h = _mm512_cvtneps_pbh(v);
  __m256i r;
  std::memcpy(&r, &h, 32);
  return r;
}

// Two dims, 16 positions each, into the position-major pair layout the dot
// instruction contracts: (d0,t0),(d1,t0),(d0,t1),(d1,t1),...
inline void store_pair_bf16(char* dst, __m512 d0, __m512 d1) {
  const __m256i a = pack_bf16_16(d0);
  const __m256i b = pack_bf16_16(d1);
  const __m256i lo = _mm256_permutex2var_epi16(
      a, _mm256_setr_epi16(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7,
                           23),
      b);
  const __m256i hi = _mm256_permutex2var_epi16(
      a, _mm256_setr_epi16(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                           15, 31),
      b);
  std::memcpy(dst, &lo, 32);
  std::memcpy(dst + 32, &hi, 32);
}

// 2^x for x <= 0: degree-5 polynomial on the fraction, exponent by insertion.
// ~10 instructions; std::exp costs more than twice that and the softmax is
// 15-30% of a call in the Mojo tree.
inline __m512 exp2v(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-127.0f));
  x = _mm512_min_ps(x, _mm512_setzero_ps());
  const __m512 n =
      _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 r = _mm512_sub_ps(x, n);
  __m512 p = _mm512_set1_ps(1.3392112e-3f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(9.6180963e-3f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.5503282e-2f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(2.4022652e-1f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(6.9314718e-1f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  const __m512i e = _mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}


// 2^x for the paged policy's softmax, where the argument is always <= 0 (the
// running max never lags the tile max there, unlike the dense FA4 slack), and
// where bf16 probabilities are rounded to 8 bits two instructions later. The
// degree-2 form is inside that rounding; fp32 keeps the degree-5 `exp2v` above.
template <bool FAST>
inline __m512 exp2v_page(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-127.0f));
  const __m512 n =
      _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 r = _mm512_sub_ps(x, n);
  __m512 p;
  if constexpr (FAST) {
    p = _mm512_fmadd_ps(_mm512_set1_ps(2.3473140e-1f), r,
                        _mm512_set1_ps(6.9335938e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  } else {
    // Taylor to degree 6 on |r| <= 0.5: truncation 1.2e-7, against the degree-5
    // minimax's 3.4e-6, which was the whole of the fp32 paged error budget
    // (a 17-key output read 1.2e-6 against a naive fp32 reference).
    p = _mm512_set1_ps(1.5403530e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3333558e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(9.6181291e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.5504109e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(2.4022651e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(6.9314718e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  }
  const __m512i e = _mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}

// 16x16 fp32 transpose, in place over 16 zmm (one per row).
inline void transpose16x16_f32(__m512* r) {
  __m512 t[16], u[16];
  for (int i = 0; i < 8; ++i) {
    t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  for (int i = 0; i < 4; ++i) {
    u[4 * i + 0] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0x44);
    u[4 * i + 1] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0xEE);
    u[4 * i + 2] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0x44);
    u[4 * i + 3] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0xEE);
  }
  for (int j = 0; j < 4; ++j) {
    const __m512 p0 = _mm512_shuffle_f32x4(u[j], u[4 + j], 0x88);
    const __m512 p1 = _mm512_shuffle_f32x4(u[j], u[4 + j], 0xDD);
    const __m512 q0 = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0x88);
    const __m512 q1 = _mm512_shuffle_f32x4(u[8 + j], u[12 + j], 0xDD);
    r[j + 0] = _mm512_shuffle_f32x4(p0, q0, 0x88);
    r[j + 8] = _mm512_shuffle_f32x4(p0, q0, 0xDD);
    r[j + 4] = _mm512_shuffle_f32x4(p1, q1, 0x88);
    r[j + 12] = _mm512_shuffle_f32x4(p1, q1, 0xDD);
  }
}

// ---------------------------------------------------------------------------
// dense policy
// ---------------------------------------------------------------------------
template <int DT, int HD>
struct Dense {
  static constexpr int RQT = 8;
  static constexpr bool BF = DT == kBF16;
  static constexpr int ESZ = BF ? 2 : 4;
  static constexpr int KTMAX = 1024;

  static int kt_() {
    static const int k = env_int("CPU_ATTENTION_KT", 256, 32, KTMAX);
    return k;
  }
  // Profiling only (`CPU_ATTENTION_VEC_MODE`): 1 skips the pack, 2 Q@K^T,
  // 4 the softmax, 8 P@V. The skipped phase still leaves defined memory behind
  // so the next one cannot trap on a signalling NaN; the output is garbage.
  static int mode() {
    static const int m = env_int("CPU_ATTENTION_VEC_MODE", 0, 0, 15);
    return m;
  }
  static int rq(int64_t seq_q, int hd) {
    static const int fixed = env_int("CPU_ATTENTION_RQ", 0, 0, 512);
    if (fixed) return static_cast<int>(std::min<int64_t>(fixed, seq_q));
    // The accumulator is rq*hd floats in L1: keep it near 16 KB so it is still
    // there when the next key tile arrives.
    int n = 16384 / (hd * 4);
    n = n < 16 ? 16 : (n > 128 ? 128 : n);
    return static_cast<int>(std::min<int64_t>(n, std::max<int64_t>(seq_q, 1)));
  }
  // `pv` runs its row loop over the whole register tile, so the accumulator is
  // sized for roundup(rq, RQT) rows.
  static int acc_rows(int rq) { return ((rq + RQT - 1) / RQT) * RQT; }
  static int64_t pad_of(int64_t seq_k) { return (seq_k + 63) & ~int64_t(63); }
  // K^T is blocked every 64 keys: `[tile][hd][64]` (bf16 `[tile][hd/2][64][2]`).
  // One dimension's 64 positions are contiguous, so a Q@K^T sweep touches four
  // dimensions per 4 KB page; a plain `[hd][seq_k]` transpose strides `seq_k*4`
  // bytes between dimensions and takes a TLB miss on every load. Both vLLM and
  // the Mojo tree block their K^T for this reason, and it is most of the fp32
  // D=128 gap here.
  static inline int64_t k_off(int64_t jj, int d) {
    return (jj >> 6) * (int64_t(HD) * 64) + int64_t(d) * 64 + (jj & 63);
  }
  static inline int64_t kb_off(int64_t jj, int d2) {
    return ((jj >> 6) * (int64_t(HD / 2) * 64) + int64_t(d2) * 64 + (jj & 63)) *
           2;
  }
  static size_t slab_bytes(int64_t seq_k, int hd) {
    return static_cast<size_t>(2 * pad_of(seq_k) * hd) * ESZ;
  }
  static size_t aux_bytes(int rq, int hd) {
    const size_t rows = static_cast<size_t>(acc_rows(rq));
    // logits (fp32) | probability pairs (bf16, one 32-bit word per key pair) |
    // accumulator | m, l, alpha.
    return static_cast<size_t>(RQT) * KTMAX * (BF ? 6 : 4) + rows * hd * 4 +
           rows * 3 * 4;
  }
  // bf16 stores probabilities as key pairs (what `dpbf16ps` contracts); fp32
  // keeps them fp32 in the logits buffer. The pair buffer sits between the
  // logits tile and the accumulator, which is what `logit_aux_floats` tells the
  // driver; pv finds it from the logits base it is handed.
  static size_t logit_aux_floats(int kt) {
    return BF ? static_cast<size_t>(RQT) * kt / 2 : 0;
  }
  static uint32_t* pairs_of(float* logits) {
    return reinterpret_cast<uint32_t*>(logits + static_cast<size_t>(RQT) * kt_());
  }
  static const uint32_t* pairs_of(const float* logits) {
    return reinterpret_cast<const uint32_t*>(
        logits + static_cast<size_t>(RQT) * kt_());
  }

  // K -> [hd][pad] (bf16: [hd/2][pad][2]); V -> [pad][hd]. One pass, 16 tokens x
  // 16 dims at a time: a 16x16 transpose is four shuffle stages and one 64-byte
  // store per output row, against 11-15 GB/s for torch's element-wise permute.
  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    if (mode() & 1) return;
    const int64_t pad = pad_of(job.seq_k);
    const char* ksrc =
        static_cast<const char*>(job.k) + (b * job.kb + kvh * job.kh) * ESZ;
    const char* vsrc =
        static_cast<const char*>(job.v) + (b * job.vb + kvh * job.vh) * ESZ;
    char* kdst = slab;
    char* vdst = slab + pad * HD * ESZ;
    for (int64_t t0 = 0; t0 < job.seq_k; t0 += 16) {
      const int64_t tn = std::min<int64_t>(16, job.seq_k - t0);
      if (tn == 16) {
        for (int64_t d0 = 0; d0 < HD; d0 += 16) {
          __m512 r[16];
          for (int i = 0; i < 16; ++i) {
            const char* s = ksrc + (t0 + i) * job.kt * ESZ + d0 * ESZ;
            r[i] = BF ? widen_bf16(s) : ld16(s);
          }
          transpose16x16_f32(r);
          if constexpr (BF) {
            for (int d = 0; d < 16; d += 2)
              store_pair_bf16(kdst + kb_off(t0, (d0 + d) / 2) * ESZ, r[d],
                              r[d + 1]);
          } else {
            for (int d = 0; d < 16; ++d)
              st16(kdst + k_off(t0, d0 + d) * 4, r[d]);
          }
        }
      } else {
        for (int64_t t = 0; t < tn; ++t) {
          const char* s = ksrc + (t0 + t) * job.kt * ESZ;
          for (int64_t d = 0; d < HD; ++d) {
            if constexpr (BF) {
              std::memcpy(kdst + (kb_off(t0 + t, d / 2) + (d & 1)) * ESZ,
                          s + d * ESZ, ESZ);
            } else {
              std::memcpy(kdst + k_off(t0 + t, d) * ESZ, s + d * ESZ, ESZ);
            }
          }
        }
      }
      if constexpr (BF) {
        // V goes to [key/2][hd][2]: one 64-byte store holds 16 dims of two
        // adjacent keys, which is the pair `dpbf16ps` contracts in P@V. That is
        // worth 2.1x on the phase (tests/bench_vec_prims.cpp, pv_dot32).
        for (int64_t t = 0; t < tn; t += 2) {
          const char* a = vsrc + (t0 + t) * job.vt * ESZ;
          const bool have_b = (t + 1) < tn;
          const char* b = have_b ? a + job.vt * ESZ : nullptr;
          char* dst = vdst + ((t0 + t) >> 1) * (HD * 2) * ESZ;
          for (int64_t d0 = 0; d0 < HD; d0 += 16) {
            if (have_b) {
              store_key_pair(dst + d0 * 2 * ESZ, a + d0 * ESZ, b + d0 * ESZ);
            } else {
              // The last key of an odd-length sequence has no partner. It still
              // has to be written in *pair* order, interleaved with zeros: a
              // plain copy would put dim d+1 in the slot of key pair (d, d+1).
              static const char zeros[32] = {0};
              store_key_pair(dst + d0 * 2 * ESZ, a + d0 * ESZ, zeros);
            }
          }
        }
      } else {
        for (int64_t t = 0; t < tn; ++t)
          std::memcpy(vdst + (t0 + t) * HD * ESZ,
                      vsrc + (t0 + t) * job.vt * ESZ, HD * ESZ);
      }
    }
    // Zero the padding so a masked lane never reads a NaN left by an earlier
    // geometry (or runs past the slab).
    if (job.seq_k < pad) {
      const int64_t tail = pad - job.seq_k;
      if constexpr (BF) {
        // Pair units: the first key without a partner is (seq_k + 1) / 2, and
        // the pair region is pad / 2 keys wide.
        const int64_t from = (job.seq_k + 1) >> 1;
        const int64_t pairs = (pad >> 1) - from;
        if (pairs > 0)
          std::memset(vdst + from * (HD * 2) * ESZ, 0,
                      pairs * (HD * 2) * ESZ);
      } else {
        std::memset(vdst + job.seq_k * HD * ESZ, 0, tail * HD * ESZ);
      }
      // Only the 16-lane group holding the last key can be read partially; the
      // rest of the padding is never addressed because the key loop stops at
      // the causal limit. Zero the dead lanes of that group so a NaN left by an
      // earlier geometry cannot poison a masked dot product.
      const int tailn = static_cast<int>((16 - (job.seq_k & 15)) & 15);
      if (tailn) {
        for (int d = 0; d < HD; ++d) {
          if constexpr (BF) {
            // Both dims of the pair are adjacent, so one store covers them.
            std::memset(kdst + kb_off(job.seq_k, d / 2) * ESZ, 0,
                        tailn * 2 * ESZ);
          } else {
            std::memset(kdst + k_off(job.seq_k, d) * ESZ, 0, tailn * ESZ);
          }
        }
      }
    }
  }

  // logits[RQT][kt] = q[RQT] @ K^T for keys [k0, k0+nkeys), causal-masked.
  static void qk(const DenseJob& job, const char* slab, int64_t, int64_t,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const int kt = kt_();
    if (mode() & 2) {
      for (int r = 0; r < nrows; ++r)
        std::memset(logits + r * kt, 0, sizeof(float) * nkeys);
      return;
    }
    const int64_t qt = job.qt;
    const int64_t kend = k0 + nkeys;
    const int64_t lim = std::min(kend, row_limit0 + 1);
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           12, 13, 14, 15);
    // The row loop is unrolled over the compile-time RQT: an array of vectors
    // indexed by a runtime bound lands in memory, which turns every FMA into a
    // load+store. Rows past `nrows` read a clamped query row and their logits
    // are never consumed.
    for (int j = 0; j < nkeys; j += LANES) {
      const int64_t jj = k0 + j;
      __m512 c[RQT];
#pragma GCC unroll 16
      for (int r = 0; r < RQT; ++r) c[r] = _mm512_setzero_ps();
      if constexpr (BF) {
        for (int d2 = 0; d2 < HD / 2; ++d2) {
          const __m512bh kb = load_pair_bf16(slab + kb_off(jj, d2) * ESZ);
#pragma GCC unroll 16
          for (int r = 0; r < RQT; ++r) {
            const int rr = r < nrows ? r : nrows - 1;
            c[r] = _mm512_dpbf16_ps(c[r],
                                    bcast_pair(q_base + (rr * qt + 2 * d2) * 2),
                                    kb);
          }
        }
      } else {
        for (int d = 0; d < HD; ++d) {
          const __m512 kb = ld16(slab + k_off(jj, d) * ESZ);
#pragma GCC unroll 16
          for (int r = 0; r < RQT; ++r) {
            const int rr = r < nrows ? r : nrows - 1;
            c[r] =
                _mm512_fmadd_ps(bcast_f32(q_base + (rr * qt + d) * 4), kb, c[r]);
          }
        }
      }
      if (jj + LANES <= lim) {
#pragma GCC unroll 16
        for (int r = 0; r < RQT; ++r)
          if (r < nrows) st16(logits + r * kt + j, c[r]);
      } else {
        const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)jj), lane);
        const __m512 ninf = _mm512_set1_ps(-INFINITY);
#pragma GCC unroll 16
        for (int r = 0; r < RQT; ++r) {
          if (r >= nrows) continue;
          const __mmask16 bad =
              _mm512_cmpge_epi32_mask(idx, _mm512_set1_epi32((int)kend)) |
              _mm512_cmpgt_epi32_mask(
                  idx, _mm512_set1_epi32((int)(row_limit0 + r)));
          st16(logits + r * kt + j, _mm512_mask_blend_ps(bad, c[r], ninf));
        }
      }
    }
  }

  // Online-softmax step over one kt-wide row tile: `logits` is replaced by the
  // probabilities. The scale is folded into the exponential instead of being
  // written back in the max pass: one fewer full pass over the tile per step
  // (9.0 -> 5.8 us of the 4096-key fp32 softmax in tests/bench_vec_prims.cpp).
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    const int kt = kt_();
    const __m512 vc = _mm512_set1_ps(c);
    if (mode() & 4) return;
    for (int r = 0; r < nrows; ++r) {
      float* p = logits + r * kt;
      __m512 mx = _mm512_set1_ps(-INFINITY);
      for (int j = 0; j < nkeys; j += LANES)
        mx = _mm512_max_ps(mx, ld16(p + j));
      const float mxv = _mm512_reduce_max_ps(mx) * c;
      const float mo = m[r];
      float mn, a;
      if (mxv == -INFINITY) {
        mn = mo;
        a = 1.0f;
      } else if (mo == -INFINITY) {
        mn = mxv;
        a = 0.0f;
      } else {
        mn = mo > mxv ? mo : mxv;
        a = std::exp2(mo - mn);
      }
      m[r] = mn;
      alpha[r] = a;
      l[r] *= a;
      if (mn == -INFINITY) {
        // No visible key in this tile: zero what P@V reads.
        if constexpr (BF) {
          uint32_t* pp0 = pairs_of(logits) +
                          static_cast<size_t>(r) * (kt_() / 2);
          std::memset(pp0, 0, sizeof(uint32_t) * static_cast<size_t>(nkeys + 1) / 2);
        } else {
          for (int j = 0; j < nkeys; j += LANES)
            st16(p + j, _mm512_setzero_ps());
        }
        continue;
      }
      const __m512 vmn = _mm512_set1_ps(mn);
      __m512 sum = _mm512_setzero_ps();
      // bf16 P@V contracts two adjacent keys per `dpbf16ps` half-lane, so the
      // probabilities are written as bf16 pairs there (`_mm512_cvtneps_pbh`
      // already emits (p0,p1),(p2,p3),... in key order). fp32 keeps fp32
      // probabilities and a plain FMA P@V: a bf16 hi/lo split would cost three
      // products, i.e. 1.5x the instructions of fp32 FMA on this ISA.
      uint32_t* pp = BF ? pairs_of(logits) + static_cast<size_t>(r) * (kt_() / 2)
                        : nullptr;
      for (int j = 0; j < nkeys; j += LANES) {
        const __m512 e = exp2v(_mm512_fmsub_ps(ld16(p + j), vc, vmn));
        if constexpr (BF) {
          const __m256i h = pack_bf16_16(e);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(pp + (j >> 1)), h);
        } else {
          st16(p + j, e);
        }
        sum = _mm512_add_ps(sum, e);
      }
      l[r] += _mm512_reduce_add_ps(sum);
    }
  }

  // acc[nrows][HD] += p[nrows][nkeys] @ V[k0, k0+nkeys)
  // acc[nrows][HD] += p[nrows][nkeys] @ V[k0, k0+nkeys)
  //
  // bf16 runs `dpbf16ps` with the contraction on keys: one instruction
  // multiplies a broadcast key-pair probability by 16 dims of two adjacent keys
  // and accumulates 32 MACs against 16 for an fp32 FMA. That is 2.1x the phase
  // (tests/bench_vec_prims.cpp, "pv_dot32") and it is why P@V was 56% of the
  // 2048:16:128 bf16 call. The accumulator is a 32-dim chunk (two zmm per row)
  // so eight rows plus the operands stay inside 32 registers.
  static void pv(const DenseJob& job, const char* slab, int64_t, int64_t,
                 int64_t k0, int nkeys, const float* p, int nrows,
                 float* acc) {
    const int kt = kt_();
    if (mode() & 8) return;
    const char* vs = slab + pad_of(job.seq_k) * HD * ESZ;
    const uint32_t* pp = BF ? pairs_of(p) : nullptr;
    const int half = kt / 2;
    for (int dc = 0; dc < HD; dc += 32) {
      __m512 c[RQT][2];
#pragma GCC unroll 16
      for (int r = 0; r < RQT; ++r) {
        c[r][0] = ld16(acc + r * HD + dc);
        c[r][1] = ld16(acc + r * HD + dc + 16);
      }
      if constexpr (BF) {
        for (int j = 0; j < nkeys; j += 2) {
          const char* v = vs + (((k0 + j) >> 1) * HD + dc) * 2 * ESZ;
          const __m512bh v0 = load_pair_bf16(v);
          const __m512bh v1 = load_pair_bf16(v + 64);
          // The probability pairs are tile-local (the softmax wrote one tile),
          // while V is addressed by the global key.
          const int word = j >> 1;
#pragma GCC unroll 16
          for (int r = 0; r < RQT; ++r) {
            const __m512bh pa = bcast_word(pp + r * half + word);
            c[r][0] = _mm512_dpbf16_ps(c[r][0], pa, v0);
            c[r][1] = _mm512_dpbf16_ps(c[r][1], pa, v1);
          }
        }
      } else {
        for (int j = 0; j < nkeys; ++j) {
          const char* v = vs + ((k0 + j) * HD + dc) * ESZ;
          const __m512 v0 = ld16(v);
          const __m512 v1 = ld16(v + 64);
#pragma GCC unroll 16
          for (int r = 0; r < RQT; ++r) {
            const __m512 pr = _mm512_set1_ps(p[r * kt + j]);
            c[r][0] = _mm512_fmadd_ps(pr, v0, c[r][0]);
            c[r][1] = _mm512_fmadd_ps(pr, v1, c[r][1]);
          }
        }
      }
#pragma GCC unroll 16
      for (int r = 0; r < RQT; ++r) {
        st16(acc + r * HD + dc, c[r][0]);
        st16(acc + r * HD + dc + 16, c[r][1]);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// paged policy
// ---------------------------------------------------------------------------
template <int DT, int HD>
struct Paged {
  static constexpr bool BF = DT == kBF16;
  static constexpr int ESZ = BF ? 2 : 4;
  // K rows are 4-byte units: fp32 is one dim, bf16 is the dim pair `dpbf16ps`
  // contracts, so both walk `NDIM` rows of `page * 4` bytes.
  static constexpr int NDIM = BF ? HD / 2 : HD;
  static constexpr int NV = HD / 16;
  static constexpr int R = 4;            // 16-key groups per Q@K^T step
  static constexpr int KSTEP = R * 16;

  static int kt_() {
    static const int k = env_int("CPU_ATTENTION_PAGED_KT", 512, 64, 4096);
    return k & ~15;
  }
  static size_t aux_bytes(int) { return static_cast<size_t>(4096) * 4; }

  // Software prefetch distance along the *current* stream, in dim rows for
  // Q@K^T and in keys for P@V (0 = off). The kernel is latency-bound -- each
  // phase reaches 70-88% of what a plain sequential read of the same bytes
  // measures -- so the question is whether a few lines fetched ahead buy
  // anything. `docs/amx/AMX_PAGED.md` has the sweep.
  static int pf_dist() {
    static const int d = env_int("CPU_ATTENTION_PAGED_PF", 0, 0, 64);
    return d;
  }

  // Loop shapes, chosen per dtype from an in-process A/B against the vec policy
  // (`docs/amx/AMX_PAGED.md` §5). bf16 takes the dim-outer Q@K^T (1.09-1.25x
  // over the vec shape at every shape) and two P@V lanes; fp32 takes the
  // dim-inner Q@K^T and one lane, which is *parity* with the vec shape at
  // 256/1024/4096 keys and 2.5-4% better at b=32 -- the dim-outer form costs
  // fp32 6-9% at three of the five shapes, because fp32 does twice the loads per
  // dim and the 4-way chains then fight for load slots instead of hiding them.
  // Both remain env-switchable for A/B: QK 1 = dim-inner, PV 1 = one lane.
  static int qk_old_shape() {
    static const int v = env_int("CPU_ATTENTION_PAGED_QK", BF ? 0 : 1, 0, 1);
    return v;
  }
  static int pv_one_lane() {
    static const int v = env_int("CPU_ATTENTION_PAGED_PV", BF ? 0 : 1, 0, 1);
    return v;
  }

  // Profiling only (`CPU_ATTENTION_PAGED_MODE`): 1 skips Q@K^T, 2 the softmax,
  // 4 P@V. Results are garbage; timings only.
  static int mode() {
    static const int m = env_int("CPU_ATTENTION_PAGED_MODE", 0, 0, 7);
    return m;
  }

  static int64_t kv_stride_of(const PagedJob& job) {
    // reshape_and_cache reaches this entry through the store-only form, which
    // carries the token stride in the q slot (there is no query there).
    return job.kv_stride ? job.kv_stride : job.q_stride;
  }

  // Same store as the vec policy: fp32 K[hd][page] / V[page][hd], bf16
  // K[hd/2][page][2] / V[page][hd].
  static void store_head(const PagedJob& job, int64_t seq, int64_t head) {
    // `head` is a *query* head: k/v arrive kv-headed (the engine's projection
    // emits one row per kv head), so the source index is the kv head. Every
    // query head of a group writes the same bytes to the same slot; the Python
    // layer orders GQA stores before any attend so the duplicate writes cannot
    // race with a reader.
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / job.page, off = slot % job.page;
    const int64_t stride = job.page * HD * ESZ;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    char* kp = static_cast<char*>(job.k_cache) + base;
    char* vp = static_cast<char*>(job.v_cache) + base;
    const int64_t kvst = kv_stride_of(job);
    const char* ks =
        static_cast<const char*>(job.k) + (seq * kvst + kvh * HD) * ESZ;
    const char* vs =
        static_cast<const char*>(job.v) + (seq * kvst + kvh * HD) * ESZ;
    if constexpr (BF) {
      for (int d = 0; d < HD; ++d)
        std::memcpy(kp + (((d / 2) * job.page + off) * 2 + (d & 1)) * ESZ,
                    ks + d * ESZ, ESZ);
    } else {
      for (int d = 0; d < HD; ++d)
        std::memcpy(kp + (d * job.page + off) * ESZ, ks + d * ESZ, ESZ);
    }
    std::memcpy(vp + off * HD * ESZ, vs, HD * ESZ);
  }

  // -------------------------------------------------------------------------
  // Q@K^T for N 16-key groups: dim-outer, one broadcast per dim, N independent
  // accumulators (N is a compile-time constant so `cv[]` stays in registers).
  // -------------------------------------------------------------------------
  static inline __attribute__((always_inline)) __m512 load_v(const void* p) {
    if constexpr (BF) return widen_bf16(p);
    else return ld16(p);
  }

  template <int N>
  static inline __attribute__((always_inline)) void qk_groups(
      const char* const* kp, int64_t krow, const char* q, float* out, int pfd) {
    __m512 cv[N];
#pragma GCC unroll 8
    for (int r = 0; r < N; ++r) cv[r] = _mm512_setzero_ps();
    if constexpr (BF) {
      for (int di = 0; di < NDIM; ++di) {
        if (pfd != 0 && (di & 1) == 0) {
#pragma GCC unroll 8
          for (int r = 0; r < N; ++r) __builtin_prefetch(kp[r] + (di + pfd) * krow);
        }
        const __m512bh qv = bcast_pair(q + di * 4);
        __m512bh kb[N];
#pragma GCC unroll 8
        for (int r = 0; r < N; ++r) kb[r] = load_pair_bf16(kp[r] + di * krow);
#pragma GCC unroll 8
        for (int r = 0; r < N; ++r) cv[r] = _mm512_dpbf16_ps(cv[r], qv, kb[r]);
      }
    } else {
      const float* qf = reinterpret_cast<const float*>(q);
      for (int di = 0; di < NDIM; ++di) {
        if (pfd != 0 && (di & 1) == 0) {
#pragma GCC unroll 8
          for (int r = 0; r < N; ++r) __builtin_prefetch(kp[r] + (di + pfd) * krow);
        }
        const __m512 qv = _mm512_set1_ps(qf[di]);
        __m512 kb[N];
#pragma GCC unroll 8
        for (int r = 0; r < N; ++r) kb[r] = ld16(kp[r] + di * krow);
#pragma GCC unroll 8
        for (int r = 0; r < N; ++r) cv[r] = _mm512_fmadd_ps(qv, kb[r], cv[r]);
      }
    }
#pragma GCC unroll 8
    for (int r = 0; r < N; ++r) st16(out + r * 16, cv[r]);
  }

  // One 16-key group, with the lane mask for a key range that ends inside it.
  static inline __attribute__((always_inline)) void qk_one(
      const char* kp, int64_t krow, const char* q, float* out, int64_t jj,
      int64_t len, const __m512i& lane, const __m512& ninf) {
    __m512 cv = _mm512_setzero_ps();
    if constexpr (BF) {
      for (int di = 0; di < NDIM; ++di)
        cv = _mm512_dpbf16_ps(cv, bcast_pair(q + di * 4), load_pair_bf16(kp + di * krow));
    } else {
      const float* qf = reinterpret_cast<const float*>(q);
      for (int di = 0; di < NDIM; ++di)
        cv = _mm512_fmadd_ps(_mm512_set1_ps(qf[di]), ld16(kp + di * krow), cv);
    }
    if (jj + 16 <= len) {
      st16(out, cv);
    } else {
      const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)jj), lane);
      const __mmask16 bad =
          _mm512_cmpge_epi32_mask(idx, _mm512_set1_epi32((int)len));
      st16(out, _mm512_mask_blend_ps(bad, cv, ninf));
    }
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char* ws) {
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t len = job.lens[seq];
    const int32_t* bt = job.block_table + seq * job.bt_stride;
    const char* q = static_cast<const char*>(job.q) +
                    (seq * job.q_stride + head * HD) * ESZ;
    char* op = static_cast<char*>(job.out) +
               (seq * job.out_stride + head * HD) * ESZ;
    float* logits = reinterpret_cast<float*>(ws);
    const int64_t page = job.page;
    const int64_t stride = page * HD * ESZ;   // bytes per (page, head)
    const int64_t krow = page * 4;            // bytes between K dim rows
    const int64_t vrow = HD * ESZ;            // bytes between V keys
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          12, 13, 14, 15);
    const __m512 ninf = _mm512_set1_ps(-INFINITY);
    const int kt = kt_();

    __m512 acc[NV];
    for (int i = 0; i < NV; ++i) acc[i] = _mm512_setzero_ps();
    float m = -INFINITY, l = 0.0f;
    const float c = job.scale * kLog2e;
    const __m512 vc = _mm512_set1_ps(c);

    for (int64_t kt0 = 0; kt0 < len; kt0 += kt) {
      const int nkeys = static_cast<int>(std::min<int64_t>(kt, len - kt0));

      // ---- Q@K^T: 64 keys per step (4 pages of work), dim-outer
      const int md = mode();
      const int pfd = pf_dist();
      int j = 0;
      int64_t p = kt0 / page;
      int64_t off = kt0 - p * page;
      const char* kpage = static_cast<const char*>(job.k_cache) +
                          (bt[p] * job.num_kv_heads + kvh) * stride;
      for (; j + KSTEP <= nkeys && (md & 1) == 0 && qk_old_shape() == 0; j += KSTEP) {
        const char* kp[R];
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) {
          if (off >= page) {
            ++p;
            off = 0;
            kpage = static_cast<const char*>(job.k_cache) +
                    (bt[p] * job.num_kv_heads + kvh) * stride;
          }
          kp[r] = kpage + off * 4;
          off += 16;
        }
        qk_groups<R>(kp, krow, q, logits + j, pfd);
      }
      for (; j < nkeys && (md & 1) == 0; j += 16) {
        const int64_t jj = kt0 + j;
        if (off >= page) {
          ++p;
          off = 0;
          kpage = static_cast<const char*>(job.k_cache) +
                  (bt[p] * job.num_kv_heads + kvh) * stride;
        }
        qk_one(kpage + off * 4, krow, q, logits + j, jj, len, lane, ninf);
        off += 16;
      }

      // ---- online softmax: max pass reads only, the scale is folded into the
      // exponential (the reference wrote the scaled logits back, one extra
      // full pass over the tile per 512 keys).
      __m512 mx = _mm512_set1_ps(-INFINITY);
      for (int t = 0; t < nkeys && (md & 2) == 0; t += 16)
        mx = _mm512_max_ps(mx, ld16(logits + t));
      const float mxv = _mm512_reduce_max_ps(mx) * c;
      float mn, a;
      if (mxv == -INFINITY) {
        mn = m;
        a = 1.0f;
      } else if (m == -INFINITY) {
        mn = mxv;
        a = 0.0f;
      } else {
        mn = m > mxv ? m : mxv;
        a = std::exp2(m - mn);
      }
      m = mn;
      l *= a;
      if (a != 1.0f) {
        const __m512 va = _mm512_set1_ps(a);
        for (int i = 0; i < NV; ++i) acc[i] = _mm512_mul_ps(acc[i], va);
      }
      if (mn == -INFINITY) {
        for (int t = 0; t < nkeys; t += 16)
          st16(logits + t, _mm512_setzero_ps());
      } else {
        const __m512 vmn = _mm512_set1_ps(mn);
        __m512 sum = _mm512_setzero_ps();
        for (int t = 0; t < nkeys; t += 16) {
          const __m512 e = exp2v_page<BF>(_mm512_fmsub_ps(ld16(logits + t), vc, vmn));
          st16(logits + t, e);
          sum = _mm512_add_ps(sum, e);
        }
        l += _mm512_reduce_add_ps(sum);
      }

      // ---- P@V: walk pages (one division per page, not per key), two
      // accumulator lanes over key parity.
      __m512 a0[NV], a1[NV];
#pragma GCC unroll 8
      for (int i = 0; i < NV; ++i) {
        a0[i] = acc[i];
        a1[i] = _mm512_setzero_ps();
      }
      int done = (md & 4) ? nkeys : 0;
      int64_t vp64 = kt0 / page;
      int64_t voff = kt0 - vp64 * page;
      while (done < nkeys) {
        const char* vp = static_cast<const char*>(job.v_cache) +
                         (bt[vp64] * job.num_kv_heads + kvh) * stride +
                         voff * vrow;
        const int64_t chunk = std::min<int64_t>(page - voff, nkeys - done);
        const float* lp = logits + done;
        int64_t t = 0;
        for (; t + 1 < chunk && pv_one_lane() == 0; t += 2) {
          if (pfd != 0 && (t & 1) == 0) __builtin_prefetch(vp + (t + 2 * pfd) * vrow);
          const char* vb0 = vp + t * vrow;
          const char* vb1 = vb0 + vrow;
          const __m512 p0 = _mm512_set1_ps(lp[t]);
          const __m512 p1 = _mm512_set1_ps(lp[t + 1]);
          __m512 v0[NV], v1[NV];
#pragma GCC unroll 8
          for (int i = 0; i < NV; ++i) {
            v0[i] = load_v(vb0 + i * 16 * ESZ);
            v1[i] = load_v(vb1 + i * 16 * ESZ);
          }
#pragma GCC unroll 8
          for (int i = 0; i < NV; ++i) {
            a0[i] = _mm512_fmadd_ps(p0, v0[i], a0[i]);
            a1[i] = _mm512_fmadd_ps(p1, v1[i], a1[i]);
          }
        }
        for (; t < chunk; ++t) {
          const char* vb = vp + t * vrow;
          const __m512 pj = _mm512_set1_ps(lp[t]);
#pragma GCC unroll 8
          for (int i = 0; i < NV; ++i)
            a0[i] = _mm512_fmadd_ps(pj, load_v(vb + i * 16 * ESZ), a0[i]);
        }
        done += static_cast<int>(chunk);
        voff += chunk;
        if (voff >= page) {
          voff = 0;
          ++vp64;
        }
      }
#pragma GCC unroll 8
      for (int i = 0; i < NV; ++i) acc[i] = _mm512_add_ps(a0[i], a1[i]);
    }

    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const __m512 vi = _mm512_set1_ps(inv);
    for (int i = 0; i < NV; ++i) {
      const __m512 o = _mm512_mul_ps(acc[i], vi);
      if constexpr (BF) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(op + i * 16 * ESZ),
                            pack_bf16_16(o));
      } else {
        st16(op + i * 16 * ESZ, o);
      }
    }
  }
};

}  // namespace

bool available() {
  const CpuFeatures& f = cpu();
  return f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq;
}

void dense(const DenseJob& job, int workers) {
#define ATTN_DENSE(DT, HD) dense_run<Dense<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_DENSE(kF32, 32);
    else if (job.head_dim == 64) ATTN_DENSE(kF32, 64);
    else ATTN_DENSE(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_DENSE(kBF16, 32);
    else if (job.head_dim == 64) ATTN_DENSE(kBF16, 64);
    else ATTN_DENSE(kBF16, 128);
  }
#undef ATTN_DENSE
}

void paged(const PagedJob& job, int workers) {
#define ATTN_PAGED(DT, HD) paged_run<Paged<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_PAGED(kF32, 32);
    else if (job.head_dim == 64) ATTN_PAGED(kF32, 64);
    else ATTN_PAGED(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_PAGED(kBF16, 32);
    else if (job.head_dim == 64) ATTN_PAGED(kBF16, 64);
    else ATTN_PAGED(kBF16, 128);
  }
#undef ATTN_PAGED
}

void store(const PagedJob& job, int workers) {
#define ATTN_STORE(DT, HD) store_run<Paged<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_STORE(kF32, 32);
    else if (job.head_dim == 64) ATTN_STORE(kF32, 64);
    else ATTN_STORE(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_STORE(kBF16, 32);
    else if (job.head_dim == 64) ATTN_STORE(kBF16, 64);
    else ATTN_STORE(kBF16, 128);
  }
#undef ATTN_STORE
}

}  // namespace vec
}  // namespace attn
