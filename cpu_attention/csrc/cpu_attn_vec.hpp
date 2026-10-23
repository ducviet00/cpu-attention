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
// K plus RQT broadcast-FMAs, i.e. <= 1.25 loads per FMA, which these cores can
// issue two of per cycle. P@V keeps the whole head-dim accumulator in registers
// (paged: one row, 4-8 zmm) or an 8x32 tile (dense) so it is FMA-bound too.
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

// fp32 -> bf16, 16 lanes, as a bit pattern the stores can take.
inline __m256i pack_bf16_16(__m512 v) {
  const __m256bh h = _mm512_cvtneps_pbh(v);
  __m256i r;
  std::memcpy(&r, &h, 32);
  return r;
}

// 2^x for x <= 0: degree-5 polynomial on the fraction, exponent by insertion.
// ~10 instructions; std::exp costs more than twice that and the softmax is
// 15-30% of a call.
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

// ---------------------------------------------------------------------------
// dense policy
// ---------------------------------------------------------------------------
template <int DT, int HD>
struct Dense {
  static constexpr int RQT = 8;
  static constexpr int KT = HD == 128 ? 256 : 512;
  static constexpr bool BF = DT == kBF16;
  static constexpr int ESZ = BF ? 2 : 4;

  static int rq(int64_t seq_q, int hd) {
    // The accumulator is rq*hd floats in L1: keep it near 32 KB so it is still
    // there when the next key tile arrives, and stop growing when the query
    // block already covers the whole sequence.
    int n = 32768 / (hd * 4);
    n = n < 16 ? 16 : (n > 128 ? 128 : n);
    return static_cast<int>(std::min<int64_t>(n, std::max<int64_t>(seq_q, 1)));
  }
  static int64_t pad_of(int64_t seq_k) { return (seq_k + 63) & ~int64_t(63); }
  static size_t slab_bytes(int64_t seq_k, int hd) {
    return static_cast<size_t>(2 * pad_of(seq_k) * hd) * ESZ;
  }
  static size_t aux_bytes(int rq, int hd) {
    return static_cast<size_t>(RQT * KT) * 4 + static_cast<size_t>(rq) * hd * 4 +
           static_cast<size_t>(rq) * 3 * 4;
  }

  // K -> [hd][pad] (bf16: [hd/2][pad][2]); V -> [pad][hd]. One pass, 16 tokens x
  // 16 dims at a time: a 16x16 transpose is 4 shuffle stages and one 64-byte
  // store per output row, against 40-70 GB/s for torch's element-wise permute.
  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    const int64_t pad = pad_of(job.seq_k);
    const char* ksrc =
        static_cast<const char*>(job.k) + (b * job.kb + kvh * job.kh) * ESZ;
    const char* vsrc =
        static_cast<const char*>(job.v) + (b * job.vb + kvh * job.vh) * ESZ;
    char* kdst = slab;
    char* vdst = slab + pad * HD * ESZ;
    for (int64_t t0 = 0; t0 < pad; t0 += 16) {
      const int64_t tn = std::min<int64_t>(16, job.seq_k - t0);
      for (int64_t d0 = 0; d0 < HD; d0 += 16) {
        if (tn == 16) {
          __m512 r[16];
          for (int i = 0; i < 16; ++i) {
            const char* s = ksrc + (t0 + i) * job.kt * ESZ + d0 * ESZ;
            r[i] = BF ? widen_bf16(s) : ld16(s);
          }
          transpose16x16_f32(r);
          if constexpr (BF) {
            for (int d = 0; d < 16; d += 2)
              store_pair_bf16(kdst + (((d0 + d) / 2 * pad + t0) * 2) * ESZ,
                              r[d], r[d + 1]);
          } else {
            for (int d = 0; d < 16; ++d)
              st16(kdst + ((d0 + d) * pad + t0) * 4, r[d]);
          }
        } else {
          for (int64_t t = 0; t < tn; ++t) {
            const char* s = ksrc + (t0 + t) * job.kt * ESZ + d0 * ESZ;
            for (int d = 0; d < 16; ++d) {
              if constexpr (BF) {
                std::memcpy(kdst + (((d0 + d) / 2 * pad + t0 + t) * 2 +
                                    ((d0 + d) & 1)) * ESZ,
                            s + d * ESZ, ESZ);
              } else {
                std::memcpy(kdst + ((d0 + d) * pad + t0 + t) * ESZ,
                            s + d * ESZ, ESZ);
              }
            }
          }
        }
      }
      for (int64_t t = 0; t < tn; ++t)
        std::memcpy(vdst + (t0 + t) * HD * ESZ, vsrc + (t0 + t) * job.vt * ESZ,
                    HD * ESZ);
      if (tn < 16) {
        std::memset(vdst + (t0 + tn) * HD * ESZ, 0, (16 - tn) * HD * ESZ);
        for (int d = 0; d < HD; ++d) {
          if constexpr (BF) {
            std::memset(kdst + (((d / 2) * pad + t0 + tn) * 2 + (d & 1)) * ESZ,
                        0, ESZ);
          } else {
            std::memset(kdst + ((d * pad) + t0 + tn) * ESZ, 0, ESZ);
          }
        }
      }
    }
  }

  static void qk(const DenseJob& job, const char* slab, int64_t, int64_t,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const int64_t pad = pad_of(job.seq_k);
    const int64_t qt = job.qt;
    const int64_t kend = k0 + nkeys;
    const int64_t lim = std::min(kend, row_limit0 + 1);
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           12, 13, 14, 15);
    for (int j = 0; j < nkeys; j += LANES) {
      const int64_t jj = k0 + j;
      __m512 c[RQT];
      for (int r = 0; r < nrows; ++r) c[r] = _mm512_setzero_ps();
      if constexpr (BF) {
        for (int d2 = 0; d2 < HD / 2; ++d2) {
          const __m512bh kb = load_pair_bf16(slab + ((d2 * pad + jj) * 2) * 2);
          for (int r = 0; r < nrows; ++r)
            c[r] = _mm512_dpbf16_ps(c[r],
                                    bcast_pair(q_base + (r * qt + 2 * d2) * 2),
                                    kb);
        }
      } else {
        for (int d = 0; d < HD; ++d) {
          const __m512 kb = ld16(slab + ((d * pad) + jj) * 4);
          for (int r = 0; r < nrows; ++r)
            c[r] =
                _mm512_fmadd_ps(bcast_f32(q_base + (r * qt + d) * 4), kb, c[r]);
        }
      }
      if (jj + LANES <= lim) {
        for (int r = 0; r < nrows; ++r) st16(logits + r * KT + j, c[r]);
      } else {
        const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)jj), lane);
        const __m512 ninf = _mm512_set1_ps(-INFINITY);
        for (int r = 0; r < nrows; ++r) {
          const __mmask16 bad =
              _mm512_cmpge_epi32_mask(idx, _mm512_set1_epi32((int)kend)) |
              _mm512_cmpgt_epi32_mask(
                  idx, _mm512_set1_epi32((int)(row_limit0 + r)));
          st16(logits + r * KT + j, _mm512_mask_blend_ps(bad, c[r], ninf));
        }
      }
    }
  }

  // Online-softmax step over one KT-wide row tile: `logits` is scaled in place
  // by c (= scale*log2e) and replaced by the probabilities.
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    const __m512 vc = _mm512_set1_ps(c);
    for (int r = 0; r < nrows; ++r) {
      float* p = logits + r * KT;
      __m512 mx = _mm512_set1_ps(-INFINITY);
      for (int j = 0; j < nkeys; j += LANES) {
        const __m512 s = _mm512_mul_ps(ld16(p + j), vc);
        st16(p + j, s);
        mx = _mm512_max_ps(mx, s);
      }
      const float mxv = _mm512_reduce_max_ps(mx);
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
        for (int j = 0; j < nkeys; j += LANES) st16(p + j, _mm512_setzero_ps());
        continue;
      }
      const __m512 vmn = _mm512_set1_ps(mn);
      __m512 sum = _mm512_setzero_ps();
      for (int j = 0; j < nkeys; j += LANES) {
        const __m512 e = exp2v(_mm512_sub_ps(ld16(p + j), vmn));
        st16(p + j, e);
        sum = _mm512_add_ps(sum, e);
      }
      l[r] += _mm512_reduce_add_ps(sum);
    }
  }

  // acc[nrows][HD] += p[nrows][nkeys] @ V[k0, k0+nkeys)
  static void pv(const DenseJob& job, const char* slab, int64_t, int64_t,
                 int64_t k0, int nkeys, const float* p, int nrows,
                 float* acc) {
    const char* vs = slab + pad_of(job.seq_k) * HD * ESZ;
    for (int dc = 0; dc < HD; dc += 32) {
      __m512 c[RQT][2];
      for (int r = 0; r < nrows; ++r) {
        c[r][0] = ld16(acc + r * HD + dc);
        c[r][1] = ld16(acc + r * HD + dc + 16);
      }
      for (int j = 0; j < nkeys; ++j) {
        const char* v = vs + ((k0 + j) * HD + dc) * ESZ;
        __m512 v0, v1;
        if constexpr (BF) {
          v0 = widen_bf16(v);
          v1 = widen_bf16(v + 32);
        } else {
          v0 = ld16(v);
          v1 = ld16(v + 64);
        }
        for (int r = 0; r < nrows; ++r) {
          const __m512 pr = _mm512_set1_ps(p[r * KT + j]);
          c[r][0] = _mm512_fmadd_ps(pr, v0, c[r][0]);
          c[r][1] = _mm512_fmadd_ps(pr, v1, c[r][1]);
        }
      }
      for (int r = 0; r < nrows; ++r) {
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
  static constexpr int KT = 512;
  static constexpr int NV = HD / 16;

  static size_t aux_bytes(int) { return static_cast<size_t>(KT) * 4; }

  static int64_t kv_stride_of(const PagedJob& job) {
    // reshape_and_cache reaches this entry through the store-only form, which
    // carries the token stride in the q slot (there is no query there).
    return job.kv_stride ? job.kv_stride : job.q_stride;
  }

  static void store_head(const PagedJob& job, int64_t seq, int64_t head) {
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / job.page, off = slot % job.page;
    const int64_t esz = ESZ;
    const int64_t stride = job.page * HD * esz;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    char* kp = static_cast<char*>(job.k_cache) + base;
    char* vp = static_cast<char*>(job.v_cache) + base;
    const int64_t kvst = kv_stride_of(job);
    const char* ks =
        static_cast<const char*>(job.k) + (seq * kvst + head * HD) * esz;
    const char* vs =
        static_cast<const char*>(job.v) + (seq * kvst + head * HD) * esz;
    if constexpr (BF) {
      for (int d = 0; d < HD; ++d)
        std::memcpy(kp + (((d / 2) * job.page + off) * 2 + (d & 1)) * esz,
                    ks + d * esz, esz);
    } else {
      for (int d = 0; d < HD; ++d)
        std::memcpy(kp + (d * job.page + off) * esz, ks + d * esz, esz);
    }
    std::memcpy(vp + off * HD * esz, vs, HD * esz);
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
    const int64_t stride = job.page * HD * ESZ;
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           12, 13, 14, 15);

    __m512 acc[NV];
    for (int i = 0; i < NV; ++i) acc[i] = _mm512_setzero_ps();
    float m = -INFINITY, l = 0.0f;
    const float c = job.scale * kLog2e;

    for (int64_t kt0 = 0; kt0 < len; kt0 += KT) {
      const int nkeys = static_cast<int>(std::min<int64_t>(KT, len - kt0));
      for (int j = 0; j < nkeys; j += LANES) {
        const int64_t jj = kt0 + j;
        const char* kb = static_cast<const char*>(job.k_cache) +
                         (bt[jj / job.page] * job.num_kv_heads + kvh) * stride;
        const int64_t off = jj % job.page;
        __m512 cv = _mm512_setzero_ps();
        if constexpr (BF) {
          for (int d2 = 0; d2 < HD / 2; ++d2) {
            const __m512bh kbv =
                load_pair_bf16(kb + (((d2 * job.page) + off) * 2) * 2);
            cv = _mm512_dpbf16_ps(cv, bcast_pair(q + 2 * d2 * 2), kbv);
          }
        } else {
          for (int d = 0; d < HD; ++d) {
            const __m512 kbv = ld16(kb + ((d * job.page) + off) * 4);
            cv = _mm512_fmadd_ps(bcast_f32(q + d * 4), kbv, cv);
          }
        }
        if (jj + LANES <= len) {
          st16(logits + j, cv);
        } else {
          const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)jj), lane);
          const __mmask16 bad = _mm512_cmpge_epi32_mask(
              idx, _mm512_set1_epi32((int)len));
          st16(logits + j,
               _mm512_mask_blend_ps(bad, cv, _mm512_set1_ps(-INFINITY)));
        }
      }
      const __m512 vc = _mm512_set1_ps(c);
      __m512 mx = _mm512_set1_ps(-INFINITY);
      for (int j = 0; j < nkeys; j += LANES) {
        const __m512 s = _mm512_mul_ps(ld16(logits + j), vc);
        st16(logits + j, s);
        mx = _mm512_max_ps(mx, s);
      }
      const float mxv = _mm512_reduce_max_ps(mx);
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
        for (int j = 0; j < nkeys; j += LANES)
          st16(logits + j, _mm512_setzero_ps());
      } else {
        const __m512 vmn = _mm512_set1_ps(mn);
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < nkeys; j += LANES) {
          const __m512 e = exp2v(_mm512_sub_ps(ld16(logits + j), vmn));
          st16(logits + j, e);
          sum = _mm512_add_ps(sum, e);
        }
        l += _mm512_reduce_add_ps(sum);
      }
      for (int j = 0; j < nkeys; ++j) {
        const int64_t jj = kt0 + j;
        const char* vb = static_cast<const char*>(job.v_cache) +
                         (bt[jj / job.page] * job.num_kv_heads + kvh) * stride +
                         ((jj % job.page) * HD) * ESZ;
        const __m512 pj = _mm512_set1_ps(logits[j]);
        for (int i = 0; i < NV; ++i) {
          __m512 v;
          if constexpr (BF) {
            v = widen_bf16(vb + i * 16 * ESZ);
          } else {
            v = ld16(vb + i * 16 * ESZ);
          }
          acc[i] = _mm512_fmadd_ps(pj, v, acc[i]);
        }
      }
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
