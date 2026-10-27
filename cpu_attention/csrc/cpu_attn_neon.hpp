// NEON kernel (aarch64, armv8.2-a+bf16+i8mm): fp32 and bf16, dense and paged.
//
// Layouts (this policy defines them; `pack`/`store_head` write them):
//
//   dense fp32   K [hd][pad]           V [pad][hd]
//   dense bf16   K [tile][hd/4][32][8] V [tile][hd/2][16][8] (64-key tiles;
//                                       D=64 keeps V [hd/2][pad/4][8])
//   paged fp32   K [hd][page]          V [page][hd]      (byte-compatible with
//                                                        the VEC and Mojo paths)
//   paged bf16   K [hd/2][page][2]     V [page/2][hd][2] (the Mojo ARM layout)
//
// The dense bf16 layouts are the two operands of `bfmmla`, whose result is a 2x2
// fp32 tile: `acc[i][j] += dot(a[4i..4i+3], b[4j..4j+3])`, i.e. `a` holds two
// *rows* of four contraction lanes and `b` two *columns*. For Q@K^T that is two
// query rows x 4 head dims against two keys x 4 head dims; for P@V it is two
// query rows x 4 keys against two head dims x 4 keys (verified against a scalar
// reference in cpu_attention/_mojo/tests/bfmmla_layout.mojo).
//
// `bfmmla` retires 16 MACs an instruction against 4 for a 4-wide fp32 FMA, and
// this core is instruction-issue bound rather than MAC bound, so that ratio is
// why the bf16 dense path exists at all. The paged path has one query row per
// head, where half of `bfmmla`'s 2x2 tile would be dead weight, so it uses
// `bfdot` (8 MACs, one row) for both phases.
//
// RQT = 8 rows per register tile on the dense path: that is the K/V traffic
// lever (a key tile is read once per RQT rows) and it leaves room for two 2x2
// accumulator panels in the 32-register file. KT = 512 keys per online-softmax
// step amortises the per-tile fixed cost (row max, alpha, and the accumulator
// rescale the shared driver does) while keeping the fp32 logits and the
// probability panel inside the 64 KB L1.
//
// Apache-2.0; the loop structure follows vLLM's csrc/cpu NEON kernels, and this
// repo's Mojo dense.mojo / attention.mojo are the correctness oracles.
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "cpu_attn_impl.hpp"
#include "cpu_isa.hpp"

namespace attn {
namespace neon {
namespace {

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------
constexpr float kNegInf = -INFINITY;

// Profiling only (`CPU_ATTENTION_NEON_MODE`): 1 skips the pack, 2 Q@K^T, 4 the
// softmax, 8 P@V. The skipped phase leaves defined memory behind; results are
// garbage and timings only.
inline int neon_mode() {
  static const int m = env_int("CPU_ATTENTION_NEON_MODE", 0, 0, 15);
  return m;
}

inline float32x4_t ldf(const void* p) {
  return vld1q_f32(static_cast<const float*>(p));
}
inline void stf(void* p, float32x4_t v) {
  vst1q_f32(static_cast<float*>(p), v);
}
inline bfloat16x8_t ldb(const void* p) {
  return vld1q_bf16(static_cast<const bfloat16_t*>(p));
}
inline void stb(void* p, bfloat16x8_t v) {
  vst1q_bf16(static_cast<bfloat16_t*>(p), v);
}
inline bfloat16x4_t ldb4(const void* p) {
  return vld1_bf16(static_cast<const bfloat16_t*>(p));
}
inline void stb4(void* p, bfloat16x4_t v) {
  vst1_bf16(static_cast<bfloat16_t*>(p), v);
}

// bf16 output of `hd` floats scaled by `inv`: one 16-byte store per 8 dims.
// `vcvt_bf16_f32` rounds to nearest even, the same as the scalar
// `f32_to_bf16`, and the vector form is 2.5x the scalar loop (which the NaN
// branch in `f32_to_bf16` keeps GCC from vectorising).
inline void store_bf16_out(char* dst, const float* src, int hd, float inv) {
  const float32x4_t vi = vdupq_n_f32(inv);
  for (int d = 0; d < hd; d += 8)
    stb(dst + d * 2,
        vcombine_bf16(vcvt_bf16_f32(vmulq_f32(vld1q_f32(src + d), vi)),
                      vcvt_bf16_f32(vmulq_f32(vld1q_f32(src + d + 4), vi))));
}

// Two adjacent bf16 (a head-dim pair of one key, or a key pair of one
// probability row) broadcast to all four fp32 lanes: the A operand of bfdot.
inline bfloat16x8_t dup_pair(const void* p) {
  return vreinterpretq_bf16_u32(
      vld1q_dup_u32(reinterpret_cast<const uint32_t*>(p)));
}

inline float32x4_t bfdot(float32x4_t acc, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfdotq_f32(acc, a, b);
}
inline float32x4_t bfmmla(float32x4_t acc, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmmlaq_f32(acc, a, b);
}

// 2^x for x <= 0: polynomial on the fraction plus an exponent insert, about ten
// instructions. DEG 5 is utils.hpp's `exp2_poly5` (3.4e-6 relative, so the fp32
// error against a scalar reference stays where the vector path had it); DEG 3 is
// the least-squares cubic (1.9e-4), two orders inside the bf16 store that
// follows it.
template <int DEG>
inline float32x4_t exp2v(float32x4_t x) {
  x = vmaxq_f32(x, vdupq_n_f32(-126.0f));
  // n = round(x) once as an integer (fcvtns) and once as a float; 2^n is then
  // an integer add into p's exponent field instead of building 2^n and
  // multiplying, one instruction fewer per vector.
  const int32x4_t ni = vcvtnq_s32_f32(x);
  const float32x4_t r = vsubq_f32(x, vcvtq_f32_s32(ni));
  // Horner with FMA: p = c_hi, then p = c_next + p*r.
  float32x4_t p;
  if constexpr (DEG >= 6) {
    p = vdupq_n_f32(1.5403530e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, r);
  } else if constexpr (DEG >= 5) {
    p = vdupq_n_f32(1.3392112e-3f);
    p = vfmaq_f32(vdupq_n_f32(9.6180963e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.5503282e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(2.4022652e-1f), p, r);
  } else if constexpr (DEG == 2) {
    p = vdupq_n_f32(2.3473140e-1f);
  } else {
    p = vdupq_n_f32(5.587586976e-2f);
    p = vfmaq_f32(vdupq_n_f32(2.422964968e-1f), p, r);
  }
  p = vfmaq_f32(vdupq_n_f32(DEG == 2 ? 6.9335938e-1f : 6.9314718e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  // p is in [0.70, 1.42], so its exponent field is 126 or 127 and adding
  // n >= -126 cannot underflow past zero (at the clamp it lands in the
  // subnormal range, ~1e-38, where only the "exactly tiny" property matters).
  return vreinterpretq_f32_s32(
      vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(ni, 23)));
}

inline float hmax(float32x4_t v) { return vmaxvq_f32(v); }
inline float hsum(float32x4_t v) { return vaddvq_f32(v); }

inline int32x4_t lane_idx() {
  static const int32_t k[4] = {0, 1, 2, 3};
  return vld1q_s32(k);
}

// 4x4 fp32 transpose, in place (rows -> columns).
inline void tr4x4(float32x4_t& a, float32x4_t& b, float32x4_t& c,
                  float32x4_t& d) {
  const float32x4_t t0 = vtrn1q_f32(a, b), t1 = vtrn2q_f32(a, b);
  const float32x4_t t2 = vtrn1q_f32(c, d), t3 = vtrn2q_f32(c, d);
  a = vreinterpretq_f32_f64(
      vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
  b = vreinterpretq_f32_f64(
      vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
  c = vreinterpretq_f32_f64(
      vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
  d = vreinterpretq_f32_f64(
      vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

// bfmmla 2x2 tiles (r0c0, r0c1, r1c0, r1c1) from two adjacent column pairs
// zipped into one row's four values.
inline float32x4_t zip_lo(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_f64(
      vzip1q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
}
inline float32x4_t zip_hi(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_f64(
      vzip2q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
}

// Keys per packed slab: whole 64-key K^T tiles (see `k_off`), so the tail is
// zero padded to a tile boundary and every tile in the slab is uniform.
inline int64_t pad_of(int64_t seq_k) { return (seq_k + 63) & ~int64_t(63); }

inline size_t round64(size_t n) { return (n + 63) & ~size_t(63); }

template <int RQT, int KT>
inline void mask_and_max(float* logits, int nrows, int nkeys, int64_t k0,
                         int64_t row_limit0, float* tmax) {
  const float32x4_t ninf = vdupq_n_f32(kNegInf);
  const int32x4_t lane = lane_idx();
  for (int r = 0; r < RQT; ++r) {
    const int64_t limit = row_limit0 + std::min(r, nrows - 1) - k0 + 1;
    const int valid = static_cast<int>(std::clamp<int64_t>(limit, 0, nkeys));
    float* row = logits + r * KT;
    float32x4_t mx = ninf;
    int j = 0;
    for (; j + 8 <= valid; j += 8)
      mx = vmaxq_f32(mx, vmaxq_f32(ldf(row + j), ldf(row + j + 4)));
    for (; j < nkeys; j += 4) {
      const int32x4_t idx = vaddq_s32(vdupq_n_s32(j), lane);
      const float32x4_t x = vbslq_f32(vcgeq_s32(idx, vdupq_n_s32(valid)),
                                     ninf, ldf(row + j));
      stf(row + j, x);
      mx = vmaxq_f32(mx, x);
    }
    tmax[r] = hmax(mx);
  }
}

// ===========================================================================
// dense fp32
// ===========================================================================
template <int HD, int KTC>
struct DenseF32 {
  static constexpr int RQT = 8;
  // `pv` writes the accumulator on its first key tile, so the driver's memset of
  // `acc_rows * hd` floats per work item is dead work: 256 KB per call at
  // 128:8:64 (a quarter of the row) and 3-11% at 2048-4096 keys.
  static constexpr bool zeroes_acc = true;
  static void store_bf16(char* dst, const float* src, int hd, float inv) {
    store_bf16_out(dst, src, hd, inv);
  }
  static constexpr int KTMAXV = 2048;

  static constexpr int kt_() { return KTC; }
  // Work item = (batch, q head, q tile) with vLLM's q tile: as many rows as fit
  // the tile, so one K/V sweep feeds them and the accumulator lives in L2 rather
  // than the register file.
  static int& rq_cached() {
    static int v = 64;
    return v;
  }
  static int rq(int64_t seq_q, int) {
    static const int fixed = env_int("CPU_ATTENTION_RQ", 0, 0, 4096);
    const int64_t base =
        seq_q <= 512 ? seq_q : (HD >= 128 ? int64_t(128) : int64_t(64));
    const int64_t want = fixed > 0 ? fixed : base;
    const int64_t capped = std::min<int64_t>(want, std::max<int64_t>(seq_q, 1));
    const int v =
        static_cast<int>(std::max<int64_t>(RQT, (capped / RQT) * RQT));
    rq_cached() = v;
    return v;
  }
  // `pv` runs its row loop over the whole register tile, so the accumulator is
  // sized for roundup(rq, RQT) rows even when the last block is short.
  static int acc_rows(int rq_) { return ((rq_ + RQT - 1) / RQT) * RQT; }

  static size_t slab_bytes(int64_t seq_k, int) {
    return static_cast<size_t>(2) * pad_of(seq_k) * HD * 4;
  }
  // [logits RQT*kt f32][acc acc_rows(rq)*hd f32][m,l,alpha rq][tmax RQT][P_T]
  // [Qpack RQT*hd f32]
  static size_t aux_bytes(int rq_, int) {
    return round64(static_cast<size_t>(RQT) * kt_() * 4 +
                   static_cast<size_t>(acc_rows(rq_)) * HD * 4 +
                   static_cast<size_t>(rq_) * 3 * 4 +
                   static_cast<size_t>(RQT) * 4 + 64 +
                   static_cast<size_t>(kt_()) * RQT * 4 +
                   static_cast<size_t>(RQT) * HD * 4);
  }
  static float* tmax_of(float* logits) {
    const int rq_ = rq_cached();
    return logits + static_cast<size_t>(RQT) * kt_() +
           static_cast<size_t>(acc_rows(rq_)) * HD +
           static_cast<size_t>(rq_) * 3;
  }
  static float* pof(float* logits) { return tmax_of(logits) + RQT + 16; }
  static float* qpack_of(float* logits) {
    return pof(logits) + static_cast<size_t>(kt_()) * RQT;
  }

  // K^T is blocked every 64 keys: `[tile][hd][64]`. One dimension's 64
  // positions are contiguous (256 B at D=64) instead of `pad` floats apart, so
  // a Q@K^T sweep touches HD/16 pages per key tile instead of one page per
  // dimension -- the TLB effect the AVX-512 path measured at 1.17x on the fp32
  // D=128 rows.
  static inline int64_t k_off(int64_t jj, int d) {
    return (jj >> 6) * (int64_t(HD) * 64) + int64_t(d) * 64 + (jj & 63);
  }

  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    if (neon_mode() & 1) return;
    const int64_t pad = pad_of(job.seq_k);
    const int64_t n = job.seq_k;
    const float* ks =
        reinterpret_cast<const float*>(job.k) + b * job.kb + kvh * job.kh;
    const float* vs =
        reinterpret_cast<const float*>(job.v) + b * job.vb + kvh * job.vh;
    float* kd = reinterpret_cast<float*>(slab);  // [HD][pad]
    float* vd = kd + HD * pad;                   // [pad][HD]
    for (int64_t t = 0; t < n; ++t)
      std::memcpy(vd + t * HD, vs + t * job.vt, sizeof(float) * HD);
    std::memset(vd + n * HD, 0, sizeof(float) * (pad - n) * HD);
    int64_t t0 = 0;
    for (; t0 + 4 <= n; t0 += 4) {
      for (int d0 = 0; d0 < HD; d0 += 4) {
        float32x4_t r0 = ldf(ks + (t0 + 0) * job.kt + d0);
        float32x4_t r1 = ldf(ks + (t0 + 1) * job.kt + d0);
        float32x4_t r2 = ldf(ks + (t0 + 2) * job.kt + d0);
        float32x4_t r3 = ldf(ks + (t0 + 3) * job.kt + d0);
        tr4x4(r0, r1, r2, r3);
        stf(kd + k_off(t0, d0 + 0), r0);
        stf(kd + k_off(t0, d0 + 1), r1);
        stf(kd + k_off(t0, d0 + 2), r2);
        stf(kd + k_off(t0, d0 + 3), r3);
      }
    }
    for (; t0 < n; ++t0)
      for (int d = 0; d < HD; ++d) kd[k_off(t0, d)] = ks[t0 * job.kt + d];
    for (int64_t t = n; t < pad; ++t)
      for (int d = 0; d < HD; ++d) kd[k_off(t, d)] = 0.0f;
  }

  // logits[r][j] = sum_d q[r][d] * K[d][k0+j], -inf past each row's limit.
  static void qk(const DenseJob& job, const char* slab, int64_t b, int64_t kvh,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const int KT = kt_();
    const int64_t qt = job.qt;
    const float* kd = reinterpret_cast<const float*>(slab);
    if (neon_mode() & 2) {
      for (int r = 0; r < nrows; ++r)
        std::memset(logits + r * KT, 0, sizeof(float) * nkeys);
      return;
    }
    // RQT is a compile-time trip count on purpose: rows past `nrows` are packed
    // as copies of the last real row (no out-of-bounds load, no NaN) and are
    // stored into the logits tile, which the softmax then ignores. That keeps
    // every accumulator index constant, which is the difference between
    // sixteen registers and sixteen stack slots. The pack itself puts the rows
    // in one contiguous L1 panel: the caller's rows are a whole token stride
    // apart, and the loop below reads each of them once per eight keys.
    float* qpk = qpack_of(logits);
    for (int r = 0; r < RQT; ++r) {
      const char* src = q_base + static_cast<int64_t>(std::min(r, nrows - 1)) *
                                     qt * 4;
      std::memcpy(qpk + r * HD, src, sizeof(float) * HD);
    }
    for (int j = 0; j < nkeys; j += 8) {
      const int64_t jj = k0 + j;
      float32x4_t c[RQT][2];
      for (int r = 0; r < RQT; ++r) {
        c[r][0] = vdupq_n_f32(0.0f);
        c[r][1] = vdupq_n_f32(0.0f);
      }
      // Keys jj..jj+7 of one dimension are 32 contiguous bytes of the tile.
      const float* kp = kd + k_off(jj, 0);
      for (int d0 = 0; d0 < HD; d0 += 4) {
        float32x4_t kv[8];
        for (int dd = 0; dd < 4; ++dd) {
          kv[2 * dd + 0] = ldf(kp + (d0 + dd) * 64);
          kv[2 * dd + 1] = ldf(kp + (d0 + dd) * 64 + 4);
        }
        for (int r = 0; r < RQT; ++r) {
          const float32x4_t qv = ldf(qpk + r * HD + d0);
          c[r][0] = vfmaq_laneq_f32(c[r][0], kv[0], qv, 0);
          c[r][1] = vfmaq_laneq_f32(c[r][1], kv[1], qv, 0);
          c[r][0] = vfmaq_laneq_f32(c[r][0], kv[2], qv, 1);
          c[r][1] = vfmaq_laneq_f32(c[r][1], kv[3], qv, 1);
          c[r][0] = vfmaq_laneq_f32(c[r][0], kv[4], qv, 2);
          c[r][1] = vfmaq_laneq_f32(c[r][1], kv[5], qv, 2);
          c[r][0] = vfmaq_laneq_f32(c[r][0], kv[6], qv, 3);
          c[r][1] = vfmaq_laneq_f32(c[r][1], kv[7], qv, 3);
        }
      }
      for (int r = 0; r < RQT; ++r) {
        stf(logits + r * KT + j, c[r][0]);
        stf(logits + r * KT + j + 4, c[r][1]);
      }
    }
    mask_and_max<RQT, KT>(logits, nrows, nkeys, k0, row_limit0,
                          tmax_of(logits));
  }

  // Scales the tile by c and turns it into probabilities, transposed into the P
  // panel (P_T[j][r]) so P@V loads one vector of rows per key.
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    // Four rows at a time: the four exp vectors are transposed so the P panel
    // ([key][row]) is written with one 16-byte store per key instead of four
    // strided 4-byte lane stores, and the four chains give the exp some ILP.
    const int KT = kt_();
    float* pt = pof(logits);
    const float* tm = tmax_of(logits);
    const float32x4_t vc = vdupq_n_f32(c);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int r0 = 0; r0 < nrows; r0 += 4) {
      float mn[4], av[4];
      for (int k = 0; k < 4; ++k) {
        const int r = r0 + k;
        const float mxv = tm[r] * c;  // c > 0: max(c*x) == c*max(x)
        const float mo = m[r];
        if (mxv == kNegInf) {
          mn[k] = mo;
          av[k] = 1.0f;
        } else if (mo == kNegInf) {
          mn[k] = mxv;
          av[k] = 0.0f;
        } else {
          mn[k] = mxv;
          av[k] = std::exp2(mo - mn[k]);
        }
        m[r] = mn[k];
        alpha[r] = av[k];
        l[r] *= av[k];
      }
      float32x4_t sum[4];
      for (int k = 0; k < 4; ++k) sum[k] = vdupq_n_f32(0.0f);
      const float32x4_t vmn0 = vdupq_n_f32(-mn[0]);
      const float32x4_t vmn1 = vdupq_n_f32(-mn[1]);
      const float32x4_t vmn2 = vdupq_n_f32(-mn[2]);
      const float32x4_t vmn3 = vdupq_n_f32(-mn[3]);
      const float* __restrict__ p0 = logits + (r0 + 0) * KT;
      const float* __restrict__ p1 = logits + (r0 + 1) * KT;
      const float* __restrict__ p2 = logits + (r0 + 2) * KT;
      const float* __restrict__ p3 = logits + (r0 + 3) * KT;
      float* __restrict__ pt0 = pt + r0;
      if (mn[0] == kNegInf || mn[1] == kNegInf || mn[2] == kNegInf ||
          mn[3] == kNegInf) {
        // A row with no visible key at all: its logits are all -inf and
        // exp2(-inf - -inf) is NaN, so that row's probabilities are exactly 0.
        // Rare (empty or padding rows), so the branch belongs outside the loop
        // -- inside it, it kept four predicated selects and a branch per 4 keys.
        for (int j = 0; j < nkeys; j += 4) {
          float32x4_t e[4];
          e[0] = mn[0] == kNegInf ? zero : exp2v<5>(vfmaq_f32(vmn0, vc, ldf(p0 + j)));
          e[1] = mn[1] == kNegInf ? zero : exp2v<5>(vfmaq_f32(vmn1, vc, ldf(p1 + j)));
          e[2] = mn[2] == kNegInf ? zero : exp2v<5>(vfmaq_f32(vmn2, vc, ldf(p2 + j)));
          e[3] = mn[3] == kNegInf ? zero : exp2v<5>(vfmaq_f32(vmn3, vc, ldf(p3 + j)));
          for (int k = 0; k < 4; ++k) sum[k] = vaddq_f32(sum[k], e[k]);
          tr4x4(e[0], e[1], e[2], e[3]);
          for (int k = 0; k < 4; ++k) stf(pt0 + (j + k) * RQT, e[k]);
        }
      } else {
        int j = 0;
        // Eight keys per step: eight independent exp chains instead of four.
        for (; j + 8 <= nkeys; j += 8) {
          float32x4_t e[4], f[4];
          e[0] = exp2v<5>(vfmaq_f32(vmn0, vc, ldf(p0)));
          e[1] = exp2v<5>(vfmaq_f32(vmn1, vc, ldf(p1)));
          e[2] = exp2v<5>(vfmaq_f32(vmn2, vc, ldf(p2)));
          e[3] = exp2v<5>(vfmaq_f32(vmn3, vc, ldf(p3)));
          f[0] = exp2v<5>(vfmaq_f32(vmn0, vc, ldf(p0 + 4)));
          f[1] = exp2v<5>(vfmaq_f32(vmn1, vc, ldf(p1 + 4)));
          f[2] = exp2v<5>(vfmaq_f32(vmn2, vc, ldf(p2 + 4)));
          f[3] = exp2v<5>(vfmaq_f32(vmn3, vc, ldf(p3 + 4)));
          p0 += 8;
          p1 += 8;
          p2 += 8;
          p3 += 8;
          for (int k = 0; k < 4; ++k) sum[k] = vaddq_f32(sum[k], e[k]);
          for (int k = 0; k < 4; ++k) sum[k] = vaddq_f32(sum[k], f[k]);
          tr4x4(e[0], e[1], e[2], e[3]);
          for (int k = 0; k < 4; ++k) stf(pt0 + (j + k) * RQT, e[k]);
          tr4x4(f[0], f[1], f[2], f[3]);
          for (int k = 0; k < 4; ++k) stf(pt0 + (j + 4 + k) * RQT, f[k]);
        }
        for (; j < nkeys; j += 4) {
          float32x4_t e[4];
          e[0] = exp2v<5>(vfmaq_f32(vmn0, vc, ldf(p0)));
          e[1] = exp2v<5>(vfmaq_f32(vmn1, vc, ldf(p1)));
          e[2] = exp2v<5>(vfmaq_f32(vmn2, vc, ldf(p2)));
          e[3] = exp2v<5>(vfmaq_f32(vmn3, vc, ldf(p3)));
          p0 += 4;
          p1 += 4;
          p2 += 4;
          p3 += 4;
          for (int k = 0; k < 4; ++k) sum[k] = vaddq_f32(sum[k], e[k]);
          tr4x4(e[0], e[1], e[2], e[3]);
          for (int k = 0; k < 4; ++k) stf(pt0 + (j + k) * RQT, e[k]);
        }
      }
      for (int k = 0; k < 4; ++k) l[r0 + k] += hsum(sum[k]);
    }
  }

  static void pv(const DenseJob& job, const char* slab, int64_t b, int64_t kvh,
                 int64_t k0, int nkeys, const float* logits, int nrows,
                 float* acc) {
    const int KT = kt_();
    const float* pt = pof(const_cast<float*>(logits));
    const float* vd =
        reinterpret_cast<const float*>(slab) + pad_of(job.seq_k) * HD;
    if (neon_mode() & 8) return;
    for (int dc = 0; dc < HD; dc += 8) {
      float32x4_t a[RQT][2];
      if (k0 == 0) {
        for (int r = 0; r < RQT; ++r) {
          a[r][0] = vdupq_n_f32(0.0f);
          a[r][1] = vdupq_n_f32(0.0f);
        }
      } else {
        for (int r = 0; r < RQT; ++r) {
          a[r][0] = ldf(acc + r * HD + dc);
          a[r][1] = ldf(acc + r * HD + dc + 4);
        }
      }
      // Pointer walk: the V row of the next key is `HD` floats on and the P
      // panel's next key is `RQT`, so neither the key index nor the (key, dim)
      // product has to be recomputed per key.
      const float* vp = vd + (k0 * HD + dc);
      const float* pp = pt;
      for (int j = 0; j < nkeys; ++j) {
        const float32x4_t v0 = ldf(vp), v1 = ldf(vp + 4);
        const float32x4_t pl = ldf(pp), ph = ldf(pp + 4);
        vp += HD;
        pp += RQT;
        for (int r = 0; r < RQT; ++r) {
          const float32x4_t p = r < 4 ? pl : ph;
          const int lane = r & 3;
          a[r][0] = vfmaq_laneq_f32(a[r][0], v0, p, lane);
          a[r][1] = vfmaq_laneq_f32(a[r][1], v1, p, lane);
        }
      }
      for (int r = 0; r < RQT; ++r) {
        stf(acc + r * HD + dc, a[r][0]);
        stf(acc + r * HD + dc + 4, a[r][1]);
      }
    }
  }
};

// ===========================================================================
// dense bf16 (bfmmla on both phases)
// ===========================================================================
template <int HD, int KTC>
struct DenseBF {
  static constexpr int RQT = 10;
  static constexpr bool zeroes_acc = true;  // see DenseF32
  static void store_bf16(char* dst, const float* src, int hd, float inv) {
    store_bf16_out(dst, src, hd, inv);
  }
  static constexpr int KTMAXV = 2048;
  static constexpr int DG = HD / 4;   // head-dim groups of 4
  static constexpr int RP = RQT / 2;  // row pairs
  // Nearby dimension pairs avoid the long-sequence V stride on D=32/128.
  // D=64 measured slightly slower with blocking, so it keeps the old layout.
  static constexpr bool VBLOCK = HD != 64;
  static constexpr int VKEYS = 64;
  static constexpr int VGROUPS = VKEYS / 4;
  static constexpr int VSTRIDE = VGROUPS * 16;

  static constexpr int kt_() { return KTC; }
  static int& rq_cached() {
    static int v = 64;
    return v;
  }
  static int rq(int64_t seq_q, int) {
    static const int fixed = env_int("CPU_ATTENTION_RQ", 0, 0, 4096);
    const int64_t base =
        seq_q <= 512 ? seq_q : (HD >= 128 ? int64_t(128) : int64_t(64));
    const int64_t want = fixed > 0 ? fixed : base;
    const int64_t capped = std::min<int64_t>(want, std::max<int64_t>(seq_q, 1));
    const int v =
        static_cast<int>(std::max<int64_t>(RQT, (capped / RQT) * RQT));
    rq_cached() = v;
    return v;
  }

  static int acc_rows(int rq_) { return ((rq_ + RQT - 1) / RQT) * RQT; }

  static size_t slab_bytes(int64_t seq_k, int) {
    return static_cast<size_t>(pad_of(seq_k)) * HD * 2 * 2;
  }
  // [logits RQT*kt f32][acc acc_rows(rq)*hd f32][m,l,alpha rq][tmax RQT]
  // [Qpack RQT*HD bf16][P RP*(kt/4)*8 bf16]
  static size_t aux_bytes(int rq_, int) {
    return round64(static_cast<size_t>(RQT) * kt_() * 4 +
                   static_cast<size_t>(acc_rows(rq_)) * HD * 4 +
                   static_cast<size_t>(rq_) * 3 * 4 +
                   static_cast<size_t>(RQT) * 4 + 64 +
                   static_cast<size_t>(RQT) * HD * 2 +
                   static_cast<size_t>(RP) * (kt_() / 4) * 8 * 2);
  }
  static float* tmax_of(float* logits) {
    const int rq_ = rq_cached();
    return logits + static_cast<size_t>(RQT) * kt_() +
           static_cast<size_t>(acc_rows(rq_)) * HD +
           static_cast<size_t>(rq_) * 3;
  }
  static char* qpack_of(float* logits) {
    return reinterpret_cast<char*>(tmax_of(logits) + RQT + 16);
  }
  static char* pof(float* logits) {
    return qpack_of(logits) + static_cast<size_t>(RQT) * HD * 2;
  }

  // K^T blocked every 64 keys: [tile][HD/4][32 key pairs][8], the same two-key
  // by four-dim `bfmmla` operand, with the key dimension in 64-key tiles.
  static inline int64_t kb_grp(int64_t jj, int dg) {
    return (jj >> 6) * (int64_t(DG) * 32) + int64_t(dg) * 32 + ((jj & 63) >> 1);
  }

  // K -> [tile][HD/4][32 pairs][8]: four head dims of two adjacent keys.
  // V -> [tile][HD/2][16][8], or the unblocked D=64 layout: two head dims
  // of four adjacent keys. The key order within a dot product is unchanged.
  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    if (neon_mode() & 1) return;
    const int64_t pad = pad_of(job.seq_k);
    const int64_t n = job.seq_k;
    const char* ks =
        static_cast<const char*>(job.k) + (b * job.kb + kvh * job.kh) * 2;
    const char* vs =
        static_cast<const char*>(job.v) + (b * job.vb + kvh * job.vh) * 2;
    char* kd = slab;
    char* vd = slab + pad * HD * 2;
    const bfloat16x4_t zero = vdup_n_bf16(0);
    // Four keys per step: two adjacent key pairs of one dim group are 16 bytes
    // apart in the packed tile, so their 32 bytes go out as two 16-byte stores
    // instead of four 8-byte ones. `pad` is a multiple of 64 and t0 a multiple
    // of 4, so a step never straddles a K^T tile.
    for (int64_t t0 = 0; t0 < pad; t0 += 4) {
      for (int dg = 0; dg < DG; ++dg) {
        bfloat16x4_t r[4];
        for (int t = 0; t < 4; ++t)
          r[t] = (t0 + t) < n
                     ? ldb4(ks + ((t0 + t) * job.kt + dg * 4) * 2)
                     : zero;
        char* dst = kd + kb_grp(t0, dg) * 16;
        stb(dst, vcombine_bf16(r[0], r[1]));
        stb(dst + 16, vcombine_bf16(r[2], r[3]));
      }
    }
    const int64_t kg_n = pad / 4;
    for (int64_t kg = 0; kg < kg_n; ++kg) {
      const int64_t t0 = 4 * kg;
      for (int d = 0; d < HD; d += 8) {
        uint16x8_t r[4];
        for (int t = 0; t < 4; ++t)
          r[t] = t0 + t < n
                     ? vld1q_u16(reinterpret_cast<const uint16_t*>(
                           vs + ((t0 + t) * job.vt + d) * 2))
                     : vdupq_n_u16(0);
        const uint32x4_t a = vreinterpretq_u32_u16(vzip1q_u16(r[0], r[1]));
        const uint32x4_t b = vreinterpretq_u32_u16(vzip2q_u16(r[0], r[1]));
        const uint32x4_t c = vreinterpretq_u32_u16(vzip1q_u16(r[2], r[3]));
        const uint32x4_t e = vreinterpretq_u32_u16(vzip2q_u16(r[2], r[3]));
        const int64_t stride = VBLOCK ? VSTRIDE : kg_n * 16;
        const int64_t off = VBLOCK
            ? ((t0 / VKEYS) * (HD / 2) + d / 2) * VGROUPS +
                  (t0 % VKEYS) / 4
            : (d / 2) * kg_n + kg;
        char* dst = vd + off * 16;
        vst1q_u32(reinterpret_cast<uint32_t*>(dst), vzip1q_u32(a, c));
        vst1q_u32(reinterpret_cast<uint32_t*>(dst + stride), vzip2q_u32(a, c));
        vst1q_u32(reinterpret_cast<uint32_t*>(dst + 2 * stride), vzip1q_u32(b, e));
        vst1q_u32(reinterpret_cast<uint32_t*>(dst + 3 * stride), vzip2q_u32(b, e));
      }
    }
  }

  static void qk(const DenseJob& job, const char* slab, int64_t b, int64_t kvh,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const int KT = kt_();
    const int64_t pad = pad_of(job.seq_k);
    const int64_t qt = job.qt;
    const char* kd = slab;
    char* qp = qpack_of(logits);
    const bfloat16x4_t zero = vdup_n_bf16(0);
    // This chunk's query rows -> (row pair, dim group of 4). One group is two
    // 8-byte loads and two 8-byte stores, per key *tile* rather than per key.
    for (int rp = 0; rp < RP; ++rp) {
      for (int dg = 0; dg < DG; ++dg) {
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        const bfloat16x4_t lo =
            r0 < nrows ? ldb4(q_base + (r0 * qt + dg * 4) * 2) : zero;
        const bfloat16x4_t hi =
            r1 < nrows ? ldb4(q_base + (r1 * qt + dg * 4) * 2) : zero;
        stb4(qp + (rp * DG + dg) * 16, lo);
        stb4(qp + (rp * DG + dg) * 16 + 8, hi);
      }
    }
    const int64_t kend = k0 + nkeys;
    const int64_t lim = std::min<int64_t>(row_limit0, kend - 1);
    const int32x4_t lane = lane_idx();
    const float32x4_t ninf = vdupq_n_f32(kNegInf);
    // Running maxima per row pair, lanes (r0, r0, r1, r1): five registers
    // instead of ten, so they stay live beside the twenty accumulators rather
    // than round-tripping through the stack every eight keys.
    float32x4_t tmax[RP];
    for (int rp = 0; rp < RP; ++rp) tmax[rp] = ninf;
    for (int j = 0; j < nkeys; j += 8) {
      const int64_t jj = k0 + j;
      float32x4_t acc[RP][4];
      for (int rp = 0; rp < RP; ++rp)
        for (int kp = 0; kp < 4; ++kp) acc[rp][kp] = vdupq_n_f32(0.0f);
      for (int dg = 0; dg < DG; ++dg) {
        bfloat16x8_t a[RP], b[4];
        for (int rp = 0; rp < RP; ++rp) a[rp] = ldb(qp + (rp * DG + dg) * 16);
        const int64_t kg = kb_grp(jj, dg);
        for (int kp = 0; kp < 4; ++kp) b[kp] = ldb(kd + (kg + kp) * 16);
        for (int rp = 0; rp < RP; ++rp)
          for (int kp = 0; kp < 4; ++kp)
            acc[rp][kp] = bfmmla(acc[rp][kp], a[rp], b[kp]);
      }
      auto emit = [&](int rp, float32x4_t ev0, float32x4_t ev1,
                      float32x4_t od0, float32x4_t od1) {
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        stf(logits + r0 * KT + j, ev0);
        stf(logits + r0 * KT + j + 4, ev1);
        stf(logits + r1 * KT + j, od0);
        stf(logits + r1 * KT + j + 4, od1);
        tmax[rp] = vmaxq_f32(
            tmax[rp], vpmaxq_f32(vmaxq_f32(ev0, ev1), vmaxq_f32(od0, od1)));
      };
      if (jj + 7 <= lim) {
        // No key of this block is past any row's limit: the common case, kept
        // free of the per-row-pair mask tests.
        for (int rp = 0; rp < RP; ++rp)
          emit(rp, zip_lo(acc[rp][0], acc[rp][1]),
               zip_lo(acc[rp][2], acc[rp][3]), zip_hi(acc[rp][0], acc[rp][1]),
               zip_hi(acc[rp][2], acc[rp][3]));
        continue;
      }
      const int rsplit =
          jj + 7 <= kend - 1
              ? static_cast<int>(jj + 7 -
                                 std::min<int64_t>(row_limit0, kend - 1))
              : RQT;
      const int32x4_t i0 =
          vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj)), lane);
      const int32x4_t i1 =
          vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj + 4)), lane);
      for (int rp = 0; rp < RP; ++rp) {
        float32x4_t ev0 = zip_lo(acc[rp][0], acc[rp][1]);
        float32x4_t ev1 = zip_lo(acc[rp][2], acc[rp][3]);
        float32x4_t od0 = zip_hi(acc[rp][0], acc[rp][1]);
        float32x4_t od1 = zip_hi(acc[rp][2], acc[rp][3]);
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        if (rsplit > r0) {
          const int32_t l0 = static_cast<int32_t>(
              std::min<int64_t>(row_limit0 + r0, kend - 1));
          ev0 = vbslq_f32(vcgtq_s32(i0, vdupq_n_s32(l0)), ninf, ev0);
          ev1 = vbslq_f32(vcgtq_s32(i1, vdupq_n_s32(l0)), ninf, ev1);
          if (rsplit > r1) {
            const int32_t l1 = static_cast<int32_t>(
                std::min<int64_t>(row_limit0 + r1, kend - 1));
            od0 = vbslq_f32(vcgtq_s32(i0, vdupq_n_s32(l1)), ninf, od0);
            od1 = vbslq_f32(vcgtq_s32(i1, vdupq_n_s32(l1)), ninf, od1);
          }
        }
        emit(rp, ev0, ev1, od0, od1);
      }
    }
    float* tmv = tmax_of(logits);
    for (int rp = 0; rp < RP; ++rp) {
      const float32x4_t m = vpmaxq_f32(tmax[rp], tmax[rp]);
      tmv[2 * rp] = vgetq_lane_f32(m, 0);
      tmv[2 * rp + 1] = vgetq_lane_f32(m, 1);
    }
  }

  // Probabilities straight into the P@V operand layout: [row pair][4 keys] holds
  // (row0's four keys, row1's four keys).
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    // A row pair at a time: the two rows of a P@V operand group are adjacent
    // halves of one 16-byte group, so both are written by a single full store
    // (and the two exp chains interleave).
    const int KT = kt_();
    const int KG = KT / 4;
    char* pt = pof(logits);
    const float* tmv = tmax_of(logits);
    const float32x4_t vc = vdupq_n_f32(c);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    if (neon_mode() & 4) return;
    for (int r0 = 0; r0 < nrows; r0 += 2) {
      const int r1 = r0 + 1 < RQT ? r0 + 1 : r0;
      float mn[2], av[2];
      const int rr[2] = {r0, r1};
      for (int k = 0; k < 2; ++k) {
        const int r = rr[k];
        const float mxv = tmv[r] * c;
        const float mo = m[r];
        if (mxv == kNegInf) {
          mn[k] = mo;
          av[k] = 1.0f;
        } else if (mo == kNegInf) {
          mn[k] = mxv;
          av[k] = 0.0f;
        } else {
          mn[k] = mxv;
          av[k] = std::exp2(mo - mn[k]);
        }
        m[r] = mn[k];
        alpha[r] = av[k];
        l[r] *= av[k];
      }
      char* dst = pt + (r0 / 2) * KG * 16;
      const float* __restrict__ p0 = logits + rr[0] * KT;
      const float* __restrict__ p1 = logits + rr[1] * KT;
      const float32x4_t vmn0 = vdupq_n_f32(-mn[0]);
      const float32x4_t vmn1 = vdupq_n_f32(-mn[1]);
      float32x4_t s0 = zero, s1 = zero;
      // Two loops instead of a `? :` inside the inner one: the test is on the
      // row's running max, i.e. loop invariant, and leaving it in the body made
      // the compiler keep a branch and re-splat the polynomial constants every
      // 4 keys (54 instructions where 40 are the polynomial). The -inf case is
      // a row with no visible key at all -- exp2(-inf - -inf) is NaN, so that
      // row's half of the operand group must be an exact zero.
      if (mn[0] == kNegInf || mn[1] == kNegInf) {
        const bool z0 = mn[0] == kNegInf, z1 = mn[1] == kNegInf;
        for (int j = 0; j < nkeys; j += 4) {
          const float32x4_t e0 =
               z0 ? zero : exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0 + j)));
          const float32x4_t e1 =
               z1 ? zero : exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1 + j)));
          s0 = vaddq_f32(s0, e0);
          s1 = vaddq_f32(s1, e1);
          stb(dst + (j / 4) * 16,
              vcombine_bf16(vcvt_bf16_f32(e0), vcvt_bf16_f32(e1)));
        }
      } else {
        int j = 0;
        // 16 keys per step: the softmax is latency-bound, not issue-bound (the
        // exp is a ~6-link dependent chain and one iteration only keeps four
        // chains in flight, which measures as 1.6 cycles per row-key for twelve
        // instructions' worth of work). Eight chains in flight is what closes
        // that gap.
        for (; j + 16 <= nkeys; j += 16) {
          const float32x4_t c0 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0)));
          const float32x4_t d0 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1)));
          const float32x4_t c1 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0 + 4)));
          const float32x4_t d1 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1 + 4)));
          const float32x4_t c2 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0 + 8)));
          const float32x4_t d2 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1 + 8)));
          const float32x4_t c3 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0 + 12)));
          const float32x4_t d3 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1 + 12)));
          p0 += 16;
          p1 += 16;
          s0 = vaddq_f32(s0, c0);
          s1 = vaddq_f32(s1, d0);
          s0 = vaddq_f32(s0, c1);
          s1 = vaddq_f32(s1, d1);
          s0 = vaddq_f32(s0, c2);
          s1 = vaddq_f32(s1, d2);
          s0 = vaddq_f32(s0, c3);
          s1 = vaddq_f32(s1, d3);
          stb(dst, vcombine_bf16(vcvt_bf16_f32(c0), vcvt_bf16_f32(d0)));
          stb(dst + 16, vcombine_bf16(vcvt_bf16_f32(c1), vcvt_bf16_f32(d1)));
          stb(dst + 32, vcombine_bf16(vcvt_bf16_f32(c2), vcvt_bf16_f32(d2)));
          stb(dst + 48, vcombine_bf16(vcvt_bf16_f32(c3), vcvt_bf16_f32(d3)));
          dst += 64;
        }
        for (; j + 8 <= nkeys; j += 8) {
          const float32x4_t a0 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0)));
          const float32x4_t a1 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1)));
          const float32x4_t b0 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0 + 4)));
          const float32x4_t b1 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1 + 4)));
          p0 += 8;
          p1 += 8;
          s0 = vaddq_f32(s0, a0);
          s1 = vaddq_f32(s1, a1);
          s0 = vaddq_f32(s0, b0);
          s1 = vaddq_f32(s1, b1);
          stb(dst, vcombine_bf16(vcvt_bf16_f32(a0), vcvt_bf16_f32(a1)));
          stb(dst + 16, vcombine_bf16(vcvt_bf16_f32(b0), vcvt_bf16_f32(b1)));
          dst += 32;
        }
        // ceil(nkeys/4) groups in total: `pv` walks the same rounded-up count,
        // and the extra keys carry a ~1e-38 probability because their logits
        // were masked to -inf.
        for (; j < nkeys; j += 4) {
          const float32x4_t a0 = exp2v<2>(vfmaq_f32(vmn0, vc, ldf(p0)));
          const float32x4_t a1 = exp2v<2>(vfmaq_f32(vmn1, vc, ldf(p1)));
          p0 += 4;
          p1 += 4;
          s0 = vaddq_f32(s0, a0);
          s1 = vaddq_f32(s1, a1);
          stb(dst, vcombine_bf16(vcvt_bf16_f32(a0), vcvt_bf16_f32(a1)));
          dst += 16;
        }
      }
      l[rr[0]] += hsum(s0);
      if (rr[1] != rr[0]) l[rr[1]] += hsum(s1);
    }
  }

  static void pv(const DenseJob& job, const char* slab, int64_t b, int64_t kvh,
                 int64_t k0, int nkeys, const float* logits, int nrows,
                 float* acc) {
    const int KG = kt_() / 4;
    const char* pt = pof(const_cast<float*>(logits));
    const int64_t pad = pad_of(job.seq_k);
    const int64_t vs = VBLOCK ? VSTRIDE : (pad / 4) * 16;
    const char* vd = slab + pad * HD * 2;
    if (neon_mode() & 8) return;
    // Eight head dims per pass: the accumulator is 4 row pairs x 4 dim pairs =
    // 16 registers, which leaves room for the operands. Sixteen dims per pass
    // would halve the number of sweeps over V but needs 32 accumulator
    // registers and spills 1.16x slower (measured).
    for (int dc = 0; dc < HD; dc += 8) {
      float32x4_t a[RP][4];
      if (k0 == 0) {
        for (int rp = 0; rp < RP; ++rp)
          for (int i = 0; i < 4; ++i) a[rp][i] = vdupq_n_f32(0.0f);
      } else {
        for (int rp = 0; rp < RP; ++rp) {
          const int r0 = 2 * rp, r1 = 2 * rp + 1;
          for (int i = 0; i < 4; ++i)
            a[rp][i] = vcombine_f32(vld1_f32(acc + r0 * HD + dc + 2 * i),
                                    vld1_f32(acc + r1 * HD + dc + 2 * i));
        }
      }
      // Both operands advance by one 16-byte group per four keys, so the walk is
      // two pointers and an add per group. Recomputing the V address from
      // (dim pair, key group) inside the loop cost 16 of the body's 42
      // instructions -- more than the matmul's own loads.
      const char* p = pt;
      for (int jb = 0; jb < (VBLOCK ? nkeys : 1); jb += VKEYS) {
        const int block_keys = VBLOCK ? std::min(VKEYS, nkeys - jb) : nkeys;
        const char* v = VBLOCK
            ? vd + (((k0 + jb) / VKEYS) * (HD / 2) + dc / 2) * vs
            : vd + int64_t(dc / 2) * vs + (k0 / 4) * 16;
        int j = 0;
        // Keep the output accumulator live across V tiles; only the operand
        // pointers change. Sixteen keys per step let loads run ahead of dots.
        for (; j + 16 <= block_keys; j += 16) {
          for (int h = 0; h < 4; ++h) {
            bfloat16x8_t pa[RP], vb[4];
            for (int rp = 0; rp < RP; ++rp)
              pa[rp] = ldb(p + rp * KG * 16 + h * 16);
            for (int i = 0; i < 4; ++i) vb[i] = ldb(v + i * vs + h * 16);
            for (int rp = 0; rp < RP; ++rp)
              for (int i = 0; i < 4; ++i)
                a[rp][i] = bfmmla(a[rp][i], pa[rp], vb[i]);
          }
          p += 64;
          v += 64;
        }
        for (; j + 8 <= block_keys; j += 8) {
          for (int h = 0; h < 2; ++h) {
            bfloat16x8_t pa[RP], vb[4];
            for (int rp = 0; rp < RP; ++rp)
              pa[rp] = ldb(p + rp * KG * 16 + h * 16);
            for (int i = 0; i < 4; ++i) vb[i] = ldb(v + i * vs + h * 16);
            for (int rp = 0; rp < RP; ++rp)
              for (int i = 0; i < 4; ++i)
                a[rp][i] = bfmmla(a[rp][i], pa[rp], vb[i]);
          }
          p += 32;
          v += 32;
        }
        for (; j < block_keys; j += 4) {
          bfloat16x8_t pa[RP], vb[4];
          for (int rp = 0; rp < RP; ++rp) pa[rp] = ldb(p + rp * KG * 16);
          for (int i = 0; i < 4; ++i) vb[i] = ldb(v + i * vs);
          for (int rp = 0; rp < RP; ++rp)
            for (int i = 0; i < 4; ++i)
              a[rp][i] = bfmmla(a[rp][i], pa[rp], vb[i]);
          p += 16;
          v += 16;
        }
      }
      for (int rp = 0; rp < RP; ++rp) {
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        const float32x4_t e0 = zip_lo(a[rp][0], a[rp][1]);
        const float32x4_t e1 = zip_lo(a[rp][2], a[rp][3]);
        const float32x4_t o0 = zip_hi(a[rp][0], a[rp][1]);
        const float32x4_t o1 = zip_hi(a[rp][2], a[rp][3]);
        stf(acc + r0 * HD + dc, e0);
        stf(acc + r0 * HD + dc + 4, e1);
        stf(acc + r1 * HD + dc, o0);
        stf(acc + r1 * HD + dc + 4, o1);
      }
    }
  }
};

// ===========================================================================
// paged fp32
// ===========================================================================
template <int HD, int PAGE = 0>
struct PagedF32 {
  static constexpr int PT = 16;       // keys per page-resolution unit
  static constexpr int PG = 4;        // units per online-softmax step
  static constexpr int NK = PT * PG;  // keys per softmax step
  static constexpr int DB = HD <= 64 ? HD : 64;  // head dims held in registers

  static size_t aux_bytes(int) {
    return round64(static_cast<size_t>(2) * NK * 4 +
                   static_cast<size_t>(HD) * 4 + 64);
  }

  static int64_t kv_stride_of(const PagedJob& job) {
    return job.kv_stride ? job.kv_stride : job.q_stride;
  }

  static void store_head(const PagedJob& job, int64_t seq, int64_t head) {
    const int64_t page = PAGE ? PAGE : job.page;
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / page, off = slot % page;
    const int64_t stride = page * HD * 4;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    float* kp =
        reinterpret_cast<float*>(static_cast<char*>(job.k_cache) + base);
    float* vp =
        reinterpret_cast<float*>(static_cast<char*>(job.v_cache) + base);
    const int64_t kvst = kv_stride_of(job);
    // The new K/V carries one head per *kv* head (GQA packs several query
    // heads on one), so the source is indexed by `kvh`, not by the query head.
    const float* ks =
        reinterpret_cast<const float*>(job.k) + seq * kvst + kvh * HD;
    const float* vs =
        reinterpret_cast<const float*>(job.v) + seq * kvst + kvh * HD;
    for (int d = 0; d < HD; ++d) kp[d * page + off] = ks[d];
    std::memcpy(vp + off * HD, vs, sizeof(float) * HD);
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char* ws) {
    const int32_t page = PAGE ? PAGE : static_cast<int32_t>(job.page);
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t len = job.lens[seq];
    const int32_t* bt = job.block_table + seq * job.bt_stride;
    const float* q =
        reinterpret_cast<const float*>(job.q) + seq * job.q_stride + head * HD;
    float* op =
        reinterpret_cast<float*>(job.out) + seq * job.out_stride + head * HD;
    float* logits = reinterpret_cast<float*>(ws);
    float* pbuf = logits + NK;
    float* acc = pbuf + NK;
    for (int i = 0; i < HD; ++i) acc[i] = 0.0f;
    if (len <= 0) {
      for (int i = 0; i < HD; ++i) op[i] = 0.0f;
      return;
    }
    const int64_t stride = int64_t(page) * HD * 4;
    const float c = job.scale * kLog2e;
    float m = kNegInf, l = 0.0f;
    const float32x4_t vc = vdupq_n_f32(c);
    const int32x4_t lane = lane_idx();
    int64_t pos = 0;
    while (pos < len) {
      const int64_t in_page = pos % page;
      const int tiles = static_cast<int>(
          std::min<int64_t>(PG, (len - pos + PT - 1) / PT));
      const char* kbase[PG];
      const char* vbase[PG];
      int32_t boff[PG];
      for (int g = 0; g < tiles; ++g) {
        const int64_t tpos = in_page + g * PT;
        const int64_t pg = bt[pos / page + tpos / page];
        boff[g] = static_cast<int32_t>(tpos % page);
        const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
        kbase[g] = static_cast<const char*>(job.k_cache) + base;
        vbase[g] = static_cast<const char*>(job.v_cache) + base;
      }
      // Q@K^T for the whole step: 4 key groups of 4 keys per 16-key unit.
      float32x4_t lg[NK / 4];
      for (int i = 0; i < NK / 4; ++i) lg[i] = vdupq_n_f32(0.0f);
      for (int g = 0; g < tiles; ++g) {
        float32x4_t c[4][4];
        for (int h = 0; h < 4; ++h)
          for (int d = 0; d < 4; ++d) c[h][d] = vdupq_n_f32(0.0f);
        const float* kb = reinterpret_cast<const float*>(kbase[g]) + boff[g];
        for (int d0 = 0; d0 < HD; d0 += 4) {
          const float32x4_t qv = ldf(q + d0);
          for (int h = 0; h < 4; ++h) {
            float32x4_t k0 = ldf(kb + (d0 + 0) * page + 4 * h);
            c[h][0] = vfmaq_laneq_f32(c[h][0], k0, qv, 0);
            k0 = ldf(kb + (d0 + 1) * page + 4 * h);
            c[h][1] = vfmaq_laneq_f32(c[h][1], k0, qv, 1);
            k0 = ldf(kb + (d0 + 2) * page + 4 * h);
            c[h][2] = vfmaq_laneq_f32(c[h][2], k0, qv, 2);
            k0 = ldf(kb + (d0 + 3) * page + 4 * h);
            c[h][3] = vfmaq_laneq_f32(c[h][3], k0, qv, 3);
          }
        }
        for (int h = 0; h < 4; ++h)
          lg[4 * g + h] = vaddq_f32(vaddq_f32(c[h][0], c[h][1]),
                                    vaddq_f32(c[h][2], c[h][3]));
      }
      const int64_t gvalid = std::min<int64_t>(NK, len - pos);
      for (int i = 0; i < NK / 4; ++i) {
        const int64_t i0 = pos + 4 * i;
        if (i0 + 4 <= gvalid) continue;
        const int32x4_t idx =
            vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i0)), lane);
        const uint32x4_t bad =
            vcgtq_s32(idx, vdupq_n_s32(static_cast<int32_t>(len - 1)));
        lg[i] = vbslq_f32(bad, vdupq_n_f32(kNegInf), lg[i]);
      }
      // One online-softmax step for the whole group.
      float32x4_t mxv = vdupq_n_f32(kNegInf);
      for (int i = 0; i < NK / 4; ++i) mxv = vmaxq_f32(mxv, lg[i]);
      const float mx = hmax(mxv) * c;
      float mn, a;
      if (mx == kNegInf) {
        mn = m;
        a = 1.0f;
      } else if (m == kNegInf) {
        mn = mx;
        a = 0.0f;
      } else {
        mn = m > mx ? m : mx;
        a = std::exp2(m - mn);
      }
      m = mn;
      l *= a;
      const float32x4_t vmn = vdupq_n_f32(-mn);
      float32x4_t sum = vdupq_n_f32(0.0f);
      for (int i = 0; i < NK / 4; ++i) {
        const float32x4_t e = exp2v<6>(vfmaq_f32(vmn, vc, lg[i]));
        stf(pbuf + 4 * i, e);
        sum = vaddq_f32(sum, e);
      }
      l += hsum(sum);
      // P@V with the accumulator in registers over a DB-dim block.
      const int64_t nk = std::min<int64_t>(NK, len - pos);
      for (int c0 = 0; c0 < HD; c0 += DB) {
        constexpr int nb = DB / 4;
        float32x4_t ac[nb];
        for (int i = 0; i < nb; ++i) ac[i] = ldf(acc + c0 + 4 * i);
        if (a != 1.0f) {
          const float32x4_t va = vdupq_n_f32(a);
          for (int i = 0; i < nb; ++i) ac[i] = vmulq_f32(ac[i], va);
        }
        for (int g = 0; g < tiles; ++g) {
          const float* vb = reinterpret_cast<const float*>(vbase[g]);
          const int64_t kv = std::min<int64_t>(PT, nk - g * PT);
          const float* pp = pbuf + g * PT;
          for (int64_t t = 0; t < kv; ++t) {
            const float pv = pp[t];
            const float* vrow = vb + (boff[g] + t) * HD + c0;
            for (int i = 0; i < nb; ++i)
              ac[i] = vfmaq_n_f32(ac[i], ldf(vrow + 4 * i), pv);
          }
        }
        for (int i = 0; i < nb; ++i) stf(acc + c0 + 4 * i, ac[i]);
      }
      pos += tiles * PT;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const float32x4_t vi = vdupq_n_f32(inv);
    for (int i = 0; i < HD / 4; ++i)
      stf(op + 4 * i, vmulq_f32(ldf(acc + 4 * i), vi));
  }
};

// ===========================================================================
// paged bf16
// ===========================================================================
template <int HD, int PAGE = 0>
struct PagedBF {
  static constexpr int PT = 16;
  static constexpr int PG = 4;
  static constexpr int NK = PT * PG;
  static constexpr int DB = HD <= 64 ? HD : 64;  // head dims in registers

  static size_t aux_bytes(int) {
    return round64(static_cast<size_t>(NK) * 2 + static_cast<size_t>(HD) * 4 +
                   64);
  }

  static int64_t kv_stride_of(const PagedJob& job) {
    return job.kv_stride ? job.kv_stride : job.q_stride;
  }

  // K -> [hd/2][page][2] (one head-dim pair of one token contiguous), V ->
  // [page/2][hd][2] (one head dim of a key pair contiguous). Both are bfdot
  // operand shapes. The paired V needs `page % 16 == 0`, which is the documented
  // block-size contract.
  static void store_head(const PagedJob& job, int64_t seq, int64_t head) {
    const int64_t page = PAGE ? PAGE : job.page;
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / page, off = slot % page;
    const int64_t stride = page * HD * 2;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    char* kp = static_cast<char*>(job.k_cache) + base;
    char* vp = static_cast<char*>(job.v_cache) + base;
    const int64_t kvst = kv_stride_of(job);
    const char* ks =
        static_cast<const char*>(job.k) + (seq * kvst + kvh * HD) * 2;
    const char* vs =
        static_cast<const char*>(job.v) + (seq * kvst + kvh * HD) * 2;
    for (int dp = 0; dp < HD / 2; ++dp)
      std::memcpy(kp + (dp * page + off) * 4, ks + dp * 4, 4);
    char* vdst = vp + ((off / 2) * HD + 0) * 4 + (off & 1) * 2;
    for (int d = 0; d < HD; ++d) std::memcpy(vdst + d * 4, vs + d * 2, 2);
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char* ws) {
    const int32_t page = PAGE ? PAGE : static_cast<int32_t>(job.page);
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t len = job.lens[seq];
    const int32_t* bt = job.block_table + seq * job.bt_stride;
    const char* q = static_cast<const char*>(job.q) +
                    (seq * job.q_stride + head * HD) * 2;
    char* op = static_cast<char*>(job.out) +
               (seq * job.out_stride + head * HD) * 2;
    char* pbuf = ws;
    float* acc = reinterpret_cast<float*>(ws + round64(NK * 2));
    for (int i = 0; i < HD; ++i) acc[i] = 0.0f;
    if (len <= 0) {
      for (int i = 0; i < HD; ++i) {
        const bf16 z = f32_to_bf16(0.0f);
        std::memcpy(op + i * 2, &z, 2);
      }
      return;
    }
    const int64_t stride = int64_t(page) * HD * 2;
    const float c = job.scale * kLog2e;
    const int32x4_t lane = lane_idx();
    float m = kNegInf, l = 0.0f;
    const float32x4_t vc = vdupq_n_f32(c);
    int64_t pos = 0;
    while (pos < len) {
      const int64_t in_page = pos % page;
      const int tiles = static_cast<int>(
          std::min<int64_t>(PG, (len - pos + PT - 1) / PT));
      const char* kbase[PG];
      const char* vbase[PG];
      int32_t boff[PG];
      for (int g = 0; g < tiles; ++g) {
        const int64_t tpos = in_page + g * PT;
        const int64_t pg = bt[pos / page + tpos / page];
        boff[g] = static_cast<int32_t>(tpos % page);
        const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
        kbase[g] = static_cast<const char*>(job.k_cache) + base;
        vbase[g] = static_cast<const char*>(job.v_cache) + base;
      }
      // Q@K^T: one bfdot retires four keys x two head dims.
      float32x4_t lg[NK / 4];
      for (int i = 0; i < NK / 4; ++i) lg[i] = vdupq_n_f32(0.0f);
      for (int dp = 0; dp < HD / 2; ++dp) {
        const bfloat16x8_t qa = dup_pair(q + dp * 4);
        for (int g = 0; g < tiles; ++g) {
          const char* kb =
              kbase[g] + (static_cast<int64_t>(dp) * page + boff[g]) * 4;
          for (int h = 0; h < 4; ++h)
            lg[4 * g + h] = bfdot(lg[4 * g + h], qa, ldb(kb + h * 16));
        }
      }
      const int64_t gvalid = std::min<int64_t>(NK, len - pos);
      for (int i = 0; i < NK / 4; ++i) {
        const int64_t i0 = pos + 4 * i;
        if (i0 + 4 <= gvalid) continue;
        const int32x4_t idx =
            vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i0)), lane);
        const uint32x4_t bad =
            vcgtq_s32(idx, vdupq_n_s32(static_cast<int32_t>(len - 1)));
        lg[i] = vbslq_f32(bad, vdupq_n_f32(kNegInf), lg[i]);
      }
      float32x4_t mxv = vdupq_n_f32(kNegInf);
      for (int i = 0; i < NK / 4; ++i) mxv = vmaxq_f32(mxv, lg[i]);
      const float mx = hmax(mxv) * c;
      float mn, a;
      if (mx == kNegInf) {
        mn = m;
        a = 1.0f;
      } else if (m == kNegInf) {
        mn = mx;
        a = 0.0f;
      } else {
        mn = m > mx ? m : mx;
        a = std::exp2(m - mn);
      }
      m = mn;
      l *= a;
      const float32x4_t vmn = vdupq_n_f32(-mn);
      float32x4_t sum = vdupq_n_f32(0.0f);
      for (int i = 0; i < NK / 4; ++i) {
        const float32x4_t e = exp2v<3>(vfmaq_f32(vmn, vc, lg[i]));
        stb4(pbuf + 8 * i, vcvt_bf16_f32(e));
        sum = vaddq_f32(sum, e);
      }
      l += hsum(sum);
      // P@V: one bfdot retires four head dims x two keys.
      const int64_t nk = std::min<int64_t>(NK, len - pos);
      for (int c0 = 0; c0 < HD; c0 += DB) {
        constexpr int nb = DB / 4;
        float32x4_t ac[nb];
        for (int i = 0; i < nb; ++i) ac[i] = ldf(acc + c0 + 4 * i);
        if (a != 1.0f) {
          const float32x4_t va = vdupq_n_f32(a);
          for (int i = 0; i < nb; ++i) ac[i] = vmulq_f32(ac[i], va);
        }
        for (int g = 0; g < tiles; ++g) {
          const int64_t kv = std::min<int64_t>(PT, nk - g * PT);
          for (int64_t t = 0; t < kv; t += 2) {
            const int64_t key = boff[g] + t;
            const char* vrow = vbase[g] + (key / 2) * (HD * 2) * 2;
            const bfloat16x8_t pa = dup_pair(pbuf + (g * PT + t) * 2);
            for (int i = 0; i < nb; ++i)
              ac[i] = bfdot(ac[i], pa, ldb(vrow + (c0 + 4 * i) * 4));
          }
        }
        for (int i = 0; i < nb; ++i) stf(acc + c0 + 4 * i, ac[i]);
      }
      pos += tiles * PT;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const float32x4_t vi = vdupq_n_f32(inv);
    for (int i = 0; i < HD / 4; ++i) {
      const float32x4_t o = vmulq_f32(ldf(acc + 4 * i), vi);
      stb4(op + 8 * i, vcvt_bf16_f32(o));
    }
  }
};

}  // namespace

bool available() {
  const CpuFeatures& f = cpu();
  return f.asimd && f.bf16;
}

void dense(const DenseJob& job, int workers) {
  // The k tile is a compile-time constant in the kernel (it is the logits row
  // stride, the probability-panel geometry and every tile offset), so the two
  // shipped sizes get their own instantiation and the choice is one branch.
  // `CPU_ATTENTION_KT` still selects between them (>= 512 means the big tile).
  const int kt_env = env_int("CPU_ATTENTION_KT", 0, 0, DenseF32<128, 256>::KTMAXV);
  const bool big = kt_env ? (kt_env >= 512) : (job.head_dim >= 128);
#define ATTN_DN(DT, HD)                                                    \
  do {                                                                     \
    if (big) dense_run<Dense##DT<HD, 1024>>(job, workers);                  \
    else dense_run<Dense##DT<HD, 256>>(job, workers);                       \
  } while (0)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_DN(F32, 32);
    else if (job.head_dim == 64) ATTN_DN(F32, 64);
    else ATTN_DN(F32, 128);
  } else {
    if (job.head_dim == 32) ATTN_DN(BF, 32);
    else if (job.head_dim == 64) ATTN_DN(BF, 64);
    else ATTN_DN(BF, 128);
  }
#undef ATTN_DN
}

void paged(const PagedJob& job, int workers) {
  // Common pages get constant strides and shifts. Page-16 fp32 stays general
  // at batch one: specializing it measured 5-11% slower there.
#define ATTN_PG(DT, HD) paged_run<Paged##DT<HD>>(job, workers)
#define ATTN_PGS(DT, HD)                                                 \
  do {                                                                  \
    if (job.page == 16 && (job.dtype != kF32 || job.num_seqs > 1))         \
      paged_run<Paged##DT<HD, 16>>(job, workers);                         \
    else if (job.page == 32) paged_run<Paged##DT<HD, 32>>(job, workers);    \
    else ATTN_PG(DT, HD);                                                \
  } while (0)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_PGS(F32, 32);
    else if (job.head_dim == 64) ATTN_PGS(F32, 64);
    else ATTN_PGS(F32, 128);
  } else {
    if (job.head_dim == 32) ATTN_PGS(BF, 32);
    else if (job.head_dim == 64) ATTN_PGS(BF, 64);
    else ATTN_PGS(BF, 128);
  }
#undef ATTN_PG
#undef ATTN_PGS
}

void store(const PagedJob& job, int workers) {
#define ATTN_ST(DT, HD) store_run<Paged##DT<HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_ST(F32, 32);
    else if (job.head_dim == 64) ATTN_ST(F32, 64);
    else ATTN_ST(F32, 128);
  } else {
    if (job.head_dim == 32) ATTN_ST(BF, 32);
    else if (job.head_dim == 64) ATTN_ST(BF, 64);
    else ATTN_ST(BF, 128);
  }
#undef ATTN_ST
}

}  // namespace neon
}  // namespace attn
