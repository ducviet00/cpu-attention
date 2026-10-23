// NEON kernel (aarch64, armv8.2-a+bf16+i8mm): fp32 and bf16, dense and paged.
//
// Layouts (this policy defines them; `pack`/`store_head` write them):
//
//   dense fp32   K [hd][pad]           V [pad][hd]
//   dense bf16   K [hd/4][pad/2][8]    V [hd/2][pad/4][8]
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
  const float32x4_t n = vrndnq_f32(x);
  const float32x4_t r = vsubq_f32(x, n);
  // Horner with FMA: p = c_hi, then p = c_next + p*r.
  float32x4_t p;
  if constexpr (DEG >= 5) {
    p = vdupq_n_f32(1.3392112e-3f);
    p = vfmaq_f32(vdupq_n_f32(9.6180963e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.5503282e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(2.4022652e-1f), p, r);
  } else {
    p = vdupq_n_f32(5.587586976e-2f);
    p = vfmaq_f32(vdupq_n_f32(2.422964968e-1f), p, r);
  }
  p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  const int32x4_t e =
      vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
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

// ===========================================================================
// dense fp32
// ===========================================================================
template <int HD, int KTC>
struct DenseF32 {
  static constexpr int RQT = 8;
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
  static size_t aux_bytes(int rq_, int) {
    return round64(static_cast<size_t>(RQT) * kt_() * 4 +
                   static_cast<size_t>(acc_rows(rq_)) * HD * 4 +
                   static_cast<size_t>(rq_) * 3 * 4 +
                   static_cast<size_t>(RQT) * 4 + 64 +
                   static_cast<size_t>(kt_()) * RQT * 4);
  }
  static float* tmax_of(float* logits) {
    const int rq_ = rq_cached();
    return logits + static_cast<size_t>(RQT) * kt_() +
           static_cast<size_t>(acc_rows(rq_)) * HD +
           static_cast<size_t>(rq_) * 3;
  }
  static float* pof(float* logits) { return tmax_of(logits) + RQT + 16; }

  // K^T is blocked every 64 keys: `[tile][hd][64]`. One dimension's 64
  // positions are contiguous (256 B at D=64) instead of `pad` floats apart, so
  // a Q@K^T sweep touches HD/16 pages per key tile instead of one page per
  // dimension -- the TLB effect the AVX-512 path measured at 1.17x on the fp32
  // D=128 rows.
  static inline int64_t k_off(int64_t jj, int d) {
    return (jj >> 6) * (int64_t(HD) * 64) + int64_t(d) * 64 + (jj & 63);
  }

  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
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
    const int64_t pad = pad_of(job.seq_k);
    const int64_t qt = job.qt;
    const float* kd = reinterpret_cast<const float*>(slab);
    const int64_t kend = k0 + nkeys;
    const int64_t lim = std::min<int64_t>(row_limit0, kend - 1);
    const int32x4_t lane = lane_idx();
    const float32x4_t ninf = vdupq_n_f32(kNegInf);
    // RQT is a compile-time trip count on purpose: rows past `nrows` read the
    // last real row again (no out-of-bounds load, no NaN) and are stored into
    // the logits tile, which the softmax then ignores. That keeps every
    // accumulator index constant, which is the difference between sixteen
    // registers and sixteen stack slots.
    const int rclamp = nrows - 1;
    float32x4_t tmax[RQT];
    for (int r = 0; r < RQT; ++r) tmax[r] = vdupq_n_f32(kNegInf);
    for (int j = 0; j < nkeys; j += 8) {
      const int64_t jj = k0 + j;
      float32x4_t c[RQT][2];
      for (int r = 0; r < RQT; ++r) {
        c[r][0] = vdupq_n_f32(0.0f);
        c[r][1] = vdupq_n_f32(0.0f);
      }
      for (int d0 = 0; d0 < HD; d0 += 4) {
        float32x4_t kv[8];
        for (int dd = 0; dd < 4; ++dd) {
          kv[2 * dd + 0] = ldf(kd + k_off(jj, d0 + dd));
          kv[2 * dd + 1] = ldf(kd + k_off(jj + 4, d0 + dd));
        }
        for (int r = 0; r < RQT; ++r) {
          const int rr = r < nrows ? r : rclamp;
          const float32x4_t qv = ldf(q_base + (rr * qt + d0) * 4);
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
      // Row r sees keys <= row_limit0 + r, so from row `rsplit` on the panel is
      // wholly visible and only the rows below it need the blend. When the panel
      // runs past the chunk's last key (`kend-1`) the chunk bound is what masks,
      // and that one does not grow with r: every row needs it.
      const bool nomask = jj + 7 <= lim;
      const int rsplit =
          nomask ? 0
                 : (jj + 7 <= kend - 1
                        ? static_cast<int>(jj + 7 -
                                           std::min<int64_t>(row_limit0,
                                                             kend - 1))
                        : RQT);
      if (rsplit <= 0) {
        for (int r = 0; r < RQT; ++r) {
          stf(logits + r * KT + j, c[r][0]);
          stf(logits + r * KT + j + 4, c[r][1]);
          tmax[r] = vmaxq_f32(tmax[r], vmaxq_f32(c[r][0], c[r][1]));
        }
      } else {
        const int32x4_t idx0 =
            vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj)), lane);
        const int32x4_t idx1 =
            vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj + 4)), lane);
        for (int r = 0; r < RQT; ++r) {
          float32x4_t a0 = c[r][0], a1 = c[r][1];
          if (r < rsplit) {
            const int32_t lr = static_cast<int32_t>(std::min<int64_t>(
                row_limit0 + (r < nrows ? r : rclamp), kend - 1));
            const int32x4_t lv = vdupq_n_s32(lr);
            a0 = vbslq_f32(vcgtq_s32(idx0, lv), ninf, a0);
            a1 = vbslq_f32(vcgtq_s32(idx1, lv), ninf, a1);
          }
          stf(logits + r * KT + j, a0);
          stf(logits + r * KT + j + 4, a1);
          tmax[r] = vmaxq_f32(tmax[r], vmaxq_f32(a0, a1));
        }
      }
    }
    float* tm = tmax_of(logits);
    for (int r = 0; r < RQT; ++r) tm[r] = hmax(tmax[r]);
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
          mn[k] = mo > mxv ? mo : mxv;
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
      const float32x4_t zero = vdupq_n_f32(0.0f);
      const float* __restrict__ p0 = logits + (r0 + 0) * KT;
      const float* __restrict__ p1 = logits + (r0 + 1) * KT;
      const float* __restrict__ p2 = logits + (r0 + 2) * KT;
      const float* __restrict__ p3 = logits + (r0 + 3) * KT;
      float* __restrict__ pt0 = pt + r0;
      for (int j = 0; j < nkeys; j += 4) {
        float32x4_t e0 = mn[0] == kNegInf
                             ? zero
                             : exp2v<5>(vfmaq_f32(vmn0, vc, ldf(p0 + j)));
        float32x4_t e1 = mn[1] == kNegInf
                             ? zero
                             : exp2v<5>(vfmaq_f32(vmn1, vc, ldf(p1 + j)));
        float32x4_t e2 = mn[2] == kNegInf
                             ? zero
                             : exp2v<5>(vfmaq_f32(vmn2, vc, ldf(p2 + j)));
        float32x4_t e3 = mn[3] == kNegInf
                             ? zero
                             : exp2v<5>(vfmaq_f32(vmn3, vc, ldf(p3 + j)));
        sum[0] = vaddq_f32(sum[0], e0);
        sum[1] = vaddq_f32(sum[1], e1);
        sum[2] = vaddq_f32(sum[2], e2);
        sum[3] = vaddq_f32(sum[3], e3);
        tr4x4(e0, e1, e2, e3);
        stf(pt0 + (j + 0) * RQT, e0);
        stf(pt0 + (j + 1) * RQT, e1);
        stf(pt0 + (j + 2) * RQT, e2);
        stf(pt0 + (j + 3) * RQT, e3);
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
    for (int dc = 0; dc < HD; dc += 8) {
      float32x4_t a[RQT][2];
      for (int r = 0; r < RQT; ++r) {
        a[r][0] = ldf(acc + r * HD + dc);
        a[r][1] = ldf(acc + r * HD + dc + 4);
      }
      for (int j = 0; j < nkeys; ++j) {
        const float* vp = vd + (k0 + j) * HD + dc;
        const float32x4_t v0 = ldf(vp), v1 = ldf(vp + 4);
        const float32x4_t pl = ldf(pt + j * RQT);
        const float32x4_t ph = ldf(pt + j * RQT + 4);
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
  static constexpr int RQT = 8;
  static constexpr int KTMAXV = 2048;
  static constexpr int DG = HD / 4;   // head-dim groups of 4
  static constexpr int RP = RQT / 2;  // row pairs

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
  // V -> [HD/2][pad/4][8]: two head dims of four adjacent keys.
  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    const int64_t pad = pad_of(job.seq_k);
    const int64_t n = job.seq_k;
    const char* ks =
        static_cast<const char*>(job.k) + (b * job.kb + kvh * job.kh) * 2;
    const char* vs =
        static_cast<const char*>(job.v) + (b * job.vb + kvh * job.vh) * 2;
    char* kd = slab;
    char* vd = slab + pad * HD * 2;
    const bfloat16x4_t zero = vdup_n_bf16(0);
    for (int64_t kp = 0; kp < pad / 2; ++kp) {
      const int64_t t0 = 2 * kp;
      for (int dg = 0; dg < DG; ++dg) {
        const bfloat16x4_t lo =
            t0 < n ? ldb4(ks + (t0 * job.kt + dg * 4) * 2) : zero;
        const bfloat16x4_t hi =
            (t0 + 1) < n ? ldb4(ks + ((t0 + 1) * job.kt + dg * 4) * 2) : zero;
        char* dst = kd + kb_grp(t0, dg) * 16;
        stb4(dst, lo);
        stb4(dst + 8, hi);
      }
    }
    const int64_t kg_n = pad / 4;
    for (int64_t kg = 0; kg < kg_n; ++kg) {
      const int64_t t0 = 4 * kg;
      for (int dp = 0; dp < HD / 2; ++dp) {
        uint32_t w[4] = {0u, 0u, 0u, 0u};
        for (int t = 0; t < 4; ++t)
          if (t0 + t < n)
            std::memcpy(&w[t], vs + ((t0 + t) * job.vt + 2 * dp) * 2, 4);
        // w[t] = (v[t][2dp], v[t][2dp+1]); the group wants the four keys of one
        // head dim contiguous (column-major over the contraction).
        const uint16x8_t vv = vreinterpretq_u16_u32(vld1q_u32(w));
        char* dst = vd + (dp * kg_n + kg) * 16;
        vst1_u16(reinterpret_cast<uint16_t*>(dst),
                 vuzp1_u16(vget_low_u16(vv), vget_high_u16(vv)));
        vst1_u16(reinterpret_cast<uint16_t*>(dst + 8),
                 vuzp2_u16(vget_low_u16(vv), vget_high_u16(vv)));
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
    float32x4_t tmax[RQT];
    for (int r = 0; r < RQT; ++r) tmax[r] = vdupq_n_f32(kNegInf);
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
      const bool nomask = jj + 7 <= lim;
      const int rsplit =
          nomask ? 0
                 : (jj + 7 <= kend - 1
                        ? static_cast<int>(jj + 7 -
                                           std::min<int64_t>(row_limit0,
                                                             kend - 1))
                        : RQT);
      float32x4_t tm[RQT];
      for (int r = 0; r < RQT; ++r) tm[r] = vdupq_n_f32(kNegInf);
      for (int rp = 0; rp < RP; ++rp) {
        float32x4_t ev0 = zip_lo(acc[rp][0], acc[rp][1]);
        float32x4_t ev1 = zip_lo(acc[rp][2], acc[rp][3]);
        float32x4_t od0 = zip_hi(acc[rp][0], acc[rp][1]);
        float32x4_t od1 = zip_hi(acc[rp][2], acc[rp][3]);
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        if (rsplit > r0) {
          const int32_t l0 = static_cast<int32_t>(
              std::min<int64_t>(row_limit0 + r0, kend - 1));
          const int32x4_t i0 =
              vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj)), lane);
          const int32x4_t i1 =
              vaddq_s32(vdupq_n_s32(static_cast<int32_t>(jj + 4)), lane);
          ev0 = vbslq_f32(vcgtq_s32(i0, vdupq_n_s32(l0)), ninf, ev0);
          ev1 = vbslq_f32(vcgtq_s32(i1, vdupq_n_s32(l0)), ninf, ev1);
          if (rsplit > r1) {
            const int32_t l1 = static_cast<int32_t>(
                std::min<int64_t>(row_limit0 + r1, kend - 1));
            od0 = vbslq_f32(vcgtq_s32(i0, vdupq_n_s32(l1)), ninf, od0);
            od1 = vbslq_f32(vcgtq_s32(i1, vdupq_n_s32(l1)), ninf, od1);
          }
        }
        stf(logits + r0 * KT + j, ev0);
        stf(logits + r0 * KT + j + 4, ev1);
        stf(logits + r1 * KT + j, od0);
        stf(logits + r1 * KT + j + 4, od1);
        tm[r0] = vmaxq_f32(tm[r0], vmaxq_f32(ev0, ev1));
        tm[r1] = vmaxq_f32(tm[r1], vmaxq_f32(od0, od1));
      }
      for (int r = 0; r < RQT; ++r) tmax[r] = vmaxq_f32(tmax[r], tm[r]);
    }
    float* tmv = tmax_of(logits);
    for (int r = 0; r < RQT; ++r) tmv[r] = hmax(tmax[r]);
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
          mn[k] = mo > mxv ? mo : mxv;
          av[k] = std::exp2(mo - mn[k]);
        }
        m[r] = mn[k];
        alpha[r] = av[k];
        l[r] *= av[k];
      }
      char* dst = pt + (r0 / 2) * KG * 8 * 2;
      const float* __restrict__ p0 = logits + rr[0] * KT;
      const float* __restrict__ p1 = logits + rr[1] * KT;
      const float32x4_t vmn0 = vdupq_n_f32(-mn[0]);
      const float32x4_t vmn1 = vdupq_n_f32(-mn[1]);
      const float32x4_t zero = vdupq_n_f32(0.0f);
      float32x4_t s0 = zero, s1 = zero;
      for (int j = 0; j < nkeys; j += 4) {
        const float32x4_t e0 =
            mn[0] == kNegInf ? zero : exp2v<3>(vfmaq_f32(vmn0, vc, ldf(p0 + j)));
        const float32x4_t e1 =
            mn[1] == kNegInf ? zero : exp2v<3>(vfmaq_f32(vmn1, vc, ldf(p1 + j)));
        s0 = vaddq_f32(s0, e0);
        s1 = vaddq_f32(s1, e1);
        stb(dst + (j / 4) * 16,
            vcombine_bf16(vcvt_bf16_f32(e0), vcvt_bf16_f32(e1)));
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
    const int64_t kgn = pad / 4;
    const char* vd = slab + pad * HD * 2;
    // Eight head dims per pass: the accumulator is 4 row pairs x 4 dim pairs =
    // 16 registers, which leaves room for the operands. Sixteen dims per pass
    // would halve the number of sweeps over V but needs 32 accumulator
    // registers and spills 1.16x slower (measured).
    for (int dc = 0; dc < HD; dc += 8) {
      float32x4_t a[RP][4];
      for (int rp = 0; rp < RP; ++rp) {
        const int r0 = 2 * rp, r1 = 2 * rp + 1;
        for (int i = 0; i < 4; ++i)
          a[rp][i] = vcombine_f32(vld1_f32(acc + r0 * HD + dc + 2 * i),
                                  vld1_f32(acc + r1 * HD + dc + 2 * i));
      }
      for (int j = 0; j < nkeys; j += 4) {
        // The P panel is per key tile (the softmax never sees k0); the V slab
        // is indexed by the global key group.
        const int64_t kgp = j / 4, kgv = (k0 + j) / 4;
        bfloat16x8_t pa[RP];
        for (int rp = 0; rp < RP; ++rp) pa[rp] = ldb(pt + (rp * KG + kgp) * 16);
        bfloat16x8_t vb[4];
        for (int i = 0; i < 4; ++i)
          vb[i] = ldb(vd + ((dc / 2 + i) * kgn + kgv) * 16);
        for (int rp = 0; rp < RP; ++rp)
          for (int i = 0; i < 4; ++i)
            a[rp][i] = bfmmla(a[rp][i], pa[rp], vb[i]);
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
template <int HD>
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
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / job.page, off = slot % job.page;
    const int64_t stride = job.page * HD * 4;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    float* kp =
        reinterpret_cast<float*>(static_cast<char*>(job.k_cache) + base);
    float* vp =
        reinterpret_cast<float*>(static_cast<char*>(job.v_cache) + base);
    const int64_t kvst = kv_stride_of(job);
    const float* ks =
        reinterpret_cast<const float*>(job.k) + seq * kvst + head * HD;
    const float* vs =
        reinterpret_cast<const float*>(job.v) + seq * kvst + head * HD;
    for (int d = 0; d < HD; ++d) kp[d * job.page + off] = ks[d];
    std::memcpy(vp + off * HD, vs, sizeof(float) * HD);
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char* ws) {
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
    const int64_t stride = job.page * HD * 4;
    const float c = job.scale * kLog2e;
    const int32_t page = static_cast<int32_t>(job.page);
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
      for (int d0 = 0; d0 < HD; d0 += 4) {
        const float32x4_t qv = ldf(q + d0);
        for (int g = 0; g < tiles; ++g) {
          const float* kb = reinterpret_cast<const float*>(kbase[g]) + boff[g];
          for (int h = 0; h < 4; ++h) {
            float32x4_t k0 = ldf(kb + (d0 + 0) * job.page + 4 * h);
            lg[0 + 4 * g + h] = vfmaq_laneq_f32(lg[0 + 4 * g + h], k0, qv, 0);
            k0 = ldf(kb + (d0 + 1) * job.page + 4 * h);
            lg[0 + 4 * g + h] = vfmaq_laneq_f32(lg[0 + 4 * g + h], k0, qv, 1);
            k0 = ldf(kb + (d0 + 2) * job.page + 4 * h);
            lg[0 + 4 * g + h] = vfmaq_laneq_f32(lg[0 + 4 * g + h], k0, qv, 2);
            k0 = ldf(kb + (d0 + 3) * job.page + 4 * h);
            lg[0 + 4 * g + h] = vfmaq_laneq_f32(lg[0 + 4 * g + h], k0, qv, 3);
          }
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
        const float32x4_t e = exp2v<5>(vfmaq_f32(vmn, vc, lg[i]));
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
template <int HD>
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
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / job.page, off = slot % job.page;
    const int64_t stride = job.page * HD * 2;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    char* kp = static_cast<char*>(job.k_cache) + base;
    char* vp = static_cast<char*>(job.v_cache) + base;
    const int64_t kvst = kv_stride_of(job);
    const char* ks =
        static_cast<const char*>(job.k) + (seq * kvst + head * HD) * 2;
    const char* vs =
        static_cast<const char*>(job.v) + (seq * kvst + head * HD) * 2;
    for (int dp = 0; dp < HD / 2; ++dp)
      std::memcpy(kp + (dp * job.page + off) * 4, ks + dp * 4, 4);
    char* vdst = vp + ((off / 2) * HD + 0) * 4 + (off & 1) * 2;
    for (int d = 0; d < HD; ++d) std::memcpy(vdst + d * 4, vs + d * 2, 2);
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char* ws) {
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
    const int64_t stride = job.page * HD * 2;
    const float c = job.scale * kLog2e;
    const int32_t page = static_cast<int32_t>(job.page);
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
              kbase[g] + (static_cast<int64_t>(dp) * job.page + boff[g]) * 4;
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
#define ATTN_PG(DT, HD) paged_run<Paged##DT<HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_PG(F32, 32);
    else if (job.head_dim == 64) ATTN_PG(F32, 64);
    else ATTN_PG(F32, 128);
  } else {
    if (job.head_dim == 32) ATTN_PG(BF, 32);
    else if (job.head_dim == 64) ATTN_PG(BF, 64);
    else ATTN_PG(BF, 128);
  }
#undef ATTN_PG
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
