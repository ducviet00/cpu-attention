// AMX dense-attention kernel (x86-64 with amx-tile + amx-bf16).
//
// One `tdpbf16ps` accumulates a 16x16 fp32 tile from two bf16 operands, i.e.
// 512 MACs per instruction against AVX-512's 32 for `vdpbf16ps`. That ratio is
// why this path exists: on a Xeon the vector kernel is 1.7-6.8x behind torch
// SDPA on bf16 and the tile kernel is 1.4-2.4x ahead of it.
//
// ## The instruction, exactly as measured
//
// Both operands are 16 rows x 64 bytes and the tile is a 16x16 matrix of
// **32-bit units** (two bf16 each); the dot product is
//
//     C[i][j] += sum_r dot2( A[i][2r..2r+1], B[r][2j..2j+1] )
//
// so A's rows are the M index and B's rows are the *contraction* index, with N
// along B's columns. For Q@K^T that means Q is used straight out of the
// caller's tensor (row = query, 32 head dims contiguous) and K has to be packed
// as `[dim pair][key]`; for P@V, P is row-major bf16 (row = query, 32 keys
// contiguous) and V is packed as `[key pair][output dim]`. `tests/amx_smoke.mojo`
// in the Mojo tree derives the same two layouts from 512 single-bit probes, and
// vLLM's `reshape_and_cache` states them ("head elements packed as quadwords").
//
// ## Work item and blocking
//
// One work item is (batch, q head, 32 query rows) -- the shared driver's `rq` /
// `RQT`. Inside it the key range is walked in super-tiles of `kt` (256 default,
// the best point measured in the Mojo tree on the same box):
//
//   Q@K^T  32 keys at a time (2 K tiles x 2 Q row groups = 4 C tiles), into an
//          fp32 logits buffer: 32 rows x kt floats.
//   softmax one pass over the super-tile, in place, emitting bf16 P (and the
//          hi/lo halves of it for fp32) into a per-thread P buffer.
//   P@V     32 keys at a time, 32 output dims at a time (2 V tiles x 2 P row
//          groups = 4 C tiles) accumulating into the output buffer, so the
//          output is loaded/stored once per super-tile rather than once per 32
//          keys.
//
// Eight tile registers are the hard budget: 2 A + 2 B + 4 C. head_dim 128
// therefore needs four passes over the super-tile in P@V (the C tiles only span
// 32 dims); P is re-read from L1 each pass, V is read once.
//
// ## fp32
//
// AMX has no fp32 product, so fp32 rides the same kernel with every operand
// split into bf16 hi + bf16 lo and three products kept per phase
// (hi*hi + hi*lo + lo*hi, dropping lo*lo at 2^-16):
//
//     x = hi + lo, hi = bf16(x), lo = bf16(x - fp32(hi)),  |lo| <= 2^-8 |x|
//
// The products are exact in bf16xbf16->fp32 and the accumulation is fp32, so
// the result is fp32-accurate at 3x the tile ops -- against AMX's ~11x
// instruction advantage over FMA. Measured max|diff| against a naive fp32
// reference: 3-8e-5 (gate 2e-4).
//
// ## Numerics
//
// Q@K^T is bf16 x bf16 with fp32 accumulation, P is rounded to bf16 once, P@V
// accumulates in fp32, and the row sum divides out at the end (the driver does
// that). `exp2` is the degree-5 Cephes polynomial plus an exponent bit insert,
// clamped at -127 so a fully masked lane is exactly +0 -- a bf16 denormal in an
// AMX tile is not free and `0 * garbage` must be 0.
//
// Apache-2.0; the packed layouts and the 2-2-4 tile pattern follow vLLM's
// csrc/cpu attention (Apache-2.0) and the Mojo kernel in this repo.
#pragma once

#if defined(__x86_64__)

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "cpu_attn_impl.hpp"
#include "cpu_isa.hpp"

namespace attn {
namespace amx {
namespace {

// ---------------------------------------------------------------------------
// tile config
// ---------------------------------------------------------------------------
// palette 1, 16 rows x 64 bytes (32 bf16 or 16 fp32) for every tile: the same
// geometry serves A, B and C in both phases, so one config is loaded per call.
struct alignas(64) TileCfg {
  uint8_t palette;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

constexpr TileCfg make_cfg() {
  TileCfg c{};
  c.palette = 1;
  // Only palette 1's eight tiles (tmm0-7) are configurable; the upper eight
  // colsb/rows entries are reserved and must stay zero or `ldtilecfg` is #GP.
  for (int i = 0; i < 8; ++i) {
    c.colsb[i] = 64;
    c.rows[i] = 16;
  }
  return c;
}

alignas(64) constexpr TileCfg kTileCfg = make_cfg();

// `ldtilecfg` is per *thread*, and the tile state can be dropped behind our back
// (`tilerelease` in the Mojo kernel, which this process also loads, or a
// thread-pool migration). A tile instruction in INIT state is #UD, so the config
// is re-established on entry to every tile-using phase instead of being cached
// in a thread-local flag. ~15 cycles against thousands.
inline void tile_cfg_load() { _tile_loadconfig(&kTileCfg); }

// ---------------------------------------------------------------------------
// vector helpers (the pack and the softmax run on AVX-512, the dots on AMX)
// ---------------------------------------------------------------------------
inline __m512i join_bf16x2(__m256i a, __m256i b) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

inline __m256i cvt_bf16x16(__m512 v) {
  const __m256bh h = _mm512_cvtneps_pbh(v);
  __m256i r;
  std::memcpy(&r, &h, sizeof(r));
  return r;
}

// 16 bf16 -> 16 fp32 (the exponent lands in the high half).
inline __m512 widen_bf16x16(const void* p) {
  const __m256i raw =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

inline __m512 widen_bf16x16(__m256i raw) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

// bf16 pair interleave: (a0,b0,a1,b1,...) out of two 16-lane halves. One
// `vpermt2w`; the 256-bit unpack form is lane-wise and needs a fix-up.
inline __m512i interleave_bf16x2(__m256i a, __m256i b) {
  constexpr uint16_t idx[32] = {0,  32, 1,  33, 2,  34, 3,  35, 4,  36, 5,
                                37, 6,  38, 7,  39, 8,  40, 9,  41, 10, 42,
                                11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
  return _mm512_permutex2var_epi16(_mm512_castsi256_si512(a),
                                   _mm512_loadu_si512(idx),
                                   _mm512_castsi256_si512(b));
}

// 2^x for x in [-127, 127]: degree-5 polynomial on the fraction, exponent by
// insertion. The positive side is reachable because the online softmax may hold
// its running max below the tile max by up to kMaxSlack (the FA4 rule).
inline __m512 exp2v(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-127.0f));
  x = _mm512_min_ps(x, _mm512_set1_ps(127.0f));
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

// 16x16 transpose of 32-bit units, in place over 16 zmm. Four shuffle stages;
// every stage writes back the pair it read, so the peak is 16 + 2 live
// registers rather than 32 (the two-array form cost ~25% of the K pack in the
// Mojo tree). Bit-identical for integer data: the float shuffles are lane
// permutations.
inline void transpose16x16(__m512* r) {
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
  static constexpr bool BF = DT == kBF16;
  static constexpr int ESZ = BF ? 2 : 4;
  // One packed operand slot holds the bf16 tile and, for fp32, its lo half.
  static constexpr int NLO = BF ? 1 : 2;

  // Query rows per work item and per register tile: two 16-row AMX groups.
  static constexpr int RQT = 32;
  // Largest super-tile: the level the driver's logits/acc/P buffers are sized
  // for. `kt` is a runtime knob (kelvin: 128/256/512 all legal).
  static constexpr int KTMAX = 1024;
  // Largest `rq` the per-thread buffers are laid out for (env RQ sweep).
  static constexpr int AROWS = 64;
  // Keys per step in both phases (two 16-key tile columns).
  static constexpr int KSTEP = 32;

  static constexpr int DC32 = HD / 32;  // Q@K^T dim chunks (32 dims each)
  static constexpr int DC16 = HD / 16;  // P@V dim chunks (16 dims each)
  static constexpr int KSLOT = 512 * NLO;  // bf16 per (key block, dim chunk)
  static constexpr int VSLOT = 512 * NLO;

  // aux layout. The driver places logits at 0, then acc/m/l/alpha from the
  // runtime `kt` and `rq`, so everything the policy keeps for itself starts
  // past the *largest* region the driver can use.
  static constexpr size_t LOGITS_MAX = size_t(RQT) * KTMAX * 4;
  static constexpr size_t ACC_MAX = size_t(AROWS) * HD * 4 + AROWS * 3 * 4;
  static constexpr size_t PROB_OFF = LOGITS_MAX + ACC_MAX;
  static constexpr size_t PROB_MAX = size_t(RQT) * KTMAX * NLO * 2;
  static constexpr size_t PANEL_OFF = PROB_OFF + PROB_MAX;
  static constexpr size_t PANEL_MAX = size_t(RQT) * HD * NLO * 2;

  // Running-max slack in log2 units. 1 = the exact max (a rescale whenever the
  // tile max moves at all); 8 = vLLM/FA4's threshold, which keeps the rescale
  // (and the driver's 32*head_dim scalar multiplies) off most super-tiles at the
  // cost of exp2 arguments up to +8 (P up to 256, still finite and still bf16).
  static float max_slack() {
    static const float v = static_cast<float>(
        env_int("CPU_ATTENTION_AMX_SLACK", 1, 1, 16));
    return v;
  }

  static int kt_() {
    static const int k = env_int("CPU_ATTENTION_KT", 256, 32, KTMAX);
    return k & ~31;
  }

  static int rq(int64_t seq_q, int hd) {
    static const int fixed = env_int("CPU_ATTENTION_RQ", 0, 0, 512);
    const int want = fixed > 0 ? fixed : RQT;
    return static_cast<int>(
        std::min<int64_t>(want, std::max<int64_t>(seq_q, 1)));
  }

  static int acc_rows(int rq) { return ((rq + RQT - 1) / RQT) * RQT; }

  static size_t aux_bytes(int rq, int hd) {
    (void)rq;
    (void)hd;
    return PANEL_OFF + PANEL_MAX;
  }

  static inline int64_t pad_of(int64_t seq_k) {
    return (seq_k + KSTEP - 1) & ~int64_t(KSTEP - 1);
  }

  static inline int64_t k_slot(int64_t kb, int64_t dc) {
    return (kb * DC32 + dc) * KSLOT;
  }
  static inline int64_t v_slot(int64_t kb, int64_t dc) {
    return (kb * DC16 + dc) * VSLOT;
  }
  static inline int64_t k_elems(int64_t seq_k) {
    return (pad_of(seq_k) / 16) * DC32 * KSLOT;
  }

  static size_t slab_bytes(int64_t seq_k, int hd) {
    (void)hd;
    const int64_t kd = k_elems(seq_k);
    const int64_t vd = (pad_of(seq_k) / 32) * DC16 * VSLOT;
    return static_cast<size_t>(kd + vd) * 2;
  }

  // -------------------------------------------------------------------------
  // pack: K -> [key block 16][dim chunk 32][dim pair][key]  (1 KB per tile)
  //       V -> [key block 32][dim chunk 16][key pair][dim]   (1 KB per tile)
  // -------------------------------------------------------------------------
  static void pack(const DenseJob& job, int64_t b, int64_t kvh, char* slab) {
    const int64_t seq_k = job.seq_k;
    const int64_t pad = pad_of(seq_k);
    const char* ksrc =
        static_cast<const char*>(job.k) + (b * job.kb + kvh * job.kh) * ESZ;
    const char* vsrc =
        static_cast<const char*>(job.v) + (b * job.vb + kvh * job.vh) * ESZ;
    const int64_t kts = job.kt * ESZ;  // bytes between K tokens
    const int64_t vts = job.vt * ESZ;
    bf16* kd = reinterpret_cast<bf16*>(slab);
    bf16* vd = kd + k_elems(seq_k);

    for (int64_t kb = 0; kb < pad / 16; ++kb) {
      const int64_t t0 = kb * 16;
      for (int dc = 0; dc < DC32; ++dc) {
        bf16* dst = kd + k_slot(kb, dc);
        if (t0 + 16 <= seq_k) {
          // 16 keys x 32 dims: one 16x16 transpose of 32-bit (dim-pair) units.
          __m512 r[16], l[16];
          for (int i = 0; i < 16; ++i) {
            const char* s = ksrc + (t0 + i) * kts + dc * 32 * ESZ;
            if constexpr (BF) {
              r[i] = _mm512_castsi512_ps(_mm512_loadu_si512(s));
            } else {
              const __m512 f0 = _mm512_loadu_ps(s);
              const __m512 f1 = _mm512_loadu_ps(s + 64);
              const __m256i h0 = cvt_bf16x16(f0);
              const __m256i h1 = cvt_bf16x16(f1);
              r[i] = _mm512_castsi512_ps(join_bf16x2(h0, h1));
              l[i] = _mm512_castsi512_ps(join_bf16x2(
                  cvt_bf16x16(_mm512_sub_ps(f0, widen_bf16x16(h0))),
                  cvt_bf16x16(_mm512_sub_ps(f1, widen_bf16x16(h1)))));
            }
          }
          transpose16x16(r);
          for (int i = 0; i < 16; ++i)
            _mm512_storeu_si512(dst + i * 32, _mm512_castps_si512(r[i]));
          if constexpr (!BF) {
            transpose16x16(l);
            for (int i = 0; i < 16; ++i)
              _mm512_storeu_si512(dst + 512 + i * 32,
                                  _mm512_castps_si512(l[i]));
          }
        } else {
          // Tail block: zero the tile, then copy whole dim pairs of the keys
          // that exist (a padded key must never inject a NaN).
          std::memset(dst, 0, size_t(512) * NLO * 2);
          for (int64_t t = t0; t < seq_k; ++t) {
            const char* s = ksrc + t * kts + dc * 32 * ESZ;
            for (int dp = 0; dp < 16; ++dp) {
              if constexpr (BF) {
                std::memcpy(dst + dp * 32 + 2 * (t - t0), s + dp * 4, 4);
              } else {
                float f0, f1;
                std::memcpy(&f0, s + dp * 8, 4);
                std::memcpy(&f1, s + dp * 8 + 4, 4);
                const bf16 h0 = f32_to_bf16(f0), h1 = f32_to_bf16(f1);
                const bf16 q0 = f32_to_bf16(f0 - bf16_to_f32(h0));
                const bf16 q1 = f32_to_bf16(f1 - bf16_to_f32(h1));
                bf16* p = dst + dp * 32 + 2 * (t - t0);
                p[0] = h0;
                p[1] = h1;
                bf16* pl = dst + 512 + dp * 32 + 2 * (t - t0);
                pl[0] = q0;
                pl[1] = q1;
              }
            }
          }
        }
      }
    }

    for (int64_t kb = 0; kb < pad / 32; ++kb) {
      const int64_t t0 = kb * 32;
      for (int dc = 0; dc < DC16; ++dc) {
        bf16* dst = vd + v_slot(kb, dc);
        for (int r = 0; r < 16; ++r) {
          const int64_t ka = t0 + 2 * r, kb2 = ka + 1;
          // One tile row = two adjacent keys of 16 output dims, interleaved.
          __m256i a_hi, c_hi, a_lo, c_lo;
          if (ka < seq_k) {
            const char* s = vsrc + ka * vts + dc * 16 * ESZ;
            if constexpr (BF) {
              a_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
            } else {
              const __m512 f = _mm512_loadu_ps(s);
              a_hi = cvt_bf16x16(f);
              a_lo = cvt_bf16x16(
                  _mm512_sub_ps(f, widen_bf16x16(a_hi)));
            }
          } else {
            a_hi = _mm256_setzero_si256();
            a_lo = _mm256_setzero_si256();
          }
          if (kb2 < seq_k) {
            const char* s = vsrc + kb2 * vts + dc * 16 * ESZ;
            if constexpr (BF) {
              c_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
            } else {
              const __m512 f = _mm512_loadu_ps(s);
              c_hi = cvt_bf16x16(f);
              c_lo = cvt_bf16x16(
                  _mm512_sub_ps(f, widen_bf16x16(c_hi)));
            }
          } else {
            c_hi = _mm256_setzero_si256();
            c_lo = _mm256_setzero_si256();
          }
          _mm512_storeu_si512(dst + r * 32, interleave_bf16x2(a_hi, c_hi));
          if constexpr (!BF)
            _mm512_storeu_si512(dst + 512 + r * 32,
                                interleave_bf16x2(a_lo, c_lo));
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Q@K^T: logits[RQT][kt] = q[RQT] @ K[k0, k0+nkeys)^T, causal-masked
  // -------------------------------------------------------------------------
  static void qk(const DenseJob& job, const char* slab, int64_t, int64_t,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const int kt = kt_();
    const bf16* kbase = reinterpret_cast<const bf16*>(slab);
    char* const extra = reinterpret_cast<char*>(logits);
    bf16* const panel = reinterpret_cast<bf16*>(extra + PANEL_OFF);
    bf16* const panel_lo = panel + RQT * HD;

    // A operand: the caller's Q when the block is full (any stride works, the
    // tile load takes it in bytes), else a zero-padded panel -- a partial block
    // still computes RQT rows and the padding must not reach the logits.
    const char* a_hi = q_base;
    const char* a_lo = nullptr;
    int astride = static_cast<int>(job.qt * ESZ);
    if (!BF || nrows < RQT) {
      for (int r = 0; r < RQT; ++r) {
        const char* s = q_base + static_cast<int64_t>(r) * job.qt * ESZ;
        bf16* d = panel + static_cast<int64_t>(r) * HD;
        bf16* dl = panel_lo + static_cast<int64_t>(r) * HD;
        for (int dc = 0; dc < DC32; ++dc) {
          if (r < nrows) {
            if constexpr (BF) {
              std::memcpy(d + dc * 32, s + dc * 32 * ESZ, 64);
            } else {
              const __m512 f0 = _mm512_loadu_ps(s + dc * 32 * ESZ);
              const __m512 f1 = _mm512_loadu_ps(s + dc * 32 * ESZ + 64);
              const __m256i h0 = cvt_bf16x16(f0);
              const __m256i h1 = cvt_bf16x16(f1);
              _mm512_storeu_si512(
                  d + dc * 32, join_bf16x2(h0, h1));
              _mm512_storeu_si512(
                  dl + dc * 32,
                  join_bf16x2(
                      cvt_bf16x16(_mm512_sub_ps(f0, widen_bf16x16(h0))),
                      cvt_bf16x16(_mm512_sub_ps(f1, widen_bf16x16(h1)))));
            }
          } else {
            std::memset(d + dc * 32, 0, 64);
            if constexpr (!BF) std::memset(dl + dc * 32, 0, 64);
          }
        }
      }
      a_hi = reinterpret_cast<const char*>(panel);
      if constexpr (!BF) a_lo = reinterpret_cast<const char*>(panel_lo);
      astride = HD * 2;
    }

    tile_cfg_load();
    const int64_t lim = std::min<int64_t>(k0 + nkeys, row_limit0 + 1);
    const __m512i lane =
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512 ninf = _mm512_set1_ps(-INFINITY);
    const int nrow = std::min(nrows, RQT);

    for (int j = 0; j < nkeys; j += KSTEP) {
      const int64_t jj = k0 + j;
      const int64_t kb = jj >> 4;
      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);
      for (int dc = 0; dc < DC32; ++dc) {
        const bf16* kp = kbase + k_slot(kb, dc);
        // Key block kb+1 at the same dim chunk is DC32 slots on, not one: the
        // slots are (key block, dim chunk) row-major.
        const bf16* kp1 = kbase + k_slot(kb + 1, dc);
        _tile_loadd(0, a_hi + dc * 64, astride);
        _tile_loadd(1, a_hi + 16 * astride + dc * 64, astride);
        _tile_loadd(2, kp, 64);
        _tile_loadd(3, kp1, 64);
        _tile_dpbf16ps(4, 0, 2);
        _tile_dpbf16ps(5, 0, 3);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);
        if constexpr (!BF) {
          // q_hi @ k_lo, then q_lo @ k_hi (k_hi reloads from L1).
          _tile_loadd(2, kp + 512, 64);
          _tile_loadd(3, kp1 + 512, 64);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
          _tile_loadd(0, a_lo + dc * 64, astride);
          _tile_loadd(1, a_lo + 16 * astride + dc * 64, astride);
          _tile_loadd(2, kp, 64);
          _tile_loadd(3, kp1, 64);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
        }
      }
      float* lp = logits + j;
      _tile_stored(4, lp, kt * 4);
      _tile_stored(5, lp + 16, kt * 4);
      _tile_stored(6, lp + 16 * kt, kt * 4);
      _tile_stored(7, lp + 16 * kt + 16, kt * 4);

      if (jj + KSTEP > lim) {
        // Boundary step: per row, keys past its causal limit and past the key
        // range become -inf. The store above already happened, so a NaN coming
        // out of a padded K tile is overwritten rather than propagated.
        const int kep = static_cast<int>(k0 + nkeys);
        for (int r = 0; r < nrow; ++r) {
          const int rlim = static_cast<int>(row_limit0 + r);
          float* rp = logits + static_cast<int64_t>(r) * kt + j;
          for (int half = 0; half < 2; ++half) {
            const __m512i idx = _mm512_add_epi32(
                _mm512_set1_epi32(static_cast<int>(jj + 16 * half)), lane);
            const __mmask16 bad =
                _mm512_cmpgt_epi32_mask(idx, _mm512_set1_epi32(rlim)) |
                _mm512_cmpge_epi32_mask(idx, _mm512_set1_epi32(kep));
            __m512* pp = reinterpret_cast<__m512*>(rp + 16 * half);
            _mm512_storeu_ps(pp, _mm512_mask_blend_ps(bad, *pp, ninf));
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // online softmax over one kt-key super-tile, in place; P in bf16
  // -------------------------------------------------------------------------
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    const int kt = kt_();
    bf16* const prob =
        reinterpret_cast<bf16*>(reinterpret_cast<char*>(logits) + PROB_OFF);
    bf16* const prob_lo = prob + static_cast<int64_t>(RQT) * KTMAX;
    const __m512 vc = _mm512_set1_ps(c);
    const float slack = max_slack();
    const int ntail = ((nkeys + KSTEP - 1) & ~(KSTEP - 1));

    for (int r = 0; r < nrows; ++r) {
      float* p = logits + static_cast<int64_t>(r) * kt;
      bf16* pp = prob + static_cast<int64_t>(r) * KTMAX;
      bf16* pl = prob_lo + static_cast<int64_t>(r) * KTMAX;
      __m512 mx = _mm512_set1_ps(-INFINITY);
      for (int j = 0; j < nkeys; j += 16) {
        const __m512 s = _mm512_mul_ps(_mm512_loadu_ps(p + j), vc);
        _mm512_storeu_ps(p + j, s);
        mx = _mm512_max_ps(mx, s);
      }
      const float mxv = _mm512_reduce_max_ps(mx);
      const float mo = m[r];
      float mn, a;
      if (mxv == -INFINITY) {
        // Whole tile masked for this row: keep the running max, rescale
        // nothing, and write exact zeros.
        mn = mo;
        a = 1.0f;
      } else if (mo == -INFINITY) {
        mn = mxv;
        a = 0.0f;
      } else if (mxv > mo + slack) {
        mn = mxv;
        a = std::exp2((mo - mn) * c);
      } else {
        mn = mo;
        a = 1.0f;
      }
      m[r] = mn;
      alpha[r] = a;
      l[r] *= a;
      if (mn == -INFINITY) {
        for (int j = 0; j < ntail; j += 16) {
          _mm512_storeu_ps(p + j, _mm512_setzero_ps());
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(pp + j),
                              _mm256_setzero_si256());
          if constexpr (!BF)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pl + j),
                                _mm256_setzero_si256());
        }
        continue;
      }
      const __m512 vmn = _mm512_set1_ps(mn);
      __m512 sum = _mm512_setzero_ps();
      for (int j = 0; j < nkeys; j += 16) {
        const __m512 e = exp2v(_mm512_sub_ps(_mm512_loadu_ps(p + j), vmn));
        sum = _mm512_add_ps(sum, e);
        const __m256i h = cvt_bf16x16(e);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pp + j), h);
        if constexpr (!BF)
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(pl + j),
                              cvt_bf16x16(_mm512_sub_ps(e, widen_bf16x16(h))));
      }
      l[r] += _mm512_reduce_add_ps(sum);
      // P@V reads whole 32-key steps, so the tail lanes of the last step must be
      // exactly zero: V is zero there too, and `NaN * 0` is not 0.
      for (int j = nkeys; j < ntail; ++j) {
        pp[j] = bf16{0};
        if constexpr (!BF) pl[j] = bf16{0};
      }
    }
  }

  // -------------------------------------------------------------------------
  // P@V: acc[RQT][HD] += P[RQT][nkeys] @ V[k0, k0+nkeys)
  // -------------------------------------------------------------------------
  static void pv(const DenseJob& job, const char* slab, int64_t, int64_t,
                 int64_t k0, int nkeys, const float* p, int nrows, float* acc) {
    const bf16* const prob =
        reinterpret_cast<const bf16*>(reinterpret_cast<const char*>(p) + PROB_OFF);
    const bf16* const prob_lo = prob + static_cast<int64_t>(RQT) * KTMAX;
    const bf16* const vbase =
        reinterpret_cast<const bf16*>(slab) + k_elems(job.seq_k);
    (void)nrows;
    tile_cfg_load();
    for (int dc = 0; dc < DC16; dc += 2) {
      float* c0 = acc + dc * 16;
      float* c1 = acc + 16 * HD + dc * 16;
      if (k0 == 0) {
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
      } else {
        _tile_loadd(4, c0, HD * 4);
        _tile_loadd(5, c0 + 16, HD * 4);
        _tile_loadd(6, c1, HD * 4);
        _tile_loadd(7, c1 + 16, HD * 4);
      }
      for (int ks = 0; ks < nkeys; ks += KSTEP) {
        const int64_t kb = (k0 + ks) >> 5;
        const bf16* vp = vbase + v_slot(kb, dc);
        _tile_loadd(0, prob + ks, KTMAX * 2);
        _tile_loadd(1, prob + 16 * KTMAX + ks, KTMAX * 2);
        _tile_loadd(2, vp, 64);
        _tile_loadd(3, vp + VSLOT, 64);
        _tile_dpbf16ps(4, 0, 2);
        _tile_dpbf16ps(5, 0, 3);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);
        if constexpr (!BF) {
          // p_hi @ v_lo, then p_lo @ v_hi
          _tile_loadd(2, vp + 512, 64);
          _tile_loadd(3, vp + VSLOT + 512, 64);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
          _tile_loadd(0, prob_lo + ks, KTMAX * 2);
          _tile_loadd(1, prob_lo + 16 * KTMAX + ks, KTMAX * 2);
          _tile_loadd(2, vp, 64);
          _tile_loadd(3, vp + VSLOT, 64);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
        }
      }
      _tile_stored(4, c0, HD * 4);
      _tile_stored(5, c0 + 16, HD * 4);
      _tile_stored(6, c1, HD * 4);
      _tile_stored(7, c1 + 16, HD * 4);
    }
  }
};

}  // namespace

bool available() {
  const CpuFeatures& f = cpu();
  return f.amx_tile && f.amx_bf16 && amx_init();
}

void dense(const DenseJob& job, int workers) {
#define ATTN_DENSE(DT, HD) dense_run<Dense<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_DENSE(kF32, 32);
    else if (job.head_dim == 64) ATTN_DENSE(kF32, 64);
    else if (job.head_dim == 128) ATTN_DENSE(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_DENSE(kBF16, 32);
    else if (job.head_dim == 64) ATTN_DENSE(kBF16, 64);
    else if (job.head_dim == 128) ATTN_DENSE(kBF16, 128);
  }
#undef ATTN_DENSE
}

}  // namespace amx
}  // namespace attn

#endif  // __x86_64__
