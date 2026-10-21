// Dense attention on Intel AMX (bf16) -- a second implementation of
// `flash_attn_func`'s kernel, in C, called from `cpu_attention/_kernel.py`
// through ctypes when the CPU has AMX and the dtype is bf16.
//
// ## Why this exists
//
// Measured on the m8i box (Xeon 6975P-C), per (query row, key) pair at D=128:
//   AVX-512 `vdpbf16ps`   16.0 cycles   (32 flops/cycle -- same as fp32 FMA)
//   AMX bf16               1.4 cycles   (372 flops/cycle; oneDNN bf16 GEMM is
//                                        1.45 TFLOP/s/core, `amx_probe.py`)
//   vLLM's AMX kernel      3.0 cycles   (1.4 matmul + ~1.6 softmax)
// so an AMX kernel is softmax-bound, not matmul-bound, and the design below
// spends its effort on the softmax and on keeping the AMX tiles fed.
//
// ## Structure (vLLM's `TileGemm224`, re-derived)
//
// One work item = 32 consecutive query rows of one (batch, head), swept over the
// whole key range. 32 rows because `tdpbf16ps` gives a 16x16 fp32 tile and 4
// accumulators (2 row groups x 2 key groups) is what fits beside the operand
// tiles in AMX's 8-tile file.
//
//   QK^T: A = Q (16 rows x 32 head dims, read straight from the caller's Q with
//         tileloadd(stride = q_stride_t*2) -- no Q packing), B = K packed by
//         `pack_k_block`.
//   PV:   A = P (16 rows x 32 keys, row-major), B = V packed by `pack_v_block`.
//
// Two-level key blocking: QK runs in 32-key chunks (4 accumulators = 32 rows x
// 32 keys) into an fp32 logits buffer, then the online softmax runs once over
// the whole `kt`-key super-tile, then P@V sweeps the super-tile. That amortises
// the output accumulator load/store over `kt` keys instead of 32.
//
// ## The tile config (constant for the whole kernel, set once per thread)
//
//   tiles 0,1  A   (QK: Q rows 0-15, 16-31; PV: P rows 0-15, 16-31)
//   tiles 2,3  B   (QK: keys 0-15, 16-31; PV: dims 0-15, 16-31)
//   tiles 4-7  C   (rowgroup x {B0,B1})
// all `rows = 16`, `colsb = 64` (32 bf16 or 16 fp32).
//
// `_tilecfg` field offsets that matter: palette at byte 0, `colsb[8]` is
// **uint16** at offset 16, `rows[8]` is uint8 at offset 48.
//
// AMX permission: `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` once per
// process, or every tile instruction is SIGILL.
//
// ## Numerics
//
// Same class as the AVX-512 bf16 path it replaces: Q@K^T is bf16 x bf16 with
// fp32 accumulate (`vdpbf16ps`'s arithmetic), P is rounded to bf16 once, P@V
// accumulates in fp32. `scale` multiplies the logits, P is unnormalised and the
// output is divided by the row sum at the end, as flash-attn does. `exp` is a
// degree-4 `exp2` polynomial; masked lanes become exactly 0 because the
// exponent clamps below the subnormal boundary.
//
// Build: `scripts/build.py` (gcc -O3 -mamx-tile -mamx-bf16 -mavx512bf16).
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef uint16_t bf16;

#define QROWS 32  /* query rows per work item (= 2 AMX row groups) */
#define KEYCH 32  /* keys per QK step (4 accumulator tiles) */
#define MAXKT 512 /* upper bound on the super-tile */

/* Softmax A/B switch (build-time so both variants cost nothing at run time).
 *
 * MEASURED, in one process, 2 threads, medians of 5 (`scripts/prof/x86/amx_ab.py`
 * over three builds, kt=256): 1 is the winner and 2/0 are 3-40% *slower*.
 *   1 = bit-insert exp + one row at a time
 *   2 = vscalefps exp + two rows in flight   (10-30% slower at short S)
 *   0 = bit-insert exp + two rows in flight  (10-40% slower)
 * Two rows in flight is slower because four `exp2` chains plus four sum
 * accumulators exceed the vector register file and spill; `vscalefps` for the
 * 2^n scale is neutral against the integer bit-insert (m2 vs m0). */
#ifndef AMX_SM
#define AMX_SM 1
#endif

static const float kLog2e = 1.4426950408889634f;
static const float kNegInf = -3.4028234663852886e38f;

static int g_amx = 0; /* 0 unknown, 1 usable, -1 not usable */

static int amx_init(void) {
  if (g_amx) return g_amx > 0 ? 0 : -1;
  long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  g_amx = (r == 0) ? 1 : -1;
  return g_amx > 0 ? 0 : -1;
}

static void tile_setup(void) {
  uint8_t cfg[64] __attribute__((aligned(64)));
  memset(cfg, 0, sizeof cfg);
  cfg[0] = 1; /* palette */
  uint16_t *colsb = (uint16_t *)(cfg + 16);
  for (int i = 0; i < 8; i++) {
    colsb[i] = 64;
    cfg[48 + i] = 16;
  }
  _tile_loadconfig(cfg);
}

/* ------------------------------------------------------------------ softmax */

/* 2^x, and *exactly* +0 once x is far enough below zero.
 *
 * Degree-4 polynomial on [-0.5, 0.5] and `vscalefps` for the 2^n scale (one
 * instruction, vs the four of a float-to-int + shift + bit-insert). The clamp
 * at -150 relies on 2^-150 underflowing fp32 to +0: a fully masked tile then
 * contributes a probability of exactly 0, which P@V needs (0 * garbage must be
 * 0, and a bf16 denormal in an AMX tile is not free). Relative error 4e-5,
 * two orders below the bf16 probabilities this path already rounds to. */
static inline __m512 exp2_ps(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(AMX_SM == 1 ? -127.0f : -150.0f));
  x = _mm512_min_ps(x, _mm512_set1_ps(127.0f));
  __m512 n =
      _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 f = _mm512_sub_ps(x, n);
  __m512 p = _mm512_set1_ps(0.009618129107628477f);
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.05550410866482158f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.2402265069591007f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.6931471805599453f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
#if AMX_SM == 1
  __m512i e = _mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
#else
  return _mm512_scalef_ps(p, n);
#endif
}

/* ------------------------------------------------------- K pack (transpose) */

/* The exact `tdpbf16ps` B-operand mapping, recovered by single-bit probes
 * (`/tmp`-style script over `dense_amx_debug_qk_raw`; the table below is the
 * measured result, all 512 positions clean):
 *
 *   bf16 element (tile row r, index t) of a 1 KB B tile holds
 *       B[2r + (t & 1)][t >> 1]
 * i.e. in 32-bit units: unit (r, u) = the dim pair (2r, 2r+1) column u. So the
 * K pack is a plain 16x16 **32-bit** transpose ([dim pair][key]) with the pair
 * interleaved inside the unit -- the same conclusion vLLM's
 * `reshape_and_cache` states ("head elements should be packed as quadwords").
 * A 16-bit-granularity transpose would be wrong; that cost me an hour. */
static void tp16x16_epi32(__m512i *r) {
  __m512i y[16];
  for (int a = 0; a < 4; a++) {
    const __m512i x0 = r[4 * a + 0], x1 = r[4 * a + 1];
    const __m512i x2 = r[4 * a + 2], x3 = r[4 * a + 3];
    __m512i t0 = _mm512_unpacklo_epi32(x0, x1);
    __m512i t1 = _mm512_unpackhi_epi32(x0, x1);
    __m512i t2 = _mm512_unpacklo_epi32(x2, x3);
    __m512i t3 = _mm512_unpackhi_epi32(x2, x3);
    y[4 * a + 0] = _mm512_unpacklo_epi64(t0, t2);
    y[4 * a + 1] = _mm512_unpackhi_epi64(t0, t2);
    y[4 * a + 2] = _mm512_unpacklo_epi64(t1, t3);
    y[4 * a + 3] = _mm512_unpackhi_epi64(t1, t3);
  }
  /* y[4a+j] lane L holds column 4L+j of row group a; output row 4L+j needs lane
   * a of y[4a+j] in lane position a. */
  const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15);
  __m512i rotidx[4];
  for (int s = 0; s < 4; s++)
    rotidx[s] = _mm512_and_si512(
        _mm512_add_epi32(iota, _mm512_set1_epi32(4 * s)), _mm512_set1_epi32(15));
  __m512i out[16];
  for (int L = 0; L < 4; L++) {
    for (int j = 0; j < 4; j++) {
      __m512i g = _mm512_setzero_si512();
      for (int a = 0; a < 4; a++) {
        __m512i rot =
            _mm512_permutexvar_epi32(rotidx[(L - a + 4) & 3], y[4 * a + j]);
        g = _mm512_mask_mov_epi32(g, (__mmask16)(0xF << (4 * a)), rot);
      }
      out[4 * L + j] = g;
    }
  }
  for (int i = 0; i < 16; i++) r[i] = out[i];
}

/* -------------------------------------------------------------- pack helpers */

typedef struct {
  int64_t seq_k;
  int64_t nkv;
  int64_t ksb, kst, ksh, vsb, vst, vsh;
  const bf16 *k, *v;
  bf16 *kp, *vp;
  int64_t kb16, kb32; /* key blocks, padded to a multiple of 32 keys */
  int64_t dc32, dc16; /* D/32, D/16 */
  int64_t kp_bh, vp_bh;
} PackJob;

/* K -> the QK B-operand layout.
 *
 * For (16-key block, 32-dim block) a 1 KB tile whose row r is
 * [K[k0+0..k0+15][d0+2r], K[k0+0..k0+15][d0+2r+1]]: the contraction index runs
 * over head dims (32 = 16 tile rows x 2) and N = the 16 keys. That is the
 * 32-bit transpose of the 16 (key) x 16 (dim-pair) block.
 * Keys past `seq_k` are zero, and so is V there, so a stale P cannot inject a
 * NaN through a padded key. */
static void pack_k_block(const PackJob *pj, int64_t b, int64_t h, int64_t kb,
                         int64_t dc) {
  const bf16 *src = pj->k + b * pj->ksb + h * pj->ksh;
  bf16 *dst =
      pj->kp + (b * pj->nkv + h) * pj->kp_bh + (kb * pj->dc32 + dc) * 512;
  const int64_t k0 = kb * 16, d0 = dc * 32;
  __m512i r[16];
  for (int i = 0; i < 16; i++) {
    const int64_t key = k0 + i;
    if (key < pj->seq_k)
      r[i] = _mm512_loadu_si512((const void *)(src + key * pj->kst + d0));
    else
      r[i] = _mm512_setzero_si512();
  }
  tp16x16_epi32(r);
  for (int i = 0; i < 16; i++)
    _mm512_storeu_si512((void *)(dst + i * 32), r[i]);
}

/* V -> the PV B-operand layout.
 *
 * B is 32 keys (contraction) x 16 output dims (N). By the measured mapping
 * above, element (tile row r, index t) must hold B[2r + (t&1)][t>>1], i.e. for
 * each dim u the two keys' values interleaved as a 32-bit unit:
 *   dst[r*32 + 2u]     = V[k0+2r][d0+u]
 *   dst[r*32 + 2u + 1] = V[k0+2r+1][d0+u]
 * Two 32-byte loads and four shuffles per tile row. */
static void pack_v_block(const PackJob *pj, int64_t b, int64_t h, int64_t kb,
                         int64_t dc) {
  const bf16 *src = pj->v + b * pj->vsb + h * pj->vsh;
  bf16 *dst =
      pj->vp + (b * pj->nkv + h) * pj->vp_bh + (kb * pj->dc16 + dc) * 512;
  const int64_t k0 = kb * 32, d0 = dc * 16;
  for (int r = 0; r < 16; r++) {
    const int64_t ka = k0 + 2 * r, kc = ka + 1;
    __m256i a = (ka < pj->seq_k)
                    ? _mm256_loadu_si256(
                          (const __m256i *)(src + ka * pj->vst + d0))
                    : _mm256_setzero_si256();
    __m256i c = (kc < pj->seq_k)
                    ? _mm256_loadu_si256(
                          (const __m256i *)(src + kc * pj->vst + d0))
                    : _mm256_setzero_si256();
    __m256i lo = _mm256_unpacklo_epi16(a, c);
    __m256i hi = _mm256_unpackhi_epi16(a, c);
    _mm256_storeu_si256((__m256i *)(dst + r * 32),
                        _mm256_permute2x128_si256(lo, hi, 0x20));
    _mm256_storeu_si256((__m256i *)(dst + r * 32 + 16),
                        _mm256_permute2x128_si256(lo, hi, 0x31));
  }
}

/* One task = one (batch, kv head, 16-key block): every dim chunk of that slab.
 * Doing all dim chunks inside a task turns the source reads into one linear run
 * over `16 * head_dim` elements instead of `64 bytes at stride head_dim*2` per
 * task, which measured latency-bound at short seq_k. */
static void pack_k_slab(const PackJob *pj, int64_t b, int64_t h, int64_t kb) {
  for (int64_t dc = 0; dc < pj->dc32; dc++) pack_k_block(pj, b, h, kb, dc);
}

static void pack_v_slab(const PackJob *pj, int64_t b, int64_t h, int64_t kb) {
  for (int64_t dc = 0; dc < pj->dc16; dc++) pack_v_block(pj, b, h, kb, dc);
}

/* ---------------------------------------------------------------- main loop */

typedef struct {
  const bf16 *q;
  bf16 *out;
  int64_t seq_q, seq_k, nh, nkv, D;
  int64_t qb, qt, qh, ob, ot, oh;
  float scale;
  int causal;
  const bf16 *kp, *vp;
  int64_t dc32, dc16;
  int64_t kp_bh, vp_bh;
  int64_t nqg;
  int kt;
  int ablate; /* profiling only: 1 = no QK, 2 = no softmax, 4 = no P@V */
} Ctx;

typedef struct {
  float *logits; /* [QROWS][kt] */
  bf16 *prob;    /* [QROWS][kt] */
  float *acc;    /* [QROWS][D] */
  bf16 *qpad;    /* [QROWS][D], only for a partial last row block */
} Scratch;

static void dense_item(const Ctx *c, Scratch *s, int64_t b, int64_t h,
                       int64_t q0) {
  const int D = (int)c->D;
  const int kt = c->kt;
  const int64_t seq_k = c->seq_k;
  const int64_t nrows = (c->seq_q - q0 < QROWS) ? (c->seq_q - q0) : QROWS;
  const int64_t diag = seq_k - c->seq_q;
  const int64_t kvh = (c->nh == c->nkv) ? h : h / (c->nh / c->nkv);
  const bf16 *kbase = c->kp + (b * c->nkv + kvh) * c->kp_bh;
  const bf16 *vbase = c->vp + (b * c->nkv + kvh) * c->vp_bh;
  bf16 *obase = c->out + b * c->ob + h * c->oh;

  const bf16 *qptr;
  int64_t qstride; /* bytes between query rows */
  if (nrows == QROWS) {
    qptr = c->q + b * c->qb + q0 * c->qt + h * c->qh;
    qstride = c->qt * 2;
  } else {
    memset(s->qpad, 0, (size_t)QROWS * D * sizeof(bf16));
    for (int64_t r = 0; r < nrows; r++)
      memcpy(s->qpad + r * D, c->q + b * c->qb + (q0 + r) * c->qt + h * c->qh,
             (size_t)D * sizeof(bf16));
    qptr = s->qpad;
    qstride = (int64_t)D * 2;
  }
  const bf16 *qptr1 = (const bf16 *)((const char *)qptr + 16 * qstride);

  float *logits = s->logits;
  bf16 *prob = s->prob;
  float *acc = s->acc;
  const int64_t lstride = (int64_t)kt * 4;
  const int64_t pstride = (int64_t)kt * 2;
  const int64_t astride = (int64_t)D * 4;

  /* Running max / sum per row: two 16-lane vectors (rows 0-15, 16-31). The max
   * starts at 0 rather than -inf -- the scaled logits are O(10), so exp2 cannot
   * overflow, and a row that sees no key at all has all logits at -3e38, which
   * clamps to p = 0 and is written out as zeros. */
  __m512 m0 = _mm512_setzero_ps(), m1 = _mm512_setzero_ps();
  __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
  memset(acc, 0, (size_t)QROWS * D * sizeof(float));

  int64_t loop_end = seq_k;
  if (c->causal) {
    int64_t e = q0 + nrows + diag;
    if (e < loop_end) loop_end = e;
  }
  if (loop_end < 0) loop_end = 0;

  const __m512 c1v = _mm512_set1_ps(c->scale * kLog2e);
  const float c1f = c->scale * kLog2e;
  const __m512 ninf = _mm512_set1_ps(kNegInf);
  const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15);

  for (int64_t t0 = 0; t0 < loop_end; t0 += kt) {
    const int nk = (int)((loop_end - t0 < kt) ? (loop_end - t0) : kt);
    const int nch = (nk + KEYCH - 1) / KEYCH;
    /* Lanes the QK actually wrote (the tile is 32 keys wide). The mask, the
     * softmax and the P buffer only ever need these: PV reads ceil(nk/32)
     * blocks. Running the softmax over all of `kt` instead wastes (kt-nk) lanes
     * on every partial super-tile, which is most of them at short seq_k. */
    const int nkp = nch * KEYCH;

    /* ---- Q@K^T: KEYCH keys at a time, 32 rows -> fp32 logits ---- */
    for (int ch = 0; !(c->ablate & 1) && ch < nch; ch++) {
      const int64_t k0 = t0 + ch * KEYCH;
      const int64_t kb = (k0 / 16) * c->dc32 * 512;
      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);
      for (int dc = 0; dc < D / 32; dc++) {
        _tile_loadd(0, (const void *)(qptr + dc * 32), qstride);
        _tile_loadd(1, (const void *)(qptr1 + dc * 32), qstride);
        _tile_loadd(2, (const void *)(kbase + kb + dc * 512), 64);
        _tile_loadd(3, (const void *)(kbase + kb + dc * 512 + c->dc32 * 512), 64);
        _tile_dpbf16ps(4, 0, 2);
        _tile_dpbf16ps(5, 0, 3);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);
      }
      float *lg = logits + ch * KEYCH;
      _tile_stored(4, (void *)(lg), lstride);
      _tile_stored(5, (void *)(lg + 16), lstride);
      _tile_stored(6, (void *)(lg + 16 * kt), lstride);
      _tile_stored(7, (void *)(lg + 16 * kt + 16), lstride);
    }

    /* ---- causal / padding mask, per row, only on a partial super-tile ---- */
    {
      int64_t lim_min = c->causal ? (q0 + diag + 1) : seq_k;
      if (lim_min > seq_k) lim_min = seq_k;
      if (t0 + nkp > lim_min) {
        for (int64_t r = 0; r < nrows; r++) {
          int64_t lim = c->causal ? (q0 + r + diag + 1) : seq_k;
          if (lim > seq_k) lim = seq_k;
          lim -= t0;
          if (lim >= nkp) continue;
          float *lg = logits + r * kt;
          if (lim <= 0) {
            for (int j = 0; j < nkp; j += 16) _mm512_storeu_ps(lg + j, ninf);
          } else {
            for (int j = (int)(lim & ~15L); j < nkp; j += 16) {
              __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(j), lane);
              __mmask16 keep =
                  _mm512_cmpgt_epi32_mask(_mm512_set1_epi32((int)lim), idx);
              _mm512_storeu_ps(
                  lg + j,
                  _mm512_mask_mov_ps(ninf, keep, _mm512_loadu_ps(lg + j)));
            }
          }
        }
      }
    }

    /* ---- online softmax over the whole super-tile ------------------------ */
    for (int rg = 0; !(c->ablate & 2) && rg < 2; rg++) {
      float *lg = logits + rg * 16 * kt;
      __m512 m_old = rg ? m1 : m0;
      float rm[16];
      for (int r = 0; r < 16; r++) {
        __m512 v = _mm512_loadu_ps(lg + r * kt);
        for (int j = 16; j < nkp; j += 16)
          v = _mm512_max_ps(v, _mm512_loadu_ps(lg + r * kt + j));
        rm[r] = _mm512_reduce_max_ps(v);
      }
      __m512 m_cand = _mm512_loadu_ps(rm);
      /* FA4's -8 threshold: rescale only when the max really moved. */
      __mmask16 need = _mm512_cmp_ps_mask(
          m_cand, _mm512_add_ps(m_old, _mm512_set1_ps(8.0f)), _CMP_GT_OQ);
      __m512 m_new = _mm512_mask_mov_ps(m_old, need, m_cand);
      __m512 alpha = exp2_ps(_mm512_mul_ps(_mm512_sub_ps(m_old, m_new), c1v));
      if (rg)
        m1 = m_new;
      else
        m0 = m_new;
      if (need) {
        float a_sc[16];
        _mm512_storeu_ps(a_sc, alpha);
        for (int r = 0; r < 16; r++) {
          if (!(need & (1u << r))) continue;
          __m512 k = _mm512_set1_ps(a_sc[r]);
          float *arow = acc + (rg * 16 + r) * D;
          for (int j = 0; j < D; j += 16)
            _mm512_storeu_ps(arow + j,
                             _mm512_mul_ps(_mm512_loadu_ps(arow + j), k));
        }
      }
      float m_sc[16];
      _mm512_storeu_ps(m_sc, m_new);
      /* exp + P in bf16 + row sum */
      if (!(c->ablate & 8) && AMX_SM != 1) {
        /* Two rows in flight and two sum accumulators per row: the polynomial
         * is a serial FMA chain, so one row at a time leaves the vector pipes
         * idle and the `sum` add is a second serial chain. */
        for (int r = 0; r < 16; r += 2) {
          const float *lp0 = lg + r * kt, *lp1 = lp0 + kt;
          bf16 *pp0 = prob + (rg * 16 + r) * kt, *pp1 = pp0 + kt;
          const __m512 off0 = _mm512_set1_ps(-m_sc[r] * c1f);
          const __m512 off1 = _mm512_set1_ps(-m_sc[r + 1] * c1f);
          __m512 s0a = _mm512_setzero_ps(), s0b = _mm512_setzero_ps();
          __m512 s1a = _mm512_setzero_ps(), s1b = _mm512_setzero_ps();
          int j = 0;
          for (; j + 32 <= nkp; j += 32) {
            __m512 e00 =
                exp2_ps(_mm512_fmadd_ps(_mm512_loadu_ps(lp0 + j), c1v, off0));
            __m512 e10 =
                exp2_ps(_mm512_fmadd_ps(_mm512_loadu_ps(lp1 + j), c1v, off1));
            __m512 e01 = exp2_ps(
                _mm512_fmadd_ps(_mm512_loadu_ps(lp0 + j + 16), c1v, off0));
            __m512 e11 = exp2_ps(
                _mm512_fmadd_ps(_mm512_loadu_ps(lp1 + j + 16), c1v, off1));
            s0a = _mm512_add_ps(s0a, e00);
            s0b = _mm512_add_ps(s0b, e01);
            s1a = _mm512_add_ps(s1a, e10);
            s1b = _mm512_add_ps(s1b, e11);
            _mm256_storeu_si256((__m256i *)(pp0 + j),
                                (__m256i)_mm512_cvtneps_pbh(e00));
            _mm256_storeu_si256((__m256i *)(pp1 + j),
                                (__m256i)_mm512_cvtneps_pbh(e10));
            _mm256_storeu_si256((__m256i *)(pp0 + j + 16),
                                (__m256i)_mm512_cvtneps_pbh(e01));
            _mm256_storeu_si256((__m256i *)(pp1 + j + 16),
                                (__m256i)_mm512_cvtneps_pbh(e11));
          }
          for (; j < nkp; j += 16) {
            __m512 e0 =
                exp2_ps(_mm512_fmadd_ps(_mm512_loadu_ps(lp0 + j), c1v, off0));
            __m512 e1 =
                exp2_ps(_mm512_fmadd_ps(_mm512_loadu_ps(lp1 + j), c1v, off1));
            s0a = _mm512_add_ps(s0a, e0);
            s1a = _mm512_add_ps(s1a, e1);
            _mm256_storeu_si256((__m256i *)(pp0 + j),
                                (__m256i)_mm512_cvtneps_pbh(e0));
            _mm256_storeu_si256((__m256i *)(pp1 + j),
                                (__m256i)_mm512_cvtneps_pbh(e1));
          }
          rm[r] = _mm512_reduce_add_ps(_mm512_add_ps(s0a, s0b));
          rm[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(s1a, s1b));
        }
      } else {
        for (int r = 0; r < 16; r++) {
          const float *lp = lg + r * kt;
          bf16 *pp = prob + (rg * 16 + r) * kt;
          const __m512 off = _mm512_set1_ps(-m_sc[r] * c1f);
          __m512 sum = _mm512_setzero_ps();
          for (int j = 0; j < nkp; j += 16) {
            __m512 e =
                exp2_ps(_mm512_fmadd_ps(_mm512_loadu_ps(lp + j), c1v, off));
            sum = _mm512_add_ps(sum, e);
            _mm256_storeu_si256((__m256i *)(pp + j),
                                (__m256i)_mm512_cvtneps_pbh(e));
          }
          rm[r] = _mm512_reduce_add_ps(sum);
        }
      }
      __m512 rows = _mm512_loadu_ps(rm);
      if (rg)
        s1 = _mm512_add_ps(_mm512_mul_ps(s1, alpha), rows);
      else
        s0 = _mm512_add_ps(_mm512_mul_ps(s0, alpha), rows);
    }

    /* ---- P@V over the super-tile, 2 dim chunks (4 accumulator tiles) ----- */
    if (!(c->ablate & 4)) {
      const int nks = (nk + 31) / 32;
      for (int dc = 0; dc < D / 16; dc += 2) {
        _tile_loadd(4, (void *)(acc + dc * 16), astride);
        _tile_loadd(5, (void *)(acc + (dc + 1) * 16), astride);
        _tile_loadd(6, (void *)(acc + 16 * D + dc * 16), astride);
        _tile_loadd(7, (void *)(acc + 16 * D + (dc + 1) * 16), astride);
        for (int ks = 0; ks < nks; ks++) {
          const int64_t vb = ((t0 / 32) + ks) * c->dc16 * 512;
          _tile_loadd(0, (const void *)(prob + ks * 32), pstride);
          _tile_loadd(1, (const void *)(prob + 16 * kt + ks * 32), pstride);
          _tile_loadd(2, (const void *)(vbase + vb + dc * 512), 64);
          _tile_loadd(3, (const void *)(vbase + vb + (dc + 1) * 512), 64);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
        }
        _tile_stored(4, (void *)(acc + dc * 16), astride);
        _tile_stored(5, (void *)(acc + (dc + 1) * 16), astride);
        _tile_stored(6, (void *)(acc + 16 * D + dc * 16), astride);
        _tile_stored(7, (void *)(acc + 16 * D + (dc + 1) * 16), astride);
      }
    }
  }

  /* ---- write the output rows -------------------------------------------- */
  float sv[2][16];
  _mm512_storeu_ps(sv[0], s0);
  _mm512_storeu_ps(sv[1], s1);
  for (int64_t r = 0; r < nrows; r++) {
    bf16 *orow = obase + (q0 + r) * c->ot;
    const float *arow = acc + r * D;
    const float sum = sv[r / 16][r % 16];
    if (!(sum > 0.0f)) {
      for (int j = 0; j < D; j += 16)
        _mm256_storeu_si256((__m256i *)(orow + j), _mm256_setzero_si256());
      continue;
    }
    const __m512 iv = _mm512_set1_ps(1.0f / sum);
    for (int j = 0; j < D; j += 16)
      _mm256_storeu_si256((__m256i *)(orow + j),
                          (__m256i)_mm512_cvtneps_pbh(
                              _mm512_mul_ps(_mm512_loadu_ps(arow + j), iv)));
  }
}

/* ------------------------------------------------------------------- driver */

typedef struct {
  void *p;
  size_t cap;
} Buf;

static Buf g_pack_k, g_pack_v, g_thread;

static void *grow(Buf *b, size_t need) {
  if (b->cap < need) {
    free(b->p);
    b->cap = need + (need >> 2);
    b->p = NULL;
    if (posix_memalign(&b->p, 64, b->cap)) {
      b->p = NULL;
      b->cap = 0;
      return NULL;
    }
  }
  return b->p;
}

/* Returns 0 on success, non-zero when the shape is unsupported (the caller then
 * falls back to the Mojo kernel). */
int dense_amx_bf16(const void *q, const void *k, const void *v, void *out,
                   int64_t batch, int64_t seq_q, int64_t seq_k, int64_t nh,
                   int64_t nkv, int64_t head_dim, int64_t qb, int64_t qt,
                   int64_t qh, int64_t ksb, int64_t kst, int64_t ksh,
                   int64_t vsb, int64_t vst, int64_t vsh, int64_t ob,
                   int64_t ot, int64_t oh, double scale, int64_t causal,
                   int64_t nthreads, int64_t kt_in) {
  /* mode lives in kt_in's top bits so the ctypes signature stays put:
   * bit 16 set = skip the pack, bit 17 set = skip the attention. Profiling
   * only (a skipped pack reads stale buffers on purpose). */
  const int ablate = (int)((kt_in >> 18) & 63);
  const int skip_pack = (kt_in & (1 << 16)) != 0;
  const int skip_attn = (kt_in & (1 << 17)) != 0;
  kt_in &= 0xFFFF;
  if (amx_init()) return 1;
  if (head_dim != 32 && head_dim != 64 && head_dim != 128) return 1;
  if (nkv <= 0 || nh % nkv) return 1;
  if (seq_q <= 0 || seq_k <= 0) return 1;
  int kt = (int)kt_in;
  if (kt <= 0) kt = 256;
  if (kt > MAXKT) kt = MAXKT;
  if (kt % KEYCH) return 1;
  if (nthreads < 1) nthreads = 1;

  const int64_t kb32 = (seq_k + 31) / 32;
  const int64_t kb16 = kb32 * 2;
  const int64_t dc32 = head_dim / 32, dc16 = head_dim / 16;
  const size_t kp_bytes = (size_t)batch * nkv * kb16 * dc32 * 512 * sizeof(bf16);
  const size_t vp_bytes = (size_t)batch * nkv * kb32 * dc16 * 512 * sizeof(bf16);
  bf16 *kp = (bf16 *)grow(&g_pack_k, kp_bytes);
  bf16 *vp = (bf16 *)grow(&g_pack_v, vp_bytes);
  const size_t per_thread =
      (size_t)kt * QROWS * 4 + (size_t)kt * QROWS * 2 +
      (size_t)QROWS * head_dim * 4 + (size_t)QROWS * head_dim * 2 + 128;
  Scratch *scr = (Scratch *)grow(&g_thread, per_thread * (size_t)nthreads);
  if (!kp || !vp || !scr) return 1;
  char *sp = (char *)scr;

  PackJob pj;
  pj.seq_k = seq_k;
  pj.nkv = nkv;
  pj.ksb = ksb;
  pj.kst = kst;
  pj.ksh = ksh;
  pj.vsb = vsb;
  pj.vst = vst;
  pj.vsh = vsh;
  pj.k = (const bf16 *)k;
  pj.v = (const bf16 *)v;
  pj.kp = kp;
  pj.vp = vp;
  pj.kb16 = kb16;
  pj.kb32 = kb32;
  pj.dc32 = dc32;
  pj.dc16 = dc16;
  pj.kp_bh = kb16 * dc32 * 512;
  pj.vp_bh = kb32 * dc16 * 512;

  Ctx c;
  c.q = (const bf16 *)q;
  c.out = (bf16 *)out;
  c.seq_q = seq_q;
  c.seq_k = seq_k;
  c.nh = nh;
  c.nkv = nkv;
  c.D = head_dim;
  c.qb = qb;
  c.qt = qt;
  c.qh = qh;
  c.ob = ob;
  c.ot = ot;
  c.oh = oh;
  c.scale = (float)scale;
  c.causal = (int)causal;
  c.kp = kp;
  c.vp = vp;
  c.dc32 = dc32;
  c.dc16 = dc16;
  c.kp_bh = pj.kp_bh;
  c.vp_bh = pj.vp_bh;
  c.nqg = (seq_q + QROWS - 1) / QROWS;
  c.kt = kt;
  c.ablate = ablate;

  const int64_t nk_tasks = batch * nkv * kb16;
  const int64_t nv_tasks = batch * nkv * kb32;
  const int64_t items = batch * nh * c.nqg;
  const int nthreads_i = (int)nthreads;

  /* One parallel region for the pack *and* the attention, so the call pays a
   * single fork-join on the pool torch already has awake. */
#pragma omp parallel num_threads(nthreads_i)
  {
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    tile_setup();
    for (int64_t i = tid; !skip_pack && i < nk_tasks + nv_tasks; i += nt) {
      if (i < nk_tasks) {
        int64_t t = i;
        const int64_t kb = t % kb16;
        t /= kb16;
        const int64_t h = t % nkv;
        const int64_t b = t / nkv;
        pack_k_slab(&pj, b, h, kb);
      } else {
        int64_t t = i - nk_tasks;
        const int64_t kb = t % kb32;
        t /= kb32;
        const int64_t h = t % nkv;
        const int64_t b = t / nkv;
        pack_v_slab(&pj, b, h, kb);
      }
    }
#pragma omp barrier
    char *base = sp + (size_t)tid * per_thread;
    Scratch sc;
    sc.logits = (float *)base;
    sc.prob = (bf16 *)(base + (size_t)kt * QROWS * 4);
    sc.acc = (float *)(base + (size_t)kt * QROWS * 4 + (size_t)kt * QROWS * 2);
    sc.qpad = (bf16 *)(base + (size_t)kt * QROWS * 4 +
                       (size_t)kt * QROWS * 2 + (size_t)QROWS * head_dim * 4);
    for (int64_t t = tid; !skip_attn && t < items; t += nt) {
      const int64_t g = t % c.nqg;
      const int64_t rest = t / c.nqg;
      const int64_t h2 = rest % nh;
      const int64_t b2 = rest / nh;
      dense_item(&c, &sc, b2, h2, g * QROWS);
    }
  }
  return 0;
}

/* Probe used by the Python side: 1 when AMX is usable here. */
int dense_amx_ok(void) { return amx_init() == 0 ? 1 : 0; }

/* Debug probe: pack K for a single contiguous (batch 1, one kv head) tensor and
 * run one 16-row x 32-key QK chunk, storing the raw logits as [16][32] floats.
 * Exists so tests/test_amx.py can compare the AMX matmul against q @ k^T
 * directly instead of guessing which stage a mismatch came from. */
int dense_amx_debug_qk(const void *q, const void *k, int64_t seq_k, int64_t D,
                       void *logits) {
  if (amx_init()) return 1;
  static bf16 kp[64 * 1024];
  const int64_t kb32 = (seq_k + 31) / 32, kb16 = kb32 * 2;
  const int64_t dc32 = D / 32;
  PackJob pj;
  memset(&pj, 0, sizeof pj);
  pj.seq_k = seq_k;
  pj.nkv = 1;
  pj.kst = D;
  pj.k = (const bf16 *)k;
  pj.kp = kp;
  pj.kb16 = kb16;
  pj.dc32 = dc32;
  pj.kp_bh = kb16 * dc32 * 512;
  for (int64_t kb = 0; kb < kb16; kb++)
    for (int64_t dc = 0; dc < dc32; dc++) pack_k_block(&pj, 0, 0, kb, dc);

  tile_setup();
  float *lg = (float *)logits;
  _tile_zero(4);
  _tile_zero(5);
  _tile_zero(6);
  _tile_zero(7);
  const bf16 *q0p = (const bf16 *)q;
  for (int dc = 0; dc < (int)dc32; dc++) {
    _tile_loadd(0, (const void *)(q0p + dc * 32), D * 2);
    _tile_loadd(1, (const void *)(q0p + 16 * D + dc * 32), D * 2);
    _tile_loadd(2, (const void *)(kp + dc * 512), 64);
    _tile_loadd(3, (const void *)(kp + dc * 512 + dc32 * 512), 64);
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
  }
  _tile_stored(4, (void *)(lg), 32 * 4);
  _tile_stored(5, (void *)(lg + 16), 32 * 4);
  _tile_stored(6, (void *)(lg + 16 * 32), 32 * 4);
  _tile_stored(7, (void *)(lg + 16 * 32 + 16), 32 * 4);
  return 0;
}

/* Debug probe #2: same as above but the caller supplies the *packed* K (one
 * head, `(kb16 * dc32) * 512` bf16), so tests can fill the tile bytes directly
 * and recover the hardware's operand mapping. */
int dense_amx_debug_qk_raw(const void *q, const void *kp, int64_t D,
                           void *logits) {
  if (amx_init()) return 1;
  const int64_t dc32 = D / 32;
  tile_setup();
  float *lg = (float *)logits;
  _tile_zero(4);
  _tile_zero(5);
  _tile_zero(6);
  _tile_zero(7);
  const bf16 *q0p = (const bf16 *)q;
  const bf16 *kpp = (const bf16 *)kp;
  for (int dc = 0; dc < (int)dc32; dc++) {
    _tile_loadd(0, (const void *)(q0p + dc * 32), D * 2);
    _tile_loadd(1, (const void *)(q0p + 16 * D + dc * 32), D * 2);
    _tile_loadd(2, (const void *)(kpp + dc * 512), 64);
    _tile_loadd(3, (const void *)(kpp + dc * 512 + dc32 * 512), 64);
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
  }
  _tile_stored(4, (void *)(lg), 32 * 4);
  _tile_stored(5, (void *)(lg + 16), 32 * 4);
  _tile_stored(6, (void *)(lg + 16 * 32), 32 * 4);
  _tile_stored(7, (void *)(lg + 16 * 32 + 16), 32 * 4);
  return 0;
}

/* ------------------------------------------------------------ pack self-test */
/* Verifies the 16x16 transpose and both pack layouts against a scalar model;
 * 0 when they match. Called from tests/test_amx.py. */
int dense_amx_pack_selftest(void) {
  enum { N = 16 };
  static int32_t in[N][N], ref[N][N], got[N][N];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) in[i][j] = i * 100 + j;
  for (int k = 0; k < N; k++)
    for (int c = 0; c < N; c++) ref[c][k] = in[k][c];
  __m512i r[N];
  for (int i = 0; i < N; i++) memcpy(&r[i], in[i], sizeof r[i]);
  tp16x16_epi32(r);
  for (int i = 0; i < N; i++) memcpy(got[i], &r[i], sizeof got[i]);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (got[i][j] != ref[i][j]) return 100 + i * 16 + j;

  /* pack_k_block: seq_k 32, D 32, one kv head. */
  {
    enum { SK = 32, D = 32 };
    static bf16 k[SK][D], kp[4 * 512];
    static bf16 want[512];
    for (int t = 0; t < SK; t++)
      for (int d = 0; d < D; d++) k[t][d] = (bf16)(t * 64 + d + 1);
    PackJob pj;
    memset(&pj, 0, sizeof pj);
    pj.seq_k = SK;
    pj.nkv = 1;
    pj.kst = D;
    pj.k = &k[0][0];
    pj.kp = kp;
    pj.kb16 = 2;
    pj.dc32 = 1;
    pj.kp_bh = 2 * 512;
    pack_k_block(&pj, 0, 0, 0, 0);
    /* MEASURED hardware layout: element (row rr, index t) = K[key t/2][dim
     * 2*rr + (t&1)], i.e. a 32-bit transpose with the dim pair interleaved. */
    for (int rr = 0; rr < 16; rr++)
      for (int t = 0; t < 32; t++)
        want[rr * 32 + t] = k[t >> 1][2 * rr + (t & 1)];
    for (int i = 0; i < 512; i++)
      if (kp[i] != want[i]) return 200 + i;
  }

  /* pack_v_block: seq_k 32, D 32. */
  {
    enum { SK = 32, D = 32 };
    static bf16 v[SK][D], vp[2 * 512];
    static bf16 want[512];
    for (int t = 0; t < SK; t++)
      for (int d = 0; d < D; d++) v[t][d] = (bf16)(t * 64 + d + 1);
    PackJob pj;
    memset(&pj, 0, sizeof pj);
    pj.seq_k = SK;
    pj.nkv = 1;
    pj.vst = D;
    pj.v = &v[0][0];
    pj.vp = vp;
    pj.kb32 = 1;
    pj.dc16 = 2;
    pj.vp_bh = 1024;
    pack_v_block(&pj, 0, 0, 0, 0); /* key block 0, dim block 0 (dims 0-15) */
    for (int rr = 0; rr < 16; rr++)
      for (int u = 0; u < 16; u++) {
        want[rr * 32 + 2 * u] = v[2 * rr][u];
        want[rr * 32 + 2 * u + 1] = v[2 * rr + 1][u];
      }
    for (int i = 0; i < 512; i++)
      if (vp[i] != want[i]) return 300 + i;
    pack_v_block(&pj, 0, 0, 0, 1); /* dims 16-31 */
    for (int rr = 0; rr < 16; rr++)
      for (int u = 0; u < 16; u++) {
        want[rr * 32 + 2 * u] = v[2 * rr][16 + u];
        want[rr * 32 + 2 * u + 1] = v[2 * rr + 1][16 + u];
      }
    for (int i = 0; i < 512; i++)
      if (vp[512 + i] != want[i]) return 400 + i;
  }
  return 0;
}
