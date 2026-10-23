// Phase micro-benchmark for the AVX-512 policy: how many cycles each of
// Q@K^T, the online softmax and P@V spends per (query row, key) pair.
//
// A whole-call profile cannot say which phase is short: the threads share L3, so
// the same code measures differently at 8 threads and the phases overlap. This
// runs one phase at a time on a slab that is already packed, which is what the
// inner loops actually see.
//
//   g++ -O3 -std=c++20 -fopenmp -mavx512f -mavx512bw -mavx512vl -mavx512dq \
//       -mavx512bf16 -mavx512vnni -mfma -mavx2 -I cpu_attention/csrc \
//       tests/bench_vec_prims.cpp -o /tmp/bvp && /tmp/bvp
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_attn_vec.hpp"

using namespace attn;

// The two symbols the policy file references from the shared core; the phases
// under test never use them, so a minimal implementation keeps this benchmark
// linkable without pulling in the dispatcher.
namespace attn {
char* workspace(uint64_t, size_t bytes) {
  return static_cast<char*>(aligned_alloc64(bytes));
}
}  // namespace attn

namespace {

namespace {
inline void st16v(void* p, __m512 v) { _mm512_storeu_ps(p, v); }
inline __m512 ld16v(const void* p) { return _mm512_loadu_ps(p); }
inline __m512 widen(const void* p) {
  const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

// The P@V the bf16 policy used before the key-pair layout: fp32 FMA over
// widened V, which is what the shipped `dpbf16ps` form replaced. Same output
// buffer, same 32-dim chunk; only the multiply changes.
template <int HD>
double bench_pv_fma(const float* p, const char* v, int64_t nkeys, int kt,
                    float* acc, int iters) {
  const int RQT = 8;
  double best = 1e30;
  for (int r = 0; r < 3; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      for (int dc = 0; dc < HD; dc += 32) {
        __m512 c[RQT][2];
        for (int j = 0; j < RQT; ++j) {
          c[j][0] = ld16v(acc + j * HD + dc);
          c[j][1] = ld16v(acc + j * HD + dc + 16);
        }
        for (int64_t j = 0; j < nkeys; ++j) {
          const char* vv = v + (j * HD + dc) * 2;
          const __m512 v0 = widen(vv);
          const __m512 v1 = widen(vv + 32);
          for (int k = 0; k < RQT; ++k) {
            const __m512 pr = _mm512_set1_ps(p[k * kt + j]);
            c[k][0] = _mm512_fmadd_ps(pr, v0, c[k][0]);
            c[k][1] = _mm512_fmadd_ps(pr, v1, c[k][1]);
          }
        }
        for (int j = 0; j < RQT; ++j) {
          st16v(acc + j * HD + dc, c[j][0]);
          st16v(acc + j * HD + dc + 16, c[j][1]);
        }
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
    if (sec < best) best = sec;
  }
  return best;
}
}  // namespace

template <int DT, int HD>
void run(int64_t seq_k, int rows, int iters) {
  using P = vec::Dense<DT, HD>;
  const int64_t pad = P::pad_of(seq_k);
  const size_t slab_bytes = P::slab_bytes(seq_k, HD);
  char* slab = static_cast<char*>(aligned_alloc64(slab_bytes));
  std::memset(slab, 0, slab_bytes);
  // Fill the slab with small finite values so the softmax sees real logits.
  const int esz = DT == kBF16 ? 2 : 4;
  auto* fp = reinterpret_cast<float*>(slab);
  for (size_t i = 0; i < slab_bytes / 4; ++i) fp[i] = (i % 17) * 0.01f - 0.08f;
  (void)esz;
  std::vector<float> qbuf(static_cast<size_t>(rows) * HD, 0.01f);
  const int kt = P::kt_();
  // The bf16 policy keeps its probability pairs between the logits tile and the
  // accumulator (`logit_aux_floats`), so the tile needs that space too.
  std::vector<float> logits(static_cast<size_t>(P::RQT) * kt +
                            P::logit_aux_floats(kt));
  std::vector<float> acc(static_cast<size_t>(P::acc_rows(rows)) * HD);
  std::vector<float> ml(static_cast<size_t>(P::acc_rows(rows)) * 3);
  DenseJob job;
  job.seq_k = seq_k;
  job.seq_q = rows;
  job.qt = HD;
  job.dtype = DT;

  const int64_t tiles = (seq_k + kt - 1) / kt;
  const double pairs_per_tile =
      static_cast<double>(rows) * static_cast<double>(kt) * tiles;

  std::printf("%s D=%d keys=%ld rows=%d  ", DT == kBF16 ? "bf16" : "fp32", HD,
              (long)seq_k, rows);
  auto bench = [&](const char* name, int which) {
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        for (int64_t k0 = 0; k0 < seq_k; k0 += kt) {
          const int nk = static_cast<int>(std::min<int64_t>(kt, seq_k - k0));
          if (which == 0)
            P::qk(job, slab, 0, 0, reinterpret_cast<const char*>(qbuf.data()),
                  P::RQT, k0, nk, seq_k, logits.data());
          else if (which == 1)
            P::pv(job, slab, 0, 0, k0, nk, logits.data(), P::RQT, acc.data());
          else
            P::softmax(logits.data(), P::RQT, nk, 0.05f, ml.data(),
                       ml.data() + P::acc_rows(rows),
                       ml.data() + 2 * P::acc_rows(rows));
        }
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double s = std::chrono::duration<double>(t1 - t0).count() / iters;
      if (s < best) best = s;
    }
    std::printf("%s %8.1f us", name, best * 1e6);
    return best;
  };
  const double t1 = bench("qk", 0);
  const double t2 = bench("softmax", 2);
  const double t3 = bench("pv", 1);
  // Same P@V as bf16 runs it now (`dpbf16ps`, key pairs) against the fp32 FMA
  // form, so the 2.1x that decided the layout is reproducible here.
  if constexpr (DT == kBF16) {
    // The shipped bf16 P@V is the `dpbf16ps` form, so `pv` above is that one;
    // this is the fp32-FMA-over-widened-V body it replaced.
    std::vector<char> vb(slab_bytes, 1);
    const double t4 = bench_pv_fma<HD>(logits.data(), vb.data(), seq_k, kt,
                                       acc.data(), iters);
    std::printf(" pv_fma %5.1f us  dpbf16ps/fma %.2fx", t4 * 1e6, t4 / t3);
  }
  // zmm FMAs: qk = rows*keys*HD/16, pv = the same, softmax ~ 3 ops per pair/16.
  const double qk_fma = pairs_per_tile * HD / 16.0;
  const double pv_fma = pairs_per_tile * HD / 16.0;
  std::printf(" | qk %.2f  pv %.2f  softmax %.2f FMA-equiv/GHz",
              qk_fma / (t1 * 4.0e9) * 1e9, pv_fma / (t3 * 4.0e9) * 1e9,
              (t2 / t1));
  std::printf("  [qk %.2f cyc/pair, pv %.2f, sm %.2f]\n",
              t1 * 4.0e9 / pairs_per_tile, t3 * 4.0e9 / pairs_per_tile,
              t2 * 4.0e9 / pairs_per_tile);
  std::free(slab);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 3;
  run<kF32, 128>(4096, 32, iters);
  run<kF32, 128>(4096, 8, iters);
  run<kF32, 64>(4096, 64, iters);
  run<kBF16, 128>(4096, 32, iters);
  run<kBF16, 64>(4096, 64, iters);
  return 0;
}
