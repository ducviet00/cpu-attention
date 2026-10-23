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
  std::vector<float> logits(static_cast<size_t>(P::RQT) * kt);
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
