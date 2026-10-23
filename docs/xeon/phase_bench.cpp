// In-situ phase benchmark for the AMX dense policy: calls the *shipped*
// pack/qk/softmax/pv with a prepared slab and reports ns per logit.
//
// The point is to compare like with like: these are the exact functions the
// kernel runs, compiled with the exact flags `scripts/build_cpp.py` uses, so a
// number here is the kernel's number and not a re-implementation's.
//
//   g++ -O3 -std=c++20 -fopenmp -mavx512f -mavx512bw -mavx512vl -mavx512dq \
//       -mavx512bf16 -mavx512vnni -mfma -mamx-tile -mamx-bf16 -mamx-int8 \
//       -I cpu_attention/csrc docs/xeon/phase_bench.cpp -o /tmp/pb && /tmp/pb
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cpu_attn_amx.hpp"

using namespace attn;

template <int DT, int HD>
static void bench(const char* tag, int64_t seq_k, int rounds) {
  using P = amx::Dense<DT, HD>;
  const int64_t seq_q = 32;
  DenseJob job{};
  job.batch = 1;
  job.seq_q = seq_q;
  job.seq_k = seq_k;
  job.heads = 1;
  job.kv_heads = 1;
  job.head_dim = HD;
  job.qb = seq_q * HD;
  job.qt = HD;
  job.qh = HD;
  job.kb = job.kh = job.vb = job.vh = HD;   // one head, one batch
  job.kt = job.vt = HD;                     // token stride
  job.ob = job.ot = job.oh = HD;
  job.scale = 1.0f / std::sqrt(float(HD));
  job.dtype = DT;
  job.causal = true;

  std::vector<float> qt_(size_t(seq_q) * HD, 0.5f);
  std::vector<float> kt_(size_t(seq_k) * HD, 0.25f);
  std::vector<float> vt_(size_t(seq_k) * HD, 0.125f);
  std::vector<bf16> kbuf(size_t(seq_k) * HD), vbuf(size_t(seq_k) * HD);
  if (DT == kBF16) {
    for (size_t i = 0; i < kbuf.size(); ++i) kbuf[i] = f32_to_bf16(kt_[i]);
    for (size_t i = 0; i < vbuf.size(); ++i) vbuf[i] = f32_to_bf16(vt_[i]);
    job.q = qt_.data();
    job.k = kbuf.data();
    job.v = vbuf.data();
  } else {
    job.q = qt_.data();
    job.k = kt_.data();
    job.v = vt_.data();
  }

  const size_t slab = P::slab_bytes(seq_k, HD);
  const size_t aux = P::aux_bytes(32, HD);
  char* sl = static_cast<char*>(std::aligned_alloc(64, (slab + 63) & ~size_t(63)));
  char* ax = static_cast<char*>(std::aligned_alloc(64, (aux + 63) & ~size_t(63)));
  std::memset(sl, 0, slab);
  std::memset(ax, 0, aux);
  float* logits = reinterpret_cast<float*>(ax);
  float* acc = logits + size_t(P::RQT) * P::kt_();
  float* m = acc + size_t(32) * HD;
  float* l = m + 32;
  float* alpha = l + 32;

  const int kt = P::kt_();
  const int nkeys = static_cast<int>(std::min<int64_t>(kt, seq_k));
  const double logits_n = double(nkeys) * seq_q;
  const float c = job.scale * kLog2e;

  auto timeit = [&](const char* what, auto&& fn) {
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      fn();
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::printf("%-10s %-4s d=%3d nk=%4d  %8.2f ns/logit  %6.1f GF/s\n", tag,
                DT == kBF16 ? "bf16" : "fp32", HD, nkeys, best / logits_n,
                what[0] == 'q' ? 2.0 * logits_n * HD / best
                               : (what[0] == 'p' ? 2.0 * logits_n * HD / best : 0.0));
    return best;
  };

  std::fprintf(stderr, "pack\n");
  P::pack(job, 0, 0, sl);
  for (int r = 0; r < 32; ++r) {
    m[r] = -1e30f;
    l[r] = 0.f;
  }
  timeit("pack", [&] { P::pack(job, 0, 0, sl); });
  std::fprintf(stderr, "qk\n");
  double qk_ns = timeit("qk", [&] {
    P::qk(job, sl, 0, 0, static_cast<const char*>(job.q), 32, 0, nkeys,
          seq_k - seq_q, logits);
  });
  std::fprintf(stderr, "sm\n");
  double sm_ns = timeit("softmax", [&] {
    P::softmax(logits, 32, nkeys, c, m, l, alpha);
  });
  std::fprintf(stderr, "pv\n");
  double pv_ns = timeit("pv", [&] {
    P::pv(job, sl, 0, 0, 0, nkeys, logits, 32, acc);
  });
  std::printf("            %-4s d=%3d  total %8.2f ns/logit "
              "(qk %.0f%% sm %.0f%% pv %.0f%%)\n",
              DT == kBF16 ? "bf16" : "fp32", HD, qk_ns / logits_n + sm_ns / logits_n + pv_ns / logits_n,
              100 * qk_ns / (qk_ns + sm_ns + pv_ns), 100 * sm_ns / (qk_ns + sm_ns + pv_ns),
              100 * pv_ns / (qk_ns + sm_ns + pv_ns));
  std::free(sl);
  std::free(ax);
}

// How much is one `ldtilecfg`? The kernel issues two per work item (one in
// `qk`, one in `pv`) and the Mojo kernel issues one per worker.
static void cfg_cost(int n) {
  double best = 1e30;
  for (int t = 0; t < 7; ++t) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) attn::amx::tile_cfg_load();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  std::printf("ldtilecfg: %.2f ns each (%d)\n", best / n, n);
}

int main(int argc, char** argv) {
  int rounds = argc > 1 ? std::atoi(argv[1]) : 200;
  cfg_cost(2000000);
  if (!amx_init()) {
    std::printf("no amx\n");
    return 1;
  }
  for (int64_t nk : {256, 1024}) {
    for (int hd : {64, 128}) {
      if (hd == 64) {
        bench<kBF16, 64>("seq", nk, rounds / 4);
        bench<kF32, 64>("seq", nk, rounds / 4);
      } else {
        bench<kBF16, 128>("seq", nk, rounds / 4);
        bench<kF32, 128>("seq", nk, rounds / 4);
      }
    }
  }
  return 0;
}
