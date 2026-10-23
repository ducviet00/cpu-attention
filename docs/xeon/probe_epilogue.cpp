// Driver-overhead probe: the two scalar-ish loops `dense_run` runs per work
// item -- the accumulator memset and the output epilogue -- copied verbatim
// from cpu_attn_impl.hpp, timed against their vectorisable equivalents.
//
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 \
//       -mfma probe_epilogue.cpp -o /tmp/probe && /tmp/probe
#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

struct bf16 { uint16_t v; };

static inline bf16 f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return bf16{static_cast<uint16_t>((u >> 16) | 0x40)};
  uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
  u += bias;
  return bf16{static_cast<uint16_t>(u >> 16)};
}

// exactly the driver's loop
static void epilogue_scalar(const float* acc, bf16* out, int nrows, int hd,
                            const float* inv) {
  for (int r = 0; r < nrows; ++r) {
    const float s = inv[r];
    const float* src = acc + r * hd;
    bf16* o = out + r * hd;
    for (int d = 0; d < hd; ++d) o[d] = f32_to_bf16(src[d] * s);
  }
}

// the same thing written for the vector unit
static void epilogue_vec(const float* acc, bf16* out, int nrows, int hd,
                         const float* inv) {
  const __m512i rne = _mm512_set1_epi32(0x7fff);
  for (int r = 0; r < nrows; ++r) {
    const __m512 s = _mm512_set1_ps(inv[r]);
    int d = 0;
    for (; d + 16 <= hd; d += 16) {
      const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(acc + r * hd + d), s);
      const __m256bh hv = _mm512_cvtneps_pbh(v);
      __m256i h;
      std::memcpy(&h, &hv, sizeof(h));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + r * hd + d), h);
    }
    for (; d < hd; ++d) out[r * hd + d] = f32_to_bf16(acc[r * hd + d] * inv[r]);
  }
  (void)rne;
}

int main() {
  // 4096:16:128 -> 4096*16 rows of 128, i.e. the largest row in the matrix
  for (int cfg = 0; cfg < 3; ++cfg) {
    int hd = cfg == 0 ? 64 : 128;
    int items = cfg == 2 ? 128 * 32 : 65536;  // (rows x dim) / hd
    int nrows = 32;
    std::vector<float> acc(size_t(nrows) * hd);
    std::vector<bf16> out(acc.size());
    std::vector<float> inv(nrows, 1.0f / 3.0f);
    for (size_t i = 0; i < acc.size(); ++i) acc[i] = float(i % 7);
    int reps = items / nrows;
    for (int which = 0; which < 2; ++which) {
      double best = 1e30;
      for (int t = 0; t < 7; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
          if (which == 0) epilogue_scalar(acc.data(), out.data(), nrows, hd, inv.data());
          else epilogue_vec(acc.data(), out.data(), nrows, hd, inv.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      std::printf("hd=%3d rows=%7d %s: %9.1f us  (%.2f ns/elem)\n", hd,
                  items, which == 0 ? "scalar" : "vec   ", best,
                  1000.0 * best / items / hd);
    }
    // memset of the accumulator, `reps` times
    double best = 1e30;
    for (int t = 0; t < 7; ++t) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) std::memset(acc.data(), 0, sizeof(float) * size_t(nrows) * hd);
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::printf("hd=%3d rows=%7d memset: %9.1f us  (%.2f ns/row)\n", hd, items,
                best, 1000.0 * best / items);
  }
  return 0;
}
