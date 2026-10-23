// Portable fallback: no vector ISA, or an ISA whose kernel is not built for
// this architecture. One element at a time; it exists so the extension always
// loads and so a wrong ISA pick is a slow call rather than a crash.
//
// Its cache layout is K [hd][tokens], V [tokens][hd] -- the same addressing the
// AVX-512 fp32 path uses, which keeps `reshape_and_cache` identical on the two
// and means a cache staged through the fallback still reads correctly.
#pragma once

#include <algorithm>
#include <cstring>

#include "cpu_attn_impl.hpp"

namespace attn {
namespace scalar {
namespace {

template <int DT, int HD>
struct Kern {
  static constexpr bool BF = DT == kBF16;

  static size_t aux_bytes(int) { return 0; }

  static float load(const char* p) {
    if constexpr (BF) {
      bf16 v;
      std::memcpy(&v, p, 2);
      return bf16_to_f32(v);
    } else {
      float v;
      std::memcpy(&v, p, 4);
      return v;
    }
  }
  static void store(char* p, float v) {
    if constexpr (BF) {
      bf16 b = f32_to_bf16(v);
      std::memcpy(p, &b, 2);
    } else {
      std::memcpy(p, &v, 4);
    }
  }

  static void store_head(const PagedJob& job, int64_t seq, int64_t head) {
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t slot = job.slots[seq];
    const int64_t pg = slot / job.page, off = slot % job.page;
    const int64_t esz = BF ? 2 : 4;
    const int64_t stride = job.page * HD * esz;
    const int64_t base = (pg * job.num_kv_heads + kvh) * stride;
    const int64_t kvst = job.kv_stride ? job.kv_stride : job.q_stride;
    const char* ks =
        static_cast<const char*>(job.k) + (seq * kvst + head * HD) * esz;
    const char* vs =
        static_cast<const char*>(job.v) + (seq * kvst + head * HD) * esz;
    char* kp = static_cast<char*>(job.k_cache) + base;
    char* vp = static_cast<char*>(job.v_cache) + base;
    for (int d = 0; d < HD; ++d) {
      if constexpr (BF) {
        std::memcpy(kp + (((d / 2) * job.page + off) * 2 + (d & 1)) * esz,
                    ks + d * esz, esz);
      } else {
        std::memcpy(kp + (d * job.page + off) * esz, ks + d * esz, esz);
      }
    }
    std::memcpy(vp + off * HD * esz, vs, HD * esz);
  }

  static void attend_head(const PagedJob& job, int64_t seq, int64_t head,
                          char*) {
    const int64_t group = job.num_heads / job.num_kv_heads;
    const int64_t kvh = head / group;
    const int64_t len = job.lens[seq];
    const int32_t* bt = job.block_table + seq * job.bt_stride;
    const int64_t esz = BF ? 2 : 4;
    const int64_t stride = job.page * HD * esz;
    const char* q = static_cast<const char*>(job.q) +
                    (seq * job.q_stride + head * HD) * esz;
    char* op = static_cast<char*>(job.out) +
               (seq * job.out_stride + head * HD) * esz;
    float acc[HD];
    for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const float c = job.scale * kLog2e;
    float p[4096];
    for (int64_t k0 = 0; k0 < len; k0 += 4096) {
      const int n = static_cast<int>(std::min<int64_t>(4096, len - k0));
      float mx = -INFINITY;
      for (int j = 0; j < n; ++j) {
        const int64_t jj = k0 + j;
        const char* kb = static_cast<const char*>(job.k_cache) +
                         (bt[jj / job.page] * job.num_kv_heads + kvh) * stride;
        const int64_t off = jj % job.page;
        float s = 0.0f;
        for (int d = 0; d < HD; ++d) {
          const char* kp;
          if constexpr (BF) {
            kp = kb + (((d / 2) * job.page + off) * 2 + (d & 1)) * esz;
          } else {
            kp = kb + (d * job.page + off) * esz;
          }
          s += load(q + d * esz) * load(kp);
        }
        p[j] = s * c;
        mx = mx > p[j] ? mx : p[j];
      }
      float a;
      if (mx == -INFINITY) {
        a = 1.0f;
      } else if (m == -INFINITY) {
        m = mx;
        a = 0.0f;
      } else {
        const float mn = m > mx ? m : mx;
        a = std::exp2(m - mn);
        m = mn;
      }
      l *= a;
      for (int d = 0; d < HD; ++d) acc[d] *= a;
      for (int j = 0; j < n; ++j) {
        p[j] = std::exp2(p[j] - m);
        l += p[j];
      }
      for (int j = 0; j < n; ++j) {
        const int64_t jj = k0 + j;
        const char* vb = static_cast<const char*>(job.v_cache) +
                         (bt[jj / job.page] * job.num_kv_heads + kvh) * stride +
                         ((jj % job.page) * HD) * esz;
        for (int d = 0; d < HD; ++d)
          acc[d] += p[j] * load(vb + d * esz);
      }
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = 0; d < HD; ++d) store(op + d * esz, acc[d] * inv);
  }
};

template <int DT, int HD>
struct DenseKern {
  static constexpr bool BF = DT == kBF16;
  static constexpr int RQT = 4;
  static constexpr int KT = 256;
  static int kt_() { return KT; }

  static int rq(int64_t seq_q, int) {
    return static_cast<int>(std::min<int64_t>(16, std::max<int64_t>(seq_q, 1)));
  }
  static size_t slab_bytes(int64_t, int) { return 0; }
  static int acc_rows(int rq) { return ((rq + RQT - 1) / RQT) * RQT; }
  static size_t aux_bytes(int rq, int hd) {
    const size_t rows = static_cast<size_t>(acc_rows(rq));
    return static_cast<size_t>(RQT * KT) * 4 + rows * hd * 4 + rows * 3 * 4;
  }
  static void pack(const DenseJob&, int64_t, int64_t, char*) {}

  static float load(const char* p) {
    if constexpr (BF) {
      bf16 v;
      std::memcpy(&v, p, 2);
      return bf16_to_f32(v);
    } else {
      float v;
      std::memcpy(&v, p, 4);
      return v;
    }
  }

  static void qk(const DenseJob& job, const char*, int64_t b, int64_t kvh,
                 const char* q_base, int nrows, int64_t k0, int nkeys,
                 int64_t row_limit0, float* logits) {
    const char* kbase = static_cast<const char*>(job.k) +
                        (b * job.kb + kvh * job.kh) * (BF ? 2 : 4);
    for (int j = 0; j < nkeys; ++j) {
      const int64_t jj = k0 + j;
      for (int r = 0; r < nrows; ++r) {
        const int64_t lim = row_limit0 + r;
        float s = 0.0f;
        if (jj <= lim) {
          for (int d = 0; d < HD; ++d)
            s += load(q_base + (r * job.qt + d) * (BF ? 2 : 4)) *
                 load(kbase + (jj * job.kt + d) * (BF ? 2 : 4));
        } else {
          s = -INFINITY;
        }
        logits[r * KT + j] = s;
      }
    }
  }
  static void softmax(float* logits, int nrows, int nkeys, float c, float* m,
                      float* l, float* alpha) {
    for (int r = 0; r < nrows; ++r) {
      float* p = logits + r * KT;
      float mx = -INFINITY;
      for (int j = 0; j < nkeys; ++j) {
        p[j] *= c;
        mx = mx > p[j] ? mx : p[j];
      }
      float mn, a;
      if (mx == -INFINITY) {
        mn = m[r];
        a = 1.0f;
      } else if (m[r] == -INFINITY) {
        mn = mx;
        a = 0.0f;
      } else {
        mn = m[r] > mx ? m[r] : mx;
        a = std::exp2(m[r] - mn);
      }
      m[r] = mn;
      alpha[r] = a;
      l[r] *= a;
      float sum = 0.0f;
      for (int j = 0; j < nkeys; ++j) {
        p[j] = mn == -INFINITY ? 0.0f : std::exp2(p[j] - mn);
        sum += p[j];
      }
      l[r] += sum;
    }
  }
  static void pv(const DenseJob& job, const char*, int64_t b, int64_t kvh,
                 int64_t k0, int nkeys, const float* p, int nrows,
                 float* acc) {
    const char* vbase = static_cast<const char*>(job.v) +
                        (b * job.vb + kvh * job.vh) * (BF ? 2 : 4);
    for (int j = 0; j < nkeys; ++j) {
      const int64_t jj = k0 + j;
      for (int d = 0; d < HD; ++d) {
        const float v = load(vbase + (jj * job.vt + d) * (BF ? 2 : 4));
        for (int r = 0; r < nrows; ++r) acc[r * HD + d] += p[r * KT + j] * v;
      }
    }
  }
};

}  // namespace

bool available() { return true; }

void dense(const DenseJob& job, int workers) {
#define ATTN_SD(DT, HD) dense_run<DenseKern<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_SD(kF32, 32);
    else if (job.head_dim == 64) ATTN_SD(kF32, 64);
    else ATTN_SD(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_SD(kBF16, 32);
    else if (job.head_dim == 64) ATTN_SD(kBF16, 64);
    else ATTN_SD(kBF16, 128);
  }
#undef ATTN_SD
}

void paged(const PagedJob& job, int workers) {
#define ATTN_SP(DT, HD) paged_run<Kern<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_SP(kF32, 32);
    else if (job.head_dim == 64) ATTN_SP(kF32, 64);
    else ATTN_SP(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_SP(kBF16, 32);
    else if (job.head_dim == 64) ATTN_SP(kBF16, 64);
    else ATTN_SP(kBF16, 128);
  }
#undef ATTN_SP
}

void store(const PagedJob& job, int workers) {
#define ATTN_SS(DT, HD) store_run<Kern<DT, HD>>(job, workers)
  if (job.dtype == kF32) {
    if (job.head_dim == 32) ATTN_SS(kF32, 32);
    else if (job.head_dim == 64) ATTN_SS(kF32, 64);
    else ATTN_SS(kF32, 128);
  } else {
    if (job.head_dim == 32) ATTN_SS(kBF16, 32);
    else if (job.head_dim == 64) ATTN_SS(kBF16, 64);
    else ATTN_SS(kBF16, 128);
  }
#undef ATTN_SS
}

}  // namespace scalar
}  // namespace attn
