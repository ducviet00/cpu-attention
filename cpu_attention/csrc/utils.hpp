// Small shared pieces: the bf16 type, aligned buffers, CPU topology, and the
// exp approximation every ISA's softmax uses.
//
// Apache-2.0. The page model, the scheduler shape and the softmax/GEMM
// structure follow vLLM's csrc/cpu attention (Apache-2.0); the kernel code is
// this project's.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace attn {

// ---------------------------------------------------------------------------
// bf16
// ---------------------------------------------------------------------------
// A plain 16-bit container rather than __bf16: the portable TU is compiled
// without -mavx512bf16 / +bf16, and __bf16 is not available there. Conversions
// are bit operations, so they compile to the same code the intrinsic would.
struct bf16 {
  uint16_t v;
};
static_assert(sizeof(bf16) == 2, "bf16 must be 16 bits");

inline float bf16_to_f32(bf16 x) {
  uint32_t u = static_cast<uint32_t>(x.v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round to nearest even, the same rounding _mm512_cvtneps_pbh uses, so the
// scalar and vector paths agree bit for bit.
inline bf16 f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  // NaN must stay a NaN: an all-ones payload would otherwise round to inf.
  if ((u & 0x7fffffffu) > 0x7f800000u) return bf16{static_cast<uint16_t>(
      (u >> 16) | 0x40)};
  uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
  u += bias;
  return bf16{static_cast<uint16_t>(u >> 16)};
}

enum DType { kF32 = 0, kBF16 = 1 };

inline int elem_size(int dtype) { return dtype == kF32 ? 4 : 2; }

// ---------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------
// 2^x for x in [-127, 0], as a degree-5 polynomial on the fraction and an
// exponent bit insert (Cephes). std::exp lowers to a call or a long inline
// sequence; this is ~10 instructions and the softmax is 15-30% of a call.
//
// The polynomial is evaluated on r = x - round(x) in [-0.5, 0.5]:
//   2^r = 1 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
inline float exp2_poly5(float x) {
  x = x < -127.0f ? -127.0f : (x > 0.0f ? 0.0f : x);
  float n = std::nearbyint(x);
  float r = x - n;
  float p = 1.3392112e-3f;
  p = p * r + 9.6180963e-3f;
  p = p * r + 5.5503282e-2f;
  p = p * r + 2.4022652e-1f;
  p = p * r + 6.9314718e-1f;
  p = p * r + 1.0f;
  // 2^n by exponent insertion; n <= 0 so the result stays finite.
  uint32_t e = static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23;
  float s;
  std::memcpy(&s, &e, 4);
  return p * s;
}

// bf16 needs less: the store two instructions later is 8 bits of mantissa, so a
// degree-2 polynomial is inside its rounding (measured 7.8e-3 -> 1.6e-2 against
// a fixed 3e-2 gate on the dense path).
inline float exp2_poly2(float x) {
  x = x < -127.0f ? -127.0f : (x > 0.0f ? 0.0f : x);
  float n = std::nearbyint(x);
  float r = x - n;
  float p = (2.3473140e-1f * r + 6.9335938e-1f) * r + 1.0f;
  uint32_t e = static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23;
  float s;
  std::memcpy(&s, &e, 4);
  return p * s;
}

constexpr float kLog2e = 1.4426950408889634f;

// ---------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------
inline void* aligned_alloc64(size_t bytes) {
  if (bytes == 0) bytes = 64;
  void* p = std::aligned_alloc(64, (bytes + 63) & ~size_t(63));
  if (p == nullptr) {
    std::fprintf(stderr, "cpu_attention(cpp): out of memory (%zu bytes)\n",
                 bytes);
    std::abort();
  }
  return p;
}

// ---------------------------------------------------------------------------
// topology
// ---------------------------------------------------------------------------
// Physical cores (packages x core_id), because two SMT siblings share the
// vector units and a fork-join over 2x the physical count measured 1.5x slower.
inline int physical_cores() {
  static int cached = 0;
  if (cached) return cached;
  int total = 0;
  std::vector<std::pair<int, int>> seen;
  for (int cpu = 0; cpu < 256; ++cpu) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) break;
    int core = -1;
    if (std::fscanf(f, "%d", &core) != 1) core = -1;
    std::fclose(f);
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                  cpu);
    f = std::fopen(path, "r");
    int pkg = 0;
    if (f != nullptr) {
      if (std::fscanf(f, "%d", &pkg) != 1) pkg = 0;
      std::fclose(f);
    }
    if (core < 0) continue;
    std::pair<int, int> key{pkg, core};
    bool dup = false;
    for (const auto& s : seen) dup |= (s == key);
    if (!dup) seen.push_back(key);
    ++total;
  }
  cached = static_cast<int>(seen.size());
  if (cached == 0) cached = total > 0 ? total : 1;
  return cached;
}

inline int last_level_cache_bytes() {
  static int cached = 0;
  if (cached) return cached;
  for (int idx : {3, 2, 1, 0}) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) continue;
    char buf[32] = {0};
    if (std::fgets(buf, sizeof(buf), f) == nullptr) buf[0] = 0;
    std::fclose(f);
    int mult = 1;
    size_t n = std::strlen(buf);
    if (n == 0) continue;
    if (buf[n - 1] == 'K') {
      mult = 1024;
    } else if (buf[n - 1] == 'M') {
      mult = 1024 * 1024;
    }
    cached = std::atoi(buf) * mult;
    if (cached > 0) return cached;
  }
  cached = 32 << 20;
  return cached;
}

}  // namespace attn
