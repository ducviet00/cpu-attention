// Entry points and ISA dispatch.
//
// There is one dispatch point per path, and the choice is made from CPUID /
// HWCAP (plus, for AMX, the process having asked the kernel for tile state), so
// one artifact runs everywhere. No `-march=native`: the committed build is
// AVX-512 + AMX / armv8.2+bf16+i8mm and everything else falls back.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cpu_attn.hpp"
#include "cpu_isa.hpp"

#if defined(__x86_64__)
namespace attn {
namespace vec {
bool available();
void dense(const DenseJob&, int);
void paged(const PagedJob&, int);
void store(const PagedJob&, int);
}  // namespace vec
namespace amx {
bool available();
void dense(const DenseJob&, int);
}  // namespace amx
}  // namespace attn
#elif defined(__aarch64__)
namespace attn {
namespace neon {
bool available();
void dense(const DenseJob&, int);
void paged(const PagedJob&, int);
void store(const PagedJob&, int);
}  // namespace neon
}  // namespace attn
#endif

namespace attn {
namespace scalar {
bool available();
void dense(const DenseJob&, int);
void paged(const PagedJob&, int);
void store(const PagedJob&, int);
}  // namespace scalar

// ---------------------------------------------------------------------------
// workspace
// ---------------------------------------------------------------------------
namespace {
struct Slot {
  uint64_t key;
  char* ptr;
  size_t bytes;
};
thread_local std::vector<Slot> g_slots;
constexpr size_t kWorkspaceCap = size_t(1536) << 20;
}  // namespace

char* workspace(uint64_t key, size_t bytes) {
  for (auto& s : g_slots) {
    if (s.key == key) {
      if (s.bytes >= bytes) return s.ptr;
      std::free(s.ptr);
      s.ptr = static_cast<char*>(aligned_alloc64(bytes));
      s.bytes = bytes;
      return s.ptr;
    }
  }
  size_t total = 0;
  for (const auto& s : g_slots) total += s.bytes;
  if (total + bytes > kWorkspaceCap) {
    for (auto& s : g_slots) std::free(s.ptr);
    g_slots.clear();
  }
  char* p = static_cast<char*>(aligned_alloc64(bytes));
  g_slots.push_back(Slot{key, p, bytes});
  return p;
}

void workspace_clear() {
  for (auto& s : g_slots) std::free(s.ptr);
  g_slots.clear();
}

size_t workspace_bytes() {
  size_t total = 0;
  for (const auto& s : g_slots) total += s.bytes;
  return total;
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------
namespace {

bool env_off(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '0' && v[1] == '\0';
}

int pick_paged_isa() {
#if defined(__x86_64__)
  if (vec::available()) return kIsaVec;
#elif defined(__aarch64__)
  if (neon::available()) return kIsaNeon;
#endif
  return kIsaScalar;
}

// AMX is the dense kernel on x86 when the process holds tile state: it runs
// 16x16 tiles so both phases retire 256 MACs per instruction, which is 2.8-9.8x
// the vector path. fp32 rides the same kernel with the operands split into bf16
// hi/lo halves and three products per phase (error 3-8e-5, gate 2e-4).
int pick_dense_isa(const DenseJob& job) {
#if defined(__x86_64__)
  if (amx::available() && !env_off("CPU_ATTENTION_AMX")) {
    const bool fp32 = job.dtype == kF32;
    if (!fp32 || !env_off("CPU_ATTENTION_AMX_FP32")) {
      if (amx_init()) return kIsaAmx;
    }
  }
  if (vec::available()) return kIsaVec;
#elif defined(__aarch64__)
  if (neon::available()) return kIsaNeon;
#endif
  (void)job;
  return kIsaScalar;
}

}  // namespace

void attention_entry(const PagedJob& job, int workers, bool do_store,
                     bool do_attend) {
  const int isa = pick_paged_isa();
  if (do_store) {
    if (isa == kIsaVec) {
#if defined(__x86_64__)
      vec::store(job, workers);
#endif
#if defined(__aarch64__)
    } else if (isa == kIsaNeon) {
      neon::store(job, workers);
#endif
    } else {
      scalar::store(job, workers);
    }
  }
  if (do_attend) {
    if (isa == kIsaVec) {
#if defined(__x86_64__)
      vec::paged(job, workers);
#endif
#if defined(__aarch64__)
    } else if (isa == kIsaNeon) {
      neon::paged(job, workers);
#endif
    } else {
      scalar::paged(job, workers);
    }
  }
}

int dense_isa_name() {
  DenseJob probe;
  probe.head_dim = 64;
  probe.dtype = kBF16;
  return pick_dense_isa(probe);
}

void dense_entry(const DenseJob& job, int workers) {
  const int isa = pick_dense_isa(job);
  if (isa == kIsaAmx) {
#if defined(__x86_64__)
    amx::dense(job, workers);
    return;
#endif
  }
  if (isa == kIsaVec) {
#if defined(__x86_64__)
    vec::dense(job, workers);
    return;
#endif
  }
  if (isa == kIsaNeon) {
#if defined(__aarch64__)
    neon::dense(job, workers);
    return;
#endif
  }
  scalar::dense(job, workers);
}

}  // namespace attn
