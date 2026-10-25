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
#include <memory>
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
// The paged kernel lives in this TU because it is the fastest AVX-512 paged
// path on the box `available()` is true for, but it is not a tile kernel: an
// AMX tile is 16 rows and a decode item is one live query row (31 useful
// MAC/cycle against `dpbf16ps`'s 32), so `docs/amx/AMX_PAGED.md` measures it and
// keeps the vector form. Routing paged here is a per-dtype loop-shape choice --
// dim-outer Q@K^T with 64 keys per step for bf16 -- not an ISA change.
void paged(const PagedJob&, int);
void store(const PagedJob&, int);
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
  std::unique_ptr<char, decltype(&std::free)> ptr;
  size_t bytes;
};
thread_local std::vector<Slot> g_slots;
constexpr size_t kWorkspaceCap = size_t(1536) << 20;
}  // namespace

char* workspace(uint64_t key, size_t bytes) {
  for (auto& s : g_slots) {
    if (s.key == key) {
      if (s.bytes >= bytes) return s.ptr.get();
      s.ptr.reset(static_cast<char*>(aligned_alloc64(bytes)));
      s.bytes = bytes;
      return s.ptr.get();
    }
  }
  size_t total = 0;
  for (const auto& s : g_slots) total += s.bytes;
  if (total + bytes > kWorkspaceCap) {
    g_slots.clear();
  }
  char* p = static_cast<char*>(aligned_alloc64(bytes));
  // The owning slot also frees its buffer when the calling Python thread exits.
  g_slots.push_back(Slot{key, {p, &std::free}, bytes});
  return p;
}

void workspace_clear() {
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

int pick_paged_isa(int dtype) {
  const char* forced = std::getenv("CPU_ATTENTION_ISA");
  if (forced && std::strcmp(forced, "scalar") == 0) return kIsaScalar;
#if defined(__x86_64__)
  if (dtype == kBF16 && !cpu().avx512bf16) return kIsaScalar;
  if ((!forced || std::strcmp(forced, "amx") == 0) && amx::available())
    return kIsaAmx;
  if (vec::available()) return kIsaVec;
#elif defined(__aarch64__)
  if (dtype == kBF16 && !cpu().bf16) return kIsaScalar;
  if (neon::available()) return kIsaNeon;
#endif
  return kIsaScalar;
}

// AMX is the dense kernel on x86 when the process holds tile state: it runs
// 16x16 tiles so both phases retire 256 MACs per instruction, which is 2.8-9.8x
// the vector path. fp32 rides the same kernel with the operands split into bf16
// hi/lo halves and three products per phase (error 3-8e-5, gate 2e-4).
int pick_dense_isa(const DenseJob& job) {
  // CPU_ATTENTION_ISA=scalar|vec|amx pins the dense kernel for an A/B; the
  // paged path follows the same lever through pick_paged_isa.
  const char* forced = std::getenv("CPU_ATTENTION_ISA");
  if (forced && std::strcmp(forced, "scalar") == 0) return kIsaScalar;
#if defined(__x86_64__)
  if (job.dtype == kBF16 && !cpu().avx512bf16) return kIsaScalar;
  if ((!forced || std::strcmp(forced, "amx") == 0) &&
      amx::available() && !env_off("CPU_ATTENTION_AMX")) {
    const bool fp32 = job.dtype == kF32;
    if (!fp32 || !env_off("CPU_ATTENTION_AMX_FP32")) {
      if (amx_init()) return kIsaAmx;
    }
  }
  if (vec::available()) return kIsaVec;
#elif defined(__aarch64__)
  if (job.dtype == kBF16 && !cpu().bf16) return kIsaScalar;
  if (neon::available()) return kIsaNeon;
#endif
  (void)job;
  return kIsaScalar;
}

}  // namespace

void attention_entry(const PagedJob& job, int workers, bool do_store,
                      bool do_attend) {
  workers = std::max(1, std::min({workers, physical_cores(), omp_get_max_threads()}));
  const int isa = pick_paged_isa(job.dtype);
  // The attend entry takes the store with it when both are asked for: one
  // OpenMP region instead of two (the second fork-join is ~1.1 us, a sixth of a
  // batch-1 call). `shape_only` below is the store-only form reshape_and_cache
  // uses.
  if (do_attend) {
    PagedJob j = job;
    j.do_store = do_store;
    if (isa == kIsaVec) {
#if defined(__x86_64__)
      vec::paged(j, workers);
    } else if (isa == kIsaAmx) {
      amx::paged(j, workers);
#endif
#if defined(__aarch64__)
    } else if (isa == kIsaNeon) {
      neon::paged(j, workers);
#endif
    } else {
      scalar::paged(j, workers);
    }
    return;
  }
  if (do_store) {
    if (isa == kIsaVec) {
#if defined(__x86_64__)
      vec::store(job, workers);
    } else if (isa == kIsaAmx) {
      amx::store(job, workers);
#endif
#if defined(__aarch64__)
    } else if (isa == kIsaNeon) {
      neon::store(job, workers);
#endif
    } else {
      scalar::store(job, workers);
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
  workers = std::max(1, std::min({workers, physical_cores(), omp_get_max_threads()}));
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
