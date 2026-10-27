#include "cpu_isa.hpp"

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1UL << 14)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif

namespace attn {
namespace {

CpuFeatures detect() {
  CpuFeatures f;
#if defined(__x86_64__) || defined(__i386__)
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid_max(0, nullptr) >= 7) {
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    f.avx512f = (ebx >> 16) & 1;
    f.avx512dq = (ebx >> 17) & 1;
    f.avx512bw = (ebx >> 30) & 1;
    f.avx512vl = (ebx >> 31) & 1;
    const unsigned max_subleaf = eax;
    f.amx_bf16 = (edx >> 22) & 1;
    f.amx_tile = (edx >> 24) & 1;
    f.amx_int8 = (edx >> 25) & 1;
    f.avx512fp16 = (edx >> 23) & 1;
    if (max_subleaf >= 1) {
      __cpuid_count(7, 1, eax, ebx, ecx, edx);
      f.avx512bf16 = (eax >> 5) & 1;
    }
  }
  // Both AVX-512 and AMX are only usable when the OS enabled the state; without
  // an XSAVE-enabled build (some VMs) the first instruction is SIGILL.
  unsigned ax = 0, bx = 0, cx = 0, dx = 0;
  __cpuid_count(1, 0, ax, bx, cx, dx);
  const bool osxsave = (cx >> 27) & 1;
  const bool avx = (cx >> 28) & 1;
  if (osxsave && avx) {
    unsigned long long xcr0 = 0;
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
    const bool zmm = (xcr0 & 0xe6) == 0xe6;
    f.avx512f &= zmm;
    f.avx512bw &= zmm;
    f.avx512vl &= zmm;
    f.avx512dq &= zmm;
    f.avx512bf16 &= zmm;
    f.avx512fp16 &= zmm;
    const bool tiles = (xcr0 & 0x60000) == 0x60000;  // XTILECFG | XTILEDATA
    f.amx_tile &= tiles;
    f.amx_bf16 &= tiles;
    f.amx_int8 &= tiles;
  } else {
    f.avx512f = f.avx512bw = f.avx512vl = f.avx512dq = false;
    f.avx512bf16 = f.avx512fp16 = false;
    f.amx_tile = f.amx_bf16 = f.amx_int8 = false;
  }
#elif defined(__aarch64__)
  const unsigned long hw = getauxval(AT_HWCAP);
  const unsigned long hw2 = getauxval(AT_HWCAP2);
  f.asimd = (hw & (1UL << 1)) != 0;   // HWCAP_ASIMD
  f.bf16 = (hw2 & HWCAP2_BF16) != 0;  // bfdot/bfmmla
  f.i8mm = (hw2 & HWCAP2_I8MM) != 0;
  f.sve_bits = static_cast<int64_t>(getauxval(AT_HWCAP) & (1UL << 22) ? 1 : 0);
#endif
  return f;
}

}  // namespace

const CpuFeatures& cpu() {
  static const CpuFeatures f = detect();
  return f;
}

bool amx_init() {
#if defined(__x86_64__)
  if (!cpu().amx_tile || !cpu().amx_bf16) return false;
  // The tile registers are in a kernel-managed XSAVE state that is off by
  // default; without this the first tile instruction is SIGILL.
  static const bool permitted =
      syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
  return permitted;
#else
  return false;
#endif
}

const char* isa_name(int isa) {
  switch (isa) {
    case kIsaVec:
      return "vec";
    case kIsaNeon:
      return "neon";
    case kIsaAmx:
      return "amx";
    default:
      return "scalar";
  }
}

int vector_isa() {
  const CpuFeatures& f = cpu();
#if defined(__x86_64__)
  if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq) return kIsaVec;
  return kIsaScalar;
#elif defined(__aarch64__)
  if (f.asimd) return kIsaNeon;
  return kIsaScalar;
#else
  return kIsaScalar;
#endif
}

}  // namespace attn
