// Keep CPUID subleaf decoding consistent with the compiler's runtime detector.
#include <cassert>
#include <iostream>

#include "cpu_isa.hpp"

int main() {
  const auto& f = attn::cpu();
#if defined(__x86_64__)
  __builtin_cpu_init();
  assert(f.avx512f == bool(__builtin_cpu_supports("avx512f")));
  assert(f.avx512bw == bool(__builtin_cpu_supports("avx512bw")));
  assert(f.avx512vl == bool(__builtin_cpu_supports("avx512vl")));
  assert(f.avx512dq == bool(__builtin_cpu_supports("avx512dq")));
  assert(f.avx512bf16 == bool(__builtin_cpu_supports("avx512bf16")));
#endif
  std::cout << "ISA detection OK; bf16=" << f.avx512bf16
            << " amx=" << f.amx_bf16 << '\n';
}
