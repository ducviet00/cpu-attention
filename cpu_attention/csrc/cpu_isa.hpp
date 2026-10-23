// Runtime ISA detection: CPUID on x86, HWCAP on aarch64.
//
// The build compiles every path the platform can have and picks at run time, so
// one artifact runs on any machine of its architecture. That is the opposite of
// the Mojo build, which targets the host CPU and has to be rebuilt per box.
#pragma once

#include <cstdint>

namespace attn {

enum Isa : int { kIsaNone = 0, kIsaVec = 1, kIsaNeon = 2, kIsaAmx = 3, kIsaScalar = 4 };

struct CpuFeatures {
  // x86
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512dq = false;
  bool avx512bf16 = false;
  bool amx_tile = false;
  bool amx_bf16 = false;
  bool amx_int8 = false;
  bool avx512fp16 = false;  // unused today; the fp16 seam is here
  // aarch64
  bool asimd = false;
  bool bf16 = false;
  bool i8mm = false;
  int64_t sve_bits = 0;
  // set once the kernel's tile state is usable
  bool amx_permitted = false;
};

const CpuFeatures& cpu();
const char* isa_name(int isa);

// Ask the kernel for permission to use tile registers and mirror the XTILEDATA
// component of the extended state. Returns true when tile ops are legal.
// Called once from the plan build, never from a kernel.
bool amx_init();

// The ISA this process should use for the vector path.
int vector_isa();

}  // namespace attn
