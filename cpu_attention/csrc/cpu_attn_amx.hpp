// AMX kernel (x86-64 with amx-tile + amx-bf16). Stub until the Xeon work lands:
// `available()` false routes every dense call to the AVX-512 path.
#pragma once

#include "cpu_attn_impl.hpp"

namespace attn {
namespace amx {

bool available() { return false; }

void dense(const DenseJob&, int) {}

}  // namespace amx
}  // namespace attn
