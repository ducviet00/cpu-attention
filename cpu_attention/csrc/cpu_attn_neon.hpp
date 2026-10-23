// NEON kernel (aarch64 with bf16). Stub until the Graviton work lands:
// `available()` false routes every call to the portable fallback.
#pragma once

#include "cpu_attn_impl.hpp"

namespace attn {
namespace neon {

bool available() { return false; }

void dense(const DenseJob&, int) {}

void paged(const PagedJob&, int) {}

void store(const PagedJob&, int) {}

}  // namespace neon
}  // namespace attn
