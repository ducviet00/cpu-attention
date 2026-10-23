// Public C++ surface of the kernel: the jobs, the two entry points and the
// workspace. `binding.cpp` is the only caller.
#pragma once

#include <cstddef>

#include "cpu_attn_impl.hpp"

namespace attn {

// Run the paged path: scatter K/V into the cache (`do_store`) and/or attend
// over it (`do_attend`). `reshape_and_cache` uses the store-only form.
void attention_entry(const PagedJob& job, int workers, bool do_store,
                     bool do_attend);

// Run the dense path over whole prompts.
void dense_entry(const DenseJob& job, int workers);

// Diagnostics: the ISA the dense path would pick, as an `Isa` value.
int dense_isa_name();

}  // namespace attn
