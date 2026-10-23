// AVX-512 translation unit: the build gives it -mavx512f/-bw/-vl/-dq/-bf16 so
// the intrinsics in cpu_attn_vec.hpp are available, while every other TU stays
// baseline.
#include "cpu_attn_vec.hpp"
