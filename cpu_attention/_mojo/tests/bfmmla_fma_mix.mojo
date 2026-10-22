"""Can bfmmla and fp32 fma co-issue? (throughput probe)

If the bf16 MMA pipe and the fp32 FMA pipe are independent, a loop of
`n bfmmla + m fma` should cost max(n/2, m/3.3) cycles per iteration rather than
(n/2 + m/3.3). If they share the vector datapath, the two terms add and a
"split the contraction across both pipes" kernel buys nothing.
"""

from std.sys import llvm_intrinsic
from std.time import perf_counter_ns


def bfmmla(
    acc: SIMD[DType.float32, 4],
    a: SIMD[DType.bfloat16, 8],
    b: SIMD[DType.bfloat16, 8],
) -> SIMD[DType.float32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.bfmmla", SIMD[DType.float32, 4], has_side_effect=False
    ](acc, a, b)


def main():
    comptime M = 2000000
    var a = SIMD[DType.bfloat16, 8](1.0)
    var b = SIMD[DType.bfloat16, 8](1.0)
    var f = SIMD[DType.float32, 4](1.0000001)
    var g = SIMD[DType.float32, 4](0.9999999)

    var a0 = SIMD[DType.float32, 4](0.0)
    var a1 = SIMD[DType.float32, 4](0.0)
    var a2 = SIMD[DType.float32, 4](0.0)
    var a3 = SIMD[DType.float32, 4](0.0)
    var a4 = SIMD[DType.float32, 4](0.0)
    var a5 = SIMD[DType.float32, 4](0.0)
    var a6 = SIMD[DType.float32, 4](0.0)
    var a7 = SIMD[DType.float32, 4](0.0)
    var f0 = SIMD[DType.float32, 4](0.0)
    var f1 = SIMD[DType.float32, 4](0.0)
    var f2 = SIMD[DType.float32, 4](0.0)
    var f3 = SIMD[DType.float32, 4](0.0)
    var f4 = SIMD[DType.float32, 4](0.0)
    var f5 = SIMD[DType.float32, 4](0.0)
    var f6 = SIMD[DType.float32, 4](0.0)
    var f7 = SIMD[DType.float32, 4](0.0)

    # warm
    for _ in range(10000):
        a0 = bfmmla(a0, a, b)
        f0 = f * f0 + g
    var t0 = perf_counter_ns()
    for _ in range(M):
        a0 = bfmmla(a0, a, b)
        a1 = bfmmla(a1, a, b)
        a2 = bfmmla(a2, a, b)
        a3 = bfmmla(a3, a, b)
        a4 = bfmmla(a4, a, b)
        a5 = bfmmla(a5, a, b)
        a6 = bfmmla(a6, a, b)
        a7 = bfmmla(a7, a, b)
    var t1 = perf_counter_ns()
    var bf = Float64(t1 - t0) / Float64(8 * M)
    print("8 bfmmla/iter          ns/instr:", bf, " -> /cycle:", 1.0 / bf / 3.25)

    var u0 = perf_counter_ns()
    for _ in range(M):
        f0 = f * f0 + g
        f1 = f * f1 + g
        f2 = f * f2 + g
        f3 = f * f3 + g
        f4 = f * f4 + g
        f5 = f * f5 + g
        f6 = f * f6 + g
        f7 = f * f7 + g
    var u1 = perf_counter_ns()
    var fm = Float64(u1 - u0) / Float64(8 * M)
    print("8 fma/iter             ns/instr:", fm, " -> /cycle:", 1.0 / fm / 3.25)

    # 8 bfmmla + 8 fma per iteration. Shared pipe predicts 8*bf + 8*fm;
    # independent pipes predict max(8*bf, 8*fm).
    var v0 = perf_counter_ns()
    for _ in range(M):
        a0 = bfmmla(a0, a, b)
        f0 = f * f0 + g
        a1 = bfmmla(a1, a, b)
        f1 = f * f1 + g
        a2 = bfmmla(a2, a, b)
        f2 = f * f2 + g
        a3 = bfmmla(a3, a, b)
        f3 = f * f3 + g
        a4 = bfmmla(a4, a, b)
        f4 = f * f4 + g
        a5 = bfmmla(a5, a, b)
        f5 = f * f5 + g
        a6 = bfmmla(a6, a, b)
        f6 = f * f6 + g
        a7 = bfmmla(a7, a, b)
        f7 = f * f7 + g
    var v1 = perf_counter_ns()
    var mx = Float64(v1 - v0) / Float64(8 * M)
    print("8 bfmmla + 8 fma /iter ns/instr-pair:", mx)
    print("  shared-pipe prediction:", 8.0 * bf + 8.0 * fm,
          " independent prediction:", 8.0 * max(bf, fm))
    print(a0[0] + a1[0] + a2[0] + a3[0] + a4[0] + a5[0] + a6[0] + a7[0]
          + f0[0] + f1[0] + f2[0] + f3[0] + f4[0] + f5[0] + f6[0] + f7[0])
