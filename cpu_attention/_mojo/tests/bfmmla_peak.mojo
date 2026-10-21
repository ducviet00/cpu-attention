"""Measure the box's clock and the NEON bfmmla throughput ceiling."""

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
    comptime N = 20000000
    var x = 0
    var t0 = perf_counter_ns()
    for _ in range(N):
        x += 1
    var t1 = perf_counter_ns()
    print("clock GHz (1 add/iter):", Float64(N) / Float64(t1 - t0))

    var a = SIMD[DType.bfloat16, 8](1.0)
    var b = SIMD[DType.bfloat16, 8](1.0)
    var acc = SIMD[DType.float32, 4](0.0)
    comptime M = 2000000
    var t2 = perf_counter_ns()
    for _ in range(M):
        # 8 independent chains: throughput, not latency.
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
        acc = bfmmla(acc, a, b)
    var t3 = perf_counter_ns()
    var per = Float64(t3 - t2) / Float64(8 * M)
    print("bfmmla ns/instr (dependent chain):", per)
    print("  -> GFLOP/s single core:", 32.0 / per)

    # Independent accumulators: 8 chains of 1, i.e. 8 in flight.
    var a0 = SIMD[DType.float32, 4](0.0)
    var a1 = SIMD[DType.float32, 4](0.0)
    var a2 = SIMD[DType.float32, 4](0.0)
    var a3 = SIMD[DType.float32, 4](0.0)
    var a4 = SIMD[DType.float32, 4](0.0)
    var a5 = SIMD[DType.float32, 4](0.0)
    var a6 = SIMD[DType.float32, 4](0.0)
    var a7 = SIMD[DType.float32, 4](0.0)
    var t4 = perf_counter_ns()
    for _ in range(M):
        a0 = bfmmla(a0, a, b)
        a1 = bfmmla(a1, a, b)
        a2 = bfmmla(a2, a, b)
        a3 = bfmmla(a3, a, b)
        a4 = bfmmla(a4, a, b)
        a5 = bfmmla(a5, a, b)
        a6 = bfmmla(a6, a, b)
        a7 = bfmmla(a7, a, b)
    var t5 = perf_counter_ns()
    var per2 = Float64(t5 - t4) / Float64(8 * M)
    print("bfmmla ns/instr (8 chains):", per2)
    print("  -> GFLOP/s single core:", 32.0 / per2)
    print(a0[0] + a1[0] + a2[0] + a3[0] + a4[0] + a5[0] + a6[0] + a7[0] + acc[0])
