"""Cost + accuracy of std.math exp vs a hand-rolled exp2 trick on SIMD[4]."""

from std.math import exp
from std.memory import bitcast
from std.time import perf_counter_ns


def exp_fast(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """exp(x) as 2^(x*log2e): round, degree-5 poly on [-0.5, 0.5], exponent add.

    No branches, no special cases; inputs below -127*ln2 give exactly 0 because
    the exponent field underflows to zero bits.
    """
    var y = x * SIMD[DType.float32, 4](1.4426950408889634)
    y = y.lt(SIMD[DType.float32, 4](-127.0)).select(
        SIMD[DType.float32, 4](-127.0), y
    )
    var n = (y + SIMD[DType.float32, 4](12582912.0)) - SIMD[DType.float32, 4](
        12582912.0
    )
    var f = y - n
    var p = SIMD[DType.float32, 4](0.0013333558146428443) * f + SIMD[
        DType.float32, 4
    ](0.009618129107628477)
    p = p * f + SIMD[DType.float32, 4](0.05550410866482158)
    p = p * f + SIMD[DType.float32, 4](0.2402265069591007)
    p = p * f + SIMD[DType.float32, 4](0.6931471805599453)
    p = p * f + SIMD[DType.float32, 4](1.0)
    var e = (n.cast[DType.int32]() + SIMD[DType.int32, 4](127)) << SIMD[
        DType.int32, 4
    ](23)
    return p * bitcast[DType.float32](e)


def main():
    print("fast(-1e38) =", exp_fast(SIMD[DType.float32, 4](-3.4028234663852886e38)))
    print("fast(0) =", exp_fast(SIMD[DType.float32, 4](0.0)))
    print("fast(-1) =", exp_fast(SIMD[DType.float32, 4](-1.0)))
    print("std(-1)  =", exp(SIMD[DType.float32, 4](-1.0)))
    var worst = Float32(0.0)
    for i in range(200):
        var x = SIMD[DType.float32, 4](
            -Float32(i) * 0.05, -Float32(i) * 0.07, Float32(i) * 0.03, -1.5
        )
        var a = exp(x)
        var b = exp_fast(x)
        for j in range(4):
            var rel = abs(b[j] - a[j]) / a[j]
            if rel > worst:
                worst = rel
    print("max rel err vs exp over [-10, 6]:", worst)

    # Throughput: eight independent calls per iteration, so the dependency
    # chain is not what is being measured.
    comptime N = 500000
    var r0 = SIMD[DType.float32, 4](0.0)
    var t0 = perf_counter_ns()
    for i in range(N):
        var x = SIMD[DType.float32, 4](Float32(i % 7) * -0.1)
        var a = SIMD[DType.float32, 4](0.0)
        for j in range(8):
            a += exp(x + SIMD[DType.float32, 4](Float32(j)))
        r0 = a
    var t1 = perf_counter_ns()
    var r1 = SIMD[DType.float32, 4](0.0)
    for i in range(N):
        var x = SIMD[DType.float32, 4](Float32(i % 7) * -0.1)
        var a = SIMD[DType.float32, 4](0.0)
        for j in range(8):
            a += exp_fast(x + SIMD[DType.float32, 4](Float32(j)))
        r1 = a
    var t2 = perf_counter_ns()
    print("std exp   ns/vector", Float64(t1 - t0) / (8 * N))
    print("fast exp  ns/vector", Float64(t2 - t1) / (8 * N))
    print(r0[0], r1[0])
