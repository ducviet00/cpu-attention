"""Throughput ceilings on this core: which MMA instruction is actually fastest?

The whole ARM plan rests on one number -- that `bfmmla` issues at 2/cycle and is
therefore ~96% consumed by vLLM's kernel at S=4096/D=128 -- so the alternatives
have to be measured, not assumed:

* `fmmla` (FP16 MMA): same 16 MACs per instruction as `bfmmla`, but fp16 has a
  10-bit mantissa instead of bf16's 7. If it also issues at 2/cycle it is *free
  precision*, which would matter for the fp32 rows (a 2-term fp16 split instead
  of 32 `fmla`) and for how much parity headroom any lossy trick has. If it
  issues at 4/cycle it is a 2x lever.
* `smmla` (int8 MMA, i8mm): 32 MACs per instruction. The previous campaign used
  "2x bfmmla" as the payoff of int8 Q@K^T; if the real rate is 2/cycle on the
  same port then the payoff is exactly 2x per MAC, and if it is 4/cycle int8 is
  worth re-examining despite its parity problem.
* `bfmmla` + `fmla` co-issue: if 8 fmla really fit inside 8 bfmmla's 4 cycles,
  then 25% of the MACs could move off the MMA port for free -- but only if the
  loop has spare *issue* slots, which a real Q@K^T body does not (26 instructions
  per 16 bfmmla). Measure it here, then count instructions in the loop that is
  supposed to host it.

Run on the ARM box (needs NEON + bf16 + i8mm + fp16):

    mojo run cpu_attention/_mojo/tests/mma_peak.mojo

Every loop keeps 8 independent accumulators so it measures throughput, not
latency, and the clock is measured in the same process (these boxes' clock moves
20-30% between sessions, so a hard-coded GHz invalidates the comparison).
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


def fmmla(
    acc: SIMD[DType.float32, 4],
    a: SIMD[DType.float16, 8],
    b: SIMD[DType.float16, 8],
) -> SIMD[DType.float32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.fmmla", SIMD[DType.float32, 4], has_side_effect=False
    ](acc, a, b)


def smmla(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.smmla", SIMD[DType.int32, 4], has_side_effect=False
    ](acc, a, b)


comptime M = 2000000
"""Outer iterations: 8 instructions each, ~2 s per probe at 3.3 GHz."""


def clock_ghz() -> Float64:
    comptime N = 20000000
    var x = 0
    var t0 = perf_counter_ns()
    for _ in range(N):
        x += 1
    var t1 = perf_counter_ns()
    print(x)
    return Float64(N) / Float64(t1 - t0)


def main():
    var ghz = clock_ghz()
    print("clock GHz:", ghz)

    # --- bfmmla: 8 independent accumulators -------------------------------
    var ba = SIMD[DType.bfloat16, 8](1.0)
    var bb = SIMD[DType.bfloat16, 8](1.0)
    var b0 = SIMD[DType.float32, 4](0.0)
    var b1 = b0
    var b2 = b0
    var b3 = b0
    var b4 = b0
    var b5 = b0
    var b6 = b0
    var b7 = b0
    var t0 = perf_counter_ns()
    for _ in range(M):
        b0 = bfmmla(b0, ba, bb)
        b1 = bfmmla(b1, ba, bb)
        b2 = bfmmla(b2, ba, bb)
        b3 = bfmmla(b3, ba, bb)
        b4 = bfmmla(b4, ba, bb)
        b5 = bfmmla(b5, ba, bb)
        b6 = bfmmla(b6, ba, bb)
        b7 = bfmmla(b7, ba, bb)
    var dt = Float64(perf_counter_ns() - t0) / Float64(8 * M)
    var bf_rate = ghz / dt
    print("bfmmla  ns/instr", dt, " instr/cycle", bf_rate,
          " GFLOP/s/core", 32.0 / dt)

    # --- fmmla (fp16) -----------------------------------------------------
    var fa = SIMD[DType.float16, 8](1.0)
    var fb = SIMD[DType.float16, 8](1.0)
    var f0 = SIMD[DType.float32, 4](0.0)
    var f1 = f0
    var f2 = f0
    var f3 = f0
    var f4 = f0
    var f5 = f0
    var f6 = f0
    var f7 = f0
    t0 = perf_counter_ns()
    for _ in range(M):
        f0 = fmmla(f0, fa, fb)
        f1 = fmmla(f1, fa, fb)
        f2 = fmmla(f2, fa, fb)
        f3 = fmmla(f3, fa, fb)
        f4 = fmmla(f4, fa, fb)
        f5 = fmmla(f5, fa, fb)
        f6 = fmmla(f6, fa, fb)
        f7 = fmmla(f7, fa, fb)
    dt = Float64(perf_counter_ns() - t0) / Float64(8 * M)
    var f16_rate = ghz / dt
    print("fmmla   ns/instr", dt, " instr/cycle", f16_rate,
          " GFLOP/s/core", 32.0 / dt)

    # --- smmla (int8) -----------------------------------------------------
    var sa = SIMD[DType.int8, 16](1)
    var sb = SIMD[DType.int8, 16](1)
    var s0 = SIMD[DType.int32, 4](0)
    var s1 = s0
    var s2 = s0
    var s3 = s0
    var s4 = s0
    var s5 = s0
    var s6 = s0
    var s7 = s0
    t0 = perf_counter_ns()
    for _ in range(M):
        s0 = smmla(s0, sa, sb)
        s1 = smmla(s1, sa, sb)
        s2 = smmla(s2, sa, sb)
        s3 = smmla(s3, sa, sb)
        s4 = smmla(s4, sa, sb)
        s5 = smmla(s5, sa, sb)
        s6 = smmla(s6, sa, sb)
        s7 = smmla(s7, sa, sb)
    dt = Float64(perf_counter_ns() - t0) / Float64(8 * M)
    var i8_rate = ghz / dt
    print("smmla   ns/instr", dt, " instr/cycle", i8_rate,
          " GFLOP/s/core(int)", 64.0 / dt)

    # --- co-issue: 8 bfmmla + 8 fmla in one loop --------------------------
    # If this takes the same time as 8 bfmmla alone, the FMA pipes are free
    # during a matmul loop and 25% of the MACs could migrate onto them.
    var fa2 = SIMD[DType.float32, 4](1.0)
    var fe = SIMD[DType.float32, 4](1.0)
    var fa3 = fa2
    var fa4 = fa2
    var fa5 = fa2
    var fa6 = fa2
    var fa7 = fa2
    var fa8 = fa2
    var fa9 = fa2
    t0 = perf_counter_ns()
    for _ in range(M):
        b0 = bfmmla(b0, ba, bb)
        fa2 = fa2 * fe + fe
        b1 = bfmmla(b1, ba, bb)
        fa3 = fa3 * fe + fe
        b2 = bfmmla(b2, ba, bb)
        fa4 = fa4 * fe + fe
        b3 = bfmmla(b3, ba, bb)
        fa5 = fa5 * fe + fe
        b4 = bfmmla(b4, ba, bb)
        fa6 = fa6 * fe + fe
        b5 = bfmmla(b5, ba, bb)
        fa7 = fa7 * fe + fe
        b6 = bfmmla(b6, ba, bb)
        fa8 = fa8 * fe + fe
        b7 = bfmmla(b7, ba, bb)
        fa9 = fa9 * fe + fe
    dt = Float64(perf_counter_ns() - t0) / Float64(M)
    print("8bfmmla+8fmla  cycles/iter", dt * ghz,
          " (4.0 = fully co-issued, 6.4 = serialised)")

    print("sums:", b0[0] + b7[0] + f0[0] + f7[0] + Float32(s0[0] + s7[0])
          + fa2[0] + fa9[0])
