"""Throughput ceilings on this core: which matrix-multiply instruction exists, and how fast?

The ARM plan rests on one number -- that `bfmmla` issues at 2/cycle and is
therefore ~95% consumed by vLLM's kernel at S=4096/D=128 -- so what else the
part offers has to be measured, not assumed:

* `fmmla` (FP16 MMA): **does not exist** here. It is the `f32mm` extension, which
  no shipping core implements, and `mojo run` dies with "Cannot select:
  intrinsic %llvm.aarch64.neon.fmmla" with and without `--target-cpu`. So an
  fp16 operand pays `bfmmla`'s rate at best.
* `smmla` (int8 MMA, i8mm): 32 MACs per instruction, twice `bfmmla`'s. If it
  really issues at 2/cycle then int8 is 3.2x `fmla` and 2x `bfmmla` per MAC --
  which is what makes a *two-term int8 split* interesting for the fp32 rows
  (16 bits of operand at 64 MACs/cycle against `fmla`'s 13.2), and irrelevant
  for the bf16 rows (8-bit operands miss the 3e-2 gate; see
  `tests/int8_qk_error.py`).
* `bfmmla` + `fmla` co-issue: if 8 fmla fit inside 8 bfmmla's 4 cycles then up
  to 25% of the MACs can move off the MMA port for free.

Run on the ARM box:

    .venv/bin/mojo run --target-cpu neoverse-v3 \
        cpu_attention/_mojo/tests/mma_peak.mojo

Two traps this file has already fallen into, both of which produced plausible
garbage: the clock loop (`x += 1` 20 M times) was folded to a constant, so the
"GHz" was 689655 and every instr/cycle figure with it; and the int8 loop with
constant operands was strength-reduced, reporting 8 instructions/cycle. So the
clock accumulator starts from a runtime value and every operand is loaded from
memory filled at runtime.
"""

from std.memory import stack_allocation
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


def smmla(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.smmla", SIMD[DType.int32, 4], has_side_effect=False
    ](acc, a, b)


comptime M = 2000000
"""Outer iterations: 8 instructions each, ~0.5-2 s per probe."""


def clock_ghz() -> Float64:
    # Start from a runtime value, or the loop is folded and the "clock" comes
    # back 7 orders of magnitude too fast.
    comptime N = 20000000
    var x = Int(perf_counter_ns() % 7)
    var t0 = perf_counter_ns()
    for _ in range(N):
        x += 1
    var t1 = perf_counter_ns()
    print("  (clock accumulator, must not be a constant):", x)
    return Float64(N) / Float64(t1 - t0)


def main():
    var ghz = clock_ghz()
    print("clock GHz:", ghz)

    # Operands live in memory filled from the clock, so nothing folds.
    var seed = Int(perf_counter_ns() % 3) + 1
    var store = stack_allocation[64, DType.float32]()
    for i in range(64):
        store.unsafe_store(i, Float32(seed + i % 5))

    var ba = SIMD[DType.bfloat16, 8](0.0)
    var bb = SIMD[DType.bfloat16, 8](0.0)
    for i in range(8):
        ba[i] = Scalar[DType.bfloat16](store.unsafe_load(i) * 0.25)
        bb[i] = Scalar[DType.bfloat16](store.unsafe_load(i + 8) * 0.25)

    # --- bfmmla: 8 independent accumulators -------------------------------
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
    print("bfmmla  ns/instr", dt, " cycles/instr", dt * ghz,
          " GFLOP/s/core", 32.0 / dt)

    # --- smmla (int8) -----------------------------------------------------
    var sa = SIMD[DType.int8, 16](0)
    var sb = SIMD[DType.int8, 16](0)
    for i in range(16):
        sa[i] = Scalar[DType.int8](Int(store.unsafe_load(i)) % 7 - 3)
        sb[i] = Scalar[DType.int8](Int(store.unsafe_load(i + 16)) % 5 - 2)
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
    print("smmla   ns/instr", dt, " cycles/instr", dt * ghz,
          " GMAC/s/core", 32.0 / dt)

    # --- co-issue: 8 bfmmla + 8 fmla in one loop --------------------------
    var fa = SIMD[DType.float32, 4](1.0)
    var fe = SIMD[DType.float32, 4](1.0)
    var fa1 = fa
    var fa2 = fa
    var fa3 = fa
    var fa4 = fa
    var fa5 = fa
    var fa6 = fa
    var fa7 = fa
    var fa8 = fa
    t0 = perf_counter_ns()
    for _ in range(M):
        b0 = bfmmla(b0, ba, bb)
        fa1 = fa1 * fe + fe
        b1 = bfmmla(b1, ba, bb)
        fa2 = fa2 * fe + fe
        b2 = bfmmla(b2, ba, bb)
        fa3 = fa3 * fe + fe
        b3 = bfmmla(b3, ba, bb)
        fa4 = fa4 * fe + fe
        b4 = bfmmla(b4, ba, bb)
        fa5 = fa5 * fe + fe
        b5 = bfmmla(b5, ba, bb)
        fa6 = fa6 * fe + fe
        b6 = bfmmla(b6, ba, bb)
        fa7 = fa7 * fe + fe
        b7 = bfmmla(b7, ba, bb)
        fa8 = fa8 * fe + fe
    dt = Float64(perf_counter_ns() - t0) / Float64(M)
    print("8bfmmla+8fmla cycles/iter", dt * ghz,
          " (2.0 = co-issued and bfmmla-bound, 4.0 = bfmmla only,",
          " >4 = fmla costs cycles)")

    print("sums:", b0[0] + b7[0] + Float32(s0[0] + s7[0]) + fa1[0] + fa8[0])
