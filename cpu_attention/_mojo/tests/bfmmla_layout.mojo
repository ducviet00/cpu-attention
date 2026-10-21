"""Check the ``bfmmla`` operand layout against a scalar reference.

``vbfmmlaq_f32(d, n, m)`` adds ``n @ m`` into ``d`` seen as a 2x2 matrix, with
the contraction over *four* elements and both operands stored as two 4-lane
halves::

    d = (d0, d1, d2, d3)  ->  [[d0, d1], [d2, d3]]
    d[i][j] += dot(n[4i .. 4i+3], m[4j .. 4j+3])

i.e. ``n`` holds *rows* (lanes 0-3 = row 0) and ``m`` holds *columns* (lanes 0-3
= column 0) -- not the interleaved ``(k0c0, k0c1, k1c0, ...)`` some references
show. For Q@K^T that is ``n`` = 2 query rows x 4 head dims and ``m`` = 2 key
positions x 4 head dims, and for P@V ``n`` = 2 query rows x 4 keys, ``m`` = 2
head dims x 4 keys, so both "pair" operand layouts are four positions of one
row/column, contiguous.

Run on an arm64 box with ``mojo run cpu_attention/_mojo/tests/bfmmla_layout.mojo``.
"""

from std.sys import llvm_intrinsic


def bfmmla(
    acc: SIMD[DType.float32, 4],
    a: SIMD[DType.bfloat16, 8],
    b: SIMD[DType.bfloat16, 8],
) -> SIMD[DType.float32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.bfmmla", SIMD[DType.float32, 4], has_side_effect=False
    ](acc, a, b)


def main():
    # A: 2 rows x 4 contraction dims, B: 2 columns x 4 contraction dims.
    var a = SIMD[DType.bfloat16, 8](1, 2, 3, 4, 10, 20, 30, 40)
    var b = SIMD[DType.bfloat16, 8](1, 2, 3, 4, 5, 6, 7, 8)
    var acc = bfmmla(SIMD[DType.float32, 4](0.0), a, b)

    # reference, with the documented packing
    var want = SIMD[DType.float32, 4](0.0)
    for i in range(2):
        for j in range(2):
            var s = Scalar[DType.float32](0.0)
            for k in range(4):
                s += Scalar[DType.float32](a[4 * i + k]) * Scalar[DType.float32](
                    b[4 * j + k]
                )
            want[2 * i + j] = s

    print("acc ", acc)
    print("want", want)
    if acc == want:
        print("bfmmla layout OK")
    else:
        print("bfmmla layout MISMATCH")
