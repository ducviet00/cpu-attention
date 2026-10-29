"""What every kernel shares: the build tier, lane width, pointers, `exp2`, bf16.

One source builds several tiers (`scripts/build.py`): each `.so` is compiled for
an explicit target, and the Python loader picks the highest tier this CPU and OS
support. So every ISA decision below is `comptime`, made from the tier's target
features; nothing here probes the running CPU except `amx_permitted` (AMX needs
OS permission, which only a syscall can ask for).
"""

from std.ffi import external_call
from std.memory import MutPointer, bitcast
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic, simd_width_of
from std.sys.info import CompilationTarget

comptime LANES = simd_width_of[DType.float32]()
"""fp32 lanes per vector: 16 on AVX-512, 8 on AVX2, 4 on NEON/SSE."""

comptime X86_BF16 = CompilationTarget.is_x86() and CompilationTarget._has_feature[
    "avx512bf16"
]()
"""`vdpbf16ps`: 2 bf16 MACs per fp32 lane, exact products, fp32 accumulate."""

comptime NEON_BF16 = CompilationTarget.has_neon() and CompilationTarget._has_feature[
    "bf16"
]()
"""`bfdot` / `bfmmla` (Armv8.6 BF16)."""

comptime BF16_DOT = X86_BF16 or NEON_BF16

comptime BCAST_FOLD = CompilationTarget.is_x86()
"""A scalar broadcast folds into the FMA's memory operand (`{1to16}`), so a
broadcast per query dim / probability is free; NEON instead loads 4 values as a
vector and multiplies by a lane (`fmla v, v, v.s[i]`)."""

comptime AMX_TIER = X86_BF16
"""The AVX-512-BF16 tier also carries the AMX kernels (inline asm assembles for
any x86 target); they run only after `amx_permitted()`."""


def tier_name() -> String:
    comptime if X86_BF16:
        return "x86-avx512bf16"
    elif CompilationTarget.is_x86() and LANES == 8:
        return "x86-avx2"
    elif CompilationTarget.is_x86():
        return "x86-64"
    elif NEON_BF16:
        return "aarch64-bf16"
    else:
        return "aarch64-neon" if CompilationTarget.has_neon() else "portable"


@always_inline
def ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


comptime LOG2E = 1.4426950408889634


@always_inline
def vmax[w: Int](a: SIMD[DType.float32, w], b: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """Fast lane-wise maximum (one `vmaxps` on x86, one `fmaxnm` on aarch64);
    `max()` on SIMD floats is NaN-propagating and emits compare + select on x86."""
    comptime if CompilationTarget.has_neon() and w == 4:
        return llvm_intrinsic[
            "llvm.maxnum.v4f32", SIMD[DType.float32, w], has_side_effect=False
        ](a, b)
    else:
        return a.gt(b).select(a, b)


@always_inline
def exp2[
    degree: Int, width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """`2^x` for `x <= 0`: round to nearest, a polynomial on the fraction, the
    integer part inserted into the exponent. About 9-10 instructions against
    `std.math.exp`'s ~22 (docs/Mojo.md §7). Inputs below -127 clamp, so a masked
    lane (-inf) yields exactly 0 rather than NaN (on NEON ~1e-38, see below).

    `degree` trades accuracy for instructions: 2 sits inside a bf16 rounding
    (relative 2e-3), 4 and 5 are minimax at 2.6e-6 and 3e-6, 6 (Taylor) at
    1.2e-7 -- the paged fp32 gate (1e-6 absolute) needs 6.
    """
    # NEON takes the exponent from the bits of the magic-added value: `fcvt*`
    # issues on one pipe at 1/cycle on Neoverse (measured), so converting n cost
    # the fp32 flash softmax ~8%. The integer exponent add needs x >= -126 (a
    # masked lane is then ~1e-38, not 0; fully masked rows are zeroed by the
    # callers).
    comptime NEON_EXP = CompilationTarget.has_neon()
    var xc = vmax(x, SIMD[DType.float32, width](-126.0 if NEON_EXP else -127.0))
    var n: SIMD[DType.float32, width]
    var y = SIMD[DType.float32, width](0)
    comptime if X86_BF16 and width == 16:
        var x16 = rebind[SIMD[DType.float32, 16]](xc)
        n = rebind[SIMD[DType.float32, width]](
            llvm_intrinsic[
                "llvm.x86.avx512.mask.rndscale.ps.512",
                SIMD[DType.float32, 16],
                has_side_effect=False,
            ](x16, Int32(8), x16, Int16(-1), Int32(4))
        )
    else:
        comptime MAGIC = 12582912.0  # 1.5 * 2^23: adding it rounds to nearest
        y = xc + MAGIC
        n = y - MAGIC
    var r = xc - n
    var p: SIMD[DType.float32, width]
    comptime if degree <= 2:
        p = r.fma(2.3473140e-1, 6.9335938e-1)
        p = p.fma(r, 1.0)
    elif degree <= 4:
        # Remez minimax on [-0.5, 0.5]: relative 2.6e-6, degree 5's class.
        p = SIMD[DType.float32, width](9.570101908e-3)
        p = p.fma(r, 5.591786032e-2)
        p = p.fma(r, 2.402474483e-1)
        p = p.fma(r, 6.931218147e-1)
        p = p.fma(r, 9.999992614e-1)
    elif degree == 5:
        p = SIMD[DType.float32, width](1.3392112e-3)
        p = p.fma(r, 9.6180963e-3)
        p = p.fma(r, 5.5503282e-2)
        p = p.fma(r, 2.4022652e-1)
        p = p.fma(r, 6.9314718e-1)
        p = p.fma(r, 1.0)
    else:
        p = SIMD[DType.float32, width](1.5403530e-4)
        p = p.fma(r, 1.3333558e-3)
        p = p.fma(r, 9.6181291e-3)
        p = p.fma(r, 5.5504109e-2)
        p = p.fma(r, 2.4022651e-1)
        p = p.fma(r, 6.9314718e-1)
        p = p.fma(r, 1.0)
    comptime if NEON_EXP:
        return bitcast[DType.float32, width](
            bitcast[DType.int32, width](p) + (bitcast[DType.int32, width](y) << 23)
        )
    var e = (n.cast[DType.int32]() + 127) << 23
    return p * bitcast[DType.float32, width](e)


@always_inline
def exp2s[degree: Int](x: Float32) -> Float32:
    """Scalar `exp2` (the online-softmax rescale factor)."""
    return exp2[degree, 1](SIMD[DType.float32, 1](x))[0]


@always_inline
def dot_bf16[
    width: Int
](
    acc: SIMD[DType.float32, width],
    a: SIMD[DType.bfloat16, 2 * width],
    b: SIMD[DType.bfloat16, 2 * width],
) -> SIMD[DType.float32, width]:
    """`acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]`: products of two bf16 values are
    exact in fp32, so this is widening math with a different summation order.
    `has_side_effect=False` matters: without it the calls stay ordered and the
    phase is ~4x slower (docs/Mojo.md §5)."""
    comptime if X86_BF16 and width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps.512",
            SIMD[DType.float32, width],
            has_side_effect=False,
        ](acc, a, b)
    elif NEON_BF16 and width == 4:
        return llvm_intrinsic[
            "llvm.aarch64.neon.bfdot",
            SIMD[DType.float32, width],
            has_side_effect=False,
        ](acc, a, b)
    else:
        var af = a.cast[DType.float32]()
        var bf = b.cast[DType.float32]()
        var ea = af.deinterleave()
        var eb = bf.deinterleave()
        comptime H = SIMD[DType.float32, width]
        return rebind[H](ea[0]).fma(
            rebind[H](eb[0]), rebind[H](ea[1]).fma(rebind[H](eb[1]), acc)
        )


@always_inline
def to_bf16[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.bfloat16, width]:
    """Round-to-nearest-even fp32 -> bf16 (one `vcvtneps2bf16` on AVX-512-BF16)."""
    return x.cast[DType.bfloat16]()


def amx_permitted() -> Bool:
    """Ask the kernel for AMX tile-data permission (`arch_prctl`), once per
    process; False on every non-AMX tier or when the OS refuses."""
    comptime if not AMX_TIER:
        return False
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    comptime SYS_ARCH_PRCTL = 158
    return (
        Int(
            external_call["syscall", Int, Int64, Int64, Int64](
                Int64(SYS_ARCH_PRCTL),
                Int64(ARCH_REQ_XCOMP_PERM),
                Int64(XFEATURE_XTILEDATA),
            )
        )
        == 0
    )
