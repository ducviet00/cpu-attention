"""Intel AMX (tile) primitives, emitted from Mojo as inline assembly.

Why this file exists: LLVM's AMX *intrinsics* (`llvm.x86.tileloadd64` and
friends) are unreachable from Mojo because they traffic in the `x86_amx` type,
which Mojo cannot name. That is not the same as "AMX is unreachable": the
`x86_amx` type only exists so a compiler can track tile registers, and a tile
register operand can equally be written *into the instruction text*. MAX reaches
Apple's AMX exactly this way (`max/kernels/src/linalg/arch/cpu/
apple_amx_intrinsics.mojo` emits raw `.word` encodings through
`std.sys.inlined_assembly`), and this file does the same for the Intel form:

    ldtilecfg (ptr)                     ; 64-byte config, palette 1
    tileloadd (base,stride,1), %tmmK
    tdpbf16ps %tmmD, %tmmA, %tmmB
    tilestored %tmmK, (base,stride,1)
    tilerelease

Tile numbers are compile-time parameters interpolated into the asm string, so
one call site serves every tile — no 8^3 variants.

LLVM does not track tile registers, so it would hoist a `tileloadd` above a
`tdpbf16ps` that still reads that tile (a WAR hazard, silently wrong results).
Every memory-touching primitive therefore carries a `~{memory}` clobber, and
`tdpbf16ps` does too: correct but conservative (the cost is software-pipelining
opportunity). `tests/test_flash.py` exercises the AMX policy on AMX CPUs.
"""

from std.collections.string.string_span import get_static_string
from std.sys import inlined_assembly
from std.sys.info import CompilationTarget


@always_inline
def _tmm[i: Int]() -> StaticString:
    """``%tmm<i>`` for a compile-time tile index."""
    comptime assert 0 <= i < 8, "AMX has tiles 0-7"
    return get_static_string["%tmm", String(i)]()


@always_inline
def tile_cfg_load(cfg: Int) -> None:
    """`ldtilecfg` from a 64-byte, 64-byte-aligned config."""
    comptime if CompilationTarget.is_x86():
        inlined_assembly[
            "ldtilecfg ($0)",
            NoneType,
            constraints="r,~{memory}",
            has_side_effect=True,
        ](cfg)


@always_inline
def tile_load[tile: Int](base: Int, stride: Int) -> None:
    """`tileloadd`: 16 rows of the tile, `stride` bytes apart, from `base`."""
    comptime asm = "tileloadd ($0,$1,1), " + _tmm[tile]()
    comptime if CompilationTarget.is_x86():
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,~{memory}",
            has_side_effect=True,
        ](base, stride)


@always_inline
def tile_store[tile: Int](base: Int, stride: Int) -> None:
    """`tilestored`: write the tile back, 16 rows `stride` bytes apart."""
    comptime asm = "tilestored " + _tmm[tile]() + ", ($0,$1,1)"
    comptime if CompilationTarget.is_x86():
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,~{memory}",
            has_side_effect=True,
        ](base, stride)


@always_inline
def tile_zero[tile: Int]() -> None:
    """`tilezero`: register-only, no memory operand."""
    comptime asm = "tilezero " + _tmm[tile]()
    # Register-only, so no memory clobber (see `tile_dpbf16ps`).
    comptime if CompilationTarget.is_x86():
        inlined_assembly[
            asm,
            NoneType,
            constraints="",
            has_side_effect=True,
        ]()


@always_inline
def tile_dpbf16ps[dst: Int, a: Int, b: Int]() -> None:
    """`tdpbf16ps`: `dst += a * b`, 16x16 bf16 tiles accumulating in fp32."""
    # AT&T operand order is (src2, src1, dst) -- the destination tile comes
    # LAST, unlike the Intel-syntax form gcc's intrinsic emits as an
    # alternative. Writing the destination first assembles fine and silently
    # computes `b += a * dst_old`, which measured as "C is all zeros" in the
    # smoke test (tmm2 was the zeroed accumulator and also a source).
    comptime asm = "tdpbf16ps " + _tmm[b]() + ", " + _tmm[a]() + ", " + _tmm[dst]()
    # No `~{memory}` here: the dot touches no memory, and the clobber cost 10-28%
    # in the kernel (it forces the compiler to treat every dot as a full barrier,
    # so the surrounding loads/stores cannot be scheduled around it). Ordering is
    # still safe because `has_side_effect=True` keeps side-effecting asm in
    # program order, and the tile loads/stores around it keep their own clobber.
    # Measured, alternating processes, against the gcc-built C kernel.
    comptime if CompilationTarget.is_x86():
        inlined_assembly[
            asm,
            NoneType,
            constraints="",
            has_side_effect=True,
        ]()
