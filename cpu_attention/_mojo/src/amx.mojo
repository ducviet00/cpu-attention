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

**The one thing to be careful about** is that LLVM does not track tile registers,
so it will happily hoist a `tileloadd` above a `tdpbf16ps` that still reads that
tile (a WAR hazard, silently wrong results). Every memory-touching primitive
here therefore carries a `~{memory}` clobber, and `tdpbf16ps` does too: correct
but conservative, and the reason a Mojo AMX kernel must be A/B'd against the C
one rather than assumed equal. Loads and dots are already ordered in this
kernel, so the cost is software-pipelining opportunity, not correctness.

Verified compile-time only here (`mojo build --emit asm --target-cpu
sapphirerapids`): the emitted assembly contains the instructions above.
`test_amx_mojo.mojo` runs a 16x16x32 bf16 GEMM against a scalar reference and
needs an AMX CPU, i.e. run it on the Xeon box.
"""

from std.ffi import external_call
from std.memory import MutPointer, stack_allocation
from std.origin import MutUntrackedOrigin
from std.os import abort
from std.sys import inlined_assembly


@always_inline
def _tmm[i: Int]() -> StaticString:
    """``%tmm<i>`` for a compile-time tile index."""
    comptime if i == 0:
        return "%tmm0"
    comptime if i == 1:
        return "%tmm1"
    comptime if i == 2:
        return "%tmm2"
    comptime if i == 3:
        return "%tmm3"
    comptime if i == 4:
        return "%tmm4"
    comptime if i == 5:
        return "%tmm5"
    comptime if i == 6:
        return "%tmm6"
    comptime if i == 7:
        return "%tmm7"
    abort("AMX tile index out of range")


comptime PALETTE_BF16 = 1
"""Tile palette: 1 = bf16 rows (64 bytes = 32 bf16 per tile row)."""


def amx_init() -> Bool:
    """Ask the kernel for tile-data permission; False means AMX is unusable.

    Required before any tile instruction (`#XF` state is off by default), and
    it is per thread, so each worker calls this once.
    """
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    return (
        Int(
            external_call["syscall", Int, Int64, Int64, Int64](
                Int64(157),  # SYS_arch_prctl on x86_64
                Int64(ARCH_REQ_XCOMP_PERM),
                Int64(XFEATURE_XTILEDATA),
            )
        )
        == 0
    )


def tile_config_bf16(
    rows: Int, colsb: Int, cfg: MutPointer[UInt8, MutUntrackedOrigin]
) -> None:
    """Fill a 64-byte tile config for `rows` tiles of `colsb` bytes each.

    The layout that matters (and that cost the x86 agent an hour): palette at
    byte 0, `colsb[8]` are **uint16** at offset 16, `rows[8]` are **uint8** at
    offset 48. Writing colsb as bytes makes `ldtilecfg` fault.
    """
    for i in range(64):
        cfg.unsafe_store(i, UInt8(0))
    cfg.unsafe_store(0, UInt8(PALETTE_BF16))
    for i in range(8):
        var c16 = MutPointer[UInt16, MutUntrackedOrigin](
            unsafe_from_address=Int(cfg) + 16
        )
        c16.unsafe_store(i, UInt16(colsb))
        cfg.unsafe_store(48 + i, UInt8(rows))


@always_inline
def tile_cfg_load(cfg: Int) -> None:
    """`ldtilecfg` from a 64-byte, 64-byte-aligned config."""
    inlined_assembly[
        "ldtilecfg ($0)",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](cfg)


@always_inline
def tile_release() -> None:
    """`tilerelease`: drop the config (must be re-established afterwards)."""
    inlined_assembly[
        "tilerelease",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def tile_load[tile: Int](base: Int, stride: Int) -> None:
    """`tileloadd`: 16 rows of the tile, `stride` bytes apart, from `base`."""
    comptime asm = "tileloadd ($0,$1,1), " + _tmm[tile]()
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
    inlined_assembly[
        asm,
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def tile_dpbf16ps[dst: Int, a: Int, b: Int]() -> None:
    """`tdpbf16ps`: `dst += a * b`, 16x16 bf16 tiles accumulating in fp32."""
    comptime asm = (
        "tdpbf16ps " + _tmm[dst]() + ", " + _tmm[a]() + ", " + _tmm[b]()
    )
    inlined_assembly[
        asm,
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()
