"""Dense attention on Intel AMX, in Mojo.

The bf16 path for x86 CPUs that have AMX (`amx-tile` + `amx-bf16`): one
`tdpbf16ps` accumulates a 16x16 fp32 tile from bf16 operands, which is 372
flops/cycle against AVX-512 `vdpbf16ps`'s 32 -- the difference between being
1.7-6.8x behind torch SDPA on a Xeon and being 1.4-2.4x ahead of it.

This started as a structural port of a gcc-built C kernel that lived in
`cpu_attention/_csrc/`; it was A/B'd against it on the Xeon and measured
identical (0.97-1.03x on every shape, both dtypes), so the C version is gone --
one build artifact, no ctypes, no second language.

## Structure

One work item = `QROWS` (32) consecutive query rows of one (batch, head), swept
over the whole key range. 32 rows because `tdpbf16ps` produces a 16x16 fp32 tile
and four accumulators (2 row groups x 2 key/dim groups) is what fits beside the
operand tiles in AMX's eight-tile file.

    QK^T  A = Q (16 rows x 32 head dims), read straight from the caller's tensor
              with `tileloadd(stride = q_stride_t * 2)` -- no Q packing, and any
              strides work; a partial last row block is a zero-padded copy.
          B = K packed by `_pack_k_block`.
    P@V   A = P (16 rows x 32 keys, row-major bf16)
          B = V packed by `_pack_v_block`.

Two-level key blocking: QK runs in 32-key chunks (4 accumulators = 32 rows x 32
keys) into an fp32 logits buffer, then one online-softmax pass over the whole
`kt`-key super-tile (<= `MAXKT`), then P@V sweeps that super-tile. That amortises
the output-accumulator load/store over `kt` keys instead of 32, and it is why the
softmax is one flat pass rather than a per-tile dependency chain.

Tile config, constant for the whole kernel and set once per worker:

    tiles 0,1  A  (QK: Q rows 0-15, 16-31; P@V: P rows 0-15, 16-31)
    tiles 2,3  B  (QK: keys 0-15, 16-31; P@V: output dims 0-15, 16-31)
    tiles 4-7  C  (row group x {B0, B1})

all `rows = 16`, `colsb = 64` (32 bf16 or 16 fp32), palette 1.

## The `tdpbf16ps` B operand, the one easy thing to get wrong

A 1 KB B tile is 16x16 **32-bit units**, and bf16 element (row `r`, index `t`)
holds `B[2r + (t & 1)][t >> 1]`. So the contraction pair (two adjacent head dims)
lives *inside* a 32-bit unit and the N index is the unit: the K pack is a 16x16
**32-bit** transpose, not a 16-bit one. V is the same shape with the pair being
two adjacent *keys* of one output dim. Recovered by single-bit probes in the C
version (`tests/amx_smoke.mojo` on a 16x16x32 GEMM, and single-bit probes in the
C version before it was retired); vLLM's `reshape_and_cache` says the same
("head elements packed as quadwords").

## Numerics

Same class as the AVX-512 bf16 path it replaces: Q@K^T is bf16 x bf16 with fp32
accumulation, P is rounded to bf16 once, P@V accumulates in fp32, and the row sum
is divided out at the end. `exp2` is a degree-4 polynomial plus an exponent
bit-insert, clamped at -127 so a fully masked lane is *exactly* +0 (a bf16
denormal in an AMX tile is not free, and `0 * garbage` must be 0).

## Permission

`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` must have succeeded before
any tile instruction, or the first one is SIGILL. The Python side does that once
(`cpu_attention._kernel.amx_available()`, cached) and only calls in when it
worked, so this module does not pay a syscall per call.
"""

from std.ffi import external_call
from std.math import fma, iota
from std.memory import MutPointer, bitcast, stack_allocation
from std.origin import MutUntrackedOrigin
from std.sys.info import CompilationTarget
from amx import (
    tile_cfg_load,
    tile_config_bf16,
    tile_dpbf16ps,
    tile_load,
    tile_release,
    tile_store,
    tile_zero,
)

comptime AMX = (
    CompilationTarget.is_x86()
    and CompilationTarget._has_feature["amx-tile"]()
    and CompilationTarget._has_feature["amx-bf16"]()
)
"""Whether this build targets a CPU with AMX.

Everything below is instantiated only when this is true, so a non-AMX target
emits no tile code at all -- no second source file and no runtime dispatch are
needed to keep older CPUs working.
"""

comptime QROWS = 32
"""Query rows per work item: two AMX row groups (tiles 0/1 and the C tiles)."""

comptime KEYCH = 32
"""Keys per QK step: four accumulators hold 32 rows x 32 keys."""

comptime MAXKT = 512
"""Upper bound on the keys in one softmax super-tile (sizes the logits buffer)."""

comptime DMAX = 128
"""Largest supported head_dim: the per-worker buffers are sized for it."""

comptime NEG_INF = -3.4028234663852886e38


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


@inline(.always)
def _ea(base: Int, elem: Int, esize: Int) -> Int:
    """Byte address of element ``elem`` of an ``esize``-byte type at ``base``.

    Mojo has no typed pointers, so every offset has to be scaled explicitly --
    and mixing the two conventions is silent: half of this file's first version
    added element offsets to a byte address, which is right only for offset 0
    (single batch, first head, first key block) and produced NaNs and a 2x in
    P@V everywhere else. Every address in this file now goes through _ea or is
    already a byte quantity, and `_EA_CHECK` below keeps it that way.
    """
    return base + elem * esize


@inline(.always)
def _exp2(x: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """2**x, and exactly +0 once x is far enough below zero.

    Degree-4 polynomial on [-0.5, 0.5] (relative error 4e-5, two orders below
    the bf16 the probabilities are rounded to) and an exponent bit-insert for
    the 2**n scale. The clamp at -127 makes a fully masked lane underflow to +0,
    which P@V needs: `0 * garbage` must be 0.

    Do not "optimise" this without an in-process A/B against this exact form:
    a degree-3 polynomial for bf16, `round()` instead of the magic-constant
    trick, a one-sided clamp and `llvm.ldexp` (one `vscalefps` per vector
    instead of the bit insert) were all tried together and measured **0.73-0.91x**
    -- i.e. 10-27% *slower* on every shape up to 1024 keys -- while a
    single-threaded probe of the same loop showed it 8% faster. The probe runs
    one thread with an L1-resident buffer; the kernel runs four. Suspect the
    `vscalefps` dependency on the vector port that the tile loads also need;
    untested, and the honest statement is that the replacement lost.
    """
    comptime V = SIMD[DType.float32, 16]
    comptime MAGIC = 12582912.0  # 1.5 * 2**23: round-to-nearest in fp32
    var xc = min(max(x, V(-127.0)), V(127.0))
    var n = (xc + V(MAGIC)) - V(MAGIC)
    var f = xc - n
    var p = V(0.009618129107628477)
    p = fma(p, f, V(0.05550410866482158))
    p = fma(p, f, V(0.2402265069591007))
    p = fma(p, f, V(0.6931471805599453))
    p = fma(p, f, V(1.0))
    var ni = n.cast[DType.int32]()
    var bits = (ni + SIMD[DType.int32, 16](127)) << SIMD[DType.int32, 16](23)
    return p * bitcast[DType.float32, 16](bits)


# --------------------------------------------------- fp32 -> bf16 hi/lo split

# fp32 has no AMX instruction, so the same kernel runs fp32 by splitting every
# operand into two bf16 halves and keeping three terms of each product:
#
#     x = hi + lo,  hi = bf16(x),  lo = bf16(x - hi)      (|lo| <= 2^-8 |x|)
#     x*y ~= hi_x*hi_y + hi_x*lo_y + lo_x*hi_y             (drops lo*lo, 2^-16)
#
# Products are exact and the accumulation is fp32, so the result is fp32-accurate
# at 3x the tile ops -- against AMX's ~11x instruction advantage over AVX-512
# fp32 FMA. Both phases need it (either one alone in plain bf16 is 1e-2);
# measured 5.5-7.6e-5 max|diff| against SDPA on every benchmark shape, gate 2e-4.


@inline(.always)
def _hi(x: SIMD[DType.float32, 16]) -> SIMD[DType.bfloat16, 16]:
    """The bf16 half of an fp32 vector (round to nearest even)."""
    return x.cast[DType.bfloat16]()


@inline(.always)
def _lo(
    x: SIMD[DType.float32, 16], hi: SIMD[DType.bfloat16, 16]
) -> SIMD[DType.bfloat16, 16]:
    """The remainder, as bf16: `bf16(x - fp32(bf16(x)))`."""
    return (x - hi.cast[DType.float32]()).cast[DType.bfloat16]()


@inline(.always)
def _row32(
    lo: SIMD[DType.bfloat16, 16], hi: SIMD[DType.bfloat16, 16]
) -> SIMD[DType.int32, 16]:
    """Join two 16-bf16 halves into the 16x 32-bit units a tile row wants."""
    return bitcast[DType.int32, 16](lo.join(hi))


# --------------------------------------------------------------------- pack


@inline(.always)
def _tp16_i32(m: Array[SIMD[DType.int32, 16], 16]) -> Array[
    SIMD[DType.int32, 16], 16
]:
    """16x16 transpose of 32-bit units, in place, two registers of temporaries.

    Stage `k` swaps bit `k` of the row index with bit `k` of the lane index, and
    the product of the four stages is the transpose. The first version built a
    *new* 16-register array per stage (`o`, then `a = o`), so 32 zmm were live on
    a 32-register machine; each stage now writes back the pair it read, so the
    peak is 16 + 2. Measured: worth ~25% of the K pack (a tile stays in
    registers instead of round-tripping 2 KB through the stack).
    """
    var a = m.copy()
    comptime for k in range(4):
        comptime step = 1 << k
        comptime for i in range(16):
            comptime if (i >> k) & 1 == 0:
                comptime b = i | step
                comptime if k == 0:
                    var t0 = a[i].shuffle[
                        0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30
                    ](a[b])
                    var t1 = a[i].shuffle[
                        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 1:
                    var t0 = a[i].shuffle[
                        0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
                    ](a[b])
                    var t1 = a[i].shuffle[
                        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 2:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27
                    ](a[b])
                    var t1 = a[i].shuffle[
                        4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 3:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
                    ](a[b])
                    var t1 = a[i].shuffle[
                        8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
    return a.copy()


struct PackJob(Copyable, ImplicitlyCopyable, Movable):
    """What the K/V pack needs: raw addresses plus element strides."""

    var k: Int
    var v: Int
    var kp: Int
    var vp: Int
    var seq_k: Int
    var head_dim: Int
    var k_stride_b: Int
    var k_stride_t: Int
    var k_stride_h: Int
    var v_stride_b: Int
    var v_stride_t: Int
    var v_stride_h: Int
    var num_kv_heads: Int
    var batch: Int
    var kb16: Int
    var kb32: Int
    var dc32: Int
    var dc16: Int
    var kp_bh: Int
    var vp_bh: Int

    def __init__(out self):
        self.k = 0
        self.v = 0
        self.kp = 0
        self.vp = 0
        self.seq_k = 0
        self.head_dim = 0
        self.k_stride_b = 0
        self.k_stride_t = 0
        self.k_stride_h = 0
        self.v_stride_b = 0
        self.v_stride_t = 0
        self.v_stride_h = 0
        self.num_kv_heads = 0
        self.batch = 0
        self.kb16 = 0
        self.kb32 = 0
        self.dc32 = 0
        self.dc16 = 0
        self.kp_bh = 0
        self.vp_bh = 0


@inline(.always)
def _pack_k_block[
    fp32: Bool
](
    job: PackJob, batch: Int, head: Int, kb: Int, dc: Int
) -> None:
    """One 1 KB QK B tile: 16 keys x 16 head-dim pairs, transposed.

    Tile row `r` holds `[K[k0+j][d0+2r], K[k0+j][d0+2r+1]]` for `j` in 0..15,
    i.e. the contraction pair (two adjacent head dims) inside a 32-bit unit and
    the 16 keys as the N index. Keys past `seq_k` are zero, and V is zero there
    too, so a stale probability cannot inject a NaN through a padded key.
    """
    comptime stride = 1024 if fp32 else 512
    var dst32 = _ptr[Int32](
        Int(job.kp)
        + (
            (batch * job.num_kv_heads + head) * job.kp_bh
            + (kb * job.dc32 + dc) * stride
        )
        * 2
    )
    var k0 = kb * 16
    var d0 = dc * 32
    var rows = Array[SIMD[DType.int32, 16], 16](uninitialized=True)
    if fp32:
        # One pass produces both slabs: hi in tiles 0..511, lo in 512..1023.
        var srcf = _ptr[Float32](
            _ea(Int(job.k), batch * job.k_stride_b + head * job.k_stride_h, 4)
        )
        var rows_lo = Array[SIMD[DType.int32, 16], 16](uninitialized=True)
        comptime for i in range(16):
            if k0 + i < job.seq_k:
                var f0 = _ptr[Float32](
                    _ea(Int(srcf), (k0 + i) * job.k_stride_t + d0, 4)
                ).unsafe_load[width=16]()
                var f1 = _ptr[Float32](
                    _ea(Int(srcf), (k0 + i) * job.k_stride_t + d0 + 16, 4)
                ).unsafe_load[width=16]()
                var h0 = _hi(f0)
                var h1 = _hi(f1)
                rows[i] = _row32(h0, h1)
                rows_lo[i] = _row32(_lo(f0, h0), _lo(f1, h1))
            else:
                rows[i] = SIMD[DType.int32, 16](0)
                rows_lo[i] = SIMD[DType.int32, 16](0)
        var cols = _tp16_i32(rows)
        var cols_lo = _tp16_i32(rows_lo)
        comptime for i in range(16):
            dst32.unsafe_store[width=16](i * 16, cols[i])
            dst32.unsafe_store[width=16](256 + i * 16, cols_lo[i])
    else:
        var src = _ptr[BFloat16](
            _ea(job.k, batch * job.k_stride_b + head * job.k_stride_h, 2)
        )
        comptime for i in range(16):
            if k0 + i < job.seq_k:
                rows[i] = _ptr[Int32](
                    _ea(Int(src), (k0 + i) * job.k_stride_t + d0, 2)
                ).unsafe_load[width=16]()
            else:
                rows[i] = SIMD[DType.int32, 16](0)
        var cols = _tp16_i32(rows)
        comptime for i in range(16):
            dst32.unsafe_store[width=16](i * 16, cols[i])


@inline(.always)
def _pack_v_block[
    fp32: Bool
](
    job: PackJob, batch: Int, head: Int, kb: Int, dc: Int
) -> None:
    """One 1 KB P@V B tile: 32 keys x 16 output dims.

    Element (tile row `r`, index `t`) must hold `V[2r + (t & 1)][t >> 1]`, so for
    each output dim `u` the two keys' values are interleaved as a 32-bit unit:
    one `interleave` of two 16-element key rows produces a whole tile row.
    """
    var k0 = kb * 32
    var d0 = dc * 16
    var slab_base = _ea(
        job.vp,
        (batch * job.num_kv_heads + head) * job.vp_bh
        + (kb * job.dc16 + dc) * (1024 if fp32 else 512),
        2,
    )
    if fp32:
        # hi and lo as two separate pointers: one base + a shared 512-element
        # offset expression silently landed both stores on the same slab.
        var hi_p = _ptr[BFloat16](slab_base)
        var lo_p = _ptr[BFloat16](slab_base + 512 * 2)  # +512 bf16
        var srcf = _ptr[Float32](
            _ea(Int(job.v), batch * job.v_stride_b + head * job.v_stride_h, 4)
        )
        comptime for r in range(16):
            var ka = k0 + 2 * r
            var fa = SIMD[DType.float32, 16](0)
            var fc = SIMD[DType.float32, 16](0)
            if ka < job.seq_k:
                fa = srcf.unsafe_load[width=16](ka * job.v_stride_t + d0)
            if ka + 1 < job.seq_k:
                fc = srcf.unsafe_load[width=16]((ka + 1) * job.v_stride_t + d0)
            # Locals in the same scope as their use, exactly as the K pack does:
            # assigning an outer `var` inside an `if` and then reading it in a
            # later `_lo(...)` argument handed the *pre-assignment* value to the
            # subtraction, so the lo slab came out equal to the hi slab (a
            # silent 2x in P@V -- caught by comparing the packed workspace
            # against the source, not by reading the kernel).
            var h0 = _hi(fa)
            var h1 = _hi(fc)
            hi_p.unsafe_store[width=32](r * 32, h0.interleave(h1))
            lo_p.unsafe_store[width=32](
                r * 32, _lo(fa, h0).interleave(_lo(fc, h1))
            )
    else:
        var dst = _ptr[BFloat16](slab_base)
        var src = _ptr[BFloat16](
            _ea(job.v, batch * job.v_stride_b + head * job.v_stride_h, 2)
        )
        comptime for r in range(16):
            var ka = k0 + 2 * r
            var a = SIMD[DType.bfloat16, 16](0)
            var c = SIMD[DType.bfloat16, 16](0)
            if ka < job.seq_k:
                a = src.unsafe_load[width=16](ka * job.v_stride_t + d0)
            if ka + 1 < job.seq_k:
                c = src.unsafe_load[width=16]((ka + 1) * job.v_stride_t + d0)
            dst.unsafe_store[width=32](r * 32, a.interleave(c))


@inline(.always)
def _pack_k_slab[
    fp32: Bool
](job: PackJob, batch: Int, head: Int, kb: Int) -> None:
    """One task = one 16-key block, every dim chunk: one linear read of K."""
    for dc in range(job.dc32):
        _pack_k_block[fp32](job, batch, head, kb, dc)


@inline(.always)
def _pack_v_slab[
    fp32: Bool
](job: PackJob, batch: Int, head: Int, kb: Int) -> None:
    """One task = one 32-key block, every dim chunk: one linear read of V."""
    for dc in range(job.dc16):
        _pack_v_block[fp32](job, batch, head, kb, dc)


# ------------------------------------------------------------------ attention


struct AmxJob(Copyable, ImplicitlyCopyable, Movable):
    """One call, as raw addresses: the packed K/V slabs plus the caller's q/out.
    """

    var q: Int
    var kp: Int
    var vp: Int
    var dst: Int
    var batch: Int
    var seq_q: Int
    var seq_k: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var q_stride_b: Int
    var q_stride_t: Int
    var q_stride_h: Int
    var kp_bh: Int
    var vp_bh: Int
    var o_stride_b: Int
    var o_stride_t: Int
    var o_stride_h: Int
    var scale: Float32
    var causal: Bool
    var kt: Int
    var dc32: Int
    var dc16: Int
    var fp32: Bool
    var mode: Int
    """Profiling only (CPU_ATTENTION_AMX_MODE): bit 0 skips the pack, bit 1 the
    whole attention loop, bit 2 the V pack, bit 3 the K pack, bits 4/5/6 the
    Q@K^T, softmax and P@V phases, bit 7 the per-item Q prologue (the Q tile
    loads and the zero-padded panel), bit 8 the per-item output epilogue.
    Results are garbage whenever a bit is set."""

    def __init__(out self):
        self.q = 0
        self.kp = 0
        self.vp = 0
        self.dst = 0
        self.batch = 0
        self.seq_q = 0
        self.seq_k = 0
        self.num_heads = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.q_stride_b = 0
        self.q_stride_t = 0
        self.q_stride_h = 0
        self.kp_bh = 0
        self.vp_bh = 0
        self.o_stride_b = 0
        self.o_stride_t = 0
        self.o_stride_h = 0
        self.scale = 1.0
        self.causal = False
        self.kt = 256
        self.dc32 = 0
        self.dc16 = 0
        self.fp32 = False
        self.mode = 0


@inline(.always)
def _amx_dense_item[
    fp32: Bool, D: Int
](
    job: AmxJob,
    logits: MutPointer[Float32, MutUntrackedOrigin],
    prob: MutPointer[BFloat16, MutUntrackedOrigin],
    acc: MutPointer[Float32, MutUntrackedOrigin],
    qpad: MutPointer[BFloat16, MutUntrackedOrigin],
    qpad_lo: MutPointer[BFloat16, MutUntrackedOrigin],
    prob_lo: MutPointer[BFloat16, MutUntrackedOrigin],
    batch: Int,
    head: Int,
    q0: Int,
) -> None:
    """Attend ``QROWS`` query rows of one (batch, head) over the key range."""
    comptime V16 = SIMD[DType.float32, 16]
    comptime I16 = SIMD[DType.int32, 16]
    # `D` is a compile-time parameter: every offset, loop bound and tile index
    # below then folds to a constant. The kernel used to read head_dim out of
    # the job, which left a dynamic multiply in each tile-load address and a
    # dynamic bound on the dim-chunk loop.
    comptime DC32 = D // 32
    comptime DC16 = D // 16
    comptime ASTRIDE = D * 4
    var kt = job.kt
    var seq_k = job.seq_k
    var nrows = min(QROWS, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var kvh = head
    if job.num_heads != job.num_kv_heads:
        kvh = head // (job.num_heads // job.num_kv_heads)
    var kbase = _ptr[BFloat16](
        _ea(job.kp, (batch * job.num_kv_heads + kvh) * job.kp_bh, 2)
    )
    var vbase = _ptr[BFloat16](
        _ea(job.vp, (batch * job.num_kv_heads + kvh) * job.vp_bh, 2)
    )
    var obase = _ptr[BFloat16](
        _ea(job.dst, batch * job.o_stride_b + head * job.o_stride_h, 2)
    )

    # Q is read straight from the caller's tensor when the row block is full;
    # otherwise copy the real rows into a zero-padded slab, because a partial
    # block still computes 32 rows and the padding must not contribute.
    var qptr = _ptr[BFloat16](0)
    var qptr_lo = _ptr[BFloat16](0)
    var qstride = 0
    if (job.mode & 128) != 0:
        # profiling: skip the Q prologue (panel build); the pointers stay valid
        # so the tile loads below read whatever is there.
        qptr = qpad
        qstride = D * 2
    elif fp32:
        # The A operand needs no transpose, so splitting Q is a straight
        # conversion into a per-work-item hi/lo panel. Only the rows past
        # `nrows` need zeroing -- a full block needs none, and zeroing all
        # QROWS*D first and then overwriting it was 2048-4096 stores per work
        # item, i.e. ~10% of an fp32 call at 256:16:128.
        for i in range(nrows * D, QROWS * D):
            qpad.unsafe_store(i, Scalar[DType.bfloat16](0))
            qpad_lo.unsafe_store(i, Scalar[DType.bfloat16](0))
        for r in range(nrows):
            var srcf = _ptr[Float32](
                _ea(
                    job.q,
                    batch * job.q_stride_b
                    + (q0 + r) * job.q_stride_t
                    + head * job.q_stride_h,
                    4,
                )
            )
            for j in range(0, D, 16):
                var f = srcf.unsafe_load[width=16](j)
                var h = _hi(f)
                qpad.unsafe_store[width=16](r * D + j, h)
                qpad_lo.unsafe_store[width=16](r * D + j, _lo(f, h))
        qptr = qpad
        qptr_lo = qpad_lo
        qstride = D * 2
    elif nrows == QROWS:
        qptr = _ptr[BFloat16](
            _ea(
                job.q,
                batch * job.q_stride_b + q0 * job.q_stride_t + head * job.q_stride_h,
                2,
            )
        )
        qstride = job.q_stride_t * 2
    else:
        for i in range(QROWS * D):
            qpad.unsafe_store(i, Scalar[DType.bfloat16](0))
        for r in range(nrows):
            var s = _ptr[BFloat16](
                _ea(
                    job.q,
                    batch * job.q_stride_b
                    + (q0 + r) * job.q_stride_t
                    + head * job.q_stride_h,
                    2,
                )
            )
            for i in range(D):
                qpad.unsafe_store(r * D + i, s.unsafe_load(i))
        qptr = qpad
        qstride = D * 2

    comptime kstride = 1024 if fp32 else 512  # bf16 per packed (hi, lo) pair
    comptime vstride = 1024 if fp32 else 512

    var lstride = kt * 4
    var pstride = kt * 2
    comptime astride = ASTRIDE
    # Running max and sum per row, as two 16-lane vectors (rows 0-15, 16-31).
    # The max starts at 0, not -inf: scaled logits are O(10), so exp2 cannot
    # overflow, and a row that sees no key at all has every logit at -3e38,
    # which clamps to p = 0 and is written out as zeros.
    var m0 = V16(0.0)
    var m1 = V16(0.0)
    var s0 = V16(0.0)
    var s1 = V16(0.0)
    var rm = stack_allocation[16, Float32]()

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)
    loop_end = max(loop_end, 0)

    var c1v = V16(job.scale * 1.4426950408889634)
    var c1f = job.scale * 1.4426950408889634
    var ninf = V16(NEG_INF)
    var lane = iota[DType.int32, 16](0)

    var t0 = 0
    while t0 < loop_end:
        var nk = min(kt, loop_end - t0)
        var nch = (nk + KEYCH - 1) // KEYCH
        # Lanes the QK actually wrote. The mask, the softmax and the P buffer
        # only ever need these: running them over all of `kt` wastes (kt-nk)
        # lanes on every partial super-tile, which is most of them at short S.
        var nkp = nch * KEYCH

        # ---- Q@K^T: KEYCH keys at a time, 32 rows, into fp32 logits ----
        for ch in range(0 if (job.mode & 16) != 0 else nch):
            var k0 = t0 + ch * KEYCH
            # kstride, not 512: fp32 pairs a (hi, lo) slab per (key block, dim
            # chunk), so the pitch doubles. Getting this wrong is invisible for
            # key block 0 and wrong for every later one (it showed up only at
            # seq_k > 32, and only in fp32).
            var kb = (k0 // 16) * DC32 * kstride
            tile_zero[4]()
            tile_zero[5]()
            tile_zero[6]()
            tile_zero[7]()
            comptime for dc in range(DC32):
                var kt_hi = Int(kbase) + (kb + dc * kstride) * 2
                if (job.mode & 128) == 0:
                    tile_load[0](Int(qptr) + dc * 64, qstride)
                    tile_load[1](Int(qptr) + dc * 64 + 16 * qstride, qstride)
                tile_load[2](kt_hi, 64)
                tile_load[3](kt_hi + DC32 * kstride * 2, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                if fp32:
                    # Q_hi@K_lo, then Q_lo@K_hi (K_hi reloads from L1).
                    tile_load[2](kt_hi + 1024, 64)
                    tile_load[3](kt_hi + DC32 * kstride * 2 + 1024, 64)
                    tile_dpbf16ps[4, 0, 2]()
                    tile_dpbf16ps[5, 0, 3]()
                    tile_dpbf16ps[6, 1, 2]()
                    tile_dpbf16ps[7, 1, 3]()
                    tile_load[0](Int(qptr_lo) + dc * 64, qstride)
                    tile_load[1](Int(qptr_lo) + dc * 64 + 16 * qstride, qstride)
                    tile_load[2](kt_hi, 64)
                    tile_load[3](kt_hi + DC32 * kstride * 2, 64)
                    tile_dpbf16ps[4, 0, 2]()
                    tile_dpbf16ps[5, 0, 3]()
                    tile_dpbf16ps[6, 1, 2]()
                    tile_dpbf16ps[7, 1, 3]()
            var lp = Int(logits) + ch * KEYCH * 4
            tile_store[4](lp, lstride)
            tile_store[5](lp + 64, lstride)
            tile_store[6](lp + 16 * lstride, lstride)
            tile_store[7](lp + 16 * lstride + 64, lstride)

        # ---- causal / padding mask, per row, on a partial super-tile only ----
        var lim_min = seq_k
        if job.causal:
            lim_min = min(seq_k, q0 + diag + 1)
        if t0 + nkp > lim_min:
            for r in range(nrows):
                var lim = seq_k
                if job.causal:
                    lim = min(seq_k, q0 + r + diag + 1)
                lim -= t0
                if lim >= nkp:
                    continue
                var lp = logits.unsafe_offset(r * kt)
                if lim <= 0:
                    for j in range(0, nkp, 16):
                        lp.unsafe_store[width=16](j, ninf)
                else:
                    var j = (lim // 16) * 16
                    while j < nkp:
                        var keep = I16(Int32(lim)).gt(lane + I16(Int32(j)))
                        lp.unsafe_store[width=16](
                            j, keep.select(lp.unsafe_load[width=16](j), ninf)
                        )
                        j += 16

        # ---- online softmax over the whole super-tile ----
        for rg in range(0 if (job.mode & 32) != 0 else 2):
            var lp = logits.unsafe_offset(rg * 16 * kt)
            var m_old = m1 if rg == 1 else m0
            for r in range(16):
                var v = lp.unsafe_load[width=16](r * kt)
                var j = 16
                while j < nkp:
                    v = max(v, lp.unsafe_load[width=16](r * kt + j))
                    j += 16
                rm.unsafe_store(r, Scalar[DType.float32](v.reduce_max()))
            var m_cand = rm.unsafe_load[width=16]()
            # FA4's -8 threshold: rescale only when the max really moved.
            var need = m_cand.gt(m_old + V16(8.0))
            var m_new = need.select(m_cand, m_old)
            var alpha = _exp2((m_old - m_new) * c1v)
            if rg == 1:
                m1 = m_new
            else:
                m0 = m_new
            var any_need = False
            for i in range(16):
                any_need = any_need or need[i]
            if any_need and t0 > 0:
                var a_sc = stack_allocation[16, Float32]()
                a_sc.unsafe_store[width=16](0, alpha)
                for r in range(16):
                    # Rows the threshold skipped have alpha == 1.0 exactly, so
                    # rescaling them unconditionally is numerically a no-op.
                    var k = V16(a_sc.unsafe_load(r))
                    var arow = acc.unsafe_offset((rg * 16 + r) * D)
                    for j in range(0, D, 16):
                        arow.unsafe_store[width=16](
                            j, arow.unsafe_load[width=16](j) * k
                        )
            var m_sc = stack_allocation[16, Float32]()
            m_sc.unsafe_store[width=16](0, m_new)
            # exp + P in bf16 + row sum, one row at a time: two rows in flight
            # spills (four exp2 chains plus four sum accumulators).
            for r in range(16):
                var lr = lp.unsafe_offset(r * kt)
                var pp = prob.unsafe_offset((rg * 16 + r) * kt)
                var off = V16(-m_sc.unsafe_load(r) * c1f)
                var acc_s = V16(0.0)
                for j in range(0, nkp, 16):
                    var e = _exp2(fma(lr.unsafe_load[width=16](j), c1v, off))
                    acc_s = acc_s + e
                    var h = e.cast[DType.bfloat16]()
                    pp.unsafe_store[width=16](j, h)
                    if fp32:
                        prob_lo.unsafe_store[width=16](
                            (rg * 16 + r) * kt + j, _lo(e, h)
                        )
                rm.unsafe_store(r, Scalar[DType.float32](acc_s.reduce_add()))
            var rows = rm.unsafe_load[width=16]()
            if rg == 1:
                s1 = fma(s1, alpha, rows)
            else:
                s0 = fma(s0, alpha, rows)

        # ---- P@V over the super-tile, two dim chunks (four accumulators) ----
        # prob_lo is the lo slab of the same workspace (hi slab first), so one
        # constant offset reaches it from the hi pointer.
        comptime plo_off = QROWS * MAXKT * 2
        var nks = nk // 32 if (job.mode & 64) != 0 else (nk + 31) // 32
        var first = t0 == 0
        var dc = 0
        while dc < DC16:
            # Nothing has accumulated before the first super-tile, so zero the
            # tiles instead of loading 32*head_dim floats of fresh memory.
            if first:
                tile_zero[4]()
                tile_zero[5]()
                tile_zero[6]()
                tile_zero[7]()
            else:
                tile_load[4](Int(acc) + dc * 64, astride)
                tile_load[5](Int(acc) + (dc + 1) * 64, astride)
                tile_load[6](Int(acc) + (16 * D + dc * 16) * 4, astride)
                tile_load[7](Int(acc) + (16 * D + (dc + 1) * 16) * 4, astride)
            for ks in range(nks):
                var vb = ((t0 // 32) + ks) * DC16 * vstride
                var vt_hi = Int(vbase) + (vb + dc * vstride) * 2
                var pp0 = Int(prob) + (ks * 32) * 2
                tile_load[0](pp0, pstride)
                tile_load[1](pp0 + 16 * kt * 2, pstride)
                tile_load[2](vt_hi, 64)
                tile_load[3](vt_hi + vstride * 2, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                if fp32:
                    # P_hi@V_lo, then P_lo@V_hi
                    tile_load[2](vt_hi + 1024, 64)
                    tile_load[3](vt_hi + vstride * 2 + 1024, 64)
                    tile_dpbf16ps[4, 0, 2]()
                    tile_dpbf16ps[5, 0, 3]()
                    tile_dpbf16ps[6, 1, 2]()
                    tile_dpbf16ps[7, 1, 3]()
                    tile_load[0](pp0 + plo_off, pstride)
                    tile_load[1](pp0 + 16 * kt * 2 + plo_off, pstride)
                    tile_load[2](vt_hi, 64)
                    tile_load[3](vt_hi + vstride * 2, 64)
                    tile_dpbf16ps[4, 0, 2]()
                    tile_dpbf16ps[5, 0, 3]()
                    tile_dpbf16ps[6, 1, 2]()
                    tile_dpbf16ps[7, 1, 3]()

            tile_store[4](Int(acc) + dc * 64, astride)
            tile_store[5](Int(acc) + (dc + 1) * 64, astride)
            tile_store[6](Int(acc) + (16 * D + dc * 16) * 4, astride)
            tile_store[7](Int(acc) + (16 * D + (dc + 1) * 16) * 4, astride)
            dc += 2

        t0 += kt

    # ---- write the output rows ----
    var sv = stack_allocation[32, Float32]()
    sv.unsafe_store[width=16](0, s0)
    sv.unsafe_store[width=16](16, s1)
    if (job.mode & 256) != 0:
        return  # profiling: skip the output epilogue
    for r in range(nrows):
        var orow = obase.unsafe_offset((q0 + r) * job.o_stride_t)
        var arow = acc.unsafe_offset(r * D)
        var s = sv.unsafe_load(r)
        if not (s > 0.0):
            for j in range(0, D, 16):
                if fp32:
                    var ofrow0 = _ptr[Float32](
                        _ea(
                            job.dst,
                            batch * job.o_stride_b
                            + (q0 + r) * job.o_stride_t
                            + head * job.o_stride_h,
                            4,
                        )
                    )
                    ofrow0.unsafe_store[width=16](j, V16(0.0))
                else:
                    orow.unsafe_store[width=16](j, Scalar[DType.bfloat16](0))
            continue
        var iv = V16(1.0 / s)
        for j in range(0, D, 16):
            var o = arow.unsafe_load[width=16](j) * iv
            if fp32:
                var ofrow = _ptr[Float32](
                    _ea(
                        job.dst,
                        batch * job.o_stride_b
                        + (q0 + r) * job.o_stride_t
                        + head * job.o_stride_h,
                        4,
                    )
                )
                ofrow.unsafe_store[width=16](j, o)
            else:
                orow.unsafe_store[width=16](j, o.cast[DType.bfloat16]())


# ------------------------------------------------------------------- driver


struct AmxDispatch(Copyable, ImplicitlyCopyable, Movable):
    """State of one parallel region (pack, barrier, attention), on the stack."""

    var pack: PackJob
    var job: AmxJob
    var pack_items: Int
    var items: Int
    var workers: Int

    def __init__(
        out self,
        pack: PackJob,
        job: AmxJob,
        pack_items: Int,
        items: Int,
        workers: Int,
    ):
        self.pack = pack
        self.job = job
        self.pack_items = pack_items
        self.items = items
        self.workers = workers


@inline(.always)
def _attend[
    fp32: Bool
](
    job: AmxJob,
    lp: Pointer[Float32, MutUntrackedOrigin],
    pp: Pointer[BFloat16, MutUntrackedOrigin],
    ap: Pointer[Float32, MutUntrackedOrigin],
    qp: Pointer[BFloat16, MutUntrackedOrigin],
    qplo: Pointer[BFloat16, MutUntrackedOrigin],
    pplo: Pointer[BFloat16, MutUntrackedOrigin],
    batch: Int,
    head: Int,
    q0: Int,
) -> None:
    """One work item, with head_dim turned into a compile-time constant."""
    if job.head_dim == 128:
        _amx_dense_item[fp32, 128](job, lp, pp, ap, qp, qplo, pplo, batch, head, q0)
    elif job.head_dim == 64:
        _amx_dense_item[fp32, 64](job, lp, pp, ap, qp, qplo, pplo, batch, head, q0)
    else:
        _amx_dense_item[fp32, 32](job, lp, pp, ap, qp, qplo, pplo, batch, head, q0)


@inline(.always)
def _amx_worker[
    fp32: Bool
](state: AmxDispatch, tid: Int, threads: Int) -> None:
    """One worker: pack its share, wait, then run its share of the attention."""
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    tile_cfg_load(Int(cfg))
    var nbh = state.pack.batch * state.pack.num_kv_heads
    var kinds = state.pack.kb16 * nbh  # K tasks first, then V
    var task = tid
    # Phase-split dispatch (pack everybody, barrier, attend) is only used when
    # there are fewer (batch, kv head) pairs than workers; otherwise the
    # per-head loop below packs one head's K/V and immediately attends every
    # query block of it, which needs no barrier and leaves that head's 32 KB of
    # packed operands in L1 for the blocks that read them. At 128:8:64 the
    # head's whole packed K/V fits L1, and the barrier plus the L2 re-read are
    # 5 us of a 42 us call.
    # `mode & 512` forces the phase-split path, for an in-process A/B.
    var per_head = nbh >= state.workers * 2 and (state.job.mode & 512) == 0
    if not per_head:
        if (state.job.mode & 1) != 0:
            task = state.pack_items  # profiling: skip the pack
    while (not per_head) and task < state.pack_items:
        if (state.job.mode & 4) != 0 and task < kinds:
            task += threads
            continue
        if task < kinds:
            var kb = task % state.pack.kb16
            var bh = task // state.pack.kb16
            _pack_k_slab[fp32](
                state.pack,
                bh // state.pack.num_kv_heads,
                bh % state.pack.num_kv_heads,
                kb,
            )
        elif (state.job.mode & 8) == 0:  # profiling: bit 3 skips only the K pack
            var t = task - kinds
            var kb = t % state.pack.kb32
            var bh = t // state.pack.kb32
            _pack_v_slab[fp32](
                state.pack,
                bh // state.pack.num_kv_heads,
                bh % state.pack.num_kv_heads,
                kb,
            )
        task += threads
    if threads > 1 and (not per_head):
        external_call["GOMP_barrier", NoneType]()
    comptime nlo = 2 if fp32 else 1
    # alignment=64 is not cosmetic: the tile stores/loads and every SIMD access
    # in the softmax are 64-byte accesses, and the default alignment is the
    # element size, so unaligned buffers split a cache line per access. The C
    # version got it free from posix_memalign; here it has to be asked for.
    # Measured: worth 8-15% of the whole call.
    var logits = stack_allocation[QROWS * MAXKT, Float32, alignment=64]()
    var prob = stack_allocation[QROWS * MAXKT * nlo, BFloat16, alignment=64]()
    var acc = stack_allocation[QROWS * DMAX, Float32, alignment=64]()
    var qpad = stack_allocation[QROWS * DMAX * nlo, BFloat16, alignment=64]()
    var lp = _ptr[Float32](Int(logits))
    var pp = _ptr[BFloat16](Int(prob))
    var ap = _ptr[Float32](Int(acc))
    var qp = _ptr[BFloat16](Int(qpad))
    # The hi slab is first, the lo slab right after it (MAXKT keys / D dims).
    var pplo = _ptr[BFloat16](Int(prob) + QROWS * MAXKT * 2)
    var qplo = _ptr[BFloat16](Int(qpad) + QROWS * DMAX * 2)
    var nqg = (state.job.seq_q + QROWS - 1) // QROWS
    if per_head:
        # One task = one (batch, kv head): pack it, then every query head that
        # reads it, every query block of each.
        var qpg = 1
        if state.job.num_heads != state.job.num_kv_heads:
            qpg = state.job.num_heads // state.job.num_kv_heads
        task = tid
        while task < nbh:
            var kvh = task % state.job.num_kv_heads
            var b = task // state.job.num_kv_heads
            if (state.job.mode & 1) == 0:
                if (state.job.mode & 4) == 0:
                    for kb in range(state.pack.kb16):
                        _pack_k_slab[fp32](state.pack, b, kvh, kb)
                if (state.job.mode & 8) == 0:
                    for kb in range(state.pack.kb32):
                        _pack_v_slab[fp32](state.pack, b, kvh, kb)
            if (state.job.mode & 2) == 0:
                for hh in range(qpg):
                    var head = kvh * qpg + hh
                    for g in range(nqg):
                        _attend[fp32](state.job, lp, pp, ap, qp, qplo, pplo,
                                      b, head, g * QROWS)
            task += threads
        tile_release()
        return
    task = tid
    if (state.job.mode & 2) != 0:
        task = state.items  # profiling: skip the attention
    while task < state.items:
        var g = task % nqg
        var rest = task // nqg
        var head = rest % state.job.num_heads
        var batch = rest // state.job.num_heads
        _attend[fp32](state.job, lp, pp, ap, qp, qplo, pplo, batch, head,
                      g * QROWS)
        task += threads
    tile_release()


@export
def _gomp_worker_amx_bf16(data: Int) abi("C"):
    var st = MutPointer[AmxDispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _amx_worker[False](state, tid, threads)


@export
def _gomp_worker_amx_fp32(data: Int) abi("C"):
    var st = MutPointer[AmxDispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _amx_worker[True](state, tid, threads)


def run_dense_amx(
    pack: PackJob, job: AmxJob, num_workers: Int, kt: Int
) -> None:
    """Pack K/V and run the attention, in one fork-join."""
    if job.batch == 0 or job.num_heads == 0 or job.seq_q <= 0:
        return
    var items = job.batch * job.num_heads * ((job.seq_q + QROWS - 1) // QROWS)
    var pack_items = pack.batch * pack.num_kv_heads * (pack.kb16 + pack.kb32)
    var workers = min(num_workers, max(items, max(pack_items, 1)))
    var state = stack_allocation[1, AmxDispatch]()
    state[] = AmxDispatch(pack, job, pack_items, items, workers)
    if workers <= 1:
        if job.fp32:
            _amx_worker[True](state[], 0, 1)
        else:
            _amx_worker[False](state[], 0, 1)
        return
    if job.fp32:
        comptime entry_f = _gomp_worker_amx_fp32
        external_call["GOMP_parallel", NoneType](
            entry_f, Int(state), Int32(workers), Int32(0)
        )
    else:
        comptime entry_b = _gomp_worker_amx_bf16
        external_call["GOMP_parallel", NoneType](
            entry_b, Int(state), Int32(workers), Int32(0)
        )


def has_amx() -> Bool:
    """Whether this build targets a CPU with AMX (so the kernel above exists).
    """
    return AMX
