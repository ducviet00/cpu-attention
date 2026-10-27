"""Dense attention on Intel AMX, v2: one softmax pass per logit instead of two.

Same contract, arithmetic and packed layouts as `dense_amx.mojo` (it reads the
same K/V slabs, so `cpu_attention/_kernel.py` allocates the same workspace), and
it is opt-in: `CPU_ATTENTION_AMX2=1`, v1 stays the default. Every number below
is measured on the Emerald Rapids box by in-process microbenchmarks (the
roofs), phase ablations (where a call's time goes) and
an interleaved one-process A/B, min-based (`docs/Mojo.md`).

## The roofs (EMR 8581C, 4 cores, 1 thread, per core)

    one tdpbf16ps, 2-2-4, no memory operands       4.86 ns   = 3.37 TFLOP/s/core
    + 4 tile loads (fixed lines *or* walking)     4.85 ns   loads are free
    + 4 tile stores per 16 dots                   6.15 ns   a 1 KB store is 5.2 ns
    one tile load alone                           2.33 ns
    1 dependent dot chain                         4.86 ns   issue-bound, not latency
    softmax pass, exp2, fp32 -> bf16           0.176 ns/logit
    same + a running max                       0.172 ns/logit
    same, degree-2 polynomial                  0.147 ns/logit
    4 dots + 8 exp2 chains in one loop        19.8 ns vs 17.8 for the dots alone
    4 dots + 96 vector FMAs                   17.8 ns == the dots alone

The last two say the AMX pipe and the vector pipes *do* run concurrently: vector
work is free in the shadow of a dot as long as it does not depend on it. For
reference oneDNN's bf16 GEMM reaches 1.11 TFLOP/s/core on this box, 33% of the
4.86 ns/dot roof, and vLLM's AMX attention 0.66-0.74.

## What v2 changes, and what it measures

1. **One pass over the logits, not two.** v1 computes a per-super-tile row max
   in a first pass, then exponentiates in a second. v2 keeps the max *stale*:
   the exp pass accumulates this super-tile's row maxima for free and, if one
   moved by more than 8, the accumulator and the row sum are rescaled afterwards
   (both are in the same frame, so one multiply fixes both). Only the first
   super-tile of a work item needs a max pass of its own, and it reuses that
   pass's maxima instead of paying for the fused one as well. Measured: softmax
   0.614 vs v1's 0.689 ms at 1024:16:128.
2. **`head_dim` and the dtype are compile-time** (`dc32` in 1/2/4), so the QK
   inner loop is 16 straight-line loads+dots with comptime offsets instead of a
   runtime `for dc in range(job.dc32)` with a profiling-mode branch inside it.
   Every `tile_loadd` carries a `~{memory}` clobber; a runtime loop bound between
   them makes the compiler reload everything it needs after each one.
3. **Degree-3 polynomial for bf16** (its probability is rounded to bf16 two
   instructions later; degree-3 is 6e-4 against that store's 4e-3) and degree-4
   for fp32, whose probabilities are split into bf16 hi/lo. Note the first
   version of this file chained all degrees off the degree-5 leading
   coefficient, which leaves a stray 0.0013*f^4 term: 4e-4 relative, invisible
   in bf16 and 2x over the fp32 gate (measured d=4.2e-4). Each degree now starts
   from its own leading coefficient, and bf16 parity improved to 7.8e-3 where v1
   reads 1.6e-2 against the same 3e-2 gate.
4. Per-row max/sum state in one 32-float array each, instead of two 16-lane
   vectors that had to be unpacked per row.

## Falsified on the way here (do not retry without new measurements)

* **Software-pipelining the softmax under the QK dots**, one chunk apart, with
  the block-shaped softmax that requires: 1185 -> 1217 us of QK+softmax at
  1024:16:128, and 14% *worse* at 4096:16:128. The QK's marginal cost fell 17%
  (the overlap is real) but the softmax's rose 21% (one horizontal reduce per
  row per 32-key block), and the block structure also broke any hope of the
  batch-level ILP the flat pass gets for free. The flat pass is better.
* **Super-tile size**: kt = 128 / 256 / 512 measure 0.99 / 0.97 / 0.96 against
  v1 at 1024:16:128 and 1.01 / 0.97 / 0.99 at 4096:16:128. Flat: a smaller kt
  keeps the logits closer to L1 but doubles the output-accumulator traffic, and
  the two cancel.
* **More ILP in the softmax** (2 rows x 4 vectors instead of 4 rows x 1): no
  change, and the isolated loop measures the same at 1, 2 and 4 rows in flight.
  The out-of-order engine already extracts it; the in-kernel cost is not a
  latency chain.
* **4K-aliasing between the logits and probability buffers** (they are exactly
  64 KB apart, so every softmax load/store pair is 4K-congruent): shifting the
  probability slabs by 64 bytes changed nothing.
* **A contiguous Q panel for the bf16 path** (the A tile load is strided by
  `q.stride(1)*2`, i.e. 4 KB at H=16/D=128): the probe's stride sweep read 22%
  worse at stride 4096 once and 2% *better* the next run, so the effect is not
  reproducible on this box. Not done.
* **8 workers (SMT) instead of 4**: 27620 vs 24314 us at 4096:16:128. AMX is
  per-core and the softmax does not gain enough to pay for the sharing.
* v1's `CPU_ATTENTION_AMX_MODE` bit 64 ("skip P@V") is broken: it switches
  `nk // 32` for `(nk + 31) // 32`, which is the same number for every full
  super-tile, so that phase reported 0.3-1% and the shares summed past 100%.
  With the bit fixed here P@V is 14% of a 4096:16:128 call.

## Where the time actually goes (v2, 4096:16:128 bf16, 22.8 ms)

    Q@K^T 7.9 ms (35%)   softmax 8.9 ms (39%)   P@V 1.7 ms (7%)
    K/V pack 1.3 ms (6%)   epilogue 0.5 ms (2%)   unattributed 2.7 ms (12%)

in ns per logit that is 0.235 / 0.265 / 0.049 / 0.038 / 0.016, against isolated
floors of 0.17 / 0.176 / 0.11. QK is 1.4x its floor, P@V is under it (the
running V pointer above was worth 2x on that phase: 3.6 -> 1.7 ms), and **the
softmax is 1.5x the same loop measured in isolation. Neither this kernel nor v1
beats that, and it is not any of the six things listed above** -- it is the
biggest single item left, worth ~1.3x on the whole call if it can be found.

## Net effect

`v2/v1`, min-based over 9 interleaved rounds with a duplicate-arm control at
+-2%: bf16 0.95-0.99 for seq >= 1024 and 1.02-1.03 at seq <= 256; fp32 0.95-1.00
for seq >= 2048 and 1.06-1.14 at seq <= 512, where the call is 35-44% K/V pack
and the pack is being rewritten separately. Against vLLM's AMX build (bf16) the
long rows read 0.70 (4096:8:64), 0.83 (4096:16:128), 0.85 (2048:16:128); against
its VEC build (fp32) 0.43-0.67.

## Multiple of 32 only

`head_dim` 32/64/128 (the same three v1 supports), any strides with
`stride(head_dim) == 1`, GQA/MQA, any batch/heads/seq >= 1, no state between
calls.
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

comptime AMX2 = (
    CompilationTarget.is_x86()
    and CompilationTarget._has_feature["amx-tile"]()
    and CompilationTarget._has_feature["amx-bf16"]()
)
"""Whether this build targets a CPU with AMX (the whole kernel is behind this).
"""

comptime QROWS = 32
"""Query rows per work item: 2-2-4 needs two A tiles, so 32 rows is the max."""

comptime KEYCH = 32
"""Keys per QK step: four accumulators hold 32 rows x 32 keys."""

comptime MAXKT = 512
"""Upper bound on the keys in one softmax super-tile (sizes the buffers)."""

comptime DMAX = 128
"""Largest supported head_dim: the per-worker buffers are sized for it."""

comptime LOG2E = 1.4426950408889634
comptime NEG_INF = -3.4028234663852886e38


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


@inline(.always)
def _ea(base: Int, elem: Int, esize: Int) -> Int:
    """Byte address of element ``elem`` of an ``esize``-byte type at ``base``.
    """
    return base + elem * esize


# --------------------------------------------------------------- exp2 + split


@inline(.always)
def _exp2v[deg: Int](x: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """2**x, and exactly +0 once x is far enough below zero.

    `deg` 3 is enough for a bf16 store (6e-4 against the store's own 4e-3);
    `deg` 4 is what the fp32 path's hi/lo probabilities need. The clamp at -127
    makes a fully masked lane underflow to +0, which P@V needs: `0 * garbage`
    must be 0, and a bf16 denormal in an AMX tile is not free.
    """
    comptime V = SIMD[DType.float32, 16]
    comptime MAGIC = 12582912.0  # 1.5 * 2**23: round-to-nearest in fp32
    var xc = min(max(x, V(-127.0)), V(127.0))
    var n = (xc + V(MAGIC)) - V(MAGIC)
    var f = xc - n
    # Each degree starts from *its own* leading coefficient. Chaining from the
    # degree-5 one and skipping terms (the first version) leaves a stray
    # 0.0013*f^4 term: 4e-4 relative, which is invisible in bf16 and 2x over the
    # fp32 gate (measured d=4.2e-4 against SDPA).
    var p = V(0.009618129107628477)
    comptime if deg == 2:
        p = V(0.2402265069591007)
    comptime if deg == 3:
        p = V(0.05550410866482158)
    comptime if deg >= 5:
        p = V(0.0013333558146428443)
    comptime if deg >= 5:
        p = fma(p, f, V(0.009618129107628477))
    comptime if deg >= 3:
        p = fma(p, f, V(0.05550410866482158))
    comptime if deg >= 2:
        p = fma(p, f, V(0.2402265069591007))
    p = fma(p, f, V(0.6931471805599453))
    p = fma(p, f, V(1.0))
    var ni = n.cast[DType.int32]()
    var bits = (ni + SIMD[DType.int32, 16](127)) << SIMD[DType.int32, 16](23)
    return p * bitcast[DType.float32, 16](bits)


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
#
# The packed layouts are v1's, byte for byte, because `_kernel.py` sizes the
# workspace and both kernels have to agree on it: K is a 16x16 32-bit transpose
# per (16-key block, 32-dim chunk) and V is a 2-way interleave of adjacent keys
# per (32-key block, 16-dim chunk). `tests/test_amx.py` compares the two kernels'
# outputs, which is what keeps this honest.


@inline(.always)
def _tp16_i32(
    m: Array[SIMD[DType.int32, 16], 16]
) -> Array[SIMD[DType.int32, 16], 16]:
    """16x16 transpose of 32-bit units, in place, two registers of temporaries.
    """
    var a = m.copy()
    comptime for k in range(4):
        comptime step = 1 << k
        comptime for i in range(16):
            comptime if (i >> k) & 1 == 0:
                comptime b = i | step
                comptime if k == 0:
                    var t0 = a[i].shuffle[
                        0,
                        16,
                        2,
                        18,
                        4,
                        20,
                        6,
                        22,
                        8,
                        24,
                        10,
                        26,
                        12,
                        28,
                        14,
                        30,
                    ](a[b])
                    var t1 = a[i].shuffle[
                        1,
                        17,
                        3,
                        19,
                        5,
                        21,
                        7,
                        23,
                        9,
                        25,
                        11,
                        27,
                        13,
                        29,
                        15,
                        31,
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 1:
                    var t0 = a[i].shuffle[
                        0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
                    ](a[b])
                    var t1 = a[i].shuffle[
                        2,
                        3,
                        18,
                        19,
                        6,
                        7,
                        22,
                        23,
                        10,
                        11,
                        26,
                        27,
                        14,
                        15,
                        30,
                        31,
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 2:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27
                    ](a[b])
                    var t1 = a[i].shuffle[
                        4,
                        5,
                        6,
                        7,
                        20,
                        21,
                        22,
                        23,
                        12,
                        13,
                        14,
                        15,
                        28,
                        29,
                        30,
                        31,
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 3:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
                    ](a[b])
                    var t1 = a[i].shuffle[
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
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
](job: PackJob, batch: Int, head: Int, kb: Int, dc: Int) -> None:
    """One 1 KB QK B tile: 16 keys x 16 head-dim pairs, transposed."""
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
](job: PackJob, batch: Int, head: Int, kb: Int, dc: Int) -> None:
    """One 1 KB P@V B tile: 32 keys x 16 output dims, keys interleaved 2-way."""
    var k0 = kb * 32
    var d0 = dc * 16
    var slab_base = _ea(
        job.vp,
        (batch * job.num_kv_heads + head) * job.vp_bh
        + (kb * job.dc16 + dc) * (1024 if fp32 else 512),
        2,
    )
    if fp32:
        var hi_p = _ptr[BFloat16](slab_base)
        var lo_p = _ptr[BFloat16](slab_base + 512 * 2)
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


struct V2Job(Copyable, ImplicitlyCopyable, Movable):
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
    """Profiling only (`CPU_ATTENTION_AMX_MODE`), same bits as v1: 1 skips the
    whole K/V pack, 2 the whole attention loop, 4 the V pack, 8 the K pack, 16
    the Q@K^T phase, 32 the softmax, 64 the P@V phase, 128 the per-item Q
    prologue, 256 the output epilogue. Results are garbage with any bit set."""

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
def _v2_qk[
    dc32: Int, fp32: Bool
](
    qp: Int,
    qp_lo: Int,
    qstride: Int,
    kchunk: Int,
    dc16: Int,
    lstride: Int,
    logits: MutPointer[Float32, MutUntrackedOrigin],
) -> None:
    """Q@K^T for 32 rows x 32 keys into one 32x32 block of the logits buffer.

    `kchunk` is the byte address of this chunk's first packed K block (16 keys,
    32 dims); the tiles for the 16 dims past that are `dc32` blocks further on,
    because the pack lays a key block out as `[dc][tile]`.
    """
    # One packed K block is 16 keys x 32 dims x 2 bytes = 1024 B (2048 on the
    # fp32 path, which stores an hi and a lo slab back to back). These are
    # *byte* strides; v1's `kstride` in `dense_amx.mojo` counts bf16 elements,
    # so it reads 512/1024 where this reads 1024/2048.
    comptime kslab = 2048 if fp32 else 1024
    comptime kblock = dc32 * kslab  # one 16-key block holds every dim chunk
    tile_zero[4]()
    tile_zero[5]()
    tile_zero[6]()
    tile_zero[7]()
    comptime for dc in range(dc32):
        comptime off = dc * kslab
        tile_load[0](qp + dc * 64, qstride)
        tile_load[1](qp + dc * 64 + 16 * qstride, qstride)
        tile_load[2](kchunk + off, 64)
        tile_load[3](kchunk + off + kblock, 64)
        tile_dpbf16ps[4, 0, 2]()
        tile_dpbf16ps[5, 0, 3]()
        tile_dpbf16ps[6, 1, 2]()
        tile_dpbf16ps[7, 1, 3]()
        comptime if fp32:
            # Q_hi@K_lo, then Q_lo@K_hi (K_hi and Q_hi reload from L1).
            tile_load[2](kchunk + off + 1024, 64)
            tile_load[3](kchunk + off + kblock + 1024, 64)
            tile_dpbf16ps[4, 0, 2]()
            tile_dpbf16ps[5, 0, 3]()
            tile_dpbf16ps[6, 1, 2]()
            tile_dpbf16ps[7, 1, 3]()
            tile_load[0](qp_lo + dc * 64, qstride)
            tile_load[1](qp_lo + dc * 64 + 16 * qstride, qstride)
            tile_load[2](kchunk + off, 64)
            tile_load[3](kchunk + off + kblock, 64)
            tile_dpbf16ps[4, 0, 2]()
            tile_dpbf16ps[5, 0, 3]()
            tile_dpbf16ps[6, 1, 2]()
            tile_dpbf16ps[7, 1, 3]()
    tile_store[4](Int(logits), lstride)
    tile_store[5](Int(logits) + 64, lstride)
    tile_store[6](Int(logits) + 16 * lstride, lstride)
    tile_store[7](Int(logits) + 16 * lstride + 64, lstride)


@inline(.always)
def _v2_block_max[
    rilp: Int
](
    lp: MutPointer[Float32, MutUntrackedOrigin],
    kt: Int,
    j0: Int,
    marr: MutPointer[Float32, MutUntrackedOrigin],
) -> None:
    """Bootstrap a row max from one 32-key block (first super-tile only).

    The steady state does not need this: `_v2_block_exp` accumulates the max of
    every block it exponentiates, so all but the very first block of an item use
    a max that is one super-tile stale (v1's -8 threshold decides whether the
    accumulator has to be rescaled).
    """
    comptime V16 = SIMD[DType.float32, 16]
    for rg in range(2):
        for rq in range(0, 16, rilp):
            var row = rg * 16 + rq
            var mx = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
            comptime for u in range(rilp):
                mx[u] = lp.unsafe_load[width=16]((row + u) * kt + j0)
            comptime for u in range(rilp):
                mx[u] = max(
                    mx[u], lp.unsafe_load[width=16]((row + u) * kt + j0 + 16)
                )
            comptime for u in range(rilp):
                var cand = Scalar[DType.float32](mx[u].reduce_max())
                if cand > marr.unsafe_load(row + u):
                    marr.unsafe_store(row + u, cand)


@inline(.always)
def _v2_maxpass[
    rilp: Int
](
    lp: MutPointer[Float32, MutUntrackedOrigin],
    kt: Int,
    nk: Int,
    marr: MutPointer[Float32, MutUntrackedOrigin],
) -> None:
    """Row maxima over a whole super-tile, one horizontal reduce per row.

    Only the first super-tile of a work item needs this: everything after it
    exponentiates against a max that is one super-tile stale and lets
    `_v2_softmax` accumulate the new one for free. Doing it per 32-key block
    instead (32 reduces per row per super-tile) measured 5% worse at
    256:16:128, where the whole call is one super-tile.
    """
    comptime V16 = SIMD[DType.float32, 16]
    for rg in range(2):
        for rq in range(0, 16, rilp):
            var row = rg * 16 + rq
            var mx = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
            comptime for u in range(rilp):
                mx[u] = V16(-3.4028234663852886e38)
            var j = 0
            while j < nk:
                comptime for u in range(rilp):
                    mx[u] = max(
                        mx[u], lp.unsafe_load[width=16]((row + u) * kt + j)
                    )
                j += 16
            comptime for u in range(rilp):
                var cand = Scalar[DType.float32](mx[u].reduce_max())
                if cand > marr.unsafe_load(row + u):
                    marr.unsafe_store(row + u, cand)


@inline(.always)
def _v2_softmax[
    fp32: Bool, deg: Int, rilp: Int, kilp: Int, withmax: Bool
](
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[BFloat16, MutUntrackedOrigin],
    plo: MutPointer[BFloat16, MutUntrackedOrigin],
    kt: Int,
    nk: Int,
    c1v: SIMD[DType.float32, 16],
    c1f: Float32,
    marr: MutPointer[Float32, MutUntrackedOrigin],
    sums: MutPointer[Float32, MutUntrackedOrigin],
    tmax: MutPointer[Float32, MutUntrackedOrigin],
) -> None:
    """One pass over a super-tile's logits: exponentiate, write bf16, sum, max.

    `rilp` rows x `kilp` 16-lane vectors are kept in flight. In isolation the
    ILP width does not matter on this core (1 row 0.1786 ns/logit, 4 rows
    0.1761), but inside the kernel the logits come back from L2 rather than L1
    (the Q@K phase streamed 16 KB of operands per chunk through L1), so more
    loads in flight is what hides that latency: measured 0.343 -> see
    docs/Mojo.md for the sweep.

    The max is accumulated here rather than in a pass of its own: all but the
    first super-tile of a work item exponentiate against a max that is one
    super-tile stale, and `_v2_item` rescales afterwards if it moved by more
    than 8 (v1's threshold, v1's raw-logit frame).
    """
    comptime V16 = SIMD[DType.float32, 16]
    for rg in range(2):
        for rq in range(0, 16, rilp):
            var row = rg * 16 + rq
            var offs = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
            var acc = Array[SIMD[DType.float32, 16], kilp * rilp](
                uninitialized=True
            )
            var mx = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
            comptime for u in range(rilp):
                offs[u] = V16(-marr.unsafe_load(row + u) * c1f)
                mx[u] = V16(-3.4028234663852886e38)
            comptime for w in range(kilp * rilp):
                acc[w] = V16(0.0)
            var j = 0
            while j < nk:
                comptime for kk in range(kilp):
                    comptime for u in range(rilp):
                        var x = lp.unsafe_load[width=16](
                            (row + u) * kt + j + kk * 16
                        )
                        comptime if withmax:
                            mx[u] = max(mx[u], x)
                        var e = _exp2v[deg](fma(x, c1v, offs[u]))
                        acc[kk * rilp + u] += e
                        pp.unsafe_store[width=16](
                            (row + u) * kt + j + kk * 16,
                            e.cast[DType.bfloat16](),
                        )
                        comptime if fp32:
                            plo.unsafe_store[width=16](
                                (row + u) * kt + j + kk * 16,
                                _lo(e, e.cast[DType.bfloat16]()),
                            )
                j += 16 * kilp
            comptime for u in range(rilp):
                var s = acc[u]
                comptime for kk in range(1, kilp):
                    s += acc[kk * rilp + u]
                sums.unsafe_store(
                    row + u,
                    sums.unsafe_load(row + u)
                    + Scalar[DType.float32](s.reduce_add()),
                )
                comptime if withmax:
                    tmax.unsafe_store(
                        row + u,
                        max(
                            tmax.unsafe_load(row + u),
                            Scalar[DType.float32](mx[u].reduce_max()),
                        ),
                    )


@inline(.always)
def _v2_mask_chunk(
    logits: MutPointer[Float32, MutUntrackedOrigin],
    kt: Int,
    j0: Int,
    t0: Int,
    seq_k: Int,
    diag: Int,
    q0: Int,
    nrows: Int,
    causal: Bool,
) -> None:
    """Write -inf over the masked lanes of one 32-key block.

    Range-based, not lane-by-lane: the causal limit is contiguous, so the
    masked part of a block is one or two 16-lane stores. Lanes past the block
    are left alone (`-inf` there would be harmless but this keeps the store
    footprint to what is really masked).
    """
    comptime V16 = SIMD[DType.float32, 16]
    comptime I16 = SIMD[DType.int32, 16]
    var jend = j0 + KEYCH
    var lim_min = seq_k
    if causal:
        lim_min = min(seq_k, q0 + diag + 1)
    if t0 + jend <= lim_min:
        return
    for r in range(nrows):
        var lim = seq_k
        if causal:
            lim = min(seq_k, q0 + r + diag + 1)
        lim -= t0
        if lim >= jend:
            continue
        var lp = logits.unsafe_offset(r * kt)
        if lim <= j0:
            for j in range(j0, jend, 16):
                lp.unsafe_store[width=16](j, V16(NEG_INF))
        else:
            var j = max(j0, (lim // 16) * 16)
            while j < jend:
                var keep = I16(Int32(lim)).gt(
                    I16(Int32(j)) + iota[DType.int32, 16](0)
                )
                lp.unsafe_store[width=16](
                    j, keep.select(lp.unsafe_load[width=16](j), V16(NEG_INF))
                )
                j += 16


@inline(.always)
def _v2_pv[
    dc32: Int, fp32: Bool
](
    pp: MutPointer[BFloat16, MutUntrackedOrigin],
    plo: MutPointer[BFloat16, MutUntrackedOrigin],
    vrow: Int,
    nks: Int,
    kt: Int,
    accp: MutPointer[Float32, MutUntrackedOrigin],
    astride: Int,
    first: Bool,
) -> None:
    """P@V over the super-tile: 32 rows x 32 dims per pass, acc in memory.

    `vrow` is the byte address of this super-tile's first packed V block (32
    keys x 16 dims); a dim chunk is `vslab` bytes further on.
    """
    comptime vslab = 2048 if fp32 else 1024  # bytes per packed 32-key V block
    comptime D = dc32 * 32
    comptime ndc = dc32  # dim chunks of 32 in the output
    var pstride = kt * 2
    for dcp in range(ndc):
        var a0 = Int(accp) + dcp * 128
        if first:
            tile_zero[4]()
            tile_zero[5]()
            tile_zero[6]()
            tile_zero[7]()
        else:
            tile_load[4](a0, astride)
            tile_load[5](a0 + 64, astride)
            tile_load[6](a0 + 16 * D * 4, astride)
            tile_load[7](a0 + 16 * D * 4 + 64, astride)
        # One running pointer for V, stepped by a comptime constant: the packed
        # V slab for a 32-key block is `dc16 * vslab` bytes on, and `dc16` is
        # `2 * dc32` at compile time.
        var vp_ks = vrow + dcp * 2 * vslab
        comptime vstep = 2 * dc32 * vslab
        for ks in range(nks):
            tile_load[0](Int(pp) + ks * 64, pstride)
            tile_load[1](Int(pp) + 16 * kt * 2 + ks * 64, pstride)
            tile_load[2](vp_ks, 64)
            tile_load[3](vp_ks + vslab, 64)
            tile_dpbf16ps[4, 0, 2]()
            tile_dpbf16ps[5, 0, 3]()
            tile_dpbf16ps[6, 1, 2]()
            tile_dpbf16ps[7, 1, 3]()
            comptime if fp32:
                tile_load[2](vp_ks + 1024, 64)
                tile_load[3](vp_ks + vslab + 1024, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                tile_load[0](Int(plo) + ks * 64, pstride)
                tile_load[1](Int(plo) + 16 * kt * 2 + ks * 64, pstride)
                tile_load[2](vp_ks, 64)
                tile_load[3](vp_ks + vslab, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
            vp_ks += vstep
        tile_store[4](a0, astride)
        tile_store[5](a0 + 64, astride)
        tile_store[6](a0 + 16 * D * 4, astride)
        tile_store[7](a0 + 16 * D * 4 + 64, astride)


@inline(.always)
def _v2_item[
    dc32: Int, fp32: Bool
](
    job: V2Job,
    logits: MutPointer[Float32, MutUntrackedOrigin],
    prob: MutPointer[BFloat16, MutUntrackedOrigin],
    prob_lo: MutPointer[BFloat16, MutUntrackedOrigin],
    acc: MutPointer[Float32, MutUntrackedOrigin],
    qpad: MutPointer[BFloat16, MutUntrackedOrigin],
    qpad_lo: MutPointer[BFloat16, MutUntrackedOrigin],
    batch: Int,
    head: Int,
    q0: Int,
) -> None:
    """Attend QROWS query rows of one (batch, head) over the key range."""
    comptime D = dc32 * 32
    comptime V16 = SIMD[DType.float32, 16]
    comptime I16 = SIMD[DType.int32, 16]
    comptime kslab = 2048 if fp32 else 1024  # bytes per packed 16-key K block
    comptime vslab = 2048 if fp32 else 1024  # bytes per packed 32-key V block
    var kt = job.kt
    var seq_k = job.seq_k
    var nrows = min(QROWS, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var kvh = head
    if job.num_heads != job.num_kv_heads:
        kvh = head // (job.num_heads // job.num_kv_heads)
    var kbase = _ea(job.kp, (batch * job.num_kv_heads + kvh) * job.kp_bh, 2)
    var vbase = _ea(job.vp, (batch * job.num_kv_heads + kvh) * job.vp_bh, 2)

    # ---- Q prologue: the A operand, plus its hi/lo panel on the fp32 path ----
    var qptr = 0
    var qptr_lo = 0
    var qstride = D * 2
    if fp32:
        # The A operand needs no transpose, so splitting Q is a straight
        # conversion into a per-work-item panel, zero padded past nrows.
        for i in range(QROWS * D):
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
            comptime for j in range(0, D, 16):
                var f = srcf.unsafe_load[width=16](j)
                var h = _hi(f)
                qpad.unsafe_store[width=16](r * D + j, h)
                qpad_lo.unsafe_store[width=16](r * D + j, _lo(f, h))
        qptr = Int(qpad)
        qptr_lo = Int(qpad_lo)
    elif nrows == QROWS:
        qptr = _ea(
            job.q,
            batch * job.q_stride_b
            + q0 * job.q_stride_t
            + head * job.q_stride_h,
            2,
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
            comptime for i in range(D):
                qpad.unsafe_store(r * D + i, s.unsafe_load(i))
        qptr = Int(qpad)

    var lstride = kt * 4
    var pstride = kt * 2
    var astride = D * 4
    var dc16 = 2 * dc32
    var c1v = V16(job.scale * LOG2E)
    var c1f = job.scale * LOG2E

    # marr: the running row max in *raw* logit units (v1's frame, so the -8
    # threshold means the same thing); sarr: the accumulated row sum.
    var marr = stack_allocation[QROWS, Float32, alignment=64]()
    var sarr = stack_allocation[QROWS, Float32, alignment=64]()
    var tmax = stack_allocation[QROWS, Float32, alignment=64]()
    for r in range(QROWS):
        marr.unsafe_store(r, Float32(0.0))
        sarr.unsafe_store(r, Float32(0.0))
    var sums = stack_allocation[QROWS, Float32, alignment=64]()

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)
    loop_end = max(loop_end, 0)
    var t0 = 0
    var first = True
    while t0 < loop_end:
        var nk = min(kt, loop_end - t0)
        var nch = (nk + KEYCH - 1) // KEYCH
        var nkp = nch * KEYCH
        var kb0 = t0 // 16
        var kchunk = kbase + kb0 * dc32 * kslab
        for r in range(QROWS):
            sums.unsafe_store(r, Float32(0.0))
            tmax.unsafe_store(r, Float32(-3.4028234663852886e38))

        # ---- Q@K^T and softmax, interleaved one 32-key chunk apart ----
        #
        # The dots of chunk c and the exponentiation of chunk c-1 use different
        # execution units (AMX vs the vector pipes) and share no register, so
        # the second hides under the first. That is worth ~35% of the two
        # phases in series, and it is the reason the softmax is block-shaped
        # (32 keys) instead of one pass over the super-tile: a block per chunk
        # is what can be issued while the next chunk's dots are in flight.
        # The mask has to run before its own block's exp, so it moves in here
        # too; it only ever writes the two lanes of the block it is on.
        # ---- Q@K^T: 32 keys at a time into the logits buffer ----
        if (job.mode & 16) == 0:
            for c in range(nch):
                _v2_qk[dc32, fp32](
                    qptr,
                    qptr_lo,
                    qstride,
                    kchunk + c * dc32 * kslab * 2,
                    dc16,
                    lstride,
                    logits.unsafe_offset(c * KEYCH),
                )

        # ---- causal / padding mask: one contiguous [-inf] range per row ----
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
                var lpr = logits.unsafe_offset(r * kt)
                if lim <= 0:
                    for j in range(0, nkp, 16):
                        lpr.unsafe_store[width=16](j, V16(NEG_INF))
                else:
                    var j = (lim // 16) * 16
                    while j < nkp:
                        var keep = I16(Int32(lim)).gt(
                            I16(Int32(j)) + iota[DType.int32, 16](0)
                        )
                        lpr.unsafe_store[width=16](
                            j,
                            keep.select(
                                lpr.unsafe_load[width=16](j), V16(NEG_INF)
                            ),
                        )
                        j += 16

        # ---- softmax: one pass, max stale by one super-tile ----
        if (job.mode & 32) == 0:
            # The first super-tile needs its own max *before* it exponentiates,
            # and that pass already produced the row maxima -- so it must not
            # pay for the fused max as well. Every later super-tile is one
            # whole pass cheaper than v1 instead of two: measured 0.614 vs
            # 0.689 ms of softmax at 1024:16:128.
            if first:
                _v2_maxpass[4](logits, kt, nkp, marr)
                for r in range(QROWS):
                    tmax.unsafe_store(r, marr.unsafe_load(r))
                _v2_softmax[fp32, 3 if not fp32 else 4, 4, 1, False](
                    logits, prob, prob_lo, kt, nkp, c1v, c1f, marr, sums, tmax
                )
            else:
                _v2_softmax[fp32, 3 if not fp32 else 4, 4, 1, True](
                    logits, prob, prob_lo, kt, nkp, c1v, c1f, marr, sums, tmax
                )

        # ---- P@V over the super-tile ----
        if (job.mode & 64) == 0:
            _v2_pv[dc32, fp32](
                prob,
                prob_lo,
                vbase + (t0 // 32) * dc16 * vslab,
                nch,
                kt,
                acc,
                astride,
                first,
            )

        # ---- fold in the row sums, then rescale if the stale max was too low
        for r in range(QROWS):
            sarr.unsafe_store(r, sarr.unsafe_load(r) + sums.unsafe_load(r))
        if True:
            var any = False
            for r in range(QROWS):
                if tmax.unsafe_load(r) > marr.unsafe_load(r) + Float32(8.0):
                    any = True
            if any:
                var alphas = stack_allocation[QROWS, Float32, alignment=64]()
                for r in range(QROWS):
                    # alpha is measured against the max this super-tile's
                    # probabilities were actually exponentiated with, so
                    # `m_old` has to be read before `marr` is updated.
                    var m_old = marr.unsafe_load(r)
                    var m_new = m_old
                    var t = tmax.unsafe_load(r)
                    if t > m_old + Float32(8.0):
                        m_new = t
                        marr.unsafe_store(r, m_new)
                    alphas.unsafe_store(
                        r,
                        Scalar[DType.float32](
                            _exp2v[4](V16(c1f * (m_old - m_new)))[0]
                        ),
                    )
                for r in range(QROWS):
                    sarr.unsafe_store(
                        r, sarr.unsafe_load(r) * alphas.unsafe_load(r)
                    )
                for r in range(QROWS):
                    var arow = acc.unsafe_offset(r * D)
                    var k = V16(alphas.unsafe_load(r))
                    comptime for j in range(0, D, 16):
                        arow.unsafe_store[width=16](
                            j, arow.unsafe_load[width=16](j) * k
                        )
        t0 += kt
        first = False

    # ---- write the output rows ----
    if (job.mode & 256) != 0:
        return
    var obase = _ptr[BFloat16](
        _ea(job.dst, batch * job.o_stride_b + head * job.o_stride_h, 2)
    )
    for r in range(nrows):
        var s = sarr.unsafe_load(r)
        var arow = acc.unsafe_offset(r * D)
        if not (s > 0.0):
            for j in range(0, D, 16):
                if fp32:
                    _ptr[Float32](
                        _ea(
                            job.dst,
                            batch * job.o_stride_b
                            + (q0 + r) * job.o_stride_t
                            + head * job.o_stride_h,
                            4,
                        )
                    ).unsafe_store[width=16](j, V16(0.0))
                else:
                    obase.unsafe_store[width=16](
                        (q0 + r) * job.o_stride_t + j,
                        SIMD[DType.bfloat16, 16](0),
                    )
            continue
        var iv = V16(1.0 / s)
        comptime for j in range(0, D, 16):
            var o = arow.unsafe_load[width=16](j) * iv
            if fp32:
                _ptr[Float32](
                    _ea(
                        job.dst,
                        batch * job.o_stride_b
                        + (q0 + r) * job.o_stride_t
                        + head * job.o_stride_h,
                        4,
                    )
                ).unsafe_store[width=16](j, o)
            else:
                obase.unsafe_store[width=16](
                    (q0 + r) * job.o_stride_t + j, o.cast[DType.bfloat16]()
                )


# ------------------------------------------------------------------- driver


struct V2Dispatch(Copyable, ImplicitlyCopyable, Movable):
    """State of one parallel region (pack, barrier, attention), on the stack.

    The per-worker scratch buffers are allocated by the caller (they must
    outlive the region) and travel as addresses, because the GOMP entry point
    only gets one `Int`.
    """

    var pack: PackJob
    var job: V2Job
    var pack_items: Int
    var items: Int

    def __init__(
        out self, pack: PackJob, job: V2Job, pack_items: Int, items: Int
    ):
        self.pack = pack
        self.job = job
        self.pack_items = pack_items
        self.items = items


@inline(.always)
def _v2_worker[
    dc32: Int, fp32: Bool
](state: V2Dispatch, tid: Int, threads: Int) -> None:
    """One worker: pack its share, wait, then run its share of the attention."""
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    tile_cfg_load(Int(cfg))
    # The scratch buffers live *here*, not in the caller: one set per GOMP
    # thread. Allocating them once and handing the address to every worker --
    # which is what the first version of this file did -- is a data race that
    # corrupts the logits of two work items running at the same time, and it
    # measured 2.5-3.8x slower on top.
    # alignment=64 is not cosmetic: every tile access and every SIMD access in
    # the softmax is a 64-byte access, and the default alignment is the element
    # size, so an unaligned buffer splits a cache line per access.
    comptime nlo = 2 if fp32 else 1
    # The softmax's inner loop reads logits[i] and writes prob[i] with i the
    # same offset, so if the two buffers are a multiple of 4 KB apart every
    # one of those pairs 4K-aliases and the core cannot disambiguate the
    # store from the load in the same cycle (Intel's partial-address alias
    # stall). `QROWS * MAXKT * 4` is exactly 64 KB, i.e. the worst possible
    # distance, so the probability slabs are shifted by 64 bytes: they stay
    # 64-byte aligned for the tile loads and stop being 4K-congruent.
    # Measured: **no change**, so 4K aliasing is not what makes the softmax
    # cost 2x its isolated rate. The shift is kept because it is free.
    var logits = stack_allocation[QROWS * MAXKT, Float32, alignment=64]()
    var prob_pad = stack_allocation[
        QROWS * MAXKT * nlo + 96, BFloat16, alignment=64
    ]()
    var acc = stack_allocation[QROWS * DMAX, Float32, alignment=64]()
    var qpad = stack_allocation[QROWS * DMAX * nlo, BFloat16, alignment=64]()
    var lp = _ptr[Float32](Int(logits))
    var pp = _ptr[BFloat16](Int(prob_pad) + 64)
    var pplo = _ptr[BFloat16](Int(prob_pad) + 64 + QROWS * MAXKT * 2 + 64)
    var ap = _ptr[Float32](Int(acc))
    var qp = _ptr[BFloat16](Int(qpad))
    var qplo = _ptr[BFloat16](Int(qpad) + QROWS * DMAX * 2)
    var nbh = state.pack.batch * state.pack.num_kv_heads
    var kinds = state.pack.kb16 * nbh
    var task = tid
    if (state.job.mode & 1) == 0:
        while task < state.pack_items:
            if task < kinds:
                if (state.job.mode & 8) == 0:
                    var kb = task % state.pack.kb16
                    var bh = task // state.pack.kb16
                    _pack_k_slab[fp32](
                        state.pack,
                        bh // state.pack.num_kv_heads,
                        bh % state.pack.num_kv_heads,
                        kb,
                    )
            elif (state.job.mode & 4) == 0:
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
    if threads > 1:
        external_call["GOMP_barrier", NoneType]()
    task = tid
    if (state.job.mode & 2) == 0:
        var nqg = (state.job.seq_q + QROWS - 1) // QROWS
        while task < state.items:
            var g = task % nqg
            var rest = task // nqg
            var head = rest % state.job.num_heads
            var batch = rest // state.job.num_heads
            _v2_item[dc32, fp32](
                state.job, lp, pp, pplo, ap, qp, qplo, batch, head, g * QROWS
            )
            task += threads
    tile_release()


@export
def _v2_gomp_bf16_d1(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[1, False](state, tid, threads)


@export
def _v2_gomp_bf16_d2(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[2, False](state, tid, threads)


@export
def _v2_gomp_bf16_d4(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[4, False](state, tid, threads)


@export
def _v2_gomp_fp32_d1(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[1, True](state, tid, threads)


@export
def _v2_gomp_fp32_d2(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[2, True](state, tid, threads)


@export
def _v2_gomp_fp32_d4(data: Int) abi("C"):
    var st = MutPointer[V2Dispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _v2_worker[4, True](state, tid, threads)


def run_dense_amx_v2(pack: PackJob, job: V2Job, num_workers: Int) -> None:
    """Pack K/V and run the attention, in one fork-join."""
    if job.batch == 0 or job.num_heads == 0 or job.seq_q <= 0:
        return
    var items = job.batch * job.num_heads * ((job.seq_q + QROWS - 1) // QROWS)
    var pack_items = pack.batch * pack.num_kv_heads * (pack.kb16 + pack.kb32)
    var workers = min(num_workers, max(items, max(pack_items, 1)))
    var state = stack_allocation[1, V2Dispatch]()
    state[] = V2Dispatch(pack, job, pack_items, items)
    if workers <= 1:
        if job.fp32:
            if job.dc32 == 4:
                _v2_worker[4, True](state[], 0, 1)
            elif job.dc32 == 2:
                _v2_worker[2, True](state[], 0, 1)
            else:
                _v2_worker[1, True](state[], 0, 1)
        else:
            if job.dc32 == 4:
                _v2_worker[4, False](state[], 0, 1)
            elif job.dc32 == 2:
                _v2_worker[2, False](state[], 0, 1)
            else:
                _v2_worker[1, False](state[], 0, 1)
        return
    var key = job.dc32 * 2 + Int(job.fp32)
    if key == 9:
        comptime e = _v2_gomp_fp32_d4
        external_call["GOMP_parallel", NoneType](
            e, Int(state), Int32(workers), Int32(0)
        )
    elif key == 5:
        comptime e2 = _v2_gomp_fp32_d2
        external_call["GOMP_parallel", NoneType](
            e2, Int(state), Int32(workers), Int32(0)
        )
    elif key == 3:
        comptime e3 = _v2_gomp_fp32_d1
        external_call["GOMP_parallel", NoneType](
            e3, Int(state), Int32(workers), Int32(0)
        )
    elif key == 8:
        comptime e4 = _v2_gomp_bf16_d4
        external_call["GOMP_parallel", NoneType](
            e4, Int(state), Int32(workers), Int32(0)
        )
    elif key == 4:
        comptime e5 = _v2_gomp_bf16_d2
        external_call["GOMP_parallel", NoneType](
            e5, Int(state), Int32(workers), Int32(0)
        )
    else:
        comptime e6 = _v2_gomp_bf16_d1
        external_call["GOMP_parallel", NoneType](
            e6, Int(state), Int32(workers), Int32(0)
        )


def has_amx2() -> Bool:
    """Whether this build targets a CPU with AMX (so the kernel above exists).
    """
    return AMX2
