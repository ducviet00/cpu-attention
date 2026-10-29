"""The AMX policy of flash_attn_func (x86 with amx-tile + amx-bf16).

One `tdpbf16ps` accumulates a 16x16 fp32 tile from two bf16 tiles. A register
tile is RQT = 32 query rows (two 16-row groups) and the key range is walked in
super-tiles of KT keys:

    Q@K^T  32 keys at a time (2 K tiles x 2 row groups = 4 accumulators), into
           an fp32 logits buffer [32][KT]
    softmax one pass over the super-tile, bf16 P (and, for fp32, its lo half)
    P@V    32 keys at a time, 32 output dims at a time, accumulating into the
           driver's fp32 accumulator rows

fp32 rides the same tiles with every operand split into bf16 hi + lo and three
products kept per phase (hi*hi + hi*lo + lo*hi): products are exact and the
accumulation is fp32, so the result is fp32-accurate (3-8e-5, gate 2e-4).

Layouts (1 KB per tile): K -> [key block 16][dim chunk 32][dim pair][key], V ->
[key block 32][dim chunk 16][key pair][dim]; a `tdpbf16ps` B tile is 16 rows of
16 32-bit units, the contraction pair sitting inside a unit. fp32 stores the lo
tile right after the hi tile of each slot.

A masked or padded logit is -1e30, so exp2 gives exactly +0, which P@V needs:
0 * garbage must be 0. Row statistics (max, running max, row sum) are computed 16
rows at a time and reduced across rows by a shuffle network, not per row.
"""

from std.math import iota
from std.memory import bitcast
from std.sys import llvm_intrinsic

from amx import tile_cfg_load, tile_dpbf16ps, tile_load, tile_store, tile_zero
from flash_attn import FlashJob, FlashPolicy, NEG
from isa import X86_BF16, bf16_lo, exp2, online_max, ptr, vmax

comptime RQT = 32  # query rows per register tile
comptime KSTEP = 32  # keys per Q@K^T / P@V step
comptime SLACK = 8.0  # FA4's running-max slack (log2 units): rescale only past it
comptime PAD = 64  # bytes added to the logits / P row strides: a power-of-two stride
# aliases in L1 (measured 3-4% on the whole call)
comptime MASKED = -1.0e30  # a masked logit; finite so the exponential needs no clamp
comptime F = DType.float32
comptime BF16 = DType.bfloat16
comptime V16 = SIMD[DType.float32, 16]
comptime I16 = SIMD[DType.int32, 16]


@always_inline
def _pad(n: Int) -> Int:
    return (n + KSTEP - 1) // KSTEP * KSTEP


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
@always_inline
def _row32(a: SIMD[BF16, 16], b: SIMD[BF16, 16]) -> I16:
    return bitcast[DType.int32, 16](a.join(b))


@always_inline
def _tp16(m: Array[I16, 16]) -> Array[I16, 16]:
    """16x16 transpose of 32-bit units in registers, in the instruction sequence
    the C++ pack uses (64 shuffles): unpack ps, unpack pd, then two stages of
    128-bit-lane shuffles. The index-vector form (`vpermt2d/q`, 16 data + the
    index registers) spilled and was 18% slower on the pack."""
    var r = m.copy()
    comptime for i in range(0, 16, 2):
        var a = r[i]
        var b = r[i + 1]
        r[i] = a.shuffle[0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29](b)
        r[i + 1] = a.shuffle[2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31](b)
    comptime for i in range(0, 16, 4):
        var a0 = r[i]
        var a1 = r[i + 1]
        var a2 = r[i + 2]
        var a3 = r[i + 3]
        r[i] = a0.shuffle[0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29](a2)
        r[i + 1] = a0.shuffle[2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31](a2)
        r[i + 2] = a1.shuffle[0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29](a3)
        r[i + 3] = a1.shuffle[2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31](a3)
    comptime for j in range(4):
        var a0 = r[j]
        var a1 = r[j + 4]
        var a2 = r[j + 8]
        var a3 = r[j + 12]
        var p0 = a0.shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](a1)
        var p1 = a0.shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](a1)
        var q0 = a2.shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](a3)
        var q1 = a2.shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](a3)
        r[j] = p0.shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](q0)
        r[j + 8] = p0.shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](q0)
        r[j + 4] = p1.shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](q1)
        r[j + 12] = p1.shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](q1)
    return r.copy()


@always_inline
def _k_bytes[dtype: DType, hd: Int](seq_k: Int) -> Int:
    comptime NLO = 1 if dtype == DType.bfloat16 else 2
    return (_pad(seq_k) // 16) * (hd // 32) * 512 * NLO * 2


@always_inline
def _pack_k_block[
    dtype: DType, hd: Int, full: Bool
](job: FlashJob, ks: Int, t0: Int, kb: Int, dc: Int, slab: Int):
    """One 1 KB tile: 16 keys x 32 dims transposed as 32-bit (dim-pair) units.
    Keys past seq_k are zero (a padded key must never inject a NaN); `full` is the
    common case with every key present and no per-row test."""
    comptime BF = dtype == DType.bfloat16
    comptime E = 2 if BF else 4
    comptime DC32 = hd // 32
    comptime SLOT = 512 * (1 if BF else 2)  # bf16 elements per (key block, dim chunk)
    var kts = job.kt * E
    var dst = ptr[Int32](slab + ((kb * DC32 + dc) * SLOT) * 2)
    var rows = Array[I16, 16](uninitialized=True)
    var rlo = Array[I16, 16](uninitialized=True)
    comptime for i in range(16):
        comptime if full:
            var s = ks + (t0 + i) * kts + dc * 32 * E
            comptime if BF:
                rows[i] = ptr[Int32](s).unsafe_load[width=16](0)
            else:
                var f0 = ptr[Float32](s).unsafe_load[width=16](0)
                var f1 = ptr[Float32](s).unsafe_load[width=16](16)
                var h0 = f0.cast[BF16]()
                var h1 = f1.cast[BF16]()
                rows[i] = _row32(h0, h1)
                rlo[i] = _row32(bf16_lo[16](f0, h0), bf16_lo[16](f1, h1))
        else:
            rows[i] = I16(0)
            rlo[i] = I16(0)
            if t0 + i < job.seq_k:
                var s = ks + (t0 + i) * kts + dc * 32 * E
                comptime if BF:
                    rows[i] = ptr[Int32](s).unsafe_load[width=16](0)
                else:
                    var f0 = ptr[Float32](s).unsafe_load[width=16](0)
                    var f1 = ptr[Float32](s).unsafe_load[width=16](16)
                    var h0 = f0.cast[BF16]()
                    var h1 = f1.cast[BF16]()
                    rows[i] = _row32(h0, h1)
                    rlo[i] = _row32(bf16_lo[16](f0, h0), bf16_lo[16](f1, h1))
    var cols = _tp16(rows)
    comptime for i in range(16):
        dst.unsafe_store[width=16](i * 16, cols[i])
    comptime if not BF:
        var clo = _tp16(rlo)
        comptime for i in range(16):
            dst.unsafe_store[width=16](256 + i * 16, clo[i])


@always_inline
def _pack_k[dtype: DType, hd: Int](job: FlashJob, b: Int, kvh: Int, slab: Int):
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var ks = job.k + (b * job.kb + kvh * job.kh) * E
    for kb in range(_pad(job.seq_k) // 16):
        var t0 = kb * 16
        if t0 + 16 <= job.seq_k:
            comptime for dc in range(hd // 32):
                _pack_k_block[dtype, hd, True](job, ks, t0, kb, dc, slab)
        else:
            comptime for dc in range(hd // 32):
                _pack_k_block[dtype, hd, False](job, ks, t0, kb, dc, slab)


@always_inline
def _pack_v_block[
    dtype: DType, hd: Int, full: Bool
](job: FlashJob, vs: Int, t0: Int, kb: Int, dc: Int, vbase: Int):
    """One 1 KB tile: 32 keys x 16 dims, a tile row = two adjacent keys of 16
    dims interleaved."""
    comptime BF = dtype == DType.bfloat16
    comptime E = 2 if BF else 4
    comptime DC16 = hd // 16
    comptime SLOT = 512 * (1 if BF else 2)
    var vts = job.vt * E
    var dst = ptr[BFloat16](vbase + ((kb * DC16 + dc) * SLOT) * 2)
    comptime for r in range(16):
        var ka = t0 + 2 * r
        comptime if BF:
            var a = SIMD[BF16, 16](0)
            var c = SIMD[BF16, 16](0)
            comptime if full:
                a = ptr[BFloat16](vs + ka * vts + dc * 16 * E).unsafe_load[width=16](0)
                c = ptr[BFloat16](vs + (ka + 1) * vts + dc * 16 * E).unsafe_load[width=16](0)
            else:
                if ka < job.seq_k:
                    a = ptr[BFloat16](vs + ka * vts + dc * 16 * E).unsafe_load[width=16](0)
                if ka + 1 < job.seq_k:
                    c = ptr[BFloat16](vs + (ka + 1) * vts + dc * 16 * E).unsafe_load[width=16](0)
            dst.unsafe_store[width=32](r * 32, a.interleave(c))
        else:
            var fa = V16(0)
            var fc = V16(0)
            comptime if full:
                fa = ptr[Float32](vs + ka * vts + dc * 16 * E).unsafe_load[width=16](0)
                fc = ptr[Float32](vs + (ka + 1) * vts + dc * 16 * E).unsafe_load[width=16](0)
            else:
                if ka < job.seq_k:
                    fa = ptr[Float32](vs + ka * vts + dc * 16 * E).unsafe_load[width=16](0)
                if ka + 1 < job.seq_k:
                    fc = ptr[Float32](vs + (ka + 1) * vts + dc * 16 * E).unsafe_load[width=16](0)
            var h0 = fa.cast[BF16]()
            var h1 = fc.cast[BF16]()
            dst.unsafe_store[width=32](r * 32, h0.interleave(h1))
            dst.unsafe_store[width=32](512 + r * 32, bf16_lo[16](fa, h0).interleave(bf16_lo[16](fc, h1)))


@always_inline
def _pack_v[dtype: DType, hd: Int](job: FlashJob, b: Int, kvh: Int, slab: Int):
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var vs = job.v + (b * job.vb + kvh * job.vh) * E
    var vbase = slab + _k_bytes[dtype, hd](job.seq_k)
    for kb in range(_pad(job.seq_k) // 32):
        var t0 = kb * 32
        if t0 + 32 <= job.seq_k:
            comptime for dc in range(hd // 16):
                _pack_v_block[dtype, hd, True](job, vs, t0, kb, dc, vbase)
        else:
            comptime for dc in range(hd // 16):
                _pack_v_block[dtype, hd, False](job, vs, t0, kb, dc, vbase)


# ---------------------------------------------------------------------------
# exp2 and the bf16 probability store
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Q@K^T
# ---------------------------------------------------------------------------
@always_inline
def _q_operand[
    dtype: DType, hd: Int
](job: FlashJob, qaddr: Int, nrows: Int, panel: Int) -> Tuple[Int, Int, Int]:
    """(hi address, lo address, row stride in bytes) of the A operand: the caller's
    Q when the block is full bf16 (the tile load takes any stride in bytes), else a
    contiguous zero-padded hi/lo panel -- a partial block still computes RQT rows
    and the padding must not reach the logits; fp32 needs its halves.

    A full bf16 block is also panelled when its rows are >= 2 KB apart (heads * hd
    * 2) and the sweep has >= 256 keys: all 32 rows then share one L1 set (12 ways)
    and the physical L2 sets vary per process. 257:16:128 on 4 threads was bimodal,
    208-226 us in place against 197-200 us through the panel; 512:32:128 -10%,
    1024:16:128 -6%. Below that the copy is a net cost (129:16:128 +6%, 2048:8:64
    +2%), and C++ reads Q in place everywhere."""
    comptime BF = dtype == DType.bfloat16
    comptime E = 2 if BF else 4
    comptime DC32 = hd // 32
    if BF and nrows == RQT and not (job.qt * E >= 2048 and job.seq_k >= 256):
        return (qaddr, 0, job.qt * E)
    var ph = ptr[BFloat16](panel)
    var pl = ptr[BFloat16](panel + RQT * hd * 2)
    for r in range(RQT):
        comptime for dc in range(DC32):
            if r < nrows:
                var s = qaddr + r * job.qt * E + dc * 32 * E
                comptime if BF:
                    ph.unsafe_store[width=32](r * hd + dc * 32, ptr[BFloat16](s).unsafe_load[width=32](0))
                else:
                    var f0 = ptr[Float32](s).unsafe_load[width=16](0)
                    var f1 = ptr[Float32](s).unsafe_load[width=16](16)
                    var h0 = f0.cast[BF16]()
                    var h1 = f1.cast[BF16]()
                    ph.unsafe_store[width=32](r * hd + dc * 32, h0.join(h1))
                    pl.unsafe_store[width=32](r * hd + dc * 32, bf16_lo[16](f0, h0).join(bf16_lo[16](f1, h1)))
            else:
                ph.unsafe_store[width=32](r * hd + dc * 32, SIMD[BF16, 32](0))
                comptime if not BF:
                    pl.unsafe_store[width=32](r * hd + dc * 32, SIMD[BF16, 32](0))
    return (panel, panel + RQT * hd * 2, hd * 2)


@always_inline
def _qk[
    dtype: DType, hd: Int, KT: Int
](
    kslab: Int,
    a_hi: Int,
    a_lo: Int,
    astride: Int,
    nrows: Int,
    k0: Int,
    nkeys: Int,
    limit0: Int,
    lg: Int,
):
    """logits[RQT][nkeys] for keys k0.. , 32 keys per step, masked past a row's
    causal limit and past the key range."""
    comptime BF = dtype == DType.bfloat16
    comptime DC32 = hd // 32
    comptime SLOT = 512 * (1 if BF else 2)
    comptime LS = KT * 4 + PAD
    var lim = min(k0 + nkeys, limit0 + 1)
    var kend = k0 + nkeys
    var lane = iota[DType.int32, 16](0)
    var masked = V16(MASKED)
    var j = 0
    while j < nkeys:
        var jj = k0 + j
        var kb = jj >> 4
        tile_zero[4]()
        tile_zero[5]()
        tile_zero[6]()
        tile_zero[7]()
        comptime for dc in range(DC32):
            var kp = kslab + ((kb * DC32 + dc) * SLOT) * 2
            var kp1 = kslab + (((kb + 1) * DC32 + dc) * SLOT) * 2
            tile_load[0](a_hi + dc * 64, astride)
            tile_load[1](a_hi + 16 * astride + dc * 64, astride)
            tile_load[2](kp, 64)
            tile_load[3](kp1, 64)
            tile_dpbf16ps[4, 0, 2]()
            tile_dpbf16ps[5, 0, 3]()
            tile_dpbf16ps[6, 1, 2]()
            tile_dpbf16ps[7, 1, 3]()
            comptime if not BF:
                # q_hi @ k_lo, then q_lo @ k_hi (k_hi reloads from L1).
                tile_load[2](kp + 1024, 64)
                tile_load[3](kp1 + 1024, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                tile_load[0](a_lo + dc * 64, astride)
                tile_load[1](a_lo + 16 * astride + dc * 64, astride)
                tile_load[2](kp, 64)
                tile_load[3](kp1, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
        var lp = lg + j * 4
        tile_store[4](lp, LS)
        tile_store[5](lp + 64, LS)
        tile_store[6](lp + 16 * LS, LS)
        tile_store[7](lp + 16 * LS + 64, LS)
        if jj + KSTEP > lim:
            # Boundary step: the stores above already happened, so a NaN out of a
            # padded K tile is overwritten, not propagated.
            for r in range(nrows):
                var rlim = Int32(limit0 + r)
                var rp = ptr[Float32](lg + r * LS + j * 4)
                comptime for half in range(2):
                    var idx = lane + I16(Int32(jj + 16 * half))
                    var bad = idx.gt(I16(rlim)) | idx.ge(I16(Int32(kend)))
                    rp.unsafe_store[width=16](16 * half, bad.select(masked, rp.unsafe_load[width=16](16 * half)))
        j += KSTEP


# ---------------------------------------------------------------------------
# online softmax: row statistics vectorised over 16 rows
# ---------------------------------------------------------------------------
@always_inline
def _op[is_max: Bool](a: V16, b: V16) -> V16:
    comptime if is_max:
        return vmax[16](a, b)
    else:
        return a + b


@always_inline
def _hnet[is_max: Bool](v: Array[V16, 16]) -> V16:
    """Lane i of the result = the horizontal max / sum of v[i], for 16 vectors at
    once (45 vector ops against ~100 for sixteen separate reductions): unpack ps
    and pd halve the vector count in-lane, then two 128-bit-lane stages."""
    var u = Array[V16, 8](uninitialized=True)
    comptime for p in range(8):
        var a = v[2 * p]
        var b = v[2 * p + 1]
        u[p] = _op[is_max](
            a.shuffle[0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29](b),
            a.shuffle[2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31](b),
        )
    var w = Array[V16, 4](uninitialized=True)
    comptime for q in range(4):
        var a = u[2 * q]
        var b = u[2 * q + 1]
        w[q] = _op[is_max](
            a.shuffle[0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29](b),
            a.shuffle[2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31](b),
        )
    var z0 = _op[is_max](
        w[0].shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](w[1]),
        w[0].shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](w[1]),
    )
    var z1 = _op[is_max](
        w[2].shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](w[3]),
        w[2].shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](w[3]),
    )
    return _op[is_max](
        z0.shuffle[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27](z1),
        z0.shuffle[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31](z1),
    )


@always_inline
def _ld16(p: Int, valid: Int, fill: Float32) -> V16:
    """16 floats from p; lanes past `valid` read as `fill` (the arrays hold only
    one entry per row of the work item)."""
    if valid >= 16:
        return ptr[Float32](p).unsafe_load[width=16](0)
    var t = V16(fill)
    for i in range(valid):
        t[i] = ptr[Float32](p).unsafe_load(i)
    return t


@always_inline
def _st16(p: Int, v: V16, valid: Int):
    if valid >= 16:
        ptr[Float32](p).unsafe_store[width=16](0, v)
    else:
        for i in range(valid):
            ptr[Float32](p).unsafe_store(i, v[i])


@always_inline
def _stats[
    KT: Int
](lg: Int, nrows: Int, n: Int, c: Float32, m: Int, l: Int, alpha: Int) -> Bool:
    """Max pass and the running-max update, 16 rows at a time. Sets m (new),
    alpha (the rescale factor, 1 where none) and scales l by it; returns whether
    any row needs a rescale."""
    comptime LS = KT * 4 + PAD
    var changed = False
    var rg = 0
    while rg < nrows:
        var valid = min(16, nrows - rg)
        var rmax = Array[V16, 16](uninitialized=True)
        comptime for r4 in range(0, 16, 4):
            # Four independent rows hide the max's dependency chain.
            var mx = Array[V16, 4](fill=V16(MASKED))
            var j = 0
            while j < n:
                comptime for rr in range(4):
                    mx[rr] = vmax[16](mx[rr], ptr[Float32](lg + (rg + r4 + rr) * LS + j * 4).unsafe_load[width=16](0))
                j += 16
            comptime for rr in range(4):
                rmax[r4 + rr] = mx[rr]
        var raw = _hnet[True](rmax)
        var mxv = raw * V16(c)
        var mo = _ld16(m + rg * 4, valid, NEG)
        var visible = raw.gt(V16(MASKED * 0.1))
        var first = mo.le(V16(NEG))
        var need = mxv.gt(mo + V16(SLACK))
        var take = visible & (first | need)
        var rescale = visible & need & ~first
        var a = V16(1)
        var any = rescale.reduce_or()
        if any:
            a = rescale.select(online_max[16, 4, SLACK](mo, mxv)[1], V16(1))
            changed = True
        _st16(m + rg * 4, take.select(mxv, mo), valid)
        _st16(alpha + rg * 4, a, valid)
        if any:
            _st16(l + rg * 4, _ld16(l + rg * 4, valid, 0) * a, valid)
        rg += 16
    return changed


@always_inline
def _cvt2(a: V16, b: V16) -> SIMD[BF16, 32]:
    """a (keys 0-15) and b (keys 16-31) -> 32 bf16: one `vcvtne2ps2bf16` (3 uops)
    against two `vcvtneps2bf16` + an insert (5)."""
    comptime if X86_BF16:
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.cvtne2ps2bf16.512", SIMD[BF16, 32], has_side_effect=False
        ](b, a)
    else:
        return a.join(b).cast[BF16]()


@always_inline
def _emit32[
    dtype: DType
](lp: Int, pp: Int, pl: Int, vc: V16, nbase: V16) -> V16:
    """P for 32 keys of one row (and its lo half for fp32); returns their sum."""
    comptime BF = dtype == DType.bfloat16
    var l = ptr[Float32](lp)
    var e0 = exp2[2 if BF else 4, 16](l.unsafe_load[width=16](0).fma(vc, nbase))
    var e1 = exp2[2 if BF else 4, 16](l.unsafe_load[width=16](16).fma(vc, nbase))
    var hh = _cvt2(e0, e1)
    ptr[BFloat16](pp).unsafe_store[width=32](0, hh)
    comptime if not BF:
        var hf = hh.cast[F]()
        var lo = _cvt2(e0 - hf.slice[16, offset=0](), e1 - hf.slice[16, offset=16]())
        ptr[BFloat16](pl).unsafe_store[width=32](0, lo)
    return e0 + e1


@always_inline
def _emit[
    dtype: DType, KT: Int
](lg: Int, pb: Int, nrows: Int, n: Int, c: Float32, m: Int, l: Int):
    """P = 2^(c*logit - m) into the P buffer, and l += row sums (reduced 16 rows
    at a time). Keys in [n, pad32(n)) are masked, so they come out as the exact
    zeros P@V needs past the key range."""
    comptime LS = KT * 4 + PAD
    comptime PS = KT * 2 + PAD
    var ntail = _pad(n)
    var vc = V16(c)
    var mp = ptr[Float32](m)
    var rg = 0
    while rg < nrows:
        var valid = min(16, nrows - rg)
        var sums = Array[V16, 16](uninitialized=True)
        comptime for r4 in range(0, 16, 4):
            # Four independent rows hide the exponential's dependency chain. Rows
            # past the work item's last repeat its final row (their P is unused).
            var nb = Array[V16, 4](uninitialized=True)
            var sm = Array[V16, 4](fill=V16(0))
            comptime for rr in range(4):
                var mr = mp.unsafe_load(min(rg + r4 + rr, nrows - 1))
                nb[rr] = V16(-mr if mr > NEG else Float32(0))
            var j = 0
            while j < ntail:
                comptime for rr in range(4):
                    var row = rg + r4 + rr
                    sm[rr] += _emit32[dtype](
                        lg + row * LS + j * 4,
                        pb + row * PS + j * 2,
                        pb + RQT * PS + row * PS + j * 2,
                        vc,
                        nb[rr],
                    )
                j += KSTEP
            comptime for rr in range(4):
                sums[r4 + rr] = sm[rr]
        _st16(l + rg * 4, _ld16(l + rg * 4, valid, 0) + _hnet[False](sums), valid)
        rg += 16


# ---------------------------------------------------------------------------
# P@V
# ---------------------------------------------------------------------------
@always_inline
def _pv[
    dtype: DType, hd: Int, KT: Int
](vslab: Int, k0: Int, nkeys: Int, pb: Int, acc: Int, first: Bool):
    """acc[32][hd] += P[32][nkeys] @ V[k0:k0+nkeys]; `first` zeroes the tiles
    instead of loading the accumulator (the driver does not clear it)."""
    comptime BF = dtype == DType.bfloat16
    comptime DC16 = hd // 16
    comptime SLOT = 512 * (1 if BF else 2)
    comptime PS = KT * 2 + PAD
    var plo = pb + RQT * PS
    var dc = 0
    while dc < DC16:
        var cc0 = acc + dc * 64
        var cc1 = acc + 16 * hd * 4 + dc * 64
        if first:
            tile_zero[4]()
            tile_zero[5]()
            tile_zero[6]()
            tile_zero[7]()
        else:
            tile_load[4](cc0, hd * 4)
            tile_load[5](cc0 + 64, hd * 4)
            tile_load[6](cc1, hd * 4)
            tile_load[7](cc1 + 64, hd * 4)
        var ks = 0
        while ks < nkeys:
            var kb = (k0 + ks) >> 5
            var vp = vslab + ((kb * DC16 + dc) * SLOT) * 2
            var pp = pb + ks * 2
            tile_load[0](pp, PS)
            tile_load[1](pp + 16 * PS, PS)
            tile_load[2](vp, 64)
            tile_load[3](vp + SLOT * 2, 64)
            tile_dpbf16ps[4, 0, 2]()
            tile_dpbf16ps[5, 0, 3]()
            tile_dpbf16ps[6, 1, 2]()
            tile_dpbf16ps[7, 1, 3]()
            comptime if not BF:
                # p_hi @ v_lo, then p_lo @ v_hi
                tile_load[2](vp + 1024, 64)
                tile_load[3](vp + SLOT * 2 + 1024, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                tile_load[0](plo + ks * 2, PS)
                tile_load[1](plo + ks * 2 + 16 * PS, PS)
                tile_load[2](vp, 64)
                tile_load[3](vp + SLOT * 2, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
            ks += KSTEP
        tile_store[4](cc0, hd * 4)
        tile_store[5](cc0 + 64, hd * 4)
        tile_store[6](cc1, hd * 4)
        tile_store[7](cc1 + 64, hd * 4)
        dc += 2


@always_inline
def _cfg(addr: Int):
    """Palette 1, eight 16-row x 64-byte tiles: the same geometry serves A, B and
    C in both phases."""
    var v = SIMD[DType.uint8, 64](0)
    v[0] = 1
    comptime for i in range(8):
        v[16 + 2 * i] = 64
        v[48 + i] = 16
    ptr[UInt8](addr).unsafe_store[width=64](0, v)


struct AmxPolicy[dtype: DType, hd: Int](FlashPolicy):
    comptime DT = Self.dtype
    comptime HD = Self.hd
    comptime RQT = RQT
    # 256 keys for bf16, 512 for fp32 (whose P also carries a lo half).
    comptime KT = 256 if Self.dtype == DType.bfloat16 else 512
    comptime ZEROES_ACC = True  # P@V starts with tile_zero on a row's first key tile

    @staticmethod
    def slab_bytes(seq_k: Int) -> Int:
        comptime NLO = 1 if Self.dtype == DType.bfloat16 else 2
        return _k_bytes[Self.dtype, Self.hd](seq_k) + (_pad(seq_k) // 32) * (Self.hd // 16) * 512 * NLO * 2

    @staticmethod
    def tile_bytes() -> Int:
        # logits | bf16 P (hi, then lo for fp32) | Q panel (hi, lo) | tile config
        comptime NLO = 1 if Self.dtype == DType.bfloat16 else 2
        return RQT * (Self.KT * 4 + PAD) + RQT * (Self.KT * 2 + PAD) * NLO + RQT * Self.hd * 2 * NLO + 64

    @staticmethod
    def pack(job: FlashJob, b: Int, kvh: Int, slab: Int):
        _pack_k[Self.dtype, Self.hd](job, b, kvh, slab)
        _pack_v[Self.dtype, Self.hd](job, b, kvh, slab)

    @staticmethod
    def tile(
        job: FlashJob,
        slab: Int,
        b: Int,
        kvh: Int,
        q0: Int,
        sub: Int,
        k0: Int,
        nk: Int,
        limit0: Int,
        c: Float32,
        scratch: Int,
        acc: Int,
        m: Int,
        l: Int,
        alpha: Int,
    ):
        comptime NLO = 1 if Self.dtype == DType.bfloat16 else 2
        comptime KT = Self.KT
        var lg = scratch
        var pb = lg + RQT * (KT * 4 + PAD)
        var panel = pb + RQT * (KT * 2 + PAD) * NLO
        if k0 == 0:
            # A query block's sweep always starts at key 0: load the tile config
            # once there instead of per call. The tile state can be dropped
            # between calls (another AMX user, a `tilerelease`, a migration) and
            # a tile instruction in INIT state is #UD, but nothing else runs on
            # this thread inside one sweep.
            var cfg = panel + RQT * Self.hd * 2 * NLO
            _cfg(cfg)
            tile_cfg_load(cfg)
        var qo = _q_operand[Self.dtype, Self.hd](job, q0, sub, panel)
        _qk[Self.dtype, Self.hd, KT](slab, qo[0], qo[1], qo[2], sub, k0, nk, limit0, lg)
        if _stats[KT](lg, sub, nk, c, m, l, alpha):
            # Some row's running max moved (rare with the slack): rescale its
            # accumulator before this tile is added.
            var ap = ptr[Float32](alpha)
            var acp = ptr[Float32](acc)
            for r in range(sub):
                var a = ap.unsafe_load(r)
                if a != 1.0:
                    comptime for d in range(0, Self.hd, 16):
                        acp.unsafe_store[width=16](r * Self.hd + d, acp.unsafe_load[width=16](r * Self.hd + d) * a)
        _emit[Self.dtype, KT](lg, pb, sub, nk, c, m, l)
        _pv[Self.dtype, Self.hd, KT](slab + _k_bytes[Self.dtype, Self.hd](job.seq_k), k0, nk, pb, acc, k0 == 0)
