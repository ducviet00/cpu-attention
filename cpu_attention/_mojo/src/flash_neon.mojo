"""The aarch64 `bfmmla` policy of flash_attn_func (bf16 dense, Armv8.6 BF16).

`bfmmla` is a 2x2 fp32 tile `acc[i][j] += dot(a[4i..4i+3], b[4j..4j+3])`: `a`
holds two *rows* of four contraction lanes, `b` two *columns*. 16 MACs per
instruction against 8 for `bfdot` (2.2-3.2x over it in the previous kernel).

* Q@K^T: 2 query rows x 4 head dims against 2 keys x 4 head dims. The logits
  *are* the accumulators, in tile order (r0k0, r0k1, r1k0, r1k1).
* P@V: 2 query rows x 4 keys against 2 head dims x 4 keys; the output tile is
  (r0d0, r0d1, r1d0, r1d1).
* The softmax runs on the tile as it is: a row pair's max / offset / alpha are
  vectors in (r0, r0, r1, r1) lane order, and the bf16 probabilities leave as
  the P@V `a` operand ([r0 k0..k3][r1 k0..k3]) with one zip.

Packed slab (per batch, head; keys padded to 16):
  K  `[key block of 8][hd/4][4 key pairs][8]`: four dims of two keys per 16 B,
     so Q@K^T walks one linear stream (hd * 16 B per 8 keys);
  V  `[hd/8 chunk][key group of 4][4 dim pairs][8]`: a P@V pass over 8 dims
     reads one linear stream, 64 B per 4 keys.
"""

from std.memory import bitcast
from std.sys import llvm_intrinsic

from flash_attn import FlashJob, FlashPolicy, NEG
from isa import exp2, online_max, ptr, vmax

comptime F4 = SIMD[DType.float32, 4]
comptime F2 = SIMD[DType.float32, 2]
comptime B8 = SIMD[DType.bfloat16, 8]
comptime B4 = SIMD[DType.bfloat16, 4]
comptime U4 = SIMD[DType.uint32, 4]
comptime U2 = SIMD[DType.uint32, 2]

comptime KT = 512  # keys per softmax tile
comptime RQT = 8  # query rows per register tile (4 row pairs)
comptime PAD = 16  # slab keys are padded to a multiple of this


@always_inline
def _pad(n: Int) -> Int:
    return (n + PAD - 1) // PAD * PAD


@always_inline
def _bfmmla(acc: F4, a: B8, b: B8) -> F4:
    return llvm_intrinsic[
        "llvm.aarch64.neon.bfmmla", F4, has_side_effect=False
    ](acc, a, b)


@always_inline
def _ninf() -> F4:
    return F4(bitcast[DType.float32, 1](SIMD[DType.uint32, 1](0xFF800000))[0])


@always_inline
def _swap(v: F4) -> F4:
    """(v1, v0, v3, v2): exchange the two key lanes of each row."""
    return v.shuffle[1, 0, 3, 2]()


@always_inline
def _bf16(e: F4) -> SIMD[DType.uint16, 4]:
    """fp32 -> bf16 bits by round-half-up and a narrowing shift (add + shrn):
    `bfcvtn` issues once per cycle on one pipe (measured), these spread."""
    var b = bitcast[DType.uint32, 4](e) + SIMD[DType.uint32, 4](0x8000)
    return (b >> SIMD[DType.uint32, 4](16)).cast[DType.uint16]()


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
def _pack[hd: Int](job: FlashJob, b: Int, h: Int, slab: Int):
    comptime DG = hd // 4
    var n = job.seq_k
    var pad = _pad(n)
    var ks = ptr[BFloat16](job.k + (b * job.kb + h * job.kh) * 2)
    var vs = ptr[BFloat16](job.v + (b * job.vb + h * job.vh) * 2)
    var kd = ptr[BFloat16](slab)
    var vd = ptr[BFloat16](slab + pad * hd * 2)
    var zero = B8(0)
    # K: four keys at a time; their four dims of one group are 32 contiguous
    # bytes in the packed block (two adjacent key pairs).
    var t0 = 0
    while t0 < pad:
        var blk = kd + ((t0 >> 3) * DG * 4 + ((t0 >> 1) & 3)) * 8
        comptime for dg2 in range(0, DG, 2):
            var r0 = zero
            var r1 = zero
            var r2 = zero
            var r3 = zero
            if t0 < n:
                r0 = (ks + t0 * job.kt).unsafe_load[width=8](dg2 * 4)
            if t0 + 1 < n:
                r1 = (ks + (t0 + 1) * job.kt).unsafe_load[width=8](dg2 * 4)
            if t0 + 2 < n:
                r2 = (ks + (t0 + 2) * job.kt).unsafe_load[width=8](dg2 * 4)
            if t0 + 3 < n:
                r3 = (ks + (t0 + 3) * job.kt).unsafe_load[width=8](dg2 * 4)
            blk.unsafe_store[width=8](
                dg2 * 32,
                r0.slice[4, offset=0]().join(r1.slice[4, offset=0]()),
            )
            blk.unsafe_store[width=8](
                dg2 * 32 + 8,
                r2.slice[4, offset=0]().join(r3.slice[4, offset=0]()),
            )
            blk.unsafe_store[width=8](
                (dg2 + 1) * 32,
                r0.slice[4, offset=4]().join(r1.slice[4, offset=4]()),
            )
            blk.unsafe_store[width=8](
                (dg2 + 1) * 32 + 8,
                r2.slice[4, offset=4]().join(r3.slice[4, offset=4]()),
            )
        t0 += 4
    # V: [chunk of 8 dims][key group][4 dim pairs][8]; a dim pair holds, for
    # each of its two dims, the four keys of the group.
    var kgn = pad // 4
    for kg in range(kgn):
        var t = kg * 4
        comptime for dc in range(hd // 8):
            var r0 = zero
            var r1 = zero
            var r2 = zero
            var r3 = zero
            if t < n:
                r0 = (vs + t * job.vt).unsafe_load[width=8](dc * 8)
            if t + 1 < n:
                r1 = (vs + (t + 1) * job.vt).unsafe_load[width=8](dc * 8)
            if t + 2 < n:
                r2 = (vs + (t + 2) * job.vt).unsafe_load[width=8](dc * 8)
            if t + 3 < n:
                r3 = (vs + (t + 3) * job.vt).unsafe_load[width=8](dc * 8)
            var z01 = r0.interleave(r1)  # (k0 k1) per dim, dims 0..7
            var z23 = r2.interleave(r3)
            var a = bitcast[DType.uint32, 4](z01.slice[8, offset=0]())  # dims 0-3
            var bb = bitcast[DType.uint32, 4](z01.slice[8, offset=8]())  # dims 4-7
            var c = bitcast[DType.uint32, 4](z23.slice[8, offset=0]())
            var e = bitcast[DType.uint32, 4](z23.slice[8, offset=8]())
            var dst = vd + (dc * kgn + kg) * 32
            var p0 = a.interleave(c)  # (a0 c0 a1 c1 | a2 c2 a3 c3)
            var p1 = bb.interleave(e)
            dst.unsafe_store[width=8](0, bitcast[DType.bfloat16, 16](p0).slice[8, offset=0]())
            dst.unsafe_store[width=8](8, bitcast[DType.bfloat16, 16](p0).slice[8, offset=8]())
            dst.unsafe_store[width=8](16, bitcast[DType.bfloat16, 16](p1).slice[8, offset=0]())
            dst.unsafe_store[width=8](24, bitcast[DType.bfloat16, 16](p1).slice[8, offset=8]())


# ---------------------------------------------------------------------------
# Q@K^T: logits in tile order, row maxima per row pair
# ---------------------------------------------------------------------------
@always_inline
def _qk[
    hd: Int, RP: Int
](kd: Int, qp: Int, lg: Int, mxo: Int, k0: Int, nk: Int, limit0: Int):
    comptime DG = hd // 4
    var kend = k0 + nk
    var lim = min(kend - 1, limit0)
    var kbase = ptr[BFloat16](kd)
    var qpp = ptr[BFloat16](qp)
    var lgp = ptr[Float32](lg)
    var ninf = _ninf()
    var mx = Array[F4, RP](fill=ninf)
    var j = 0
    while j < nk:
        var jj = k0 + j
        var kp_ = kbase + (jj >> 3) * (DG * 32)
        var accs = Array[F4, RP * 4](fill=F4(0))
        var dg = 0
        # Runtime loop over dim groups with a small unroll: fully unrolled, LLVM
        # hoists the Q panels out of the key loop and spills them.
        while dg < DG:
            comptime for u in range(4):
                var bv = Array[B8, 4](uninitialized=True)
                comptime for kp in range(4):
                    bv[kp] = kp_.unsafe_load[width=8]((dg + u) * 32 + kp * 8)
                comptime for rp in range(RP):
                    var a = qpp.unsafe_load[width=8](((dg + u) * RP + rp) * 8)
                    comptime for kp in range(4):
                        accs[rp * 4 + kp] = _bfmmla(accs[rp * 4 + kp], a, bv[kp])
            dg += 4
        if jj + 7 <= lim:
            comptime for rp in range(RP):
                var o = lgp + (rp * (KT // 2) + (j >> 1)) * 4
                comptime for kp in range(4):
                    o.unsafe_store[width=4](kp * 4, accs[rp * 4 + kp])
                mx[rp] = vmax[4](
                    mx[rp],
                    vmax[4](
                        vmax[4](accs[rp * 4], accs[rp * 4 + 1]),
                        vmax[4](accs[rp * 4 + 2], accs[rp * 4 + 3]),
                    ),
                )
        else:
            var ib = SIMD[DType.int32, 4](
                Int32(jj), Int32(jj + 1), Int32(jj), Int32(jj + 1)
            )
            comptime for rp in range(RP):
                var l0 = Int32(min(limit0 + 2 * rp, kend - 1))
                var l1 = Int32(min(limit0 + 2 * rp + 1, kend - 1))
                var lv = SIMD[DType.int32, 4](l0, l0, l1, l1)
                var o = lgp + (rp * (KT // 2) + (j >> 1)) * 4
                var m4 = ninf
                comptime for kp in range(4):
                    var x = (ib + Int32(2 * kp)).gt(lv).select(ninf, accs[rp * 4 + kp])
                    o.unsafe_store[width=4](kp * 4, x)
                    m4 = vmax[4](m4, x)
                mx[rp] = vmax[4](mx[rp], m4)
        j += 8
    comptime for rp in range(RP):
        ptr[Float32](mxo).unsafe_store[width=4](rp * 4, mx[rp])


# ---------------------------------------------------------------------------
# online softmax: P (bf16, the P@V `a` operand layout) and m / l / alpha
# ---------------------------------------------------------------------------
@always_inline
def _softmax[
    RP: Int, FIX: Bool
](
    lg: Int,
    pb: Int,
    mxv: F4,
    rp: Int,
    nblk: Int,
    c: Float32,
    sub: Int,
    m: Int,
    l: Int,
    alpha: Int,
):
    var r0 = 2 * rp
    var r1 = min(2 * rp + 1, sub - 1)
    var mp = ptr[Float32](m)
    var lp = ptr[Float32](l)
    var ap = ptr[Float32](alpha)
    var mt = vmax[4](mxv, _swap(mxv))  # (r0, r0, r1, r1) tile max, logit units
    var mo0 = mp.unsafe_load(r0)
    var mo1 = mp.unsafe_load(r1)
    var mo = F4(mo0, mo0, mo1, mo1)
    var step = online_max[4, 4](mo, mt * F4(c))
    var mn = step[0]
    var a4 = step[1]
    var live = 2 * rp + 1 < sub
    mp.unsafe_store(r0, mn[0])
    ap.unsafe_store(r0, a4[0])
    if live:
        mp.unsafe_store(r1, mn[2])
        ap.unsafe_store(r1, a4[2])
    var off = F4(0) - mn
    var c4 = F4(c)
    var lgp = ptr[Float32](lg) + rp * (KT // 2) * 4
    var pp = ptr[BFloat16](pb)
    var s = F4(0)
    var dead = SIMD[DType.bool, 4](fill=False)
    comptime if FIX:
        dead = mt.eq(_ninf())
    for blk in range(nblk):
        var e0 = exp2[2, 4](lgp.unsafe_load[width=4](blk * 16).fma(c4, off))
        var e1 = exp2[2, 4](lgp.unsafe_load[width=4](blk * 16 + 4).fma(c4, off))
        var e2 = exp2[2, 4](lgp.unsafe_load[width=4](blk * 16 + 8).fma(c4, off))
        var e3 = exp2[2, 4](lgp.unsafe_load[width=4](blk * 16 + 12).fma(c4, off))
        comptime if FIX:
            # A row that sees no key in this tile contributes exactly 0.
            e0 = dead.select(F4(0), e0)
            e1 = dead.select(F4(0), e1)
            e2 = dead.select(F4(0), e2)
            e3 = dead.select(F4(0), e3)
        s += (e0 + e1) + (e2 + e3)
        var q0 = bitcast[DType.uint32, 2](_bf16(e0))
        var q1 = bitcast[DType.uint32, 2](_bf16(e1))
        var q2 = bitcast[DType.uint32, 2](_bf16(e2))
        var q3 = bitcast[DType.uint32, 2](_bf16(e3))
        # (r0 k0k1, r1 k0k1) and (r0 k2k3, r1 k2k3) -> r0 k0..k3 | r1 k0..k3
        pp.unsafe_store[width=8](
            ((2 * blk) * RP + rp) * 8,
            bitcast[DType.bfloat16, 8](q0.interleave(q1)),
        )
        pp.unsafe_store[width=8](
            ((2 * blk + 1) * RP + rp) * 8,
            bitcast[DType.bfloat16, 8](q2.interleave(q3)),
        )
    lp.unsafe_store(r0, lp.unsafe_load(r0) * a4[0] + (s[0] + s[1]))
    if live:
        lp.unsafe_store(r1, lp.unsafe_load(r1) * a4[2] + (s[2] + s[3]))


# ---------------------------------------------------------------------------
# P@V: acc += alpha * (P @ V) in passes of 8 dims
# ---------------------------------------------------------------------------
@always_inline
def _pv[
    hd: Int, RP: Int
](vd: Int, kgn: Int, pb: Int, acc: Int, alpha: Int, sub: Int, k0: Int, nkg: Int):
    var ap = ptr[Float32](alpha)
    var accp = ptr[Float32](acc)
    var vbase = ptr[BFloat16](vd)
    var pbase = ptr[BFloat16](pb)
    for dc in range(hd // 8):
        var accs = Array[F4, RP * 4](fill=F4(0))
        if k0 > 0:
            # (the driver zeroes the accumulator, so the first tile skips this)
            comptime for rp in range(RP):
                var r1 = min(2 * rp + 1, sub - 1)
                var a0 = ap.unsafe_load(2 * rp)
                var a1 = ap.unsafe_load(r1)
                var a4 = F4(a0, a0, a1, a1)
                comptime for i in range(4):
                    var lo = accp.unsafe_load[width=2]((2 * rp) * hd + dc * 8 + 2 * i)
                    var hi = accp.unsafe_load[width=2]((2 * rp + 1) * hd + dc * 8 + 2 * i)
                    accs[rp * 4 + i] = lo.join(hi) * a4
        var vp = vbase + (dc * kgn + (k0 >> 2)) * 32
        var pp = pbase
        var kg = 0
        while kg + 2 <= nkg:
            comptime for u in range(2):
                var vb = Array[B8, 4](uninitialized=True)
                comptime for i in range(4):
                    vb[i] = vp.unsafe_load[width=8](u * 32 + i * 8)
                comptime for rp in range(RP):
                    var pa = pp.unsafe_load[width=8]((u * RP + rp) * 8)
                    comptime for i in range(4):
                        accs[rp * 4 + i] = _bfmmla(accs[rp * 4 + i], pa, vb[i])
            vp += 64
            pp += 2 * RP * 8
            kg += 2
        if kg < nkg:
            var vb = Array[B8, 4](uninitialized=True)
            comptime for i in range(4):
                vb[i] = vp.unsafe_load[width=8](i * 8)
            comptime for rp in range(RP):
                var pa = pp.unsafe_load[width=8](rp * 8)
                comptime for i in range(4):
                    accs[rp * 4 + i] = _bfmmla(accs[rp * 4 + i], pa, vb[i])
        comptime for rp in range(RP):
            comptime for i in range(4):
                var t = accs[rp * 4 + i]
                accp.unsafe_store[width=2]((2 * rp) * hd + dc * 8 + 2 * i, t.slice[2, offset=0]())
                accp.unsafe_store[width=2]((2 * rp + 1) * hd + dc * 8 + 2 * i, t.slice[2, offset=2]())


# ---------------------------------------------------------------------------
# scratch layout (bytes): logits | P | Q panel | row maxima
# ---------------------------------------------------------------------------
comptime LG_BYTES = RQT * KT * 4
comptime P_BYTES = (RQT // 2) * (KT // 4) * 16


@always_inline
def _tile[
    hd: Int, RP: Int
](
    job: FlashJob,
    slab: Int,
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
    comptime DG = hd // 4
    var pad = _pad(job.seq_k)
    var lg = scratch
    var pb = scratch + LG_BYTES
    var qp = pb + P_BYTES
    var mxo = qp + (RQT // 2) * DG * 16
    # Q panel [dim group][row pair][8]: four dims of row 2p, then of row 2p+1.
    var qb = ptr[BFloat16](q0)
    var qpp = ptr[BFloat16](qp)
    var zero = B4(0)
    for dg in range(DG):
        comptime for rp in range(RP):
            var lo = zero
            var hi = zero
            if 2 * rp < sub:
                lo = (qb + (2 * rp) * job.qt).unsafe_load[width=4](dg * 4)
            if 2 * rp + 1 < sub:
                hi = (qb + (2 * rp + 1) * job.qt).unsafe_load[width=4](dg * 4)
            qpp.unsafe_store[width=8]((dg * RP + rp) * 8, lo.join(hi))
    _qk[hd, RP](slab, qp, lg, mxo, k0, nk, limit0)
    var nblk = (nk + 7) >> 3
    var mxp = ptr[Float32](mxo)
    comptime for rp in range(RP):
        var mxv = mxp.unsafe_load[width=4](rp * 4)
        # A row is dead when it sees no key in this tile (both lanes -inf).
        var mt = vmax[4](mxv, _swap(mxv))
        var any_dead = mt.eq(_ninf()).reduce_or()
        if any_dead:
            _softmax[RP, True](lg, pb, mxv, rp, nblk, c, sub, m, l, alpha)
        else:
            _softmax[RP, False](lg, pb, mxv, rp, nblk, c, sub, m, l, alpha)
    _pv[hd, RP](slab + pad * hd * 2, pad >> 2, pb, acc, alpha, sub, k0, (nk + 3) >> 2)


struct NeonMmlaPolicy[hd: Int](FlashPolicy):
    comptime DT = DType.bfloat16
    comptime HD = Self.hd
    comptime RQT = RQT
    comptime KT = KT
    comptime ZEROES_ACC = True

    @staticmethod
    def slab_bytes(seq_k: Int) -> Int:
        return 2 * _pad(seq_k) * Self.hd * 2

    @staticmethod
    def tile_bytes() -> Int:
        return LG_BYTES + P_BYTES + (RQT // 2) * (Self.hd // 4) * 16 + (RQT // 2) * 16

    @staticmethod
    def pack(job: FlashJob, b: Int, h: Int, slab: Int):
        _pack[Self.hd](job, b, h, slab)

    @staticmethod
    def tile(
        job: FlashJob,
        slab: Int,
        b: Int,
        h: Int,
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
        # A partial register tile dispatches fewer row pairs.
        if sub <= 2:
            _tile[Self.hd, 1](job, slab, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
        elif sub <= 4:
            _tile[Self.hd, 2](job, slab, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
        elif sub <= 6:
            _tile[Self.hd, 3](job, slab, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
        else:
            _tile[Self.hd, 4](job, slab, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
