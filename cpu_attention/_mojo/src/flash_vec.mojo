"""The vector policy of flash_attn_func: AVX-512 (AVX-512-BF16 dot products),
AVX2, and NEON fp32 from one LANES-generic source; the per-target differences are
compile-time constants (KV32, NS, KU32 from LANES and BCAST_FOLD).

Layouts: K^T blocked every KP keys, `[tile][hd][KP]` (bf16 `[tile][hd/2][KP][2]`,
the dim pair a dot instruction contracts); V row-major (fp32, read in place
while all of V fits LLC/2) or bf16 key pairs `[key/2][hd][2]` for a `dpbf16ps`
P@V. Q@K^T is an RQT-row x LANES-key register tile; P@V an RQT-row x 2*LANES-dim
tile.
"""

from std.math import iota
from std.memory import bitcast

from flash_attn import FlashJob, FlashPolicy, NEG, V
from isa import BCAST_FOLD, LANES, dot_bf16, exp2, exp2s, ptr, vmax

comptime KP = 64  # keys per K^T block
comptime KT = 256  # keys per softmax tile
comptime RQT = 8  # query rows per register tile
comptime PV_VECS = 2  # dim vectors per P@V pass: 1 was 1.28x slower at hd=128 fp32 (V re-read)
comptime KV32 = max(1, 8 // LANES)
"""fp32 key vectors per Q@K^T pass: at least 8 keys, so 4-lane NEON keeps
RQT x 2 = 16 independent FMA chains (4 pipes x 4 cycles on Neoverse-V2)."""
comptime NS = max(1, 8 // LANES)
"""Partial row sums in the softmax (independent adds on 4 lanes)."""
comptime KU32 = 1 if BCAST_FOLD else 4
"""fp32 P@V keys per step: 4 probabilities as one vector, a lane each, where a
broadcast does not fold into the FMA."""
comptime QPANEL = not BCAST_FOLD
"""fp32 query rows through a contiguous panel where they are loaded as vectors."""


@always_inline
def _pad(n: Int) -> Int:
    return (n + KP - 1) // KP * KP


@always_inline
def _k_off[dtype: DType, hd: Int](j: Int, d: Int) -> Int:
    """Element offset of key j, dim d in the K^T slab."""
    comptime if dtype == DType.bfloat16:
        return ((j // KP) * (hd // 2 * KP) + (d >> 1) * KP + j % KP) * 2 + (d & 1)
    else:
        return (j // KP) * (hd * KP) + d * KP + j % KP


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
@always_inline
def _transpose(mut r: Array[V, LANES]):
    """LANES x LANES in registers: log2(LANES) rounds of interleaving row i
    with row i + LANES/2."""
    comptime ROUNDS = 4 if LANES == 16 else (3 if LANES == 8 else 2)
    comptime for _s in range(ROUNDS):
        var t = Array[V, LANES](uninitialized=True)
        comptime for i in range(LANES // 2):
            var w = r[i].interleave(r[i + LANES // 2])
            t[2 * i] = w.slice[LANES, offset=0]()
            t[2 * i + 1] = w.slice[LANES, offset=LANES]()
        r = t^


def _pack[dtype: DType, hd: Int](job: FlashJob, b: Int, kvh: Int, slab: Int):
    comptime E = 2 if dtype == DType.bfloat16 else 4
    comptime BF = dtype == DType.bfloat16
    var pad = _pad(job.seq_k)
    var ks = ptr[Scalar[dtype]](job.k + (b * job.kb + kvh * job.kh) * E)
    var vs = ptr[Scalar[dtype]](job.v + (b * job.vb + kvh * job.vh) * E)
    var kd = ptr[Scalar[dtype]](slab)
    var vd = ptr[Scalar[dtype]](slab + pad * hd * E)
    var t0 = 0
    while t0 + LANES <= job.seq_k:
        comptime for d0 in range(0, hd, LANES):
            var r = Array[V, LANES](uninitialized=True)
            comptime for i in range(LANES):
                r[i] = ks.unsafe_load[width=LANES]((t0 + i) * job.kt + d0).cast[
                    DType.float32
                ]()
            _transpose(r)
            comptime if BF:
                comptime for d in range(0, LANES, 2):
                    kd.unsafe_store[width = 2 * LANES](
                        _k_off[dtype, hd](t0, d0 + d),
                        r[d].interleave(r[d + 1]).cast[dtype](),
                    )
            else:
                comptime for d in range(LANES):
                    kd.unsafe_store[width=LANES](
                        _k_off[dtype, hd](t0, d0 + d), r[d].cast[dtype]()
                    )
        t0 += LANES
    # Tail tokens; the unused lanes of their group are masked in Q@K^T.
    for t in range(t0, job.seq_k):
        for d in range(hd):
            kd.unsafe_store(_k_off[dtype, hd](t, d), ks.unsafe_load(t * job.kt + d))
    comptime if BF:
        # V as key pairs [key/2][hd][2]; an odd last key pairs with zeros.
        var t = 0
        while t < job.seq_k:
            var a = vs + t * job.vt
            var dst = vd + (t >> 1) * hd * 2
            comptime for d0 in range(0, hd, LANES):
                var va = a.unsafe_load[width=LANES](d0)
                var vb = SIMD[dtype, LANES](0)
                if t + 1 < job.seq_k:
                    vb = (a + job.vt).unsafe_load[width=LANES](d0)
                dst.unsafe_store[width = 2 * LANES](2 * d0, va.interleave(vb))
            t += 2
    else:
        if not job.v_direct:
            for t in range(job.seq_k):
                comptime for d0 in range(0, hd, LANES):
                    vd.unsafe_store[width=LANES](
                        t * hd + d0, (vs + t * job.vt).unsafe_load[width=LANES](d0)
                    )


# ---------------------------------------------------------------------------
# Q@K^T: logits[r][j] for ROWS rows x nkeys keys, masked
# ---------------------------------------------------------------------------
@always_inline
def _qk[
    dtype: DType, hd: Int, ROWS: Int
](
    job: FlashJob,
    kd: Int,
    q0: Int,
    qs: Int,
    nrows: Int,
    k0: Int,
    nkeys: Int,
    limit0: Int,
    logits: Int,
    tmax: Int,
):
    """Query rows are `qs` elements apart. Also writes each row's tile maximum
    to `tmax` (the softmax then skips its max pass over the tile)."""
    comptime BF = dtype == DType.bfloat16
    # bf16 K is stored as dim pairs on every tier, so bf16 always goes through
    # `dot_bf16` (one instruction with AVX-512-BF16/NEON-BF16, widened otherwise).
    comptime DOT = BF
    var kend = k0 + nkeys
    var mx = Array[V, ROWS](fill=V(NEG))
    var lim = min(kend, limit0 + 1)
    var lg = ptr[Float32](logits)
    var j = 0
    comptime KV = 1 if DOT else KV32
    comptime assert KP % (KV * LANES) == 0, "a Q@K^T pass stays inside one K^T block"
    while j < nkeys:
        var jj = k0 + j
        var c = Array[V, ROWS * KV](fill=V(0))
        # The pass's keys share one K^T block (KP is a multiple of the pass), so
        # one base and constant key/dim offsets: per-load block arithmetic kept
        # NEON from pairing the loads (`ldp`) and cost ~10% more instructions.
        var kb = ptr[Scalar[dtype]](kd) + _k_off[dtype, hd](jj, 0)
        comptime if DOT:
            var kp = ptr[BFloat16](Int(kb))
            # Runtime loop over dim pairs: fully unrolled, LLVM hoists the
            # query broadcasts out of the key loop and spills them.
            for d2 in range(0, hd // 2, 4):
                comptime for u in range(4):
                    var kv = kp.unsafe_load[width = 2 * LANES]((d2 + u) * 2 * KP)
                    comptime for r in range(ROWS):
                        var rr = min(r, nrows - 1)
                        var qv = SIMD[DType.int32, LANES](
                            ptr[Int32](q0 + rr * qs * 2).unsafe_load(d2 + u)
                        )
                        # The query pair goes last: `vdpbf16ps` folds a
                        # 32-bit memory broadcast only as its last source.
                        c[r] = dot_bf16[LANES](
                            c[r], kv, bitcast[DType.bfloat16, 2 * LANES](qv)
                        )
        else:
            var kp = kb
            for d in range(0, hd, 4):
                comptime if BCAST_FOLD:
                    comptime for u in range(4):
                        var kv = Array[V, KV](uninitialized=True)
                        comptime for i in range(KV):
                            kv[i] = kp.unsafe_load[width=LANES](
                                (d + u) * KP + i * LANES
                            ).cast[DType.float32]()
                        comptime for r in range(ROWS):
                            var rr = min(r, nrows - 1)
                            var qd = Float32(ptr[Scalar[dtype]](q0 + rr * qs * 4).unsafe_load(d + u))
                            comptime for i in range(KV):
                                c[r * KV + i] = V(qd).fma(kv[i], c[r * KV + i])
                else:
                    # K for 4 dims first, then per row one 4-dim query vector
                    # used by lane (one live query register).
                    var kv = Array[V, 4 * KV](uninitialized=True)
                    comptime for u in range(4):
                        comptime for i in range(KV):
                            kv[u * KV + i] = kp.unsafe_load[width=LANES](
                                (d + u) * KP + i * LANES
                            ).cast[DType.float32]()
                    comptime for r in range(ROWS):
                        var rr = min(r, nrows - 1)
                        var qv = ptr[Scalar[dtype]](q0 + rr * qs * 4).unsafe_load[width=4](d).cast[DType.float32]()
                        comptime for u in range(4):
                            comptime for i in range(KV):
                                c[r * KV + i] = V(qv[u]).fma(kv[u * KV + i], c[r * KV + i])
        comptime for i in range(KV):
            var ji = jj + i * LANES
            if ji + LANES <= lim:
                comptime for r in range(ROWS):
                    if r < nrows:
                        lg.unsafe_store[width=LANES](r * KT + j + i * LANES, c[r * KV + i])
                        mx[r] = vmax[LANES](mx[r], c[r * KV + i])
            else:
                var idx = iota[DType.int32, LANES](Int32(ji))
                var past = idx.ge(SIMD[DType.int32, LANES](Int32(kend)))
                # Row r sees key k iff k - limit0 <= r: one subtraction, then a
                # compile-time constant per row (8 runtime per-row limits spilled
                # and were reloaded on every masked pass on NEON).
                var rel = idx - Int32(limit0)
                comptime for r in range(ROWS):
                    if r < nrows:
                        var bad = past | rel.gt(SIMD[DType.int32, LANES](Int32(r)))
                        var x = bad.select(V(NEG), c[r * KV + i])
                        lg.unsafe_store[width=LANES](r * KT + j + i * LANES, x)
                        mx[r] = vmax[LANES](mx[r], x)
        j += KV * LANES
    comptime for r in range(ROWS):
        if r < nrows:
            ptr[Float32](tmax).unsafe_store(r, mx[r].reduce_max())


# ---------------------------------------------------------------------------
# online softmax over one tile; the scale is folded into exp2
# ---------------------------------------------------------------------------
@always_inline
def _softmax[
    dtype: DType
](logits: Int, nrows: Int, nkeys: Int, c: Float32, m: Int, l: Int, alpha: Int):
    comptime BF = dtype == DType.bfloat16
    var lg = ptr[Float32](logits)
    var pp = ptr[BFloat16](logits + RQT * KT * 4)  # bf16 probabilities
    var mp = ptr[Float32](m)
    var lp = ptr[Float32](l)
    var ap = ptr[Float32](alpha)
    for r in range(nrows):
        var row = lg + r * KT
        var tm = ap.unsafe_load(r)  # the tile max `_qk` left here
        var j = 0
        if tm == NEG:
            # No visible key in this tile (a causal row inside a register tile
            # whose last row does see keys): zero what P@V reads. Going through
            # exp2 would compare two roundings of -MAX*c, whose difference is far
            # outside exp2's range in either direction.
            ap.unsafe_store(r, 1.0)
            while j < nkeys:
                comptime if BF:
                    (pp + r * KT).unsafe_store[width=LANES](j, SIMD[DType.bfloat16, LANES](0))
                else:
                    row.unsafe_store[width=LANES](j, V(0))
                j += LANES
            continue
        var mxv = tm * c
        var mo = mp.unsafe_load(r)
        var mn = max(mo, mxv)
        var a = exp2s[5](mo - mn)
        mp.unsafe_store(r, mn)
        ap.unsafe_store(r, a)
        var s = Array[V, NS](fill=V(0))

        @always_inline
        def one(j: Int) capturing -> V:
            # bf16 probabilities are rounded to 8 bits right after this, so a
            # degree-2 polynomial (relative 1.7e-3) is inside that rounding;
            # fp32 takes the degree-4 minimax (2.6e-6, gate 2e-4).
            var e = exp2[2 if BF else 4, LANES](
                row.unsafe_load[width=LANES](j).fma(V(c), V(-mn))
            )
            comptime if BF:
                (pp + r * KT).unsafe_store[width=LANES](j, e.cast[DType.bfloat16]())
            else:
                row.unsafe_store[width=LANES](j, e)
            return e

        while j + NS * LANES <= nkeys:
            comptime for u in range(NS):
                s[u] += one(j + u * LANES)
            j += NS * LANES
        while j < nkeys:
            s[0] += one(j)
            j += LANES
        var sum = s[0]
        comptime for u in range(1, NS):
            sum += s[u]
        lp.unsafe_store(r, lp.unsafe_load(r) * a + sum.reduce_add())


# ---------------------------------------------------------------------------
# P@V: acc[r][hd] += p[r][0:nk] @ V[k0:k0+nk], dims in register chunks
# ---------------------------------------------------------------------------
@always_inline
def _pv[
    dtype: DType, hd: Int, ROWS: Int
](vsrc: Int, vstride: Int, k0: Int, nk: Int, logits: Int, acc: Int, alpha: Int, sub: Int):
    """acc[r] = acc[r] * alpha[r] + p[r] @ V over `nk` keys, in passes of NV dim
    vectors. The rescale rides the accumulator load, and a row tile's first key
    tile (always k0 == 0) starts from zero: no separate rescale pass, no clear."""
    comptime BF = dtype == DType.bfloat16
    comptime NV = PV_VECS
    comptime DC = NV * LANES
    var ap = ptr[Float32](acc)
    var lg = ptr[Float32](logits)
    var pp = ptr[Int32](logits + RQT * KT * 4)  # bf16 key pairs as words

    @always_inline
    def chunk(dc: Int) capturing:
        var c = Array[V, ROWS * NV](fill=V(0))
        if k0 > 0:
            comptime for r in range(ROWS):
                var a = ptr[Float32](alpha).unsafe_load(min(r, sub - 1))
                comptime for i in range(NV):
                    var x = ap.unsafe_load[width=LANES](r * hd + dc + i * LANES)
                    c[r * NV + i] = x if a == 1.0 else x * a
        comptime if BF:
            # Key pairs: one `dpbf16ps` folds two keys x LANES dims (2.05x the
            # phase over fp32 FMA in the C++ measurement).
            var vp = ptr[BFloat16](vsrc)
            var j = 0
            while j < nk:
                var row = ((k0 + j) >> 1) * hd * 2 + dc * 2
                var vv = Array[SIMD[DType.bfloat16, 2 * LANES], NV](uninitialized=True)
                comptime for i in range(NV):
                    vv[i] = vp.unsafe_load[width = 2 * LANES](row + i * 2 * LANES)
                comptime for r in range(ROWS):
                    var w = SIMD[DType.int32, LANES](pp.unsafe_load((r * KT + j) >> 1))
                    var pa = bitcast[DType.bfloat16, 2 * LANES](w)
                    comptime for i in range(NV):
                        c[r * NV + i] = dot_bf16[LANES](c[r * NV + i], vv[i], pa)
                j += 2
        else:
            var vp = ptr[Scalar[dtype]](vsrc)
            var j0 = 0
            comptime if KU32 > 1:
                while j0 + KU32 <= nk:
                    var vv = Array[V, KU32 * NV](uninitialized=True)
                    comptime for k in range(KU32):
                        comptime for i in range(NV):
                            vv[k * NV + i] = vp.unsafe_load[width=LANES](
                                (k0 + j0 + k) * vstride + dc + i * LANES
                            ).cast[DType.float32]()
                    comptime for r in range(ROWS):
                        var pk = lg.unsafe_load[width=KU32](r * KT + j0)
                        comptime for k in range(KU32):
                            comptime for i in range(NV):
                                c[r * NV + i] = vv[k * NV + i].fma(V(pk[k]), c[r * NV + i])
                    j0 += KU32
            for j in range(j0, nk):
                var row = (k0 + j) * vstride + dc
                var vv = Array[V, NV](uninitialized=True)
                comptime for i in range(NV):
                    vv[i] = vp.unsafe_load[width=LANES](row + i * LANES).cast[DType.float32]()
                comptime for r in range(ROWS):
                    var pr = V(lg.unsafe_load(r * KT + j))
                    comptime for i in range(NV):
                        c[r * NV + i] = pr.fma(vv[i], c[r * NV + i])
        comptime for r in range(ROWS):
            comptime for i in range(NV):
                ap.unsafe_store[width=LANES](r * hd + dc + i * LANES, c[r * NV + i])

    # Few small chunk bodies unroll (x86: 2-4 chunks, one key per step); NEON's
    # 4-key bodies loop, as an unrolled copy per chunk costs instruction cache.
    comptime if (hd // DC) * (1 if BF else KU32) <= 4:
        comptime for dc in range(0, hd, DC):
            chunk(dc)
    else:
        for dc in range(0, hd, DC):
            chunk(dc)


# ---------------------------------------------------------------------------
# one work item: rq query rows of (b, h)
# ---------------------------------------------------------------------------
@always_inline
def _tile[
    dtype: DType, hd: Int, ROWS: Int
](
    job: FlashJob,
    kd: Int,
    vsrc: Int,
    vstride: Int,
    q0: Int,
    sub: Int,
    k0: Int,
    nk: Int,
    limit0: Int,
    c: Float32,
    aux: Int,
    acc: Int,
    m: Int,
    l: Int,
    alpha: Int,
):
    comptime if QPANEL and dtype == DType.float32:
        # The caller's rows are a token stride apart (16 KB at 32 heads x 128
        # dims: all rows in one L1 set): copy them into a contiguous panel.
        var panel = aux + RQT * KT * 4
        for r in range(ROWS):
            var src = ptr[Float32](q0) + min(r, sub - 1) * job.qt
            comptime for d in range(0, hd, 4):
                ptr[Float32](panel).unsafe_store[width=4](r * hd + d, src.unsafe_load[width=4](d))
        _qk[dtype, hd, ROWS](job, kd, panel, hd, sub, k0, nk, limit0, aux, alpha)
    else:
        _qk[dtype, hd, ROWS](job, kd, q0, job.qt, sub, k0, nk, limit0, aux, alpha)
    _softmax[dtype](aux, sub, nk, c, m, l, alpha)
    _pv[dtype, hd, ROWS](vsrc, vstride, k0, nk, aux, acc, alpha, sub)


struct VecPolicy[dtype: DType, hd: Int](FlashPolicy):
    comptime DT = Self.dtype
    comptime HD = Self.hd
    comptime RQT = RQT
    comptime KT = KT
    comptime ZEROES_ACC = True

    @staticmethod
    def slab_bytes(seq_k: Int) -> Int:
        return 2 * _pad(seq_k) * Self.hd * (2 if Self.dtype == DType.bfloat16 else 4)

    @staticmethod
    def tile_bytes() -> Int:
        # fp32 logits | bf16 probabilities (bf16) or the Q panel (fp32, QPANEL)
        comptime BF = Self.dtype == DType.bfloat16
        return RQT * KT * 4 + (RQT * KT * 2 if BF else (RQT * Self.hd * 4 if QPANEL else 0))

    @staticmethod
    def pack(job: FlashJob, b: Int, kvh: Int, slab: Int):
        _pack[Self.dtype, Self.hd](job, b, kvh, slab)

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
        comptime E = 2 if Self.dtype == DType.bfloat16 else 4
        var vsrc = slab + _pad(job.seq_k) * Self.hd * E
        var vstride = Self.hd
        if job.v_direct:
            vsrc = job.v + (b * job.vb + kvh * job.vh) * E
            vstride = job.vt
        # A partial register tile dispatches a smaller instantiation instead of
        # repeating the last query row.
        if sub <= 2:
            _tile[Self.dtype, Self.hd, 2](job, slab, vsrc, vstride, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
        elif sub <= 4:
            _tile[Self.dtype, Self.hd, 4](job, slab, vsrc, vstride, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
        else:
            _tile[Self.dtype, Self.hd, RQT](job, slab, vsrc, vstride, q0, sub, k0, nk, limit0, c, scratch, acc, m, l, alpha)
