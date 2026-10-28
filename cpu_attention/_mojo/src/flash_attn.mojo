"""`flash_attn_func`: FlashAttention over supplied Q/K/V tensors.

Work item = (batch, query head, `rq` consecutive query rows). Each (batch, kv
head) is first packed into a private slab -- K^T blocked every KP keys (the
`[tile][hd][KP]` / bf16 `[tile][hd/2][KP][2]` page layout the KV cache uses),
V row-major (fp32, or read in place while all of V fits LLC/2) or as bf16 key
pairs for a `dpbf16ps` P@V. The key range is then walked in tiles of KT keys,
the rows in register tiles of RQT: K/V are read once per `rq` rows.

Causal is bottom-right aligned (row i sees keys j <= i + seq_k - seq_q). The
key loop stops at the last row's limit and each register tile clamps to its
own, so no fully masked block is computed; a row that sees no key is exactly 0.

Order: *fused* (a thread packs a head, then attends all of its query blocks,
no barrier) while workers' slabs fit 0.6 of the LLC; otherwise *split* (pack
every head, barrier, then a dynamic boustrophedon over query blocks so the
threads share one head's slab through the LLC).
"""

from std.atomic import Atomic
from std.math import iota
from std.memory import bitcast, stack_allocation

from isa import BF16_DOT, LANES, LOG2E, dot_bf16, exp2, exp2s, ptr
from parallel import barrier, parallel

comptime KP = 64  # keys per K^T block
comptime KT = 256  # keys per softmax tile
comptime RQT = 8  # query rows per register tile
comptime NEG = -Float32.MAX
comptime V = SIMD[DType.float32, LANES]


struct FlashJob(ImplicitlyCopyable, Movable):
    var q: Int
    var k: Int
    var v: Int
    var out: Int
    var batch: Int
    var seq_q: Int
    var seq_k: Int
    var heads: Int
    var kv_heads: Int
    var qb: Int
    var qt: Int
    var qh: Int
    var kb: Int
    var kt: Int
    var kh: Int
    var vb: Int
    var vt: Int
    var vh: Int
    var ob: Int
    var ot: Int
    var oh: Int
    var causal: Bool
    var workers: Int
    var fuse: Bool
    var rq: Int
    var ws: Int
    var slab: Int  # bytes per (batch, kv head) slab
    var aux: Int  # bytes per thread
    var v_direct: Bool
    var scale: Float32

    def __init__(out self, plan: Int, q: Int, k: Int, v: Int, out_: Int, scale: Float32):
        var p = ptr[Int64](plan)
        self.q = q
        self.k = k
        self.v = v
        self.out = out_
        self.batch = Int(p[0])
        self.seq_q = Int(p[1])
        self.seq_k = Int(p[2])
        self.heads = Int(p[3])
        self.kv_heads = Int(p[4])
        self.qb = Int(p[6])
        self.qt = Int(p[7])
        self.qh = Int(p[8])
        self.kb = Int(p[9])
        self.kt = Int(p[10])
        self.kh = Int(p[11])
        self.vb = Int(p[12])
        self.vt = Int(p[13])
        self.vh = Int(p[14])
        self.ob = Int(p[15])
        self.ot = Int(p[16])
        self.oh = Int(p[17])
        self.causal = p[19] != 0
        self.workers = Int(p[20])
        self.fuse = p[21] != 0
        self.rq = Int(p[22])
        self.ws = Int(p[23])
        self.slab = Int(p[24])
        self.aux = Int(p[25])
        self.v_direct = p[26] != 0
        self.scale = scale


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
    nrows: Int,
    k0: Int,
    nkeys: Int,
    limit0: Int,
    logits: Int,
):
    comptime BF = dtype == DType.bfloat16
    comptime DOT = BF and BF16_DOT
    var kend = k0 + nkeys
    var lim = min(kend, limit0 + 1)
    var lg = ptr[Float32](logits)
    var j = 0
    while j < nkeys:
        var jj = k0 + j
        var c = Array[V, ROWS](fill=V(0))
        comptime if DOT:
            var kp = ptr[BFloat16](kd)
            # Runtime loop over dim pairs: fully unrolled, LLVM hoists the
            # query broadcasts out of the key loop and spills them.
            for d2 in range(0, hd // 2, 4):
                comptime for u in range(4):
                    var kv = kp.unsafe_load[width = 2 * LANES](
                        _k_off[dtype, hd](jj, 2 * (d2 + u))
                    )
                    comptime for r in range(ROWS):
                        var rr = min(r, nrows - 1)
                        var qv = SIMD[DType.int32, LANES](
                            ptr[Int32](q0 + rr * job.qt * 2).unsafe_load(d2 + u)
                        )
                        c[r] = dot_bf16[LANES](
                            c[r], bitcast[DType.bfloat16, 2 * LANES](qv), kv
                        )
        else:
            var kp = ptr[Scalar[dtype]](kd)
            for d in range(0, hd, 4):
                comptime for u in range(4):
                    var kv = kp.unsafe_load[width=LANES](
                        _k_off[dtype, hd](jj, d + u)
                    ).cast[DType.float32]()
                    comptime for r in range(ROWS):
                        var rr = min(r, nrows - 1)
                        var qs = Float32(
                            ptr[Scalar[dtype]](q0 + rr * job.qt * 4).unsafe_load(d + u)
                        )
                        c[r] = V(qs).fma(kv, c[r])
        if jj + LANES <= lim:
            comptime for r in range(ROWS):
                if r < nrows:
                    lg.unsafe_store[width=LANES](r * KT + j, c[r])
        else:
            var idx = iota[DType.int32, LANES](Int32(jj))
            var past = idx.ge(SIMD[DType.int32, LANES](Int32(kend)))
            comptime for r in range(ROWS):
                if r < nrows:
                    var bad = past | idx.gt(SIMD[DType.int32, LANES](Int32(limit0 + r)))
                    lg.unsafe_store[width=LANES](r * KT + j, bad.select(V(NEG), c[r]))
        j += LANES


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
        var mx = V(NEG)
        var j = 0
        while j < nkeys:
            mx = max(mx, row.unsafe_load[width=LANES](j))
            j += LANES
        var mxv = mx.reduce_max() * c
        var mo = mp.unsafe_load(r)
        var mn = max(mo, mxv)
        var a = exp2s[5](mo - mn)
        mp.unsafe_store(r, mn)
        ap.unsafe_store(r, a)
        var sum = V(0)
        j = 0
        while j < nkeys:
            var e = exp2[5, LANES](row.unsafe_load[width=LANES](j).fma(V(c), V(-mn)))
            comptime if BF:
                (pp + r * KT).unsafe_store[width=LANES](j, e.cast[DType.bfloat16]())
            else:
                row.unsafe_store[width=LANES](j, e)
            sum += e
            j += LANES
        lp.unsafe_store(r, lp.unsafe_load(r) * a + sum.reduce_add())


# ---------------------------------------------------------------------------
# P@V: acc[r][hd] += p[r][0:nk] @ V[k0:k0+nk], dims in register chunks
# ---------------------------------------------------------------------------
@always_inline
def _pv[
    dtype: DType, hd: Int, ROWS: Int
](vsrc: Int, vstride: Int, k0: Int, nk: Int, logits: Int, acc: Int):
    comptime BF = dtype == DType.bfloat16
    comptime DOT = BF and BF16_DOT
    comptime DC = 2 * LANES
    var ap = ptr[Float32](acc)
    var lg = ptr[Float32](logits)
    var pp = ptr[Int32](logits + RQT * KT * 4)  # bf16 key pairs as words
    comptime for dc in range(0, hd, DC):
        var c0 = Array[V, ROWS](uninitialized=True)
        var c1 = Array[V, ROWS](uninitialized=True)
        comptime for r in range(ROWS):
            c0[r] = ap.unsafe_load[width=LANES](r * hd + dc)
            c1[r] = ap.unsafe_load[width=LANES](r * hd + dc + LANES)
        comptime if BF:
            # Key pairs: one `dpbf16ps` folds two keys x LANES dims (2.05x the
            # phase over fp32 FMA in the C++ measurement).
            var vp = ptr[BFloat16](vsrc)
            var j = 0
            while j < nk:
                var row = ((k0 + j) >> 1) * hd * 2 + dc * 2
                var v0 = vp.unsafe_load[width = 2 * LANES](row)
                var v1 = vp.unsafe_load[width = 2 * LANES](row + 2 * LANES)
                comptime for r in range(ROWS):
                    var w = SIMD[DType.int32, LANES](pp.unsafe_load((r * KT + j) >> 1))
                    var pa = bitcast[DType.bfloat16, 2 * LANES](w)
                    c0[r] = dot_bf16[LANES](c0[r], pa, v0)
                    c1[r] = dot_bf16[LANES](c1[r], pa, v1)
                j += 2
        else:
            var vp = ptr[Scalar[dtype]](vsrc)
            for j in range(nk):
                var row = (k0 + j) * vstride + dc
                var v0 = vp.unsafe_load[width=LANES](row).cast[DType.float32]()
                var v1 = vp.unsafe_load[width=LANES](row + LANES).cast[DType.float32]()
                comptime for r in range(ROWS):
                    var pr = V(lg.unsafe_load(r * KT + j))
                    c0[r] = pr.fma(v0, c0[r])
                    c1[r] = pr.fma(v1, c1[r])
        comptime for r in range(ROWS):
            ap.unsafe_store[width=LANES](r * hd + dc, c0[r])
            ap.unsafe_store[width=LANES](r * hd + dc + LANES, c1[r])


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
    _qk[dtype, hd, ROWS](job, kd, q0, sub, k0, nk, limit0, aux)
    _softmax[dtype](aux, sub, nk, c, m, l, alpha)
    var ap = ptr[Float32](alpha)
    var acp = ptr[Float32](acc)
    for r in range(sub):
        var a = ap.unsafe_load(r)
        # a == 1: no-op; a == 0: the row's first contributing tile, where the
        # accumulator is still exactly 0.
        if a != 1.0 and a != 0.0:
            comptime for d in range(0, hd, LANES):
                acp.unsafe_store[width=LANES](
                    r * hd + d, acp.unsafe_load[width=LANES](r * hd + d) * a
                )
    _pv[dtype, hd, ROWS](vsrc, vstride, k0, nk, aux, acc)


def _attend[
    dtype: DType, hd: Int
](job: FlashJob, b: Int, h: Int, qblk: Int, slab: Int, aux: Int):
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var group = job.heads // job.kv_heads
    var kvh = h // group
    var q0 = qblk * job.rq
    var nrows = min(job.rq, job.seq_q - q0)
    var acc = aux + RQT * KT * 4 + RQT * KT * 2
    var arows = (job.rq + RQT - 1) // RQT * RQT
    var m = acc + arows * hd * 4
    var l = m + job.rq * 4
    var alpha = l + job.rq * 4
    var c = job.scale * Float32(LOG2E)
    var diag = job.seq_k - job.seq_q
    var limit0 = (q0 + diag) if job.causal else (1 << 30)
    var pad = _pad(job.seq_k)
    var kd = slab
    var vsrc = slab + pad * hd * E
    var vstride = hd
    if job.v_direct:
        vsrc = job.v + (b * job.vb + kvh * job.vh) * E
        vstride = job.vt
    var accp = ptr[Float32](acc)
    for i in range(0, arows * hd, LANES):
        accp.unsafe_store[width=LANES](i, V(0))
    for r in range(nrows):
        ptr[Float32](m).unsafe_store(r, NEG)
        ptr[Float32](l).unsafe_store(r, 0)
    var qbase = job.q + (b * job.qb + q0 * job.qt + h * job.qh) * E
    var empty = job.causal and limit0 + nrows - 1 < 0
    if not empty:
        var kend = min(job.seq_k, limit0 + nrows) if job.causal else job.seq_k
        var k0 = 0
        while k0 < kend:
            var nkeys = min(KT, job.seq_k - k0)
            var r0 = 0
            while r0 < nrows:
                var sub = min(RQT, nrows - r0)
                var lim = limit0 + r0
                if not (job.causal and lim + sub - 1 < k0):
                    var nk = min(nkeys, lim + sub - k0) if job.causal else nkeys
                    if nk > 0:
                        var qr = qbase + r0 * job.qt * E
                        var ar = acc + r0 * hd * 4
                        if sub <= 2:
                            _tile[dtype, hd, 2](job, kd, vsrc, vstride, qr, sub, k0, nk, lim, c, aux, ar, m + r0 * 4, l + r0 * 4, alpha + r0 * 4)
                        elif sub <= 4:
                            _tile[dtype, hd, 4](job, kd, vsrc, vstride, qr, sub, k0, nk, lim, c, aux, ar, m + r0 * 4, l + r0 * 4, alpha + r0 * 4)
                        else:
                            _tile[dtype, hd, RQT](job, kd, vsrc, vstride, qr, sub, k0, nk, lim, c, aux, ar, m + r0 * 4, l + r0 * 4, alpha + r0 * 4)
                r0 += RQT
            k0 += KT
    for r in range(nrows):
        var o = ptr[Scalar[dtype]](job.out + (b * job.ob + (q0 + r) * job.ot + h * job.oh) * E)
        var lr = ptr[Float32](l).unsafe_load(r)
        var inv = Float32(1) / lr if lr > 0 else Float32(0)
        var src = ptr[Float32](acc + r * hd * 4)
        comptime for d in range(0, hd, LANES):
            o.unsafe_store[width=LANES](d, (src.unsafe_load[width=LANES](d) * inv).cast[dtype]())


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _body[dtype: DType, hd: Int](ctx: Int, tid: Int, threads: Int):
    var job = ptr[FlashJob](ctx)[]
    var nqb = (job.seq_q + job.rq - 1) // job.rq
    var nbh = job.batch * job.kv_heads
    var group = job.heads // job.kv_heads
    var cells = threads if job.fuse else nbh
    var aux = job.ws + cells * job.slab + tid * job.aux
    if job.fuse:
        # (batch, kv head) round-robin: neighbouring heads on neighbouring
        # threads (adjacent source chunks in token-major K/V).
        var slab = job.ws + tid * job.slab
        var i = tid
        while i < nbh:
            var b = i // job.kv_heads
            var kvh = i - b * job.kv_heads
            _pack[dtype, hd](job, b, kvh, slab)
            for qb in range(nqb):
                _attend[dtype, hd](job, b, kvh, qb, slab, aux)
            i += threads
        return
    var i = tid
    while i < nbh:
        _pack[dtype, hd](job, i // job.kv_heads, i % job.kv_heads, job.ws + i * job.slab)
        i += threads
    barrier()
    var counter = ptr[Int64](job.ws + cells * job.slab + threads * job.aux)
    var total = job.batch * job.heads * nqb
    while True:
        var t = Int(Atomic[Int64].fetch_add(counter, 1))
        if t >= total:
            break
        # Boustrophedon over one head's query blocks: causal work grows with
        # the block index, so pair the cheapest with the most expensive.
        var hp = t // nqb
        var raw = t - hp * nqb
        var qb = (nqb - 1 - (raw >> 1)) if (raw & 1) else (raw >> 1)
        var b = hp // job.heads
        var h = hp - b * job.heads
        _attend[dtype, hd](job, b, h, qb, job.ws + (b * job.kv_heads + h // group) * job.slab, aux)


def _run[dtype: DType, hd: Int](job: FlashJob):
    var st = stack_allocation[1, FlashJob]()
    st[] = job
    if not job.fuse:
        var cells = job.batch * job.kv_heads
        ptr[Int64](job.ws + cells * job.slab + job.workers * job.aux)[] = 0
    comptime body = _body[dtype, hd]
    parallel(body, Int(st), job.workers)


def flash_run(job: FlashJob, dtype: Int, hd: Int) raises:
    if dtype == 0:
        _hd[DType.float32](job, hd)
    elif dtype == 1:
        _hd[DType.bfloat16](job, hd)
    else:
        raise Error("cpu_attention: unsupported dtype")


def _hd[dtype: DType](job: FlashJob, hd: Int) raises:
    if hd == 64:
        _run[dtype, 64](job)
    elif hd == 128:
        _run[dtype, 128](job)
    elif hd == 32:
        _run[dtype, 32](job)
    else:
        raise Error("cpu_attention: head_dim must be 32, 64 or 128")


def flash_aux_bytes(rq: Int, hd: Int) -> Int:
    """Per-thread scratch: logits | bf16 probabilities | accumulator | m, l, alpha."""
    var arows = (rq + RQT - 1) // RQT * RQT
    return RQT * KT * 4 + RQT * KT * 2 + arows * hd * 4 + 3 * rq * 4 + 64


def flash_slab_bytes(seq_k: Int, hd: Int, dtype: Int) -> Int:
    var e = 2 if dtype == 1 else 4
    return 2 * _pad(seq_k) * hd * e
