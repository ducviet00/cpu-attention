"""`flash_attn_func`: FlashAttention over supplied Q/K/V tensors.

MHA. Work item = (batch, head, `rq` consecutive query rows). Each (batch, head)
is first packed into a private slab -- K^T blocked every KP keys (the
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
from std.math import align_up, ceildiv
from std.memory import stack_allocation
from std.sys import size_of

from isa import LANES, LOG2E, NEG_INF, ptr
from parallel import barrier, parallel

comptime V = SIMD[DType.float32, LANES]


@fieldwise_init
struct FlashJob(ImplicitlyCopyable, Movable):
    """One call: addresses, the geometry with every stride (b/t/h: batch, token,
    head; in elements), and the launch parameters `_C.flash` computed."""

    var q: Int
    var k: Int
    var v: Int
    var out: Int
    var batch: Int
    var seq_q: Int
    var seq_k: Int
    var heads: Int
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
    var slab: Int  # bytes per (batch, head) slab
    var aux: Int  # bytes per thread
    var v_direct: Bool
    var scale: Float32


trait FlashPolicy:
    """One ISA's half of flash_attn_func: the packed K/V layout and a register
    tile of up to RQT query rows over one key tile. The driver owns scheduling,
    causal bounds, the online-softmax state and the epilogue."""

    comptime DT: DType
    comptime HD: Int
    comptime RQT: Int  # query rows per register tile
    comptime KT: Int  # keys per tile (also the logits row width)
    comptime ZEROES_ACC: Bool = False
    """The policy's first P@V on a row writes the accumulator instead of adding
    to it, so the driver need not clear it (16 KB of stores per work item at
    rq=64, hd=64)."""

    @staticmethod
    def slab_bytes(seq_k: Int) -> Int:
        """Packed K/V bytes for one (batch, head)."""
        ...

    @staticmethod
    def tile_bytes() -> Int:
        """Per-thread scratch the tile needs before the accumulator."""
        ...

    @staticmethod
    def pack(job: FlashJob, b: Int, h: Int, slab: Int):
        ...

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
        """Rows q0..q0+sub-1 against keys k0..k0+nk-1: update m/l, rescale the
        accumulator rows by alpha and add P@V into them. Row r sees keys
        <= limit0 + r."""
        ...


def aux_bytes[P: FlashPolicy](rq: Int) -> Int:
    """Per-thread scratch: tile scratch | accumulator | m, l, alpha."""
    var n = P.tile_bytes() + align_up(rq, P.RQT) * P.HD * 4 + 3 * rq * 4
    # 64-byte multiples: the split order's work counter (an Int64 atomic) sits
    # after the per-thread scratch, and aarch64 atomics fault when misaligned.
    return align_up(n, 64)


def _attend[
    P: FlashPolicy
](job: FlashJob, b: Int, h: Int, qblk: Int, slab: Int, aux: Int):
    comptime dtype = P.DT
    comptime hd = P.HD
    comptime RQT = P.RQT
    comptime KT = P.KT
    comptime E = size_of[dtype]()
    var q0 = qblk * job.rq
    var nrows = min(job.rq, job.seq_q - q0)
    var acc = aux + P.tile_bytes()
    var arows = align_up(job.rq, RQT)
    var m = acc + arows * hd * 4
    var l = m + job.rq * 4
    var alpha = l + job.rq * 4
    var c = job.scale * Float32(LOG2E)
    var diag = job.seq_k - job.seq_q
    var limit0 = (q0 + diag) if job.causal else (1 << 30)
    var accp = ptr[Float32](acc)
    comptime if not P.ZEROES_ACC:
        for i in range(0, arows * hd, LANES):
            accp.unsafe_store[width=LANES](i, V(0))
    for r in range(nrows):
        ptr[Float32](m).unsafe_store(r, NEG_INF)
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
                        P.tile(job, slab, b, h, qr, sub, k0, nk, lim, c, aux, ar, m + r0 * 4, l + r0 * 4, alpha + r0 * 4)
                r0 += RQT
            k0 += KT
    for r in range(nrows):
        var o = ptr[Scalar[dtype]](job.out + (b * job.ob + (q0 + r) * job.ot + h * job.oh) * E)
        var lr = ptr[Float32](l).unsafe_load(r)
        var inv = Float32(1) / lr if lr > 0 else Float32(0)
        var src = ptr[Float32](acc + r * hd * 4)
        if lr > 0:
            comptime for d in range(0, hd, LANES):
                o.unsafe_store[width=LANES](d, (src.unsafe_load[width=LANES](d) * inv).cast[dtype]())
        else:
            # A row that saw no key: exactly 0, without reading an accumulator
            # that a ZEROES_ACC policy never initialised (NaN * 0 is NaN).
            comptime for d in range(0, hd, LANES):
                o.unsafe_store[width=LANES](d, SIMD[dtype, LANES](0))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _body[P: FlashPolicy](ctx: Int, tid: Int, threads: Int):
    var job = ptr[FlashJob](ctx)[]
    var nqb = ceildiv(job.seq_q, job.rq)
    var nbh = job.batch * job.heads
    var cells = threads if job.fuse else nbh
    var aux = job.ws + cells * job.slab + tid * job.aux
    if job.fuse:
        # (batch, head) round-robin: neighbouring heads on neighbouring threads
        # (adjacent source chunks in token-major K/V).
        var slab = job.ws + tid * job.slab
        var i = tid
        while i < nbh:
            var b = i // job.heads
            var h = i - b * job.heads
            P.pack(job, b, h, slab)
            for qb in range(nqb):
                _attend[P](job, b, h, qb, slab, aux)
            i += threads
        return
    var i = tid
    while i < nbh:
        P.pack(job, i // job.heads, i % job.heads, job.ws + i * job.slab)
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
        _attend[P](job, b, h, qb, job.ws + hp * job.slab, aux)


def flash_drive[P: FlashPolicy](job: FlashJob):
    var st = stack_allocation[1, FlashJob]()
    st[] = job
    if not job.fuse:
        var cells = job.batch * job.heads
        ptr[Int64](job.ws + cells * job.slab + job.workers * job.aux)[] = 0
    comptime body = _body[P]
    parallel(body, Int(st), job.workers)


