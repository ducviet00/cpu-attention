"""KV-cache attention: `flash_attn_with_kvcache` decode and `reshape_and_cache`.

Contract: one query token per sequence, attending every cached position (the
query is the last position, so causality is the cache length). The cache is
`[num_blocks, kv_heads, page, head_dim]` with this backend's in-page layout
(`PageLayout`); `reshape_and_cache` writes it and nothing else reads it.

Work item = (sequence, kv head, a chunk of that kv head's query heads). Every
query head of a GQA/MQA group is a row of one register tile, so the group shares
one scan of K/V -- the cache is read once per kv head, not once per query head.
When there are fewer (sequence, kv head) pairs than workers, the group is split
into chunks so every worker has a share; the store then happens in the same
region, before a barrier.
"""

from std.math import iota
from std.memory import bitcast, stack_allocation
from std.sys.info import CompilationTarget

from isa import LANES, LOG2E, BF16_DOT, dot_bf16, exp2, exp2s, ptr
from parallel import barrier, parallel

comptime V_PAIRS_ARM = CompilationTarget.has_neon()
"""aarch64 stores bf16 V as key pairs (`[page/2][hd][2]`) so `bfdot` can contract
two keys; x86 keeps `[page][hd]` (fp32 P@V, bit-parity class with C++)."""


struct PagedJob(ImplicitlyCopyable, Movable):
    """One call, as raw addresses. `k`/`v`/`slots` are only read by the store."""

    var q: Int
    var k: Int
    var v: Int
    var k_cache: Int
    var v_cache: Int
    var out: Int
    var slots: Int
    var block_table: Int
    var lens: Int
    var num_seqs: Int
    var num_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var page: Int
    var bt_stride: Int
    var q_stride: Int
    var kv_stride: Int
    var out_stride: Int
    var scale: Float32
    var dtype: Int  # 0 fp32, 1 bf16
    var workers: Int
    var do_store: Bool
    var do_attend: Bool
    var chunks: Int  # query-head chunks per (sequence, kv head)

    def __init__(out self):
        self.q = 0
        self.k = 0
        self.v = 0
        self.k_cache = 0
        self.v_cache = 0
        self.out = 0
        self.slots = 0
        self.block_table = 0
        self.lens = 0
        self.num_seqs = 0
        self.num_heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.page = 0
        self.bt_stride = 0
        self.q_stride = 0
        self.kv_stride = 0
        self.out_stride = 0
        self.scale = 1.0
        self.dtype = 0
        self.workers = 1
        self.do_store = False
        self.do_attend = True
        self.chunks = 1


# ---------------------------------------------------------------------------
# page layout
# ---------------------------------------------------------------------------
struct PageLayout[dtype: DType, hd: Int]:
    """Where token `t` of a (page, kv head) lives. K is dim-major so one vector
    load is LANES consecutive tokens of one dimension; bf16 pairs the two
    dimensions a dot instruction contracts: `[hd/2][page][2]`."""

    comptime BF = Self.dtype == DType.bfloat16
    comptime V_PAIRS = Self.BF and V_PAIRS_ARM
    comptime SPLIT = Self.BF and not Self.V_PAIRS and Self.hd % (2 * LANES) == 0
    """bf16 V is widened even/odd-split, so the fp32 accumulator holds each
    2*LANES block of dims as [even dims][odd dims] (see `_pv`)."""

    @staticmethod
    @always_inline
    def k_elem(page: Int, t: Int, d: Int) -> Int:
        comptime if Self.BF:
            return ((d >> 1) * page + t) * 2 + (d & 1)
        else:
            return d * page + t

    @staticmethod
    @always_inline
    def v_elem(page: Int, t: Int, d: Int) -> Int:
        comptime if Self.V_PAIRS:
            return ((t >> 1) * Self.hd + d) * 2 + (t & 1)
        else:
            return t * Self.hd + d

    @staticmethod
    @always_inline
    def store_token(k_page: Int, v_page: Int, page: Int, t: Int, k_src: Int, v_src: Int):
        """Scatter one token's K and V (one kv head) into its page."""
        var ks = ptr[Scalar[Self.dtype]](k_src)
        var vs = ptr[Scalar[Self.dtype]](v_src)
        var kd = ptr[Scalar[Self.dtype]](k_page)
        var vd = ptr[Scalar[Self.dtype]](v_page)
        comptime if Self.BF:
            comptime for d2 in range(Self.hd // 2):
                kd.unsafe_store[width=2](
                    Self.k_elem(page, t, 2 * d2), ks.unsafe_load[width=2](2 * d2)
                )
        else:
            comptime for d in range(Self.hd):
                kd.unsafe_store(Self.k_elem(page, t, d), ks.unsafe_load(d))
        comptime if Self.V_PAIRS:
            comptime for d in range(Self.hd):
                vd.unsafe_store(Self.v_elem(page, t, d), vs.unsafe_load(d))
        else:
            vd.unsafe_store[width=Self.hd](
                Self.v_elem(page, t, 0), vs.unsafe_load[width=Self.hd]()
            )


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
@always_inline
def _store_kv[dtype: DType, hd: Int](job: PagedJob, seq: Int, kvh: Int):
    var slot = Int(ptr[Int64](job.slots).unsafe_load(seq))
    if slot < 0:  # padding: no token to store (C++ semantics)
        return
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var pg = slot // job.page
    var base = ((pg * job.kv_heads + kvh) * job.page * hd) * E
    var src = (seq * job.kv_stride + kvh * hd) * E
    L.store_token(
        job.k_cache + base, job.v_cache + base, job.page, slot - pg * job.page,
        job.k + src, job.v + src,
    )


# ---------------------------------------------------------------------------
# attend: R query rows of one kv head against one sequence's pages
# ---------------------------------------------------------------------------
@always_inline
def _kt[hd: Int]() -> Int:
    # Keys per softmax step. The probability tile (R x KT fp32) and one KT-key V
    # tile stay inside L1 next to the K stream.
    return 256 if hd <= 64 else 128


@always_inline
def _attend_rows[
    dtype: DType, hd: Int, R: Int, PAGE: Int
](job: PagedJob, seq: Int, kvh: Int, h0: Int, nrows: Int):
    """Query heads `h0 .. h0+nrows-1` (nrows <= R, all of kv head `kvh`)."""
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    comptime KT = _kt[hd]()
    comptime NV = hd // LANES
    comptime DEG = 2 if dtype == DType.bfloat16 else 6
    comptime DOT = dtype == DType.bfloat16 and BF16_DOT
    var page = PAGE if PAGE else job.page
    var len = Int(ptr[Int32](job.lens).unsafe_load(seq))
    var bt = ptr[Int32](job.block_table + seq * job.bt_stride * 4)
    var head_bytes = page * hd * E  # one (page, kv head) block
    var c = job.scale * Float32(LOG2E)

    # Queries: fp32 copies (fp32 path, scale folded into exp2) or broadcast bf16
    # dim pairs (dot path).
    var qf = stack_allocation[R * hd, DType.float32, alignment=64]()
    var qp = stack_allocation[R * hd // 2, DType.int32, alignment=64]()
    for r in range(R):
        var rr = min(r, nrows - 1)
        var src = job.q + (seq * job.q_stride + (h0 + rr) * hd) * E
        comptime if DOT:
            var s = ptr[Int32](src)
            comptime for d2 in range(hd // 2):
                qp.unsafe_store(r * (hd // 2) + d2, s.unsafe_load(d2))
        else:
            var s = ptr[Scalar[dtype]](src)
            comptime for d in range(0, hd, LANES):
                qf.unsafe_store[width=LANES](
                    r * hd + d, s.unsafe_load[width=LANES](d).cast[DType.float32]()
                )

    var logits = stack_allocation[R * KT, DType.float32, alignment=64]()
    var acc = stack_allocation[R * hd, DType.float32, alignment=64]()
    for i in range(0, R * hd, LANES):
        acc.unsafe_store[width=LANES](i, SIMD[DType.float32, LANES](0))
    var m = SIMD[DType.float32, R](-Float32.MAX)
    var l = SIMD[DType.float32, R](0)

    var k0 = 0
    while k0 < len:
        var nk = min(KT, len - k0)
        # ---- Q@K^T. A small register tile scores G key groups per step so the
        # dot chains are independent (one bf16 chain over head_dim is latency
        # bound: ~6 cycles per `vdpbf16ps`); groups resolve their own pages.
        comptime G = 1 if dtype == DType.float32 else (4 if R == 1 else (2 if R == 2 else 1))
        var j = 0
        while j + G * LANES <= nk:
            _qk_groups[dtype, hd, R, G, KT, PAGE](
                job, bt, kvh, head_bytes, k0 + j, j, qf, qp, logits, LANES
            )
            j += G * LANES
        while j < nk:
            _qk_groups[dtype, hd, R, 1, KT, PAGE](
                job, bt, kvh, head_bytes, k0 + j, j, qf, qp, logits, nk - j
            )
            j += LANES

        # ---- online softmax per row; p = 2^(logit*c - m), m in scaled log2 units
        var alpha = SIMD[DType.float32, R](1)
        comptime for r in range(R):
            var mx = SIMD[DType.float32, LANES](-Float32.MAX)
            var t = 0
            while t < nk:
                mx = max(mx, logits.unsafe_load[width=LANES](r * KT + t))
                t += LANES
            var mn = max(m[r], mx.reduce_max() * c)
            alpha[r] = exp2s[DEG](m[r] - mn)
            m[r] = mn
            var sum = SIMD[DType.float32, LANES](0)
            t = 0
            while t < nk:
                var e = exp2[DEG, LANES](
                    logits.unsafe_load[width=LANES](r * KT + t).fma(
                        SIMD[DType.float32, LANES](c), SIMD[DType.float32, LANES](-mn)
                    )
                )
                logits.unsafe_store[width=LANES](r * KT + t, e)
                sum += e
                t += LANES
            l[r] = l[r] * alpha[r] + sum.reduce_add()

        # ---- P@V over exactly nk keys (never past `len`: 0 * NaN is NaN)
        _pv[dtype, hd, R, KT, PAGE](job, bt, kvh, head_bytes, k0, nk, logits, acc, alpha)
        k0 += nk

    # ---- epilogue: a row that saw no key is exactly 0
    for r in range(nrows):
        var inv = Float32(1) / l[r] if l[r] > 0 else Float32(0)
        var dst = job.out + (seq * job.out_stride + (h0 + r) * hd) * E
        var o = ptr[Scalar[dtype]](dst)
        comptime if L.SPLIT:
            comptime for d in range(0, hd, 2 * LANES):
                var ev = acc.unsafe_load[width=LANES](r * hd + d) * inv
                var od = acc.unsafe_load[width=LANES](r * hd + d + LANES) * inv
                o.unsafe_store[width = 2 * LANES](d, ev.interleave(od).cast[dtype]())
        else:
            comptime for d in range(0, hd, LANES):
                o.unsafe_store[width=LANES](
                    d, (acc.unsafe_load[width=LANES](r * hd + d) * inv).cast[dtype]()
                )



@always_inline
def _qk_groups[
    dtype: DType, hd: Int, R: Int, G: Int, KT: Int, PAGE: Int
](
    job: PagedJob,
    bt: MutPointer[Int32, _],
    kvh: Int,
    head_bytes: Int,
    key0: Int,
    j: Int,
    qf: MutPointer[Float32, _],
    qp: MutPointer[Int32, _],
    logits: MutPointer[Float32, _],
    valid: Int,
):
    """logits[r][j + g*LANES ..] for G groups of LANES keys starting at `key0`;
    lanes at or past `valid` (only with G == 1) are masked to -max."""
    comptime L = PageLayout[dtype, hd]
    comptime DOT = dtype == DType.bfloat16 and BF16_DOT
    comptime V = SIMD[DType.float32, LANES]
    var page = PAGE if PAGE else job.page
    var base = Array[Int, G](fill=0)
    comptime for g in range(G):
        var key = key0 + g * LANES
        var pg = key // page
        var blk = (Int(bt.unsafe_load(pg)) * job.kv_heads + kvh) * head_bytes
        base[g] = job.k_cache + blk + L.k_elem(page, key - pg * page, 0) * (
            2 if dtype == DType.bfloat16 else 4
        )
    var cv = Array[V, R * G](fill=V(0))
    comptime if DOT:
        # A runtime loop over head-dim pairs (unrolled by U): fully unrolled,
        # LLVM hoists every query broadcast out of the key loop and spills
        # them, so each dot pays a stack reload (30% of samples, measured).
        comptime U = 4
        for d0 in range(0, hd // 2, U):
            comptime for u in range(U):
                var d2 = d0 + u
                comptime for g in range(G):
                    var kv = ptr[BFloat16](base[g]).unsafe_load[width = 2 * LANES](
                        d2 * page * 2
                    )
                    comptime for r in range(R):
                        var qv = SIMD[DType.int32, LANES](
                            qp.unsafe_load(r * (hd // 2) + d2)
                        )
                        cv[r * G + g] = dot_bf16[LANES](
                            cv[r * G + g], bitcast_bf16[LANES](qv), kv
                        )
    else:
        # CH independent partial dot products per (row, group): one chain over
        # all of head_dim 128 rounds past the 1e-6 fp32 gate (1.2-1.6e-6
        # measured), and the chains also overlap the K loads.
        comptime CH = max(1, min(4, 16 // (R * G)))
        var part = Array[V, R * G * CH](fill=V(0))
        comptime U = max(CH, 4)
        for d0 in range(0, hd, U):
            comptime for u in range(U):
                var d = d0 + u
                comptime for g in range(G):
                    var kv = ptr[Scalar[dtype]](base[g]).unsafe_load[width=LANES](
                        d * page
                    ).cast[DType.float32]()
                    comptime for r in range(R):
                        comptime i = (r * G + g) * CH + u % CH
                        part[i] = V(qf.unsafe_load(r * hd + d)).fma(kv, part[i])
        comptime for rg in range(R * G):
            var sum = part[rg * CH]
            comptime for c in range(1, CH):
                sum += part[rg * CH + c]
            cv[rg] = sum
    comptime for g in range(G):
        comptime for r in range(R):
            var v = cv[r * G + g]
            comptime if G == 1:
                if valid < LANES:
                    var lane = iota[DType.int32, LANES]()
                    var keep = lane.lt(SIMD[DType.int32, LANES](Int32(valid)))
                    v = keep.select(v, V(-Float32.MAX))
            logits.unsafe_store[width=LANES](r * KT + j + g * LANES, v)


@always_inline
def bitcast_bf16[width: Int](x: SIMD[DType.int32, width]) -> SIMD[DType.bfloat16, 2 * width]:
    from std.memory import bitcast

    return bitcast[DType.bfloat16, 2 * width](x)


@always_inline
def _pv[
    dtype: DType, hd: Int, R: Int, KT: Int, PAGE: Int
](
    job: PagedJob,
    bt: MutPointer[Int32, _],
    kvh: Int,
    head_bytes: Int,
    k0: Int,
    nk: Int,
    p: MutPointer[Float32, _],
    acc: MutPointer[Float32, _],
    alpha: SIMD[DType.float32, R],
):
    """acc[r] = acc[r]*alpha[r] + sum_j p[r][j] * V[k0+j], dims in register
    chunks of DC so R rows x DC dims stay inside the vector registers."""
    comptime L = PageLayout[dtype, hd]
    comptime V = SIMD[DType.float32, LANES]
    comptime SPLIT = L.SPLIT
    comptime DC = hd if R * hd // LANES <= 16 else max(
        2 * LANES if SPLIT else LANES, 16 * LANES // R
    )
    comptime NC = DC // LANES
    # Two key-parity lanes while the tile is small: the even and odd keys feed
    # independent FMA chains (one chain per dim vector is latency-bound at R=1).
    comptime LN = 2 if R * NC <= 8 else 1
    var page = PAGE if PAGE else job.page
    comptime for dc in range(0, hd, DC):
        var a = Array[V, LN * R * NC](fill=V(0))
        comptime for r in range(R):
            comptime for i in range(NC):
                a[r * NC + i] = acc.unsafe_load[width=LANES](r * hd + dc + i * LANES) * alpha[r]
        var done = 0
        while done < nk:
            var key = k0 + done
            var pg = key // page
            var off = key - pg * page
            var chunk = min(page - off, nk - done)
            var blk = (Int(bt.unsafe_load(pg)) * job.kv_heads + kvh) * head_bytes
            var vp = ptr[Scalar[dtype]](job.v_cache + blk)

            @always_inline
            def vrow(t: Int, i: Int) capturing -> V:
                comptime if SPLIT:
                    # Dims are kept even/odd-split per 2*LANES block: one
                    # 64-byte load is LANES words; `<< 16` is the even dims
                    # and `& 0xffff0000` the odd ones, 2 instructions per 32
                    # values against 4 for zero-extend + shift (the widen was
                    # 28% of a bf16 decode). Each dim's fp32 math is unchanged.
                    var w = bitcast[DType.int32, LANES](
                        vp.unsafe_load[width = 2 * LANES](
                            L.v_elem(page, off + t, dc + (i // 2) * 2 * LANES)
                        )
                    )
                    if i % 2 == 0:
                        return bitcast[DType.float32, LANES](w << 16)
                    return bitcast[DType.float32, LANES](w & Int32(-65536))
                elif L.V_PAIRS:
                    var pair = vp.unsafe_load[width = 2 * LANES](
                        L.v_elem(page, (off + t) & ~1, dc + i * LANES)
                    ).cast[DType.float32]().deinterleave()
                    return rebind[V](pair[0]) if ((off + t) & 1) == 0 else rebind[V](pair[1])
                else:
                    return vp.unsafe_load[width=LANES](
                        L.v_elem(page, off + t, dc + i * LANES)
                    ).cast[DType.float32]()

            var t = 0
            comptime if LN == 2:
                while t + 1 < chunk:
                    comptime for i in range(NC):
                        var v0 = vrow(t, i)
                        var v1 = vrow(t + 1, i)
                        comptime for r in range(R):
                            var pr = p + r * KT + done + t
                            a[r * NC + i] = V(pr[0]).fma(v0, a[r * NC + i])
                            a[(R + r) * NC + i] = V(pr[1]).fma(v1, a[(R + r) * NC + i])
                    t += 2
            while t < chunk:
                comptime for i in range(NC):
                    var v0 = vrow(t, i)
                    comptime for r in range(R):
                        a[r * NC + i] = V(p.unsafe_load(r * KT + done + t)).fma(
                            v0, a[r * NC + i]
                        )
                t += 1
            done += chunk
        comptime for r in range(R):
            comptime for i in range(NC):
                var s = a[r * NC + i]
                comptime if LN == 2:
                    s += a[(R + r) * NC + i]
                acc.unsafe_store[width=LANES](r * hd + dc + i * LANES, s)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
comptime RMAX = 8  # query heads per register tile


@always_inline
def _rows_tile[dtype: DType, hd: Int, PAGE: Int](job: PagedJob, seq: Int, kvh: Int, h0: Int, n: Int):
    # A partial tile instantiates a smaller R instead of repeating rows.
    var h = h0
    var left = n
    while left > 0:
        if left >= 8 and hd * 8 // LANES <= 32:
            _attend_rows[dtype, hd, 8, PAGE](job, seq, kvh, h, 8)
            h += 8
            left -= 8
        elif left >= 4:
            _attend_rows[dtype, hd, 4, PAGE](job, seq, kvh, h, 4)
            h += 4
            left -= 4
        elif left >= 2:
            _attend_rows[dtype, hd, 2, PAGE](job, seq, kvh, h, 2)
            h += 2
            left -= 2
        else:
            _attend_rows[dtype, hd, 1, PAGE](job, seq, kvh, h, 1)
            h += 1
            left -= 1


def _body[dtype: DType, hd: Int, PAGE: Int](ctx: Int, tid: Int, threads: Int):
    var job = ptr[PagedJob](ctx)[]
    var group = job.num_heads // job.kv_heads
    var pairs = job.num_seqs * job.kv_heads
    if job.do_store and job.chunks > 1:
        # Split groups: every store of the call precedes every read.
        var s0 = tid * pairs // threads
        var s1 = (tid + 1) * pairs // threads
        for i in range(s0, s1):
            _store_kv[dtype, hd](job, i // job.kv_heads, i % job.kv_heads)
        barrier()
    if not job.do_attend:
        if job.chunks == 1 and job.do_store:
            var s0 = tid * pairs // threads
            var s1 = (tid + 1) * pairs // threads
            for i in range(s0, s1):
                _store_kv[dtype, hd](job, i // job.kv_heads, i % job.kv_heads)
        return
    var tasks = pairs * job.chunks
    var per = (group + job.chunks - 1) // job.chunks
    var t0 = tid * tasks // threads
    var t1 = (tid + 1) * tasks // threads
    for t in range(t0, t1):
        var pair = t // job.chunks
        var ch = t - pair * job.chunks
        var seq = pair // job.kv_heads
        var kvh = pair - seq * job.kv_heads
        if job.do_store and job.chunks == 1:
            # The whole group is this task's: store, then read it back.
            _store_kv[dtype, hd](job, seq, kvh)
        var h0 = kvh * group + ch * per
        var n = min(per, group - ch * per)
        if n > 0:
            _rows_tile[dtype, hd, PAGE](job, seq, kvh, h0, n)


def _run[dtype: DType, hd: Int](job: PagedJob):
    var j = job
    var st = stack_allocation[1, PagedJob]()
    st[] = j
    # The engine's page sizes (32 on x86, 16 on aarch64) compile to constant K
    # strides; with a runtime page LLVM keeps one address per head-dim pair on
    # the stack and every K load pays a second load for it.
    if j.page == 32:
        comptime body = _body[dtype, hd, 32]
        parallel(body, Int(st), j.workers)
    elif j.page == 16:
        comptime body = _body[dtype, hd, 16]
        parallel(body, Int(st), j.workers)
    else:
        comptime body = _body[dtype, hd, 0]
        parallel(body, Int(st), j.workers)


def paged_run(job: PagedJob) raises:
    """Store and/or attend. `job.chunks` and `job.workers` come from the plan."""
    if job.num_seqs == 0:
        return
    if job.dtype == 0:
        _dispatch_hd[DType.float32](job)
    elif job.dtype == 1:
        _dispatch_hd[DType.bfloat16](job)
    else:
        raise Error("cpu_attention: unsupported dtype")


def _dispatch_hd[dtype: DType](job: PagedJob) raises:
    if job.page % 16 != 0 or job.page <= 0:
        raise Error("cpu_attention: page size must be a positive multiple of 16")
    if job.head_dim == 64:
        _run[dtype, 64](job)
    elif job.head_dim == 128:
        _run[dtype, 128](job)
    elif job.head_dim == 32:
        _run[dtype, 32](job)
    else:
        raise Error("cpu_attention: head_dim must be 32, 64 or 128")
