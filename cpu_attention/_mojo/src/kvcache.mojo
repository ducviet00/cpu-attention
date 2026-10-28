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

from isa import LANES, LOG2E, NEON_BF16, dot_bf16, exp2, exp2s, ptr, vmax
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
        var kdt = ptr[Scalar[Self.dtype]](k_page + t * 4)
        var vd = ptr[Scalar[Self.dtype]](v_page)
        comptime if Self.BF and not V_PAIRS_ARM:
            for d0 in range(0, Self.hd // 2, 4):
                comptime for u in range(4):
                    var d2 = d0 + u
                    kdt.unsafe_store[width=2](
                        d2 * page * 2, ks.unsafe_load[width=2](2 * d2)
                    )
        elif Self.BF:
            comptime for d2 in range(Self.hd // 2):
                kdt.unsafe_store[width=2](
                    d2 * page * 2, ks.unsafe_load[width=2](2 * d2)
                )
        else:
            for d0 in range(0, Self.hd, 4):
                comptime for u in range(4):
                    var d = d0 + u
                    kdt.unsafe_store(d * page, ks.unsafe_load(d))
        comptime if Self.V_PAIRS:
            var vdt = ptr[Scalar[Self.dtype]](
                v_page + (((t >> 1) * Self.hd) * 2 + (t & 1)) * 2
            )
            comptime for d in range(Self.hd):
                vdt.unsafe_store(d * 2, vs.unsafe_load(d))
        else:
            vd.unsafe_store[width=Self.hd](
                t * Self.hd, vs.unsafe_load[width=Self.hd]()
            )


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
@always_inline
def _div_page[PAGE: Int](x: Int, page: Int, out res: Tuple[Int, Int]):
    comptime if PAGE == 16:
        return (x >> 4, x & 15)
    elif PAGE == 32:
        return (x >> 5, x & 31)
    elif PAGE == 64:
        return (x >> 6, x & 63)
    else:
        var q = x // page
        return (q, x - q * page)


@always_inline
def _store_kv[dtype: DType, hd: Int, PAGE: Int](job: PagedJob, seq: Int, kvh: Int):
    var page = PAGE if PAGE else job.page
    var pg: Int
    var off: Int
    if job.slots != 0:
        var slot = Int(ptr[Int64](job.slots).unsafe_load(seq))
        if slot < 0:  # padding: no token to store (C++ semantics)
            return
        var d = _div_page[PAGE](slot, page)
        pg = d[0]
        off = d[1]
    else:
        var before = Int(ptr[Int32](job.lens).unsafe_load(seq))
        if before < 0:
            return
        var d = _div_page[PAGE](before, page)
        pg = Int(ptr[Int32](job.block_table + seq * job.bt_stride * 4).unsafe_load(d[0]))
        off = d[1]
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var base = ((pg * job.kv_heads + kvh) * page * hd) * E
    var src = (seq * job.kv_stride + kvh * hd) * E
    L.store_token(
        job.k_cache + base, job.v_cache + base, page, off,
        job.k + src, job.v + src,
    )


# ---------------------------------------------------------------------------
# attend: R query rows of one kv head against one sequence's pages
# ---------------------------------------------------------------------------
@always_inline
def _kt[hd: Int, R: Int]() -> Int:
    # Keys per softmax step. At hd <= 64, 256 keys (1 KB) covers ctx=256 in a
    # single pass; at hd == 128 and R == 1 on x86, 512 keys (2 KB) matches C++
    # vec::Paged; larger R or NEON keeps R x KT inside L1 next to the K stream.
    return 256 if hd <= 64 else (512 if (LANES == 16 and R == 1) else 128)


def _attend_rows[
    dtype: DType, hd: Int, R: Int, PAGE: Int
](job: PagedJob, seq: Int, kvh: Int, h0: Int, nrows: Int):
    """Query heads `h0 .. h0+nrows-1` (nrows <= R, all of kv head `kvh`)."""
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    comptime KT = _kt[hd, R]()
    comptime NV = hd // LANES
    # bf16 on x86: degree 2, inside the bf16 output rounding, the C++ class the
    # x86 gate compares against (2 ulp). aarch64's gate is also <= 1e-3 against
    # the fp32 reference, which degree 2 (relative 1.7e-3) misses by 4x.
    comptime DEG = (2 if not V_PAIRS_ARM else 5) if dtype == DType.bfloat16 else 6
    # bf16 K is dim pairs on every tier; `dot_bf16` widens where there is no
    # dot instruction.
    comptime DOT = dtype == DType.bfloat16
    var page = PAGE if PAGE else job.page
    var len = Int(ptr[Int32](job.lens).unsafe_load(seq)) + (
        1 if (job.do_store and job.slots == 0) else 0
    )
    var bt = ptr[Int32](job.block_table + seq * job.bt_stride * 4)
    var head_bytes = page * hd * E  # one (page, kv head) block
    var c = job.scale * Float32(LOG2E)

    # Queries: at R == 1 read directly from `job.q`; for R > 1 pack R rows into
    # a contiguous stack buffer.
    var qf_buf = stack_allocation[
        (R * hd) if (R > 1 and not DOT) else 1, DType.float32, alignment=64
    ]()
    var qp_buf = stack_allocation[
        (R * hd // 2) if (R > 1 and DOT) else 1, DType.int32, alignment=64
    ]()
    var q0_addr = job.q + (seq * job.q_stride + h0 * hd) * E
    var qf = ptr[Float32](q0_addr) if R == 1 else qf_buf
    var qp = ptr[Int32](q0_addr) if R == 1 else qp_buf
    comptime if R > 1:
        for r in range(R):
            var rr = min(r, nrows - 1)
            var src = job.q + (seq * job.q_stride + (h0 + rr) * hd) * E
            comptime if DOT:
                var s = ptr[Int32](src)
                comptime for d2 in range(hd // 2):
                    qp_buf.unsafe_store(r * (hd // 2) + d2, s.unsafe_load(d2))
            else:
                var s = ptr[Scalar[dtype]](src)
                comptime for d in range(0, hd, LANES):
                    qf_buf.unsafe_store[width=LANES](
                        r * hd + d, s.unsafe_load[width=LANES](d).cast[DType.float32]()
                    )

    var logits = stack_allocation[R * KT, DType.float32, alignment=64]()
    var acc = stack_allocation[R * hd, DType.float32, alignment=64]()
    var m = SIMD[DType.float32, R](-Float32.MAX)
    var l = SIMD[DType.float32, R](0)

    var k0 = 0
    while k0 < len:
        var nk = min(KT, len - k0)
        # ---- Q@K^T. A small register tile scores G key groups per step so the
        # dot chains are independent (one bf16 chain over head_dim is latency
        # bound: ~6 cycles per `vdpbf16ps`); page pointers step incrementally.
        comptime G = (
            (4 if R == 1 else (2 if R == 2 else 1))
            if (dtype == DType.float32 and V_PAIRS_ARM)
            else (
                1
                if dtype == DType.float32
                else (
                    (8 if V_PAIRS_ARM else 4)
                    if R == 1
                    else (2 if R == 2 else 1)
                )
            )
        )
        var d0 = _div_page[PAGE](k0, page)
        var pg = d0[0]
        var off = d0[1]
        var kpage = job.k_cache + (Int(bt.unsafe_load(pg)) * job.kv_heads + kvh) * head_bytes
        var j = 0
        while j + G * LANES <= nk:
            var base = Array[Int, G](uninitialized=True)
            comptime for g in range(G):
                if off >= page:
                    pg += 1
                    off = 0
                    kpage = job.k_cache + (
                        Int(bt.unsafe_load(pg)) * job.kv_heads + kvh
                    ) * head_bytes
                base[g] = kpage + off * 4
                off += LANES
            _qk_groups[dtype, hd, R, G, KT, PAGE, False](
                page, base, j, qf, qp, logits, LANES
            )
            j += G * LANES
        while j < nk:
            if off >= page:
                pg += 1
                off = 0
                kpage = job.k_cache + (
                    Int(bt.unsafe_load(pg)) * job.kv_heads + kvh
                ) * head_bytes
            var base1 = Array[Int, 1](fill=kpage + off * 4)
            off += LANES
            _qk_groups[dtype, hd, R, 1, KT, PAGE, True](
                page, base1, j, qf, qp, logits, nk - j
            )
            j += LANES

        # ---- online softmax per row; p = 2^(logit*c - m), m in scaled log2 units
        var alpha = SIMD[DType.float32, R](1)
        comptime for r in range(R):
            var mx = SIMD[DType.float32, LANES](-Float32.MAX)
            var t = 0
            while t < nk:
                mx = vmax(mx, logits.unsafe_load[width=LANES](r * KT + t))
                t += LANES
            var mx_r = mx.reduce_max() * c
            var mn: Float32
            if k0 == 0:
                mn = mx_r
                alpha[r] = 0.0
                m[r] = mn
            elif mx_r > m[r]:
                mn = mx_r
                alpha[r] = exp2s[DEG](m[r] - mn)
                m[r] = mn
            else:
                mn = m[r]
                alpha[r] = 1.0
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
            l[r] = (l[r] * alpha[r] if k0 > 0 else Float32(0)) + sum.reduce_add()

        # ---- P@V over exactly nk keys (never past `len`: 0 * NaN is NaN)
        _pv[dtype, hd, R, KT, PAGE](
            job, bt, kvh, head_bytes, k0, nk, len, logits, acc, alpha
        )
        k0 += nk

    # ---- epilogue: a row that saw no key is exactly 0
    for r in range(nrows):
        var dst = job.out + (seq * job.out_stride + (h0 + r) * hd) * E
        var o = ptr[Scalar[dtype]](dst)
        if l[r] <= 0:
            comptime for d in range(0, hd, LANES):
                o.unsafe_store[width=LANES](d, SIMD[dtype, LANES](0))
            continue
        var inv = Float32(1) / l[r]
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
    dtype: DType, hd: Int, R: Int, G: Int, KT: Int, PAGE: Int, MASK: Bool
](
    page: Int,
    base: Array[Int, G],
    j: Int,
    qf: MutPointer[Float32, _],
    qp: MutPointer[Int32, _],
    logits: MutPointer[Float32, _],
    valid: Int,
):
    """logits[r][j + g*LANES ..] for G groups of LANES keys;
    lanes at or past `valid` (only when MASK is True) are masked to -max."""
    comptime DOT = dtype == DType.bfloat16
    comptime V = SIMD[DType.float32, LANES]
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
                        # Query last: `vdpbf16ps` folds a broadcast load
                        # only as its last source (exact, symmetric products).
                        cv[r * G + g] = dot_bf16[LANES](
                            cv[r * G + g], kv, bitcast_bf16[LANES](qv)
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
            comptime if CH == 4:
                cv[rg] = (part[rg * 4] + part[rg * 4 + 1]) + (
                    part[rg * 4 + 2] + part[rg * 4 + 3]
                )
            else:
                var sum = part[rg * CH]
                comptime for c in range(1, CH):
                    sum += part[rg * CH + c]
                cv[rg] = sum
    comptime for g in range(G):
        comptime for r in range(R):
            var v = cv[r * G + g]
            comptime if MASK:
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
    len: Int,
    p: MutPointer[Float32, _],
    acc: MutPointer[Float32, _],
    alpha: SIMD[DType.float32, R],
):
    """acc[r] = acc[r]*alpha[r] + sum_j p[r][j] * V[k0+j], dims in register
    chunks of DC so R rows x DC dims stay inside the vector registers."""
    comptime if dtype == DType.bfloat16 and NEON_BF16:
        _pv_arm[hd, R, KT, PAGE](
            job, bt, kvh, head_bytes, k0, nk, len, p, acc, alpha
        )
        return
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
    var d0 = _div_page[PAGE](k0, page)
    comptime for dc in range(0, hd, DC):
        var a = Array[V, LN * R * NC](fill=V(0))
        if k0 > 0:
            comptime for r in range(R):
                var ar = alpha[r]
                if ar == 1.0:
                    comptime for i in range(NC):
                        a[r * NC + i] = acc.unsafe_load[width=LANES](
                            r * hd + dc + i * LANES
                        )
                elif ar != 0.0:
                    var va = V(ar)
                    comptime for i in range(NC):
                        a[r * NC + i] = (
                            acc.unsafe_load[width=LANES](r * hd + dc + i * LANES)
                            * va
                        )
        var done = 0
        var pg = d0[0]
        var off = d0[1]
        while done < nk:
            var chunk = min(page - off, nk - done)
            var blk = (Int(bt.unsafe_load(pg)) * job.kv_heads + kvh) * head_bytes
            var vp = ptr[Scalar[dtype]](job.v_cache + blk)
            var vp_off = vp + off * hd + dc

            @always_inline
            def vrow(t: Int, i: Int) capturing -> V:
                comptime if SPLIT:
                    # Dims are kept even/odd-split per 2*LANES block: one
                    # 64-byte load is LANES words; `<< 16` is the even dims
                    # and `& 0xffff0000` the odd ones, 2 instructions per 32
                    # values against 4 for zero-extend + shift (the widen was
                    # 28% of a bf16 decode). Each dim's fp32 math is unchanged.
                    var w = bitcast[DType.int32, LANES](
                        (vp_off + t * hd).unsafe_load[width = 2 * LANES](
                            (i // 2) * 2 * LANES
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
                    return (vp_off + t * hd).unsafe_load[width=LANES](
                        i * LANES
                    ).cast[DType.float32]()

            var t = 0
            comptime if LN == 2:
                while t + 1 < chunk:
                    var p0 = Array[V, R](uninitialized=True)
                    var p1 = Array[V, R](uninitialized=True)
                    comptime for r in range(R):
                        var base_p = r * KT + done + t
                        p0[r] = V(p.unsafe_load(base_p))
                        p1[r] = V(p.unsafe_load(base_p + 1))
                    comptime for i in range(NC):
                        var v0 = vrow(t, i)
                        var v1 = vrow(t + 1, i)
                        comptime for r in range(R):
                            a[r * NC + i] = p0[r].fma(v0, a[r * NC + i])
                            a[(R + r) * NC + i] = p1[r].fma(
                                v1, a[(R + r) * NC + i]
                            )
                    t += 2
            while t < chunk:
                var p0 = Array[V, R](uninitialized=True)
                comptime for r in range(R):
                    p0[r] = V(p.unsafe_load(r * KT + done + t))
                comptime for i in range(NC):
                    var v0 = vrow(t, i)
                    comptime for r in range(R):
                        a[r * NC + i] = p0[r].fma(v0, a[r * NC + i])
                t += 1
            done += chunk
            off = 0
            pg += 1
        comptime for r in range(R):
            comptime for i in range(NC):
                var s = a[r * NC + i]
                comptime if LN == 2:
                    s += a[(R + r) * NC + i]
                acc.unsafe_store[width=LANES](r * hd + dc + i * LANES, s)


@always_inline
def _pv_arm[hd: Int, R: Int, KT: Int, PAGE: Int](
    job: PagedJob,
    bt: MutPointer[Int32, _],
    kvh: Int,
    head_bytes: Int,
    k0: Int,
    nk: Int,
    len: Int,
    p: MutPointer[Float32, _],
    acc: MutPointer[Float32, _],
    alpha: SIMD[DType.float32, R],
):
    # Axion perf: widening/deinterleaving V and fp32 P@V take 70% of the MHA
    # kernel. The existing paired layout is already the bfdot operand.
    comptime V = SIMD[DType.float32, 4]
    comptime L = PageLayout[DType.bfloat16, hd]
    comptime DC = min(hd, max(4, 64 // R))
    comptime NC = DC // 4
    var page = PAGE if PAGE else job.page
    var d0 = _div_page[PAGE](k0, page)
    comptime for dc in range(0, hd, DC):
        var a = Array[V, R * NC](fill=V(0))
        if k0 > 0:
            comptime for r in range(R):
                var ar = alpha[r]
                if ar == 1.0:
                    comptime for i in range(NC):
                        a[r * NC + i] = acc.unsafe_load[width=4](
                            r * hd + dc + i * 4
                        )
                elif ar != 0.0:
                    var va = V(ar)
                    comptime for i in range(NC):
                        a[r * NC + i] = (
                            acc.unsafe_load[width=4](r * hd + dc + i * 4) * va
                        )
        var done = 0
        var pg = d0[0]
        var off = d0[1]
        while done < nk:
            var chunk = min(page - off, nk - done)
            var blk = (Int(bt.unsafe_load(pg)) * job.kv_heads + kvh) * head_bytes
            var vp = ptr[BFloat16](job.v_cache + blk)
            var t = 0
            if len >= 128:
                while t + 1 < chunk:
                    var probs = Array[SIMD[DType.bfloat16, 8], R](
                        uninitialized=True
                    )
                    comptime for r in range(R):
                        var pair = p.unsafe_load[width=2](
                            r * KT + done + t
                        ).cast[DType.bfloat16]()
                        var twice = pair.join(pair)
                        probs[r] = twice.join(twice)
                    comptime for i in range(NC):
                        var v = vp.unsafe_load[width=8](
                            L.v_elem(page, off + t, dc + i * 4)
                        )
                        comptime for r in range(R):
                            a[r * NC + i] = dot_bf16[4](
                                a[r * NC + i], v, probs[r]
                            )
                    t += 2
            else:
                # Short contexts (< 128 keys, e.g. L=17/33 in heterogeneous
                # batches) can have dominant single-key weights where single-bf16
                # P exceeds the 1e-3 excess gate against the scalar fp32 peer;
                # adding the low bf16 residual retains fp32 P accuracy.
                while t + 1 < chunk:
                    var probs = Array[SIMD[DType.bfloat16, 8], R](
                        uninitialized=True
                    )
                    var residuals = Array[SIMD[DType.bfloat16, 8], R](
                        uninitialized=True
                    )
                    comptime for r in range(R):
                        var values = p.unsafe_load[width=2](r * KT + done + t)
                        var pair = values.cast[DType.bfloat16]()
                        var twice = pair.join(pair)
                        probs[r] = twice.join(twice)
                        var low = (values - pair.cast[DType.float32]()).cast[
                            DType.bfloat16
                        ]()
                        var low_twice = low.join(low)
                        residuals[r] = low_twice.join(low_twice)
                    comptime for i in range(NC):
                        var v = vp.unsafe_load[width=8](
                            L.v_elem(page, off + t, dc + i * 4)
                        )
                        comptime for r in range(R):
                            a[r * NC + i] = dot_bf16[4](
                                a[r * NC + i], v, probs[r]
                            )
                            a[r * NC + i] = dot_bf16[4](
                                a[r * NC + i], v, residuals[r]
                            )
                    t += 2
            if t < chunk:
                # Do not dot a padded key: zero probability times garbage NaN
                # is still NaN. Only the valid half of the pair participates.
                comptime for i in range(NC):
                    var pair = vp.unsafe_load[width=8](L.v_elem(page, off + t, dc + i * 4))
                    var v = pair.shuffle[0, 2, 4, 6, 0, 2, 4, 6]().slice[4]().cast[DType.float32]()
                    comptime for r in range(R):
                        var prob = p.unsafe_load(r * KT + done + t)
                        a[r * NC + i] = V(prob).fma(v, a[r * NC + i])
            done += chunk
            off = 0
            pg += 1
        comptime for r in range(R):
            comptime for i in range(NC):
                acc.unsafe_store[width=4](r * hd + dc + i * 4, a[r * NC + i])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
comptime RMAX = 8  # query heads per register tile


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
    var kvh_n = job.kv_heads
    var pairs = job.num_seqs * kvh_n
    if not job.do_attend:
        if job.do_store:
            var s0 = tid * pairs // threads
            var s1 = (tid + 1) * pairs // threads
            if s0 < s1:
                var seq = s0 // kvh_n
                var kvh = s0 - seq * kvh_n
                for _ in range(s0, s1):
                    _store_kv[dtype, hd, PAGE](job, seq, kvh)
                    kvh += 1
                    if kvh == kvh_n:
                        kvh = 0
                        seq += 1
        return
    if job.num_heads == kvh_n:
        # MHA fast path: 1 query head per kv head, chunks == 1, no barrier.
        var t0 = tid * pairs // threads
        var t1 = (tid + 1) * pairs // threads
        if t0 < t1:
            var seq = t0 // kvh_n
            var kvh = t0 - seq * kvh_n
            for _ in range(t0, t1):
                if job.do_store:
                    _store_kv[dtype, hd, PAGE](job, seq, kvh)
                _attend_rows[dtype, hd, 1, PAGE](job, seq, kvh, kvh, 1)
                kvh += 1
                if kvh == kvh_n:
                    kvh = 0
                    seq += 1
        return
    var group = job.num_heads // kvh_n
    if job.do_store and job.chunks > 1:
        # Split groups: every store of the call precedes every read.
        var s0 = tid * pairs // threads
        var s1 = (tid + 1) * pairs // threads
        if s0 < s1:
            var seq = s0 // kvh_n
            var kvh = s0 - seq * kvh_n
            for _ in range(s0, s1):
                _store_kv[dtype, hd, PAGE](job, seq, kvh)
                kvh += 1
                if kvh == kvh_n:
                    kvh = 0
                    seq += 1
        barrier()
    var tasks = pairs * job.chunks
    var per = (group + job.chunks - 1) // job.chunks
    var t0 = tid * tasks // threads
    var t1 = (tid + 1) * tasks // threads
    for t in range(t0, t1):
        var pair = t // job.chunks
        var ch = t - pair * job.chunks
        var seq = pair // kvh_n
        var kvh = pair - seq * kvh_n
        if job.do_store and job.chunks == 1:
            # The whole group is this task's: store, then read it back.
            _store_kv[dtype, hd, PAGE](job, seq, kvh)
        var h0 = kvh * group + ch * per
        var n = min(per, group - ch * per)
        if n > 0:
            _rows_tile[dtype, hd, PAGE](job, seq, kvh, h0, n)


def _run[dtype: DType, hd: Int](job: PagedJob):
    var j = job
    var st = stack_allocation[1, PagedJob]()
    st[] = j
    # Compile-time page sizes (16, 32, 48, 64) give constant K strides and
    # shift/mask page indexing; anything else falls back to runtime `job.page`.
    if j.page == 32:
        comptime body = _body[dtype, hd, 32]
        parallel(body, Int(st), j.workers)
    elif j.page == 16:
        comptime body = _body[dtype, hd, 16]
        parallel(body, Int(st), j.workers)
    elif j.page == 64:
        comptime body = _body[dtype, hd, 64]
        parallel(body, Int(st), j.workers)
    elif j.page == 48:
        comptime body = _body[dtype, hd, 48]
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
