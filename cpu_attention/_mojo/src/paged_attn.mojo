"""KV-cache attention: `flash_attn_with_kvcache` decode and `reshape_and_cache`.

Contract: one query token per sequence, attending every cached position (the
query is the last position, so causality is the cache length). The cache is
`[num_blocks, heads, page, head_dim]` with this backend's in-page layout
(`PageLayout`); `reshape_and_cache` writes it and nothing else reads it.

MHA: work item = (sequence, head), one query row against its own head's K/V.
Workers take contiguous ranges of items balanced on context length (`_share`).
"""

from std.math import iota
from std.memory import bitcast, stack_allocation
from std.sys.info import CompilationTarget

from isa import BCAST_FOLD, LANES, LOG2E, bf16_lo, dot_bf16, exp2, online_max, ptr, transpose, vmax
from parallel import parallel

comptime V_PAIRS_ARM = CompilationTarget.has_neon()
"""aarch64 stores bf16 V as key pairs (`[page/2][hd][2]`) so `bfdot` can contract
two keys; x86 keeps `[page][hd]` (fp32 P against widened V rows)."""


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
    var heads: Int
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
        self.heads = 0
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


# ---------------------------------------------------------------------------
# page layout
# ---------------------------------------------------------------------------
struct PageLayout[dtype: DType, hd: Int]:
    """Where token `t` of a (page, head) lives. K is dim-major so one vector
    load is LANES consecutive tokens of one dimension; bf16 pairs the two
    dimensions a dot instruction contracts: `[hd/2][page][2]`. V is `[page][hd]`,
    bf16 on aarch64 key pairs `[page/2][hd][2]` (the `dot_bf16` operand)."""

    comptime BF = Self.dtype == DType.bfloat16
    comptime V_PAIRS = Self.BF and V_PAIRS_ARM
    comptime SPLIT = Self.BF and not Self.V_PAIRS and Self.hd % (2 * LANES) == 0
    """bf16 V is widened even/odd-split, so the fp32 accumulator holds each
    2*LANES block of dims as [even dims][odd dims] (see `_pv`)."""

    @staticmethod
    @always_inline
    def store_token(k_page: Int, v_page: Int, page: Int, t: Int, k_src: Int, v_src: Int):
        """Scatter one token's K and V (one head) into its page."""
        var ks = ptr[Scalar[Self.dtype]](k_src)
        var kdt = ptr[Scalar[Self.dtype]](k_page + t * 4)
        comptime if Self.BF and not V_PAIRS_ARM:
            for d0 in range(0, Self.hd // 2, 4):
                comptime for u in range(4):
                    var d2 = d0 + u
                    kdt.unsafe_store[width=2](
                        d2 * page * 2, ks.unsafe_load[width=2](2 * d2)
                    )
        elif Self.BF:
            var ksi = ptr[Int32](k_src)
            var kdti = ptr[Int32](k_page + t * 4)
            comptime for d0 in range(0, Self.hd // 2, 4):
                var v4 = ksi.unsafe_load[width=4](d0)
                comptime for u in range(4):
                    kdti.unsafe_store((d0 + u) * page, v4[u])
        else:
            comptime for d0 in range(0, Self.hd, 4):
                var v4 = ks.unsafe_load[width=4](d0)
                comptime for u in range(4):
                    kdt.unsafe_store((d0 + u) * page, v4[u])
        Self.store_v(v_page, page, t, v_src)

    @staticmethod
    @always_inline
    def store_v(v_page: Int, page: Int, t: Int, v_src: Int):
        """One token's V (one head) into its page."""
        var vs = ptr[Scalar[Self.dtype]](v_src)
        var vd = ptr[Scalar[Self.dtype]](v_page)
        comptime if Self.V_PAIRS:
            var vdt = ptr[Scalar[Self.dtype]](
                v_page + (((t >> 1) * Self.hd) * 2 + (t & 1)) * 2
            )
            comptime for d0 in range(0, Self.hd, 8):
                var v8 = vs.unsafe_load[width=8](d0)
                comptime for u in range(8):
                    vdt.unsafe_store((d0 + u) * 2, v8[u])
        else:
            vd.unsafe_store[width=Self.hd](
                t * Self.hd, vs.unsafe_load[width=Self.hd]()
            )

    @staticmethod
    @always_inline
    def store_k_block(k_page: Int, page: Int, t0: Int, k_src: Int, src_stride: Int):
        """K of LANES consecutive tokens from `t0` (one head, one page) through
        a LANES x LANES transpose of 32-bit K words (an fp32 dim, or a bf16 dim
        pair): one vector store per word row, where per-token stores are LANES
        scalar stores each (store-bound: fp32 store-only ran 1.14x slower that way on the C4D)."""
        comptime W = Self.hd * (2 if Self.BF else 4) // 4
        comptime assert W % LANES == 0, "K words tile by LANES"
        comptime for w0 in range(0, W, LANES):
            var r = Array[SIMD[DType.int32, LANES], LANES](uninitialized=True)
            comptime for i in range(LANES):
                r[i] = ptr[Int32](k_src + i * src_stride).unsafe_load[width=LANES](w0)
            transpose[DType.int32](r)
            comptime for j in range(LANES):
                ptr[Int32](k_page + ((w0 + j) * page + t0) * 4).unsafe_store[width=LANES](0, r[j])


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
    elif PAGE:
        var q = x // PAGE  # a constant divisor: multiply and shift
        return (q, x - q * PAGE)
    else:
        var q = x // page
        return (q, x - q * page)


@always_inline
def _store_kv[dtype: DType, hd: Int, PAGE: Int](job: PagedJob, seq: Int, h: Int):
    var page = PAGE if PAGE else job.page
    var pg: Int
    var off: Int
    if job.slots != 0:
        var slot = Int(ptr[Int64](job.slots).unsafe_load(seq))
        if slot < 0:  # padding: no token to store (vLLM semantics)
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
    var base = ((pg * job.heads + h) * page * hd) * E
    var src = (seq * job.kv_stride + h * hd) * E
    L.store_token(
        job.k_cache + base, job.v_cache + base, page, off,
        job.k + src, job.v + src,
    )


@always_inline
def _store_block[dtype: DType, hd: Int, PAGE: Int](job: PagedJob, h: Int, s0: Int, n: Int):
    """Store-only: tokens s0 .. s0+n-1 of head `h`. LANES tokens in
    consecutive slots of one page take the transposed K store; anything else
    (padding, a page boundary, scattered slots) stores token by token."""
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var page = PAGE if PAGE else job.page
    if n == LANES and job.slots != 0:
        var sl = ptr[Int64](job.slots)
        var first = Int(sl.unsafe_load(s0))
        var d = _div_page[PAGE](first, page)
        var run = first >= 0 and d[1] + LANES <= page
        var i = 1
        while run and i < LANES:  # scattered slots (one token per sequence) stop at once
            run = Int(sl.unsafe_load(s0 + i)) == first + i
            i += 1
        if run:
            var base = ((d[0] * job.heads + h) * page * hd) * E
            var src = (s0 * job.kv_stride + h * hd) * E
            L.store_k_block(job.k_cache + base, page, d[1], job.k + src, job.kv_stride * E)
            for i in range(LANES):
                L.store_v(job.v_cache + base, page, d[1] + i, job.v + src + i * job.kv_stride * E)
            return
    for i in range(n):
        _store_kv[dtype, hd, PAGE](job, s0 + i, h)


# ---------------------------------------------------------------------------
# attend: one query row against one sequence's pages of its head
# ---------------------------------------------------------------------------
# One driver for every tier. A softmax step of KT keys is resolved into 16-key
# units (page % 16 == 0 is checked at dispatch, so a unit never straddles a
# page): one key vector on AVX-512, two on AVX2, four on NEON. Per target only
# compile-time constants (`Decode`) and the P@V operand differ:
#   x86      fp32 P against V rows (bf16 V widened);
#   aarch64  bf16 P against bf16 V key pairs with `dot_bf16` (`bfdot`): widening
#            V was 70% of the kernel on Axion (docs/Mojo.md §9).


def _kt(hd: Int) -> Int:
    """Keys per softmax step. aarch64: 64 (the four unit bases of a step stay in
    registers). x86: at hd <= 64, 256 keys (1 KB) covers ctx=256 in a single
    pass; at hd == 128, 512 keys (2 KB) on 16 lanes, 128 on 8."""
    if V_PAIRS_ARM:
        return 64
    return 256 if hd <= 64 else (512 if LANES == 16 else 128)


def _groups(bf: Bool) -> Int:
    """Key vectors scored per Q@K^T call: independent dot chains (one bf16 chain
    over head_dim is latency bound, ~6 cycles per `vdpbf16ps`) that fit the
    registers. aarch64 bf16 scores a whole 64-key step in one sweep."""
    if bf:
        return 16 if V_PAIRS_ARM else 4
    return 4 if V_PAIRS_ARM else 1


def _nc(hd: Int) -> Int:
    """P@V dim vectors per register chunk (at most 16 accumulators)."""
    return min(hd // LANES, 16)


struct Decode[dtype: DType, hd: Int]:
    """The decode kernel's compile-time shape on this tier."""

    comptime BF = Self.dtype == DType.bfloat16
    comptime E = 2 if Self.BF else 4
    comptime UNIT = 16
    comptime GPU = Self.UNIT // LANES
    """Key vectors per unit."""
    comptime KT = _kt(Self.hd)
    comptime UNITS = Self.KT // Self.UNIT
    """Units per step; a step spans at most UNITS + 1 pages (pages are >= 16 keys)."""
    comptime G = _groups(Self.BF)
    comptime G_TAIL = min(Self.G, Self.GPU)
    """A masked tail call never leaves its unit, so it never reads past the step's pages."""
    comptime PAIRS = PageLayout[Self.dtype, Self.hd].V_PAIRS
    # bf16: degree 5. The gate is <= 1e-3 against the fp32 reference beyond the
    # bf16 output rounding; degree 2 (relative 1.7e-3) missed it by 4x (x86:
    # 3.9e-3 worst, 91 of 252 sweep cases; degree 5: 7e-7) for +1.2% instructions.
    comptime DEG = 5 if Self.BF else 6
    comptime NS = 2 if V_PAIRS_ARM else 1
    """Partial row sums in the softmax (independent adds on 4-lane NEON)."""
    comptime FULL = V_PAIRS_ARM
    """A full step gets its own P@V body with constant unit lengths."""
    comptime NC = _nc(Self.hd)
    comptime PV: PVKernel = PVPairs[Self.hd] if Self.PAIRS else PVRows[Self.dtype, Self.hd]
    """The P@V kernel: the only per-target code in the driver."""


@always_inline
def _rescale[DEG: Int](k0: Int, mx: Float32, m: Float32) -> Tuple[Float32, Float32]:
    """Online softmax step: the new running max and the factor for the old sum
    and accumulator (0 on the first step, which also discards their garbage)."""
    if k0 == 0:
        return (mx, Float32(0))
    var s = online_max[1, DEG](m, mx)
    return (s[0][0], s[1][0])


@always_inline
def _acc_load[
    NC: Int, W: Int, N: Int
](mut a: Array[SIMD[DType.float32, W], N], acc: MutPointer[Float32, _], dc: Int, k0: Int, alpha: Float32):
    """a[i] = acc[dc + i*W ..] * alpha; `a` is zero-filled, which stands for the
    first step and for alpha == 0."""
    if k0 > 0:
        if alpha == 1.0:
            comptime for i in range(NC):
                a[i] = acc.unsafe_load[width=W](dc + i * W)
        elif alpha != 0.0:
            var va = SIMD[DType.float32, W](alpha)
            comptime for i in range(NC):
                a[i] = acc.unsafe_load[width=W](dc + i * W) * va


@always_inline
def _store_row[dtype: DType, hd: Int](job: PagedJob, seq: Int, h: Int, l: Float32, acc: MutPointer[Float32, _]):
    """out = acc / l; a row that saw no key is exactly 0.

    Runtime loops on purpose: with only constant offsets into the accumulator
    buffer, LLVM promotes it to vector registers live across the key loop and
    spills the Q@K^T accumulators (aarch64 bf16 d=128: 155 stack accesses in that
    loop against 47, decode 4-14% slower)."""
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
    var o = ptr[Scalar[dtype]](job.out + (seq * job.out_stride + h * hd) * E)
    if l <= 0:
        comptime for d in range(0, hd, LANES):
            o.unsafe_store[width=LANES](d, SIMD[dtype, LANES](0))
        return
    var inv = Float32(1) / l
    comptime if L.SPLIT:
        for d in range(0, hd, 2 * LANES):
            var ev = acc.unsafe_load[width=LANES](d) * inv
            var od = acc.unsafe_load[width=LANES](d + LANES) * inv
            o.unsafe_store[width = 2 * LANES](d, ev.interleave(od).cast[dtype]())
    else:
        for d in range(0, hd, LANES):
            o.unsafe_store[width=LANES](d, (acc.unsafe_load[width=LANES](d) * inv).cast[dtype]())


@always_inline
def bitcast_bf16[width: Int](x: SIMD[DType.int32, width]) -> SIMD[DType.bfloat16, 2 * width]:
    return bitcast[DType.bfloat16, 2 * width](x)


@always_inline
def _page_bases[E: Int, hd: Int](
    job: PagedJob,
    bt: MutPointer[Int32, _],
    h: Int,
    page: Int,
    p0: Int,
    npg: Int,
    kpg: MutPointer[Int64, _],
    vpg: MutPointer[Int64, _],
):
    """K and V block addresses of a step's npg pages from page p0: one
    block-table lookup per page (per 16-key unit cost 5-11% more instructions
    per call, measured on Zen 4)."""
    var head_bytes = page * hd * E
    for i in range(npg):
        var blk = (Int(bt.unsafe_load(p0 + i)) * job.heads + h) * head_bytes
        kpg.unsafe_store(i, Int64(job.k_cache + blk))
        vpg.unsafe_store(i, Int64(job.v_cache + blk))


@always_inline
def _key_addr[PAGE: Int, BYTES: Int](pg: MutPointer[Int64, _], page: Int, x: Int) -> Int:
    """Address of key x of a step, counted from its first page's start."""
    var d = _div_page[PAGE](x, page)
    return Int(pg.unsafe_load(d[0])) + d[1] * BYTES


@always_inline
def _qk[
    dtype: DType, hd: Int, G: Int, PAGE: Int, MASK: Bool
](
    kpg: MutPointer[Int64, _],
    page: Int,
    o0: Int,
    q: Int,
    j: Int,
    nk: Int,
    logits: MutPointer[Float32, _],
    mut mx: SIMD[DType.float32, LANES],
):
    """logits[j + g*LANES ..] for G key vectors from key j, and the running lane
    max; with MASK, lanes at or past nk are -max and vectors past nk are neither
    scored into the max nor stored."""
    comptime V = SIMD[DType.float32, LANES]
    # A call of whole units (j is then a multiple of 16) addresses each unit's
    # base plus compile-time offsets for its key vectors: 16 runtime group
    # pointers cost NEON an address add per load. A call inside one unit
    # (G < GPU) starts mid-unit.
    comptime GPU = 16 // LANES
    comptime GU = GPU if G >= GPU else G  # key vectors per unit in this call
    comptime UC = G // GU  # units in this call
    var kb = Array[Int, UC](uninitialized=True)
    comptime if G >= GPU:
        comptime for u in range(UC):
            kb[u] = _key_addr[PAGE, 4](kpg, page, o0 + j + 16 * u)
    else:
        kb[0] = _key_addr[PAGE, 4](kpg, page, o0 + j)
    var cv = Array[V, G](fill=V(0))
    comptime if dtype == DType.bfloat16:
        # bf16 K and Q are dim pairs. A runtime loop over them (unrolled by U):
        # fully unrolled, LLVM hoists every query broadcast out of the key loop
        # and spills them (30% of samples on x86, measured).
        var qp = ptr[Int32](q)
        comptime U = 4
        for d0 in range(0, hd // 2, U):
            comptime for u in range(U):
                var d2 = d0 + u
                comptime for g in range(G):
                    var k = ptr[BFloat16](kb[g // GU] + (g % GU) * LANES * 4).unsafe_load[width = 2 * LANES](d2 * page * 2)
                    # Query last: `vdpbf16ps` folds a broadcast load only as its
                    # last source (exact, symmetric products).
                    var qv = SIMD[DType.int32, LANES](qp.unsafe_load(d2))
                    cv[g] = dot_bf16[LANES](cv[g], k, bitcast_bf16[LANES](qv))
    else:
        # CH independent partial dot products per vector: one chain over all of
        # head_dim 128 rounds past the 1e-6 fp32 gate (1.2-1.6e-6 measured), and
        # the chains also overlap the K loads.
        var qf = ptr[Float32](q)
        comptime CH = max(1, min(4, 16 // G))
        var part = Array[V, G * CH](fill=V(0))
        comptime U = max(CH, 4)
        for d0 in range(0, hd, U):
            # Where a broadcast does not fold into the FMA (NEON), U query dims
            # as one vector, then a lane each (by-element FMA).
            var qv = SIMD[DType.float32, U]()
            comptime if not BCAST_FOLD:
                qv = qf.unsafe_load[width=U](d0)
            comptime for u in range(U):
                var d = d0 + u
                comptime for g in range(G):
                    var k = ptr[Float32](kb[g // GU] + (g % GU) * LANES * 4).unsafe_load[width=LANES](d * page)
                    comptime i = g * CH + u % CH
                    var qs: Float32
                    comptime if BCAST_FOLD:
                        qs = qf.unsafe_load(d)
                    else:
                        qs = qv[u]
                    part[i] = V(qs).fma(k, part[i])
        comptime for g in range(G):
            comptime if CH == 4:
                cv[g] = (part[g * 4] + part[g * 4 + 1]) + (part[g * 4 + 2] + part[g * 4 + 3])
            else:
                var sum = part[g * CH]
                comptime for c in range(1, CH):
                    sum += part[g * CH + c]
                cv[g] = sum
    comptime assert G & (G - 1) == 0 and G <= 32, "G is a power of two"
    var t = Array[V, G](uninitialized=True)
    comptime for g in range(G):
        var key = j + g * LANES
        var v = cv[g]
        comptime if MASK:
            if key + LANES > nk:
                var lane = iota[DType.int32, LANES]() + Int32(key)
                v = lane.lt(SIMD[DType.int32, LANES](Int32(nk))).select(v, V(-Float32.MAX))
            if key < nk:
                logits.unsafe_store[width=LANES](key, v)
        else:
            logits.unsafe_store[width=LANES](key, v)
        t[g] = v
    # The call's max as a tree: a chain of G `vmax` after a 16-vector sweep sits
    # on the softmax's critical path.
    comptime for lvl in range(5):
        comptime s = 16 >> lvl
        comptime if s < G:
            comptime for i in range(s):
                t[i] = vmax[LANES](t[i], t[i + s])
    mx = vmax[LANES](mx, t[0])


trait PVKernel:
    """How a step keeps its probabilities and contracts them with V: the one
    per-target piece of the decode driver (`Decode.PV`), as `InnerMatmulKernel`
    is for Modular's matmul."""

    comptime P16: Bool
    """Probabilities are also kept as bf16 (`pbuf`, and `pbuf_lo` when LO)."""
    comptime LN: Int
    """Key-parity accumulator sets per dim vector."""

    @staticmethod
    def store_p[LO: Bool, W: Int](e: SIMD[DType.float32, W], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        """Keep the probabilities `e` of keys i .. i+W-1 (`odd`: the step has an
        odd number of keys)."""
        ...

    @staticmethod
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        """a += P[kf : kf+kv] @ V for the dims from `dc`, over a run of kv keys
        contiguous from address `vb` (FULL: one 16-key unit)."""
        ...


struct PVRows[dtype: DType, hd: Int](PVKernel):
    """fp32 P against V rows `[key][hd]` (bf16 V widened): x86."""

    comptime P16 = False
    comptime NC = _nc(Self.hd)
    comptime LN = 2 if Self.NC <= 8 else 1
    """Even and odd keys feed independent FMA chains while the tile is small (one
    chain per dim vector is latency bound)."""

    @staticmethod
    @always_inline
    def store_p[LO: Bool, W: Int](e: SIMD[DType.float32, W], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        ptr[Float32](logits).unsafe_store[width=W](i, e)

    @staticmethod
    @always_inline
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        comptime V = SIMD[DType.float32, LANES]
        comptime NC = Self.NC
        comptime SPLIT = PageLayout[Self.dtype, Self.hd].SPLIT
        var vp = ptr[Scalar[Self.dtype]](vb + dc * (2 if Self.dtype == DType.bfloat16 else 4))
        var pk = ptr[Float32](p) + kf

        @always_inline
        def vrow(t: Int, i: Int) capturing -> V:
            comptime if SPLIT:
                # Dims are kept even/odd-split per 2*LANES block: one 64-byte
                # load is LANES words; `<< 16` is the even dims and
                # `& 0xffff0000` the odd ones, 2 instructions per 32 values
                # against 4 for zero-extend + shift (the widen was 28% of a bf16
                # decode). Each dim's fp32 math is unchanged.
                var w = bitcast[DType.int32, LANES]((vp + t * Self.hd).unsafe_load[width = 2 * LANES]((i // 2) * 2 * LANES))
                if i % 2 == 0:
                    return bitcast[DType.float32, LANES](w << 16)
                return bitcast[DType.float32, LANES](w & Int32(-65536))
            else:
                return (vp + t * Self.hd).unsafe_load[width=LANES](i * LANES).cast[DType.float32]()

        var t = 0
        comptime if not BCAST_FOLD:
            # 4 probabilities as one vector, a lane each (NEON's by-element FMA;
            # a broadcast load per key costs a load there).
            while t + 3 < kv:
                var pv = pk.unsafe_load[width=4](t)
                comptime for k in range(4):
                    comptime for i in range(NC):
                        comptime s = (k % Self.LN) * NC + i
                        a[s] = V(pv[k]).fma(vrow(t + k, i), a[s])
                t += 4
        comptime if Self.LN == 2:

            @always_inline
            def pair(t: Int) capturing:
                var p0 = V(pk.unsafe_load(t))
                var p1 = V(pk.unsafe_load(t + 1))
                comptime for i in range(NC):
                    var v0 = vrow(t, i)
                    var v1 = vrow(t + 1, i)
                    a[i] = p0.fma(v0, a[i])
                    a[NC + i] = p1.fma(v1, a[NC + i])

            # Two pairs per iteration: the loop's own adds and compares were 11%
            # of a cache-resident bf16 decode (Zen 4 profile).
            while t + 3 < kv:
                pair(t)
                pair(t + 2)
                t += 4
            if t + 1 < kv:
                pair(t)
                t += 2
        while t < kv:
            var p0 = V(pk.unsafe_load(t))
            comptime for i in range(NC):
                a[i] = p0.fma(vrow(t, i), a[i])
            t += 1


struct PVPairs[hd: Int](PVKernel):
    """bf16 P against bf16 V key pairs `[key/2][hd][2]` with `dot_bf16`
    (`bfdot`): one load is LANES dims of a key pair, contracted against the
    pair's two probabilities. aarch64, where widening V was 70% of the kernel."""

    comptime P16 = True
    comptime NC = _nc(Self.hd)
    comptime LN = 1

    @staticmethod
    @always_inline
    def store_p[LO: Bool, W: Int](e: SIMD[DType.float32, W], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        var hi = e.cast[DType.bfloat16]()
        ptr[BFloat16](pbuf).unsafe_store[width=W](i, hi)
        comptime if LO:
            ptr[BFloat16](pbuf_lo).unsafe_store[width=W](i, bf16_lo(e, hi))
        if odd:  # the unpaired last key is an fp32 FMA
            ptr[Float32](logits).unsafe_store[width=W](i, e)

    @staticmethod
    @always_inline
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        comptime V = SIMD[DType.float32, LANES]
        comptime B = SIMD[DType.bfloat16, 2 * LANES]
        comptime NC = Self.NC
        var vbg = vb + dc * 4
        var w0 = kf >> 1
        var pw = ptr[Int32](pbuf)  # one bf16 key pair per word
        var plw = ptr[Int32](pbuf_lo)

        @always_inline
        def prob(w: MutPointer[Int32, _], t: Int) capturing -> B:
            return bitcast_bf16[LANES](SIMD[DType.int32, LANES](w.unsafe_load(w0 + (t >> 1))))

        var t = 0
        if len >= 128:
            # Two key pairs per step.
            while t + 3 < kv:
                var p0 = prob(pw, t)
                var p1 = prob(pw, t + 2)
                var vp0 = ptr[BFloat16](vbg + t * Self.hd * 2)
                var vp1 = ptr[BFloat16](vbg + (t + 2) * Self.hd * 2)
                comptime for i in range(NC):
                    a[i] = dot_bf16[LANES](a[i], vp0.unsafe_load[width = 2 * LANES](i * 2 * LANES), p0)
                comptime for i in range(NC):
                    a[i] = dot_bf16[LANES](a[i], vp1.unsafe_load[width = 2 * LANES](i * 2 * LANES), p1)
                t += 4
            while t + 1 < kv:
                var pr = prob(pw, t)
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    a[i] = dot_bf16[LANES](a[i], vp.unsafe_load[width = 2 * LANES](i * 2 * LANES), pr)
                t += 2
        else:
            # Short contexts can have one dominant key, where single-bf16 P
            # misses the 1e-3 gate: add P's low bf16 residual.
            while t + 1 < kv:
                var pr = prob(pw, t)
                var lo = prob(plw, t)
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    var v = vp.unsafe_load[width = 2 * LANES](i * 2 * LANES)
                    a[i] = dot_bf16[LANES](a[i], v, pr)
                    a[i] = dot_bf16[LANES](a[i], v, lo)
                t += 2
        comptime if not FULL:
            if t < kv:
                # The unpaired last key: only its half of the pair (0 * a padded
                # NaN is still NaN).
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    var v = rebind[V](vp.unsafe_load[width = 2 * LANES](i * 2 * LANES).cast[DType.float32]().deinterleave()[0])
                    a[i] = V(ptr[Float32](p).unsafe_load(kf + t)).fma(v, a[i])


@always_inline
def _exp_row[
    dtype: DType, hd: Int, LO: Bool
](logits: Int, pbuf: Int, pbuf_lo: Int, nk: Int, c: Float32, mn: Float32) -> Float32:
    """p = 2^(logit*c - mn) for a step, kept by `Decode.PV`; returns the sum (NS
    partial sums)."""
    comptime D = Decode[dtype, hd]
    comptime V = SIMD[DType.float32, LANES]
    var lg = ptr[Float32](logits)
    var odd = (nk & 1) == 1
    var s = Array[V, D.NS](fill=V(0))

    @always_inline
    def ex(t: Int) capturing -> V:
        return exp2[D.DEG, LANES](lg.unsafe_load[width=LANES](t).fma(V(c), V(-mn)))

    var t = 0
    comptime if D.NS == 2:
        # Two vectors per step, kept with one store (16-byte bf16 stores on NEON).
        while t + 2 * LANES <= nk:
            var e0 = ex(t)
            var e1 = ex(t + LANES)
            D.PV.store_p[LO, 2 * LANES](e0.join(e1), logits, pbuf, pbuf_lo, t, odd)
            s[0] += e0
            s[1] += e1
            t += 2 * LANES
    while t < nk:
        var e = ex(t)
        D.PV.store_p[LO, LANES](e, logits, pbuf, pbuf_lo, t, odd)
        s[0] += e
        t += LANES
    var sum = s[0]
    comptime for u in range(1, D.NS):
        sum += s[u]
    return sum.reduce_add()


@always_inline
def _pv[
    dtype: DType, hd: Int, PAGE: Int, FULL: Bool
](
    k0: Int, nk: Int, len: Int, o0: Int, npg: Int, page: Int, alpha: Float32,
    vpg: MutPointer[Int64, _], pbuf: Int, pbuf_lo: Int, p: Int, acc: MutPointer[Float32, _],
):
    """acc = acc*alpha + sum_j P[j] * V[k0+j] over exactly nk keys (never past
    `len`: 0 * a padded NaN is NaN), NC dim vectors at a time, page by page
    (FULL: 16-key unit by unit, a constant length)."""
    comptime D = Decode[dtype, hd]
    comptime V = SIMD[DType.float32, LANES]
    comptime NC = D.NC
    comptime LN = D.PV.LN
    comptime for dc in range(0, hd, NC * LANES):
        var a = Array[V, LN * NC](fill=V(0))
        _acc_load[NC, LANES, LN * NC](a, acc, dc, k0, alpha)
        comptime if FULL:
            for g in range(D.UNITS):
                var vb = _key_addr[PAGE, hd * D.E](vpg, page, o0 + g * D.UNIT)
                D.PV.keys[LN * NC, True](a, vb, dc, p, pbuf, pbuf_lo, g * D.UNIT, D.UNIT, len)
        else:
            var pgs = PAGE if PAGE else page
            for i in range(npg):
                var ks = max(0, i * pgs - o0)
                var ke = min(nk, (i + 1) * pgs - o0)
                var vb = Int(vpg.unsafe_load(i)) + (ks + o0 - i * pgs) * hd * D.E
                D.PV.keys[LN * NC, False](a, vb, dc, p, pbuf, pbuf_lo, ks, ke - ks, len)
        comptime for i in range(NC):
            var s = a[i]
            comptime if LN == 2:
                s += a[NC + i]
            acc.unsafe_store[width=LANES](dc + i * LANES, s)


def _attend[dtype: DType, hd: Int, PAGE: Int](job: PagedJob, seq: Int, h: Int):
    """Query head `h` of sequence `seq` against its cached K/V."""
    comptime D = Decode[dtype, hd]
    comptime KT = D.KT
    comptime assert D.UNIT % LANES == 0 and KT % D.UNIT == 0, "a step is whole units of whole vectors"
    comptime assert D.NS == 1 or D.NS == 2, "the softmax keeps one or two partial sums"
    comptime assert hd % (D.NC * LANES) == 0, "P@V chunks tile head_dim"
    comptime V = SIMD[DType.float32, LANES]
    var page = PAGE if PAGE else job.page
    var len = Int(ptr[Int32](job.lens).unsafe_load(seq)) + (
        1 if (job.do_store and job.slots == 0) else 0
    )
    var bt = ptr[Int32](job.block_table + seq * job.bt_stride * 4)
    var c = job.scale * Float32(LOG2E)
    var q = job.q + (seq * job.q_stride + h * hd) * D.E  # read in place

    var kpg = stack_allocation[D.UNITS + 1, DType.int64, alignment=64]()
    var vpg = stack_allocation[D.UNITS + 1, DType.int64, alignment=64]()
    var logits = stack_allocation[KT, DType.float32, alignment=64]()
    var pbuf = stack_allocation[KT if D.PV.P16 else 1, DType.bfloat16, alignment=64]()
    var pbuf_lo = stack_allocation[KT if D.PV.P16 else 1, DType.bfloat16, alignment=64]()
    var acc = stack_allocation[hd, DType.float32, alignment=64]()
    var m = -Float32.MAX
    var l = Float32(0)

    var k0 = 0
    while k0 < len:
        var nk = min(KT, len - k0)
        # The step's pages: it starts o0 keys into page p0 (o0 % 16 == 0).
        var d0 = _div_page[PAGE](k0, page)
        var o0 = d0[1]
        var npg = _div_page[PAGE](o0 + nk - 1, page)[0] + 1
        _page_bases[D.E, hd](job, bt, h, page, d0[0], npg, kpg, vpg)

        # ---- Q@K^T: G key vectors per call; the tail in masked calls of at
        # most one unit, and a single vector when that is all that is left.
        var mx = V(-Float32.MAX)
        var j = 0
        while j + D.G * LANES <= nk:
            _qk[dtype, hd, D.G, PAGE, False](kpg, page, o0, q, j, nk, logits, mx)
            j += D.G * LANES
        while j < nk:
            if nk - j <= LANES:
                _qk[dtype, hd, 1, PAGE, True](kpg, page, o0, q, j, nk, logits, mx)
                break
            _qk[dtype, hd, D.G_TAIL, PAGE, True](kpg, page, o0, q, j, nk, logits, mx)
            j += D.G_TAIL * LANES

        # ---- online softmax; p = 2^(logit*c - m), m in scaled log2 units
        var s = _rescale[D.DEG](k0, mx.reduce_max() * c, m)
        m = s[0]
        var alpha = s[1]
        var sum: Float32
        if D.PV.P16 and len < 128:
            sum = _exp_row[dtype, hd, True](Int(logits), Int(pbuf), Int(pbuf_lo), nk, c, s[0])
        else:
            sum = _exp_row[dtype, hd, False](Int(logits), Int(pbuf), Int(pbuf_lo), nk, c, s[0])
        l = (l * alpha if k0 > 0 else Float32(0)) + sum

        # ---- P@V
        comptime if D.FULL:
            if nk == KT:
                _pv[dtype, hd, PAGE, True](k0, nk, len, o0, npg, page, alpha, vpg, Int(pbuf), Int(pbuf_lo), Int(logits), acc)
                k0 += nk
                continue
        _pv[dtype, hd, PAGE, False](k0, nk, len, o0, npg, page, alpha, vpg, Int(pbuf), Int(pbuf_lo), Int(logits), acc)
        k0 += nk

    _store_row[dtype, hd](job, seq, h, l, acc)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _share(job: PagedJob, tid: Int, threads: Int) -> Tuple[Int, Int]:
    """Worker `tid`'s (sequence, head) items `[t0, t1)` in sequence-major order:
    contiguous, and balanced on cost (an item's context plus one 16-key unit of
    fixed work), not on count. Equal counts left a mixed-length batch's critical
    path on whichever worker held its longest sequence (Axion, 8 sequences of
    17-2047 keys x 16 heads on 8 threads: 3.3x the mean, 2.7x slower than vLLM)."""
    var lens = ptr[Int32](job.lens)
    var per_seq = job.heads
    var tasks = job.num_seqs * per_seq
    var total = 0
    for s in range(job.num_seqs):
        total += Int(lens.unsafe_load(s)) + 16
    # Worker tid starts at the first item whose preceding cost reaches
    # tid / threads of the total (everything scaled by threads: exact integers).
    var lo = tid * total * per_seq
    var hi = lo + total * per_seq
    var t0 = tasks
    var t1 = tasks
    var cum = 0
    for s in range(job.num_seqs):
        var c = (Int(lens.unsafe_load(s)) + 16) * threads
        var last = cum + c * (per_seq - 1)  # cost before this sequence's last item
        if t0 == tasks and lo <= last:
            t0 = s * per_seq + max(0, (lo - cum + c - 1) // c)
        if hi <= last:
            t1 = s * per_seq + max(0, (hi - cum + c - 1) // c)
            break
        cum += c * per_seq
    return (t0, t1)


def _body[dtype: DType, hd: Int, PAGE: Int](ctx: Int, tid: Int, threads: Int):
    var job = ptr[PagedJob](ctx)[]
    var heads = job.heads
    if not job.do_attend:
        if job.do_store:
            # Prefill-style staging (the first LANES slots consecutive) runs in
            # (head, LANES-token block) tasks, so a block is one thread's;
            # scattered appends keep (token, head) order: their source rows are
            # read contiguously (blocks cost b=32 ctx=1 ~0.4 us).
            var blocks = job.slots != 0 and job.num_seqs >= LANES
            if blocks:
                var sl = ptr[Int64](job.slots)
                for i in range(1, LANES):
                    blocks = blocks and Int(sl.unsafe_load(i)) == Int(sl.unsafe_load(0)) + i
            if blocks:
                var nblk = (job.num_seqs + LANES - 1) // LANES
                var tasks = heads * nblk
                for t in range(tid * tasks // threads, (tid + 1) * tasks // threads):
                    var h = t // nblk
                    var a = (t - h * nblk) * LANES
                    _store_block[dtype, hd, PAGE](job, h, a, min(LANES, job.num_seqs - a))
            else:
                var pairs = job.num_seqs * heads
                var s0 = tid * pairs // threads
                var seq = s0 // heads
                var h = s0 - seq * heads
                for _ in range(s0, (tid + 1) * pairs // threads):
                    _store_kv[dtype, hd, PAGE](job, seq, h)
                    h += 1
                    if h == heads:
                        h = 0
                        seq += 1
        return
    var t0, t1 = _share(job, tid, threads)
    if t0 < t1:
        var seq = t0 // heads
        var h = t0 - seq * heads
        for _ in range(t0, t1):
            if job.do_store:
                _store_kv[dtype, hd, PAGE](job, seq, h)
            _attend[dtype, hd, PAGE](job, seq, h)
            h += 1
            if h == heads:
                h = 0
                seq += 1


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
    """Store and/or attend. `job.workers` comes from the plan."""
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
