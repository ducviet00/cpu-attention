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

from isa import LANES, LOG2E, dot_bf16, exp2, exp2s, ptr, vmax
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
# One driver for every tier. A softmax step of KT keys is resolved into 16-key
# units (page % 16 == 0 is checked at dispatch, so a unit never straddles a
# page): one key vector on AVX-512, two on AVX2, four on NEON. Per target only
# compile-time constants (`Decode`) and the P@V operand differ:
#   x86      fp32 P against V rows (bf16 V widened): the C++ class the 2-ulp x86
#            bf16 gate compares against;
#   aarch64  bf16 P against bf16 V key pairs with `dot_bf16` (`bfdot`): widening
#            V was 70% of the kernel on Axion (docs/Mojo.md §9).


def _kt(hd: Int, R: Int) -> Int:
    """Keys per softmax step. aarch64: 64 (the four unit bases of a step stay in
    registers). x86: at hd <= 64, 256 keys (1 KB) covers ctx=256 in a single
    pass; at hd == 128 and R == 1 on 16 lanes, 512 keys (2 KB) matches C++
    vec::Paged; larger R keeps R x KT inside L1 next to the K stream."""
    if V_PAIRS_ARM:
        return 64
    return 256 if hd <= 64 else (512 if (LANES == 16 and R == 1) else 128)


def _groups(bf: Bool, R: Int) -> Int:
    """Key vectors scored per Q@K^T call: independent dot chains (one bf16 chain
    over head_dim is latency bound, ~6 cycles per `vdpbf16ps`) that fit the
    registers. aarch64 bf16 R == 1 scores a whole 64-key step in one sweep."""
    if bf:
        if V_PAIRS_ARM:
            return 16 if R == 1 else (4 if R == 2 else 2)
        return 4 if R == 1 else (2 if R == 2 else 1)
    return 4 if (V_PAIRS_ARM and R == 1) else 1


struct Decode[dtype: DType, hd: Int, R: Int]:
    """The decode kernel's compile-time shape on this tier."""

    comptime BF = Self.dtype == DType.bfloat16
    comptime E = 2 if Self.BF else 4
    comptime UNIT = 16
    comptime GPU = Self.UNIT // LANES
    """Key vectors per unit."""
    comptime KT = _kt(Self.hd, Self.R)
    comptime UNITS = Self.KT // Self.UNIT
    comptime G = _groups(Self.BF, Self.R)
    comptime G_TAIL = min(Self.G, Self.GPU)
    """A masked tail call never leaves its unit (only resolved units are read)."""
    comptime PAIRS = PageLayout[Self.dtype, Self.hd].V_PAIRS
    # bf16 on x86: degree 2, inside the bf16 output rounding (the C++ class of
    # the x86 gate). aarch64's gate is also <= 1e-3 against the fp32 reference,
    # which degree 2 (relative 1.7e-3) misses by 4x.
    comptime DEG = (5 if V_PAIRS_ARM else 2) if Self.BF else 6
    comptime NS = 2 if V_PAIRS_ARM else 1
    """Partial row sums in the softmax (independent adds on 4-lane NEON)."""
    comptime FULL = V_PAIRS_ARM and Self.R == 1
    """A full R == 1 step gets its own P@V body with constant unit lengths."""
    comptime NC = _nc(Self.dtype, Self.hd, Self.R)
    comptime PV: PVKernel = PVPairs[Self.hd, Self.R] if Self.PAIRS else PVRows[Self.dtype, Self.hd, Self.R]
    """The P@V kernel: the only per-target code in the driver."""


@always_inline
def _rescale[DEG: Int](k0: Int, mx: Float32, m: Float32) -> Tuple[Float32, Float32]:
    """Online softmax step: the new running max and the factor for the old sum
    and accumulator (0 on the first step, which also discards their garbage)."""
    if k0 == 0:
        return (mx, Float32(0))
    if mx > m:
        return (mx, exp2s[DEG](m - mx))
    return (m, Float32(1))


@always_inline
def _acc_load[
    hd: Int, R: Int, NC: Int, W: Int, N: Int
](
    mut a: Array[SIMD[DType.float32, W], N],
    acc: MutPointer[Float32, _],
    dc: Int,
    k0: Int,
    alpha: SIMD[DType.float32, R],
):
    """a[r*NC + i] = acc[r][dc + i*W ..] * alpha[r]; `a` is zero-filled, which
    stands for the first step and for alpha == 0."""
    if k0 > 0:
        comptime for r in range(R):
            var ar = alpha[r]
            if ar == 1.0:
                comptime for i in range(NC):
                    a[r * NC + i] = acc.unsafe_load[width=W](r * hd + dc + i * W)
            elif ar != 0.0:
                var va = SIMD[DType.float32, W](ar)
                comptime for i in range(NC):
                    a[r * NC + i] = acc.unsafe_load[width=W](r * hd + dc + i * W) * va


@always_inline
def _store_rows[dtype: DType, hd: Int, R: Int](
    job: PagedJob,
    seq: Int,
    h0: Int,
    nrows: Int,
    l: SIMD[DType.float32, R],
    acc: MutPointer[Float32, _],
):
    """out = acc / l; a row that saw no key is exactly 0."""
    comptime L = PageLayout[dtype, hd]
    comptime E = 2 if dtype == DType.bfloat16 else 4
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
def bitcast_bf16[width: Int](x: SIMD[DType.int32, width]) -> SIMD[DType.bfloat16, 2 * width]:
    return bitcast[DType.bfloat16, 2 * width](x)


@always_inline
def _unit_bases[E: Int, hd: Int, PAGE: Int](
    job: PagedJob,
    bt: MutPointer[Int32, _],
    kvh: Int,
    page: Int,
    k0: Int,
    units: Int,
    kbase: MutPointer[Int64, _],
    vbase: MutPointer[Int64, _],
):
    """K and V addresses of the first key of each 16-key unit from `k0`."""
    var head_bytes = page * hd * E
    for g in range(units):
        var d = _div_page[PAGE](k0 + g * 16, page)
        var blk = (Int(bt.unsafe_load(d[0])) * job.kv_heads + kvh) * head_bytes
        kbase.unsafe_store(g, Int64(job.k_cache + blk + d[1] * 4))
        vbase.unsafe_store(g, Int64(job.v_cache + blk + d[1] * hd * E))


@always_inline
def _qk[
    dtype: DType, hd: Int, R: Int, G: Int, KT: Int, MASK: Bool
](
    kbase: MutPointer[Int64, _],
    page: Int,
    q: Int,
    j: Int,
    nk: Int,
    logits: MutPointer[Float32, _],
    mut mx: Array[SIMD[DType.float32, LANES], R],
):
    """logits[r][j + g*LANES ..] for G key vectors from key j, and the running
    lane max; with MASK, lanes at or past nk are -max and vectors past nk are
    neither scored into the max nor stored."""
    comptime V = SIMD[DType.float32, LANES]
    var kb = Array[Int, G](uninitialized=True)
    comptime for g in range(G):
        var key = j + g * LANES
        kb[g] = Int(kbase.unsafe_load(key >> 4)) + (key & 15) * 4
    var cv = Array[V, R * G](fill=V(0))
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
                    var k = ptr[BFloat16](kb[g]).unsafe_load[width = 2 * LANES](d2 * page * 2)
                    comptime for r in range(R):
                        # Query last: `vdpbf16ps` folds a broadcast load only
                        # as its last source (exact, symmetric products).
                        var qv = SIMD[DType.int32, LANES](qp.unsafe_load(r * (hd // 2) + d2))
                        cv[r * G + g] = dot_bf16[LANES](cv[r * G + g], k, bitcast_bf16[LANES](qv))
    else:
        # CH independent partial dot products per (row, vector): one chain over
        # all of head_dim 128 rounds past the 1e-6 fp32 gate (1.2-1.6e-6
        # measured), and the chains also overlap the K loads.
        var qf = ptr[Float32](q)
        comptime CH = max(1, min(4, 16 // (R * G)))
        var part = Array[V, R * G * CH](fill=V(0))
        comptime U = max(CH, 4)
        for d0 in range(0, hd, U):
            # U query dims as one vector, then a lane each: NEON's by-element
            # FMA (a scalar broadcast load per dim costs a load each there);
            # x86 folds it into an embedded broadcast either way.
            var qv = Array[SIMD[DType.float32, U], R](uninitialized=True)
            comptime for r in range(R):
                qv[r] = qf.unsafe_load[width=U](r * hd + d0)
            comptime for u in range(U):
                var d = d0 + u
                comptime for g in range(G):
                    var k = ptr[Float32](kb[g]).unsafe_load[width=LANES](d * page)
                    comptime for r in range(R):
                        comptime i = (r * G + g) * CH + u % CH
                        part[i] = V(qv[r][u]).fma(k, part[i])
        comptime for rg in range(R * G):
            comptime if CH == 4:
                cv[rg] = (part[rg * 4] + part[rg * 4 + 1]) + (part[rg * 4 + 2] + part[rg * 4 + 3])
            else:
                var sum = part[rg * CH]
                comptime for c in range(1, CH):
                    sum += part[rg * CH + c]
                cv[rg] = sum
    comptime assert G & (G - 1) == 0 and G <= 32, "G is a power of two"
    comptime for r in range(R):
        var t = Array[V, G](uninitialized=True)
        comptime for g in range(G):
            var key = j + g * LANES
            var v = cv[r * G + g]
            comptime if MASK:
                if key + LANES > nk:
                    var lane = iota[DType.int32, LANES]() + Int32(key)
                    v = lane.lt(SIMD[DType.int32, LANES](Int32(nk))).select(v, V(-Float32.MAX))
                if key < nk:
                    logits.unsafe_store[width=LANES](r * KT + key, v)
            else:
                logits.unsafe_store[width=LANES](r * KT + key, v)
            t[g] = v
        # The call's max as a tree: a chain of G `vmax` after a 16-vector sweep
        # sits on the softmax's critical path.
        comptime for lvl in range(5):
            comptime s = 16 >> lvl
            comptime if s < G:
                comptime for i in range(s):
                    t[i] = vmax[LANES](t[i], t[i + s])
        mx[r] = vmax[LANES](mx[r], t[0])


trait PVKernel:
    """How a step keeps its probabilities and contracts them with V: the one
    per-target piece of the decode driver (`Decode.PV`), as `InnerMatmulKernel`
    is for Modular's matmul."""

    comptime P16: Bool
    """Probabilities are also kept as bf16 (`pbuf`, and `pbuf_lo` when LO)."""
    comptime LN: Int
    """Key-parity accumulator sets per dim vector."""

    @staticmethod
    def store_p[LO: Bool](e: SIMD[DType.float32, LANES], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        """Keep the probabilities `e` of keys i .. i+LANES-1 (`odd`: the step has
        an odd number of keys)."""
        ...

    @staticmethod
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        """a += P[:, kf : kf+kv] @ V for the dims from `dc`, over a run of kv keys
        contiguous from address `vb` (FULL: one 16-key unit)."""
        ...


def _nc(dtype: DType, hd: Int, R: Int) -> Int:
    """P@V dim vectors per register chunk: R x NC accumulators stay in registers."""
    if R * hd // LANES <= 16:
        return hd // LANES
    var split = dtype == DType.bfloat16 and not V_PAIRS_ARM and hd % (2 * LANES) == 0
    return max(2 if split else 1, 16 // R)


struct PVRows[dtype: DType, hd: Int, R: Int](PVKernel):
    """fp32 P against V rows `[key][hd]` (bf16 V widened): x86, the C++ class of
    the 2-ulp bf16 gate."""

    comptime P16 = False
    comptime NC = _nc(Self.dtype, Self.hd, Self.R)
    comptime LN = 2 if Self.R * Self.NC <= 8 else 1
    """Even and odd keys feed independent FMA chains while the tile is small
    (one chain per dim vector is latency bound at R=1)."""

    @staticmethod
    @always_inline
    def store_p[LO: Bool](e: SIMD[DType.float32, LANES], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        ptr[Float32](logits).unsafe_store[width=LANES](i, e)

    @staticmethod
    @always_inline
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        comptime V = SIMD[DType.float32, LANES]
        comptime KT = _kt(Self.hd, Self.R)
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
        comptime if Self.LN == 2:
            while t + 1 < kv:
                var p0 = Array[V, Self.R](uninitialized=True)
                var p1 = Array[V, Self.R](uninitialized=True)
                comptime for r in range(Self.R):
                    p0[r] = V(pk.unsafe_load(r * KT + t))
                    p1[r] = V(pk.unsafe_load(r * KT + t + 1))
                comptime for i in range(NC):
                    var v0 = vrow(t, i)
                    var v1 = vrow(t + 1, i)
                    comptime for r in range(Self.R):
                        a[r * NC + i] = p0[r].fma(v0, a[r * NC + i])
                        a[(Self.R + r) * NC + i] = p1[r].fma(v1, a[(Self.R + r) * NC + i])
                t += 2
        while t < kv:
            var p0 = Array[V, Self.R](uninitialized=True)
            comptime for r in range(Self.R):
                p0[r] = V(pk.unsafe_load(r * KT + t))
            comptime for i in range(NC):
                var v0 = vrow(t, i)
                comptime for r in range(Self.R):
                    a[r * NC + i] = p0[r].fma(v0, a[r * NC + i])
            t += 1


struct PVPairs[hd: Int, R: Int](PVKernel):
    """bf16 P against bf16 V key pairs `[key/2][hd][2]` with `dot_bf16`
    (`bfdot`): one load is LANES dims of a key pair, contracted against the
    pair's two probabilities. aarch64, where widening V was 70% of the kernel."""

    comptime P16 = True
    comptime NC = _nc(DType.bfloat16, Self.hd, Self.R)
    comptime LN = 1

    @staticmethod
    @always_inline
    def store_p[LO: Bool](e: SIMD[DType.float32, LANES], logits: Int, pbuf: Int, pbuf_lo: Int, i: Int, odd: Bool):
        var hi = e.cast[DType.bfloat16]()
        ptr[BFloat16](pbuf).unsafe_store[width=LANES](i, hi)
        comptime if LO:
            ptr[BFloat16](pbuf_lo).unsafe_store[width=LANES](i, (e - hi.cast[DType.float32]()).cast[DType.bfloat16]())
        if odd:  # the unpaired last key is an fp32 FMA
            ptr[Float32](logits).unsafe_store[width=LANES](i, e)

    @staticmethod
    @always_inline
    def keys[N: Int, FULL: Bool](
        mut a: Array[SIMD[DType.float32, LANES], N], vb: Int, dc: Int, p: Int, pbuf: Int, pbuf_lo: Int,
        kf: Int, kv: Int, len: Int,
    ):
        comptime V = SIMD[DType.float32, LANES]
        comptime B = SIMD[DType.bfloat16, 2 * LANES]
        comptime KT = _kt(Self.hd, Self.R)
        comptime NC = Self.NC
        var vbg = vb + dc * 4
        var w0 = kf >> 1
        var pw = ptr[Int32](pbuf)  # one bf16 key pair per word
        var plw = ptr[Int32](pbuf_lo)

        @always_inline
        def prob(w: MutPointer[Int32, _], r: Int, t: Int) capturing -> B:
            return bitcast_bf16[LANES](SIMD[DType.int32, LANES](w.unsafe_load(r * (KT // 2) + w0 + (t >> 1))))

        var t = 0
        if len >= 128:
            comptime if Self.R == 1:
                # Two key pairs per step.
                while t + 3 < kv:
                    var p0 = prob(pw, 0, t)
                    var p1 = prob(pw, 0, t + 2)
                    var vp0 = ptr[BFloat16](vbg + t * Self.hd * 2)
                    var vp1 = ptr[BFloat16](vbg + (t + 2) * Self.hd * 2)
                    comptime for i in range(NC):
                        a[i] = dot_bf16[LANES](a[i], vp0.unsafe_load[width = 2 * LANES](i * 2 * LANES), p0)
                    comptime for i in range(NC):
                        a[i] = dot_bf16[LANES](a[i], vp1.unsafe_load[width = 2 * LANES](i * 2 * LANES), p1)
                    t += 4
            while t + 1 < kv:
                var pr = Array[B, Self.R](uninitialized=True)
                comptime for r in range(Self.R):
                    pr[r] = prob(pw, r, t)
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    var v = vp.unsafe_load[width = 2 * LANES](i * 2 * LANES)
                    comptime for r in range(Self.R):
                        a[r * NC + i] = dot_bf16[LANES](a[r * NC + i], v, pr[r])
                t += 2
        else:
            # Short contexts can have one dominant key, where single-bf16 P
            # misses the 1e-3 gate: add P's low bf16 residual.
            while t + 1 < kv:
                var pr = Array[B, Self.R](uninitialized=True)
                var lo = Array[B, Self.R](uninitialized=True)
                comptime for r in range(Self.R):
                    pr[r] = prob(pw, r, t)
                    lo[r] = prob(plw, r, t)
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    var v = vp.unsafe_load[width = 2 * LANES](i * 2 * LANES)
                    comptime for r in range(Self.R):
                        a[r * NC + i] = dot_bf16[LANES](a[r * NC + i], v, pr[r])
                        a[r * NC + i] = dot_bf16[LANES](a[r * NC + i], v, lo[r])
                t += 2
        comptime if not FULL:
            if t < kv:
                # The unpaired last key: only its half of the pair (0 * a padded
                # NaN is still NaN).
                var vp = ptr[BFloat16](vbg + t * Self.hd * 2)
                comptime for i in range(NC):
                    var v = rebind[V](vp.unsafe_load[width = 2 * LANES](i * 2 * LANES).cast[DType.float32]().deinterleave()[0])
                    comptime for r in range(Self.R):
                        a[r * NC + i] = V(ptr[Float32](p).unsafe_load(r * KT + kf + t)).fma(v, a[r * NC + i])


@always_inline
def _exp_row[
    dtype: DType, hd: Int, R: Int, LO: Bool
](logits: Int, pbuf: Int, pbuf_lo: Int, row: Int, nk: Int, c: Float32, mn: Float32) -> Float32:
    """p = 2^(logit*c - mn) for one row of a step, kept by `Decode.PV`; returns
    the row sum (NS partial sums)."""
    comptime D = Decode[dtype, hd, R]
    comptime V = SIMD[DType.float32, LANES]
    var lg = ptr[Float32](logits)
    var odd = (nk & 1) == 1
    var s = Array[V, D.NS](fill=V(0))

    @always_inline
    def one(t: Int) capturing -> V:
        var e = exp2[D.DEG, LANES](lg.unsafe_load[width=LANES](row + t).fma(V(c), V(-mn)))
        D.PV.store_p[LO](e, logits, pbuf, pbuf_lo, row + t, odd)
        return e

    var t = 0
    while t + D.NS * LANES <= nk:
        comptime for u in range(D.NS):
            s[u] += one(t + u * LANES)
        t += D.NS * LANES
    while t < nk:
        s[0] += one(t)
        t += LANES
    var sum = s[0]
    comptime for u in range(1, D.NS):
        sum += s[u]
    return sum.reduce_add()


@always_inline
def _pv[
    dtype: DType, hd: Int, R: Int, FULL: Bool
](
    k0: Int, nk: Int, len: Int, units: Int, alpha: SIMD[DType.float32, R], vbase: MutPointer[Int64, _],
    pbuf: Int, pbuf_lo: Int, p: Int, acc: MutPointer[Float32, _],
):
    """acc[r] = acc[r]*alpha[r] + sum_j P[r][j] * V[k0+j] over exactly nk keys
    (never past `len`: 0 * a padded NaN is NaN), NC dim vectors at a time, in
    runs of units contiguous in memory (all of a page past 16 keys, so the key
    loops are not cut every 16 keys; FULL: every unit of a step, 16 keys each)."""
    comptime D = Decode[dtype, hd, R]
    comptime V = SIMD[DType.float32, LANES]
    comptime NC = D.NC
    comptime LN = D.PV.LN
    comptime for dc in range(0, hd, NC * LANES):
        var a = Array[V, LN * R * NC](fill=V(0))
        _acc_load[hd, R, NC, LANES, LN * R * NC](a, acc, dc, k0, alpha)
        var g = 0
        var gend = D.UNITS if FULL else units
        while g < gend:
            var g1 = g + 1
            comptime if not FULL:
                comptime RUN = D.UNIT * hd * D.E
                while g1 < gend and Int(vbase.unsafe_load(g1)) == Int(vbase.unsafe_load(g1 - 1)) + RUN:
                    g1 += 1
            var kv = D.UNIT if FULL else min(g1 * D.UNIT, nk) - g * D.UNIT
            D.PV.keys[LN * R * NC, FULL](a, Int(vbase.unsafe_load(g)), dc, p, pbuf, pbuf_lo, g * D.UNIT, kv, len)
            g = g1
        comptime for r in range(R):
            comptime for i in range(NC):
                var s = a[r * NC + i]
                comptime if LN == 2:
                    s += a[(R + r) * NC + i]
                acc.unsafe_store[width=LANES](r * hd + dc + i * LANES, s)


def _attend_rows[
    dtype: DType, hd: Int, R: Int, PAGE: Int
](job: PagedJob, seq: Int, kvh: Int, h0: Int, nrows: Int):
    """Query heads `h0 .. h0+nrows-1` (nrows <= R, all of kv head `kvh`)."""
    comptime D = Decode[dtype, hd, R]
    comptime KT = D.KT
    comptime assert D.UNIT % LANES == 0 and KT % D.UNIT == 0, "a step is whole units of whole vectors"
    comptime assert hd % (D.NC * LANES) == 0, "P@V chunks tile head_dim"
    comptime V = SIMD[DType.float32, LANES]
    var page = PAGE if PAGE else job.page
    var len = Int(ptr[Int32](job.lens).unsafe_load(seq)) + (
        1 if (job.do_store and job.slots == 0) else 0
    )
    var bt = ptr[Int32](job.block_table + seq * job.bt_stride * 4)
    var c = job.scale * Float32(LOG2E)

    # Queries as int32 words (bf16 dim pairs or fp32): at R == 1 read in place,
    # for R > 1 pack R rows (the last one repeated) into a stack buffer.
    comptime QW = hd * D.E // 4
    var qbuf = stack_allocation[(R * QW) if R > 1 else 1, DType.int32, alignment=64]()
    var q = (job.q + (seq * job.q_stride + h0 * hd) * D.E) if R == 1 else Int(qbuf)
    comptime if R > 1:
        for r in range(R):
            var s = ptr[Int32](job.q + (seq * job.q_stride + (h0 + min(r, nrows - 1)) * hd) * D.E)
            comptime for w in range(0, QW, 4):
                qbuf.unsafe_store[width=4](r * QW + w, s.unsafe_load[width=4](w))

    var kbase = stack_allocation[D.UNITS, DType.int64, alignment=64]()
    var vbase = stack_allocation[D.UNITS, DType.int64, alignment=64]()
    var logits = stack_allocation[R * KT, DType.float32, alignment=64]()
    var pbuf = stack_allocation[(R * KT) if D.PV.P16 else 1, DType.bfloat16, alignment=64]()
    var pbuf_lo = stack_allocation[(R * KT) if D.PV.P16 else 1, DType.bfloat16, alignment=64]()
    var acc = stack_allocation[R * hd, DType.float32, alignment=64]()
    var m = SIMD[DType.float32, R](-Float32.MAX)
    var l = SIMD[DType.float32, R](0)

    var k0 = 0
    while k0 < len:
        var nk = min(KT, len - k0)
        var units = (nk + D.UNIT - 1) // D.UNIT
        _unit_bases[D.E, hd, PAGE](job, bt, kvh, page, k0, units, kbase, vbase)

        # ---- Q@K^T: G key vectors per call; the tail in masked calls of at
        # most one unit, and a single vector when that is all that is left.
        var mx = Array[V, R](fill=V(-Float32.MAX))
        var j = 0
        while j + D.G * LANES <= nk:
            _qk[dtype, hd, R, D.G, KT, False](kbase, page, q, j, nk, logits, mx)
            j += D.G * LANES
        while j < nk:
            if nk - j <= LANES:
                _qk[dtype, hd, R, 1, KT, True](kbase, page, q, j, nk, logits, mx)
                break
            _qk[dtype, hd, R, D.G_TAIL, KT, True](kbase, page, q, j, nk, logits, mx)
            j += D.G_TAIL * LANES

        # ---- online softmax per row; p = 2^(logit*c - m), m in scaled log2 units
        var alpha = SIMD[DType.float32, R](1)
        comptime for r in range(R):
            var s = _rescale[D.DEG](k0, mx[r].reduce_max() * c, m[r])
            m[r] = s[0]
            alpha[r] = s[1]
            var sum: Float32
            if D.PV.P16 and len < 128:
                sum = _exp_row[dtype, hd, R, True](Int(logits), Int(pbuf), Int(pbuf_lo), r * KT, nk, c, s[0])
            else:
                sum = _exp_row[dtype, hd, R, False](Int(logits), Int(pbuf), Int(pbuf_lo), r * KT, nk, c, s[0])
            l[r] = (l[r] * alpha[r] if k0 > 0 else Float32(0)) + sum

        # ---- P@V
        comptime if D.FULL:
            if nk == KT:
                _pv[dtype, hd, R, True](k0, nk, len, units, alpha, vbase, Int(pbuf), Int(pbuf_lo), Int(logits), acc)
                k0 += nk
                continue
        _pv[dtype, hd, R, False](k0, nk, len, units, alpha, vbase, Int(pbuf), Int(pbuf_lo), Int(logits), acc)
        k0 += nk

    _store_rows[dtype, hd, R](job, seq, h0, nrows, l, acc)



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
