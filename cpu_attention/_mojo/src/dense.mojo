"""Dense (non-paged) causal self-attention -- ``flash_attn_func``'s kernel.

Where ``attention.mojo`` implements the *paged* decode step (one query token per
sequence, K/V read through a block table), this is the other half of
flash-attention: whole ``seqlen_q x seqlen_k`` attention on contiguous tensors,

    q [batch, seq_q, nheads,   head_dim]
    k [batch, seq_k, nheads_k, head_dim]
    v [batch, seq_k, nheads_k, head_dim]
    o [batch, seq_q, nheads,   head_dim]

with the causal mask aligned to the bottom-right corner (``j <= i + seq_k -
seq_q``), optional GQA/MQA (``nheads % nheads_k == 0``) and no dropout, ALiBi,
softcap or sliding window.

Structure, borrowd from vLLM's ``csrc/cpu`` ``VEC`` kernel:

* one work item owns ``qt`` consecutive query rows of one (batch, head) for the
  whole key sweep, so an online-softmax accumulator (``head_dim`` floats per
  row) lives in registers across the sweep and K/V are read once per row block;
* keys are scored ``KT_TILE`` at a time as ``Q_tile @ K^T`` with the *transposed*
  K, so one broadcast-FMA covers ``KT_TILE`` positions (vLLM gets the same layout
  for free because it stores K transposed inside its pages; a dense K is
  row-major, so the Python side hands us a transposed scratch copy);
* V stays row-major: ``acc[d] += p[j] * v[j][d]`` is one FMA over the head dim,
  shared by every row of the block;
* bf16 takes a bf16 dot-product path (``dpbf16ps`` on AVX-512, ``bfdot`` on
  pre-``bfmmla`` ARM), which is what makes bf16 faster than fp32 here: AVX-512
  has no native bf16 add/multiply, so the only thing bf16 may be used for is the
  dot;
* on ARM the bf16 path is instead ``_dense_rows_bfmmla``: ``vbfmmlaq_f32``
  accumulates a 2x2 fp32 tile per instruction (four fp32 lanes of output per
  lane slot, against ``bfdot``'s two) on *both* phases, which is what vLLM's
  ``csrc/cpu`` NEON kernel does and why it is the reference to beat here. K and
  V are repacked by the Python side for it (``_transpose_head`` mode 2 packs K
  position-paired over four head dims, ``_vpack_head`` packs V position-major
  over head-dim pairs, because P@V contracts over keys).

Work items are handed out round-robin over the OpenMP pool (``GOMP_parallel`` on
torch's own pool, as in ``attention.mojo``) because causal masking makes the
late query rows ~2x more expensive than the early ones -- a contiguous slice per
thread would leave half the cores idle at the end of a long prefill.
"""

from std.ffi import external_call
from std.math import exp, fma, iota
from std.memory import MutPointer, bitcast, stack_allocation
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic, simd_width_of
from std.sys.info import CompilationTarget


comptime X86_DOT = CompilationTarget.is_x86() and CompilationTarget._has_feature[
    "avx512bf16"
]()
comptime NEON_DOT = CompilationTarget.has_neon() and CompilationTarget._has_feature[
    "bf16"
]()
comptime BF16_DOT = X86_DOT or NEON_DOT

comptime LANES = simd_width_of[DType.float32]()

comptime KT_TILE = 4 * LANES
"""Keys scored per online-softmax step: four *registers* of positions.

Was a flat 64, which is four 16-lane vectors on AVX-512 (+6% bf16 / +16% fp32
over 32 there) but sixteen 4-lane vectors on NEON -- a ``qt``-row block then
needs ``qt * 16`` logit registers plus ``qt * head_dim / 4`` of accumulator on a
32-register machine, i.e. it spills everything and runs at ~30% of the ARM
roofline. Tying the tile to the lane width keeps the register budget identical
on both ISAs: 64 keys on AVX-512, 16 on NEON.
"""

comptime BFMMLA = NEON_DOT
"""NEON ``bfmmla``: a 2x2 fp32 tile per instruction, contracted over four bf16
lanes. Four fp32 lanes of output per lane slot instead of ``bfdot``'s two, and
one operand load feeds 2-4 instructions, which is why vLLM's ARM bf16 kernel
runs on this instruction and not on ``bfdot`` (their ``.so`` has 9436 ``bfmmla``
and 0 ``bfdot``). ARM-only: AVX-512's ``dpbf16ps`` already does 2x2 by itself.
"""

comptime NEG_INF = -3.4028234663852886e38


@inline(.always)
def _bfmmla(
    acc: SIMD[DType.float32, 4],
    a: SIMD[DType.bfloat16, 8],
    b: SIMD[DType.bfloat16, 8],
) -> SIMD[DType.float32, 4]:
    """``acc += a @ b`` as a 2x2 fp32 tile, contracted over four bf16 lanes.

    ``acc = (row0 col0, row0 col1, row1 col0, row1 col1)``; ``a`` holds the two
    rows in its two 4-lane halves and ``b`` the two columns the same way
    (verified against a scalar reference in ``tests/bfmmla_layout.mojo``).
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.bfmmla",
        SIMD[DType.float32, 4],
        has_side_effect=False,
    ](acc, a, b)


comptime BKS = 64
"""Keys per P@V super-tile on the ``bfmmla`` path.

The output accumulator is ``qt * head_dim`` floats, i.e. 16-32 registers per row
on a 32-register machine, so it lives in L1 and is loaded and stored once per
super-tile: ``2 * qt/2 * head_dim/2`` vector ops per super-tile whatever its
size, against ``qt/2 * head_dim/2 * keys/4`` bfmmla. At 16 keys (one softmax
step) that traffic is half the bfmmla count; at 64 keys it is an eighth. The
softmax keeps running on 16-key steps inside the super-tile, so the register
budget does not grow with it.
"""

comptime BDCP = 8
"""Head-dim *pairs* per P@V chunk (16 head dims): ``qt/2 * 8`` accumulator
registers beside one V vector and the probability panel. Mirrors vLLM's
``NTile = 8`` output columns per gemm call."""


struct DenseJob(Copyable, ImplicitlyCopyable, Movable):
    """One dense attention call, as raw addresses plus element strides."""

    var q: Int
    var kt: Int
    var v: Int
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
    # Transposed K: [batch, kv head, head_dim(/2), seq_k(padded), (2)]
    var kt_stride_bh: Int
    var kt_stride_d: Int
    var kt_stride_t: Int
    # Keys per contiguous K^T block: the transposed K is `[tile][dim][kt_blk]`,
    # so one Q@K^T tile is one linear run instead of `head_dim` rows
    # `kt_blk * sizeof(T)` apart. Set to `seq_k_pad` for the unblocked layout.
    var kt_blk: Int
    var v_stride_b: Int
    var v_stride_t: Int
    var v_stride_h: Int
    var o_stride_b: Int
    var o_stride_t: Int
    var o_stride_h: Int
    var scale: Float32
    var causal: Bool

    def __init__(out self):
        self.q = 0
        self.kt = 0
        self.v = 0
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
        self.kt_stride_bh = 0
        self.kt_stride_d = 0
        self.kt_stride_t = 0
        self.kt_blk = 0
        self.v_stride_b = 0
        self.v_stride_t = 0
        self.v_stride_h = 0
        self.o_stride_b = 0
        self.o_stride_t = 0
        self.o_stride_h = 0
        self.scale = 1.0
        self.causal = False


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


@inline(.always)
def _bf16_dot[
    dtype: DType
](
    acc: SIMD[DType.float32, LANES],
    a: SIMD[DType.bfloat16, 2 * LANES],
    b: SIMD[dtype, 2 * LANES],
) -> SIMD[DType.float32, LANES]:
    """``acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]``, fp32 accumulate."""
    comptime if X86_DOT:
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps.512",
            SIMD[DType.float32, LANES],
            has_side_effect=False,
        ](acc, a, b.cast[DType.bfloat16]())
    elif NEON_DOT:
        return llvm_intrinsic[
            "llvm.aarch64.neon.bfdot",
            SIMD[DType.float32, LANES],
            has_side_effect=False,
        ](acc, a, b.cast[DType.bfloat16]())
    else:
        var out = acc
        for i in range(LANES):
            out[i] += Scalar[DType.float32](a[2 * i]) * Scalar[DType.float32](
                b[2 * i]
            ) + Scalar[DType.float32](a[2 * i + 1]) * Scalar[DType.float32](
                b[2 * i + 1]
            )
        return out


def _vexp[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """``exp(x)`` for ``x <= 0`` as a Cephes-degree polynomial plus an exponent
    bit-trick, in ~10 FMAs and three live vectors.

    ``std.math.exp`` on a SIMD is *not* this: on the AVX-512 host it costs ~36
    cycles per lane here (it is not vectorised into a polynomial), which at
    head_dim 128 was 58% of the dense kernel's time -- and its scratch footprint
    also pushes the P@V accumulator out of registers. Clamping at -87 (exp(-87) =
    1.6e-38) keeps the 2^k scale in range; masked lanes carry NEG_INF, so they
    land on the clamp and contribute ~0.
    """
    comptime LOG2E = 1.4426950408889634
    comptime LN2_HI = 0.693145751953125
    comptime LN2_LO = 1.428606765330187e-6
    var xc = max(x, SIMD[DType.float32, width](-87.0))
    # Round to nearest through the +1.5*2^23 trick: branch-free, and the low
    # mantissa bits of `t` hold the integer k.
    comptime MAGIC = 12582912.0
    var t = fma(
        xc, SIMD[DType.float32, width](LOG2E), SIMD[DType.float32, width](MAGIC)
    )
    var kf = t - SIMD[DType.float32, width](MAGIC)
    var r = fma(-kf, SIMD[DType.float32, width](LN2_HI), xc)
    r = fma(-kf, SIMD[DType.float32, width](LN2_LO), r)
    var p = SIMD[DType.float32, width](1.9875691500e-4)
    p = fma(p, r, SIMD[DType.float32, width](1.3981999507e-3))
    p = fma(p, r, SIMD[DType.float32, width](8.3334519073e-3))
    p = fma(p, r, SIMD[DType.float32, width](4.1665795894e-2))
    p = fma(p, r, SIMD[DType.float32, width](1.6666665459e-1))
    p = fma(p, r, SIMD[DType.float32, width](5.0000001201e-1))
    p = fma(p, r, SIMD[DType.float32, width](1.0))
    p = fma(p, r, SIMD[DType.float32, width](1.0))
    var ki = kf.cast[DType.int32]()
    var sbits = (ki + SIMD[DType.int32, width](127)) << SIMD[
        DType.int32, width
    ](23)
    return p * bitcast[DType.float32, width](sbits)


@inline(.always)
def _exp2_g[
    width: Int
](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Width-generic ``2^y`` for ``y <= 0``: magic-constant rounding + degree-4
    minimax + exponent insert, ~13 instructions.

    ``_exp2_fast`` is the 4-lane version and uses ``roundeven``; this one exists
    because the fp32 tile kernel works in ``LANES``-wide vectors and the
    intrinsic's name carries the width. Degree 5 (7.5e-8 relative) because the
    probability error is amplified ~40x into the output: degree 4 (2.6e-6)
    measured 1.1e-4 against the 2e-4 tolerance, degree 3 would fail it.
    """
    comptime MAGIC = 12582912.0
    var yc = max(y, SIMD[DType.float32, width](-127.0))
    var n = (yc + SIMD[DType.float32, width](MAGIC)) - SIMD[DType.float32, width](
        MAGIC
    )
    var f = yc - n
    var p = SIMD[DType.float32, width](1.3276673810e-3)
    p = p * f + SIMD[DType.float32, width](9.6755714487e-3)
    p = p * f + SIMD[DType.float32, width](5.5507129206e-2)
    p = p * f + SIMD[DType.float32, width](2.4022119177e-1)
    p = p * f + SIMD[DType.float32, width](6.9314696713e-1)
    p = p * f + SIMD[DType.float32, width](1.0000000718e0)
    var ki = n.cast[DType.int32]()
    var sbits = (ki + SIMD[DType.int32, width](127)) << SIMD[
        DType.int32, width
    ](23)
    return p * bitcast[DType.float32, width](sbits)


@inline(.always)
def _exp2f(x: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Scalar ``2^x`` through the same polynomial (the online-softmax rescale).
    """
    return _exp2_g[1](SIMD[DType.float32, 1](x))[0]


@inline(.always)
def _vexp_f(x: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Scalar ``exp`` through the same polynomial (the online-softmax rescale).
    """
    return _vexp[1](SIMD[DType.float32, 1](x))[0]


@inline(.always)
def _dense_rows[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """Attend ``qt`` consecutive query rows of (batch, head) over all keys.

    fp32 path: Q is widened and ``scale`` folded in (as the paged kernel does).
    """
    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[Scalar[dtype]](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[Scalar[dtype]](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var v_base = _ptr[Scalar[dtype]](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[Scalar[dtype]](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    var qv = stack_allocation[qt * head_dim, DType.float32]()
    comptime for r in range(qt):
        if r < nrows:
            var qsrc = q_base.unsafe_offset((q0 + r) * job.q_stride_t)
            qv.unsafe_store[width=head_dim](
                r * head_dim,
                qsrc.unsafe_load[width=head_dim]().cast[DType.float32]()
                * job.scale,
            )

    var acc = Array[SIMD[DType.float32, head_dim], qt](uninitialized=True)
    var row_max = Array[Float32, qt](uninitialized=True)
    var row_sum = Array[Float32, qt](uninitialized=True)
    comptime for r in range(qt):
        acc[r] = SIMD[DType.float32, head_dim](0.0)
        row_max[r] = Scalar[DType.float32](NEG_INF)
        row_sum[r] = Scalar[DType.float32](0.0)

    var probs = stack_allocation[qt * KT_TILE, DType.float32]()

    # Highest key index any row of the block may see, + 1.
    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # Offset of this tile inside the (possibly blocked) K^T.
        var kbase = (t0 // job.kt_blk) * (head_dim * job.kt_blk) + (
            t0 % job.kt_blk
        ) * job.kt_stride_t
        var logits = Array[SIMD[DType.float32, KT_TILE], qt](uninitialized=True)
        comptime for r in range(qt):
            logits[r] = SIMD[DType.float32, KT_TILE](0.0)

        # Q@K^T: one K^T vector (KT_TILE positions of one head dim) feeds every
        # row of the block, so the loads are amortized over `qt` FMAs.
        var d = 0
        while d < head_dim:
            var kv = kt_base.unsafe_load[width=KT_TILE](
                d * job.kt_stride_d + kbase
            ).cast[DType.float32]()
            comptime for r in range(qt):
                # fma(), not `+=`/`*`: multiply+add pairs do not contract on
                # their own here, and an uncontracted pair is two instructions
                # per 16 MACs instead of one.
                logits[r] = fma(
                    SIMD[DType.float32, KT_TILE](qv.unsafe_load(r * head_dim + d)),
                    kv,
                    logits[r],
                )
            d += 1

        # Causal tail, or the last tile of a non-causal call: kill the lanes
        # that no row may see. Every row has its own limit, so the test is per
        # row; outside the boundary tile the whole tile is visible.
        var boundary = t0 + KT_TILE > seq_k
        if job.causal:
            boundary = t0 + KT_TILE > q0 + diag + 1
        if boundary:
            # Causal: keep t <= q0 + r + diag. Non-causal: keep t < seq_k. The
            # causal limit is always <= seq_k - 1, so one comparison covers both.
            var idx = iota[DType.int32, KT_TILE](Int32(t0))
            comptime for r in range(qt):
                var limit = Int32(seq_k)
                if job.causal:
                    # Inclusive: query row q0+r may see key q0+r+diag.
                    limit = Int32(q0 + r + diag + 1)
                logits[r] = idx.lt(SIMD[DType.int32, KT_TILE](limit)).select(
                    logits[r], SIMD[DType.float32, KT_TILE](NEG_INF)
                )

        # Online softmax for the block, then P@V.
        comptime for r in range(qt):
            var m_old = row_max[r]
            var m_new = max(
                m_old, Scalar[DType.float32](logits[r].reduce_max())
            )
            var alpha = Scalar[DType.float32](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                alpha = _vexp_f(m_old - m_new)
            var p = SIMD[DType.float32, KT_TILE](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = _vexp[KT_TILE](logits[r] - m_new)
            row_sum[r] = row_sum[r] * alpha + Scalar[DType.float32](
                p.reduce_add()
            )
            row_max[r] = m_new
            acc[r] = acc[r] * alpha
            probs.unsafe_store[width=KT_TILE](r * KT_TILE, p)

        # Only the keys that exist: a short last tile has KT_TILE - jn masked
        # lanes whose probability is exactly 0, and reading their V would walk
        # past the end of the tensor (0 * garbage is garbage, not 0).
        var jn = min(KT_TILE, loop_end - t0)
        var j = 0
        while j < jn:
            var vrow = v_base.unsafe_load[width=head_dim](
                (t0 + j) * job.v_stride_t
            ).cast[DType.float32]()
            comptime for r in range(qt):
                acc[r] = fma(
                    SIMD[DType.float32, head_dim](
                        probs.unsafe_load(r * KT_TILE + j)
                    ),
                    vrow,
                    acc[r],
                )
            j += 1

        t0 += KT_TILE

    comptime for r in range(qt):
        if r < nrows:
            var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
            var s = row_sum[r]
            if s > Scalar[DType.float32](0.0):
                out.unsafe_store[width=head_dim](
                    (acc[r] * (Scalar[DType.float32](1.0) / s)).cast[dtype]()
                )
            else:
                # Every key of this row was masked: flash-attn returns zeros.
                out.unsafe_store[width=head_dim](
                    SIMD[dtype, head_dim](Scalar[dtype](0))
                )


@inline(.always)
def _dense_rows_mem[
    dtype: DType, head_dim: Int, qt: Int, kt: Int, dc: Int, pairs: Bool
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """``_dense_rows`` with the output accumulator in memory, tiled over the head dim.

    The register version needs ``qt * head_dim`` floats of accumulator plus
    ``qt * kt / LANES`` of logits, so at ``head_dim = 128`` it spills as soon as
    ``qt > 2`` -- and a spilled accumulator is reloaded and rewritten once per
    key, which is what keeps fp32 at that shape behind SDPA. Here only ``dc``
    head dims of the ``qt`` rows are live (``qt * dc / LANES`` zmm), the full
    ``[qt, head_dim]`` accumulator lives in L1 and is touched once per key tile,
    and ``kt`` drops to 32 so the logits fit beside it. ``qt`` can then be 8,
    which halves the K^T/V traffic per query row (flops per byte scale with the
    rows held in flight, so this is the only way to buy it back).
    """
    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[Scalar[dtype]](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[Scalar[dtype]](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var v_base = _ptr[Scalar[dtype]](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[Scalar[dtype]](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    comptime OPN = 2 * LANES
    var qv = stack_allocation[qt * head_dim, DType.float32]()
    var q_pairs = stack_allocation[qt * (head_dim // 2) * OPN, BFloat16]()
    var acc = stack_allocation[qt * head_dim, DType.float32]()
    comptime for r in range(qt):
        if r < nrows:
            var qsrc = q_base.unsafe_offset((q0 + r) * job.q_stride_t)
            comptime if pairs:
                # bf16 dot: one instruction contracts two head dims of LANES
                # positions, so Q is broadcast as (q[2d], q[2d+1]) pairs.
                comptime for dp in range(head_dim // 2):
                    var q0v = Scalar[DType.bfloat16](qsrc.unsafe_load(2 * dp))
                    var q1v = Scalar[DType.bfloat16](
                        qsrc.unsafe_load(2 * dp + 1)
                    )
                    q_pairs.unsafe_store[width=OPN](
                        (r * (head_dim // 2) + dp) * OPN,
                        SIMD[DType.bfloat16, LANES](q0v).interleave(
                            SIMD[DType.bfloat16, LANES](q1v)
                        ),
                    )
            else:
                qv.unsafe_store[width=head_dim](
                    r * head_dim,
                    qsrc.unsafe_load[width=head_dim]().cast[DType.float32]()
                    * job.scale,
                )
        comptime for i in range(head_dim // LANES):
            acc.unsafe_store[width=LANES](
                r * head_dim + i * LANES, SIMD[DType.float32, LANES](0.0)
            )

    var row_max = Array[Float32, qt](uninitialized=True)
    var row_sum = Array[Float32, qt](uninitialized=True)
    comptime for r in range(qt):
        row_max[r] = Scalar[DType.float32](NEG_INF)
        row_sum[r] = Scalar[DType.float32](0.0)

    var probs = stack_allocation[qt * kt, DType.float32]()

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # Offset of this tile inside the (possibly blocked) K^T.
        var kbase = (t0 // job.kt_blk) * (head_dim * job.kt_blk) + (
            t0 % job.kt_blk
        ) * job.kt_stride_t
        var logits = Array[SIMD[DType.float32, kt], qt](uninitialized=True)
        comptime if pairs:
            comptime NV = kt // LANES
            var accs = Array[SIMD[DType.float32, LANES], qt * NV](
                uninitialized=True
            )
            comptime for i in range(qt * NV):
                accs[i] = SIMD[DType.float32, LANES](0.0)
            var dp = 0
            while dp < head_dim // 2:
                comptime for r in range(qt):
                    var a = q_pairs.unsafe_load[width=OPN](
                        (r * (head_dim // 2) + dp) * OPN
                    )
                    comptime for v in range(NV):
                        accs[r * NV + v] = _bf16_dot[dtype](
                            accs[r * NV + v],
                            a,
                            kt_base.unsafe_load[width=OPN](
                                dp * job.kt_stride_d
                                + kbase
                                + v * OPN
                            ),
                        )
                dp += 1
            comptime for r in range(qt):
                var l = SIMD[DType.float32, kt](0.0)
                comptime for v in range(NV):
                    comptime for i in range(LANES):
                        l[v * LANES + i] = accs[r * NV + v][i]
                logits[r] = l * job.scale
        else:
            comptime for r in range(qt):
                logits[r] = SIMD[DType.float32, kt](0.0)
            var d = 0
            while d < head_dim:
                var kv = kt_base.unsafe_load[width=kt](
                    d * job.kt_stride_d + kbase
                ).cast[DType.float32]()
                comptime for r in range(qt):
                    logits[r] = fma(
                        SIMD[DType.float32, kt](qv.unsafe_load(r * head_dim + d)),
                        kv,
                        logits[r],
                    )
                d += 1

        var boundary = t0 + kt > seq_k
        if job.causal:
            boundary = t0 + kt > q0 + diag + 1
        if boundary:
            var idx = iota[DType.int32, kt](Int32(t0))
            comptime for r in range(qt):
                var limit = Int32(seq_k)
                if job.causal:
                    limit = Int32(q0 + r + diag + 1)
                logits[r] = idx.lt(SIMD[DType.int32, kt](limit)).select(
                    logits[r], SIMD[DType.float32, kt](NEG_INF)
                )

        comptime for r in range(qt):
            var m_old = row_max[r]
            var m_new = max(
                m_old, Scalar[DType.float32](logits[r].reduce_max())
            )
            var alpha = Scalar[DType.float32](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                alpha = _vexp_f(m_old - m_new)
            var p = SIMD[DType.float32, kt](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = _vexp[kt](logits[r] - m_new)
            # Nothing accumulated yet (row_sum == 0 implies acc is all zeros) or
            # the max did not move: both are exact no-ops for the rescale.
            if row_sum[r] > Scalar[DType.float32](0.0) and alpha != Scalar[
                DType.float32
            ](1.0):
                comptime for i in range(head_dim // LANES):
                    acc.unsafe_store[width=LANES](
                        r * head_dim + i * LANES,
                        acc.unsafe_load[width=LANES](r * head_dim + i * LANES)
                        * alpha,
                    )
            row_sum[r] = row_sum[r] * alpha + Scalar[DType.float32](
                p.reduce_add()
            )
            row_max[r] = m_new
            probs.unsafe_store[width=kt](r * kt, p)

        var jn = min(kt, loop_end - t0)
        comptime for dci in range(head_dim // dc):
            var accv = Array[SIMD[DType.float32, dc], qt](uninitialized=True)
            comptime for r in range(qt):
                accv[r] = acc.unsafe_load[width=dc](r * head_dim + dci * dc)
            var j = 0
            while j < jn:
                var vrow = v_base.unsafe_load[width=dc](
                    (t0 + j) * job.v_stride_t + dci * dc
                ).cast[DType.float32]()
                comptime for r in range(qt):
                    accv[r] = fma(
                        SIMD[DType.float32, dc](probs.unsafe_load(r * kt + j)),
                        vrow,
                        accv[r],
                    )
                j += 1
            comptime for r in range(qt):
                acc.unsafe_store[width=dc](r * head_dim + dci * dc, accv[r])

        t0 += kt

    comptime for r in range(qt):
        if r < nrows:
            var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
            var s = row_sum[r]
            if s > Scalar[DType.float32](0.0):
                out.unsafe_store[width=head_dim](
                    (
                        acc.unsafe_load[width=head_dim](r * head_dim).cast[
                            DType.float32
                        ]()
                        * (Scalar[DType.float32](1.0) / s)
                    ).cast[dtype]()
                )
            else:
                out.unsafe_store[width=head_dim](
                    SIMD[dtype, head_dim](Scalar[dtype](0))
                )


@inline(.always)
def _dense_rows_dot[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """``_dense_rows`` with Q@K^T as bf16 dot products (AVX512-BF16 / NEON).

    Q stays bf16 and ``scale`` multiplies the logits, so Q is never rounded
    through a scaled multiply. The transposed K holds ``(k[2d], k[2d+1])`` pairs
    of one position together, so one instruction covers two head dims of LANES
    positions; P@V stays fp32 against bf16 V widened on load.
    """
    comptime NV = KT_TILE // LANES
    comptime OPN = 2 * LANES
    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[Scalar[dtype]](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[Scalar[dtype]](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var v_base = _ptr[Scalar[dtype]](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[Scalar[dtype]](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    # One broadcast pair vector per (row, head-dim pair), built once per block.
    var q_pairs = stack_allocation[qt * (head_dim // 2) * OPN, BFloat16]()
    comptime for r in range(qt):
        if r < nrows:
            var qsrc = q_base.unsafe_offset((q0 + r) * job.q_stride_t)
            comptime for dp in range(head_dim // 2):
                var q0v = Scalar[DType.bfloat16](qsrc.unsafe_load(2 * dp))
                var q1v = Scalar[DType.bfloat16](qsrc.unsafe_load(2 * dp + 1))
                q_pairs.unsafe_store[width=OPN](
                    (r * (head_dim // 2) + dp) * OPN,
                    SIMD[DType.bfloat16, LANES](q0v).interleave(
                        SIMD[DType.bfloat16, LANES](q1v)
                    ),
                )

    var acc = Array[SIMD[DType.float32, head_dim], qt](uninitialized=True)
    var row_max = Array[Float32, qt](uninitialized=True)
    var row_sum = Array[Float32, qt](uninitialized=True)
    comptime for r in range(qt):
        acc[r] = SIMD[DType.float32, head_dim](0.0)
        row_max[r] = Scalar[DType.float32](NEG_INF)
        row_sum[r] = Scalar[DType.float32](0.0)

    var probs = stack_allocation[qt * KT_TILE, DType.float32]()

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # Offset of this tile inside the (possibly blocked) K^T.
        var kbase = (t0 // job.kt_blk) * (head_dim * job.kt_blk) + (
            t0 % job.kt_blk
        ) * job.kt_stride_t
        var accs = Array[SIMD[DType.float32, LANES], qt * NV](
            uninitialized=True
        )
        comptime for i in range(qt * NV):
            accs[i] = SIMD[DType.float32, LANES](0.0)
        var dp = 0
        while dp < head_dim // 2:
            comptime for r in range(qt):
                var a = q_pairs.unsafe_load[width=OPN](
                    (r * (head_dim // 2) + dp) * OPN
                )
                comptime for v in range(NV):
                    accs[r * NV + v] = _bf16_dot[dtype](
                        accs[r * NV + v],
                        a,
                        kt_base.unsafe_load[width=OPN](
                            dp * job.kt_stride_d
                            + kbase
                            + v * OPN
                        ),
                    )
            dp += 1

        var logits = Array[SIMD[DType.float32, KT_TILE], qt](uninitialized=True)
        comptime for r in range(qt):
            var l = SIMD[DType.float32, KT_TILE](0.0)
            comptime for v in range(NV):
                comptime for i in range(LANES):
                    l[v * LANES + i] = accs[r * NV + v][i]
            logits[r] = l * job.scale

        var boundary = t0 + KT_TILE > seq_k
        if job.causal:
            boundary = t0 + KT_TILE > q0 + diag + 1
        if boundary:
            # Causal: keep t <= q0 + r + diag. Non-causal: keep t < seq_k. The
            # causal limit is always <= seq_k - 1, so one comparison covers both.
            var idx = iota[DType.int32, KT_TILE](Int32(t0))
            comptime for r in range(qt):
                var limit = Int32(seq_k)
                if job.causal:
                    # Inclusive: query row q0+r may see key q0+r+diag.
                    limit = Int32(q0 + r + diag + 1)
                logits[r] = idx.lt(SIMD[DType.int32, KT_TILE](limit)).select(
                    logits[r], SIMD[DType.float32, KT_TILE](NEG_INF)
                )

        comptime for r in range(qt):
            var m_old = row_max[r]
            var m_new = max(
                m_old, Scalar[DType.float32](logits[r].reduce_max())
            )
            var alpha = Scalar[DType.float32](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                alpha = _vexp_f(m_old - m_new)
            var p = SIMD[DType.float32, KT_TILE](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = _vexp[KT_TILE](logits[r] - m_new)
            row_sum[r] = row_sum[r] * alpha + Scalar[DType.float32](
                p.reduce_add()
            )
            row_max[r] = m_new
            acc[r] = acc[r] * alpha
            probs.unsafe_store[width=KT_TILE](r * KT_TILE, p)

        # Only the keys that exist: a short last tile has KT_TILE - jn masked
        # lanes whose probability is exactly 0, and reading their V would walk
        # past the end of the tensor (0 * garbage is garbage, not 0).
        var jn = min(KT_TILE, loop_end - t0)
        var j = 0
        while j < jn:
            var vrow = v_base.unsafe_load[width=head_dim](
                (t0 + j) * job.v_stride_t
            ).cast[DType.float32]()
            comptime for r in range(qt):
                acc[r] = fma(
                    SIMD[DType.float32, head_dim](
                        probs.unsafe_load(r * KT_TILE + j)
                    ),
                    vrow,
                    acc[r],
                )
            j += 1

        t0 += KT_TILE

    comptime for r in range(qt):
        if r < nrows:
            var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
            var s = row_sum[r]
            if s > Scalar[DType.float32](0.0):
                out.unsafe_store[width=head_dim](
                    (acc[r] * (Scalar[DType.float32](1.0) / s)).cast[dtype]()
                )
            else:
                out.unsafe_store[width=head_dim](
                    SIMD[dtype, head_dim](Scalar[dtype](0))
                )


@inline(.always)
def _exp2_fast(y: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """``2^y`` for ``y <= 0`` in about ten instructions, no libm.

    ``y = n + f`` with ``n`` the rounded integer part (``roundeven``, one
    instruction), ``2^f`` from a degree-3 minimax polynomial on ``[-0.5, 0.5]``
    and ``2^n`` by adding ``n`` to the exponent field of 1.0. Clamping at -127 makes the exponent field underflow to zero
    bits, so every masked lane (``NEG_INF``) comes out as *exactly* 0 -- which
    the online softmax needs and a saturating conversion would not give.

    Relative error ~4e-5, i.e. two orders below the bf16 probabilities this path
    already rounds to; ``std.math.exp`` costs ~22 instructions per vector here
    (range reduction plus NaN/Inf/overflow fixups) and was 20% of the whole
    dense kernel. Same trick as torch's ``fexp_u20``, which is what vLLM's ARM
    kernel calls. Callers pass ``x * scale * log2e - m * scale * log2e`` as one
    FMA, so this is the whole softmax.
    """
    comptime MAGIC = 12582912.0  # 1.5 * 2^23: add/sub rounds to nearest even
    # One fmaxnm: NaN (a masked lane whose centre is +inf) and anything below
    # -127 both land on -127, whose exponent field is zero bits, so p is 0.
    var yc = _vmax(y, SIMD[DType.float32, 4](-127.0))
    # ``roundeven`` is one instruction with 4-cycle latency; the magic-constant
    # add/sub pair it replaces is two instructions and 8 cycles on the critical
    # path of a chain that ablation puts at 30% of the call.
    var n = llvm_intrinsic[
        "llvm.roundeven.v4f32", SIMD[DType.float32, 4], has_side_effect=False
    ](yc)
    var f = yc - n
    # Degree-3 minimax on [-0.5, 0.5]: 7.5e-5 relative. Degree 2 (1.7e-3) is
    # only 1% faster and doubles the output error (7.8e-3 -> 1.6e-2 against a
    # 3e-2 tolerance), so it is not worth the margin.
    var p = SIMD[DType.float32, 4](5.5172628285e-2)
    p = p * f + SIMD[DType.float32, 4](2.4261366164e-1)
    p = p * f + SIMD[DType.float32, 4](6.9326095397e-1)
    p = p * f + SIMD[DType.float32, 4](9.9992793009e-1)
    var e = (n.cast[DType.int32]() + SIMD[DType.int32, 4](127)) << SIMD[
        DType.int32, 4
    ](23)
    return p * bitcast[DType.float32](e)


@inline(.always)
def _vmax(
    a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]
) -> SIMD[DType.float32, 4]:
    """Elementwise max in one instruction.

    ``a.lt(b).select(b, a)`` is the portable spelling (SIMD comparisons are
    *reducing*, so ``max`` cannot be used directly) but compiles to fcmgt + bsl
    -- two instructions on the critical path of the running-max chain, which is
    serial. ``llvm.maxnum`` is ``fmaxnm``: one instruction, and it ignores NaN,
    which is what lets the softmax skip the "no visible key yet" special case
    (see ``_dense_rows_bfmmla``).

    The intrinsic name carries the width, so this specialization is four lanes;
    AVX-512 wants ``llvm.maxnum.v16f32`` with ``SIMD[DType.float32, 16]`` (same
    single ``vmaxps``).
    """
    return llvm_intrinsic[
        "llvm.maxnum.v4f32", SIMD[DType.float32, 4], has_side_effect=False
    ](a, b)


@inline(.always)
def _swap_pairs(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """``(v1, v0, v3, v2)``: exchange each row's two key lanes.

    The ``bfmmla`` accumulator tile holds ``(row0 key j, row0 key j+1, row1 key
    j, row1 key j+1)``, so lanes 0-1 belong to row 0 and 2-3 to row 1. Adding
    ``v`` to its own swap therefore duplicates each row's *pair sum* into both
    of its lanes, and ``max`` does the same for the pair max -- no horizontal
    reduce, no vector-to-scalar move.
    """
    return SIMD[DType.float32, 4](v[1], v[0], v[3], v[2])


@inline(.always)
def _dense_rows_bfmmla[
    head_dim: Int, qt: Int, ks: Int, dcp: Int, sub: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """``_dense_rows`` with ``bfmmla`` on both phases (bf16 on NEON).

    Q@K^T pairs two query rows against two key positions and P@V pairs two query
    rows against two output dims, so each phase is one instruction per 16 MACs
    where fp32 needs four and ``bfdot`` two. The whole task runs in 2x2 fp32
    tiles: the logits *are* the Q@K^T accumulators, the online softmax runs on
    both rows of a pair at once (its four lanes hold row0 twice and row1 twice),
    and the probabilities are converted straight into the A panels P@V wants.

    Layouts, all built outside this kernel:

    * Q: ``[row pair][head_dim/4][8]``, eight bf16 = four head dims of row ``2p``
      then four of row ``2p+1`` (packed here, once per work item);
    * K: ``[head_dim/4][seq_k/2][8]`` -- four head dims of key ``2p``, then four
      of key ``2p+1`` (``_transpose_head`` mode 2);
    * V: ``[head_dim/2][seq_k/4][8]`` -- one head-dim pair across four keys
      (``_vpack_head``), because P@V contracts over keys;
    * P: ``[key group of 4][row pair][8]``, written from the softmax registers.
    """
    comptime RP = qt // 2  # row pairs
    comptime DG = head_dim // 4  # Q@K^T contraction groups (4 head dims)
    comptime DP = head_dim // 2  # P@V output dim pairs
    comptime NS = sub // 2  # Q@K^T column pairs per softmax step
    comptime NST = ks // sub  # softmax steps per super-tile
    comptime NKG = ks // 4  # key groups per super-tile
    comptime KGS = sub // 4  # key groups per softmax step

    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[BFloat16](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[BFloat16](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var vt_base = _ptr[BFloat16](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[BFloat16](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    # Q A panels: one per (row pair, 4 head dims), built once per work item.
    # ``[head-dim group][row pair][8]``: one group's four row-pair panels are
    # then 64 contiguous bytes and load as two `ldp`s. Laid out ``[rp][g][8]``
    # they are 128 elements apart and cost four separate loads in the innermost
    # loop of the Q@K^T sweep.
    var q_pan = stack_allocation[RP * DG * 8, BFloat16]()
    comptime for g in range(DG):
        comptime for rp in range(RP):
            var lo = SIMD[DType.bfloat16, 4](0.0)
            var hi = SIMD[DType.bfloat16, 4](0.0)
            if 2 * rp < nrows:
                lo = q_base.unsafe_offset(
                    (q0 + 2 * rp) * job.q_stride_t + 4 * g
                ).unsafe_load[width=4]()
            if 2 * rp + 1 < nrows:
                hi = q_base.unsafe_offset(
                    (q0 + 2 * rp + 1) * job.q_stride_t + 4 * g
                ).unsafe_load[width=4]()
            q_pan.unsafe_store[width=8]((g * RP + rp) * 8, lo.join(hi))

    var acc = stack_allocation[RP * DP * 4, DType.float32]()
    comptime for i in range(RP * DP):
        acc.unsafe_store[width=4](i * 4, SIMD[DType.float32, 4](0.0))
    # The online-softmax state is one 4-lane vector per row pair, in the same
    # (row0, row0, row1, row1) lane order as an accumulator tile, so the max
    # update, the rescale factor and the exponent offset are all one vector op
    # with no vector-to-scalar round trip. Keeping them scalar cost ~20% of the
    # kernel: every 16-key step did `umov` per row for the max, a scalar
    # `fmax`/`fmul`, then a `dup` to get back into a vector for the exp.
    var m_v = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
    var s_v = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
    comptime for rp in range(RP):
        m_v[rp] = SIMD[DType.float32, 4](NEG_INF)
        s_v[rp] = SIMD[DType.float32, 4](0.0)

    var probs = stack_allocation[NKG * RP * 8, BFloat16]()
    # Logits are kept in raw Q@K^T units and converted once per accumulator:
    # exp(scale*x - m) = 2^(c1*x - m*c1) with c1 = scale * log2(e).
    var c1 = Scalar[DType.float32](job.scale) * Scalar[DType.float32](
        1.4426950408889634
    )

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # ---- Q@K^T + online softmax, `sub` keys at a time. A runtime loop, not
        # a comptime one: four unrolled copies of the softmax made the register
        # allocator interleave them and spill the whole accumulator tile.
        var st = 0
        while st < NST:
            var tok = t0 + st * sub
            var half = tok // 2
            var accs = Array[SIMD[DType.float32, 4], RP * NS](
                uninitialized=True
            )
            comptime for i in range(RP * NS):
                accs[i] = SIMD[DType.float32, 4](0.0)
            # One K vector (2 key positions x 4 head dims) feeds every row pair;
            # one Q panel feeds every key pair. K^T is blocked by ``kt_blk``
            # keys, so this walks one 8 KB tile linearly instead of jumping
            # ``seq_k/2 * 16`` bytes per head-dim group (16 scattered lines per
            # sub-block, which was a 12% branch bubble at S=2048).
            var ktp = kt_base.unsafe_offset(
                (tok // job.kt_blk) * (DG * job.kt_stride_d)
                + ((tok % job.kt_blk) // 2) * 8
            )
            var g = 0
            while g < DG:
                comptime for rp in range(RP):
                    var a = q_pan.unsafe_load[width=8]((g * RP + rp) * 8)
                    comptime for v in range(NS):
                        accs[rp * NS + v] = _bfmmla(
                            accs[rp * NS + v],
                            a,
                            ktp.unsafe_load[width=8](
                                g * job.kt_stride_d + v * 8
                            ),
                        )
                g += 1

            # Causal tail (or the last tile of a non-causal call): kill the
            # lanes no row may see. Lane 0,1 of a tile are keys 2v, 2v+1 of row
            # 2rp and lanes 2,3 the same keys of row 2rp+1.
            var boundary = tok + sub > seq_k
            if job.causal:
                boundary = tok + sub > q0 + diag + 1
            if boundary:
                comptime for rp in range(RP):
                    var l0 = Int32(seq_k)
                    var l1 = Int32(seq_k)
                    if job.causal:
                        l0 = Int32(q0 + 2 * rp + diag + 1)
                        l1 = Int32(q0 + 2 * rp + 1 + diag + 1)
                    var lim = SIMD[DType.int32, 4](l0, l0, l1, l1)
                    comptime for v in range(NS):
                        var idx = SIMD[DType.int32, 4](
                            Int32(tok + 2 * v),
                            Int32(tok + 2 * v + 1),
                            Int32(tok + 2 * v),
                            Int32(tok + 2 * v + 1),
                        )
                        accs[rp * NS + v] = idx.lt(lim).select(
                            accs[rp * NS + v], SIMD[DType.float32, 4](NEG_INF)
                        )

            comptime for rp in range(RP):
                # Max over the NS accumulators as a tree: same op count, depth
                # log2(NS) instead of NS-1, and nothing feeds a scalar.
                comptime H1 = NS // 2
                var l1 = Array[SIMD[DType.float32, 4], H1](uninitialized=True)
                comptime for v in range(H1):
                    l1[v] = _vmax(
                        accs[rp * NS + 2 * v], accs[rp * NS + 2 * v + 1]
                    )
                comptime H2 = H1 // 2
                var l2 = Array[SIMD[DType.float32, 4], H2](uninitialized=True)
                comptime for v in range(H2):
                    l2[v] = _vmax(l1[2 * v], l1[2 * v + 1])
                var mx = l2[0]
                comptime for v in range(1, H2):
                    mx = _vmax(mx, l2[v])
                # Row max: (max key pair, max key pair, ...), broadcast per row.
                var m_cand = _vmax(m_v[rp], _vmax(mx, _swap_pairs(mx)))
                var gap = m_v[rp] - m_cand
                # `gap` is <= 0 in every visible lane, so "did the max move" is
                # one horizontal max and a compare -- the branch is only taken
                # when some row's max actually grew.
                var need = gap.reduce_max() < Scalar[DType.float32](0.0)
                var m_new = m_v[rp]
                var alpha = SIMD[DType.float32, 4](1.0)
                if need:
                    m_new = m_cand
                    alpha = _exp2_fast(gap * c1)
                m_v[rp] = m_new
                # scale, max subtraction and the log2 factor in one FMA per
                # accumulator, in the same lane order as the tile.
                var off = m_new * (-c1)
                var sumv = SIMD[DType.float32, 4](0.0)
                comptime for v in range(NS):
                    var p = _exp2_fast(accs[rp * NS + v] * c1 + off)
                    accs[rp * NS + v] = p
                    sumv += p
                s_v[rp] = s_v[rp] * alpha + _swap_pairs(sumv) + sumv
                if need:
                    comptime for d in range(DP):
                        acc.unsafe_store[width=4](
                            (rp * DP + d) * 4,
                            acc.unsafe_load[width=4]((rp * DP + d) * 4) * alpha,
                        )
                    # The panels already written for this super-tile are relative
                    # to the previous max, and P@V has not consumed them yet, so
                    # they need the same rescale (the accumulator holds only
                    # earlier super-tiles). Rare: the running max stops moving
                    # after the first few tiles.
                    var av8 = alpha.interleave(alpha)
                    var g = 0
                    while g < st * KGS:
                        var pv = probs.unsafe_load[width=8]((g * RP + rp) * 8)
                        probs.unsafe_store[width=8](
                            (g * RP + rp) * 8,
                            (pv.cast[DType.float32]() * av8).cast[
                                DType.bfloat16
                            ](),
                        )
                        g += 1
                # Probabilities -> A panels: four keys of row 2rp, then four of
                # row 2rp+1 (the two 4-lane halves of the instruction's A).
                comptime for kg in range(KGS):
                    var pa = accs[rp * NS + 2 * kg].cast[DType.bfloat16]()
                    var pb = accs[rp * NS + 2 * kg + 1].cast[DType.bfloat16]()
                    probs.unsafe_store[width=8](
                        ((st * KGS + kg) * RP + rp) * 8,
                        SIMD[DType.bfloat16, 8](
                            pa[0],
                            pa[1],
                            pb[0],
                            pb[1],
                            pa[2],
                            pa[3],
                            pb[2],
                            pb[3],
                        ),
                    )
            st += 1

        # ---- P@V over the super-tile, BDCP dim pairs at a time. Only the key
        # groups that exist: the masked lanes' probability is exactly 0, but
        # their V may be past the end of the packed buffer (0 * garbage is not 0).
        var jn = min(ks, seq_k - t0)
        var kgroups = (jn + 3) // 4
        comptime for dc in range(0, DP, dcp):
            var accv = Array[SIMD[DType.float32, 4], RP * dcp](
                uninitialized=True
            )
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    accv[rp * dcp + d] = acc.unsafe_load[width=4](
                        (rp * DP + dc + d) * 4
                    )
            var vbase = vt_base.unsafe_offset(
                (t0 // 4) * job.v_stride_t + dc * 8
            )
            var kg = 0
            while kg < kgroups:
                var ap = Array[SIMD[DType.bfloat16, 8], RP](uninitialized=True)
                comptime for rp in range(RP):
                    ap[rp] = probs.unsafe_load[width=8]((kg * RP + rp) * 8)
                # One pointer bump per key group, then every dim pair is a
                # compile-time offset off it: taking the stride inside the d loop
                # turned each of the `dcp` loads into a `madd`.
                var vp = vbase.unsafe_offset(kg * job.v_stride_t)
                comptime for d in range(dcp):
                    var vv = vp.unsafe_load[width=8](d * 8)
                    comptime for rp in range(RP):
                        accv[rp * dcp + d] = _bfmmla(
                            accv[rp * dcp + d], ap[rp], vv
                        )
                kg += 1
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    acc.unsafe_store[width=4](
                        (rp * DP + dc + d) * 4, accv[rp * dcp + d]
                    )

        t0 += ks

    comptime for rp in range(RP):
        comptime for c in range(2):
            var r = 2 * rp + c
            if r < nrows:
                var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
                # A row whose whole window lies at a negative key index (the
                # bottom-right aligned causal mask with seq_q > seq_k) sees no
                # key at all: its accumulator is `2^-127`-scaled garbage, so it
                # is written as zeros, as flash-attn does.
                var vis = seq_k > 0
                if job.causal:
                    vis = q0 + r + diag + 1 > 0
                var s = s_v[rp][2 * c]
                if vis and s > Scalar[DType.float32](0.0):
                    var inv = SIMD[DType.float32, 4](
                        Scalar[DType.float32](1.0) / s
                    )
                    comptime for d4 in range(DP // 4):
                        var o8 = SIMD[DType.bfloat16, 8](0.0)
                        comptime for i in range(4):
                            var t = (
                                acc.unsafe_load[width=4](
                                    (rp * DP + d4 * 4 + i) * 4
                                )
                                * inv
                            )
                            o8[2 * i] = Scalar[DType.bfloat16](t[2 * c])
                            o8[2 * i + 1] = Scalar[DType.bfloat16](t[2 * c + 1])
                        out.unsafe_store[width=8](d4 * 8, o8)
                else:
                    # Every key of this row was masked: flash-attn returns zeros.
                    comptime for d8 in range(DP // 4):
                        out.unsafe_store[width=8](
                            d8 * 8, SIMD[DType.bfloat16, 8](0.0)
                        )


comptime TILE_KS = 64
"""Keys per tile in ``_dense_rows_tile``: the whole tile's logits live in L1.

This is the vLLM structure (one QK pass over a key tile, one straight-line
softmax over the whole tile, then P@V over it) as opposed to v1's online-softmax
chain, which re-derives the max and rescales the accumulator every 16 keys. A
tile has to be a multiple of 64 because the K scratch is padded to 64 keys and
the QK pass reads whole sub-blocks.
"""

comptime KT_PAIR_STRIDE = (KS_STRIDE // 2) * 8
"""Elements between head-dim groups in the bf16 K^T layout
(``[tile][dim group][key pair][8]``): 32 key pairs x 8 bf16."""

comptime KS_STRIDE = 64
"""Keys per contiguous K^T block, as a *compile-time* stride.

The fp32 Q@K^T sweep reads one head dim's 64 keys per iteration; taking that
stride from the job turned every increment into a `madd`. 64 is what
``_kernel.kt_blk`` sets for every fast path (the x86 blocking is also 64), and
``_dense_dispatch`` falls back to the generic kernel if it ever is not.
"""

comptime TILE_NSQ = 4
"""Key *pairs* per QK sub-block: ``qt/2 * nsq`` fp32 accumulator registers.

Four row pairs x four key pairs is 16 registers, which is what lets ``qt`` grow
to 8 (the K read is amortised over the rows, so rows are the only lever on the
K traffic) without the 32-register tile spilling.
"""


@inline(.always)
def _dense_rows_tile[
    head_dim: Int, qt: Int, ks: Int, dcp: Int, nsq: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """``bfmmla`` attention with the key tile's logits in L1 (ARM bf16).

    Same arithmetic as ``_dense_rows_bfmmla`` and the same operands, but the
    structure is vLLM's rather than flash-decoding's:

    1. **QK**, ``nsq`` key pairs (``2*nsq`` keys) at a time, ``RP*nsq`` fp32
       accumulators in registers, straight to a logits buffer. The running tile
       max is folded into this pass (it is one ``fmaxnm`` per accumulator, and
       the values are already in registers).
    2. **one softmax over the whole tile**: a single ``m_new``/``alpha`` per row
       per tile instead of per 16 keys, then one pass that reads each logit,
       scales, exponentiates, accumulates the row sum and writes the bf16
       probability panel P@V consumes.
    3. **P@V** over the tile, ``dcp`` head-dim pairs at a time.

    Why: at 16 keys per online-softmax step the row max, the row sum, the
    accumulator rescale and the block mask are all paid 4-8x more often than
    they need to be, and the 32-register logit tile caps the rows per work item
    (``qt``), which is the only thing that amortises the K/V reads. Staging the
    logits in L1 costs ~4 memory ops per row-key and buys the larger row block.
    """

    comptime RP = qt // 2  # row pairs
    comptime DG = head_dim // 4  # Q@K^T contraction groups (4 head dims each)
    comptime DP = head_dim // 2  # P@V output dim pairs
    comptime NKP = ks // 2  # key pairs per tile
    comptime NKG = ks // 4  # key groups (4 keys) per tile

    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[BFloat16](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[BFloat16](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var vt_base = _ptr[BFloat16](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[BFloat16](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    # Q A panels: [row pair][head-dim group of 4][8] -- four head dims of the
    # even row then four of the odd row, built once per work item.
    var q_pan = stack_allocation[RP * DG * 8, BFloat16]()
    comptime for rp in range(RP):
        comptime for g in range(DG):
            var lo = SIMD[DType.bfloat16, 4](0.0)
            var hi = SIMD[DType.bfloat16, 4](0.0)
            if 2 * rp < nrows:
                lo = q_base.unsafe_offset(
                    (q0 + 2 * rp) * job.q_stride_t + 4 * g
                ).unsafe_load[width=4]()
            if 2 * rp + 1 < nrows:
                hi = q_base.unsafe_offset(
                    (q0 + 2 * rp + 1) * job.q_stride_t + 4 * g
                ).unsafe_load[width=4]()
            q_pan.unsafe_store[width=8]((g * RP + rp) * 8, lo.join(hi))

    # Output accumulator: [row pair][dim pair][4], in L1 like v1's.
    var acc = stack_allocation[RP * DP * 4, DType.float32]()
    comptime for i in range(RP * DP):
        acc.unsafe_store[width=4](i * 4, SIMD[DType.float32, 4](0.0))

    # The tile's logits [key pair][row pair][4 fp32] and its bf16 probability
    # panels [key group][row pair][8]. Runtime-indexed, so they stay in L1.
    var logits = stack_allocation[ks * qt, DType.float32]()
    var probs = stack_allocation[NKG * RP * 8, BFloat16]()

    var m_v = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
    var s_v = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
    comptime for rp in range(RP):
        m_v[rp] = SIMD[DType.float32, 4](NEG_INF)
        s_v[rp] = SIMD[DType.float32, 4](0.0)

    # exp(scale*x - m) = 2^(c1*x - c1*m); logits stay in raw Q@K^T units.
    var c1 = Scalar[DType.float32](job.scale) * Scalar[DType.float32](
        1.4426950408889634
    )

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        var kt_tile = kt_base.unsafe_offset(
            (t0 // job.kt_blk) * (DG * job.kt_stride_d)
            + ((t0 % job.kt_blk) // 2) * 8
        )
        # (recomputed per sub-block below: a tile wider than ``kt_blk`` would
        # step out of its key-pair run into the next head-dim group.)
        # ---- 1. Q@K^T into the logits buffer, and the tile max on the way.
        var mt = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
        comptime for rp in range(RP):
            mt[rp] = SIMD[DType.float32, 4](NEG_INF)
        # Only the key sub-blocks that exist: the last row of the block sees no
        # key past ``t0 + jn``, so everything beyond is masked to NEG_INF anyway
        # and the whole sub-block (Q@K^T, max, exp, P@V) is wasted work. At
        # S=128 that waste was ~1.5x of the called-for work; at S>=2048 it is
        # noise.
        # (Up to the *causal* limit of the block's last row, not seq_k: past
        # ``loop_end`` every row of the block is masked anyway.)
        var jn = min(ks, loop_end - t0)
        var nkp = min(NKP, (jn + 2 * nsq - 1) // (2 * nsq) * nsq)
        # K^T is blocked by ``kt_blk`` (= the tile size here), so the whole
        # tile's K is one contiguous run and a sub-block reads it linearly.
        var sb = 0
        while sb < nkp:
            var kt_sub = kt_base.unsafe_offset(
                ((t0 + 2 * sb) // job.kt_blk) * (DG * job.kt_stride_d)
                + (((t0 + 2 * sb) % job.kt_blk) // 2) * 8
            )
            var accs = Array[SIMD[DType.float32, 4], RP * nsq](
                uninitialized=True
            )
            # Peel g=0 so the zero accumulator is one shared register: the
            # obvious `accs[i] = 0` loop emitted RP*nsq separate `movi`s per
            # sub-block.
            var zero = SIMD[DType.float32, 4](0.0)
            comptime for rp in range(RP):
                var a0 = q_pan.unsafe_load[width=8]((0 * RP + rp) * 8)
                comptime for v in range(nsq):
                    accs[rp * nsq + v] = _bfmmla(
                        zero,
                        a0,
                        kt_sub.unsafe_load[width=8](v * 8),
                    )
            var g = 1
            while g < DG:
                comptime for rp in range(RP):
                    var a = q_pan.unsafe_load[width=8]((g * RP + rp) * 8)
                    comptime for v in range(nsq):
                        accs[rp * nsq + v] = _bfmmla(
                            accs[rp * nsq + v],
                            a,
                            kt_sub.unsafe_load[width=8](
                                g * KT_PAIR_STRIDE + v * 8
                            ),
                        )
                g += 1

            # Causal tail: kill the lanes no row may see, in the accumulators so
            # the tile max and the exp pass both see NEG_INF (exp2 of which is
            # exactly 0 after the clamp). Lane 0,1 are keys 2v,2v+1 of row 2rp
            # and lanes 2,3 the same keys of row 2rp+1.
            var tok = t0 + 2 * sb
            var boundary = tok + 2 * nsq > seq_k
            if job.causal:
                boundary = tok + 2 * nsq > q0 + diag + 1
            if boundary:
                comptime for rp in range(RP):
                    var l0 = Int32(seq_k)
                    var l1 = Int32(seq_k)
                    if job.causal:
                        l0 = Int32(q0 + 2 * rp + diag + 1)
                        l1 = Int32(q0 + 2 * rp + 1 + diag + 1)
                    var lim = SIMD[DType.int32, 4](l0, l0, l1, l1)
                    comptime for v in range(nsq):
                        var idx = SIMD[DType.int32, 4](
                            Int32(tok + 2 * v),
                            Int32(tok + 2 * v + 1),
                            Int32(tok + 2 * v),
                            Int32(tok + 2 * v + 1),
                        )
                        accs[rp * nsq + v] = idx.lt(lim).select(
                            accs[rp * nsq + v], SIMD[DType.float32, 4](NEG_INF)
                        )

            comptime for rp in range(RP):
                var mx = accs[rp * nsq]
                comptime for v in range(1, nsq):
                    mx = _vmax(mx, accs[rp * nsq + v])
                # One fmaxnm per sub-block: the pair max duplicated over each
                # row's two key lanes, folded into the tile max.
                mt[rp] = _vmax(mt[rp], _vmax(mx, _swap_pairs(mx)))
                comptime for v in range(nsq):
                    logits.unsafe_store[width=4](
                        ((sb + v) * RP + rp) * 4, accs[rp * nsq + v]
                    )
            sb += nsq

        # ---- 2. One softmax update for the whole tile, then exp + panels.
        # Key groups that exist, rounded up to a multiple of two (the exp loop
        # consumes two per iteration) and never past the sub-blocks the QK pass
        # actually filled.
        var nkg = min(NKG, (jn + 7) // 8 * 2)
        comptime for rp in range(RP):
            # Does this tile show this row pair anything at all? False only in
            # the last tile of a block whose early rows' window ends earlier.
            var has = mt[rp].reduce_max() > Scalar[DType.float32](NEG_INF)
            var m_new = m_v[rp]
            var alpha = SIMD[DType.float32, 4](1.0)
            if has:
                m_new = _vmax(m_v[rp], mt[rp])
                # Degree-5 exp2, four lanes: an error in `alpha` does not cancel
                # between the accumulator and the row sum (the terms added
                # afterwards are not scaled), so it accumulates over tiles -- and
                # the two rows of the pair have their own max, so this has to stay
                # a vector.
                alpha = _exp2_g[4]((m_v[rp] - m_new) * c1)
                m_v[rp] = m_new
                # The accumulator (and the row sum) are relative to the old max.
                if s_v[rp].reduce_max() > Scalar[DType.float32](0.0):
                    comptime for d in range(DP):
                        acc.unsafe_store[width=4](
                            (rp * DP + d) * 4,
                            acc.unsafe_load[width=4]((rp * DP + d) * 4) * alpha,
                        )
                    s_v[rp] = s_v[rp] * alpha
            # One scalar-vector offset per row pair, and a *runtime* kg loop:
            # 16 unrolled copies of the polynomial produced a 4.4k-instruction
            # tile body that spilled (the compiler interleaved the chains) and
            # did not fit the I-cache.
            var off = m_new * (-c1)
            # One partial sum per key group of the two the loop consumes, folded
            # once: a single `sumv` is a 16-long serial `fadd` chain per row pair
            # per tile, and it showed up as 5% of the backend stalls.
            var sumv = Array[SIMD[DType.float32, 4], 2](uninitialized=True)
            sumv[0] = SIMD[DType.float32, 4](0.0)
            sumv[1] = SIMD[DType.float32, 4](0.0)
            # Nothing visible yet anywhere: the panels have to be exactly zero,
            # because -inf * -c1 is +inf and would exponentiate to 1, not 0.
            var blind = m_new.reduce_max() == Scalar[DType.float32](NEG_INF)
            var kg = 0
            if not has and blind:
                while kg < nkg:
                    probs.unsafe_store[width=8](
                        (kg * RP + rp) * 8, SIMD[DType.bfloat16, 8](0.0)
                    )
                    kg += 1
            else:
                # Two key groups per iteration: the polynomial is a ~10-deep
                # dependency chain and two of them per iteration (32 cycles of
                # latency against ~20 instructions of work) does not fill the
                # pipeline; four in flight does.
                while kg < nkg:
                    comptime for u in range(2):
                        var p0 = _exp2_fast(
                            logits.unsafe_load[width=4](
                                (kg + u) * (2 * RP) * 4 + rp * 4
                            )
                            * c1
                            + off
                        )
                        var p1 = _exp2_fast(
                            logits.unsafe_load[width=4](
                                ((kg + u) * 2 + 1) * RP * 4 + rp * 4
                            )
                            * c1
                            + off
                        )
                        sumv[u] += p0 + p1
                        # Row 0's four keys are lanes 0,1 of both exponentials
                        # and row 1's are lanes 2,3, so a panel is two 4-lane
                        # half-shuffles, two truncating converts and a join.
                        # The 8-lane constructor this replaces compiled to zip2
                        # plus a scalar lane insert (`mov v.h[7]`) per group.
                        var z0 = SIMD[DType.float32, 4](
                            p0[0], p0[1], p1[0], p1[1]
                        )
                        var z1 = SIMD[DType.float32, 4](
                            p0[2], p0[3], p1[2], p1[3]
                        )
                        probs.unsafe_store[width=8](
                            ((kg + u) * RP + rp) * 8,
                            z0.cast[DType.bfloat16]().join(
                                z1.cast[DType.bfloat16]()
                            ),
                        )
                    kg += 2
            var st = sumv[0] + sumv[1]
            s_v[rp] = s_v[rp] + _swap_pairs(st) + st

        # ---- 3. P@V over the tile, dcp dim pairs at a time. Only the key
        # groups that exist (``nkg``, decided in phase 1): masked lanes are
        # exactly 0 but their V may be past the end of the packed buffer
        # (0 * garbage is not 0).
        var kgroups = nkg
        comptime for dc in range(0, DP, dcp):
            var accv = Array[SIMD[DType.float32, 4], RP * dcp](
                uninitialized=True
            )
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    accv[rp * dcp + d] = acc.unsafe_load[width=4](
                        (rp * DP + dc + d) * 4
                    )
            var vbase = vt_base.unsafe_offset(
                (t0 // 4) * job.v_stride_t + dc * 8
            )
            var kg = 0
            while kg < kgroups:
                var ap = Array[SIMD[DType.bfloat16, 8], RP](uninitialized=True)
                comptime for rp in range(RP):
                    ap[rp] = probs.unsafe_load[width=8]((kg * RP + rp) * 8)
                comptime for d in range(dcp):
                    var vv = vbase.unsafe_load[width=8](
                        kg * job.v_stride_t + d * 8
                    )
                    comptime for rp in range(RP):
                        accv[rp * dcp + d] = _bfmmla(
                            accv[rp * dcp + d], ap[rp], vv
                        )
                kg += 1
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    acc.unsafe_store[width=4](
                        (rp * DP + dc + d) * 4, accv[rp * dcp + d]
                    )

        t0 += ks

    comptime for rp in range(RP):
        comptime for c in range(2):
            var r = 2 * rp + c
            if r < nrows:
                var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
                var vis = seq_k > 0
                if job.causal:
                    vis = q0 + r + diag + 1 > 0
                var s = s_v[rp][2 * c]
                if vis and s > Scalar[DType.float32](0.0):
                    var inv = SIMD[DType.float32, 4](
                        Scalar[DType.float32](1.0) / s
                    )
                    comptime for d4 in range(DP // 4):
                        var o8 = SIMD[DType.bfloat16, 8](0.0)
                        comptime for i in range(4):
                            var t = (
                                acc.unsafe_load[width=4](
                                    (rp * DP + d4 * 4 + i) * 4
                                )
                                * inv
                            )
                            o8[2 * i] = Scalar[DType.bfloat16](t[2 * c])
                            o8[2 * i + 1] = Scalar[DType.bfloat16](t[2 * c + 1])
                        out.unsafe_store[width=8](d4 * 8, o8)
                else:
                    comptime for d8 in range(DP // 4):
                        out.unsafe_store[width=8](
                            d8 * 8, SIMD[DType.bfloat16, 8](0.0)
                        )


@inline(.always)
def _dense_rows_tile_f32[
    head_dim: Int, qt: Int, ks: Int, kb: Int, dc: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """fp32 attention with the key tile's logits in L1 (the ARM fp32 path).

    Same structure as ``_dense_rows_tile`` (which is the bf16/``bfmmla`` one) and
    for the same reasons: the old fp32 path was ``_dense_rows_mem``, i.e. an
    online softmax every 16 keys, and ``perf stat`` put it at **72.7
    instructions per row-key** against vLLM's 51.6 -- the max tree, the
    accumulator rescale, the product-buffer reload per head-dim chunk and the
    mask were all paid 4x more often than the arithmetic needs.

    fp32 has no bfmmla, so the arithmetic is 16 FMA per row-key for Q@K^T and 16
    for P@V (four 4-lane MACs each) and the whole game is issuing as few other
    instructions as possible: this is 46, against a hard floor of 32.

    * Q is transposed to ``[head_dim][qt]`` once per work item so one 4-lane load
      broadcasts four rows' q[d] with no scalar loads (the fma becomes
      ``fmla vD.4s, vN.4s, vM.s[k]``).
    * Q@K^T walks ``kb`` keys (two vectors) at a time into ``qt*kb/4`` registers.
    * one softmax per 64-key tile: one max, one alpha, one accumulator rescale.
    * P@V walks a 4-key group at a time and reads the probability vector of each
      row once per (group, head-dim chunk), using the four lanes as four keys --
      so the probabilities are never re-read as scalars, which is what the old
      path did eight times per key.
    """

    comptime NV = head_dim // LANES  # vectors per row of the output accumulator
    comptime KV = kb // LANES  # vectors per Q@K^T key sub-block
    comptime DV = dc // LANES  # vectors per P@V head-dim chunk
    comptime NKB = ks // kb  # Q@K^T sub-blocks per tile
    comptime NCH = head_dim // dc  # P@V head-dim chunks

    var seq_k = job.seq_k
    var nrows = min(qt, job.seq_q - q0)
    var diag = seq_k - job.seq_q
    var q_base = _ptr[Scalar[DType.float32]](job.q).unsafe_offset(
        batch * job.q_stride_b + head * job.q_stride_h
    )
    var kt_base = _ptr[Scalar[DType.float32]](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + kv_head) * job.kt_stride_bh
    )
    var v_base = _ptr[Scalar[DType.float32]](job.v).unsafe_offset(
        batch * job.v_stride_b + kv_head * job.v_stride_h
    )
    var o_base = _ptr[Scalar[DType.float32]](job.dst).unsafe_offset(
        batch * job.o_stride_b + head * job.o_stride_h
    )

    # Q transposed to [head_dim][qt], scale folded in: one 4-lane load then feeds
    # four rows with a lane-indexed fma.
    # `scale * log2(e)` folded into Q, so every exponential below is a bare
    # exp2 with no per-element multiply.
    var c1 = Scalar[DType.float32](job.scale) * Scalar[DType.float32](
        1.4426950408889634
    )
    var q_t = stack_allocation[head_dim * qt, DType.float32]()
    var rr = 0
    while rr < qt:
        if rr < nrows:
            var src = q_base.unsafe_offset((q0 + rr) * job.q_stride_t)
            var d = 0
            while d < head_dim:
                var v = src.unsafe_load[width=LANES](d) * c1
                comptime for i in range(LANES):
                    q_t.unsafe_store((d + i) * qt + rr, v[i])
                d += LANES
        rr += 1

    var acc = stack_allocation[qt * head_dim, DType.float32]()
    comptime for i in range(qt * NV):
        acc.unsafe_store[width=LANES](i * LANES, SIMD[DType.float32, LANES](0.0))
    var logits = stack_allocation[qt * ks, DType.float32]()
    var probs = stack_allocation[qt * ks, DType.float32]()

    var row_max = Array[Scalar[DType.float32], qt](uninitialized=True)
    var row_sum = Array[SIMD[DType.float32, LANES], qt](uninitialized=True)
    comptime for r in range(qt):
        row_max[r] = Scalar[DType.float32](NEG_INF)
        row_sum[r] = SIMD[DType.float32, LANES](0.0)

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # ---- 1. Q@K^T into the logits buffer, with the tile max on the way.
        var mvec = Array[SIMD[DType.float32, LANES], qt](uninitialized=True)
        comptime for r in range(qt):
            mvec[r] = SIMD[DType.float32, LANES](NEG_INF)
        # Up to the causal limit of the block's last row (see the bf16 kernel).
        var jn = min(ks, loop_end - t0)
        var nkb = min(NKB, (jn + kb - 1) // kb)
        var sb = 0
        while sb < nkb:
            var ko = sb * kb
            var kt_sub = kt_base.unsafe_offset(
                ((t0 + ko) // job.kt_blk) * (head_dim * job.kt_blk)
                + ((t0 + ko) % job.kt_blk)
            )
            var accs = Array[SIMD[DType.float32, LANES], qt * KV](
                uninitialized=True
            )
            var zero = SIMD[DType.float32, LANES](0.0)
            comptime for i in range(qt * KV):
                accs[i] = zero
            var d = 0
            while d < head_dim:
                var kp = kt_sub.unsafe_offset(d * KS_STRIDE)
                var kv = Array[SIMD[DType.float32, LANES], KV](
                    uninitialized=True
                )
                comptime for v in range(KV):
                    kv[v] = kp.unsafe_load[width=LANES](v * LANES)
                var qp = q_t.unsafe_offset(d * qt)
                comptime for h in range(qt // LANES):
                    var qv = qp.unsafe_load[width=LANES](h * LANES)
                    comptime for i in range(LANES):
                        comptime for v in range(KV):
                            accs[(h * LANES + i) * KV + v] += (
                                kv[v] * SIMD[DType.float32, LANES](qv[i])
                            )
                d += 1

            # Causal tail: NEG_INF in the lanes no row may see (exp of it is 0).
            var boundary = t0 + ko + kb > seq_k
            if job.causal:
                boundary = t0 + ko + kb > q0 + diag + 1
            if boundary:
                comptime for r in range(qt):
                    var lim = Int32(seq_k)
                    if job.causal:
                        lim = Int32(q0 + r + diag + 1)
                    comptime for v in range(KV):
                        var idx = iota[DType.int32, LANES](
                            Int32(t0 + ko + v * LANES)
                        )
                        accs[r * KV + v] = idx.lt(
                            SIMD[DType.int32, LANES](lim)
                        ).select(
                            accs[r * KV + v], SIMD[DType.float32, LANES](NEG_INF)
                        )

            comptime for r in range(qt):
                var mx = accs[r * KV]
                comptime for v in range(1, KV):
                    mx = mx.lt(accs[r * KV + v]).select(
                        accs[r * KV + v], mx
                    )
                mvec[r] = mvec[r].lt(mx).select(mx, mvec[r])
                comptime for v in range(KV):
                    logits.unsafe_store[width=LANES](
                        r * ks + ko + v * LANES, accs[r * KV + v]
                    )
            sb += 1

        # ---- 2. One softmax step for the tile, then exp + the row sums.
        comptime for r in range(qt):
            var m_cand = mvec[r].reduce_max()
            var m_new = row_max[r]
            var alpha = Scalar[DType.float32](1.0)
            var has = m_cand > Scalar[DType.float32](NEG_INF)
            if has:
                m_new = max(m_new, m_cand)
                alpha = _exp2f(Scalar[DType.float32](row_max[r] - m_new))
                row_max[r] = m_new
            # `q_t` carries `scale * log2(e)`, so the logits are already log2
            # units: the exp argument is logit - max, one fadd.
            var off = Scalar[DType.float32](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                off = -m_new
            if has:
                if row_sum[r].reduce_max() > Scalar[DType.float32](0.0):
                    comptime for i in range(NV):
                        acc.unsafe_store[width=LANES](
                            r * head_dim + i * LANES,
                            acc.unsafe_load[width=LANES](
                                r * head_dim + i * LANES
                            )
                            * SIMD[DType.float32, LANES](alpha),
                        )
                    row_sum[r] *= SIMD[DType.float32, LANES](alpha)
            # Two partial sums, folded once: one accumulator is a 16-long
            # serial `fadd` chain per row per tile (the same fix as the bf16
            # kernel, where it was worth 8% of the whole call).
            var sumv = Array[SIMD[DType.float32, LANES], 2](uninitialized=True)
            sumv[0] = SIMD[DType.float32, LANES](0.0)
            sumv[1] = SIMD[DType.float32, LANES](0.0)
            var k = 0
            # Only the keys that exist, rounded up to a whole 4-key group (the
            # logits past `jn` are NEG_INF, so exp of them is exactly 0).
            var kend = min(ks, (jn + 3) // 4 * 4)
            while k < kend:
                var p = _exp2_g[LANES](
                    logits.unsafe_load[width=LANES](r * ks + k)
                    + SIMD[DType.float32, LANES](off)
                )
                sumv[(k // LANES) % 2] += p
                probs.unsafe_store[width=LANES](r * ks + k, p)
                k += LANES
            row_sum[r] += sumv[0] + sumv[1]

        # ---- 3. P@V over the tile, `dc` floats of the head dim at a time, four
        # keys at a time. Only the key groups that exist: the masked lanes'
        # probability is exactly 0, but their V may be past the end of the
        # tensor and 0 * garbage is not 0.
        var kgroups = (jn + 3) // 4
        comptime for c in range(NCH):
            var accv = Array[SIMD[DType.float32, LANES], qt * DV](
                uninitialized=True
            )
            comptime for r in range(qt):
                comptime for v in range(DV):
                    accv[r * DV + v] = acc.unsafe_load[width=LANES](
                        r * head_dim + c * dc + v * LANES
                    )
            var kg = 0
            while kg < kgroups:
                var pv = Array[SIMD[DType.float32, LANES], qt](
                    uninitialized=True
                )
                # One base per key group: `r * ks` is then a compile-time offset,
                # instead of an address computation per row.
                var pp = probs.unsafe_offset(kg * 4)
                comptime for r in range(qt):
                    pv[r] = pp.unsafe_load[width=LANES](r * ks)
                # The four keys of the group are the four *lanes* of the
                # probability vector, so this has to be comptime: a runtime
                # lane index would spill the vector to the stack.
                var vg = v_base.unsafe_offset(
                    (t0 + kg * 4) * job.v_stride_t + c * dc
                )
                comptime for kgi in range(4):
                    var vv = Array[SIMD[DType.float32, LANES], DV](
                        uninitialized=True
                    )
                    comptime for v in range(DV):
                        vv[v] = vg.unsafe_load[width=LANES](v * LANES)
                    vg = vg.unsafe_offset(job.v_stride_t)
                    comptime for r in range(qt):
                        comptime for v in range(DV):
                            accv[r * DV + v] += vv[v] * SIMD[DType.float32, LANES](
                                pv[r][kgi]
                            )
                kg += 1
            comptime for r in range(qt):
                comptime for v in range(DV):
                    acc.unsafe_store[width=LANES](
                        r * head_dim + c * dc + v * LANES, accv[r * DV + v]
                    )

        t0 += ks

    comptime for r in range(qt):
        if r < nrows:
            var out = o_base.unsafe_offset((q0 + r) * job.o_stride_t)
            var vis = seq_k > 0
            if job.causal:
                vis = q0 + r + diag + 1 > 0
            var s = row_sum[r].reduce_add()
            if vis and s > Scalar[DType.float32](0.0):
                var inv = SIMD[DType.float32, LANES](
                    Scalar[DType.float32](1.0) / s
                )
                comptime for i in range(NV):
                    out.unsafe_store[width=LANES](
                        i * LANES,
                        acc.unsafe_load[width=LANES](r * head_dim + i * LANES)
                        * inv,
                    )
            else:
                comptime for i in range(NV):
                    out.unsafe_store[width=LANES](
                        i * LANES, SIMD[DType.float32, LANES](0.0)
                    )


comptime TBLK = 16"""Tokens transposed at a time: 16 floats / 32 bf16 is one 64-byte store, so the
destination is written a cache line at a time instead of 4 bytes at a time (which
is what ``torch.permute(...).contiguous()`` does here -- 5-15 GB/s)."""

comptime WIDE_TP = LANES >= 16
"""Whether the transpose runs as 16x16 register tiles (AVX-512) or through L1.

Both variants move the same bytes; they differ in what the *source* read costs.
Gathering one head dim of 16 tokens is 16 scalar loads ``head_dim * sizeof(T)``
apart, i.e. 16 different cache lines, and the old loop did that once per head
dim -- so every source line was re-read ``head_dim`` times and the pass moved
``head_dim`` times the tensor through L2/L3 (at head_dim 128 / S=512 / H=32 that
measured 1.0 ms for 16.8 MB, 16 GB/s). The register form loads whole 64-byte
rows once and permutes them into columns, so DRAM traffic is one read plus one
write and the instruction count drops from 17 per 64-byte store to 6."""


@inline(.always)
def _tr16_s0[dt: DType](
    m: Array[SIMD[dt, 16], 16]
) -> Array[SIMD[dt, 16], 16]:
    """16x16 transpose, stage 0 of 4: swap row bit 0 with lane bit 0.

    Stage ``k`` swaps bit ``k`` of the row index with bit ``k`` of the lane
    index; the product of the four stages is the transpose. Each stage is one
    ``shuffle[mask](a, b)`` per output row (a permute of the 32-lane
    concatenation of the two source rows), i.e. 64 permutes per 1 KB tile.
    """
    var o = Array[SIMD[dt, 16], 16](uninitialized=True)
    comptime for i in range(8):
        var a = m[2 * i]
        var b = m[2 * i + 1]
        o[2 * i] = a.shuffle[
            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30
        ](b)
        o[2 * i + 1] = a.shuffle[
            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
        ](b)
    return o.copy()


@inline(.always)
def _tr16_s1[dt: DType](
    m: Array[SIMD[dt, 16], 16]
) -> Array[SIMD[dt, 16], 16]:
    """16x16 transpose, stage 1: swap row bit 1 with lane bit 1."""
    var o = Array[SIMD[dt, 16], 16](uninitialized=True)
    comptime for i in range(16):
        comptime if (i >> 1) & 1 == 0:
            comptime b = i | 2
            o[i] = m[i].shuffle[
                0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
            ](m[b])
            o[b] = m[i].shuffle[
                2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
            ](m[b])
    return o.copy()


@inline(.always)
def _tr16_s2[dt: DType](
    m: Array[SIMD[dt, 16], 16]
) -> Array[SIMD[dt, 16], 16]:
    """16x16 transpose, stage 2: swap row bit 2 with lane bit 2."""
    var o = Array[SIMD[dt, 16], 16](uninitialized=True)
    comptime for i in range(16):
        comptime if (i >> 2) & 1 == 0:
            comptime b = i | 4
            o[i] = m[i].shuffle[
                0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27
            ](m[b])
            o[b] = m[i].shuffle[
                4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31
            ](m[b])
    return o.copy()


@inline(.always)
def _tr16_s3[dt: DType](
    m: Array[SIMD[dt, 16], 16]
) -> Array[SIMD[dt, 16], 16]:
    """16x16 transpose, stage 3: swap row bit 3 with lane bit 3."""
    var o = Array[SIMD[dt, 16], 16](uninitialized=True)
    comptime for i in range(8):
        var a = m[i]
        var b = m[i + 8]
        o[i] = a.shuffle[
            0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
        ](b)
        o[i + 8] = a.shuffle[
            8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
        ](b)
    return o.copy()


@inline(.always)
def _tr16[dt: DType](m: Array[SIMD[dt, 16], 16]) -> Array[SIMD[dt, 16], 16]:
    """Transpose a 16x16 tile held in registers (rows in, rows out)."""
    return _tr16_s3(_tr16_s2(_tr16_s1(_tr16_s0(m))))


struct TransposeJob(Copyable, ImplicitlyCopyable, Movable):
    """K -> K^T (the Q@K^T sweep's layout) and V -> the P@V operand layout.

    ``v``/``vt`` and the ``v*``/``vt*`` strides are only read in mode 2 (the
    ``bfmmla`` path, whose P@V contracts over keys and so needs V position-major
    too).
    """

    var k: Int
    var kt: Int
    var v: Int
    var vt: Int
    var batch: Int
    var num_kv_heads: Int
    var seq_k: Int
    var head_dim: Int
    var k_stride_b: Int
    var k_stride_t: Int
    var k_stride_h: Int
    var kt_stride_bh: Int
    var kt_stride_d: Int
    var kt_blk: Int
    var v_stride_b: Int
    var v_stride_t: Int
    var v_stride_h: Int
    var vt_stride_bh: Int
    var vt_stride_d: Int

    def __init__(out self):
        self.k = 0
        self.kt = 0
        self.v = 0
        self.vt = 0
        self.batch = 0
        self.num_kv_heads = 0
        self.seq_k = 0
        self.head_dim = 0
        self.k_stride_b = 0
        self.k_stride_t = 0
        self.k_stride_h = 0
        self.kt_stride_bh = 0
        self.kt_stride_d = 0
        self.kt_blk = 0
        self.v_stride_b = 0
        self.v_stride_t = 0
        self.v_stride_h = 0
        self.vt_stride_bh = 0
        self.vt_stride_d = 0


@inline(.always)
def _transpose_head[
    dtype: DType, head_dim: Int, mode: Int
](job: TransposeJob, batch: Int, head: Int):
    """Copy one (batch, kv head) of K into its transposed slot.

    Mode 0 is one head dim per row, ``[head_dim, seq_k]``; mode 1 is the bf16
    dot layout ``[head_dim/2, seq_k, 2]`` (the two head dims a dot instruction
    contracts, next to the position they belong to); mode 2 is the ``bfmmla``
    layout ``[head_dim/4, seq_k/2, 8]`` -- four head dims of key ``2p``, then the
    same four of key ``2p+1``, which is the "2 columns x 4 contraction" operand
    Q@K^T wants when the contraction runs over head dims.
    """
    var src = _ptr[Scalar[dtype]](job.k).unsafe_offset(
        batch * job.k_stride_b + head * job.k_stride_h
    )
    var dst = _ptr[Scalar[dtype]](job.kt).unsafe_offset(
        (batch * job.num_kv_heads + head) * job.kt_stride_bh
    )
    var t0 = 0
    while t0 < job.seq_k:
        var n = min(TBLK, job.seq_k - t0)
        comptime if mode == 0:
            comptime if WIDE_TP:
                # 16 tokens x 16 head dims per register tile: whole rows in,
                # whole columns out, so the source is read once (see WIDE_TP).
                var d0 = 0
                while d0 < head_dim:
                    var rows = Array[SIMD[dtype, 16], 16](uninitialized=True)
                    comptime for i in range(TBLK):
                        if i < n:
                            rows[i] = src.unsafe_load[width=16](
                                (t0 + i) * job.k_stride_t + d0
                            )
                        else:
                            rows[i] = SIMD[dtype, 16](Scalar[dtype](0))
                    var cols = _tr16[dtype](rows)
                    comptime for d in range(16):
                        dst.unsafe_store[width=16](
                            (t0 // job.kt_blk) * (head_dim * job.kt_blk)
                            + (d0 + d) * job.kt_stride_d
                            + t0 % job.kt_blk,
                            cols[d],
                        )
                    d0 += 16
            else:
                # Lanes narrower than the tile: stage 16 whole rows in L1, then
                # gather each column out of it. Every source line is read once
                # (the old form re-read it once per head dim) and the gathers
                # are L1 hits.
                var tile = stack_allocation[TBLK * head_dim, dtype]()
                comptime for i in range(TBLK):
                    if i < n:
                        tile.unsafe_store[width=head_dim](
                            i * head_dim,
                            src.unsafe_load[width=head_dim](
                                (t0 + i) * job.k_stride_t
                            ),
                        )
                comptime for d in range(head_dim):
                    var v = SIMD[dtype, TBLK](Scalar[dtype](0))
                    comptime for i in range(TBLK):
                        v[i] = tile.unsafe_load(i * head_dim + d)
                    dst.unsafe_store[width=TBLK](
                        (t0 // job.kt_blk) * (head_dim * job.kt_blk)
                        + d * job.kt_stride_d
                        + t0 % job.kt_blk,
                        v,
                    )
        elif mode == 1:
            comptime if WIDE_TP:
                # Same tile, but the element is a head-dim *pair* (two adjacent
                # bf16), so one 16-lane register is a 64-byte run of a token.
                # The pairs are addressed as int32, since a 16-lane bitcast of a
                # 32-lane bf16 value is not a thing this compiler accepts.
                var src32 = _ptr[Int32](Int(src))
                var dst32 = _ptr[Int32](Int(dst))
                var d0 = 0
                while d0 < head_dim // 2:
                    var rows = Array[SIMD[DType.int32, 16], 16](
                        uninitialized=True
                    )
                    comptime for i in range(TBLK):
                        if i < n:
                            rows[i] = src32.unsafe_load[width=16](
                                ((t0 + i) * job.k_stride_t) // 2 + d0
                            )
                        else:
                            rows[i] = SIMD[DType.int32, 16](0)
                    var cols = _tr16[DType.int32](rows)
                    comptime for d in range(16):
                        dst32.unsafe_store[width=16](
                            (
                                (t0 // job.kt_blk) * (head_dim * job.kt_blk)
                                + (d0 + d) * job.kt_stride_d
                                + (t0 % job.kt_blk) * 2
                            )
                            // 2,
                            cols[d],
                        )
                    d0 += 16
            else:
                comptime for d2 in range(head_dim // 2):
                    var v = SIMD[dtype, 2 * TBLK](0.0)
                    comptime for i in range(TBLK):
                        if i < n:
                            var row = src.unsafe_offset(
                                (t0 + i) * job.k_stride_t + 2 * d2
                            )
                            v[2 * i] = row.unsafe_load(0)
                            v[2 * i + 1] = row.unsafe_load(1)
                    dst.unsafe_store[width=2 * TBLK](
                        (t0 // job.kt_blk) * (head_dim * job.kt_blk)
                        + d2 * job.kt_stride_d
                        + (t0 % job.kt_blk) * 2,
                        v,
                    )
        else:
            # 16 tokens = 8 key pairs = 64 bf16 per head-dim group: one 128-byte
            # store, 4 head dims of one key per 8-byte load. Blocked by key
            # tile: ``[tile][dim group][key pair][8]``, so one tile's K for every
            # dim group is contiguous (16 allocations instead of 16 strided runs
            # when the QK sweep walks the groups).
            comptime for g in range(head_dim // 4):
                var v = SIMD[dtype, 4 * TBLK](0.0)
                comptime for i in range(TBLK):
                    if i < n:
                        var q4 = src.unsafe_offset(
                            (t0 + i) * job.k_stride_t + 4 * g
                        ).unsafe_load[width=4]()
                        comptime for d in range(4):
                            v[4 * i + d] = q4[d]
                dst.unsafe_store[width=4 * TBLK](
                    (t0 // job.kt_blk) * ((head_dim // 4) * job.kt_stride_d)
                    + g * job.kt_stride_d
                    + (t0 % job.kt_blk) * 4,
                    v,
                )
        t0 += TBLK


@inline(.always)
def _vcopy_head[
    dtype: DType, head_dim: Int
](job: TransposeJob, batch: Int, head: Int):
    """Copy one (batch, kv head) of V so its tokens are contiguous.

    V arrives as ``[batch, seq_k, heads, head_dim]``, i.e. one head's V rows are
    ``heads * head_dim`` elements apart -- 8 KB at 16 heads x head_dim 128 fp32.
    The P@V sweep walks those rows once per key per d-chunk, so on a 33 MB V that
    stride is what the prefetcher cannot follow: measured on the Ryzen,
    ``S=4096 H=16 D=128`` fp32 runs **1.95x** faster when the same values are
    handed over head-major (16.8 MB: 1.14x; head_dim 64 or bf16, where V fits in
    cache: 1.0x). vLLM gets this layout for free from its paged cache
    (``[block][head][block_size][head_dim]``, 512 B per token); a dense V does
    not, so it is packed here with the K transpose in the same fork-join.

    It is a straight copy -- no transpose, no dtype change -- so the result is
    bit-identical to reading V in place; it costs 1-2% of the call (33 MB at
    40-70 GB/s against a 100 ms kernel).
    """
    var src = _ptr[Scalar[dtype]](job.v).unsafe_offset(
        batch * job.v_stride_b + head * job.v_stride_h
    )
    var dst = _ptr[Scalar[dtype]](job.vt).unsafe_offset(
        (batch * job.num_kv_heads + head) * job.vt_stride_bh
    )
    var t0 = 0
    while t0 < job.seq_k:
        var n = min(TBLK, job.seq_k - t0)
        comptime for i in range(TBLK):
            if i < n:
                dst.unsafe_store[width=head_dim](
                    i * head_dim,
                    src.unsafe_load[width=head_dim](
                        (t0 + i) * job.v_stride_t
                    ),
                )
        dst = dst.unsafe_offset(TBLK * head_dim)
        t0 += TBLK


@inline(.always)
def _vpack_head[
    dtype: DType, head_dim: Int
](job: TransposeJob, batch: Int, head: Int):
    """Copy one (batch, kv head) of V into the ``bfmmla`` P@V operand layout.

    ``[seq_k/4, head_dim/2, 8]``: eight bf16 = one head-dim pair across four
    keys (four of dim ``2d``, then four of dim ``2d+1``). P@V contracts over
    keys, so the operand's two 4-lane halves are the two *output dims*; V's
    natural row-major layout has them interleaved with the key index instead.

    Key group outermost, head-dim pair inner: P@V walks "all dim pairs of one
    key group" (a contiguous 512 B run at head_dim 64) so its inner loop needs
    no address arithmetic at all -- the old ``[dim pair][key group]`` order had
    to add a runtime stride per dim pair, which was 8 of the 35 instructions in
    that loop. The padding lanes of the last key group are written as zeros:
    the kernel reads the whole group and ``0 * NaN`` is not 0.
    """
    var src = _ptr[Scalar[dtype]](job.v).unsafe_offset(
        batch * job.v_stride_b + head * job.v_stride_h
    )
    var dst = _ptr[Scalar[dtype]](job.vt).unsafe_offset(
        (batch * job.num_kv_heads + head) * job.vt_stride_bh
    )
    var t0 = 0
    while t0 < job.seq_k:
        var n = min(TBLK, job.seq_k - t0)
        # 16 tokens = 4 key groups; one 8-element store per (group, dim pair).
        comptime for dp in range(head_dim // 2):
            comptime for j in range(TBLK // 4):
                var w = SIMD[dtype, 8](0.0)
                comptime for i4 in range(4):
                    if 4 * j + i4 < n:
                        var pair = src.unsafe_offset(
                            (t0 + 4 * j + i4) * job.v_stride_t + 2 * dp
                        ).unsafe_load[width=2]()
                        w[i4] = pair[0]
                        w[4 + i4] = pair[1]
                dst.unsafe_store[width=8](
                    (t0 // 4 + j) * job.vt_stride_d + dp * 8, w
                )
        t0 += TBLK


comptime TransposeTaskFn = def(TransposeJob, Int) thin -> None


struct TransposeDispatch(ImplicitlyCopyable, Movable):
    var job: TransposeJob
    var task_fn: TransposeTaskFn
    var num_tasks: Int

    def __init__(
        out self, job: TransposeJob, task_fn: TransposeTaskFn, num_tasks: Int
    ):
        self.job = job
        self.task_fn = task_fn
        self.num_tasks = num_tasks


@export
def _gomp_worker_transpose(data: Int) abi("C"):
    var st = MutPointer[TransposeDispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    var task = tid
    while task < state.num_tasks:
        state.task_fn(state.job, task)
        task += threads


@inline(.always)
def _transpose_task[
    dtype: DType, head_dim: Int, mode: Int
](job: TransposeJob, task: Int):
    var batch = task // job.num_kv_heads
    var head = task % job.num_kv_heads
    _transpose_head[dtype, head_dim, mode](job, batch, head)
    comptime if mode == 2:
        # The bfmmla path needs V position-major too; pack both in one pass so
        # the call pays a single fork-join.
        _vpack_head[dtype, head_dim](job, batch, head)
    else:
        # Every other path reads V with `_vcopy_head`'s contiguous layout, when
        # the Python side decided the copy pays (`vt == 0` otherwise: same
        # `v_stride_t` the caller handed us, read in place).
        if job.vt != 0:
            _vcopy_head[dtype, head_dim](job, batch, head)


def _transpose_dispatch[
    dtype: DType, head_dim: Int, mode: Int
](job: TransposeJob, num_workers: Int):
    var items = job.batch * job.num_kv_heads
    if items == 0:
        return
    var workers = min(num_workers, items)
    if workers <= 1:
        for task in range(items):
            _transpose_task[dtype, head_dim, mode](job, task)
        return
    comptime task_fn = _transpose_task[dtype, head_dim, mode]
    var state = stack_allocation[1, TransposeDispatch]()
    state[] = TransposeDispatch(job, task_fn, items)
    comptime entry = _gomp_worker_transpose
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(workers), Int32(0)
    )


def _transpose_shape[
    dtype: DType, head_dim: Int
](job: TransposeJob, num_workers: Int, mode: Int) raises:
    if mode == 0:
        _transpose_dispatch[dtype, head_dim, 0](job, num_workers)
    elif mode == 1:
        _transpose_dispatch[dtype, head_dim, 1](job, num_workers)
    elif mode == 2:
        _transpose_dispatch[dtype, head_dim, 2](job, num_workers)
    else:
        raise Error("cpu_attention dense: unknown K/V layout mode")


def _transpose_dtype[
    dtype: DType
](job: TransposeJob, num_workers: Int, mode: Int) raises:
    if job.head_dim == 32:
        _transpose_shape[dtype, 32](job, num_workers, mode)
    elif job.head_dim == 64:
        _transpose_shape[dtype, 64](job, num_workers, mode)
    elif job.head_dim == 128:
        _transpose_shape[dtype, 128](job, num_workers, mode)
    else:
        raise Error("cpu_attention dense: unsupported head_dim")


def run_transpose(
    job: TransposeJob, dtype_code: Int, num_workers: Int, mode: Int
) raises:
    """Pack K (and, in mode 2, V) into the layout the kernel reads.

    Mode 0 is the portable ``[head_dim, seq_k]`` transpose, mode 1 the ``bfdot``
    paired-head-dim layout, mode 2 the ``bfmmla`` layouts (K position-paired over
    4 head dims, V position-major over head-dim pairs). Parallel over
    (batch, kv head), one fork-join for both tensors in mode 2.
    """
    if dtype_code == 0:
        _transpose_dtype[DType.float32](job, num_workers, mode)
    elif dtype_code == 1:
        _transpose_dtype[DType.bfloat16](job, num_workers, mode)
    else:
        raise Error("cpu_attention dense: unsupported dtype")


comptime DenseTaskFn = def(DenseJob, Int) thin -> None


struct DenseDispatch(ImplicitlyCopyable, Movable):
    """State of one ``GOMP_parallel`` region, on the caller's stack."""

    var job: DenseJob
    var task_fn: DenseTaskFn
    var num_tasks: Int

    def __init__(out self, job: DenseJob, task_fn: DenseTaskFn, num_tasks: Int):
        self.job = job
        self.task_fn = task_fn
        self.num_tasks = num_tasks


@export
def _gomp_worker_dense(data: Int) abi("C"):
    """Run this thread's share of the work items, strided by thread id.

    Round-robin rather than a contiguous slice: work item ``t`` costs
    ``O(q0(t))``, so with a contiguous slice the thread that owns the last query
    rows does several times the work of the first one. Striding keeps every
    thread's mix of early and late rows the same.
    """
    var st = MutPointer[DenseDispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    var task = tid
    while task < state.num_tasks:
        state.task_fn(state.job, task)
        task += threads


@inline(.always)
comptime ARM_V2 = LANES < 16
"""Whether this build is the ARM path whose v2 kernels pair q blocks below."""


@inline(.always)
def _dense_arm_rows[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """The v2 tile kernel for this dtype (bf16 ``bfmmla`` or fp32)."""
    comptime if dtype == DType.bfloat16:
        _dense_rows_tile[head_dim, qt, TILE_KS, 4, TILE_NSQ](
            job, batch, head, kv_head, q0
        )
    else:
        _dense_rows_tile_f32[head_dim, qt, TILE_KS, 8, 8](
            job, batch, head, kv_head, q0
        )


def _dense_task_pair[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, task: Int):
    """One work item = the cheapest *and* the most expensive q block of a head.

    Causal work grows linearly with ``q0``, so round-robin over q blocks leaves
    whichever thread draws the late blocks with ~1.4x the average work: at
    S=128 15% of the call was sitting in the OpenMP barrier
    (``gomp_team_barrier_wait_end`` + ``gomp_barrier_wait_end``, 14.6% of the
    process by ``perf``). Pairing block ``g`` with block ``qgroups-1-g`` makes
    every work item the same cost and halves the number of work items, which
    also halves the per-item prologue.
    """
    var qgroups = (job.seq_q + qt - 1) // qt
    var gpairs = (qgroups + 1) // 2
    var gp = task % gpairs
    var rest = task // gpairs
    var head = rest % job.num_heads
    var batch = rest // job.num_heads
    var kv_head = head
    if job.num_heads != job.num_kv_heads:
        kv_head = head // (job.num_heads // job.num_kv_heads)
    _dense_arm_rows[dtype, head_dim, qt](job, batch, head, kv_head, gp * qt)
    var g2 = qgroups - 1 - gp
    if g2 != gp:
        _dense_arm_rows[dtype, head_dim, qt](job, batch, head, kv_head, g2 * qt)


def _dense_task[dtype: DType, head_dim: Int, qt: Int](job: DenseJob, task: Int):
    var qgroups = (job.seq_q + qt - 1) // qt
    var g = task % qgroups
    var rest = task // qgroups
    var head = rest % job.num_heads
    var batch = rest // job.num_heads
    var kv_head = head
    if job.num_heads != job.num_kv_heads:
        kv_head = head // (job.num_heads // job.num_kv_heads)
    var q0 = g * qt
    comptime if LANES < 16:
        # NEON: `head_dim / LANES` floats per row is 16 registers at head_dim 64
        # whatever qt is, so the register kernel cannot hold an accumulator.
        # bf16 runs the 2x2-tile `bfmmla` kernel, whose operands carry two query
        # rows at a time; the 4-lane paths keep the head-dim `pairs` layout.
        comptime if dtype == DType.bfloat16 and BFMMLA:
            # Tile choice: ``qt/2 * sub/2`` 2x2 accumulator tiles have to fit
            # beside the softmax working set, and the load ratio goes the other
            # way -- (qt/2 + sub/2) / (qt/2 * sub/2) loads per instruction, 0.63
            # at 4 rows x 16 keys and 0.5 at 8 x 8. The per-softmax-step fixed
            # cost dominates the load ratio at head_dim 64 (a 16-key step is 16%
            # faster than an 8-key one at qt = 4), while at head_dim 128 there
            # is 2x the matmul per key, so 8 rows x 8 keys wins there.
            comptime if qt <= 4:
                _dense_rows_bfmmla[head_dim, qt, BKS, 8, 16](
                    job, batch, head, kv_head, q0
                )
            else:
                _dense_arm_rows[dtype, head_dim, qt](
                    job, batch, head, kv_head, q0
                )
        elif dtype == DType.bfloat16 and BF16_DOT:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, True](
                job, batch, head, kv_head, q0
            )
        elif qt > 4:
            _dense_arm_rows[dtype, head_dim, qt](
                job, batch, head, kv_head, q0
            )
        else:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, False](
                job, batch, head, kv_head, q0
            )
    elif dtype == DType.bfloat16 and BF16_DOT:
        comptime if head_dim >= 128:
            # head_dim 128 x qt rows of bf16 dot accumulator is 8 zmm per row,
            # so `_dense_rows_dot`'s register accumulator is 64 zmm at qt = 8 and
            # spills. The L1-accumulator variant with the same dot instruction
            # measured 1.24-1.28x over it at 256/1024/4096 x 16 x 128 on the
            # Ryzen, interleaved, medians of 5.
            _dense_rows_mem[dtype, head_dim, qt, 32, 32, True](
                job, batch, head, kv_head, q0
            )
        else:
            _dense_rows_dot[dtype, head_dim, qt](job, batch, head, kv_head, q0)
    elif qt * head_dim >= 512:
        # acc (qt * head_dim / LANES zmm) + logits (qt * KT_TILE / LANES) do
        # not fit in 32 registers: keep acc in L1 and tile the head dim.
        # The threshold is `>=`, not `>`: 8 rows x 64 dims is exactly 512 floats
        # of accumulator *plus* 8 x KT_TILE/16 = 32 zmm of logits, i.e. 64 zmm
        # of live state, so it spills in the register kernel. Measured on the
        # Ryzen, interleaved, medians of 5: 1.05-1.10x for the L1-accumulator
        # kernel at 1024/2048/4096 x 8 x 64 fp32, and with qt=6 instead of 8 on
        # top of that (CPU_ATTENTION_QT=6) the same shapes gain 1.19-1.22x.
        _dense_rows_mem[dtype, head_dim, qt, 32, 32, False](
            job, batch, head, kv_head, q0
        )
    else:
        _dense_rows[dtype, head_dim, qt](job, batch, head, kv_head, q0)


def _dense_dispatch[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, num_workers: Int):
    var qgroups = (job.seq_q + qt - 1) // qt
    comptime paired = (
        LANES < 16
        and qt > 4
        and (dtype == DType.float32 or (dtype == DType.bfloat16 and BFMMLA))
    )
    comptime if paired:
        qgroups = (qgroups + 1) // 2
    var items = job.batch * job.num_heads * qgroups
    if items == 0:
        return
    var workers = min(num_workers, items)
    comptime if paired:
        if workers <= 1:
            for task in range(items):
                _dense_task_pair[dtype, head_dim, qt](job, task)
            return
        var state = stack_allocation[1, DenseDispatch]()
        comptime tfp = _dense_task_pair[dtype, head_dim, qt]
        state[] = DenseDispatch(job, tfp, items)
        comptime entry_p = _gomp_worker_dense
        external_call["GOMP_parallel", NoneType](
            entry_p, Int(state), Int32(workers), Int32(0)
        )
    else:
        if workers <= 1:
            for task in range(items):
                _dense_task[dtype, head_dim, qt](job, task)
            return
        var state = stack_allocation[1, DenseDispatch]()
        comptime tfn = _dense_task[dtype, head_dim, qt]
        state[] = DenseDispatch(job, tfn, items)
        comptime entry_n = _gomp_worker_dense
        external_call["GOMP_parallel", NoneType](
            entry_n, Int(state), Int32(workers), Int32(0)
        )


def _run_shape[
    dtype: DType, head_dim: Int
](job: DenseJob, num_workers: Int, qt: Int) raises:
    if qt == 2:
        _dense_dispatch[dtype, head_dim, 2](job, num_workers)
    elif qt == 4:
        _dense_dispatch[dtype, head_dim, 4](job, num_workers)
    elif qt == 6:
        _dense_dispatch[dtype, head_dim, 6](job, num_workers)
    elif qt == 8:
        _dense_dispatch[dtype, head_dim, 8](job, num_workers)
    elif qt == 10:
        # Additive arm for the x86 sweep on the Xeon; no ARM path selects it.
        _dense_dispatch[dtype, head_dim, 10](job, num_workers)
    elif qt == 12:
        _dense_dispatch[dtype, head_dim, 12](job, num_workers)
    else:
        raise Error("cpu_attention dense: qt must be 2, 4, 6, 8, 10 or 12")


def _run_dtype[dtype: DType](job: DenseJob, num_workers: Int, qt: Int) raises:
    if job.head_dim == 32:
        _run_shape[dtype, 32](job, num_workers, qt)
    elif job.head_dim == 64:
        _run_shape[dtype, 64](job, num_workers, qt)
    elif job.head_dim == 128:
        _run_shape[dtype, 128](job, num_workers, qt)
    else:
        raise Error("cpu_attention dense: unsupported head_dim")


def lane_width() -> Int:
    """fp32 lanes per vector in this build (16 on AVX-512, 4 on NEON)."""
    return LANES


def has_bfmmla() -> Bool:
    """Whether this build has NEON ``bfmmla`` (the K *and* V layouts differ)."""
    comptime if BFMMLA:
        return True
    else:
        return False


def has_bf16_dot() -> Bool:
    """Whether this build has a bf16 dot-product instruction (the K layout depends
    on it: paired head dims for the dot, one head dim per position otherwise).
    """
    comptime if BF16_DOT:
        return True
    else:
        return False


def run_dense(job: DenseJob, dtype_code: Int, num_workers: Int, qt: Int) raises:
    """Dispatch on dtype (0 = fp32, 1 = bf16), head_dim (32/64/128) and q block.
    """
    if dtype_code == 0:
        _run_dtype[DType.float32](job, num_workers, qt)
    elif dtype_code == 1:
        _run_dtype[DType.bfloat16](job, num_workers, qt)
    else:
        raise Error("cpu_attention dense: unsupported dtype")
