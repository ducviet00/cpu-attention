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
        "llvm.aarch64.neon.bfmmla", SIMD[DType.float32, 4], has_side_effect=False
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
    var t = fma(xc, SIMD[DType.float32, width](LOG2E), SIMD[DType.float32, width](MAGIC))
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
    var sbits = (ki + SIMD[DType.int32, width](127)) << SIMD[DType.int32, width](23)
    return p * bitcast[DType.float32, width](sbits)


@inline(.always)
def _vexp_f(x: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Scalar ``exp`` through the same polynomial (the online-softmax rescale)."""
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
        var logits = Array[SIMD[DType.float32, KT_TILE], qt](uninitialized=True)
        comptime for r in range(qt):
            logits[r] = SIMD[DType.float32, KT_TILE](0.0)

        # Q@K^T: one K^T vector (KT_TILE positions of one head dim) feeds every
        # row of the block, so the loads are amortized over `qt` FMAs.
        var d = 0
        while d < head_dim:
            var kv = kt_base.unsafe_load[width=KT_TILE](
                d * job.kt_stride_d + t0 * job.kt_stride_t
            ).cast[DType.float32]()
            comptime for r in range(qt):
                logits[r] += (
                    SIMD[DType.float32, KT_TILE](
                        qv.unsafe_load(r * head_dim + d)
                    )
                    * kv
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
        var jn = min(KT_TILE, seq_k - t0)
        var j = 0
        while j < jn:
            var vrow = v_base.unsafe_load[width=head_dim](
                (t0 + j) * job.v_stride_t
            ).cast[DType.float32]()
            comptime for r in range(qt):
                acc[r] += probs.unsafe_load(r * KT_TILE + j) * vrow
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
                                + t0 * job.kt_stride_t
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
                    d * job.kt_stride_d + t0 * job.kt_stride_t
                ).cast[DType.float32]()
                comptime for r in range(qt):
                    logits[r] += (
                        SIMD[DType.float32, kt](qv.unsafe_load(r * head_dim + d))
                        * kv
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

        var jn = min(kt, seq_k - t0)
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
                    accv[r] += probs.unsafe_load(r * kt + j) * vrow
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
                    (acc.unsafe_load[width=head_dim](r * head_dim).cast[
                        DType.float32
                    ]() * (Scalar[DType.float32](1.0) / s)).cast[dtype]()
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
                            + t0 * job.kt_stride_t
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
        var jn = min(KT_TILE, seq_k - t0)
        var j = 0
        while j < jn:
            var vrow = v_base.unsafe_load[width=head_dim](
                (t0 + j) * job.v_stride_t
            ).cast[DType.float32]()
            comptime for r in range(qt):
                acc[r] += probs.unsafe_load(r * KT_TILE + j) * vrow
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

    ``y = n + f`` with ``n`` the rounded integer part (the add-and-subtract
    magic-constant trick, so no round instruction), ``2^f`` from a degree-4
    polynomial on ``[-0.5, 0.5]`` and ``2^n`` by adding ``n`` to the exponent
    field of 1.0. Clamping at -127 makes the exponent field underflow to zero
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
    var n = (yc + SIMD[DType.float32, 4](MAGIC)) - SIMD[DType.float32, 4](
        MAGIC
    )
    var f = yc - n
    var p = SIMD[DType.float32, 4](0.009618129107628477) * f + SIMD[
        DType.float32, 4
    ](0.05550410866482158)
    p = p * f + SIMD[DType.float32, 4](0.2402265069591007)
    p = p * f + SIMD[DType.float32, 4](0.6931471805599453)
    p = p * f + SIMD[DType.float32, 4](1.0)
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
            q_pan.unsafe_store[width=8]((rp * DG + g) * 8, lo.join(hi))

    var acc = stack_allocation[RP * DP * 4, DType.float32]()
    comptime for i in range(RP * DP):
        acc.unsafe_store[width=4](i * 4, SIMD[DType.float32, 4](0.0))
    var row_max = Array[Float32, qt](uninitialized=True)
    var row_sum = Array[Float32, qt](uninitialized=True)
    comptime for r in range(qt):
        row_max[r] = Scalar[DType.float32](NEG_INF)
        row_sum[r] = Scalar[DType.float32](0.0)

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
            var accs = Array[SIMD[DType.float32, 4], RP * NS](uninitialized=True)
            comptime for i in range(RP * NS):
                accs[i] = SIMD[DType.float32, 4](0.0)
            # One K vector (2 key positions x 4 head dims) feeds every row pair;
            # one Q panel feeds every key pair.
            var g = 0
            while g < DG:
                comptime for rp in range(RP):
                    var a = q_pan.unsafe_load[width=8]((rp * DG + g) * 8)
                    comptime for v in range(NS):
                        accs[rp * NS + v] = _bfmmla(
                            accs[rp * NS + v],
                            a,
                            kt_base.unsafe_load[width=8](
                                g * job.kt_stride_d + (half + v) * 8
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
                var mx = accs[rp * NS]
                comptime for v in range(1, NS):
                    mx = _vmax(mx, accs[rp * NS + v])
                var m0 = max(row_max[2 * rp], max(mx[0], mx[1]))
                var m1 = max(row_max[2 * rp + 1], max(mx[2], mx[3]))
                # 2^0 = 1 exactly when the max did not move, so the two rows
                # rescale independently and a static row is a no-op.
                var a0 = _exp2_fast(
                    SIMD[DType.float32, 4]((row_max[2 * rp] - m0) * c1)
                )[0]
                var a1 = _exp2_fast(
                    SIMD[DType.float32, 4]((row_max[2 * rp + 1] - m1) * c1)
                )[0]
                # One FMA per accumulator turns the scaled logit into log2 space
                # (scale, max subtraction and the log2 factor folded into c1/c2),
                # then 2^y is the polynomial. A row with no visible key has
                # m == NEG_INF, so its centre overflows to +inf: the logit is
                # -inf too, the sum is NaN, and `_exp2_fast`'s fmaxnm turns that
                # into exactly 0 -- no special case needed.
                var off = SIMD[DType.float32, 2](-m0 * c1).join(
                    SIMD[DType.float32, 2](-m1 * c1)
                )
                var sum0 = SIMD[DType.float32, 4](0.0)
                var sum1 = SIMD[DType.float32, 4](0.0)
                comptime for v in range(NS // 2):
                    var p0 = _exp2_fast(accs[rp * NS + 2 * v] * c1 + off)
                    var p1 = _exp2_fast(accs[rp * NS + 2 * v + 1] * c1 + off)
                    accs[rp * NS + 2 * v] = p0
                    accs[rp * NS + 2 * v + 1] = p1
                    sum0 += p0
                    sum1 += p1
                var sumv = sum0 + sum1
                row_sum[2 * rp] = row_sum[2 * rp] * a0 + sumv[0] + sumv[1]
                row_sum[2 * rp + 1] = row_sum[2 * rp + 1] * a1 + sumv[2] + sumv[
                    3
                ]
                row_max[2 * rp] = m0
                row_max[2 * rp + 1] = m1
                if a0 != Scalar[DType.float32](1.0) or a1 != Scalar[
                    DType.float32
                ](1.0):
                    var av = SIMD[DType.float32, 4](a0, a0, a1, a1)
                    comptime for d in range(DP):
                        acc.unsafe_store[width=4](
                            (rp * DP + d) * 4,
                            acc.unsafe_load[width=4]((rp * DP + d) * 4) * av,
                        )
                    # The panels already written for this super-tile are relative
                    # to the previous max, and P@V has not consumed them yet, so
                    # they need the same rescale (the accumulator holds only
                    # earlier super-tiles). Rare: the running max stops moving
                    # after the first few tiles.
                    var av8 = SIMD[DType.float32, 4](a0).join(
                        SIMD[DType.float32, 4](a1)
                    )
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
                            pa[0], pa[1], pb[0], pb[1], pa[2], pa[3], pb[2], pb[3]
                        ),
                    )
            st += 1

        # ---- P@V over the super-tile, BDCP dim pairs at a time. Only the key
        # groups that exist: the masked lanes' probability is exactly 0, but
        # their V may be past the end of the packed buffer (0 * garbage is not 0).
        var jn = min(ks, seq_k - t0)
        var kgroups = (jn + 3) // 4
        comptime for dc in range(0, DP, dcp):
            var accv = Array[SIMD[DType.float32, 4], RP * dcp](uninitialized=True)
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    accv[rp * dcp + d] = acc.unsafe_load[width=4](
                        (rp * DP + dc + d) * 4
                    )
            var kg = 0
            while kg < kgroups:
                var ap = Array[SIMD[DType.bfloat16, 8], RP](uninitialized=True)
                comptime for rp in range(RP):
                    ap[rp] = probs.unsafe_load[width=8]((kg * RP + rp) * 8)
                comptime for d in range(dcp):
                    var vv = vt_base.unsafe_load[width=8](
                        (dc + d) * job.v_stride_t + (t0 // 4 + kg) * 8
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
                var s = row_sum[r]
                if s > Scalar[DType.float32](0.0):
                    var inv = SIMD[DType.float32, 4](
                        Scalar[DType.float32](1.0) / s
                    )
                    comptime for d4 in range(DP // 4):
                        var o8 = SIMD[DType.bfloat16, 8](0.0)
                        comptime for i in range(4):
                            var t = acc.unsafe_load[width=4](
                                (rp * DP + d4 * 4 + i) * 4
                            ) * inv
                            o8[2 * i] = Scalar[DType.bfloat16](t[2 * c])
                            o8[2 * i + 1] = Scalar[DType.bfloat16](t[2 * c + 1])
                        out.unsafe_store[width=8](d4 * 8, o8)
                else:
                    # Every key of this row was masked: flash-attn returns zeros.
                    comptime for d8 in range(DP // 4):
                        out.unsafe_store[width=8](
                            d8 * 8, SIMD[DType.bfloat16, 8](0.0)
                        )



comptime TBLK = 16
"""Tokens transposed at a time: 16 floats / 32 bf16 is one 64-byte store, so the
destination is written a cache line at a time instead of 4 bytes at a time (which
is what ``torch.permute(...).contiguous()`` does here -- 5-15 GB/s)."""


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
            comptime for d in range(head_dim):
                var v = SIMD[DType.float32, TBLK](0.0)
                comptime for i in range(TBLK):
                    if i < n:
                        v[i] = Scalar[DType.float32](
                            src.unsafe_load((t0 + i) * job.k_stride_t + d)
                        )
                dst.unsafe_store[width=TBLK](
                    d * job.kt_stride_d + t0, v.cast[dtype]()
                )
        elif mode == 1:
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
                    d2 * job.kt_stride_d + t0 * 2, v
                )
        else:
            # 16 tokens = 8 key pairs = 64 bf16 per head-dim group: one 128-byte
            # store, 4 head dims of one key per 8-byte load.
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
                    g * job.kt_stride_d + t0 * 4, v
                )
        t0 += TBLK


@inline(.always)
def _vpack_head[
    dtype: DType, head_dim: Int
](job: TransposeJob, batch: Int, head: Int):
    """Copy one (batch, kv head) of V into the ``bfmmla`` P@V operand layout.

    ``[head_dim/2, seq_k/4, 8]``: eight bf16 = one head-dim pair across four
    keys (four of dim ``2d``, then four of dim ``2d+1``). P@V contracts over
    keys, so the operand's two 4-lane halves are the two *output dims*; V's
    natural row-major layout has them interleaved with the key index instead.
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
        # 16 tokens = 4 groups of 4 keys: 32 bf16 per head-dim pair, written as
        # one 64-byte store from 32 2-element loads of the row-major source.
        comptime for dp in range(head_dim // 2):
            var v = SIMD[dtype, 2 * TBLK](0.0)
            comptime for i in range(TBLK):
                if i < n:
                    var pair = src.unsafe_offset(
                        (t0 + i) * job.v_stride_t + 2 * dp
                    ).unsafe_load[width=2]()
                    comptime for d in range(2):
                        v[8 * (i // 4) + 4 * d + i % 4] = pair[d]
            dst.unsafe_store[width=2 * TBLK](
                dp * job.vt_stride_d + (t0 // 4) * 8, v
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
def _transpose_task[dtype: DType, head_dim: Int, mode: Int](
    job: TransposeJob, task: Int
):
    var batch = task // job.num_kv_heads
    var head = task % job.num_kv_heads
    _transpose_head[dtype, head_dim, mode](job, batch, head)
    comptime if mode == 2:
        # The bfmmla path needs V position-major too; pack both in one pass so
        # the call pays a single fork-join.
        _vpack_head[dtype, head_dim](job, batch, head)


def _transpose_dispatch[dtype: DType, head_dim: Int, mode: Int](
    job: TransposeJob, num_workers: Int
):
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


def _transpose_shape[dtype: DType, head_dim: Int](
    job: TransposeJob, num_workers: Int, mode: Int
) raises:
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
                # 3-4 row pairs: halve the key tile and the P@V dim-pair chunk,
                # or the tile stops fitting in 32 registers.
                _dense_rows_bfmmla[head_dim, qt, BKS, 4, 8](
                    job, batch, head, kv_head, q0
                )
        elif dtype == DType.bfloat16 and BF16_DOT:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, True](
                job, batch, head, kv_head, q0
            )
        else:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, False](
                job, batch, head, kv_head, q0
            )
    elif dtype == DType.bfloat16 and BF16_DOT:
        _dense_rows_dot[dtype, head_dim, qt](job, batch, head, kv_head, q0)
    elif qt * head_dim > 512:
        # acc (qt * head_dim / LANES zmm) + logits (qt * KT_TILE / LANES) no
        # longer fit in 32 registers: keep acc in L1 and tile the head dim.
        # The threshold is 512 rather than 384 because the register kernel is 6%
        # faster at 8 rows x 64 dims where both are legal (measured, and it is
        # only unusable above 512).
        _dense_rows_mem[dtype, head_dim, qt, 32, 32, False](
            job, batch, head, kv_head, q0
        )
    else:
        _dense_rows[dtype, head_dim, qt](job, batch, head, kv_head, q0)


def _dense_dispatch[
    dtype: DType, head_dim: Int, qt: Int
](job: DenseJob, num_workers: Int):
    var qgroups = (job.seq_q + qt - 1) // qt
    var items = job.batch * job.num_heads * qgroups
    if items == 0:
        return
    comptime task_fn = _dense_task[dtype, head_dim, qt]
    var workers = min(num_workers, items)
    if workers <= 1:
        for task in range(items):
            _dense_task[dtype, head_dim, qt](job, task)
        return
    var state = stack_allocation[1, DenseDispatch]()
    state[] = DenseDispatch(job, task_fn, items)
    comptime entry = _gomp_worker_dense
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(workers), Int32(0)
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
    else:
        raise Error("cpu_attention dense: qt must be 2, 4, 6 or 8")


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
