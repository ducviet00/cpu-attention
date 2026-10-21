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
* bf16 takes the same bf16 dot-product path as the paged kernel (``dpbf16ps`` /
  ``bfdot``), which is what makes bf16 faster than fp32 here: AVX-512 has no
  native bf16 add/multiply, so the only thing bf16 may be used for is the dot.

Work items are handed out round-robin over the OpenMP pool (``GOMP_parallel`` on
torch's own pool, as in ``attention.mojo``) because causal masking makes the
late query rows ~2x more expensive than the early ones -- a contiguous slice per
thread would leave half the cores idle at the end of a long prefill.
"""

from std.ffi import external_call
from std.math import exp, iota
from std.memory import MutPointer, stack_allocation
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

comptime NEG_INF = -3.4028234663852886e38


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
                alpha = exp(m_old - m_new)
            var p = SIMD[DType.float32, KT_TILE](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = exp(logits[r] - m_new)
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
                alpha = exp(m_old - m_new)
            var p = SIMD[DType.float32, kt](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = exp(logits[r] - m_new)
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
                alpha = exp(m_old - m_new)
            var p = SIMD[DType.float32, KT_TILE](0.0)
            if m_new > Scalar[DType.float32](NEG_INF):
                p = exp(logits[r] - m_new)
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


comptime TBLK = 16
"""Tokens transposed at a time: 16 floats / 32 bf16 is one 64-byte store, so the
destination is written a cache line at a time instead of 4 bytes at a time (which
is what ``torch.permute(...).contiguous()`` does here -- 5-15 GB/s)."""


struct TransposeJob(Copyable, ImplicitlyCopyable, Movable):
    """K -> K^T (the Q@K^T sweep's layout), as raw addresses."""

    var k: Int
    var kt: Int
    var batch: Int
    var num_kv_heads: Int
    var seq_k: Int
    var head_dim: Int
    var k_stride_b: Int
    var k_stride_t: Int
    var k_stride_h: Int
    var kt_stride_bh: Int
    var kt_stride_d: Int

    def __init__(out self):
        self.k = 0
        self.kt = 0
        self.batch = 0
        self.num_kv_heads = 0
        self.seq_k = 0
        self.head_dim = 0
        self.k_stride_b = 0
        self.k_stride_t = 0
        self.k_stride_h = 0
        self.kt_stride_bh = 0
        self.kt_stride_d = 0


@inline(.always)
def _transpose_head[
    dtype: DType, head_dim: Int, pairs: Bool
](job: TransposeJob, batch: Int, head: Int):
    """Copy one (batch, kv head) of K into its transposed slot.

    ``pairs`` builds the bf16 dot layout ``[head_dim/2, seq_k, 2]`` (the two head
    dims a dot instruction contracts, next to the position they belong to);
    otherwise it is one head dim per row, ``[head_dim, seq_k]``.
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
        comptime if pairs:
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
    dtype: DType, head_dim: Int, pairs: Bool
](job: TransposeJob, task: Int):
    _transpose_head[dtype, head_dim, pairs](
        job, task // job.num_kv_heads, task % job.num_kv_heads
    )


def _transpose_dispatch[
    dtype: DType, head_dim: Int, pairs: Bool
](job: TransposeJob, num_workers: Int):
    var items = job.batch * job.num_kv_heads
    if items == 0:
        return
    var workers = min(num_workers, items)
    if workers <= 1:
        for task in range(items):
            _transpose_task[dtype, head_dim, pairs](job, task)
        return
    comptime task_fn = _transpose_task[dtype, head_dim, pairs]
    var state = stack_allocation[1, TransposeDispatch]()
    state[] = TransposeDispatch(job, task_fn, items)
    comptime entry = _gomp_worker_transpose
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(workers), Int32(0)
    )


def _transpose_shape[
    dtype: DType, head_dim: Int
](job: TransposeJob, num_workers: Int, pairs: Bool):
    if pairs:
        _transpose_dispatch[dtype, head_dim, True](job, num_workers)
    else:
        _transpose_dispatch[dtype, head_dim, False](job, num_workers)


def _transpose_dtype[
    dtype: DType
](job: TransposeJob, num_workers: Int, pairs: Bool) raises:
    if job.head_dim == 32:
        _transpose_shape[dtype, 32](job, num_workers, pairs)
    elif job.head_dim == 64:
        _transpose_shape[dtype, 64](job, num_workers, pairs)
    elif job.head_dim == 128:
        _transpose_shape[dtype, 128](job, num_workers, pairs)
    else:
        raise Error("cpu_attention dense: unsupported head_dim")


def run_transpose(
    job: TransposeJob, dtype_code: Int, num_workers: Int, pairs: Bool
) raises:
    """Transpose K into the Q@K^T layout, in parallel over (batch, kv head)."""
    if dtype_code == 0:
        _transpose_dtype[DType.float32](job, num_workers, pairs)
    elif dtype_code == 1:
        _transpose_dtype[DType.bfloat16](job, num_workers, pairs)
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
        # Keep it in L1 and tile the head dim by two vectors instead; bf16 keeps
        # its dot instruction through the `pairs` K layout.
        comptime if dtype == DType.bfloat16 and BF16_DOT:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, True](
                job, batch, head, kv_head, q0
            )
        else:
            _dense_rows_mem[dtype, head_dim, qt, KT_TILE, 2 * LANES, False](
                job, batch, head, kv_head, q0
            )
    elif dtype == DType.bfloat16 and BF16_DOT:
        _dense_rows_dot[dtype, head_dim, qt](job, batch, head, kv_head, q0)
    elif qt * head_dim > 384:
        # acc (qt * head_dim / LANES zmm) + logits (qt * KT_TILE / LANES) no
        # longer fit in 32 registers: keep acc in L1 and tile the head dim.
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
