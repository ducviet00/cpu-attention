"""Short-sequence bf16 dense attention: same geometry, less per-block overhead.

`_dense_rows_tile` in ``dense.mojo`` is tuned for long key ranges, where the work
per query block is thousands of row-keys and three per-block costs are rounding
error. At S<=256 they are not:

* **key coverage is rounded up to 16 keys.** The exp pass consumes four key
  groups per iteration, so ``nkp`` is rounded to 8 key pairs; at S=128 the
  average query block only has 68 keys, so 12 of the 80 it processes are keys no
  row may see (they are masked to -inf, i.e. pure waste). Measured against a
  non-causal run (which needs no rounding), the causal 128:8:64 call costs 43%
  more per row-key than its exact key count implies, and that fit puts ~1400
  cycles of per-block cost beside ~11% of rounding waste.
The shipped kernel rounds to 16 keys because its exponential pass consumes four
key groups per iteration; the two-group tail loop here removes that requirement.
(The obvious companion -- skipping the first tile's accumulator zeroing -- is
*not* done: a `t0 == 0` test inside the P@V chunk loop measured 4% slower.)

Tried and rejected (measured, in this file's history): **KV-outer**. Hold a row
chunk's accumulator in L1 and put the key tile outermost, so the row panels of a
tile are independent chains and the drain happens once per tile per chunk instead
of once per query block (the fit's F = 1391 cycles/block). At 128:8:64 bf16,
21 rounds, min: RB=64 24.3 -> 24.7 us (0.979x), RB=128 24.2 -> 24.9 (0.960x),
against 1.020x for the per-block form. The reorder costs the registers: the
per-chunk accumulator and the running max/sum become memory (`stack_allocation`)
round trips per (panel, tile) instead of registers, and the panel body is too big
for the out-of-order window to overlap anyway. So the per-block fixed cost is not
a loop-order artefact, and the missing cycles are not the *drain* a reorder can
collect.

Tried and rejected (measured, earlier in this file's history): making the *head* the work
item -- one call per (batch, head) with the Q panels built for 8 blocks at a time
and the block loop inside that one call -- so the per-block prologue and the
call boundary would be amortised. 128:8:64 bf16: 1.009x against the shipped
kernel where the per-block form is 1.020x, i.e. the ~1400 cycles per block that
a causal/non-causal fit attributes to fixed cost are **not** the Q panel and
**not** the call boundary. What is left is the epilogue (an `acc` load -> scale ->
shuffle -> store chain, 32 store groups, plus 8 scalar reciprocals) and the
pipeline drain at each block boundary; that needs a non-degenerate ablation
before anyone rewrites it -- the obvious "remove the epilogue" ablation is
degenerate because it kills the accumulator's live-out and reads 50% of the call.

A trap worth knowing: the K^T addressing here must match ``_rows_small`` (and
dense.mojo's mode-2 pack) exactly -- within a 64-key tile it is
``[key pair][dim group][8]``, so the dim-group stride is 8 and the key-pair
stride is ``DG * 8``. Using the older ``KT_PAIR_STRIDE`` constant reads the
packed buffer at the wrong offsets, which produces wrong logits for every row
except the first (a one-key row is just ``v[0]``) and is invisible in a
uniform-probability test.

Everything is the *same* tile geometry, the same bfmmla sequence and
the same operand layouts as the shipped kernel. It was bit-identical while it
shared the shipped kernel's degree-3 exponential; it now uses the bf16-precision
degree-2 one (`_exp2_bf16`, the form `dense.mojo`'s tile kernel uses), so the
two paths differ by ~1 bf16 ulp and `tests/bench_small.py` checks both against
SDPA inside the fixed bf16 tolerance rather than against each other. This file is
separate from ``dense.mojo`` on purpose: the bf16 D=128 tile is being rewritten
there, and this path only serves ``seq_k <= 256``.

Why not drop the K/V pack instead (which is 2.4 us of the 26.5 us call at
128:8:64)? Measured, not assumed: assembling a bfmmla operand from the native
``k[token][head][dim]`` layout costs an extra 206 instructions per 256 bfmmla
(``qk_packed`` 588 vs ``qk_native`` 794 in an isolated probe) and the P@V
operand -- four keys x two dims from rows ``v_stride_t`` apart -- costs 29 extra
instructions per 16 bfmmla (41 vs 70). That is ~10 instructions per row-key
against 2.4 us of pack for the whole call: the packed layouts are what make
every operand a single load, so the pack stays.
"""

from std.math import fma
from std.memory import stack_allocation
from std.ffi import external_call

from dense import (
    BFMMLA,
    DenseJob,
    DenseTaskFn,
    KS_STRIDE,
    KT_PAIR_STRIDE,
    NEG_INF,
    RESCALE_SLACK,
    TILE_KS,
    TILE_NSQ,
    TransposeJob,
    TransposeTaskFn,
    _bfmmla,
    _exp2_bf16,
    _exp2_g,
    _ptr,
    _swap_pairs,
    _transpose_task,
    _vmax,
)

comptime SMALL_MAXK = 256
"""Where the shipped path takes over by default (``_kernel._SMALL_MAXK``, which
the sweep can raise past this -- the host owns the threshold; this constant is
only the measured default and the docstring of the trade)."""

comptime DCP = 4
"""Head-dim pairs per P@V chunk: the shipped geometry (``dcp = 4``)."""


@inline(.always)
def _rows_small[
    head_dim: Int, qt: Int, ks: Int
](job: DenseJob, batch: Int, head: Int, kv_head: Int, q0: Int):
    """``_dense_rows_tile`` with the key range rounded to 4 keys, not 16, and no
    first-tile accumulator zeroing.

    Every operand layout, the bfmmla order and the tile size are the shipped
    ones; the loop bounds, the accumulator's initial value and the exponential
    differ. The exponential is `_exp2_bf16` (degree 2, the form `dense.mojo`'s
    tile kernel uses) instead of `_exp2_fast`, and the panel shuffle is a 4-lane
    `shuffle` instead of a lane constructor: measured 1.05x at 128:8:64 and
    1.04x at 256:16:128, interleaved, alternating processes. Precision moves
    from 7.8e-3 to 1.6e-2 worst-case against the fixed 3e-2 bf16 gate (the
    shipped D=128 rows read the same), so both paths are checked against SDPA
    inside the tolerance rather than against each other.
    """
    comptime RP = qt // 2  # row pairs
    comptime DG = head_dim // 4  # Q@K^T contraction groups (4 head dims each)
    comptime DP = head_dim // 2  # P@V output dim pairs
    comptime dcp = DCP
    comptime nsq = TILE_NSQ
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

    # inclusive Q@K^T scale: exp(scale*x) = 2^(c1*x)
    var c1 = Scalar[DType.float32](job.scale) * Scalar[DType.float32](
        1.4426950408889634
    )
    # Q A panels: [row pair][head-dim group of 4][8] -- four head dims of the
    # even row then four of the odd row.
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

    # Output accumulator: [row pair][dim pair][4], in L1. It is *not* zeroed up
    # front: the first tile's P@V starts every chunk from a zero register. The
    # one case that needs zeros is a query block with no visible key at all
    # (loop_end == 0), where nothing is ever stored and the epilogue multiplies
    # garbage by a zero reciprocal.
    # `acc` is zeroed up front even though the first tile overwrites it: hoisting
    # a `t0 == 0` test into the P@V chunk loop instead measured *4% slower*
    # (1.010x vs 1.051x at 128:8:64), because the branch sits inside an unrolled
    # bfmmla loop and costs more than the 128 stores it saves.
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

    var loop_end = seq_k
    if job.causal:
        loop_end = min(seq_k, q0 + nrows + diag)

    var t0 = 0
    while t0 < loop_end:
        # ---- 1. Q@K^T into the logits buffer, and the tile max on the way.
        var mt = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
        comptime for rp in range(RP):
            mt[rp] = SIMD[DType.float32, 4](NEG_INF)
        var jn = min(ks, loop_end - t0)
        # Keys this tile must score: whole 4-key groups (the exp and P@V loops
        # both work in groups), rounded to 8 keys so the exp loop's two-group
        # tail always finishes it. A tile of `jn` keys therefore costs
        # `ceil(jn/8)*8` keys against the shipped `ceil(jn/16)*16`.
        var nkg = min(NKG, (jn + 7) // 8 * 2)
        # The Q@K^T sweep steps by `nsq` key pairs; anything past `nkg` groups is
        # masked by the causal limit (it is past the last row's limit by
        # construction), so it only has to be *in bounds* of the padded pack.
        var nkp = min(ks // 2, (2 * nkg + nsq - 1) // nsq * nsq)
        var sb = 0
        while sb < nkp:
            var kt_sub = kt_base.unsafe_offset(
                ((t0 + 2 * sb) // KS_STRIDE) * (KS_STRIDE * head_dim)
                + (((t0 + 2 * sb) % KS_STRIDE) // 2) * (DG * 8)
            )
            var accs = Array[SIMD[DType.float32, 4], RP * nsq](
                uninitialized=True
            )
            # Peel g=0 so the zero accumulator is one shared register.
            var zero = SIMD[DType.float32, 4](0.0)
            comptime for rp in range(RP):
                var a0 = q_pan.unsafe_load[width=8]((0 * RP + rp) * 8)
                comptime for v in range(nsq):
                    accs[rp * nsq + v] = _bfmmla(
                        zero, a0, kt_sub.unsafe_load[width=8](v * (DG * 8))
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
                                v * (DG * 8) + g * 8
                            ),
                        )
                g += 1

            # Causal tail: kill the lanes no row may see, in the accumulators so
            # the tile max and the exp pass both see NEG_INF.
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
                mt[rp] = _vmax(mt[rp], _vmax(mx, _swap_pairs(mx)))
                comptime for v in range(nsq):
                    logits.unsafe_store[width=4](
                        ((sb + v) * RP + rp) * 4, accs[rp * nsq + v]
                    )
            sb += nsq

        # ---- 2. One softmax update for the whole tile, then exp + panels.
        comptime for rp in range(RP):
            var has = mt[rp].reduce_max() > Scalar[DType.float32](NEG_INF)
            var m_new = m_v[rp]
            var alpha = SIMD[DType.float32, 4](1.0)
            if has and (mt[rp] - m_v[rp]).reduce_max() > RESCALE_SLACK:
                m_new = _vmax(m_v[rp], mt[rp])
                alpha = _exp2_g[4]((m_v[rp] - m_new) * c1)
                m_v[rp] = m_new
                if s_v[rp].reduce_max() > Scalar[DType.float32](0.0):
                    comptime for d in range(DP):
                        acc.unsafe_store[width=4](
                            (rp * DP + d) * 4,
                            acc.unsafe_load[width=4]((rp * DP + d) * 4) * alpha,
                        )
                    s_v[rp] = s_v[rp] * alpha
            var off = m_new * (-c1)
            var sumv = Array[SIMD[DType.float32, 4], 4](uninitialized=True)
            comptime for u in range(4):
                sumv[u] = SIMD[DType.float32, 4](0.0)
            var blind = m_new.reduce_max() == Scalar[DType.float32](NEG_INF)
            var kg = 0
            if not has and blind:
                while kg < nkg:
                    probs.unsafe_store[width=8](
                        (kg * RP + rp) * 8, SIMD[DType.bfloat16, 8](0.0)
                    )
                    kg += 1
            else:
                # Four key groups per iteration, then a two-group tail: `nkg` is
                # only even here (the shipped kernel rounds it to a multiple of
                # four by scoring up to 15 keys nobody can see).
                while kg + 4 <= nkg:
                    comptime for u in range(4):
                        var l0 = logits.unsafe_load[width=4](
                            (kg + u) * (2 * RP) * 4 + rp * 4
                        )
                        var l1 = logits.unsafe_load[width=4](
                            ((kg + u) * 2 + 1) * RP * 4 + rp * 4
                        )
                        # Scale, max offset and log2 base in one fma, with the
                        # bf16-precision polynomial (`_exp2_bf16`): the
                        # probabilities are rounded to bf16 two instructions
                        # later, so degree 2 is inside that rounding. The
                        # separate `l * c1` multiply this replaces was one
                        # instruction per vector, and `_exp2_fast`'s degree-3
                        # polynomial two more.
                        var p0 = _exp2_bf16(fma(l0, c1, off))
                        var p1 = _exp2_bf16(fma(l1, c1, off))
                        sumv[u] += p0 + p1
                        # Row 0's four keys are lanes 0,1 of both exponentials
                        # and row 1's are lanes 2,3: a panel is two 4-lane
                        # half-shuffles (`zip1/zip2.2d`) and a join. The
                        # lane-constructor spelling emits a `mov` per lane.
                        var z0 = p0.shuffle[0, 1, 4, 5](p1)
                        var z1 = p0.shuffle[2, 3, 6, 7](p1)
                        probs.unsafe_store[width=8](
                            ((kg + u) * RP + rp) * 8,
                            z0.cast[DType.bfloat16]().join(
                                z1.cast[DType.bfloat16]()
                            ),
                        )
                    kg += 4
                while kg + 2 <= nkg:
                    comptime for u in range(2):
                        var l0 = logits.unsafe_load[width=4](
                            (kg + u) * (2 * RP) * 4 + rp * 4
                        )
                        var l1 = logits.unsafe_load[width=4](
                            ((kg + u) * 2 + 1) * RP * 4 + rp * 4
                        )
                        var p0 = _exp2_bf16(fma(l0, c1, off))
                        var p1 = _exp2_bf16(fma(l1, c1, off))
                        sumv[u] += p0 + p1
                        var z0 = p0.shuffle[0, 1, 4, 5](p1)
                        var z1 = p0.shuffle[2, 3, 6, 7](p1)
                        probs.unsafe_store[width=8](
                            ((kg + u) * RP + rp) * 8,
                            z0.cast[DType.bfloat16]().join(
                                z1.cast[DType.bfloat16]()
                            ),
                        )
                    kg += 2
            var st = (sumv[0] + sumv[1]) + (sumv[2] + sumv[3])
            s_v[rp] = s_v[rp] + _swap_pairs(st) + st

        # Is this the last tile of the block? Then the row sums are final and the
        # output can be written straight out of the P@V chunk registers, which
        # skips the accumulator's 4 KB round trip per block (the epilogue's loads
        # and stores are the largest single term the ablation found).
        var last = t0 + ks >= loop_end
        var inv4s = Array[SIMD[DType.float32, 4], RP](uninitialized=True)
        if last:
            comptime for rp in range(RP):
                var vv0 = seq_k > 0
                var vv1 = vv0
                if job.causal:
                    vv0 = q0 + 2 * rp + diag + 1 > 0
                    vv1 = q0 + 2 * rp + 1 + diag + 1 > 0
                var j0 = Scalar[DType.float32](0.0)
                var j1 = Scalar[DType.float32](0.0)
                var q0s = s_v[rp][0]
                var q1s = s_v[rp][2]
                if vv0 and q0s > Scalar[DType.float32](0.0):
                    j0 = Scalar[DType.float32](1.0) / q0s
                if vv1 and q1s > Scalar[DType.float32](0.0):
                    j1 = Scalar[DType.float32](1.0) / q1s
                inv4s[rp] = SIMD[DType.float32, 4](j0, j0, j1, j1)

        # ---- 3. P@V over the tile, dcp dim pairs at a time.
        comptime for dc in range(0, DP, dcp):
            var accv = Array[SIMD[DType.float32, 4], RP * dcp](
                uninitialized=True
            )
            # The first tile's accumulator *is* zero, so load it only when some
            # earlier tile wrote it: 128 vector stores per query block become a
            # register move (measured 2.6% of a 128-key call).
            comptime for rp in range(RP):
                comptime for d in range(dcp):
                    accv[rp * dcp + d] = acc.unsafe_load[width=4](
                        (rp * DP + dc + d) * 4
                    )
            var vbase = vt_base.unsafe_offset(
                (t0 // 4) * job.v_stride_t + dc * 8
            )
            var kg = 0
            while kg < nkg:
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
            if last:
                # Normalise and de-interleave in registers: the same arithmetic
                # the epilogue did, without the accumulator round trip.
                comptime for rp in range(RP):
                    var b0 = accv[rp * dcp + 0] * inv4s[rp]
                    var b1 = accv[rp * dcp + 1] * inv4s[rp]
                    var b2 = accv[rp * dcp + 2] * inv4s[rp]
                    var b3 = accv[rp * dcp + 3] * inv4s[rp]
                    if 2 * rp < nrows:
                        o_base.unsafe_offset(
                            (q0 + 2 * rp) * job.o_stride_t
                        ).unsafe_store[width=8](
                            2 * dc,
                            b0.shuffle[0, 1, 4, 5](b1)
                            .cast[DType.bfloat16]()
                            .join(
                                b2.shuffle[0, 1, 4, 5](b3).cast[
                                    DType.bfloat16
                                ]()
                            ),
                        )
                    if 2 * rp + 1 < nrows:
                        o_base.unsafe_offset(
                            (q0 + 2 * rp + 1) * job.o_stride_t
                        ).unsafe_store[width=8](
                            2 * dc,
                            b0.shuffle[2, 3, 6, 7](b1)
                            .cast[DType.bfloat16]()
                            .join(
                                b2.shuffle[2, 3, 6, 7](b3).cast[
                                    DType.bfloat16
                                ]()
                            ),
                        )
            else:
                comptime for rp in range(RP):
                    comptime for d in range(dcp):
                        acc.unsafe_store[width=4](
                            (rp * DP + dc + d) * 4, accv[rp * dcp + d]
                        )

        t0 += ks

    # Output: one row of `head_dim` bf16 per query row -- already written by the
    # last tile's fused store unless the block had no tile at all (loop_end <= 0),
    # where nothing was ever accumulated and the row is zero.
    if loop_end <= 0:
        comptime for rp in range(RP):
            if 2 * rp < nrows:
                var orow = o_base.unsafe_offset((q0 + 2 * rp) * job.o_stride_t)
                comptime for i in range(DP // 4):
                    orow.unsafe_store[width=8](
                        i * 8, SIMD[DType.bfloat16, 8](0.0)
                    )
            if 2 * rp + 1 < nrows:
                var orow1 = o_base.unsafe_offset(
                    (q0 + 2 * rp + 1) * job.o_stride_t
                )
                comptime for i in range(DP // 4):
                    orow1.unsafe_store[width=8](
                        i * 8, SIMD[DType.bfloat16, 8](0.0)
                    )


def _task_small[head_dim: Int, qt: Int, ks: Int](job: DenseJob, task: Int):
    """One query block of one head, addressed as ``(batch, head, g)``.

    A single block, not a pair: in the fused per-head region the *worker* owns a
    whole head and walks its blocks, so pairing would run every block twice
    (measured: 1.88x slower, 431 vs 229 us at 256:16:128).
    """
    var qgroups = (job.seq_q + qt - 1) // qt
    var g = task % qgroups
    var rest = task // qgroups
    var head = rest % job.num_heads
    var batch = rest // job.num_heads
    var kv_head = head
    if job.num_heads != job.num_kv_heads:
        kv_head = head // (job.num_heads // job.num_kv_heads)
    _rows_small[head_dim, qt, ks](job, batch, head, kv_head, g * qt)


struct SmallDispatch(ImplicitlyCopyable, Movable):
    """State of one region: the pack job, the attention job, and the two task
    functions (the pack one is the shipped ``_transpose_task``)."""

    var tjob: TransposeJob
    var djob: DenseJob
    var t_fn: TransposeTaskFn
    var d_fn: DenseTaskFn
    var num_tasks: Int
    var qgroups: Int

    def __init__(
        out self,
        tjob: TransposeJob,
        djob: DenseJob,
        t_fn: TransposeTaskFn,
        d_fn: DenseTaskFn,
        num_tasks: Int,
        qgroups: Int,
    ):
        self.tjob = tjob
        self.djob = djob
        self.t_fn = t_fn
        self.d_fn = d_fn
        self.num_tasks = num_tasks
        self.qgroups = qgroups


@export
def _gomp_worker_small(data: Int) abi("C"):
    """One ``GOMP_parallel`` region: pack head ``t``, then attend head ``t``.

    Same shape as ``_gomp_worker_head`` -- no barrier between the phases, because
    the thread that packs a head is the thread that reads it -- reusing the
    shipped pack task through the function value in ``SmallDispatch``.
    """
    var st = MutPointer[SmallDispatch, MutUntrackedOrigin](
        unsafe_from_address=data
    )
    var state = st[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    var t = tid
    while t < state.num_tasks:
        state.t_fn(state.tjob, t)
        var base = t * state.qgroups
        var g = 0
        while g < state.qgroups:
            state.d_fn(state.djob, base + g)
            g += 1
        t += threads


def can_small(job: DenseJob) -> Bool:
    """Whether this call may use the short-sequence path (MHA only).

    The fused per-head form packs a kv head and attends the *same* index, so
    query heads must map one-to-one onto kv heads.
    """
    return (
        job.seq_k > 0
        and job.seq_q > 0
        and job.num_heads == job.num_kv_heads
        and job.head_dim >= 32
    )


def _small_fused[
    head_dim: Int, ks: Int, mode: Int
](tjob: TransposeJob, djob: DenseJob, num_workers: Int):
    comptime qt = 8
    comptime TASK_PACK = _transpose_task[DType.bfloat16, head_dim, mode]
    var items = djob.batch * djob.num_heads
    if items == 0 or djob.seq_q <= 0:
        return
    var qgroups = (djob.seq_q + qt - 1) // qt
    var t_fn: TransposeTaskFn = TASK_PACK
    comptime TASK_ROWS = _task_small[head_dim, qt, ks]
    var d_fn: DenseTaskFn = TASK_ROWS
    var workers = min(num_workers, items)
    if workers <= 1:
        for i in range(items):
            t_fn(tjob, i)
            var base = i * qgroups
            for g in range(qgroups):
                d_fn(djob, base + g)
        return
    var state = stack_allocation[1, SmallDispatch]()
    state[] = SmallDispatch(tjob, djob, t_fn, d_fn, items, qgroups)
    comptime entry = _gomp_worker_small
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(workers), Int32(0)
    )


def run_small_fused(
    tjob: TransposeJob, djob: DenseJob, mode: Int, num_workers: Int
) raises:
    """Pack K/V in ``mode``'s layout and attend with the short-sequence kernel.

    bf16 ``bfmmla`` only, ``qt = 8``, tile size as the shipped path picks it
    (``TILE_KS`` below head_dim 128, 256 at 128); anything else is the caller's
    bug, and ``can_small`` plus ``_kernel.py``'s mode selection keep it out.

    The ``comptime if BFMMLA`` is load-bearing on non-ARM builds: Mojo
    instantiates this function (and therefore this file's kernel, which emits
    the ``llvm.aarch64.neon.bfmmla`` intrinsic) even where nothing calls it, so
    without the guard an x86 build dies at codegen with
    ``LLVM ERROR: Cannot select: intrinsic %llvm.aarch64.neon.bfmmla`` -- which
    is exactly what merging this file without the guard did to ``main``.
    """
    comptime if not BFMMLA:
        raise Error("cpu_attention dense: the small path is ARM bf16 only")
    else:
        if not can_small(djob):
            raise Error("cpu_attention dense: small path cannot take this call")
        if djob.head_dim == 64:
            _small_fused[64, TILE_KS, 2](tjob, djob, num_workers)
        elif djob.head_dim == 128:
            _small_fused[128, 256, 2](tjob, djob, num_workers)
        elif djob.head_dim == 32:
            _small_fused[32, TILE_KS, 2](tjob, djob, num_workers)
        else:
            raise Error("cpu_attention dense: unsupported head_dim")
