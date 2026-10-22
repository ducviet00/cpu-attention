"""Python extension module for the cpu_attention paged-attention kernel.

Everything crossing the boundary is a raw address / integer, so a call costs a
handful of `PyLong` conversions and no tensor bookkeeping. The Python side
(``cpu_attention``) owns the tensors and passes ``tensor.data_ptr()``.

Build with:  pixi run build        (from ``cpu_attention/_mojo``)
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from attention import Job, run
from dense import (
    INT8_QK,
    DenseJob,
    TransposeJob,
    run_dense_fused,
    has_bf16_dot,
    has_bfmmla,
    lane_width,
    run_dense,
    run_transpose,
)
from dense_small import run_small_fused
from amx import amx_init
from dense_amx import AMX, AmxJob, PackJob, run_dense_amx


def dense_transpose(
    k: PythonObject,
    kt: PythonObject,
    v: PythonObject,
    vt: PythonObject,
    batch: PythonObject,
    num_kv_heads: PythonObject,
    seq_k: PythonObject,
    head_dim: PythonObject,
    k_stride_b: PythonObject,
    k_stride_t: PythonObject,
    k_stride_h: PythonObject,
    kt_stride_bh: PythonObject,
    kt_stride_d: PythonObject,
    kt_blk: PythonObject,
    v_stride_b: PythonObject,
    v_stride_t: PythonObject,
    v_stride_h: PythonObject,
    vt_stride_bh: PythonObject,
    vt_stride_d: PythonObject,
    dtype: PythonObject,
    mode: PythonObject,
    num_workers: PythonObject,
    ks: PythonObject,
    ks_stride_bh: PythonObject,
) raises:
    """Pack K (and, in mode 2, V) into the layout the dense kernel reads."""
    var job = TransposeJob()
    job.k = Int(py=k)
    job.kt = Int(py=kt)
    job.v = Int(py=v)
    job.vt = Int(py=vt)
    job.batch = Int(py=batch)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.seq_k = Int(py=seq_k)
    job.head_dim = Int(py=head_dim)
    job.k_stride_b = Int(py=k_stride_b)
    job.k_stride_t = Int(py=k_stride_t)
    job.k_stride_h = Int(py=k_stride_h)
    job.kt_stride_bh = Int(py=kt_stride_bh)
    job.kt_stride_d = Int(py=kt_stride_d)
    job.kt_blk = Int(py=kt_blk)
    job.v_stride_b = Int(py=v_stride_b)
    job.v_stride_t = Int(py=v_stride_t)
    job.v_stride_h = Int(py=v_stride_h)
    job.vt_stride_bh = Int(py=vt_stride_bh)
    job.vt_stride_d = Int(py=vt_stride_d)
    job.ks = Int(py=ks)
    job.ks_stride_bh = Int(py=ks_stride_bh)
    run_transpose(
        job,
        dtype_code=Int(py=dtype),
        num_workers=Int(py=num_workers),
        mode=Int(py=mode),
    )


def dense_lanes() -> PythonObject:
    """fp32 lanes per vector (16 on AVX-512, 4 on NEON)."""
    return PythonObject(Int32(lane_width()))


def dense_bf16_dot() -> PythonObject:
    """True when the kernel uses the bf16 dot-product (paired) K layout."""
    if has_bf16_dot():
        return PythonObject(True)
    return PythonObject(False)


def dense_bfmmla() -> PythonObject:
    """True when the bf16 path runs on NEON ``bfmmla`` (K and V both repacked).
    """
    if has_bfmmla():
        return PythonObject(True)
    return PythonObject(False)


def dense_int8_qk() -> PythonObject:
    """True when bf16 Q@K^T runs on i8mm ``smmla`` with int8 K and Q operands.

    The Python side picks pack mode 3 (int8 K plus a per-64-key dequant scale)
    only when this is true, so a build without i8mm keeps the bf16 layout.
    """
    if INT8_QK:
        return PythonObject(True)
    return PythonObject(False)


def dense_amx_init() -> PythonObject:
    """Ask the kernel for AMX tile-data permission; False means unusable here.

    Must succeed before any tile instruction: `#XF` state is off by default and
    the first tile instruction is SIGILL without it. Called once per process
    from `cpu_attention._kernel.amx_available()`.
    """
    if amx_init():
        return PythonObject(True)
    return PythonObject(False)


def dense_has_amx() -> PythonObject:
    """True when this build has the AMX kernel (x86 target with amx-tile+bf16).
    """
    if AMX:
        return PythonObject(True)
    return PythonObject(False)


def dense_amx_attention(
    q: PythonObject,
    kp: PythonObject,
    vp: PythonObject,
    dst: PythonObject,
    k: PythonObject,
    v: PythonObject,
    batch: PythonObject,
    seq_q: PythonObject,
    seq_k: PythonObject,
    num_heads: PythonObject,
    num_kv_heads: PythonObject,
    head_dim: PythonObject,
    q_stride_b: PythonObject,
    q_stride_t: PythonObject,
    q_stride_h: PythonObject,
    k_stride_b: PythonObject,
    k_stride_t: PythonObject,
    k_stride_h: PythonObject,
    v_stride_b: PythonObject,
    v_stride_t: PythonObject,
    v_stride_h: PythonObject,
    o_stride_b: PythonObject,
    o_stride_t: PythonObject,
    o_stride_h: PythonObject,
    kp_bh: PythonObject,
    vp_bh: PythonObject,
    kb16: PythonObject,
    kb32: PythonObject,
    dc32: PythonObject,
    dc16: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    kt: PythonObject,
    num_workers: PythonObject,
    fp32: PythonObject,
    mode: PythonObject,
) raises:
    """Pack K/V into the AMX operand layouts and run the attention.

    ``fp32`` selects the three-term hi/lo arithmetic (every operand split into
    two bf16 halves, three products per phase), which is fp32-accurate.

    ``kp``/``vp`` are caller-owned workspace (see ``cpu_attention._kernel``);
    the pack writes every tile it will read, so nothing needs zeroing except
    keys past ``seq_k``, which the pack writes as zero.
    """
    var pack = PackJob()
    pack.k = Int(py=k)
    pack.v = Int(py=v)
    pack.kp = Int(py=kp)
    pack.vp = Int(py=vp)
    pack.seq_k = Int(py=seq_k)
    pack.head_dim = Int(py=head_dim)
    pack.k_stride_b = Int(py=k_stride_b)
    pack.k_stride_t = Int(py=k_stride_t)
    pack.k_stride_h = Int(py=k_stride_h)
    pack.v_stride_b = Int(py=v_stride_b)
    pack.v_stride_t = Int(py=v_stride_t)
    pack.v_stride_h = Int(py=v_stride_h)
    pack.num_kv_heads = Int(py=num_kv_heads)
    pack.batch = Int(py=batch)
    pack.kb16 = Int(py=kb16)
    pack.kb32 = Int(py=kb32)
    pack.dc32 = Int(py=dc32)
    pack.dc16 = Int(py=dc16)
    pack.kp_bh = Int(py=kp_bh)
    pack.vp_bh = Int(py=vp_bh)

    var job = AmxJob()
    job.q = Int(py=q)
    job.kp = Int(py=kp)
    job.vp = Int(py=vp)
    job.dst = Int(py=dst)
    job.batch = Int(py=batch)
    job.seq_q = Int(py=seq_q)
    job.seq_k = Int(py=seq_k)
    job.num_heads = Int(py=num_heads)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.head_dim = Int(py=head_dim)
    job.q_stride_b = Int(py=q_stride_b)
    job.q_stride_t = Int(py=q_stride_t)
    job.q_stride_h = Int(py=q_stride_h)
    job.kp_bh = Int(py=kp_bh)
    job.vp_bh = Int(py=vp_bh)
    job.o_stride_b = Int(py=o_stride_b)
    job.o_stride_t = Int(py=o_stride_t)
    job.o_stride_h = Int(py=o_stride_h)
    job.scale = Float32(Float64(py=scale))
    job.causal = Bool(causal)
    job.kt = Int(py=kt)
    job.dc32 = Int(py=dc32)
    job.dc16 = Int(py=dc16)
    job.fp32 = Bool(fp32)
    job.mode = Int(py=mode)
    run_dense_amx(pack, job, num_workers=Int(py=num_workers), kt=Int(py=kt))


def dense_forward(
    k: PythonObject,
    kt: PythonObject,
    v: PythonObject,
    vt: PythonObject,
    vk: PythonObject,
    batch: PythonObject,
    num_kv_heads: PythonObject,
    seq_k: PythonObject,
    head_dim: PythonObject,
    k_stride_b: PythonObject,
    k_stride_t: PythonObject,
    k_stride_h: PythonObject,
    vt_stride_bh: PythonObject,
    vt_stride_d: PythonObject,
    vs_b: PythonObject,
    vs_t: PythonObject,
    vs_h: PythonObject,
    q: PythonObject,
    dst: PythonObject,
    seq_q: PythonObject,
    num_heads: PythonObject,
    q_stride_b: PythonObject,
    q_stride_t: PythonObject,
    q_stride_h: PythonObject,
    v_stride_b: PythonObject,
    v_stride_t: PythonObject,
    v_stride_h: PythonObject,
    o_stride_b: PythonObject,
    o_stride_t: PythonObject,
    o_stride_h: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    dtype: PythonObject,
    mode: PythonObject,
    kt_stride_bh: PythonObject,
    kt_stride_d: PythonObject,
    kt_stride_t: PythonObject,
    kt_blk: PythonObject,
    qt: PythonObject,
    num_workers: PythonObject,
    ks: PythonObject,
    ks_stride_bh: PythonObject,
) raises:
    """Pack K/V and run dense attention in one OpenMP region.

    Same work as ``dense_transpose`` followed by ``dense_attention``, but one
    fork-join with a ``GOMP_barrier`` between the phases instead of two. ``kt``
    and ``vt`` are the pack's destinations and the kernel's operands; the pack's
    V source strides and the kernel's *packed* V strides are separate arguments
    because they are different tensors.
    """
    var tjob = TransposeJob()
    tjob.k = Int(py=k)
    tjob.kt = Int(py=kt)
    tjob.v = Int(py=v)
    tjob.vt = Int(py=vt)
    tjob.batch = Int(py=batch)
    tjob.num_kv_heads = Int(py=num_kv_heads)
    tjob.seq_k = Int(py=seq_k)
    tjob.head_dim = Int(py=head_dim)
    tjob.k_stride_b = Int(py=k_stride_b)
    tjob.k_stride_t = Int(py=k_stride_t)
    tjob.k_stride_h = Int(py=k_stride_h)
    tjob.kt_stride_bh = Int(py=kt_stride_bh)
    tjob.kt_stride_d = Int(py=kt_stride_d)
    tjob.kt_blk = Int(py=kt_blk)
    tjob.v_stride_b = Int(py=vs_b)
    tjob.v_stride_t = Int(py=vs_t)
    tjob.v_stride_h = Int(py=vs_h)
    tjob.vt_stride_bh = Int(py=vt_stride_bh)
    tjob.vt_stride_d = Int(py=vt_stride_d)
    tjob.ks = Int(py=ks)
    tjob.ks_stride_bh = Int(py=ks_stride_bh)

    var job = DenseJob()
    job.q = Int(py=q)
    job.kt = Int(py=kt)
    job.v = Int(py=vk)
    job.dst = Int(py=dst)
    job.ks = Int(py=ks)
    job.ks_stride_bh = Int(py=ks_stride_bh)
    job.batch = Int(py=batch)
    job.seq_q = Int(py=seq_q)
    job.seq_k = Int(py=seq_k)
    job.num_heads = Int(py=num_heads)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.head_dim = Int(py=head_dim)
    job.q_stride_b = Int(py=q_stride_b)
    job.q_stride_t = Int(py=q_stride_t)
    job.q_stride_h = Int(py=q_stride_h)
    job.kt_stride_bh = Int(py=kt_stride_bh)
    job.kt_stride_d = Int(py=kt_stride_d)
    job.kt_stride_t = Int(py=kt_stride_t)
    job.kt_blk = Int(py=kt_blk)
    job.v_stride_b = Int(py=v_stride_b)
    job.v_stride_t = Int(py=v_stride_t)
    job.v_stride_h = Int(py=v_stride_h)
    job.o_stride_b = Int(py=o_stride_b)
    job.o_stride_t = Int(py=o_stride_t)
    job.o_stride_h = Int(py=o_stride_h)
    job.scale = Float32(Float64(py=scale))
    job.causal = Bool(causal)
    # `mode >= 8`: the short-sequence path (dense_small.mojo). It has its own
    # region because it drops the pack/attention barrier by construction -- a
    # thread packs a head and then attends it -- and the pack layout is `mode-8`.
    if Int(py=mode) >= 8:
        run_small_fused(
            tjob, job, mode=Int(py=mode) - 8, num_workers=Int(py=num_workers)
        )
    else:
        run_dense_fused(
            tjob,
            job,
            dtype_code=Int(py=dtype),
            mode=Int(py=mode),
            qt=Int(py=qt),
            num_workers=Int(py=num_workers),
        )


def dense_forward_plan(
    plan: PythonObject,
    k: PythonObject,
    q: PythonObject,
    vsrc: PythonObject,
    vkern: PythonObject,
    dst: PythonObject,
    scale: PythonObject,
) raises:
    """Pack K/V and run dense attention with the geometry in an int64 buffer.

    ``dense_forward`` marshals 42 Python objects per call: measured 1.22 us of a
    24 us call at 128:8:64, against the ~0.3 us a pybind entry costs vLLM.
    Everything that only changes with the geometry rides in an int64 buffer the
    call plan owns (``plan`` is its address), so a call passes seven arguments.
    The field order is generated from the same table as
    ``_kernel._PLAN_ABI_FIELDS``, and `tests/test_dense.py` additionally requires
    bit-identical output from both entries on every shape it covers -- which is
    the check the first attempt at this change (two silent field mismatches)
    lacked.
    """
    var pb = MutPointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=plan)
    )
    var tjob = TransposeJob()
    var job = DenseJob()
    tjob.kt = Int(pb.unsafe_load(0))
    tjob.vt = Int(pb.unsafe_load(1))
    tjob.batch = Int(pb.unsafe_load(2))
    tjob.num_kv_heads = Int(pb.unsafe_load(3))
    tjob.seq_k = Int(pb.unsafe_load(4))
    tjob.head_dim = Int(pb.unsafe_load(5))
    tjob.k_stride_b = Int(pb.unsafe_load(6))
    tjob.k_stride_t = Int(pb.unsafe_load(7))
    tjob.k_stride_h = Int(pb.unsafe_load(8))
    tjob.kt_stride_bh = Int(pb.unsafe_load(9))
    tjob.kt_stride_d = Int(pb.unsafe_load(10))
    tjob.kt_blk = Int(pb.unsafe_load(11))
    tjob.v_stride_b = Int(pb.unsafe_load(12))
    tjob.v_stride_t = Int(pb.unsafe_load(13))
    tjob.v_stride_h = Int(pb.unsafe_load(14))
    tjob.vt_stride_bh = Int(pb.unsafe_load(15))
    tjob.vt_stride_d = Int(pb.unsafe_load(16))
    tjob.ks = Int(pb.unsafe_load(17))
    tjob.ks_stride_bh = Int(pb.unsafe_load(18))
    job.kt = Int(pb.unsafe_load(19))
    job.ks = Int(pb.unsafe_load(20))
    job.ks_stride_bh = Int(pb.unsafe_load(21))
    job.batch = Int(pb.unsafe_load(22))
    job.seq_q = Int(pb.unsafe_load(23))
    job.seq_k = Int(pb.unsafe_load(24))
    job.num_heads = Int(pb.unsafe_load(25))
    job.num_kv_heads = Int(pb.unsafe_load(26))
    job.head_dim = Int(pb.unsafe_load(27))
    job.q_stride_b = Int(pb.unsafe_load(28))
    job.q_stride_t = Int(pb.unsafe_load(29))
    job.q_stride_h = Int(pb.unsafe_load(30))
    job.kt_stride_bh = Int(pb.unsafe_load(31))
    job.kt_stride_d = Int(pb.unsafe_load(32))
    job.kt_stride_t = Int(pb.unsafe_load(33))
    job.kt_blk = Int(pb.unsafe_load(34))
    job.v_stride_b = Int(pb.unsafe_load(35))
    job.v_stride_t = Int(pb.unsafe_load(36))
    job.v_stride_h = Int(pb.unsafe_load(37))
    job.o_stride_b = Int(pb.unsafe_load(38))
    job.o_stride_t = Int(pb.unsafe_load(39))
    job.o_stride_h = Int(pb.unsafe_load(40))
    var d_dtype = Int(pb.unsafe_load(41))
    var d_mode = Int(pb.unsafe_load(42))
    var d_qt = Int(pb.unsafe_load(43))
    var d_workers = Int(pb.unsafe_load(44))
    job.causal = Bool(Int(pb.unsafe_load(45)))
    tjob.k = Int(py=k)
    job.q = Int(py=q)
    job.dst = Int(py=dst)
    job.v = Int(py=vkern)
    tjob.v = Int(py=vsrc)
    job.scale = Float32(Float64(py=scale))
    if d_mode >= 8:
        run_small_fused(tjob, job, mode=d_mode - 8, num_workers=d_workers)
    else:
        run_dense_fused(
            tjob, job, dtype_code=d_dtype, mode=d_mode, qt=d_qt,
            num_workers=d_workers,
        )


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[attention]("attention")
        m.def_function[dense_attention]("dense_attention")
        m.def_function[dense_forward]("dense_forward")
        m.def_function[dense_forward_plan]("dense_forward_plan")
        m.def_function[dense_transpose]("dense_transpose")
        m.def_function[dense_bf16_dot]("dense_bf16_dot")
        m.def_function[dense_bfmmla]("dense_bfmmla")
        m.def_function[dense_int8_qk]("dense_int8_qk")
        m.def_function[dense_lanes]("dense_lanes")
        m.def_function[dense_has_amx]("dense_has_amx")
        m.def_function[dense_amx_init]("dense_amx_init")
        m.def_function[dense_amx_attention]("dense_amx_attention")
        return m.finalize()
    except e:
        abort(String("error creating cpu_attention._C: ", e))


def attention(
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    dst: PythonObject,
    slots: PythonObject,
    block_table: PythonObject,
    lens: PythonObject,
    num_seqs: PythonObject,
    num_heads: PythonObject,
    num_kv_heads: PythonObject,
    head_dim: PythonObject,
    block_size: PythonObject,
    bt_stride: PythonObject,
    q_stride: PythonObject,
    kv_stride: PythonObject,
    dst_stride: PythonObject,
    dtype: PythonObject,
    scale: PythonObject,
    num_workers: PythonObject,
    do_store: PythonObject,
    do_attend: PythonObject,
) raises:
    """Run the paged attention kernel (scatter K/V when ``do_store``).

    ``dtype`` is 0 for float32 and 1 for bfloat16;
    ``head_dim`` must be 32, 64 or 128.
    """
    var job = Job()
    job.q = Int(py=q)
    job.k = Int(py=k)
    job.v = Int(py=v)
    job.k_cache = Int(py=k_cache)
    job.v_cache = Int(py=v_cache)
    job.dst = Int(py=dst)
    job.slots = Int(py=slots)
    job.block_table = Int(py=block_table)
    job.lens = Int(py=lens)
    job.num_seqs = Int(py=num_seqs)
    job.num_heads = Int(py=num_heads)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.head_dim = Int(py=head_dim)
    job.block_size = Int(py=block_size)
    job.bt_stride = Int(py=bt_stride)
    job.q_stride = Int(py=q_stride)
    job.kv_stride = Int(py=kv_stride)
    job.dst_stride = Int(py=dst_stride)
    job.scale = Float32(Float64(py=scale))
    run(
        job,
        dtype_code=Int(py=dtype),
        num_workers=Int(py=num_workers),
        do_store=Bool(do_store),
        do_attend=Bool(do_attend),
    )


def dense_attention(
    q: PythonObject,
    kt: PythonObject,
    v: PythonObject,
    dst: PythonObject,
    batch: PythonObject,
    seq_q: PythonObject,
    seq_k: PythonObject,
    num_heads: PythonObject,
    num_kv_heads: PythonObject,
    head_dim: PythonObject,
    q_stride_b: PythonObject,
    q_stride_t: PythonObject,
    q_stride_h: PythonObject,
    kt_stride_bh: PythonObject,
    kt_stride_d: PythonObject,
    kt_stride_t: PythonObject,
    kt_blk: PythonObject,
    v_stride_b: PythonObject,
    v_stride_t: PythonObject,
    v_stride_h: PythonObject,
    o_stride_b: PythonObject,
    o_stride_t: PythonObject,
    o_stride_h: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    dtype: PythonObject,
    qt: PythonObject,
    num_workers: PythonObject,
    ks: PythonObject,
    ks_stride_bh: PythonObject,
) raises:
    """Run the dense (non-paged) attention kernel.

    ``kt`` is K transposed for the Q@K^T sweep (see ``dense.mojo``); ``qt`` is
    the query rows per work item (2, 4 or 8).
    """
    var job = DenseJob()
    job.q = Int(py=q)
    job.kt = Int(py=kt)
    job.v = Int(py=v)
    job.dst = Int(py=dst)
    job.batch = Int(py=batch)
    job.seq_q = Int(py=seq_q)
    job.seq_k = Int(py=seq_k)
    job.num_heads = Int(py=num_heads)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.head_dim = Int(py=head_dim)
    job.q_stride_b = Int(py=q_stride_b)
    job.q_stride_t = Int(py=q_stride_t)
    job.q_stride_h = Int(py=q_stride_h)
    job.kt_stride_bh = Int(py=kt_stride_bh)
    job.kt_stride_d = Int(py=kt_stride_d)
    job.kt_stride_t = Int(py=kt_stride_t)
    job.kt_blk = Int(py=kt_blk)
    job.v_stride_b = Int(py=v_stride_b)
    job.v_stride_t = Int(py=v_stride_t)
    job.v_stride_h = Int(py=v_stride_h)
    job.o_stride_b = Int(py=o_stride_b)
    job.o_stride_t = Int(py=o_stride_t)
    job.o_stride_h = Int(py=o_stride_h)
    job.ks = Int(py=ks)
    job.ks_stride_bh = Int(py=ks_stride_bh)
    job.scale = Float32(Float64(py=scale))
    job.causal = Bool(causal)
    run_dense(
        job,
        dtype_code=Int(py=dtype),
        num_workers=Int(py=num_workers),
        qt=Int(py=qt),
    )
