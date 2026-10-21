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
    DenseJob,
    TransposeJob,
    has_bf16_dot,
    has_bfmmla,
    lane_width,
    run_dense,
    run_transpose,
)


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
    v_stride_b: PythonObject,
    v_stride_t: PythonObject,
    v_stride_h: PythonObject,
    vt_stride_bh: PythonObject,
    vt_stride_d: PythonObject,
    dtype: PythonObject,
    mode: PythonObject,
    num_workers: PythonObject,
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
    job.v_stride_b = Int(py=v_stride_b)
    job.v_stride_t = Int(py=v_stride_t)
    job.v_stride_h = Int(py=v_stride_h)
    job.vt_stride_bh = Int(py=vt_stride_bh)
    job.vt_stride_d = Int(py=vt_stride_d)
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
    """True when the bf16 path runs on NEON ``bfmmla`` (K and V both repacked)."""
    if has_bfmmla():
        return PythonObject(True)
    return PythonObject(False)


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[attention]("attention")
        m.def_function[dense_attention]("dense_attention")
        m.def_function[dense_transpose]("dense_transpose")
        m.def_function[dense_bf16_dot]("dense_bf16_dot")
        m.def_function[dense_bfmmla]("dense_bfmmla")
        m.def_function[dense_lanes]("dense_lanes")
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
    job.v_stride_b = Int(py=v_stride_b)
    job.v_stride_t = Int(py=v_stride_t)
    job.v_stride_h = Int(py=v_stride_h)
    job.o_stride_b = Int(py=o_stride_b)
    job.o_stride_t = Int(py=o_stride_t)
    job.o_stride_h = Int(py=o_stride_h)
    job.scale = Float32(Float64(py=scale))
    job.causal = Bool(causal)
    run_dense(
        job,
        dtype_code=Int(py=dtype),
        num_workers=Int(py=num_workers),
        qt=Int(py=qt),
    )
