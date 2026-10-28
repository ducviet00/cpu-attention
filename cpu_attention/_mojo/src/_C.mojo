"""Python extension module for the cpu_attention paged-attention kernel.

Everything crossing the boundary is a raw address / integer, so a call costs a
handful of `PyLong` conversions and no tensor bookkeeping. The Python side
(``cpu_attention``) owns the tensors and passes ``tensor.data_ptr()``.

Build with:  pixi run build        (from ``cpu_attention/_mojo``)
"""

from std.os import abort
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from kvcache import PagedJob, paged_run
from flash_attn import FlashJob
from dispatch import flash_run, flash_sizes as _flash_sizes
from isa import ptr, tier_name
from amx import amx_init
from dense_amx import AMX, AmxJob, PackJob, run_dense_amx
from dense_amx_v2 import (
    AMX2,
    PackJob as PackJob2,
    V2Job,
    run_dense_amx_v2,
)


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


comptime AMX_PLAN_MAGIC = 0x414D58504C414E31
"""'AMXPLAN1' as an int64: identifies a dense_amx plan buffer's layout."""


def dense_amx_plan(
    plan: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    dst: PythonObject,
) raises:
    """AMX pack + attention with the geometry in an int64 buffer (5 arguments).

    ``dense_amx_attention`` marshals 36 Python objects per call, which is 1.2 us
    of a 43 us call at 128:8:64 -- the row where we lose to vLLM and where the
    kernel itself is already faster than theirs, so the call path is the whole
    margin. Everything that changes only with the geometry (strides, tile
    counts, the packed-K/V workspace addresses, kt, the worker count) rides in
    the buffer; only the four tensor addresses and the buffer address cross the
    boundary. The field order below is mirrored by
    ``_kernel._AMX_PLAN_FIELDS`` and the last slot is a magic word, so a stale
    or mismatched plan is rejected instead of silently misread.
    """
    var pb = MutPointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=plan)
    )
    if Int(pb.unsafe_load(32)) != AMX_PLAN_MAGIC:
        raise Error("cpu_attention: stale dense_amx plan buffer")
    var pack = PackJob()
    pack.k = Int(py=k)
    pack.v = Int(py=v)
    pack.kp = Int(pb.unsafe_load(0))
    pack.vp = Int(pb.unsafe_load(1))
    pack.batch = Int(pb.unsafe_load(2))
    pack.num_kv_heads = Int(pb.unsafe_load(4))
    pack.seq_k = Int(pb.unsafe_load(5))
    pack.head_dim = Int(pb.unsafe_load(7))
    pack.k_stride_b = Int(pb.unsafe_load(8))
    pack.k_stride_t = Int(pb.unsafe_load(9))
    pack.k_stride_h = Int(pb.unsafe_load(10))
    pack.v_stride_b = Int(pb.unsafe_load(11))
    pack.v_stride_t = Int(pb.unsafe_load(12))
    pack.v_stride_h = Int(pb.unsafe_load(13))
    pack.kp_bh = Int(pb.unsafe_load(20))
    pack.vp_bh = Int(pb.unsafe_load(21))
    pack.kb16 = Int(pb.unsafe_load(22))
    pack.kb32 = Int(pb.unsafe_load(23))
    pack.dc32 = Int(pb.unsafe_load(24))
    pack.dc16 = Int(pb.unsafe_load(25))
    var job = AmxJob()
    job.q = Int(py=q)
    job.kp = pack.kp
    job.vp = pack.vp
    job.dst = Int(py=dst)
    job.batch = pack.batch
    job.seq_q = Int(pb.unsafe_load(6))
    job.seq_k = pack.seq_k
    job.num_heads = Int(pb.unsafe_load(3))
    job.num_kv_heads = pack.num_kv_heads
    job.head_dim = pack.head_dim
    job.q_stride_b = Int(pb.unsafe_load(14))
    job.q_stride_t = Int(pb.unsafe_load(15))
    job.q_stride_h = Int(pb.unsafe_load(16))
    job.kp_bh = pack.kp_bh
    job.vp_bh = pack.vp_bh
    job.o_stride_b = Int(pb.unsafe_load(17))
    job.o_stride_t = Int(pb.unsafe_load(18))
    job.o_stride_h = Int(pb.unsafe_load(19))
    job.scale = Float32(bitcast[DType.float32](Int32(Int(pb.unsafe_load(26)))))
    job.causal = Bool(Int(pb.unsafe_load(27)))
    job.kt = Int(pb.unsafe_load(28))
    job.dc32 = pack.dc32
    job.dc16 = pack.dc16
    job.fp32 = Bool(Int(pb.unsafe_load(30)))
    job.mode = Int(pb.unsafe_load(31))
    run_dense_amx(
        pack, job, num_workers=Int(pb.unsafe_load(29)), kt=job.kt
    )


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[kv_attention]("kv_attention")
        m.def_function[flash_forward]("flash_forward")
        m.def_function[flash_sizes]("flash_sizes")
        m.def_function[tier]("tier")
        m.def_function[dense_has_amx]("dense_has_amx")
        m.def_function[dense_amx_init]("dense_amx_init")
        m.def_function[dense_amx_attention]("dense_amx_attention")
        m.def_function[dense_amx_plan]("dense_amx_plan")
        m.def_function[dense_amx2_has_amx]("dense_amx2_has_amx")
        m.def_function[dense_amx2_attention]("dense_amx2_attention")
        return m.finalize()
    except e:
        abort(String("error creating cpu_attention._C: ", e))



def tier() raises -> PythonObject:
    """The build tier of this shared object (see `scripts/build.py`)."""
    return PythonObject(tier_name())



def flash_sizes(
    seq_k: PythonObject, rq: PythonObject, hd: PythonObject, dtype: PythonObject, isa: PythonObject
) raises -> PythonObject:
    """(slab bytes per (batch, kv head), scratch bytes per thread)."""
    var s = _flash_sizes(Int(py=seq_k), Int(py=rq), Int(py=hd), Int(py=dtype), Int(py=isa))
    return Python.tuple(s[0], s[1])


def flash_forward(
    plan: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    dst: PythonObject,
    ws: PythonObject,
    scale: PythonObject,
) raises:
    """flash_attn_func from a plan buffer (see `_kernel._flash_plan`)."""
    var job = FlashJob(
        Int(py=plan), Int(py=q), Int(py=k), Int(py=v), Int(py=dst),
        Float32(Float64(py=scale)),
    )
    job.ws = Int(py=ws)
    var p = ptr[Int64](Int(py=plan))
    flash_run(job, Int(p.unsafe_load(18)), Int(p.unsafe_load(5)), Int(p.unsafe_load(27)))

def kv_attention(
    plan: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    dst: PythonObject,
    slots: PythonObject,
    block_table: PythonObject,
    lens: PythonObject,
    q_stride: PythonObject,
    kv_stride: PythonObject,
    scale: PythonObject,
) raises:
    """KV-cache attention / store. `plan` addresses int64 geometry slots:
    num_seqs, num_heads, kv_heads, head_dim, page, bt_stride, out_stride, dtype,
    workers, do_store, do_attend, chunks."""
    var p = ptr[Int64](Int(py=plan))
    var job = PagedJob()
    job.num_seqs = Int(p[0])
    job.num_heads = Int(p[1])
    job.kv_heads = Int(p[2])
    job.head_dim = Int(p[3])
    job.page = Int(p[4])
    job.bt_stride = Int(p[5])
    job.out_stride = Int(p[6])
    job.dtype = Int(p[7])
    job.workers = Int(p[8])
    job.do_store = p[9] != 0
    job.do_attend = p[10] != 0
    job.chunks = Int(p[11])
    job.q = Int(py=q)
    job.k = Int(py=k)
    job.v = Int(py=v)
    job.k_cache = Int(py=k_cache)
    job.v_cache = Int(py=v_cache)
    job.out = Int(py=dst)
    job.slots = Int(py=slots)
    job.block_table = Int(py=block_table)
    job.lens = Int(py=lens)
    job.q_stride = Int(py=q_stride)
    job.kv_stride = Int(py=kv_stride)
    job.scale = Float32(Float64(py=scale))
    paged_run(job)

def dense_amx2_has_amx() -> PythonObject:
    """True when this build has the v2 AMX kernel (x86 + amx-tile + amx-bf16)."""
    if AMX2:
        return PythonObject(True)
    return PythonObject(False)


def dense_amx2_attention(
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
    """v2 of the AMX dense kernel: same arguments, same packed workspace as
    ``dense_amx_attention``, one softmax pass per logit instead of two.

    Selected by ``CPU_ATTENTION_AMX2=1`` in ``cpu_attention._kernel``; v1 stays
    the default so a regression cannot reach the engine.
    """
    var pack = PackJob2()
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

    var job = V2Job()
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
    run_dense_amx_v2(pack, job, num_workers=Int(py=num_workers))
