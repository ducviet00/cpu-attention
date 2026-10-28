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
from std.ffi import external_call
from pyview import data_ptr, dims, dtype_code, hdr_at, impl, is_tensor, size, stride, word
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
        m.def_function[kv_fast]("kv_fast")
        m.def_function[tensor_probe]("tensor_probe")
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


comptime KV_ENTRY0 = 16  # first plan entry slot in the kv_fast table
comptime KV_ENTRY_W = 32  # slots per entry


def tensor_probe(table: PythonObject, t: PythonObject) raises -> PythonObject:
    """(storage base, dim, sizes..., strides..., dtype code, storage offset,
    accepted type) as this module reads
    them; the Python calibration compares it with torch before enabling
    `kv_fast`."""
    var hdr = Int(py=table)
    var im = impl(Int(t._obj_ptr))
    var n = dims(hdr, im)
    var out = Python.list()
    out.append(word(word(im + hdr_at(hdr, 1)) + hdr_at(hdr, 2)))  # storage base
    out.append(n)
    for i in range(min(n, 5)):
        out.append(size(hdr, im, i))
    for i in range(min(n, 5)):
        out.append(stride(hdr, im, i))
    out.append(dtype_code(hdr, im))
    out.append(word(im + hdr_at(hdr, 3)))
    out.append(is_tensor(hdr, Int(t._obj_ptr)))
    return out


@always_inline
def _match3(e: Int, at: Int, hdr: Int, im: Int, strides: Bool) -> Bool:
    comptime for i in range(3):
        var x = stride(hdr, im, i) if strides else size(hdr, im, i)
        if x != word(e + 8 * (at + i)):
            return False
    return True


def kv_fast(
    table: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    slots: PythonObject,
    block_table: PythonObject,
    lens: PythonObject,
    scale: PythonObject,
    dst: PythonObject,
) raises -> PythonObject:
    """KV-cache attention with the tensors read here (see pyview.mojo): True if
    a plan in `table` matched this call's layout and it ran, False for the
    caller's checked path (which then registers a plan)."""
    var hdr = Int(py=table)
    var none = hdr_at(hdr, 10)
    var qo = Int(q._obj_ptr)
    var kco = Int(k_cache._obj_ptr)
    var vco = Int(v_cache._obj_ptr)
    var bto = Int(block_table._obj_ptr)
    var lo = Int(lens._obj_ptr)
    var oo = Int(dst._obj_ptr)
    var ko = Int(k._obj_ptr)
    var store = ko != none
    if not (is_tensor(hdr, qo) and is_tensor(hdr, kco) and is_tensor(hdr, vco)
            and is_tensor(hdr, bto) and is_tensor(hdr, lo) and is_tensor(hdr, oo)):
        return False
    var qi = impl(qo)
    var kci = impl(kco)
    var vci = impl(vco)
    var bti = impl(bto)
    var li = impl(lo)
    var oi = impl(oo)
    if dims(hdr, qi) != 3 or dims(hdr, kci) != 4 or dims(hdr, vci) != 4 or dims(hdr, bti) != 2 or dims(hdr, oi) != 3:
        return False
    var code = dtype_code(hdr, qi)
    var i32 = hdr_at(hdr, 12)
    if dtype_code(hdr, kci) != code or dtype_code(hdr, vci) != code or dtype_code(hdr, oi) != code:
        return False
    if dtype_code(hdr, bti) != i32 or dtype_code(hdr, li) != i32:
        return False
    # The kernel writes a contiguous [seqs, heads, hd] output and reads V with
    # the K cache's geometry.
    var nh = size(hdr, qi, 1)
    var hd = size(hdr, qi, 2)
    if size(hdr, oi, 0) != size(hdr, qi, 0) or size(hdr, oi, 1) != nh or size(hdr, oi, 2) != hd:
        return False
    if stride(hdr, oi, 0) != nh * hd or stride(hdr, oi, 1) != hd or stride(hdr, oi, 2) != 1:
        return False
    comptime for i in range(4):
        if size(hdr, vci, i) != size(hdr, kci, i) or stride(hdr, vci, i) != stride(hdr, kci, i):
            return False
    var vo = Int(v._obj_ptr)
    var so = Int(slots._obj_ptr)
    var ki = 0
    var vi = 0
    var si = 0
    if store:
        if not (is_tensor(hdr, ko) and is_tensor(hdr, vo) and is_tensor(hdr, so)):
            return False
        ki = impl(ko)
        vi = impl(vo)
        si = impl(so)
        if dims(hdr, ki) != 3 or dims(hdr, vi) != 3 or dtype_code(hdr, ki) != code or dtype_code(hdr, vi) != code:
            return False
        if dtype_code(hdr, si) != hdr_at(hdr, 13):
            return False
    var threads = Int(external_call["omp_get_max_threads", Int32]())
    var n = hdr_at(hdr, 11)
    for e_i in range(n):
        var e = hdr + 8 * (KV_ENTRY0 + KV_ENTRY_W * e_i)
        if word(e) != code or word(e + 8) != Int(store) or word(e + 16) != threads:
            continue
        if not (_match3(e, 3, hdr, qi, False) and _match3(e, 6, hdr, qi, True)):
            continue
        var ok = True
        comptime for i in range(4):
            if size(hdr, kci, i) != word(e + 8 * (9 + i)):
                ok = False
        if not ok or size(hdr, bti, 0) != word(e + 8 * 13) or size(hdr, bti, 1) != word(e + 8 * 14):
            continue
        if store and not (_match3(e, 15, hdr, ki, True) and _match3(e, 18, hdr, vi, True)):
            continue
        var item = word(e + 8 * 24)
        var p = ptr[Int64](word(e + 8 * 21))
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
        job.q = data_ptr(hdr, qi, item)
        job.k_cache = data_ptr(hdr, kci, item)
        job.v_cache = data_ptr(hdr, vci, item)
        job.out = data_ptr(hdr, oi, item)
        job.block_table = data_ptr(hdr, bti, 4)
        job.lens = data_ptr(hdr, li, 4)
        job.q_stride = word(e + 8 * 22)
        job.kv_stride = word(e + 8 * 23)
        if store:
            job.k = data_ptr(hdr, ki, item)
            job.v = data_ptr(hdr, vi, item)
            job.slots = data_ptr(hdr, si, 8)
        job.scale = Float32(Float64(py=scale))
        paged_run(job)
        return True
    return False

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
