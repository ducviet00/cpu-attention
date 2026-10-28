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
from isa import amx_permitted, ptr, tier_name
from std.ffi import external_call
from pyview import data_ptr, dims, dtype_code, hdr_at, impl, is_tensor, size, stride, word


def amx_available() -> PythonObject:
    """True when the OS grants AMX tile-data permission (`arch_prctl`); False on
    every tier that carries no AMX policy. Called once per process from
    `cpu_attention._kernel.amx_available()`, which also checks the CPU flags."""
    return PythonObject(amx_permitted())


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[kv_attention]("kv_attention")
        m.def_function[kv_fast]("kv_fast")
        m.def_function[flash_fast]("flash_fast")
        m.def_function[tensor_probe]("tensor_probe")
        m.def_function[flash_forward]("flash_forward")
        m.def_function[flash_sizes]("flash_sizes")
        m.def_function[tier]("tier")
        m.def_function[amx_available]("amx_available")
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


comptime KV_ENTRY0 = 24  # first plan entry slot (after the pyview header)
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


@always_inline
def _match3_d(e: Int, at: Int, hdr: Int, im: Int, strides: Bool, d4: Bool) -> Bool:
    var i0 = 0
    var i1 = 2 if d4 else 1
    var i2 = 3 if d4 else 2
    var x0 = stride(hdr, im, i0) if strides else size(hdr, im, i0)
    var x1 = stride(hdr, im, i1) if strides else size(hdr, im, i1)
    var x2 = stride(hdr, im, i2) if strides else size(hdr, im, i2)
    return (
        x0 == word(e + 8 * at)
        and x1 == word(e + 8 * (at + 1))
        and x2 == word(e + 8 * (at + 2))
    )


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
    """KV-cache attention / store with the tensors read here (see pyview.mojo):
    True if a plan in `table` matched this call's layout and it ran, False for
    the caller's checked path (which then registers a plan)."""
    var hdr = Int(py=table)
    var none = hdr_at(hdr, 10)
    var kco = Int(k_cache._obj_ptr)
    var vco = Int(v_cache._obj_ptr)
    var oo = Int(dst._obj_ptr)
    var ko = Int(k._obj_ptr)
    var vo = Int(v._obj_ptr)
    var so = Int(slots._obj_ptr)
    var threads = Int(external_call["omp_get_max_threads", Int32]())
    var n = hdr_at(hdr, 11)

    if oo == none:
        # Store-only fast path (`reshape_and_cache`).
        if not (
            is_tensor(hdr, ko)
            and is_tensor(hdr, vo)
            and is_tensor(hdr, kco)
            and is_tensor(hdr, vco)
            and is_tensor(hdr, so)
        ):
            return False
        var ki = impl(ko)
        var vi = impl(vo)
        var kci = impl(kco)
        var vci = impl(vco)
        var si = impl(so)
        if (
            dims(hdr, ki) != 3
            or dims(hdr, vi) != 3
            or dims(hdr, kci) != 4
            or dims(hdr, vci) != 4
            or dims(hdr, si) != 1
        ):
            return False
        var code = dtype_code(hdr, ki)
        if (
            dtype_code(hdr, vi) != code
            or dtype_code(hdr, kci) != code
            or dtype_code(hdr, vci) != code
            or dtype_code(hdr, si) != hdr_at(hdr, 13)
        ):
            return False
        if stride(hdr, si, 0) != 1 or size(hdr, si, 0) < size(hdr, ki, 0):
            return False
        comptime for i in range(3):
            if size(hdr, vi, i) != size(hdr, ki, i):
                return False
        comptime for i in range(4):
            if size(hdr, vci, i) != size(hdr, kci, i) or stride(
                hdr, vci, i
            ) != stride(hdr, kci, i):
                return False
        for e_i in range(n - 1, -1, -1):
            var e = hdr + 8 * (KV_ENTRY0 + KV_ENTRY_W * e_i)
            if (
                word(e) != code
                or word(e + 8) != 3
                or word(e + 16) != threads
            ):
                continue
            if not (
                _match3(e, 3, hdr, ki, False)
                and _match3(e, 15, hdr, ki, True)
                and _match3(e, 18, hdr, vi, True)
            ):
                continue
            var ok = True
            comptime for i in range(4):
                if size(hdr, kci, i) != word(e + 8 * (9 + i)):
                    ok = False
            if not ok:
                continue
            var item = word(e + 8 * 24)
            var p = ptr[Int64](word(e + 8 * 21))
            var job = PagedJob()
            job.num_seqs = Int(p[0])
            job.num_heads = Int(p[1])
            job.kv_heads = Int(p[2])
            job.head_dim = Int(p[3])
            job.page = Int(p[4])
            job.dtype = Int(p[7])
            job.workers = Int(p[8])
            job.do_store = True
            job.do_attend = False
            job.chunks = 1
            job.k = data_ptr(hdr, ki, item)
            job.v = data_ptr(hdr, vi, item)
            job.k_cache = data_ptr(hdr, kci, item)
            job.v_cache = data_ptr(hdr, vci, item)
            job.slots = data_ptr(hdr, si, 8)
            job.kv_stride = word(e + 8 * 23)
            paged_run(job)
            return True
        return False

    var qo = Int(q._obj_ptr)
    var bto = Int(block_table._obj_ptr)
    var lo = Int(lens._obj_ptr)
    var store = ko != none
    if not (
        is_tensor(hdr, qo)
        and is_tensor(hdr, kco)
        and is_tensor(hdr, vco)
        and is_tensor(hdr, bto)
        and is_tensor(hdr, lo)
        and is_tensor(hdr, oo)
    ):
        return False
    var qi = impl(qo)
    var kci = impl(kco)
    var vci = impl(vco)
    var bti = impl(bto)
    var li = impl(lo)
    var oi = impl(oo)
    var qd = dims(hdr, qi)
    var d4 = qd == 4
    if (
        (qd != 3 and qd != 4)
        or dims(hdr, kci) != 4
        or dims(hdr, vci) != 4
        or dims(hdr, bti) != 2
        or dims(hdr, oi) != qd
        or dims(hdr, li) != 1
    ):
        return False
    if d4 and (size(hdr, qi, 1) != 1 or size(hdr, oi, 1) != 1):
        return False
    if stride(hdr, bti, 1) != 1 or stride(hdr, li, 0) != 1:
        return False
    var nseq = size(hdr, qi, 0)
    if size(hdr, li, 0) < nseq:
        return False
    var code = dtype_code(hdr, qi)
    var i32 = hdr_at(hdr, 12)
    if (
        dtype_code(hdr, kci) != code
        or dtype_code(hdr, vci) != code
        or dtype_code(hdr, oi) != code
    ):
        return False
    if dtype_code(hdr, bti) != i32 or dtype_code(hdr, li) != i32:
        return False
    # The kernel writes a contiguous [seqs, (1,) heads, hd] output and reads V
    # with the K cache's geometry.
    var nh = size(hdr, qi, 2 if d4 else 1)
    var hd = size(hdr, qi, 3 if d4 else 2)
    if (
        size(hdr, oi, 0) != nseq
        or size(hdr, oi, 2 if d4 else 1) != nh
        or size(hdr, oi, 3 if d4 else 2) != hd
    ):
        return False
    if (
        stride(hdr, oi, 0) != nh * hd
        or stride(hdr, oi, 2 if d4 else 1) != hd
        or stride(hdr, oi, 3 if d4 else 2) != 1
    ):
        return False
    comptime for i in range(4):
        if size(hdr, vci, i) != size(hdr, kci, i) or stride(
            hdr, vci, i
        ) != stride(hdr, kci, i):
            return False
    var ki = 0
    var vi = 0
    var si = 0
    var mode = 0
    if store:
        if not (is_tensor(hdr, ko) and is_tensor(hdr, vo)):
            return False
        ki = impl(ko)
        vi = impl(vo)
        if (
            dims(hdr, ki) != qd
            or dims(hdr, vi) != qd
            or dtype_code(hdr, ki) != code
            or dtype_code(hdr, vi) != code
        ):
            return False
        if d4 and (size(hdr, ki, 1) != 1 or size(hdr, vi, 1) != 1):
            return False
        var kvh = size(hdr, kci, 1)
        if (
            size(hdr, ki, 0) != nseq
            or size(hdr, ki, 2 if d4 else 1) != kvh
            or size(hdr, ki, 3 if d4 else 2) != hd
            or size(hdr, vi, 0) != nseq
            or size(hdr, vi, 2 if d4 else 1) != kvh
            or size(hdr, vi, 3 if d4 else 2) != hd
        ):
            return False
        if so != none:
            if not is_tensor(hdr, so):
                return False
            si = impl(so)
            if (
                dtype_code(hdr, si) != hdr_at(hdr, 13)
                or dims(hdr, si) != 1
                or stride(hdr, si, 0) != 1
                or size(hdr, si, 0) < nseq
            ):
                return False
            mode = 1
        else:
            if size(hdr, li, 0) != nseq:
                return False
            mode = 2
    else:
        if vo != none or so != none:
            return False
        if d4 and size(hdr, li, 0) != nseq:
            return False
    var sc_obj = Int(scale._obj_ptr)
    var default_scale = sc_obj == none
    var sval = Float32(0)
    if not default_scale:
        if word(sc_obj + 8) != hdr_at(hdr, 16):
            return False
        sval = Float32(Float64(py=scale))
    for e_i in range(n - 1, -1, -1):
        var e = hdr + 8 * (KV_ENTRY0 + KV_ENTRY_W * e_i)
        if word(e) != code or word(e + 8) != mode or word(e + 16) != threads:
            continue
        if not (
            _match3_d(e, 3, hdr, qi, False, d4)
            and _match3_d(e, 6, hdr, qi, True, d4)
        ):
            continue
        var ok = True
        comptime for i in range(4):
            if size(hdr, kci, i) != word(e + 8 * (9 + i)):
                ok = False
        if (
            not ok
            or size(hdr, bti, 0) != word(e + 8 * 13)
            or size(hdr, bti, 1) != word(e + 8 * 14)
        ):
            continue
        if store and not (
            _match3_d(e, 15, hdr, ki, True, d4)
            and _match3_d(e, 18, hdr, vi, True, d4)
        ):
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
            job.slots = data_ptr(hdr, si, 8) if mode == 1 else 0
        job.scale = (
            ptr[Float32](e + 8 * 25).unsafe_load(0)
            if default_scale
            else sval
        )
        paged_run(job)
        return True
    return False


comptime FL_ENTRY0 = 24  # first plan entry slot (after the pyview header)
comptime FL_ENTRY_W = 32


@always_inline
def _match4(e: Int, at: Int, hdr: Int, im: Int, strides: Bool) -> Bool:
    comptime for i in range(4):
        var x = stride(hdr, im, i) if strides else size(hdr, im, i)
        if x != word(e + 8 * (at + i)):
            return False
    return True


def flash_fast(
    table: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    dst: PythonObject,
) raises -> PythonObject:
    """flash_attn_func with the tensors read here: True if a plan registered by
    a validated call matches this call exactly (layout, dtype, causal, scale,
    threads) and it ran; False for the caller's checked path."""
    var hdr = Int(py=table)
    var qo = Int(q._obj_ptr)
    var ko = Int(k._obj_ptr)
    var vo = Int(v._obj_ptr)
    var oo = Int(dst._obj_ptr)
    if not (is_tensor(hdr, qo) and is_tensor(hdr, ko) and is_tensor(hdr, vo) and is_tensor(hdr, oo)):
        return False
    var qi = impl(qo)
    var ki = impl(ko)
    var vi = impl(vo)
    var oi = impl(oo)
    if dims(hdr, qi) != 4 or dims(hdr, ki) != 4 or dims(hdr, vi) != 4 or dims(hdr, oi) != 4:
        return False
    var code = dtype_code(hdr, qi)
    if dtype_code(hdr, ki) != code or dtype_code(hdr, vi) != code or dtype_code(hdr, oi) != code:
        return False
    var row = size(hdr, qi, 2) * size(hdr, qi, 3)
    comptime for i in range(4):
        if size(hdr, oi, i) != size(hdr, qi, i) or size(hdr, vi, i) != size(hdr, ki, i):
            return False
    if (stride(hdr, oi, 3) != 1 or stride(hdr, oi, 2) != size(hdr, qi, 3)
            or stride(hdr, oi, 1) != row or stride(hdr, oi, 0) != row * size(hdr, qi, 1)):
        return False
    var co = Int(causal._obj_ptr)
    var cflag = 1 if co == hdr_at(hdr, 14) else (0 if co == hdr_at(hdr, 15) else -1)
    if cflag < 0:
        return False
    var so = Int(scale._obj_ptr)
    var default_scale = so == hdr_at(hdr, 10)
    var sval = Float64(0)
    if not default_scale:
        if word(so + 8) != hdr_at(hdr, 16):  # exactly a Python float
            return False
        sval = Float64(py=scale)
    var threads = Int(external_call["omp_get_max_threads", Int32]())
    var n = hdr_at(hdr, 11)
    for e_i in range(n - 1, -1, -1):
        var e = hdr + 8 * (FL_ENTRY0 + FL_ENTRY_W * e_i)
        if word(e) != code or word(e + 8) != cflag or word(e + 16) != threads:
            continue
        if default_scale:
            if word(e + 24) != 1:
                continue
        elif word(e + 24) != 0 or ptr[Float64](e + 32).unsafe_load(0) != sval:
            continue
        if not (_match4(e, 5, hdr, qi, False) and _match4(e, 9, hdr, qi, True)
                and _match4(e, 13, hdr, ki, False) and _match4(e, 17, hdr, ki, True)
                and _match4(e, 21, hdr, vi, True)):
            continue
        var item = word(e + 8 * 27)
        var plan = word(e + 8 * 25)
        var job = FlashJob(
            plan, data_ptr(hdr, qi, item), data_ptr(hdr, ki, item),
            data_ptr(hdr, vi, item), data_ptr(hdr, oi, item),
            ptr[Float32](e + 8 * 28).unsafe_load(0),
        )
        job.ws = word(e + 8 * 26)
        var p = ptr[Int64](plan)
        flash_run(job, Int(p.unsafe_load(18)), Int(p.unsafe_load(5)), Int(p.unsafe_load(27)))
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
