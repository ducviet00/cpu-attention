"""The Python extension: every argument is an integer or an address (the
`cpu_attention` side owns the tensors and passes `data_ptr()`s), and the
geometry rides in an int64 plan buffer built once per shape."""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from dispatch import flash_run, flash_sizes as _flash_sizes
from flash_attn import FlashJob
from isa import amx_permitted, ptr, tier_name
from paged_attn import PagedJob, paged_run


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[kv_attention]("kv_attention")
        m.def_function[flash_forward]("flash_forward")
        m.def_function[flash_sizes]("flash_sizes")
        m.def_function[tier]("tier")
        m.def_function[amx_available]("amx_available")
        return m.finalize()
    except e:
        abort(String("error creating cpu_attention._C: ", e))


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
    scale: PythonObject,
) raises:
    """Store and/or attend (`paged_attn`). The plan's int64 slots: num_seqs,
    heads, head_dim, page, bt_stride, out_stride, dtype, workers, do_store,
    do_attend, q token stride, k/v token stride (`_kernel.paged_plan`)."""
    var p = ptr[Int64](Int(py=plan))
    paged_run(PagedJob(
        q=Int(py=q), k=Int(py=k), v=Int(py=v), k_cache=Int(py=k_cache), v_cache=Int(py=v_cache),
        out=Int(py=dst), slots=Int(py=slots), block_table=Int(py=block_table), lens=Int(py=lens),
        num_seqs=Int(p[0]), heads=Int(p[1]), head_dim=Int(p[2]), page=Int(p[3]),
        bt_stride=Int(p[4]), q_stride=Int(p[10]), kv_stride=Int(p[11]), out_stride=Int(p[5]),
        scale=Float32(Float64(py=scale)), dtype=Int(p[6]), workers=Int(p[7]),
        do_store=p[8] != 0, do_attend=p[9] != 0,
    ))


def flash_forward(
    plan: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    dst: PythonObject,
    ws: PythonObject,
    scale: PythonObject,
) raises:
    """flash_attn_func from a plan buffer (`_kernel.flash_plan`)."""
    var job = FlashJob(Int(py=plan), Int(py=q), Int(py=k), Int(py=v), Int(py=dst), Float32(Float64(py=scale)))
    job.ws = Int(py=ws)
    var p = ptr[Int64](Int(py=plan))
    flash_run(job, Int(p[17]), Int(p[4]), Int(p[26]))


def flash_sizes(
    seq_k: PythonObject, rq: PythonObject, hd: PythonObject, dtype: PythonObject, isa: PythonObject
) raises -> PythonObject:
    """(slab bytes per (batch, head), scratch bytes per thread)."""
    var s = _flash_sizes(Int(py=seq_k), Int(py=rq), Int(py=hd), Int(py=dtype), Int(py=isa))
    return Python.tuple(s[0], s[1])


def tier() raises -> PythonObject:
    """The build tier of this shared object (`scripts/build.py`)."""
    return PythonObject(tier_name())


def amx_available() -> PythonObject:
    """The OS grants AMX tile data (`arch_prctl`); False on tiers without AMX."""
    return PythonObject(amx_permitted())
