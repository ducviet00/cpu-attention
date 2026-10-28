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
