"""The Python extension: one native entry per op.

Each entry reads its tensor arguments in place (`torch_tensor`), validates them,
computes the launch parameters, allocates the output and runs the kernel; there
is no plan cache. It returns None for anything it does not take (a wrong shape
or dtype, a tensor subclass, a layout that needs a copy): the Python caller then
raises the precise error, or copies and calls again.
"""

from std.ffi import external_call
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from dispatch import flash_run, flash_sizes
from flash_attn import FlashJob
from isa import NEON_BF16, amx_permitted, ptr, tier_name
from paged_attn import PagedJob, paged_run
from torch_tensor import AMX, BF16, CORES, F32, FALSE, I32, I64, LLC, NONE, TRUE, Tensor, cfg, dims4, empty


@export
def PyInit__C() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_C")
        m.def_function[decode]("decode")
        m.def_function[store]("store")
        m.def_function[flash]("flash")
        m.def_function[probe]("probe")
        m.def_function[tier]("tier")
        m.def_function[amx_available]("amx_available")
        return m.finalize()
    except e:
        abort(String("error creating cpu_attention._C: ", e))


def _workers(rt: Int, tasks: Int, pages: Int) -> Int:
    """Threads for one call. The kernel forks on torch's OpenMP pool (~2.4 us),
    which beats running inline from about 32 KV positions per call (batch 1, ctx
    64: 24.5 us inline vs 31.0 pooled; ctx 128: the pool 1.3x faster). Capped at
    physical cores: SMT threads only oversubscribe (8 on a 4-core Xeon: 1.5x
    slower at ctx 1024)."""
    var threads = Int(external_call["omp_get_max_threads", Int32]())
    var n = min(threads, cfg(rt, CORES), tasks)
    return 1 if n > 1 and tasks * pages <= 32 else max(n, 1)


def _dtype(rt: Int, t: Tensor) -> Int:
    """0 fp32, 1 bf16, -1 anything else."""
    var code = t.dtype()
    return 0 if code == cfg(rt, F32) else (1 if code == cfg(rt, BF16) else -1)


def _rows_ok(t: Tensor, hd: Int) -> Bool:
    """Heads and dims contiguous (any token stride: chunks of a fused projection)."""
    var n = t.dim()
    return t.stride(n - 1) == 1 and t.stride(n - 2) == hd


def _caches_ok(kc: Tensor, vc: Tensor, code: Int, heads: Int, hd: Int) -> Bool:
    """`[blocks, heads, page, dim]`, contiguous, matching, page a multiple of 16."""
    if not (kc.ok() and vc.ok()) or kc.dim() != 4 or vc.dim() != 4:
        return False
    for i in range(4):
        if vc.size(i) != kc.size(i):
            return False
    return (kc.dtype() == code and vc.dtype() == code and kc.size(1) == heads
            and kc.size(3) == hd and kc.size(2) > 0 and kc.size(2) % 16 == 0
            and kc.contiguous() and vc.contiguous())


def _int_vector(rt: Int, t: Tensor, code: Int, n: Int) -> Bool:
    return t.ok() and t.dim() == 1 and t.dtype() == cfg(rt, code) and t.size(0) == n and t.contiguous()


def decode(
    rt: PythonObject,
    q: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    k: PythonObject,
    v: PythonObject,
    slots: PythonObject,
    lens: PythonObject,
    page_table: PythonObject,
    scale: PythonObject,
    dst: PythonObject,
) raises -> PythonObject:
    """Paged decode of q `[seqs, (1,) heads, dim]` (MHA). With k/v, each sequence's
    new token is stored first: at `slots` (then `lens` is the length after the
    store), or appended at `lens` (the length before). Writes `dst` when given,
    else a new tensor; returns it, or None (see the module docstring)."""
    var r = Int(py=rt)
    var none = cfg(r, NONE)
    var qt = Tensor.of(r, Int(q._obj_ptr))
    var kc = Tensor.of(r, Int(k_cache._obj_ptr))
    var vc = Tensor.of(r, Int(v_cache._obj_ptr))
    var bt = Tensor.of(r, Int(page_table._obj_ptr))
    var ln = Tensor.of(r, Int(lens._obj_ptr))
    if not (qt.ok() and bt.ok()):
        return PythonObject(None)
    var nd = qt.dim()
    if (nd != 3 and nd != 4) or (nd == 4 and qt.size(1) != 1):
        return PythonObject(None)
    var b = qt.size(0)
    var heads = qt.size(nd - 2)
    var hd = qt.size(nd - 1)
    var dt = _dtype(r, qt)
    if dt < 0 or (hd != 32 and hd != 64 and hd != 128) or not _rows_ok(qt, hd):
        return PythonObject(None)
    if not _caches_ok(kc, vc, qt.dtype(), heads, hd):
        return PythonObject(None)
    var page = kc.size(2)
    if bt.dim() != 2 or bt.dtype() != cfg(r, I32) or bt.size(0) != b or not bt.contiguous():
        return PythonObject(None)
    var width = bt.size(1)
    if not _int_vector(r, ln, I32, b):
        return PythonObject(None)
    var do_store = Int(k._obj_ptr) != none
    var kt = Tensor.of(r, Int(k._obj_ptr))
    var vt = Tensor.of(r, Int(v._obj_ptr))
    if do_store:
        if not (kt.ok() and vt.ok()) or kt.dim() != nd or vt.dim() != nd or kt.dtype() != qt.dtype():
            return PythonObject(None)
        for i in range(nd):
            if kt.size(i) != qt.size(i) or vt.size(i) != qt.size(i) or vt.stride(i) != kt.stride(i):
                return PythonObject(None)
        if vt.dtype() != qt.dtype() or not _rows_ok(kt, hd):
            return PythonObject(None)
    elif Int(v._obj_ptr) != none:
        return PythonObject(None)
    var sl = 0
    if Int(slots._obj_ptr) != none:
        var st = Tensor.of(r, Int(slots._obj_ptr))
        if not do_store or not _int_vector(r, st, I64, b):
            return PythonObject(None)
        sl = st.data(8)
        for i in range(b):
            if Int(ptr[Int64](sl)[i]) >= kc.size(0) * page:
                return PythonObject(None)
    # Every length must fit the page table (the kernel reads it without checks).
    var extra = 1 if do_store and sl == 0 else 0
    var lp = ptr[Int32](ln.data(4))
    for i in range(b):
        if Int(lp[i]) + extra > width * page:
            return PythonObject(None)
    var out = dst
    if Int(dst._obj_ptr) == none:
        var sizes = dims4(b, 1, heads, hd) if nd == 4 else dims4(b, heads, hd)
        out = PythonObject(from_owned=empty(nd, sizes, dt == 1))
    var ot = Tensor.of(r, Int(out._obj_ptr))
    if not ot.ok() or ot.dim() != nd or ot.dtype() != qt.dtype() or not ot.contiguous():
        return PythonObject(None)
    for i in range(nd):
        if ot.size(i) != qt.size(i):
            return PythonObject(None)
    var e = 2 if dt == 1 else 4
    paged_run(PagedJob(
        q=qt.data(e), k=kt.data(e) if do_store else 0, v=vt.data(e) if do_store else 0,
        k_cache=kc.data(e), v_cache=vc.data(e), out=ot.data(e), slots=sl,
        block_table=bt.data(4), lens=ln.data(4), num_seqs=b, heads=heads, head_dim=hd,
        page=page, bt_stride=width, q_stride=qt.stride(0),
        kv_stride=kt.stride(0) if do_store else 0, out_stride=heads * hd,
        scale=Float32(Float64(hd) ** -0.5) if Int(scale._obj_ptr) == none else Float32(Float64(py=scale)),
        dtype=dt, workers=_workers(r, b * heads, max(width, 1)), do_store=do_store, do_attend=True,
    ))
    return out


def store(
    rt: PythonObject,
    key: PythonObject,
    value: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    slots: PythonObject,
) raises -> PythonObject:
    """`reshape_and_cache`: token i of key/value `[tokens, heads, dim]` goes to
    slot `slots[i]` (int64; negative skips it). True, or None."""
    var r = Int(py=rt)
    var kt = Tensor.of(r, Int(key._obj_ptr))
    var vt = Tensor.of(r, Int(value._obj_ptr))
    var kc = Tensor.of(r, Int(k_cache._obj_ptr))
    var vc = Tensor.of(r, Int(v_cache._obj_ptr))
    var st = Tensor.of(r, Int(slots._obj_ptr))
    if not (kt.ok() and vt.ok()) or kt.dim() != 3 or vt.dim() != 3:
        return PythonObject(None)
    var n = kt.size(0)
    var heads = kt.size(1)
    var hd = kt.size(2)
    var dt = _dtype(r, kt)
    if dt < 0 or (hd != 32 and hd != 64 and hd != 128) or vt.dtype() != kt.dtype() or not _rows_ok(kt, hd):
        return PythonObject(None)
    for i in range(3):
        if vt.size(i) != kt.size(i) or vt.stride(i) != kt.stride(i):
            return PythonObject(None)
    if not _caches_ok(kc, vc, kt.dtype(), heads, hd) or not _int_vector(r, st, I64, n):
        return PythonObject(None)
    var sp = ptr[Int64](st.data(8))
    for i in range(n):
        if Int(sp[i]) >= kc.size(0) * kc.size(2):
            return PythonObject(None)
    var e = 2 if dt == 1 else 4
    paged_run(PagedJob(
        q=0, k=kt.data(e), v=vt.data(e), k_cache=kc.data(e), v_cache=vc.data(e), out=0,
        slots=Int(sp), block_table=0, lens=0, num_seqs=n, heads=heads, head_dim=hd,
        page=kc.size(2), bt_stride=0, q_stride=0, kv_stride=kt.stride(0), out_stride=0,
        scale=1.0, dtype=dt, workers=_workers(r, n * heads, 1), do_store=True, do_attend=False,
    ))
    return PythonObject(True)


def _vector_beats_amx(dt: Int, hd: Int, seq_q: Int, seq_k: Int, scale: Float64) -> Bool:
    """Few query rows leave AMX's 16-row tiles mostly empty (and fp32 pays three
    products per phase), so the vector policy is faster there. Crossovers on the
    Xeon 8581C, 4 threads (vector / AMX time): fp32 hd=64 up to 65 rows 0.39-0.94,
    fp32 hd=128 up to 17 rows 0.80-0.97 (or a 33-key sequence: 0.59), bf16 1-2
    rows 0.76-0.89 (hd=128 only while seq_k <= 256: AMX packs a long K/V faster)."""
    if dt == 0:
        # A logit std above ~6 (scale * sqrt(hd) for unit inputs) peaks the
        # probabilities enough for the hi/lo split to reach 2.03e-4 against the
        # 2e-4 gate (scale 1.7 at hd=64): exact fp32 on the vector policy.
        if scale * Float64(hd) ** 0.5 > 6.0:
            return True
        return seq_q <= (65 if hd <= 64 else 24) or seq_k <= 64
    return seq_q <= 2 and (hd <= 64 or seq_k <= 256)


def flash(
    rt: PythonObject,
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    ws: PythonObject,
    ws_bytes: PythonObject,
) raises -> PythonObject:
    """flash_attn_func over q/k/v `[batch, seq, heads, dim]` (MHA, dims
    contiguous) with the workspace at `ws`. Returns the output, the workspace bytes
    it needs when `ws_bytes` is short (an int), or None."""
    var r = Int(py=rt)
    var qt = Tensor.of(r, Int(q._obj_ptr))
    var kt = Tensor.of(r, Int(k._obj_ptr))
    var vt = Tensor.of(r, Int(v._obj_ptr))
    if not (qt.ok() and kt.ok() and vt.ok()) or qt.dim() != 4 or kt.dim() != 4 or vt.dim() != 4:
        return PythonObject(None)
    var b = qt.size(0)
    var sq = qt.size(1)
    var sk = kt.size(1)
    var heads = qt.size(2)
    var hd = qt.size(3)
    var dt = _dtype(r, qt)
    if dt < 0 or (hd != 32 and hd != 64 and hd != 128) or sq < 1 or sk < 1:
        return PythonObject(None)
    if kt.size(0) != b or kt.size(2) != heads or kt.size(3) != hd or kt.dtype() != qt.dtype() or vt.dtype() != qt.dtype():
        return PythonObject(None)
    for i in range(4):
        if vt.size(i) != kt.size(i):
            return PythonObject(None)
    if qt.stride(3) != 1 or kt.stride(3) != 1 or vt.stride(3) != 1:
        return PythonObject(None)
    var cp = Int(causal._obj_ptr)
    if cp != cfg(r, TRUE) and cp != cfg(r, FALSE):
        return PythonObject(None)
    var is_causal = cp == cfg(r, TRUE)
    var sc = Float64(hd) ** -0.5 if Int(scale._obj_ptr) == cfg(r, NONE) else Float64(py=scale)
    # Policy: 0 vector, 1 AMX (fp32 as bf16 hi/lo), 2 aarch64 bfmmla (fp32 on the vector policy).
    var isa = 0
    comptime if NEON_BF16:
        isa = 2
    else:
        var amx = cfg(r, AMX)
        if (amx == 2 or (amx == 1 and dt == 1)) and not _vector_beats_amx(dt, hd, sq, sk, sc):
            isa = 1
    var rq: Int
    if isa == 1:
        rq = 64  # two 32-row AMX register tiles: K/V tiles load once per 64 rows
    elif isa == 2 and dt == 0 and hd == 128 and is_causal and sq == sk and sq >= 2048:
        rq = 128  # Axion: amortize the K/V scan on long fp32 D=128 (6-9%; short rows lost 9.5%)
    else:
        rq = max(16, min(128, 16384 // (hd * 4)))  # a ~16 KB accumulator stays in L1
    # At least ~8 work items per worker: causal blocks differ in cost, and at 4
    # per worker the tail cost 1024:8:32 10-11% at 16 threads (Axion). Halve rq,
    # down to one register-tile pair.
    rq = min(rq, sq)
    var floor = 64 if isa == 1 else 16
    var cap = _workers(r, 1 << 30, 1)
    while rq // 2 >= floor and b * heads * ((sq + rq - 1) // rq) < 8 * cap:
        rq //= 2
    var n = _workers(r, b * heads * ((sq + rq - 1) // rq), 8)
    var sizes = flash_sizes(sk, rq, hd, dt, isa)
    var slab = sizes[0]
    var aux = sizes[1]
    var llc = cfg(r, LLC)
    # Fused (a thread packs a head and attends it, no barrier) while the workers'
    # slabs fit 0.6 of the LLC and the heads divide evenly.
    var fuse = b * heads >= n and b * heads % n == 0 and slab * n <= llc * 3 // 5
    var need = (n if fuse else b * heads) * slab + n * aux + 128
    if need > Int(py=ws_bytes):
        return PythonObject(need)
    var out = PythonObject(from_owned=empty(4, dims4(b, sq, heads, hd), dt == 1))
    var e = 2 if dt == 1 else 4
    var job = FlashJob(
        q=qt.data(e), k=kt.data(e), v=vt.data(e), out=Tensor.of(r, Int(out._obj_ptr)).data(e),
        batch=b, seq_q=sq, seq_k=sk, heads=heads,
        qb=qt.stride(0), qt=qt.stride(1), qh=qt.stride(2), kb=kt.stride(0), kt=kt.stride(1),
        kh=kt.stride(2), vb=vt.stride(0), vt=vt.stride(1), vh=vt.stride(2),
        ob=sq * heads * hd, ot=heads * hd, oh=hd, causal=is_causal, workers=n, fuse=fuse, rq=rq,
        ws=(Int(py=ws) + 63) & ~63, slab=slab, aux=aux,
        # fp32 V is read in place while it all fits half the LLC: a copy only
        # buys a unit stride.
        v_direct=dt == 0 and isa == 0 and b * heads * sk * hd * 4 <= llc // 2,
        scale=Float32(sc),
    )
    flash_run(job, dt, hd, isa)
    return out


def probe(rt: PythonObject, t: PythonObject) raises -> PythonObject:
    """What this module reads from `t` -- [plain, storage address, storage
    offset, dims, sizes..., strides..., dtype code] -- plus a new tensor of its
    shape and dtype, for the import-time check in `_kernel.runtime`."""
    var r = Int(py=rt)
    var x = Tensor.of(r, Int(t._obj_ptr))
    var res = Python.list()
    res.append(x.ok())
    if not x.ok():
        return res
    var n = x.dim()
    res.append(x.base())
    res.append(x.offset())
    res.append(n)
    for i in range(n):
        res.append(x.size(i))
    for i in range(n):
        res.append(x.stride(i))
    res.append(x.dtype())
    var dt = _dtype(r, x)
    if 1 <= n <= 4 and dt >= 0:
        var sizes = Array[Int, 4](fill=0)
        for i in range(n):
            sizes[i] = x.size(i)
        res.append(PythonObject(from_owned=empty(n, sizes, dt == 1)))
    return res


def tier() raises -> PythonObject:
    """The build tier of this shared object (`scripts/build.py`)."""
    return PythonObject(tier_name())


def amx_available() -> PythonObject:
    """The OS grants AMX tile data (`arch_prctl`); False on tiers without AMX."""
    return PythonObject(amx_permitted())
