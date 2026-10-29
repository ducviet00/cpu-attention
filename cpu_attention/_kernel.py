"""Loader and call plumbing for the attention kernels.

Two backends, one Python API:

* ``mojo`` (default) -- built from ``cpu_attention/_mojo`` into
  ``cpu_attention/_C.so`` by ``uv run --extra build python scripts/build.py``;
* ``cpp`` -- built from ``cpu_attention/csrc`` into ``cpu_attention/_C_cpp.so``
  by ``uv run --extra build python scripts/build_cpp.py``.

``CPU_ATTENTION_BACKEND=cpp`` selects the second one. Both shared objects can be
loaded in one process, which is what the A/B harnesses do. Set
``CPU_ATTENTION_LIB`` to point at another build of the selected backend.
Everything here is optional: without the shared library (or on an unsupported
configuration) :func:`is_available` is False and every entry point raises
``RuntimeError``.
"""

import importlib.util
import os
import platform
import struct
import threading

import torch

# Keep Mojo as the default until the full cross-ISA scorecard passes. Explicit
# `auto` retains the earlier architecture-specific choice for existing callers.
_CHOSEN = os.environ.get("CPU_ATTENTION_BACKEND", "mojo").lower()
if _CHOSEN == "auto":
    _CHOSEN = "cpp" if platform.machine() in ("x86_64", "AMD64") else "mojo"
if _CHOSEN not in ("mojo", "cpp"):
    raise ValueError(
        f"CPU_ATTENTION_BACKEND must be 'auto', 'mojo' or 'cpp', not {_CHOSEN!r}"
    )
_BACKEND = _CHOSEN

_DIR = os.path.dirname(os.path.abspath(__file__))
_INITIAL_BACKEND = _BACKEND
_LIB_PATH = os.path.join(_DIR, "_C_cpp.so" if _BACKEND == "cpp" else "_C.so")


def backend() -> str:
    """Return the selected backend name (``"mojo"`` or ``"cpp"``)."""
    return _BACKEND


def set_backend(name: str) -> None:
    """Select the backend for later calls (``"mojo"`` or ``"cpp"``).

    Both shared objects stay loaded: the A/B harnesses alternate them inside one
    process, and plan/scratch caches are keyed per backend so switching never
    hands one kernel the other's buffer.
    """
    global _BACKEND
    if name not in ("mojo", "cpp"):
        raise ValueError(f"backend must be 'mojo' or 'cpp', not {name!r}")
    _BACKEND = name

# The kernel specializes on head_dim at compile time; anything else raises.
_HEAD_DIMS = {32, 64, 128}
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1}

_BUILTS = {}


# Mojo build tiers (scripts/build.py), best first, with the CPU features each
# one was compiled for. `/proc/cpuinfo` lists a feature only when the kernel has
# enabled its register state, so a listed AVX-512 flag is also an OS check. AMX
# needs a further permission syscall, which the kernel module makes itself.
_MOJO_TIERS = {
    "x86_64": [
        ("x86-avx512bf16", {"avx512f", "avx512bw", "avx512vl", "avx512dq",
                            "avx512cd", "avx512_bf16", "avx512_vnni"}),
        ("x86-avx2", {"avx2", "fma", "bmi1", "bmi2", "f16c", "movbe"}),
        ("x86-64", set()),
    ],
    "aarch64": [
        ("aarch64-bf16", {"asimd", "bf16", "i8mm"}),
        ("aarch64-neon", set()),
    ],
}
# CPU_ATTENTION_ISA (shared with the C++ backend) -> Mojo tier.
_MOJO_ISA = {"amx": "x86-avx512bf16", "vec": "x86-avx512bf16",
             "avx2": "x86-avx2", "neon": "aarch64-bf16"}
_LOAD_ERROR = {}


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _mojo_tier():
    """The tier to load: the best one this CPU supports, or the forced one."""
    arch = "aarch64" if platform.machine() in ("aarch64", "arm64") else "x86_64"
    tiers = _MOJO_TIERS[arch]
    flags = _cpu_flags()
    ok = [name for name, need in tiers if need <= flags]
    forced = os.environ.get("CPU_ATTENTION_ISA")
    if forced:
        want = _MOJO_ISA.get(forced, tiers[-1][0] if forced == "scalar" else None)
        if want is None or want not in ok:
            raise RuntimeError(f"cpu_attention: CPU_ATTENTION_ISA={forced} is not "
                               "supported on this CPU")
        return want, [want]
    return ok[0], ok


def _lib_path(name: str) -> str:
    if name == _INITIAL_BACKEND and os.environ.get("CPU_ATTENTION_LIB"):
        return os.environ["CPU_ATTENTION_LIB"]
    if name == "cpp":
        return os.path.join(_DIR, "_C_cpp.so")
    _, candidates = _mojo_tier()
    for tier in candidates:
        path = os.path.join(_DIR, f"_C_{tier}.so")
        if os.path.exists(path):
            return path
    return os.path.join(_DIR, "_C.so")


def _load(name=None):
    name = name or _BACKEND
    path = _lib_path(name)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("cpu_attention._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module(name=None):
    """Return the compiled kernel module for ``name`` (default: the selected
    backend), or None if it is unavailable."""
    name = name or _BACKEND
    if name not in _BUILTS:
        try:
            _BUILTS[name] = _load(name) or False
        except Exception as e:  # noqa: BLE001
            _LOAD_ERROR[name] = e
            _BUILTS[name] = False
    return _BUILTS[name] or None


def is_available() -> bool:
    """Return True if the compiled kernel is present and loadable."""
    return module() is not None


_BOUND = {}


def _entry(attr: str):
    """Bind ``attr`` of the selected backend's module once per backend."""
    key = (_BACKEND, attr)
    fn = _BOUND.get(key)
    if fn is None:
        mod = require()
        fn = getattr(mod, attr)
        _BOUND[key] = fn
    return fn


def require():
    """Return the compiled kernel module, raising RuntimeError if unavailable."""
    mod = module()
    if mod is None and isinstance(_LOAD_ERROR.get(_BACKEND), RuntimeError):
        raise _LOAD_ERROR[_BACKEND]
    if mod is None:
        script = "build_cpp.py" if _BACKEND == "cpp" else "build.py"
        raise RuntimeError(
            f"cpu_attention: the {_BACKEND} kernel is not built; run "
            f"`uv run --extra build python scripts/{script}` (or set "
            "CPU_ATTENTION_LIB)"
        )
    return mod


def supported(dtype, block_size: int, head_dim: int) -> bool:
    """Return True if the kernel can run this dtype / page size / head dim."""
    return (
        dtype in _DTYPE_CODES
        and head_dim in _HEAD_DIMS
        and block_size % 16 == 0
        and is_available()
    )


def _attention():
    """Return the extension's single callable, bound once per backend."""
    return _entry("attention")


_PHYSICAL = None


def _physical_cores() -> int:
    """Physical cores visible to this process, from the topology sysfs files."""
    global _PHYSICAL
    if _PHYSICAL is None:
        cores = set()
        try:
            for cpu in range(os.cpu_count() or 1):
                base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
                with open(f"{base}/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/physical_package_id") as f:
                    cores.add((f.read().strip(), core))
        except OSError:
            cores = set()
        _PHYSICAL = len(cores) or (os.cpu_count() or 1)
    return _PHYSICAL


_OVERRIDE = os.environ.get("CPU_ATTENTION_WORKERS")


def set_num_workers(workers: int) -> None:
    """Pin the worker count for every call (0 restores the automatic rule).

    It is part of the plan key, so cached plans are not invalidated here; a
    caller that changes it gets a fresh plan on the next call.
    """
    global _OVERRIDE
    _OVERRIDE = str(workers) if workers else None
    _KV_GEN[0] += 1


def _workers(num_tasks: int, pages: int = 1) -> int:
    """Workers for one call: the physical-core cap, inline only for tiny work.

    The kernel dispatches on torch's own OpenMP pool (``GOMP_parallel``), so a
    fork-join costs ~2.4 us warm and ~2.7 us with a 1 ms gap -- there is no
    sleeping second pool to wake, which is why the pool now wins almost
    everywhere: measured cold, one worker (inline, 0.36 us) beats 8 only up to
    ~32 KV positions per call (24.5 vs 31.0 us at batch 1 / ctx 64), while at
    batch 1 / ctx 128 the pool is already 1.3x faster and at batch 8 / ctx 256
    it is 8x. The cap at the *physical* core count matters on SMT boxes (8
    workers on a 4-core Xeon: 1.5x slower at ctx 1024); torch already sizes its
    pool that way, so it only bites when a parent sets a larger
    ``OMP_NUM_THREADS``. ``CPU_ATTENTION_WORKERS`` selects within that cap.
    """
    if _OVERRIDE:
        return max(1, min(int(_OVERRIDE), num_tasks, _physical_cores(),
                          torch.get_num_threads()))
    workers = min(torch.get_num_threads(), _physical_cores(), num_tasks)
    if workers > 1 and num_tasks * pages <= 32:
        return 1
    return workers


def num_workers(num_tasks: int, pages: int = 1) -> int:
    """Return the worker count :func:`_workers` would pick (diagnostics)."""
    return _workers(num_tasks, pages)


def physical_cores() -> int:
    """Return the physical core count used to cap the worker pool (diagnostics)."""
    return _physical_cores()


_PAGED_PLANS = threading.local()
_get_num_threads = torch.get_num_threads
"""Per-thread geometry -> fast call, for the Mojo paged path.

(The C++ backend keeps the same per-thread cache inside the extension:
`paged_call` / `paged_register`.)

The engine calls `self_attention`/`cross_attention` 24 times a decode step with
the same geometry, and at batch 1 / ctx 256 the whole call is 7 us: reading a
`.shape` and three `.stride()` tuples per call is a third of that. The plan holds
the 14 constant arguments and patches only the data pointers, so a call is one
`torch.empty`, six `data_ptr()`s and the extension call.

Correctness of the key: shapes plus, for a strided tensor, the strides. A
contiguous tensor's strides follow from its shape, so the common case (a fresh
contiguous q and cache) builds no stride tuples at all.
"""


def _kv_build(query, key, value, k_cache, page_table, do_store, do_attend):
    """Build the Mojo fast call for one geometry + layout.

    The geometry rides in a 12-slot int64 buffer the kernel reads, and the token
    strides are constants of the plan (they are part of its key), so a call
    marshals only the per-call pointers and the scale. Workers follow
    `_workers`; `chunks` splits a GQA/MQA group across workers only when there
    are fewer (sequence, kv head) pairs than workers (each chunk is a register
    tile of the group's query heads over one scan of the cache).
    """
    nseq, nheads, hd = query.shape
    kvh, page = k_cache.shape[1:3]
    npages = page_table.shape[1] if page_table is not None else 0
    # Heads and dims must be contiguous; a token stride is fine (the engine's
    # q/k/v are chunks of one fused projection). Anything else is copied.
    q_ok = query.stride(-1) == 1 and query.stride(-2) == hd
    kv_ok = (not do_store or (key.stride(-1) == 1 and key.stride(-2) == hd and
                              value.stride() == key.stride()))
    q_stride = query.stride(0) if q_ok else nheads * hd
    kv_stride = (key.stride(0) if kv_ok else key.shape[1] * hd) if do_store else 0
    pairs = nseq * kvh
    group = nheads // kvh
    workers = _workers(nseq * nheads, max(npages, 1))
    chunks = 1
    if do_attend and pairs < workers:
        chunks = min(group, -(-workers // pairs))
    workers = max(1, min(workers, pairs * chunks))
    buf = torch.tensor([nseq, nheads, kvh, hd, page, npages, nheads * hd,
                        _DTYPE_CODES[query.dtype], workers, int(do_store),
                        int(do_attend), chunks], dtype=torch.int64)
    plan = buf.data_ptr()
    entry = _entry("kv_attention")

    if do_store and do_attend:
        def fn(out, query, key, value, k_cache, v_cache, slots, block_table,
               lens, scale, _buf=buf):
            if not q_ok:
                query = query.contiguous()
            if not kv_ok:
                key, value = key.contiguous(), value.contiguous()
            entry(plan, query.data_ptr(), key.data_ptr(), value.data_ptr(),
                  k_cache.data_ptr(), v_cache.data_ptr(), out.data_ptr(),
                  0 if slots is None else slots.data_ptr(),
                  block_table.data_ptr(), lens.data_ptr(),
                  q_stride, kv_stride, scale)
    elif do_attend:
        def fn(out, query, key, value, k_cache, v_cache, slots, block_table,
               lens, scale, _buf=buf):
            if not q_ok:
                query = query.contiguous()
            entry(plan, query.data_ptr(), 0, 0, k_cache.data_ptr(),
                  v_cache.data_ptr(), out.data_ptr(), 0,
                  block_table.data_ptr(), lens.data_ptr(), q_stride, 0, scale)
    else:
        def fn(out, query, key, value, k_cache, v_cache, slots, block_table,
               lens, scale, _buf=buf):
            if not kv_ok:
                key, value = key.contiguous(), value.contiguous()
            entry(plan, 0, key.data_ptr(), value.data_ptr(), k_cache.data_ptr(),
                  v_cache.data_ptr(), 0, slots.data_ptr(), 0, 0, 0, kv_stride,
                  1.0)
    # What `kv_fast` needs to run this plan without Python (no copies needed).
    fn.fast = (buf, q_stride, kv_stride) if q_ok and kv_ok else None
    return fn


def _kv_plans():
    plans = getattr(_PAGED_PLANS, "p", None)
    if plans is None:
        plans = _PAGED_PLANS.p = {}
    return plans



# ---------------------------------------------------------------------------
# kv_fast: the decode call with the tensors read in Mojo (pyview.mojo)
# ---------------------------------------------------------------------------
_HDR = 24  # header slots before the plan entries (KV_ENTRY0 / FL_ENTRY0 in _C.mojo)
_KV_W = 32  # int64 slots per plan entry (see _C.mojo kv_fast)
_KV_CAP = 64
_KV_TABLE = threading.local()
_KV_GEN = [0]  # bumped by set_num_workers / _drop_workspace: tables are rebuilt
_LAYOUT = None  # calibrated header slots, or False when the fast path is off
_DT_CODE = {}


def _calibrate_layout():
    """TensorImpl/StorageImpl field offsets, found by matching what torch reports
    (`tensor._cdata` is the TensorImpl address) and verified on probe tensors,
    first through ctypes and then through the module's own reader. Any
    disagreement disables the fast path (False); nothing is assumed."""
    import ctypes

    def words(a, n):
        return [ctypes.c_int64.from_address(a + 8 * i).value for i in range(n)]

    def u16(a):
        return ctypes.c_uint16.from_address(a).value

    try:
        p0 = torch.empty(3, 5, 7)
        w = words(p0._cdata, 48)
        st = p0.untyped_storage()
        o_st = [i for i, x in enumerate(w) if x == st._cdata]
        o_dp = [i for i, x in enumerate(words(st._cdata, 16)) if x == st.data_ptr()]
        o_sz = [i for i in range(45) if w[i:i + 3] == [3, 5, 7]]
        o_sd = [i for i in range(45) if w[i:i + 3] == [35, 7, 1]]
        wv = words(torch.empty(100)[13:].view(29, 3)._cdata, 48)
        o_off = [i for i, x in enumerate(wv) if x == 13 and w[i] == 0]
        w2 = words(torch.empty(2, 2)._cdata, 48)
        w1 = words(torch.empty(4)._cdata, 48)
        o_dim = [i for i in range(48) if (w[i], w2[i], w1[i]) == (3, 2, 1)]
        dts = (torch.float32, torch.bfloat16, torch.int32, torch.int64,
               torch.float16)
        pairs = [(torch.empty(3, 5, dtype=d), torch.empty(7, dtype=d)) for d in dts]
        o_dt = [b for b in range(0, 384, 2)
                if all(u16(a._cdata + b) == u16(c._cdata + b) for a, c in pairs)
                and len({u16(a._cdata + b) for a, _ in pairs}) == len(dts)]
        if not (o_st and o_dp and o_sz and o_sd and o_off and o_dim and o_dt):
            return False
        o_st, o_dp, o_sz, o_sd, o_off, o_dim, o_dt = (
            o_st[0], o_dp[0], o_sz[0], o_sd[0], o_off[0], o_dim[0], o_dt[0])
        codes = {d: u16(a._cdata + o_dt) for d, (a, _) in zip(dts, pairs)}
        base = torch.randn(64, 3 * 16 * 8)
        probes = [
            torch.randn(5), torch.randn(4, 6), torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5).bfloat16(), torch.randn(9, 7)[2:].t(),
            torch.arange(50, dtype=torch.int32)[7:], torch.arange(9)[1::2],
            base[:, 16 * 8:32 * 8].view(64, 16, 8),
            torch.nn.Parameter(torch.randn(6, 2)),
            torch.empty(3, 16, 32, 64, dtype=torch.bfloat16)[1:],
        ]

        def ref(t):
            return (t.untyped_storage().data_ptr(), t.dim(), list(t.shape),
                    list(t.stride()), codes.get(t.dtype), t.storage_offset())

        for t in probes:
            im = t._cdata
            got = (words(words(im + 8 * o_st, 1)[0] + 8 * o_dp, 1)[0],
                   words(im + 8 * o_dim, 1)[0], words(im + 8 * o_sz, t.dim()),
                   words(im + 8 * o_sd, t.dim()), u16(im + o_dt),
                   words(im + 8 * o_off, 1)[0])
            if got != ref(t):
                return False
        hdr = [0, 8 * o_st, 8 * o_dp, 8 * o_off, 8 * o_dim, 8 * o_sz, 8 * o_sd,
               o_dt, id(torch.Tensor), id(torch.nn.Parameter), id(None), 0,
               codes[torch.int32], codes[torch.int64], id(True), id(False),
               id(float), 0]
        probe = module().tensor_probe
        buf = torch.tensor(hdr, dtype=torch.int64)
        for t in probes:
            r = probe(buf.data_ptr(), t)
            n = r[1]
            mine = (r[0], n, r[2:2 + n], r[2 + n:2 + 2 * n], r[2 + 2 * n],
                    r[3 + 2 * n])
            if mine != ref(t) or not r[4 + 2 * n]:
                return False
        _DT_CODE.update(codes)
        return hdr
    except Exception:  # noqa: BLE001
        return False


def _kv_table():
    """This thread's kv_fast table (int64 tensor), or None if the fast path is
    off (calibration failed, or CPU_ATTENTION_KV_FAST=0)."""
    global _LAYOUT
    t = getattr(_KV_TABLE, "t", None)
    if t is not None and _KV_TABLE.gen == _KV_GEN[0]:
        return t
    if _LAYOUT is None:
        _LAYOUT = (os.environ.get("CPU_ATTENTION_KV_FAST", "1") != "0" and
                   _calibrate_layout())
    if not _LAYOUT:
        return None
    t = torch.zeros(_HDR + _KV_CAP * _KV_W, dtype=torch.int64)
    t[:len(_LAYOUT)] = torch.tensor(_LAYOUT, dtype=torch.int64)
    _KV_TABLE.t, _KV_TABLE.gen, _KV_TABLE.keep = t, _KV_GEN[0], []
    _KV_TABLE.addr = t.data_ptr()
    _KV_TABLE.fn = _entry("kv_fast")
    return t


def _kv_register(fn, query, key, value, k_cache, page_table, mode):
    """Record the plan `fn` just ran so the next call with this layout runs from
    Mojo. Only plans that needed no copies qualify."""
    if fn.fast is None or _kv_table() is None or query.dtype not in _DT_CODE:
        return
    t = _KV_TABLE.t
    n = int(t[11])
    if n == _KV_CAP:
        t[11] = n = 0
        _KV_TABLE.keep = []
    buf, q_stride, kv_stride = fn.fast
    hd = query.shape[-1]
    default_scale = struct.unpack("<i", struct.pack("<f", float(hd ** -0.5)))[0] & 0xFFFFFFFF
    do_store = mode != 0
    pt_shape = page_table.shape if page_table is not None else (0, 0)
    e = [_DT_CODE[query.dtype], int(mode), torch.get_num_threads(),
         *query.shape, *query.stride(), *k_cache.shape, *pt_shape,
         *(key.stride() if do_store else (0, 0, 0)),
         *(value.stride() if do_store else (0, 0, 0)),
         buf.data_ptr(), q_stride, kv_stride, query.element_size(),
         default_scale]
    off = _HDR + n * _KV_W
    t[off:off + len(e)] = torch.tensor(e, dtype=torch.int64)
    _KV_TABLE.keep.append(buf)
    t[11] = n + 1


def _kv_store(key, value, k_cache, v_cache, slots):
    """reshape_and_cache on the Mojo backend: one token per `key` row."""
    table = getattr(_KV_TABLE, "t", None)
    if table is not None and _KV_TABLE.gen == _KV_GEN[0] and _KV_TABLE.fn(
            _KV_TABLE.addr, None, key, value, k_cache, v_cache, slots,
            None, None, None, None):
        return
    plans = _kv_plans()
    k = (key.shape, key.stride(), value.stride(), k_cache.shape,
         _get_num_threads(), _OVERRIDE, key.dtype, "store")
    fn = plans.get(k)
    if fn is None:
        if len(plans) > 64:
            plans.clear()
        fn = plans[k] = _kv_build(key, key, value, k_cache, None, True, False)
    fn(None, key, key, value, k_cache, v_cache, slots, None, None, 1.0)
    _kv_register(fn, key, key, value, k_cache, None, 3)


def _paged(
    query, key, value, k_cache, v_cache, slots, page_table, lens, scale, do_store,
    out=None,
):
    """Run one kernel call (geometry already checked).

    Allocates the output unless `out` is given: `torch.empty` is 1.3 us, which
    is a fifth of a batch-1 call and the reason the scorecard's batch-1 rows
    read badly against a reference whose output is preallocated.
    """
    if _BACKEND == "cpp":
        # The C++ entry reads the tensors and allocates `out` itself; it knows
        # every geometry this thread registered (see `paged_call`).
        call = _entry("paged_call")
        res = call(query, key, value, k_cache, v_cache, slots, page_table, lens,
                   scale, out)
        if res is NotImplemented:
            nseq, nheads = query.shape[:2]
            npages = page_table.shape[1]
            _entry("paged_register")(query, k_cache, page_table, do_store,
                                     _workers(nseq * nheads, npages))
            res = call(query, key, value, k_cache, v_cache, slots, page_table,
                       lens, scale, out)
        return res
    table = getattr(_KV_TABLE, "t", None)
    if table is not None and _KV_TABLE.gen == _KV_GEN[0]:
        res = _KV_TABLE.fn(
            _KV_TABLE.addr, query, key, value, k_cache, v_cache, slots,
            page_table, lens, scale, out)
        if res is not None:
            return res
    if out is None:
        # Paged queries have contiguous heads/dims; a fused-projection token
        # gap is compacted by empty_like without a memory-format keyword.
        out = torch.empty_like(query)
    # One tuple, one lookup: the key carries the strides, so the layout checks
    # run only when a plan is built (0.75 us of Python per call before).
    plans = _kv_plans()
    mode = (1 if slots is not None else 2) if do_store else 0
    if do_store:
        k = (query.shape, query.stride(), key.stride(), value.stride(),
             k_cache.shape, page_table.shape, _get_num_threads(), _OVERRIDE,
             query.dtype, mode)
    else:
        k = (query.shape, query.stride(), k_cache.shape, page_table.shape,
             _get_num_threads(), _OVERRIDE, query.dtype)
    fn = plans.get(k)
    if fn is None:
        if len(plans) > 64:
            plans.clear()
        fn = plans[k] = _kv_build(query, key, value, k_cache, page_table,
                                  do_store, True)
    fn(out, query, key, value, k_cache, v_cache, slots, page_table, lens, scale)
    _kv_register(fn, query, key, value, k_cache, page_table, mode)
    return out


_LLC = None


def _llc_bytes() -> int:
    """Last-level cache size in bytes, from sysfs (32 MiB if unknown).

    Only used to decide whether a dense V is worth copying into a per-head
    contiguous slab: the copy pays when V no longer fits, and the P@V sweep is
    the only reader.
    """
    global _LLC
    if _LLC is None:
        size = 0
        try:
            for idx in (3, 2, 1, 0):
                with open(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size") as f:
                    text = f.read().strip()
                mult = 1
                if text.endswith("K"):
                    mult, text = 1024, text[:-1]
                elif text.endswith("M"):
                    mult, text = 1024 * 1024, text[:-1]
                size = int(text) * mult
                break
        except (OSError, ValueError):
            size = 0
        _LLC = size or (32 << 20)
    return _LLC


_SCRATCH = threading.local()
"""Reusable scratch, per thread, keyed by what it is for.

`torch.empty` on a multi-megabyte buffer is not free: glibc serves it with
`mmap`, the matching `free` unmaps it, so the next call gets fresh pages and the
kernel faults them in one 4 KiB page at a time. `clear_page_erms` (the kernel
zeroing those pages) measured **16.8% of a whole call** at S=1024/H=16/D=128
fp32 against 49% for the row kernel itself, and the pack looked 6x its real cost
because the faults landed inside it.

Holding one buffer per purpose and shape removes the fault/zero pass for
everything except the first call. It is a workspace, not a cache of results: the
kernel writes every byte it reads, and the buffers are never returned to the
caller (the output tensor is still a fresh allocation, since the caller owns
it). Per-thread, because two threads packing into one buffer would corrupt each
other's K.
"""


_SCRATCH_BYTES = 24 << 20
"""Bytes of packed-K/V workspace kept alive for *other* geometries.

Bounded by bytes rather than by the number of geometries, so a workload with
many small shapes keeps all of them and one with a few large shapes keeps one
generation. Charged at the start of a plan build (see :func:`_dense_plan`), when
nothing is half-built, and only against workspace that belongs to *other*
geometries -- a single geometry is never evicted to make room for itself, which
is what would turn this into a per-call reallocation.
"""


def _workspace_bytes(bufs) -> int:
    return sum(t.numel() * t.element_size() for t in bufs.values())


def _drop_workspace() -> None:
    """Release every packed-K/V buffer and the call plans that point at them.

    One generation is all an engine needs, but a sweep over many *large* shapes
    would otherwise keep every geometry's K^T/V scratch alive at once -- a 22-row
    ``benchmarks/bench.py`` run reaches several hundred MB, and the rows it
    measures late are 5-15% slower than the same rows run on their own (measured
    on the Graviton box, medians of 9; the fingerprint is positional, not
    shape-dependent). The plans hold references to the buffers, so the two have
    to be dropped together.
    """
    bufs = getattr(_SCRATCH, "bufs", None)
    if bufs is not None:
        bufs.clear()
    plans = getattr(_SCRATCH, "plans", None)
    if plans is not None:
        plans.clear()
    # The flash plans (AMX included) hold the workspace's raw addresses: dropping
    # only some of them made an on/off/on A/B reuse freed memory.
    _FLASH.__dict__.pop("ws", None)
    _FLASH.__dict__.pop("p", None)
    _KV_GEN[0] += 1


def _scratch(kind, shape, dtype):
    """A workspace tensor of ``shape``/``dtype``, reused across calls."""
    bufs = getattr(_SCRATCH, "bufs", None)
    if bufs is None:
        bufs = _SCRATCH.bufs = {}
    key = (kind, shape, dtype)
    buf = bufs.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype)
        bufs[key] = buf
    return buf


def scratch_bytes() -> int:
    """Bytes this thread holds as workspace (diagnostics / leak checks)."""
    bufs = getattr(_SCRATCH, "bufs", None) or {}
    seen = {id(t) for t in bufs.values()}
    plans = getattr(_SCRATCH, "plans", None) or {}
    for plan in plans.values():
        # two of the plan's fields are the buffers themselves
        for t in plan[-3:]:
            if t is not None:
                seen.add(id(t))
    total = 0
    for t in list(bufs.values()) + [
        x for p in plans.values() for x in p[-3:] if x is not None
    ]:
        if id(t) in seen:
            seen.discard(id(t))
            total += t.numel() * t.element_size()
    return total


def clear_scratch() -> None:
    """Drop this thread's workspace buffers (tests, or after a shape sweep)."""
    _drop_workspace()


_AMX_ENV = (os.environ.get("CPU_ATTENTION_AMX", "1") != "0" and
            os.environ.get("CPU_ATTENTION_ISA", "amx") == "amx")
# fp32 goes through the same AMX policy with bf16 hi/lo operand splitting (three
# products per phase); this switch is for A/B and for falling back if a future
# CPU disagrees with the measured 3-8e-5 error (gate 2e-4).
_AMX_FP32_ENV = os.environ.get("CPU_ATTENTION_AMX_FP32", "1") != "0"

_AMX_OK = None


def amx_available() -> bool:
    """True when this build and process can run the AMX flash policy.

    The CPU must report AMX and the OS must grant tile-data permission
    (`arch_prctl`, a syscall: the answer is cached, and the first tile
    instruction is SIGILL without it). Torch's oneDNN normally asks too, but that
    is an implementation detail of the wheel, not a guarantee. The ``_AMX_ENV``
    toggle is separate and is read per call, so on/off/on works in one process.
    """
    global _AMX_OK
    if _AMX_OK is None:
        mod = module()
        gate = getattr(mod, "amx_available", None) if mod else None
        cpu = {"amx_tile", "amx_bf16"} <= _cpu_flags()
        _AMX_OK = bool(cpu and gate is not None and gate())
    return _AMX_OK and _AMX_ENV


_CPP_PLANS = threading.local()
"""Per-thread dense plans for the C++ backend: geometry -> (buffer, address).

The C++ side keeps the workspace (packed K/V, accumulators) keyed on the same
geometry, so the Python side only has to hand it the plan; every derived value
(rows per work item, ISA, worker count) is computed once, on the miss path.
"""

def _cpp_entries():
    """Bind the C++ plan/forward pair once per backend."""
    return _entry("dense_make_plan"), _entry("dense_forward")


def _cpp_plan(q, k, v, scale, causal, geom=None):
    """The plan address for this geometry, building it on the first call."""
    plans = getattr(_CPP_PLANS, "p", None)
    if plans is None:
        plans = _CPP_PLANS.p = {}
    if geom is None:
        geom = (q.shape, q.stride(), k.stride(), v.stride(), q.dtype)
    qshape, qs, ks, vs, dtype = geom
    batch, seq_q, heads, hd = qshape
    kshape = k.shape
    kv_heads = kshape[2]
    workers = _workers(batch * heads * -(-seq_q // 64), 8)
    # The public wrapper already read every stride. A contiguous Q says nothing
    # about K or V: mixed contiguous/strided inputs need their own plan too.
    key = (qshape, kshape, qs, ks, vs, dtype, causal, workers, scale)
    cached = plans.get(key)
    if cached is None:
        make_plan, _ = _cpp_entries()
        buf = torch.empty(24, dtype=torch.int64)
        make_plan(
            buf.data_ptr(), batch, seq_q, kshape[1], heads, kv_heads, hd,
            _DTYPE_CODES[dtype], int(bool(causal)), float(scale),
            qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], vs[0], vs[1], vs[2],
            seq_q * heads * hd, heads * hd, hd, workers,
        )
        if len(plans) > 64:
            plans.clear()
        addr = buf.data_ptr()
        # The buffer has to outlive the address, so hold it next to the key.
        plans[key] = cached = (buf, addr)
    return cached[1]


def _dense_cpp(q, k, v, scale: float, causal: bool, geom=None):
    """Dense attention through the C++ backend: one call, five arguments."""
    forward = _entry("dense_forward")
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    forward(_cpp_plan(q, k, v, scale, causal, geom), q.data_ptr(), k.data_ptr(),
            v.data_ptr(), out.data_ptr())
    return out


_FLASH = threading.local()
"""Per-thread flash_attn_func plans and workspace (the Mojo backend)."""
_AMX_PLANS = _FLASH
"""AMX is a policy id in the flash plan, so its plans live in the same place."""


def _flash_ws(nbytes):
    """This thread's workspace, grown on demand and never shrunk: a large
    `torch.empty` is an mmap, and freeing it means faulting the pages in again
    on the next call (docs/CPP.md §3). `_drop_workspace` releases it."""
    ws = getattr(_FLASH, "ws", None)
    if ws is None or ws.numel() < nbytes:
        _FLASH.ws = ws = torch.empty(nbytes + 4096, dtype=torch.uint8)
        _KV_GEN[0] += 1  # fast-path plans hold the old workspace address
    return (ws.data_ptr() + 63) & ~63


def _vector_beats_amx(dt, hd, seq_q, seq_k, scale):
    """Few query rows leave AMX's 16-row tiles mostly empty (and fp32 pays three
    products per phase), so the vector policy is faster there. Crossovers
    measured on the Xeon 8581C, 4 threads (vector / AMX time): fp32 hd=64 up to
    65 rows 0.39-0.94, fp32 hd=128 up to 17 rows 0.80-0.97 (or a 33-key sequence: 0.59), bf16 1-2 rows 0.76-0.89
    (hd=128 only while seq_k <= 256: AMX packs a long K/V faster)."""
    if dt == 0:
        # fp32 on AMX is bf16 hi/lo products (the dropped lo*lo term): accurate
        # to 3-8e-5 at the usual scale, but a logit std above ~6 (scale*sqrt(hd)
        # for unit-variance inputs; 1 at the default scale) makes the
        # probabilities peaked enough to reach 2.03e-4 against the 2e-4 gate
        # (scale 1.7 at hd=64). Such a call takes the exact-fp32 vector policy.
        if scale * hd ** 0.5 > 6.0:
            return True
        return seq_q <= (65 if hd <= 64 else 24) or seq_k <= 64
    return seq_q <= 2 and (hd <= 64 or seq_k <= 256)


def _flash_plan(geom, k, causal, scale):
    """Plan buffer for one geometry: the int64 slots `FlashJob` reads."""
    plans = getattr(_FLASH, "p", None)
    if plans is None:
        plans = _FLASH.p = {}
    qshape, qs, ks, vs, dtype = geom
    batch, seq_q, heads, hd = qshape
    seq_k, kv_heads = k.shape[1], k.shape[2]
    tasks_q = -(-seq_q // 64)
    dt = _DTYPE_CODES[dtype]
    # Policy ids: 0 vector, 1 AMX (fp32 hi/lo unless switched off), 2 ARM
    # bfmmla/lane-FMA. The toggles can change between calls, so key on the id.
    isa = (2 if module().tier() == "aarch64-bf16" else
           1 if amx_available() and (dt == 1 or _AMX_FP32_ENV) else 0)
    if isa == 1 and _vector_beats_amx(dt, hd, seq_q, seq_k, scale):
        isa = 0
    key = (geom, k.shape, causal, _OVERRIDE, torch.get_num_threads(),
           _llc_bytes(), isa)
    plan = plans.get(key)
    if plan is None:
        if isa == 1:
            # Two 32-row AMX register tiles per work item: K/V tiles are loaded
            # once per 64 rows.
            rq = 64
        elif (isa == 2 and dt == 0 and hd == 128 and causal and
              seq_q == seq_k and seq_q >= 2048):
            # Axion: amortize the K/V scan on long fp32 D=128 calls (6-9%);
            # applying this globally regressed short rows by up to 9.5%.
            rq = 128
        else:
            # Rows per work item: the accumulator (rq x hd fp32) stays near 16 KB,
            # so it is still in L1 when the next key tile arrives.
            rq = max(16, min(128, 16384 // (hd * 4)))
        # At least ~8 work items per worker: causal blocks differ in cost, and at
        # 4 per worker the tail cost 1024:8:32 10-11% at 16 threads (Axion, both
        # dtypes). Halve rq, down to one register-tile pair.
        floor, cap = (64 if isa == 1 else 16), _workers(1 << 30)
        while rq > floor and batch * heads * -(-seq_q // rq) < 8 * cap:
            rq //= 2
        rq = min(rq, max(seq_q, 1))
        nqb = -(-seq_q // rq)
        workers = _workers(batch * heads * nqb, 8)
        slab, aux = _entry("flash_sizes")(seq_k, rq, hd, dt, isa)
        nbh = batch * kv_heads
        llc = _llc_bytes()
        # Fused (pack own head, attend, no barrier) while every worker's slab
        # fits 0.6 of the LLC and the heads divide evenly; MHA only.
        fuse = (heads == kv_heads and nbh >= workers and nbh % workers == 0 and
                slab * workers <= llc * 3 // 5)
        # fp32 V is read in place while all of it fits half the LLC: the copy
        # only buys a unit stride (C++ measurement, docs/CPP.md §5).
        v_direct = (dt == 0 and not isa and
                    batch * kv_heads * seq_k * hd * 4 <= llc // 2)
        cells = workers if fuse else nbh
        total = cells * slab + workers * aux + 64
        buf = torch.tensor([
            batch, seq_q, seq_k, heads, kv_heads, hd,
            qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], vs[0], vs[1], vs[2],
            seq_q * heads * hd, heads * hd, hd,
            dt, int(bool(causal)), workers, int(fuse), rq, 0, slab, aux,
            int(v_direct), isa], dtype=torch.int64)
        if len(plans) > 64:
            plans.clear()
        plans[key] = plan = (buf, buf.data_ptr(), total)
    return plan


def _flash_mojo(q, k, v, scale, causal, geom=None):
    """flash_attn_func on the Mojo backend: one call, seven arguments."""
    if geom is None:
        geom = (q.shape, q.stride(), k.stride(), v.stride(), q.dtype)
    buf, plan, total = _flash_plan(geom, k, causal, scale)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    ws = _flash_ws(total)
    _entry("flash_forward")(plan, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                            out.data_ptr(), ws, scale)
    _FLASH.last = (buf, ws, scale)
    return out


_FL_W = 32


def _flash_gen():
    """Validity of this thread's flash_fast plans: the global generation plus
    the AMX toggles, which choose the policy a plan runs and may be flipped
    between calls (an A/B, or `CPU_ATTENTION_AMX` on/off/on)."""
    return (_KV_GEN[0], _AMX_ENV, _AMX_FP32_ENV)


def _flash_table():
    """This thread's flash_fast table, or None (see `_kv_table`)."""
    global _LAYOUT
    t = getattr(_FLASH, "t", None)
    if t is not None and _FLASH.gen == _flash_gen():
        return t
    if _LAYOUT is None:
        _LAYOUT = (os.environ.get("CPU_ATTENTION_KV_FAST", "1") != "0" and
                   _calibrate_layout())
    if not _LAYOUT:
        return None
    t = torch.zeros(_HDR + _KV_CAP * _FL_W, dtype=torch.int64)
    t[:len(_LAYOUT)] = torch.tensor(_LAYOUT, dtype=torch.int64)
    _FLASH.t, _FLASH.gen, _FLASH.keep = t, _flash_gen(), []
    _FLASH.addr = t.data_ptr()
    _FLASH.fn = _entry("flash_fast")
    return t


def _flash_register(q, k, v, softmax_scale, causal):
    """After a validated Mojo flash call: record its plan for `flash_fast`."""
    last = _FLASH.__dict__.pop("last", None)
    if last is None or _flash_table() is None or q.dtype not in _DT_CODE:
        return
    if softmax_scale is not None and type(softmax_scale) is not float:
        return
    if type(causal) is not bool:
        return
    buf, ws, scale = last
    t = _FLASH.t
    n = int(t[11])
    if n == _KV_CAP:
        t[11] = n = 0
        _FLASH.keep = []
    import struct as _struct
    e = [_DT_CODE[q.dtype], int(causal), torch.get_num_threads(),
         int(softmax_scale is None),
         _struct.unpack("<q", _struct.pack("<d", float(softmax_scale or 0.0)))[0],
         *q.shape, *q.stride(), *k.shape, *k.stride(), *v.stride(),
         buf.data_ptr(), ws, q.element_size(),
         _struct.unpack("<i", _struct.pack("<f", float(scale)))[0] & 0xFFFFFFFF]
    off = _HDR + n * _FL_W
    t[off:off + len(e)] = torch.tensor(e, dtype=torch.int64)
    _FLASH.keep.append(buf)
    t[11] = n + 1


def _dense(q, k, v, scale: float, causal: bool, geom=None):
    """flash_attn_func over contiguous-``head_dim`` q/k/v (already checked).

    ``geom`` is ``(q.shape, q.stride(), k.stride(), v.stride(), q.dtype)`` when
    the caller has already read them (``flash_attn_func`` has to, for its
    argument checks); otherwise they are read here.
    """
    if _BACKEND == "cpp":
        return _dense_cpp(q, k, v, scale, causal, geom)
    return _flash_mojo(q, k, v, scale, causal, geom)
