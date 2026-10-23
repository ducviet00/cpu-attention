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
import struct
import threading

import torch

_BACKEND = os.environ.get("CPU_ATTENTION_BACKEND", "mojo").lower()
if _BACKEND not in ("mojo", "cpp"):
    raise ValueError(
        f"CPU_ATTENTION_BACKEND must be 'mojo' or 'cpp', not {_BACKEND!r}"
    )

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


def _lib_path(name: str) -> str:
    if name == _INITIAL_BACKEND and os.environ.get("CPU_ATTENTION_LIB"):
        return os.environ["CPU_ATTENTION_LIB"]
    return os.path.join(_DIR, "_C_cpp.so" if name == "cpp" else "_C.so")


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
        except Exception:  # noqa: BLE001
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
    ``OMP_NUM_THREADS``. ``CPU_ATTENTION_WORKERS`` overrides everything.
    """
    if _OVERRIDE:
        return max(1, min(int(_OVERRIDE), num_tasks))
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


def _call(
    query,
    key,
    value,
    k_cache,
    v_cache,
    out,
    slots,
    block_table,
    lens,
    scale,
    do_store: bool,
):
    """Run one kernel call; ``key``/``value``/``slots`` are ignored without a store."""
    attention = _attention()
    zero = 0
    num_seqs, num_heads, head_dim = query.shape
    q_stride = query.stride()
    if q_stride[-1] != 1 or q_stride[-2] != head_dim:
        query = query.contiguous()
        q_stride = query.stride()
    # A fused q/k/v projection hands us chunks of one buffer: the head dim is
    # contiguous but the token stride is 3 * d_model, which the kernel takes
    # explicitly (a head-major layout check would silently copy).
    kv_stride = 0
    if do_store:
        k_stride = key.stride()
        if k_stride[-1] != 1 or k_stride[-2] != head_dim or value.stride() != k_stride:
            key, value = key.contiguous(), value.contiguous()
            k_stride = key.stride()
        kv_stride = k_stride[0]
    num_tasks = num_seqs * num_heads
    attention(
        query.data_ptr(),
        key.data_ptr() if do_store else zero,
        value.data_ptr() if do_store else zero,
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        out.data_ptr(),
        slots.data_ptr() if do_store else zero,
        block_table.data_ptr(),
        lens.data_ptr(),
        num_seqs,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        block_table.shape[1],
        q_stride[0],
        kv_stride,
        out.stride(0),
        _DTYPE_CODES[query.dtype],
        scale,
        _workers(num_tasks, block_table.shape[1]),
        do_store,
        True,
    )
    return out


def _paged(
    query, key, value, k_cache, v_cache, slots, page_table, lens, scale, do_store
):
    """Run one kernel call, allocating the output (geometry already checked)."""
    out = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        dtype=query.dtype,
        device=query.device,
    )
    return _call(
        query,
        key,
        value,
        k_cache,
        v_cache,
        out,
        slots,
        page_table,
        lens,
        scale,
        do_store,
    )


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


def _transpose_attention():
    """Return the extension's K-transpose entry point, bound once."""
    return _entry("dense_transpose")


_FORWARD = None


def _forward_plan():
    """The fused entry that takes a plan buffer instead of 42 arguments."""
    try:
        return _entry("dense_forward_plan")
    except AttributeError:
        raise RuntimeError(
            "cpu_attention: the mojo kernel has no dense_forward_plan; rebuild "
            "it with scripts/build.py"
        ) from None


def _forward():
    """Return the extension's fused pack+attention entry point, bound once."""
    return _entry("dense_forward")


_FUSE = not os.environ.get("CPU_ATTENTION_NO_FUSE")


def _dense_attention():
    """Return the extension's dense entry point, bound once."""
    return _entry("dense_attention")


_LANES = {}


def _lanes() -> int:
    """fp32 lanes per vector in the built kernel (16 on AVX-512, 4 on NEON)."""
    if _BACKEND not in _LANES:
        _LANES[_BACKEND] = int(_entry("dense_lanes")())
    return _LANES[_BACKEND]


_BF16_DOT = {}


def _bf16_dot() -> bool:
    """True when the built kernel has a bf16 dot-product instruction.

    The transposed-K layout depends on it (paired head dims for the dot, one
    head dim per position otherwise), so the caller has to know which one the
    compiled kernel expects.
    """
    if _BACKEND not in _BF16_DOT:
        _BF16_DOT[_BACKEND] = bool(_entry("dense_bf16_dot")())
    return _BF16_DOT[_BACKEND]


_INT8_QK = None


def _int8_qk() -> bool:
    """True when the build runs bf16 Q@K^T on i8mm with int8 operands.

    That path reads a different K layout *and* dtype (int8 blocks plus a
    per-64-key dequant scale), so the caller has to allocate both scratch
    buffers and select pack mode 3.
    """
    global _INT8_QK
    if _INT8_QK is None:
        mod = module()
        gate = getattr(mod, "dense_int8_qk", None) if mod else None
        _INT8_QK = bool(gate()) if gate is not None else False
    return _INT8_QK


_BFMMLA = {}


def _bfmmla() -> bool:
    """True when the bf16 path runs on NEON ``bfmmla``.

    That kernel reads a different K *and* V layout (position-paired K over four
    head dims, position-major V over head-dim pairs), so the caller allocates
    both scratch buffers and packs V as well as K.
    """
    if _BACKEND not in _BFMMLA:
        _BFMMLA[_BACKEND] = bool(_entry("dense_bfmmla")())
    return _BFMMLA[_BACKEND]



_QT = 4
"""Query rows per work item (``CPU_ATTENTION_QT`` overrides it).
Bounded by the register file: the output accumulator is ``qt * head_dim / LANES``
zmm and the tile logits another ``qt * 64 / LANES``, so ~32 registers cap
``qt * head_dim`` near 512. Bigger blocks amortize the K^T/V loads over more rows
and shrink the per-tile softmax, but spill the accumulator; 4 measured best for
every supported head dim and dtype on the AVX-512 boxes (2 and 8 are switchable
through the environment).
"""

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


_PLAN_ABI_FIELDS = [18, 19, 2, 3, 4, 5, 23, 24, 25, 8, 9, 11, 26, 27, 28, 16, 17, 21, 22, 18, 21, 22, 2, 0, 4, 1, 3, 5, 29, 30, 31, 8, 9, 10, 11, 32, 33, 34, 35, 36, 37, 6, 7, 12, 13]
"""Plan-tuple indices carried in the call buffer, in the order ``_C`` reads them.

Mirrors the field order in ``_C.dense_forward_plan``; both were generated from
one table. ``job.causal`` follows them as the last entry.
"""

_PLAN_ABI = os.environ.get("CPU_ATTENTION_PLAN_ABI", "1") != "0"
"""Use the seven-argument entry. ``0`` forces the 42-argument one, for the A/B in
``tests/test_dense.py``, which requires bit-identical output from both."""

_SCRATCH_STRIDES = {}


def _scratch_strides(kt, vt):
    """Cached ``.stride()`` tuples for the two pack buffers.

    ``Tensor.stride(i)`` is a Python call and the pack buffers only change when
    the shape does, so their stride tuples are cached by identity.
    """
    key = (id(kt), id(vt))
    got = _SCRATCH_STRIDES.get(key)
    # Validate on shape, not identity alone: a buffer freed by clear_scratch()
    # or by the 64-entry bound could hand its id to a tensor of another shape,
    # and a stale stride tuple would corrupt every later call silently.
    if got is None or got[0] != kt.shape:
        if len(_SCRATCH_STRIDES) > 64:
            _SCRATCH_STRIDES.clear()
        got = (kt.shape, kt.stride(), vt.stride() if vt is not None else None)
        _SCRATCH_STRIDES[key] = got
    return got[1], got[2]


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
    ``tests/bench_vs_refs.py`` run reaches several hundred MB, and the rows it
    measures late are 5-15% slower than the same rows run on their own (measured
    on the Graviton box, medians of 9; the fingerprint is positional, not
    shape-dependent). The plans hold references to the buffers, so the two have
    to be dropped together.
    """
    bufs = getattr(_SCRATCH, "bufs", None)
    if bufs is not None:
        bufs.clear()
    _SCRATCH_STRIDES.clear()
    plans = getattr(_SCRATCH, "plans", None)
    if plans is not None:
        plans.clear()


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
    # The AMX call plans hold the workspace addresses; a stale plan would feed
    # the kernel a freed buffer.
    _AMX_PLANS.__dict__.pop("p", None)


_QT_ENV = os.environ.get("CPU_ATTENTION_QT")
"""``CPU_ATTENTION_QT``: pin the query rows per work item (``None`` = the table).

Read per call by :func:`_dense` (it is part of the plan key), so a sweep can
monkeypatch this global between calls in one process -- which is how the `qt`
table in :func:`_qt_for` was measured.
"""


def set_qt(rows) -> None:
    """Pin ``qt`` for every later call (``0``/``None`` restores the table).

    The plan cache keys on it, so no invalidation is needed here.
    """
    global _QT_ENV
    _QT_ENV = str(int(rows)) if rows else None

_DEFAULT_SMALL_MAXK = 1024
"""The shipped threshold, so scripts that pin it back can restore the default."""

_SMALL_MAXK = int(
    os.environ.get("CPU_ATTENTION_SMALL_MAXK", str(_DEFAULT_SMALL_MAXK))
)
"""``seq_k`` up to which the short-sequence kernel (``dense_small.mojo``) runs.

That path is the shipped bf16 tile with the key coverage rounded to 4 keys
instead of 16, no first-tile accumulator round trip, and the pack fused in the
same per-head item. It used to lose above ``seq_k = 256`` (0.94x at 512:8:64);
once it moved to the same bf16-precision exponential as the shipped tile kernel
(the degree-2 polynomial + the 4-lane panel shuffle, 2026-10-23) it wins or ties
everywhere it was measured, so the threshold went to 1024:

    shape (bf16)     small/ship, in-process, rotated, min of 21 (control)
    512:8:64         0.969 (0.998)
    512:32:128       0.983 (1.003)
    1024:8:64        0.986 (0.999)
    1024:16:128      0.986 (1.000)
    2048:8:64        0.990 (0.997)
    2048:16:128      0.992 (0.990)   -- no gain, so 2048 stays on the ship path
    4096:16:128      0.998 (0.999)   -- likewise

`512:8:64` was the shape that used to keep this threshold at 256. Costs no
precision the gate can see: those rows move from 7.8e-3 to 1.6e-2 against the
fixed 3e-2 bf16 tolerance, the same figure the D=128 rows already report.
"""


def set_small_maxk(seq_k) -> None:
    """Pin the short-sequence threshold (sweeps/tests); 0 disables the path."""
    global _SMALL_MAXK
    _SMALL_MAXK = int(seq_k)


_HEAD_RATIO = int(os.environ.get("CPU_ATTENTION_HEAD_RATIO", "0"))
"""Max heads per thread for the per-head dispatch (fp32; bf16 ignores it).

**Off by default, and the cap must be re-justified before it is turned on.** I
measured fp32 512:32:128 as 4802 us (median of 15) capped against 4659 uncapped
and concluded 4 heads per thread was the problem; the integrator re-measured it
three separate processes with 11 rounds and a min-based ratio and got the
**opposite**: 0.84-0.85 uncapped against 0.94-0.98 capped, with `ab_two_trees` on
the same pair reading 4027/4023/4024 us uncapped against 4510/4516/4529 capped.
The 4065..5350 spread I saw was foreign load on the shared box, i.e. exactly the
median-vs-min artefact the harness now guards against. So the default is 0 (no
cap) and the knob stays for a box that measures otherwise.
"""
_HEAD_MAXK = os.environ.get("CPU_ATTENTION_HEAD_MAXK", "1073741824")
"""``CPU_ATTENTION_HEAD_MAXK``: max ``seq_k`` for the per-head dispatch.

One thread owns a whole (batch, head): it packs that head's K/V and then runs
every one of its query blocks, so the two phases share bytes and the region
needs no internal barrier. It only balances when ``batch*heads % workers == 0``
(checked in :func:`_dense_plan`), and the header cache argument only holds while
one head's K/V stays near-resident -- but it measured faster at *every* shape in
the matrix (1.00-1.18x, in-process interleaved, `tests/bench_head.py`), so the
default is on and the limit is only there for A/B and for a box where one head's
K/V no longer fits. ``0`` = off.
"""


def set_head_maxk(seq_k) -> None:
    """Enable (or disable) the per-head dispatch above ``seq_k`` (sweeps/tests)."""
    global _HEAD_MAXK
    _HEAD_MAXK = str(int(seq_k)) if seq_k else None


def set_head_ratio(heads) -> None:
    """Cap heads per thread for the per-head dispatch (0 = no cap)."""
    global _HEAD_RATIO
    _HEAD_RATIO = int(heads)


def _qt_for(
    lanes: int, is_bf16: bool, head_dim: int, seq_k: int, bfmmla: bool
) -> int:
    """Query rows per work item, per (ISA, dtype, head_dim, seq_k).

    ``qt`` is the *only* dimension across which K/V is reused, so traffic per
    row-key scales as ``1/qt``. That is what binds on the server boxes (see
    ``PROFILE_ARM.md`` §10 and ``PROFILE_X86.md``): on x86 fp32 the load-bound
    fraction is 48% vs vLLM's 15% and ``CPU_ATTENTION_QT=8`` is worth 1.20-1.34x.

    ARM (``lanes < 16``), fp32/bfdot, head_dim 64, measured on m9g
    (Neoverse-V3, 3.28 GHz, single core, interleaved, medians of 5, parity
    asserted per variant), as a fraction of qt=4::

        S        1024    2048    4096
        qt=2     0.80x   0.79x   0.77x
        qt=4     1.00x   1.00x   1.00x
        qt=6     1.09x   1.10x   1.11x
        qt=8     0.78x   0.76x   0.76x

    So 4 -> 6 is a free win once the key range is long enough for the extra
    traffic reduction to pay for the shorter per-row block, and 8 loses to the
    register wall (``head_dim / LANES`` = 16 accumulator registers per row at
    head_dim 64).

    head_dim 128 takes the same 6, which is *not* the "32 registers per row
    already" argument one would make from the register file: this kernel keeps
    the output accumulator in L1 (``acc``/``accv`` are stack allocations), so
    rows cost ``kt + dc`` registers per row, not ``head_dim``, and the row count
    is what divides the 16-byte K/V loads. Measured interleaved on m9g against
    the shipped ``qt=2``, ``qt=6`` is 1.42x at S=1024, 1.52x at S=2048 and 1.73x
    at S=4096 (fp32, H=16, D=128); ``qt=4`` and ``qt=8`` are 1.18x and 1.55x.

    bf16 on ARM is a different path (``bfmmla``) that keeps its own tile
    geometry, and head_dim picks the rows there; x86 (``lanes == 16``) keeps 4.
    """
    if bfmmla:
        # The bfmmla v2 kernel stages the key tile's logits in L1, so the row
        # block is no longer capped by the register file and 8 rows is the
        # default at every head dim (rows are what the K/V read is amortised
        # over). ``CPU_ATTENTION_QT=4`` still selects the v1 register kernel for
        # an A/B.
        #
        # 12 and 16 rows are reachable (`CPU_ATTENTION_QT`) and the kernel has
        # the geometry for them, but they are not selected: an interleaved
        # 6/8/12/16 sweep at 8 threads on the per-head dispatch measures qt8
        # best at *every* shape, including the short ones
        #
        #     shape            qt6   **qt8**  qt12   qt16
        #     128:8:64          25   **23**    26     24
        #     256:16:128       231  **207**   229    214
        #     512:8:64         249  **226**   239    235
        #     512:32:128      1682 **1524**  1615   1561
        #     1024:16:128     3145 **2876**  3014   2944
        #
        # (16 rows *was* the right answer for 128:8:64 under the older
        # round-robin dispatch, by 10%, because that dispatch wasted a third of
        # the call at the barrier -- `ca0a85e` removed that, and the extra rows
        # then cost more than the per-item overhead they save.)
        return 8
    if lanes < 16:
        # fp32 v2 (tile logits in L1) wants 8 rows at every head dim: rows are
        # what the K/V read is amortised over and the register file no longer
        # caps them. ``CPU_ATTENTION_QT=4`` selects the old online-softmax path,
        # which wanted 6 (measured 1.09-1.11x over 4 at D<=64, 1.42-1.73x at
        # D=128 back when the accumulator lived in L1 and rows cost kt + dc
        # registers -- kept here only as history).
        return 8
    # x86. Measured on m8i (4 physical cores), 1/4/8 threads, interleaved:
    # qt=8 is 1.19x at head_dim 64 fp32 and 1.14x at 128, and the row effect is
    # 1.18-1.19x no matter which kernel runs (register 1.19x, L1-accumulator
    # 1.18x), so this is the rows, not the kernel. bf16 D=64 is the exception:
    # 8 rows x 64 dims of accumulator is 32 zmm, and the spill instructions eat
    # more than the wider sweep gains (a wash). head_dim 32 gains nothing (0.94x).
    # ---- bf16 without an AMX kernel (ARM, and x86 when AMX is unavailable) --
    if is_bf16:
        if seq_k < 1024:
            return 4 if seq_k < 256 else 6
        # 6, not 8, in the L1-accumulator bf16 kernel: 1.03-1.06x at head_dim
        # 128 (48 zmm of accumulator against 64 at 8 rows). At head_dim 64 the
        # register dot kernel is not row-limited and 4 vs 6 is a wash
        # (0.98-1.04x), so 4 stays.
        return 6 if head_dim >= 128 else 4
    # ---- fp32 -------------------------------------------------------------
    # Rows are the only axis K/V is reused across, so traffic per row-key is
    # D*4/qt bytes for Q@K^T and again for P@V. Both boxes were swept; where
    # they disagree the Xeon wins, since it is the target:
    #   Ryzen (8 cores, 32 MB L3): 6 beats 4 by 1.13x at 256x16x128 and 1.25x
    #     at 512x32x128, and 8 beats 6 by nothing anywhere.
    #   Xeon (2 cores, 480 MB L3), interleaved medians of 5: head_dim 64 ->
    #     6 beats 4 by 1.19-1.40x and 8 by 5-18%; head_dim 128 -> 10 beats 8 by
    #     1.05-1.08x and 4 by 1.5x at seq >= 1024, while 512x32x128 prefers 8
    #     (1.08x over 6).
    if seq_k < 256:
        # Too little work per work item for a wide row block: 4 beats 8 by 1.5x
        # at 128x8x64.
        return 4
    if head_dim == 32:
        return 4
    if head_dim <= 64:
        return 6
    if seq_k < 1024:
        return 8 if seq_k >= 512 else 6
    return 10


_AMX_ENV = os.environ.get("CPU_ATTENTION_AMX", "1") != "0"
# fp32 goes through the same kernel with bf16 hi/lo operand splitting (three
# products per phase); this switch is for A/B and for falling back if a future
# CPU disagrees with the measured 5.5-7.6e-5 error.
_AMX_FP32_ENV = os.environ.get("CPU_ATTENTION_AMX_FP32", "1") != "0"

# Profiling only (see tests/bench_amx_phases.py): 1 skips the K/V pack, 2 skips
# the attention. The results are garbage in either mode; timings only.
# Profiling only: 1 skip pack, 2 skip attention, 4/8 skip V/K pack, 16/32/64
# skip Q@K^T / softmax / P@V. Results are garbage in any of these; timings only.
_AMX_MODE_V = int(os.environ.get("CPU_ATTENTION_AMX_MODE", "0") or 0)

_AMX_ENTRY = None

# v2 of the AMX kernel (`dense_amx_v2.mojo`: one softmax pass per logit instead
# of two). Opt-in so a regression cannot reach the engine:
# `CPU_ATTENTION_AMX2=1` at import, or flip `_AMX_IMPL` between calls (which is
# how tests/bench_amx_v2.py interleaves the two in one process).
_AMX_IMPL = os.environ.get("CPU_ATTENTION_AMX2", "0") == "1" and "v2" or "v1"
_AMX_ENTRIES = {}


def amx_available() -> bool:
    """True when this build can run the AMX dense kernel (bf16 or fp32).

    Cached: the first call asks the kernel for tile-data permission
    (`arch_prctl`), which is a syscall, and the answer cannot change inside a
    process. Doing it here (rather than per call) is also what makes the tile
    instructions legal at all -- `#XF` state is off by default, and the first
    tile instruction is SIGILL without it. Torch's oneDNN normally asks for it
    too, but that is an implementation detail of the wheel, not a guarantee.
    """
    global _AMX_ENTRY
    if _AMX_ENTRY is None:
        mod = module()
        gate = getattr(mod, "dense_has_amx", None) if mod else None
        init = getattr(mod, "dense_amx_init", None) if mod else None
        global _AMX_PLAN
        if gate is None or init is None or not gate() or not init():
            _AMX_ENTRY = False
        else:
            _AMX_ENTRIES["v1"] = mod.dense_amx_attention
            gate2 = getattr(mod, "dense_amx2_has_amx", None)
            if gate2 is not None and gate2():
                _AMX_ENTRIES["v2"] = mod.dense_amx2_attention
            _AMX_ENTRY = True
            _AMX_PLAN = getattr(mod, "dense_amx_plan", None)
    return _AMX_ENTRY is not False and _AMX_ENTRY is not None and _AMX_ENV


_AMX_PLAN = None
"""``dense_amx_plan``: the 5-argument entry (see ``_amx_plan_buffer``)."""

_AMX_PLANS = threading.local()
"""Geometry -> the int64 buffer that entry reads, per thread.

The AMX call marshalled 36 Python objects and rebuilt twelve derived ints on
every call. Rebuilding them is 3.7 us of a 43 us call at 128:8:64 -- the row
where the kernel is already faster than vLLM's, so the call path *is* the
margin there. Everything in the buffer changes only with the geometry and the
tensor strides; the four tensor addresses still cross per call.

Bounded by entry count and dropped by `clear_scratch`, like the other
workspace. The buffer holds the two packed-K/V workspace addresses, so a
cleared workspace must not leave a stale plan behind.
"""

AMX_PLAN_MAGIC = 0x414D58504C414E31
"""Mirrors ``_C.AMX_PLAN_MAGIC``: the kernel rejects a buffer without it."""

_AMX_KT = int(os.environ.get("CPU_ATTENTION_AMX_KT") or 256)


def _amx_plan_key(q, k, v, fp32: bool, causal: bool):
    return (q.shape, k.shape, q.stride(), k.stride(), v.stride(), fp32,
            causal, _AMX_KT, _AMX_MODE_V, torch.get_num_threads())


def _amx_plan_buffer(q, k, v, fp32, causal, kp, vp, kp_bh, vp_bh, kb16, kb32,
                     dc32, dc16, scale, kt, workers):
    """Pack one AMX call's geometry into an int64 buffer for ``dense_amx_plan``.

    Field order is fixed and mirrored in ``_C.dense_amx_plan``; slot 32 is a
    magic word so a stale buffer is refused rather than misread. The strides
    are the caller's element strides (the kernel scales by the element size),
    and the output strides are derived, not read: the output is a fresh
    contiguous tensor.
    """
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    qs, ks, vs = q.stride(), k.stride(), v.stride()
    scale_bits = struct.unpack("<i", struct.pack("<f", float(scale)))[0]
    plan = torch.tensor([
        kp.data_ptr(), vp.data_ptr(),
        batch, num_heads, num_kv_heads, seq_k, seq_q, head_dim,
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        qs[0], qs[1], qs[2],
        seq_q * num_heads * head_dim, num_heads * head_dim, head_dim,
        kp_bh, vp_bh, kb16, kb32, dc32, dc16,
        scale_bits, int(bool(causal)), kt, workers, int(fp32),
        int(_AMX_MODE_V), AMX_PLAN_MAGIC,
    ], dtype=torch.int64)
    return plan


def _amx_dense(q, k, v, scale: float, causal: bool):
    """Run the AMX kernel, or return None when it cannot take this call.

    bf16 is native; fp32 runs the same kernel with every operand split into bf16
    hi/lo and three products per phase (fp32-accurate at 3x the tile ops). One
    work item is 32 query rows of one (batch, head); anything else -- no AMX in
    this build, an unsupported head dim, the switch turned off -- falls back to
    the AVX-512 path in this file.
    """
    if not amx_available():
        return None
    if q.shape[-1] not in (32, 64, 128) or q.dtype not in _DTYPE_CODES:
        return None
    fp32 = q.dtype == torch.float32
    plans = _AMX_PLANS.__dict__.get("p") if _AMX_IMPL == "v1" else None
    if plans:
        plan = plans.get(
            (q.shape, k.shape, q.stride(), k.stride(), v.stride(), fp32,
             causal, _AMX_KT, _AMX_MODE_V, torch.get_num_threads())
        )
        if plan is not None:
            out = torch.empty_like(q, memory_format=torch.contiguous_format)
            _AMX_PLAN(plan.data_ptr(), q.data_ptr(), k.data_ptr(),
                      v.data_ptr(), out.data_ptr())
            return out
    if fp32 and not _AMX_FP32_ENV:
        return None
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    if seq_q < 1 or seq_k < 1 or num_kv_heads < 1:
        return None
    # The layouts the kernel reads: one 1 KB tile per (key block, dim chunk);
    # fp32 packs an (hi, lo) slab pair, so the pitch is 1024 bf16, not 512.
    kb32 = -(-seq_k // 32)          # V blocks: 32 keys
    kb16 = kb32 * 2                 # K blocks: 16 keys
    dc32, dc16 = head_dim // 32, head_dim // 16
    slab = 1024 if fp32 else 512
    kp_bh, vp_bh = kb16 * dc32 * slab, kb32 * dc16 * slab
    nbh = batch * num_kv_heads
    kp = _scratch("amx_kp", (nbh * kp_bh,), torch.bfloat16)
    vp = _scratch("amx_vp", (nbh * vp_bh,), torch.bfloat16)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    qs, ks, vs, os_ = q.stride(), k.stride(), v.stride(), out.stride()
    # 256-key super-tiles measured best on the Xeon (128 is 6% slower, 384/512
    # lose the logits+prob+acc working set out of L1).
    kt = _AMX_KT
    tasks = batch * num_heads * (-(-seq_q // 32))
    workers = _workers(tasks, 8)
    # The plan entry only exists for v1 (v2 has its own packed layouts but the
    # same arguments), so a v2 call takes the 36-argument path.
    _AMX_ENTRIES.get(_AMX_IMPL, _AMX_ENTRIES["v1"])(
        q.data_ptr(), kp.data_ptr(), vp.data_ptr(), out.data_ptr(),
        k.data_ptr(), v.data_ptr(),
        batch, seq_q, seq_k, num_heads, num_kv_heads, head_dim,
        qs[0], qs[1], qs[2],
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        os_[0], os_[1], os_[2],
        kp_bh, vp_bh, kb16, kb32, dc32, dc16,
        float(scale), int(bool(causal)), kt,
        workers, int(fp32), int(_AMX_MODE_V),
    )
    if _AMX_PLAN is not None and _AMX_IMPL == "v1":
        plans = _AMX_PLANS.__dict__.get("p")
        if plans is None:
            plans = _AMX_PLANS.p = {}
        if len(plans) > 64:
            plans.clear()
        plans[_amx_plan_key(q, k, v, fp32, causal)] = _amx_plan_buffer(
            q, k, v, fp32, causal, kp, vp, kp_bh, vp_bh, kb16, kb32, dc32,
            dc16, scale, kt, workers,
        )
    return out




def _plans():
    """This thread's dense call plans, built on first use.

    A dense call's *geometry* -- which kernel path, ``qt``, the worker count, the
    scratch buffer and its strides, and the ~30 constant arguments the extension
    takes -- is a pure function of ``(q.shape, q.stride(), k.stride(),
    v.stride(), dtype, causal)``; only the data pointers, ``scale`` and the
    freshly allocated output change between calls. Rebuilding it measured 2.0 us
    of a 5.0 us Python path (laptop, 4 threads), i.e. 4% of a whole 52 us call
    at S=128/8/64 and most of the fixed cost those rows are made of.

    The cache holds no reference to the caller's tensors -- only ints, shape
    tuples, and workspace buffers this module owns -- so it cannot keep a
    caller's tensor alive, which is what made an earlier whole-call cache lose
    when it was keyed on tensors. Each entry holds its scratch buffer, so the
    pointers inside a plan stay valid exactly as long as those buffers do
    (:func:`clear_scratch` clears both). Thread-local, because the buffers are.
    """
    plans = getattr(_SCRATCH, "plans", None)
    if plans is None:
        plans = _SCRATCH.plans = {}
    return plans


def _dense_plan(key, q, k, v, causal):
    """Build the per-geometry call plan for :func:`_dense` (slow path).

    Everything here depends only on ``(shapes, strides, dtype, causal)``, so it
    runs once per geometry and the result is cached per thread (see
    :func:`_plans`). The scratch buffers it allocates are *held by the plan*, so
    the pointers inside it stay valid for as long as the entry does.
    """
    qs, q_st, k_st, v_st, dt = key[0], key[1], key[2], key[3], key[4]
    # Bound the packed-K/V workspace by bytes before building anything: a sweep
    # over large shapes would otherwise keep every geometry's scratch alive at
    # once (a 22-row bench_vs_refs run reaches several hundred MB, and the rows
    # it measures late are 5-15% slower than the same rows run on their own --
    # medians of 9 on the Graviton box, and the fingerprint is positional, not
    # shape-dependent). Done here rather than inside `_scratch` so it can never
    # fire in the middle of a build.
    # (This runs only on a plan *miss*, so a geometry that is already planned
    # never reaches it and cannot evict itself.)
    _live = getattr(_SCRATCH, "bufs", None)
    if _live and _plans() and _workspace_bytes(_live) > _SCRATCH_BYTES:
        _drop_workspace()
    batch, seq_q, num_heads, head_dim = qs
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    bfmmla = dt == torch.bfloat16 and _bfmmla()
    is_bf16 = dt == torch.bfloat16
    if _QT_ENV:
        qt = int(_QT_ENV)
    else:
        qt = _qt_for(_lanes(), is_bf16, head_dim, seq_k, bfmmla)

    # Q@K^T wants K transposed (one head dim, all positions contiguous), which a
    # dense K is not. Each bf16 path asks for its own packing: the dot pairs two
    # head dims of a position, bfmmla pairs two positions of four head dims and
    # needs V repacked too (its P@V contracts over keys). Padded to the key tile
    # (64) so the last tile never reads past the buffer; the mask kills the
    # padding lanes.
    #
    # V gets a plain contiguous copy per (batch, kv head) when it no longer fits
    # in the LLC, because the P@V sweep reads one head's V rows once per key and
    # a dense V puts consecutive tokens `heads * head_dim` apart -- 8 KB at 16
    # heads x 128 dims, a stride the hardware prefetcher cannot follow. vLLM's
    # paged cache hands its P@V sweep that layout for free
    # (`[block][head][block_size][head_dim]`: 512 B per token); a dense call has
    # to pay one linear copy (33 MB at 40-70 GB/s, 1-2% of the call). Measured
    # on the Ryzen, interleaved, bit-identical output (`max|diff| 0`):
    #
    #   S=4096 H=16 D=128 fp32 (V 33.5 MB): 1.90x
    #   S=2048 H=16 D=128 fp32 (V 16.8 MB): 1.04x
    #   S=4096 H= 8 D= 64 fp32 (V  8.4 MB): 0.99x
    #   S=4096 H=16 D=128 bf16 (V 16.8 MB): 0.94x
    #
    # so it is gated on fp32 (whose P@V is V-bandwidth-bound; bf16 spends its
    # P@V instructions widening V instead) and on V being at least a third of
    # the LLC. The gate was `2 * V >= LLC` (half) when it was measured on a
    # 32 MB-L3 laptop; on this box (48 MB) that left 2048:16:128 fp32 (V =
    # 16.8 MB) *without* the copy, and forcing it there measured 1.02-1.07x in
    # three interleaved runs while every smaller shape stayed within noise
    # (`tests/bench_f32_vpack.py`). A third of the LLC fires at exactly the
    # shapes the half rule did on the laptop, plus that one.
    vpack = (dt == torch.float32
             and v.numel() * v.element_size() * 3 >= _llc_bytes())
    # The P@V loops work on 4-key groups, and the last tile's key range is only
    # known at runtime, so a kernel that rounds that range up to a whole group
    # reads up to three V rows past the end of the tensor. With V read in place
    # that is out of bounds -- `tests/fuzz_dense.py`'s stray-read probe catches it
    # as a NaN coming out of a neighbouring allocation, and it is a page fault
    # away from crashing (it was a real fp32 bug, `seq_q != seq_k`). Padding the
    # scratch by one group and zeroing the tail costs a copy only for the odd
    # key counts this affects, and nothing at all for the benchmark's shapes.
    vpad = (seq_k % 4) != 0
    seq_k_pad = -(-seq_k // 64) * 64
    # K^T is blocked every `kt_blk` keys so one Q@K^T tile is a single linear run
    # (`[tile][dim][kt_blk]`) instead of `head_dim` rows `kt_blk * elt` apart --
    # 16 KB at seq_k 4096. Same reason as the V copy below, and the same
    # measurement: handed the blocked layout, S=4096/H=16/D=128 fp32 gains
    # 1.2-1.4x over the plain `[head_dim, seq_k]` transpose. `seq_k_pad` (i.e. no
    # blocking) keeps the 4-lane/ARM layouts byte-identical to before.
    # Every fast path blocks K^T by 64 keys, and the tile kernels take that as a
    # *compile-time* stride (a runtime stride turned each address into a `madd`).
    kt_blk = 64
    ks = None
    if bfmmla and _int8_qk():
        mode = 3
        # int8 K for i8mm: [tile][head-dim group of 8][key pair][16 int8], plus
        # one dequant scale per 64-key tile. Half the bytes of the bf16 layout
        # and the same block structure, so the Q@K^T sweep still walks each
        # tile's K linearly.
        kt_blk = 64
        kt = _scratch("kt3", (batch, num_kv_heads, seq_k_pad // kt_blk,
                              head_dim // 8, kt_blk // 2, 16), torch.int8)
        ks = _scratch("ks3", (batch, num_kv_heads, seq_k_pad // kt_blk),
                      torch.float32)
        vt = _scratch("vt2", (batch, num_kv_heads, seq_k_pad // 4, head_dim // 2, 8),
                      dt)
    elif bfmmla:
        mode = 2
        # K blocked by key tile (64 keys): one tile's K for every head-dim group
        # is then one contiguous run, so the Q@K^T sweep walks it linearly
        # instead of jumping seq_k/2*16 bytes per head-dim group.
        # [tile][key pair][head-dim group][8]: one key pair's whole head dim
        # is a single linear run, which is the order the Q@K^T sweep reads.
        kt = _scratch("kt2", (batch, num_kv_heads, seq_k_pad // kt_blk,
                              kt_blk // 2, head_dim // 4, 8), dt)
        # V: [key group of 4][head-dim pair][8], so P@V's inner loop walks one
        # key group's dim pairs as a contiguous run (512 B at head_dim 64).
        vt = _scratch("vt2", (batch, num_kv_heads, seq_k_pad // 4, head_dim // 2, 8),
                      dt)
    elif is_bf16 and _bf16_dot():
        mode = 1
        kt = _scratch("kt1", (batch, num_kv_heads, head_dim // 2,
                              seq_k_pad // kt_blk, kt_blk, 2), dt)
        vt = (_scratch("vt1", (batch, num_kv_heads, seq_k + 4 * int(vpad),
                               head_dim), dt) if (vpack or vpad) else None)
    else:
        mode = 0
        kt = _scratch("kt0", (batch, num_kv_heads, seq_k_pad // kt_blk, head_dim,
                              kt_blk), dt)
        vt = (_scratch("vt0", (batch, num_kv_heads, seq_k + 4 * int(vpad),
                               head_dim), dt) if (vpack or vpad) else None)
    # The pack's strides: `kt_blk` keys per contiguous K^T block, and the V
    # scratch's per-head base and per-key-group stride. Both are properties of
    # the buffer, so they belong to the plan and not to the call.
    if vt is not None and vpad:
        # Once per geometry: the pack writes rows [0, seq_k) on every call and
        # never touches the tail, so one zeroing keeps the over-read harmless.
        vt[:, :, seq_k:, :].zero_()
    kt_st, vt_st = _scratch_strides(kt, vt)
    kt_bh, kt_d = kt_st[1], kt_st[3]
    vt_bh = vt_st[1] if vt is not None else 0
    vt_d = vt_st[2] if vt is not None else 0
    ks_stride_bh = (seq_k_pad // kt_blk) if ks is not None else 0
    ks_ptr = ks.data_ptr() if ks is not None else 0
    qb, qtok, qh = q_st[0], q_st[1], q_st[2]
    # In mode 2 the kernel reads the packed V (one row per head dim *pair*, one
    # column per 4-key group), so hand it those strides instead.
    if vt is not None:
        vb, vtok, vh = vt_st[0], vt_st[2], vt_st[1]
        vt_ptr = vt.data_ptr()
        v_ptr = vt_ptr
    else:
        vb, vtok, vh = v_st[0], v_st[1], v_st[2]
        vt_ptr = 0
        v_ptr = 0
    # The output is always a fresh contiguous tensor of q's shape, so its
    # strides follow from the shape -- no `.stride()` call on the hot path.
    ob, otok, oh = seq_q * num_heads * head_dim, num_heads * head_dim, head_dim
    ktb = kt_st[1]
    if mode == 2 or mode == 3:
        # mode 2 [tile][key pair][dim group][8]: stride(3) is one key pair, 8 a
        # head-dim group; mode 3 [tile][dim group][key pair][16]: stride(3) is
        # one head-dim group. The kernel derives the tile stride from the
        # dim-group count.
        ktd, ktt = kt_st[3], 8
    else:
        ktd = kt_st[3]                          # dim stride within a block
        ktt = kt_st[4] if mode == 1 else 1      # token stride inside a block
    tasks = batch * num_heads * (-(-seq_q // qt))
    workers = _workers(tasks, 8)
    # ---- per-head dispatch (mode += 4) -------------------------------------
    # One thread packs a kv head and then runs *all* of that head's query
    # blocks, with no barrier between the phases. Requirements:
    #   * MHA: the pack task index and the attention item index have to be the
    #     same (batch, head), so every query head must have its own kv head;
    #   * perfect balance: items % workers == 0, which holds for every head
    #     count in the benchmark matrix at 8 threads (8/16/32) and is checked
    #     rather than assumed;
    #   * short enough S that a thread's K/V stays in L1/L2 (the win is the
    #     per-thread footprint and the barrier, both of which vanish when one
    #     head's K/V no longer fits).
    # `perf` on 128:8:64 bf16 put 29% of the call in the region's barrier; this
    # is the dispatch that removes it.
    head_items = batch * num_heads
    head_mode = int(_HEAD_MAXK or 0)
    # Heads per thread: see `_HEAD_RATIO` -- fp32 stops paying at 2, bf16 still
    # gains at 4.
    head_ratio = head_items // max(workers, 1)
    ratio_ok = is_bf16 or _HEAD_RATIO <= 0 or head_ratio <= _HEAD_RATIO
    if (
        head_mode
        and ratio_ok
        and num_heads == num_kv_heads
        and seq_k <= head_mode
        and head_items >= workers
        and head_items % workers == 0
    ):
        mode += 4
    # ---- short-sequence kernel (mode += 8) ---------------------------------
    # `dense_small.mojo`: the same geometry with the key coverage rounded to 8
    # keys instead of 16, in its own region (it needs the pack fused per head,
    # which the mode>=8 entry does itself). bf16 only -- the fp32 tile kernel's
    # own rounding costs 2% of its call, not 11%. `_FUSE` because the non-fused
    # path packs in a separate region. Measured (in-process, 21 rounds, min):
    # 128:8:64 1.020x, 256:16:128 1.042-1.068x; 512:8:64 is 0.94x, which is why
    # the threshold is 256 and not higher.
    if (
        _SMALL_MAXK
        and mode == 6  # bf16 bfmmla layout + per-head
        and _FUSE
        and seq_k <= _SMALL_MAXK
        and head_dim <= 128  # D=128 measured too: 256:16:128 bf16 1.068x in
        # `tests/bench_small.py`; the tile geometry (ks=256) is the same kernel
    ):
        mode += 8
    plan = (
        seq_q, num_heads, batch, num_kv_heads, seq_k, head_dim,
        _DTYPE_CODES[dt], mode, ktb, ktd, ktt, kt_blk, qt, workers,
        kt_bh, kt_d, vt_bh, vt_d, kt.data_ptr(), vt_ptr, v_ptr, ks_ptr,
        ks_stride_bh, k_st[0], k_st[1], k_st[2], v_st[0], v_st[1], v_st[2],
        qb, qtok, qh, vb, vtok, vh, ob, otok, oh,
        _workers(batch * num_kv_heads, seq_k // 8),
        # held for their lifetimes, so the pointers above stay valid
        kt, vt, ks,
    )
    # The geometry as an int64 buffer, so a call passes seven arguments to the
    # extension instead of 42 (see `_C.dense_forward_plan`). Indices are into
    # the `plan` tuple above -- no names, so this cannot pick up the wrong
    # local. `tests/test_dense.py` requires bit-identical output from both
    # entries, which is the check the first attempt at this change lacked.
    pbuf = torch.tensor(
        [plan[i] for i in _PLAN_ABI_FIELDS] + [int(causal)], dtype=torch.int64
    )
    plan = plan + (pbuf,)
    plans = _plans()
    if len(plans) > 64:
        # The scratch budget in :func:`_scratch` is the bound that matters; this
        # one only stops the dict itself growing without limit.
        _drop_workspace()
    plans[key] = plan
    return plan


_CPP_PLANS = threading.local()
"""Per-thread dense plans for the C++ backend: geometry -> (buffer, address).

The C++ side keeps the workspace (packed K/V, accumulators) keyed on the same
geometry, so the Python side only has to hand it the plan; every derived value
(rows per work item, ISA, worker count) is computed once, on the miss path.
"""

def _cpp_entries():
    """Bind the C++ plan/forward pair once per backend."""
    return _entry("dense_make_plan"), _entry("dense_forward")


def _cpp_plan(q, k, v, scale, causal):
    """The plan address for this geometry, building it on the first call."""
    plans = getattr(_CPP_PLANS, "p", None)
    if plans is None:
        plans = _CPP_PLANS.p = {}
    batch, seq_q, heads, hd = q.shape
    kv_heads = k.shape[2]
    workers = _workers(batch * heads * -(-seq_q // 64), 8)
    key = (q.shape, q.stride(), k.stride(), v.stride(), causal, q.dtype,
           float(scale), workers)
    addr = plans.get(key)
    if addr is None:
        make_plan, _ = _cpp_entries()
        qs, ks, vs = q.stride(), k.stride(), v.stride()
        buf = torch.empty(24, dtype=torch.int64)
        make_plan(
            buf.data_ptr(), batch, seq_q, k.shape[1], heads, kv_heads, hd,
            _DTYPE_CODES[q.dtype], int(bool(causal)), float(scale),
            qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], vs[0], vs[1], vs[2],
            seq_q * heads * hd, heads * hd, hd, workers,
        )
        if len(plans) > 64:
            plans.clear()
        addr = buf.data_ptr()
        # The buffer has to outlive the address, so hold it next to the key.
        plans[key] = addr
        _CPP_BUFS.append(buf)
    return addr


_CPP_BUFS = []


def _dense_cpp(q, k, v, scale: float, causal: bool):
    """Dense attention through the C++ backend: one call, five arguments."""
    _, forward = _cpp_entries()
    out = torch.empty(q.shape, dtype=q.dtype)
    forward(_cpp_plan(q, k, v, scale, causal), q.data_ptr(), k.data_ptr(),
            v.data_ptr(), out.data_ptr())
    return out


def _dense(q, k, v, scale: float, causal: bool, geom=None):
    """Run the dense kernel over contiguous-``head_dim`` q/k/v (already checked).

    ``geom`` is ``(q.shape, q.stride(), k.stride(), v.stride(), q.dtype)`` when
    the caller has already read them (``flash_attn_func`` has to, for its
    argument checks); otherwise they are read here.

    The Python side of this is not free, and it is charged to us by the
    scorecard: at S=128/8/64 it is a tenth of the whole call. Three things keep
    it small. One ``.shape``/``.stride()`` per tensor and no ``.stride(i)`` (each
    is a ~0.1-0.2 us Python call, and this path once made nineteen of them).
    Everything that depends only on the geometry -- which kernel path, ``qt``,
    the worker count, the scratch buffer, its strides, and the ~30 constant
    arguments the extension takes -- is built once per geometry by
    :func:`_dense_plan` and cached, which took the path from 3.5 us to ~1.5 us.
    What is left per call is the output allocation (0.53 us), four
    ``data_ptr()`` reads (0.25), one dict lookup (0.05) and the extension call
    itself -- measured at **0.76 us for a 42-argument call with zero work in it**,
    so the marshal is the floor and cutting the argument count is the only way
    left. It is not worth it: that is 1.5% of a 51 us call at S=128 and nothing
    above it.
    """
    if _BACKEND == "cpp":
        return _dense_cpp(q, k, v, scale, causal)
    out = _amx_dense(q, k, v, scale, causal)
    if out is not None:
        return out
    if geom is None:
        geom = (q.shape, q.stride(), k.stride(), v.stride(), q.dtype)
    # Two knobs that are not part of the geometry but do change the plan, so
    # they belong in the key: `_QT_ENV` is flipped between calls by the `qt`
    # sweeps (that is how the table in `_qt_for` was measured), and the worker
    # count follows `torch.set_num_threads` and `set_num_workers`. A stale plan
    # here is silent and expensive -- it would keep running 4 workers after the
    # caller asked for 1 and look like a 3x regression in a sweep.
    key = (geom[0], geom[1], geom[2], geom[3], geom[4], causal, _QT_ENV,
           _OVERRIDE or torch.get_num_threads(), _HEAD_MAXK, _HEAD_RATIO,
           _SMALL_MAXK, _llc_bytes())
    # ... and the LLC size, because it decides whether V gets copied: a box
    # whose cache size is reported differently (or a test that patches the gate)
    # must not keep the other layout's plan.
    plan = _plans().get(key)
    if plan is None:
        plan = _dense_plan(key, q, k, v, causal)
    (seq_q, num_heads, batch, num_kv_heads, seq_k, head_dim, dtc, mode,
     ktb, ktd, ktt, kt_blk, qt, workers, kt_bh, kt_d, vt_bh, vt_d, kt_ptr,
     vt_ptr, v_ptr_packed, ks_ptr, ks_bh, k0, k1, k2, v0, v1, v2,
     qb, qtok, qh, vb, vtok, vh, ob, otok, oh, pack_workers,
     _kt, _vt, _ks, _pbuf) = plan
    out = torch.empty(geom[0], dtype=geom[4])
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()
    v_ptr = v_ptr_packed if v_ptr_packed else v.data_ptr()
    o_ptr = out.data_ptr()
    if _FUSE:
        # One OpenMP region for the pack and the attention, with a barrier
        # between them: two fork-joins cost two pool wakes, and at S=128 the
        # barrier was 14% of the process by perf. `CPU_ATTENTION_NO_FUSE=1`
        # restores the two-call form, which is what `tests/bench_phases.py`
        # instruments to split pack from kernel.
        if _PLAN_ABI:
            # Six arguments instead of 42: the geometry rides in `_pbuf`.
            _forward_plan()(
                _pbuf.data_ptr(), k_ptr,
                q_ptr, v_ptr if not vt_ptr else v.data_ptr(), v_ptr, o_ptr, scale,
            )
            return out
        _forward()(
            k_ptr, kt_ptr, v_ptr if not vt_ptr else v.data_ptr(), vt_ptr, v_ptr,
            batch, num_kv_heads, seq_k, head_dim, k0, k1, k2, vt_bh, vt_d,
            v0, v1, v2, q_ptr, o_ptr, seq_q, num_heads, qb, qtok, qh,
            vb, vtok, vh, ob, otok, oh, scale, causal, dtc, mode, ktb, ktd,
            ktt, kt_blk, qt, workers, ks_ptr, ks_bh,
        )
        return out
    _transpose_attention()(
        k_ptr, kt_ptr, v_ptr if not vt_ptr else v.data_ptr(),
        vt_ptr, batch, num_kv_heads, seq_k, head_dim, k0, k1, k2,
        kt_bh, kt_d, kt_blk, v0, v1, v2, vt_bh, vt_d, dtc, mode,
        pack_workers, ks_ptr, ks_bh,
    )
    _dense_attention()(
        q_ptr, kt_ptr, v_ptr, o_ptr, batch, seq_q, seq_k, num_heads,
        num_kv_heads, head_dim, qb, qtok, qh, ktb, ktd, ktt, kt_blk,
        vb, vtok, vh, ob, otok, oh, scale, causal, dtc, qt, workers,
        ks_ptr, ks_bh,
    )
    return out
