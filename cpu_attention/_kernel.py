"""Loader and call plumbing for the Mojo attention kernels.

The kernel is built from ``cpu_attention/_mojo`` into ``cpu_attention/_C.so``
by ``uv run --extra build python scripts/build.py``. Set ``CPU_ATTENTION_LIB``
to point at another build. Everything here is optional: without the shared
library (or on an unsupported configuration) :func:`is_available` is False and
every entry point raises ``RuntimeError``.
"""

import importlib.util
import os

import torch

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_C.so")

# The kernel specializes on head_dim at compile time; anything else raises.
_HEAD_DIMS = {32, 64, 128}
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1}

_BUILT = None


def _load():
    path = os.environ.get("CPU_ATTENTION_LIB", _LIB_PATH)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("cpu_attention._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module():
    """Return the compiled kernel module, or None if it is unavailable."""
    global _BUILT
    if _BUILT is None:
        try:
            _BUILT = _load()
        except Exception:  # noqa: BLE001
            _BUILT = False
    return _BUILT or None


def is_available() -> bool:
    """Return True if the compiled kernel is present and loadable."""
    return module() is not None


def require():
    """Return the compiled kernel module, raising RuntimeError if unavailable."""
    mod = module()
    if mod is None:
        raise RuntimeError(
            "cpu_attention: the Mojo kernel is not built; run "
            "`uv run --extra build python scripts/build.py` (or set "
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


_ATTENTION = None


def _attention():
    """Return the extension's single callable, bound once."""
    global _ATTENTION
    if _ATTENTION is None:
        _ATTENTION = require().attention
    return _ATTENTION


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
    """Pin the worker count for every call (0 restores the automatic rule)."""
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


_TRANSPOSE = None


def _transpose_attention():
    """Return the extension's K-transpose entry point, bound once."""
    global _TRANSPOSE
    if _TRANSPOSE is None:
        _TRANSPOSE = require().dense_transpose
    return _TRANSPOSE


_DENSE = None


def _dense_attention():
    """Return the extension's dense entry point, bound once."""
    global _DENSE
    if _DENSE is None:
        _DENSE = require().dense_attention
    return _DENSE


_LANES = None


def _lanes() -> int:
    """fp32 lanes per vector in the built kernel (16 on AVX-512, 4 on NEON)."""
    global _LANES
    if _LANES is None:
        _LANES = int(require().dense_lanes())
    return _LANES


_BF16_DOT = None


def _bf16_dot() -> bool:
    """True when the built kernel has a bf16 dot-product instruction.

    The transposed-K layout depends on it (paired head dims for the dot, one
    head dim per position otherwise), so the caller has to know which one the
    compiled kernel expects.
    """
    global _BF16_DOT
    if _BF16_DOT is None:
        _BF16_DOT = bool(require().dense_bf16_dot())
    return _BF16_DOT


_BFMMLA = None


def _bfmmla() -> bool:
    """True when the bf16 path runs on NEON ``bfmmla``.

    That kernel reads a different K *and* V layout (position-paired K over four
    head dims, position-major V over head-dim pairs), so the caller allocates
    both scratch buffers and packs V as well as K.
    """
    global _BFMMLA
    if _BFMMLA is None:
        _BFMMLA = bool(require().dense_bfmmla())
    return _BFMMLA



_QT = 4
"""Query rows per work item (``CPU_ATTENTION_QT`` overrides it).

Bounded by the register file: the output accumulator is ``qt * head_dim / LANES``
zmm and the tile logits another ``qt * 64 / LANES``, so ~32 registers cap
``qt * head_dim`` near 512. Bigger blocks amortize the K^T/V loads over more rows
and shrink the per-tile softmax, but spill the accumulator; 4 measured best for
every supported head dim and dtype on the AVX-512 boxes (2 and 8 are switchable
through the environment).
"""

_QT_ENV = os.environ.get("CPU_ATTENTION_QT")


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
    head_dim 64). head_dim 128 is left at 2: 32 registers per row already.

    bf16 on ARM is a different path (``bfmmla``) that keeps its own tile
    geometry, and head_dim picks the rows there; x86 (``lanes == 16``) keeps 4.
    """
    if bfmmla:
        # The bfmmla kernel carries two query rows per 2x2 tile and charges the
        # whole softmax per (row block, key tile), so rows per work item are
        # nearly free. head_dim decides: at 64 there is room for a 16-key tile
        # with 4 rows (849 GFLOP/s vs 828 with 8 rows/8 keys), at 128 the key
        # tile halves to 8 keys to stay inside 32 registers either way and 8
        # rows wins (1122 vs 1032).
        return 8 if head_dim >= 128 else 4
    if lanes < 16:
        if head_dim <= 64:
            return 6 if seq_k >= 1024 else 4
        return 2
    return _QT


def _dense(q, k, v, scale: float, causal: bool):
    """Run the dense kernel over contiguous-``head_dim`` q/k/v (already checked)."""
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    bfmmla = k.dtype == torch.bfloat16 and _bfmmla()
    if _QT_ENV:
        qt = int(_QT_ENV)
    else:
        qt = _qt_for(_lanes(), k.dtype == torch.bfloat16, head_dim, seq_k, bfmmla)

    # Q@K^T wants K transposed (one head dim, all positions contiguous), which a
    # dense K is not. Each bf16 path asks for its own packing: the dot pairs two
    # head dims of a position, bfmmla pairs two positions of four head dims and
    # needs V repacked too (its P@V contracts over keys). Padded to the key tile
    # (64) so the last tile never reads past the buffer; the mask kills the
    # padding lanes.
    seq_k_pad = -(-seq_k // 64) * 64
    if bfmmla:
        mode = 2
        kt = torch.empty(
            (batch, num_kv_heads, head_dim // 4, seq_k_pad // 2, 8), dtype=k.dtype
        )
        vt = torch.empty(
            (batch, num_kv_heads, head_dim // 2, seq_k_pad // 4, 8), dtype=v.dtype
        )
    elif k.dtype == torch.bfloat16 and _bf16_dot():
        mode = 1
        kt = torch.empty((batch, num_kv_heads, head_dim // 2, seq_k_pad, 2),
                         dtype=k.dtype)
        vt = None
    else:
        mode = 0
        kt = torch.empty((batch, num_kv_heads, head_dim, seq_k_pad), dtype=k.dtype)
        vt = None
    _transpose_attention()(
        k.data_ptr(),
        kt.data_ptr(),
        v.data_ptr() if vt is not None else 0,
        vt.data_ptr() if vt is not None else 0,
        batch,
        num_kv_heads,
        seq_k,
        head_dim,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        kt.stride(1),
        kt.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        vt.stride(1) if vt is not None else 0,
        vt.stride(2) if vt is not None else 0,
        _DTYPE_CODES[k.dtype],
        mode,
        _workers(batch * num_kv_heads, seq_k // 64),
    )

    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    qb, qtok, qh = q.stride(0), q.stride(1), q.stride(2)
    # In mode 2 the kernel reads the packed V (one row per head dim *pair*, one
    # column per 4-key group), so hand it those strides instead.
    if vt is not None:
        vb, vtok, vh = vt.stride(0), vt.stride(2), vt.stride(1)
        v_ptr = vt.data_ptr()
    else:
        vb, vtok, vh = v.stride(0), v.stride(1), v.stride(2)
        v_ptr = v.data_ptr()
    ob, otok, oh = out.stride(0), out.stride(1), out.stride(2)
    ktb, ktd, ktt = kt.stride(1), kt.stride(2), kt.stride(3)
    tasks = batch * num_heads * (-(-seq_q // qt))
    _dense_attention()(
        q.data_ptr(),
        kt.data_ptr(),
        v_ptr,
        out.data_ptr(),
        batch,
        seq_q,
        seq_k,
        num_heads,
        num_kv_heads,
        head_dim,
        qb,
        qtok,
        qh,
        ktb,
        ktd,
        ktt,
        vb,
        vtok,
        vh,
        ob,
        otok,
        oh,
        scale,
        causal,
        _DTYPE_CODES[q.dtype],
        qt,
        _workers(tasks, 8),
    )
    return out
