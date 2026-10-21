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
_CSRC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_csrc_kernels.so"
)

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

_SCRATCH = {}
"""Reusable scratch, keyed by what it is for.

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
it).
"""


def _scratch(kind, shape, dtype):
    """A workspace tensor of ``shape``/``dtype``, reused across calls."""
    key = (kind, shape, dtype)
    buf = _SCRATCH.get(key)
    if buf is None:
        # Bound the dict: one entry per (purpose, shape, dtype) alive at a time
        # is all the engine ever needs, but a caller looping over many shapes
        # would otherwise grow it without limit.
        if len(_SCRATCH) > 64:
            _SCRATCH.clear()
        buf = torch.empty(shape, dtype=dtype)
        _SCRATCH[key] = buf
    return buf


def scratch_bytes() -> int:
    """Bytes currently held as workspace (diagnostics / leak checks)."""
    return sum(t.numel() * t.element_size() for t in _SCRATCH.values())


def clear_scratch() -> None:
    """Drop every workspace buffer (tests, or after a shape sweep)."""
    _SCRATCH.clear()


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
        # head_dim 128 takes the same 6 (1.42-1.73x over the old qt=2); the
        # "32 registers per row" argument does not apply because this kernel keeps
        # the accumulator in L1, so a row costs kt + dc registers.
        return 6
    # x86. Measured on m8i (4 physical cores), 1/4/8 threads, interleaved:
    # qt=8 is 1.19x at head_dim 64 fp32 and 1.14x at 128, and the row effect is
    # 1.18-1.19x no matter which kernel runs (register 1.19x, L1-accumulator
    # 1.18x), so this is the rows, not the kernel. bf16 D=64 is the exception:
    # 8 rows x 64 dims of accumulator is 32 zmm, and the spill instructions eat
    # more than the wider sweep gains (a wash). head_dim 32 gains nothing (0.94x).
    if seq_k < 1024:
        return 4
    if head_dim == 32:
        return 4
    if is_bf16:
        return 8 if head_dim >= 128 else 4
    return 8


_CSRC = None
"""ctypes handle for ``_csrc_kernels.so`` (the gcc-build AMX kernels), or False."""

_AMX_ENV = os.environ.get("CPU_ATTENTION_AMX", "1") != "0"
_KT_ENV = os.environ.get("CPU_ATTENTION_KT", "0")
# Profiling only: "pack" skips the attention loop, "attn" skips the pack (which
# then reads a stale buffer, so the result is garbage -- timings only).
_AMX_MODE = os.environ.get("CPU_ATTENTION_AMX_MODE", "")


def _csrc():
    """The gcc-built x86 kernels, or None when they are unavailable here."""
    global _CSRC
    if _CSRC is None:
        try:
            import ctypes

            lib = ctypes.CDLL(os.environ.get("CPU_ATTENTION_CSRC", _CSRC_PATH))
            lib.dense_amx_ok.restype = ctypes.c_int
            lib.dense_amx_ok.argtypes = []
            fn = lib.dense_amx_bf16
            fn.restype = ctypes.c_int
            fn.argtypes = (
                [ctypes.c_void_p] * 4
                + [ctypes.c_int64] * 18
                + [ctypes.c_double]
                + [ctypes.c_int64] * 3
            )
            _CSRC = lib
        except (OSError, AttributeError):
            _CSRC = False
    return _CSRC or None


def amx_available() -> bool:
    """True when this build/CPU can run the AMX bf16 dense kernel."""
    lib = _csrc()
    return bool(lib is not None and _AMX_ENV and lib.dense_amx_ok() == 1)


def _amx_dense(q, k, v, scale: float, causal: bool):
    """Run the C/AMX kernel, or return None when it cannot take this call.

    One work item is 32 query rows of one (batch, head), so the kernel is only
    reachable for bf16 with a supported head dim; everything else falls back to
    the Mojo path.
    """
    lib = _csrc()
    if lib is None or not _AMX_ENV:
        return None
    if lib.dense_amx_ok() != 1 or q.shape[-1] not in (32, 64, 128):
        return None
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    qs, ks, vs, os_ = q.stride(), k.stride(), v.stride(), out.stride()
    tasks = batch * num_heads * (-(-seq_q // 32))
    rc = lib.dense_amx_bf16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        batch, seq_q, seq_k, num_heads, num_kv_heads, head_dim,
        qs[0], qs[1], qs[2],
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        os_[0], os_[1], os_[2],
        float(scale), int(bool(causal)),
        _workers(tasks, 8),
        int(_KT_ENV or 0)
        | (1 << 16 if _AMX_MODE == "pack" else 0)
        | (1 << 17 if _AMX_MODE == "attn" else 0)
        | (int(_AMX_MODE[3:], 16) << 18 if _AMX_MODE.startswith("abl") else 0),
    )
    return out if rc == 0 else None


def _dense(q, k, v, scale: float, causal: bool):
    """Run the dense kernel over contiguous-``head_dim`` q/k/v (already checked)."""
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    if q.dtype == torch.bfloat16:
        out = _amx_dense(q, k, v, scale, causal)
        if out is not None:
            return out
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
    # P@V instructions widening V instead) and on V being at least half the LLC.
    vpack = (v.dtype == torch.float32
             and v.numel() * v.element_size() * 2 >= _llc_bytes())
    seq_k_pad = -(-seq_k // 64) * 64
    # K^T is blocked every `kt_blk` keys so one Q@K^T tile is a single linear run
    # (`[tile][dim][kt_blk]`) instead of `head_dim` rows `kt_blk * elt` apart --
    # 16 KB at seq_k 4096. Same reason as the V copy below, and the same
    # measurement: handed the blocked layout, S=4096/H=16/D=128 fp32 gains
    # 1.2-1.4x over the plain `[head_dim, seq_k]` transpose. `seq_k_pad` (i.e. no
    # blocking) keeps the 4-lane/ARM layouts byte-identical to before.
    kt_blk = 4 * _lanes() if _lanes() >= 16 and not bfmmla else seq_k_pad
    if bfmmla:
        mode = 2
        kt = _scratch("kt2", (batch, num_kv_heads, head_dim // 4, seq_k_pad // 2, 8),
                      k.dtype)
        vt = _scratch("vt2", (batch, num_kv_heads, head_dim // 2, seq_k_pad // 4, 8),
                      v.dtype)
    elif k.dtype == torch.bfloat16 and _bf16_dot():
        mode = 1
        kt = _scratch("kt1", (batch, num_kv_heads, head_dim // 2,
                              seq_k_pad // kt_blk, kt_blk, 2), k.dtype)
        vt = (_scratch("vt1", (batch, num_kv_heads, seq_k, head_dim), v.dtype)
              if vpack else None)
    else:
        mode = 0
        kt = _scratch("kt0", (batch, num_kv_heads, seq_k_pad // kt_blk, head_dim,
                              kt_blk), k.dtype)
        vt = (_scratch("vt0", (batch, num_kv_heads, seq_k, head_dim), v.dtype)
              if vpack else None)
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
        kt.stride(3) if mode == 0 else (kt.stride(3) if mode == 1 else kt.stride(2)),
        kt_blk,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        vt.stride(1) if vt is not None else 0,
        vt.stride(2) if vt is not None else 0,
        _DTYPE_CODES[k.dtype],
        mode,
        # The pack is one task per (batch, kv head) and each task does
        # `seq_k` strided row copies, so "pages" has to count tokens, not
        # 64-token blocks: at 8 heads the old `seq_k // 64` kept the pack
        # single-threaded up to seq_k 512, where it costs 23.7 us of a 71 us
        # call at S=128 (8 workers: 5.5 us).
        _workers(batch * num_kv_heads, seq_k // 8),
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
    ktb = kt.stride(1)
    if mode == 2:
        ktd, ktt = kt.stride(2), kt.stride(3)   # bfmmla: its own packing
    else:
        ktd = kt.stride(3)                      # dim stride within a block
        ktt = kt.stride(4) if mode == 1 else 1  # token stride inside a block
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
        kt_blk,
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
