"""Loads the Mojo kernel and builds its launch plans.

`scripts/build.py` builds one shared object per ISA tier; the best tier this CPU
supports is loaded (`CPU_ATTENTION_ISA` forces a lower one, `CPU_ATTENTION_LIB`
another build). A plan is the int64 geometry buffer a kernel entry reads; the
callers build one per geometry, after validating it, and cache it.
"""

import importlib.util
import os
import platform
import threading

import torch

DTYPES = {torch.float32: 0, torch.bfloat16: 1}
HEAD_DIMS = (32, 64, 128)
_DIR = os.path.dirname(os.path.abspath(__file__))

# Tiers, best first, with the CPU features each was compiled for. /proc/cpuinfo
# lists a feature only when the OS enabled its register state, so a listed
# AVX-512 flag is also an OS check; AMX also needs a permission syscall, which
# the kernel module makes itself.
_TIERS = {
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
_ISA = {"amx": "x86-avx512bf16", "vec": "x86-avx512bf16", "avx2": "x86-avx2",
        "neon": "aarch64-bf16"}  # CPU_ATTENTION_ISA -> tier ("scalar": the last)


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _library():
    if os.environ.get("CPU_ATTENTION_LIB"):
        return os.environ["CPU_ATTENTION_LIB"]
    tiers = _TIERS["aarch64" if platform.machine() in ("aarch64", "arm64") else "x86_64"]
    flags = _cpu_flags()
    ok = [name for name, need in tiers if need <= flags]
    forced = os.environ.get("CPU_ATTENTION_ISA")
    if forced:
        want = tiers[-1][0] if forced == "scalar" else _ISA.get(forced)
        if want not in ok:
            raise RuntimeError(f"cpu_attention: CPU_ATTENTION_ISA={forced} is not "
                               "supported on this CPU")
        ok = [want]
    paths = [os.path.join(_DIR, f"_C_{tier}.so") for tier in ok]
    return next((p for p in paths if os.path.exists(p)), os.path.join(_DIR, "_C.so"))


_MODULE = None


def module():
    """The compiled kernel module (RuntimeError when it is not built)."""
    global _MODULE
    if _MODULE is None:
        path = _library()
        if not os.path.exists(path):
            raise RuntimeError("cpu_attention: the kernel is not built; run `uv run "
                               "--extra build python scripts/build.py`")
        spec = importlib.util.spec_from_file_location("cpu_attention._C", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE = mod
    return _MODULE


def _cached(fn):
    value = []
    return lambda: value[0] if value else value.append(fn()) or value[0]


@_cached
def physical_cores():
    """Physical cores visible to this process, from the sysfs topology."""
    cores = set()
    try:
        for cpu in range(os.cpu_count() or 1):
            base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
            with open(f"{base}/core_id") as f, open(f"{base}/physical_package_id") as g:
                cores.add((g.read().strip(), f.read().strip()))
    except OSError:
        cores = set()
    return len(cores) or (os.cpu_count() or 1)


@_cached
def llc_bytes():
    """Last-level cache size, from sysfs (32 MiB if unknown)."""
    for idx in (3, 2, 1, 0):
        try:
            with open(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size") as f:
                text = f.read().strip()
        except OSError:
            continue
        unit = {"K": 1 << 10, "M": 1 << 20}.get(text[-1], 1)
        return int(text.rstrip("KM")) * unit
    return 32 << 20


def workers(tasks, pages=1):
    """Threads for one call. The kernel forks on torch's OpenMP pool (~2.4 us),
    which beats running inline from about 32 KV positions per call (batch 1, ctx
    64: 24.5 us inline vs 31.0 pooled; ctx 128: the pool 1.3x faster). Capped at
    physical cores: SMT threads only oversubscribe (8 on a 4-core Xeon: 1.5x
    slower at ctx 1024)."""
    n = min(torch.get_num_threads(), physical_cores(), tasks)
    return 1 if n > 1 and tasks * pages <= 32 else max(n, 1)


# ---------------------------------------------------------------------------
# paged decode / reshape_and_cache
# ---------------------------------------------------------------------------
def paged_plan(q, k, k_cache, npages, store, attend):
    """The plan of a `kv_attention` call, for q `[seqs, ..., heads, dim]` and
    the new tokens' k `[tokens, ..., heads, dim]` (heads and dims contiguous; a
    token stride is fine: q/k/v may be chunks of one fused projection)."""
    rows = q if q is not None else k
    nseq, heads, hd = rows.shape[0], rows.shape[-2], rows.shape[-1]
    buf = torch.tensor([
        nseq, heads, hd, k_cache.shape[2], npages, heads * hd,
        DTYPES[rows.dtype], workers(nseq * heads, max(npages, 1)), int(store),
        int(attend), q.stride(0) if q is not None else 0,
        k.stride(0) if k is not None else 0], dtype=torch.int64)
    return buf, buf.data_ptr()


def run_paged(plan, q, k, v, k_cache, v_cache, out, slots, page_table, lens, scale):
    module().kv_attention(
        plan[1], q.data_ptr() if q is not None else 0,
        k.data_ptr() if k is not None else 0, v.data_ptr() if v is not None else 0,
        k_cache.data_ptr(), v_cache.data_ptr(), out.data_ptr() if out is not None else 0,
        slots.data_ptr() if slots is not None else 0,
        page_table.data_ptr() if page_table is not None else 0,
        lens.data_ptr() if lens is not None else 0, scale)


# ---------------------------------------------------------------------------
# flash_attn_func
# ---------------------------------------------------------------------------
_AMX_ENV = (os.environ.get("CPU_ATTENTION_AMX", "1") != "0"
            and os.environ.get("CPU_ATTENTION_ISA", "amx") == "amx")
# fp32 runs on AMX as bf16 hi/lo products (error 3-8e-5 against the 2e-4 gate);
# CPU_ATTENTION_AMX_FP32=0 keeps fp32 on the exact vector policy.
_AMX_FP32 = os.environ.get("CPU_ATTENTION_AMX_FP32", "1") != "0"


@_cached
def amx_available():
    """The CPU reports AMX and the OS grants tile data (`arch_prctl`, asked by
    the module: the first tile instruction is SIGILL without it)."""
    return {"amx_tile", "amx_bf16"} <= _cpu_flags() and bool(module().amx_available())


def _vector_beats_amx(dt, hd, seq_q, seq_k, scale):
    """Few query rows leave AMX's 16-row tiles mostly empty (and fp32 pays three
    products per phase), so the vector policy is faster there. Crossovers on the
    Xeon 8581C, 4 threads (vector / AMX time): fp32 hd=64 up to 65 rows 0.39-0.94,
    fp32 hd=128 up to 17 rows 0.80-0.97 (or a 33-key sequence: 0.59), bf16 1-2
    rows 0.76-0.89 (hd=128 only while seq_k <= 256: AMX packs a long K/V faster)."""
    if dt == 0:
        # A logit std above ~6 (scale * sqrt(hd) for unit inputs) peaks the
        # probabilities enough for the hi/lo split to reach 2.03e-4 against the
        # 2e-4 gate (scale 1.7 at hd=64): exact fp32 on the vector policy.
        if scale * hd ** 0.5 > 6.0:
            return True
        return seq_q <= (65 if hd <= 64 else 24) or seq_k <= 64
    return seq_q <= 2 and (hd <= 64 or seq_k <= 256)


def flash_plan(q, k, v, scale, causal):
    """The plan of a `flash_forward` call: (geometry buffer, its address,
    workspace bytes). q/k/v are `[batch, seq, heads, dim]` with dims contiguous."""
    batch, seq_q, heads, hd = q.shape
    seq_k = k.shape[1]
    dt = DTYPES[q.dtype]
    # Policy: 0 vector, 1 AMX (fp32 as bf16 hi/lo), 2 aarch64 bfmmla / lane FMA.
    isa = (2 if module().tier() == "aarch64-bf16" else
           1 if _AMX_ENV and amx_available() and (dt == 1 or _AMX_FP32) else 0)
    if isa == 1 and _vector_beats_amx(dt, hd, seq_q, seq_k, scale):
        isa = 0
    if isa == 1:
        rq = 64  # two 32-row AMX register tiles: K/V tiles load once per 64 rows
    elif isa == 2 and dt == 0 and hd == 128 and causal and seq_q == seq_k >= 2048:
        rq = 128  # Axion: amortize the K/V scan on long fp32 D=128 (6-9%; short rows lost 9.5%)
    else:
        rq = max(16, min(128, 16384 // (hd * 4)))  # a ~16 KB accumulator stays in L1
    # At least ~8 work items per worker: causal blocks differ in cost, and at 4
    # per worker the tail cost 1024:8:32 10-11% at 16 threads (Axion). Halve rq,
    # down to one register-tile pair.
    rq = min(rq, max(seq_q, 1))
    floor, cap = (64 if isa == 1 else 16), workers(1 << 30)
    while rq // 2 >= floor and batch * heads * -(-seq_q // rq) < 8 * cap:
        rq //= 2
    n = workers(batch * heads * -(-seq_q // rq), 8)
    slab, aux = module().flash_sizes(seq_k, rq, hd, dt, isa)
    llc = llc_bytes()
    # Fused (a thread packs a head and attends it, no barrier) while the workers'
    # slabs fit 0.6 of the LLC and the heads divide evenly.
    fuse = batch * heads >= n and batch * heads % n == 0 and slab * n <= llc * 3 // 5
    # fp32 V is read in place while it all fits half the LLC: a copy only buys a
    # unit stride.
    v_direct = dt == 0 and not isa and batch * heads * seq_k * hd * 4 <= llc // 2
    qs, ks, vs = q.stride(), k.stride(), v.stride()
    buf = torch.tensor([
        batch, seq_q, seq_k, heads, hd, qs[0], qs[1], qs[2], ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2], seq_q * heads * hd, heads * hd, hd, dt, int(causal),
        n, int(fuse), rq, 0, slab, aux, int(v_direct), isa], dtype=torch.int64)
    return buf, buf.data_ptr(), (n if fuse else batch * heads) * slab + n * aux + 64


_WS = threading.local()


def workspace(nbytes):
    """This thread's flash workspace (64-byte aligned), grown on demand and never
    shrunk: a large `torch.empty` is an mmap whose pages fault in again after a
    free (16.8% of an S=1024 H=16 D=128 fp32 call was the kernel zeroing them)."""
    ws = getattr(_WS, "t", None)
    if ws is None or ws.numel() < nbytes + 64:
        _WS.t = ws = torch.empty(nbytes + 4096, dtype=torch.uint8)
    return (ws.data_ptr() + 63) & ~63
