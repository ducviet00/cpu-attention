"""Loads the Mojo kernel and its runtime config.

`scripts/build.py` builds one shared object per ISA tier; the best tier this CPU
supports is loaded (`CPU_ATTENTION_ISA` forces a lower one, `CPU_ATTENTION_LIB`
another build). The kernel's entries read torch tensors in place
(`_mojo/src/torch_tensor.mojo`), at TensorImpl field offsets that `runtime()`
finds once by matching what torch reports and verifies on probe tensors.
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


_AMX_ENV = (os.environ.get("CPU_ATTENTION_AMX", "1") != "0"
            and os.environ.get("CPU_ATTENTION_ISA", "amx") == "amx")
# fp32 runs on AMX as bf16 hi/lo products (error 3-8e-5 against the 2e-4 gate);
# CPU_ATTENTION_AMX_FP32=0 keeps fp32 on the exact vector policy.
_AMX_FP32 = os.environ.get("CPU_ATTENTION_AMX_FP32", "1") != "0"
_DTYPES = (torch.float32, torch.bfloat16, torch.int32, torch.int64, torch.float16)


def _calibrate():
    """Byte offsets of the fields the kernel reads, found by matching values torch
    reports: the TensorImpl in the Python object (`tensor._cdata`), its storage and
    the storage's data pointer, storage offset, dim count, inline sizes and
    strides, and the dtype index. Returns (offsets, dtype codes) or None."""
    import ctypes

    def words(addr, n):
        return [ctypes.c_int64.from_address(addr + 8 * i).value for i in range(n)]

    def u16(addr):
        return ctypes.c_uint16.from_address(addr).value

    t = torch.empty(3, 5, 7)
    w = words(t._cdata, 48)
    storage = t.untyped_storage()
    impl = [8 * i for i, x in enumerate(words(id(t), 8)) if x == t._cdata]
    st = [8 * i for i, x in enumerate(w) if x == storage._cdata]
    data = [8 * i for i, x in enumerate(words(storage._cdata, 16)) if x == storage.data_ptr()]
    sizes = [8 * i for i in range(45) if w[i:i + 3] == [3, 5, 7]]
    strides = [8 * i for i in range(45) if w[i:i + 3] == [35, 7, 1]]
    view = words(torch.empty(100)[13:].view(29, 3)._cdata, 48)
    offset = [8 * i for i, x in enumerate(view) if x == 13 and w[i] == 0]
    w2, w1 = words(torch.empty(2, 2)._cdata, 48), words(torch.empty(4)._cdata, 48)
    dim = [8 * i for i in range(48) if (w[i], w2[i], w1[i]) == (3, 2, 1)]
    pairs = [(torch.empty(3, 5, dtype=d), torch.empty(7, dtype=d)) for d in _DTYPES]
    dtype = [b for b in range(0, 384, 2)
             if all(u16(a._cdata + b) == u16(c._cdata + b) for a, c in pairs)
             and len({u16(a._cdata + b) for a, _ in pairs}) == len(_DTYPES)]
    found = (impl, st, data, offset, dim, sizes, strides, dtype)
    if not all(found):
        return None
    offsets = [x[0] for x in found]
    return offsets, {d: u16(a._cdata + offsets[7]) for d, (a, _) in zip(_DTYPES, pairs)}


_RUNTIME = []


def runtime():
    """The address of the kernel's runtime config (slots in `torch_tensor.mojo`).
    Built once: calibrates the tensor field offsets and verifies them, and the
    module's output allocation, on probe tensors (RuntimeError on any mismatch:
    a torch whose tensor layout this build cannot read)."""
    if _RUNTIME:
        return _RUNTIME[1]
    mod = module()
    found = None
    try:
        found = _calibrate()
    except Exception:  # noqa: BLE001
        pass
    if found is None:
        raise RuntimeError(f"cpu_attention: cannot read torch {torch.__version__}'s tensor layout")
    offsets, codes = found
    amx = 0
    if _AMX_ENV and {"amx_tile", "amx_bf16"} <= _cpu_flags() and mod.amx_available():
        amx = 2 if _AMX_FP32 else 1
    cfg = torch.tensor(offsets + [
        id(torch.Tensor), id(torch.nn.Parameter), id(None), id(True), id(False),
        codes[torch.float32], codes[torch.bfloat16], codes[torch.int32], codes[torch.int64],
        physical_cores(), llc_bytes(), amx], dtype=torch.int64)
    addr = cfg.data_ptr()
    base = torch.randn(64, 3 * 16 * 8)
    probes = [torch.randn(5), torch.randn(4, 6), torch.randn(2, 3, 4).bfloat16(),
              torch.randn(9, 7)[2:].t(), torch.arange(50, dtype=torch.int32)[7:],
              torch.arange(9)[1::2], base[:, 16 * 8:32 * 8].view(64, 16, 8),
              torch.nn.Parameter(torch.randn(6, 2)), torch.empty(3, 16, 32, 64, dtype=torch.bfloat16)[1:]]
    for t in probes:
        got = mod.probe(addr, t)
        n = t.dim()
        want = [True, t.untyped_storage().data_ptr(), t.storage_offset(), n, *t.shape,
                *t.stride(), codes[t.dtype]]
        new = got[len(want):]
        if got[:len(want)] != want or (t.dtype in (torch.float32, torch.bfloat16) and not (
                new and new[0].shape == t.shape and new[0].dtype == t.dtype
                and new[0].is_contiguous())):
            raise RuntimeError(f"cpu_attention: cannot read torch {torch.__version__}'s tensor layout")
    _RUNTIME[:] = [cfg, addr]
    return addr


_WS = threading.local()


def workspace(nbytes=0):
    """This thread's flash workspace as (address, bytes), grown to `nbytes` and
    never shrunk: a large `torch.empty` is an mmap whose pages fault in again after
    a free (16.8% of an S=1024 H=16 D=128 fp32 call was the kernel zeroing them)."""
    ws = getattr(_WS, "t", None)
    if ws is None or ws.numel() < nbytes:
        _WS.t = ws = torch.empty(max(nbytes, 1 << 20), dtype=torch.uint8)
    return ws.data_ptr(), ws.numel()
