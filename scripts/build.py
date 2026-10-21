"""Build the Mojo kernel into ``cpu_attention/_C.so``.

    uv run --extra build python scripts/build.py            # the extension
    uv run --extra build python scripts/build.py --tests    # + standalone test

The mojo compiler comes from the ``build`` extra (the PyPI ``mojo`` wheel), so
uv is the only tool needed.  The shared library resolves the libgomp symbols at
load time (torch has libgomp loaded), but the standalone test binary links them,
so ``--tests`` needs libgomp -- the system copy, or torch's bundled
``libgomp.so.1`` -- and is skipped with a note when neither is there.
"""

import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cpu_attention", "_mojo", "src")
TESTS = os.path.join(ROOT, "cpu_attention", "_mojo", "tests")
BUILD = os.path.join(ROOT, "cpu_attention", "_mojo", "build")
LIB = os.path.join(ROOT, "cpu_attention", "_C.so")

# The AMX/AVX-512 C kernels (x86 only): a second extension module next to
# _C.so, loaded with ctypes. Kept out of the Mojo build because AMX needs
# tileloadd/tdpbf16ps on 16x16 tiles, which Mojo cannot express.
CSRC = os.path.join(ROOT, "cpu_attention", "_csrc")
CLIB = os.path.join(ROOT, "cpu_attention", "_csrc_kernels.so")
C_SOURCES = ["dense_amx.c"]
C_FLAGS = [
    "-O3", "-fPIC", "-shared", "-std=c11", "-fopenmp", "-fno-math-errno",
    "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512bf16",
    "-mamx-tile", "-mamx-bf16", "-mamx-int8",
]


def build_c() -> None:
    """gcc-build the C kernels; silently skipped off x86_64."""
    if not sys.platform.startswith("linux") or platform.machine() not in (
        "x86_64", "AMD64",
    ):
        print("skip the C kernels: linux x86_64 only")
        return
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        print(f"skip the C kernels: no {cc}")
        return
    srcs = [os.path.join(CSRC, s) for s in C_SOURCES]
    if not all(os.path.exists(s) for s in srcs):
        print("skip the C kernels: sources missing")
        return
    _run(cc, *C_FLAGS, *srcs, "-o", CLIB)
    print(f"built {CLIB}")


def _run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def _libgomp_flags() -> list[str] | None:
    """Linker flags for libgomp, or None when only torch's runtime copy exists."""
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(os.path.join(lib, "libgomp.so.1")):
            return ["-Xlinker", "-l:libgomp.so.1", "-Xlinker", f"-L{lib}"]
    except ImportError:
        pass
    dirs = ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64")
    if any(os.path.exists(os.path.join(d, "libgomp.so")) for d in dirs):
        return ["-Xlinker", "-lgomp"]
    return None


def main() -> None:
    mojo = shutil.which("mojo")
    if mojo is None:
        raise SystemExit(
            "mojo not found: run `uv sync --extra build` first, or "
            "`uv run --extra build python scripts/build.py`"
        )
    # Link libgomp into the extension instead of leaving `omp_*` undefined for
    # load time: on x86 torch happens to bring a global copy, but on aarch64 its
    # bundled libgomp.so.1 is RTLD_LOCAL, so `import` fails with "undefined
    # symbol: omp_get_thread_num". Same soname either way, so the loader reuses
    # torch's copy when it is already in the process.
    flags = _libgomp_flags()
    if flags is None:
        print("warning: no libgomp found; the extension may not import")
        flags = []
    _run(
        mojo,
        "build",
        os.path.join(SRC, "_C.mojo"),
        "--emit",
        "shared-lib",
        "-o",
        LIB,
        *flags,
    )
    print(f"built {LIB}")
    build_c()
    if "--tests" not in sys.argv:
        return
    if not flags:
        print("skip the standalone test: no libgomp to link against")
        return
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, "test_attention")
    _run(
        mojo,
        "build",
        "-I",
        SRC,
        os.path.join(TESTS, "test_attention.mojo"),
        "-o",
        out,
        *flags,
    )
    _run(out)


if __name__ == "__main__":
    main()
