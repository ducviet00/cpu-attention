"""Build the Mojo kernel into ``cpu_attention/_C.so``.

    uv run --extra build python scripts/build.py            # the extension
    uv run --extra build python scripts/build.py --tests    # + standalone test

The mojo compiler comes from the ``build`` extra (the PyPI ``mojo`` wheel), so
uv is the only tool needed. Everything the kernels do -- including Intel AMX,
which used to need a gcc-built C shim -- now comes out of this one build: the
tile instructions are emitted as inline assembly in ``_mojo/src/amx.mojo``
because LLVM's AMX *intrinsics* traffic in a type Mojo cannot name.  The shared library resolves the libgomp symbols at
load time (torch has libgomp loaded), but the standalone test binary links them,
so ``--tests`` needs libgomp -- the system copy, or torch's bundled
``libgomp.so.1`` -- and is skipped with a note when neither is there.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cpu_attention", "_mojo", "src")
TESTS = os.path.join(ROOT, "cpu_attention", "_mojo", "tests")
BUILD = os.path.join(ROOT, "cpu_attention", "_mojo", "build")
LIB = os.path.join(ROOT, "cpu_attention", "_C.so")


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
