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
    _run(mojo, "build", os.path.join(SRC, "_C.mojo"), "--emit", "shared-lib", "-o", LIB)
    print(f"built {LIB}")
    if "--tests" not in sys.argv:
        return
    flags = _libgomp_flags()
    if flags is None:
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
