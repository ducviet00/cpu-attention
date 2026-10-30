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
    """Linker flags for libgomp: torch's bundled copy, else the system one."""
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        flags: list[str] = []
        if os.path.exists(os.path.join(lib, "libgomp.so.1")):
            flags += ["-Xlinker", "-l:libgomp.so.1", "-Xlinker", f"-L{lib}"]
        elif any(
            os.path.exists(os.path.join(d, "libgomp.so"))
            for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64")
        ):
            flags += ["-Xlinker", "-lgomp"]
        else:
            return None
        return flags
    except ImportError:
        pass
    dirs = ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64")
    if any(os.path.exists(os.path.join(d, "libgomp.so")) for d in dirs):
        return ["-Xlinker", "-lgomp"]
    return None


# One shared object per ISA tier, each compiled for an explicit target (never
# the build host's CPU): `cpu_attention._kernel` loads the highest tier this CPU
# supports, and `CPU_ATTENTION_ISA` can force a lower one. The AVX-512-BF16 tier
# also carries the AMX kernels; they run only after the OS grants tile state.
TIERS = {
    "x86_64": [
        ("x86-avx512bf16", ["--target-cpu", "x86-64-v4",
                            "--target-features", "+avx512bf16,+avx512vnni"]),
        ("x86-avx2", ["--target-cpu", "x86-64-v3"]),
        ("x86-64", ["--target-cpu", "x86-64"]),
    ],
    "aarch64": [
        ("aarch64-bf16", ["--target-cpu", "neoverse-n1",
                          "--target-features", "+bf16,+i8mm"]),
        ("aarch64-neon", ["--target-cpu", "generic"]),
    ],
}


def _arch() -> str:
    import platform

    m = platform.machine()
    return "aarch64" if m in ("aarch64", "arm64") else "x86_64"


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
    test_flags = flags or []
    if flags is None:
        print("warning: no libgomp found; the extension may not import")
        flags = []
    tiers = TIERS[_arch()]
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--tier=")]
    if only:
        tiers = [t for t in tiers if t[0] in only]
    procs = []
    for name, target in tiers:
        out = os.path.join(ROOT, "cpu_attention", f"_C_{name}.so")
        cmd = [mojo, "build", os.path.join(SRC, "_C.mojo"), "--emit",
               "shared-lib", *target, "-o", out, *flags]
        print("+", " ".join(cmd))
        procs.append((name, out, subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT,
                                                  text=True)))
    failed = False
    for name, out, proc in procs:
        log = proc.communicate()[0]
        if proc.returncode != 0:
            failed = True
            print(f"tier {name} FAILED:\n" + "\n".join(
                ln for ln in log.splitlines() if "error" in ln)[:4000])
        else:
            print(f"built {out}")
    if failed:
        raise SystemExit(1)
    # `_C.so` stays the best tier for this host (what older callers load).
    best = os.path.join(ROOT, "cpu_attention", f"_C_{tiers[0][0]}.so")
    if not only or tiers[0][0] == TIERS[_arch()][0][0]:
        shutil.copyfile(best, LIB)
        print(f"{LIB} = {os.path.basename(best)}")
    if "--tests" not in sys.argv:
        return
    if not test_flags:
        print("skip the standalone test: no libgomp to link against")
        return
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, "test_attention")
    _run(
        mojo,
        "build",
        *TIERS[_arch()][0][1],
        "-I",
        SRC,
        os.path.join(TESTS, "test_attention.mojo"),
        "-o",
        out,
        *test_flags,
    )
    _run(out)


if __name__ == "__main__":
    main()
