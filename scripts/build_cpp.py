"""Build the C++ kernel into ``cpu_attention/_C_cpp.so``.

    uv run --extra build python scripts/build_cpp.py          # build
    uv run --extra build python scripts/build_cpp.py --test   # + parity smoke

One artifact per architecture, compiled for the baseline ISA with runtime
dispatch (CPUID on x86, HWCAP on aarch64). No ``-march=native``: the committed
flags are the ones the scorecard was measured with, so a build on one box means
the same instructions on another.

Per-ISA flags, as required by the intrinsics each file uses:
  vec.cpp    AVX-512 F/BW/VL/DQ + BF16 + VNNI + FMA + AVX2
  amx.cpp    the above plus -mamx-tile -mamx-bf16 -mamx-int8
  neon.cpp   -march=armv8.2-a+bf16+i8mm
Everything else is compiled baseline, so the fallback paths stay portable.
"""

import os
import platform
import subprocess
import sys
import sysconfig
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cpu_attention", "csrc")
LIB = os.path.join(ROOT, "cpu_attention", "_C_cpp.so")

X86_VEC = [
    "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mavx512bf16",
    "-mavx512vnni", "-mfma", "-mavx2",
]
X86_AMX = X86_VEC + ["-mamx-tile", "-mamx-bf16", "-mamx-int8"]
ARM = ["-march=armv8.2-a+bf16+i8mm"]

BASE = ["-O3", "-fPIC", "-std=c++20", "-fopenmp", "-Wall", "-Wextra",
        "-Wno-unused-parameter"]


def arch() -> str:
    m = platform.machine()
    if m in ("x86_64", "AMD64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    raise SystemExit(f"build_cpp: unsupported machine {m}")


def main() -> None:
    t0 = time.time()
    a = arch()
    cxx = os.environ.get("CXX", "g++")
    objs = []
    build = os.path.join(ROOT, "cpu_attention", "csrc", "build")
    os.makedirs(build, exist_ok=True)

    # (source, extra flags)
    pyinc = sysconfig.get_paths()["include"]
    units = [("cpu_attn.cpp", []), ("binding.cpp", ["-I", pyinc]),
             ("cpu_isa.cpp", []), ("scalar.cpp", [])]
    if a == "x86_64":
        units += [("vec.cpp", X86_VEC), ("amx.cpp", X86_AMX)]
    else:
        units += [("neon.cpp", ARM)]

    for name, extra in units:
        src = os.path.join(SRC, name)
        if not os.path.exists(src):
            raise SystemExit(f"build_cpp: missing {src}")
        obj = os.path.join(build, name.replace(".cpp", ".o"))
        cmd = [cxx, *BASE, *extra, "-I", SRC, "-c", src, "-o", obj]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)
        objs.append(obj)

    # Replacing the inode keeps an already-loaded A/B arm valid during a build.
    # Linking straight into LIB can truncate executable pages in another process.
    pending = os.path.join(build, f"_C_cpp.{os.getpid()}.so")
    cmd = [cxx, "-shared", "-o", pending, *objs, "-fopenmp"]
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        os.replace(pending, LIB)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)
    print(f"built {LIB} in {time.time() - t0:.1f}s")

    if "--test" in sys.argv:
        isa_test = os.path.join(build, "test_cpu_isa")
        subprocess.run([cxx, *BASE, "-I", SRC,
                        os.path.join(ROOT, "tests", "test_cpu_isa.cpp"),
                        os.path.join(build, "cpu_isa.o"), "-o", isa_test],
                       check=True)
        subprocess.run([isa_test], check=True)
        # Parity smoke: the dense suite covers the plan entry, GQA, masks and
        # both dtypes; the flash-shaped API covers the paged path.
        env = dict(os.environ, CPU_ATTENTION_BACKEND="cpp", PYTHONPATH=ROOT)
        for script in ("tests/test_dense.py", "tests/test_flash_api.py"):
            subprocess.run([sys.executable, os.path.join(ROOT, script)],
                           check=True, env=env)


if __name__ == "__main__":
    main()
