"""Independently loaded benchmark arms.

Every arm is its own copy of the ``cpu_attention`` package -- Python glue *and*
shared objects -- in a private directory, imported under a unique module name.
So two arms never share a plan cache, a workspace, a ``dlopen`` handle or
module-level state, and a duplicate arm is a real control (docs/CPP.md §7.2:
a second closure over one module is not).

* ``current(name)``: this checkout's glue and builds (the build under test).
* ``tagged(name, tag, libs)``: the glue from a git tag with the given builds,
  e.g. the Mojo oracle at ``mojo-oracle-v1`` with its saved ``_C.so``.

``Arm.env`` is applied around each timed sample, outside the timer: the C++
backend reads ``CPU_ATTENTION_ISA`` on every call, which is how one process
times each C++ ISA path.
"""

import glob
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "cpu_attention")
_TMP = tempfile.mkdtemp(prefix="ca_arms_")
_ENV_KEYS = ("CPU_ATTENTION_ISA", "CPU_ATTENTION_AMX", "CPU_ATTENTION_AMX_FP32")


def clear_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


class Arm:
    def __init__(self, name, pkg, backend, env, origin):
        self.name, self.pkg, self.backend = name, pkg, backend
        self.env, self.origin = dict(env), origin

    def enter(self):
        clear_env()
        os.environ.update(self.env)

    def module(self):
        return self.pkg._kernel.module()

    def describe(self):
        mod = self.module()
        info = {"backend": self.backend, "env": self.env, "origin": self.origin,
                 "lib": getattr(mod, "__file__", None)}
        root = os.path.dirname(self.pkg.__file__)
        info["sha256"] = {}
        for path in sorted(glob.glob(os.path.join(root, "*.py")) +
                           glob.glob(os.path.join(root, "_C*.so"))):
            with open(path, "rb") as f:
                info["sha256"][os.path.basename(path)] = hashlib.sha256(
                    f.read()).hexdigest()
        self.enter()
        for probe in ("isa", "tier", "backend"):
            fn = getattr(mod, probe, None)
            if callable(fn):
                try:
                    info[probe] = fn()
                except Exception as e:  # noqa: BLE001
                    info[probe] = f"error: {e}"
        return info


def _import(name, src_py_dir, libs, backend, env):
    dst = os.path.join(_TMP, name, "ca_" + name)
    os.makedirs(dst)
    for f in glob.glob(os.path.join(src_py_dir, "*.py")):
        shutil.copy(f, dst)
    for base, src in libs.items():
        shutil.copy(src, os.path.join(dst, base))
    saved = {k: os.environ.get(k) for k in _ENV_KEYS + ("CPU_ATTENTION_BACKEND",
                                                        "CPU_ATTENTION_LIB")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        os.environ["CPU_ATTENTION_BACKEND"] = backend
        modname = "ca_" + name
        spec = importlib.util.spec_from_file_location(
            modname, os.path.join(dst, "__init__.py"),
            submodule_search_locations=[dst])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[modname] = pkg
        spec.loader.exec_module(pkg)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    pkg._kernel.set_backend(backend)
    if pkg._kernel.module() is None:
        raise RuntimeError(f"arm {name}: {backend} build did not load from {dst}")
    return pkg


def current(name, backend, env=None):
    libs = {os.path.basename(p): p for p in glob.glob(os.path.join(PKG, "_C*.so"))}
    pkg = _import(name, PKG, libs, backend, env or {})
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain",
                            "cpu_attention"], capture_output=True,
                           text=True).stdout.strip()
    return Arm(name, pkg, backend, env or {}, f"{rev}{'+dirty' if dirty else ''}")


def snapshot(name, directory, backend="mojo", env=None):
    directory = os.path.abspath(directory)
    libs = {os.path.basename(p): p for p in glob.glob(
        os.path.join(directory, "_C*.so"))}
    pkg = _import(name, directory, libs, backend, env or {})
    return Arm(name, pkg, backend, env or {}, directory)


def tagged(name, tag, backend, libs, env=None):
    """Glue from git ``tag`` with the shared objects in ``libs``."""
    src = os.path.join(_TMP, f"src_{name}")
    os.makedirs(src)
    archive = subprocess.run(["git", "-C", ROOT, "archive", tag, "cpu_attention"],
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", src, "--strip-components=1",
                    "--wildcards", "cpu_attention/*.py"], input=archive,
                   check=True)
    pkg = _import(name, src, libs, backend, env or {})
    return Arm(name, pkg, backend, env or {}, tag)


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpp_isas(dtype):
    """The C++ ISA paths applicable on this CPU for ``dtype`` ('bf16'|'fp32')."""
    flags = cpu_flags()
    if "avx512f" in flags:
        isas = []
        if "amx_tile" in flags and "amx_bf16" in flags:
            isas.append("amx")
        if dtype == "fp32" or "avx512_bf16" in flags:
            isas.append("vec")
        return isas + ["scalar"]
    if "asimd" in flags:
        return (["neon"] if dtype == "fp32" or "bf16" in flags else []) + ["scalar"]
    return ["scalar"]
