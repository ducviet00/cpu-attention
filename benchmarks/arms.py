"""Independently loaded benchmark arms.

Every arm is its own copy of the ``cpu_attention`` package -- Python glue *and*
shared objects -- in a private directory, imported under a unique module name.
So two arms never share a plan cache, a workspace, a ``dlopen`` handle or
module-level state, and a duplicate arm is a real control (a second closure over
one module is not: it shares every cache the timing depends on).

* ``current(name)``: this checkout's glue and builds (the build under test).
* ``snapshot(name, directory)``: a saved package directory (a previous build).

``Arm.env`` (e.g. a forced ``CPU_ATTENTION_ISA``) is applied around each timed
sample, outside the timer.
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
    def __init__(self, name, pkg, env, origin):
        self.name, self.pkg = name, pkg
        self.env, self.origin = dict(env), origin

    def enter(self):
        clear_env()
        os.environ.update(self.env)

    def module(self):
        return self.pkg._kernel.module()

    def describe(self):
        mod = self.module()
        info = {"env": self.env, "origin": self.origin,
                 "lib": getattr(mod, "__file__", None)}
        root = os.path.dirname(self.pkg.__file__)
        info["sha256"] = {}
        for path in sorted(glob.glob(os.path.join(root, "*.py")) +
                           glob.glob(os.path.join(root, "_C*.so"))):
            with open(path, "rb") as f:
                info["sha256"][os.path.basename(path)] = hashlib.sha256(
                    f.read()).hexdigest()
        self.enter()
        for probe in ("isa", "tier"):
            fn = getattr(mod, probe, None)
            if callable(fn):
                try:
                    info[probe] = fn()
                except Exception as e:  # noqa: BLE001
                    info[probe] = f"error: {e}"
        return info


def _import(name, src_py_dir, libs, env):
    dst = os.path.join(_TMP, name, "ca_" + name)
    os.makedirs(dst)
    for f in glob.glob(os.path.join(src_py_dir, "*.py")):
        shutil.copy(f, dst)
    for base, src in libs.items():
        shutil.copy(src, os.path.join(dst, base))
    saved = {k: os.environ.get(k) for k in _ENV_KEYS + ("CPU_ATTENTION_LIB",)}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
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
    if pkg._kernel.module() is None:
        raise RuntimeError(f"arm {name}: the build did not load from {dst}")
    return pkg


def current(name, env=None):
    libs = {os.path.basename(p): p for p in glob.glob(os.path.join(PKG, "_C*.so"))}
    pkg = _import(name, PKG, libs, env or {})
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain",
                            "cpu_attention"], capture_output=True,
                           text=True).stdout.strip()
    return Arm(name, pkg, env or {}, f"{rev}{'+dirty' if dirty else ''}")


def snapshot(name, directory, env=None):
    directory = os.path.abspath(directory)
    libs = {os.path.basename(p): p for p in glob.glob(
        os.path.join(directory, "_C*.so"))}
    pkg = _import(name, directory, libs, env or {})
    return Arm(name, pkg, env or {}, directory)

