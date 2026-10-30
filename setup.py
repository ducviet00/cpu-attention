"""Builds the Mojo kernel into the package, so `pip install` / `uv pip install` of
this repository (git URL or path) compiles it: one `_C_<tier>.so` per ISA tier.
Set CPU_ATTENTION_SKIP_BUILD=1 to package already-built objects as they are."""

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


class BuildKernel(build_py):
    def run(self):
        if not os.environ.get("CPU_ATTENTION_SKIP_BUILD"):
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "build.py")
            subprocess.run([sys.executable, script], check=True)
        super().run()


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True  # a platform wheel: the package carries shared objects


setup(cmdclass={"build_py": BuildKernel}, distclass=NativeDistribution)
