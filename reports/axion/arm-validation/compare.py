import glob
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "tests")]
import bench
import arms as A


def build_arms(args):
    before = os.path.join(ROOT, "reports/axion/arm-validation/before")
    libs = {os.path.basename(p): p for p in glob.glob(before + "/_C*.so")}
    loaders = [(n, lambda n=n: A.current(n, "mojo"))
               for n in ("mojo_new", "mojo_ctrl")]
    for n in ("mojo_before", "mojo_before_ctrl"):
        def load(n=n):
            pkg = A._import(n, A.PKG, libs, "mojo", {})
            return A.Arm(n, pkg, "mojo", {}, "afbea06 rebuilt baseline")
        loaders.append((n, load))
    cpp = os.path.join(A.PKG, "_C_cpp.so")
    for isa in ("neon", "scalar"):
        loaders.append(("cpp_" + isa, lambda isa=isa: A.tagged(
            "cpp_" + isa, args.oracle_tag, "cpp", {"_C_cpp.so": cpp},
            {"CPU_ATTENTION_ISA": isa})))
    random.Random(20261028).shuffle(loaders)
    loaded = {n: f() for n, f in loaders}
    # The scalar reference is safe on poisoned/heterogeneous bf16 tails; the
    # NEON baseline is still selected per row whenever it is valid and faster.
    return [loaded[n] for n in ("mojo_new", "mojo_ctrl", "mojo_before",
                               "mojo_before_ctrl", "cpp_scalar", "cpp_neon")]


bench.build_arms = build_arms
raise SystemExit(bench.main())
