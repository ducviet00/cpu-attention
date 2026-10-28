"""Load and allocate independent arms in shuffled order, with a safe bf16 peer."""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "tests")]
import bench
import arms as A


def build(args):
    cpp = os.path.join(A.PKG, "_C_cpp.so")
    loaders = [lambda: A.current("mojo_new", "mojo"),
               lambda: A.snapshot("mojo_previous", args.previous),
               lambda: A.snapshot("mojo_ctrl", args.previous)]
    for isa in ("scalar", "neon"):
        loaders.append(lambda isa=isa: A.tagged(
            "cpp_" + isa, args.oracle_tag, "cpp", {"_C_cpp.so": cpp},
            {"CPU_ATTENTION_ISA": isa}))
    rng = random.Random(20261029)
    rng.shuffle(loaders)
    return [f() for f in loaders]


paged = bench.paged_row


def row(r, arms, vllm, args, page):
    args.ref_arm = "cpp_scalar"
    return paged(r, arms, vllm, args, page)


bench.build_arms = build
bench.paged_row = row
raise SystemExit(bench.main())
