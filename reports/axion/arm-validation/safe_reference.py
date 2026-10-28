import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "tests")]
import bench

build = bench.build_arms


def scalar_reference(args):
    loaded = build(args)
    # The benchmark otherwise chooses cpp_neon as the bf16 peer even on rows
    # where that arm returns NaN. The safe scalar arm is still eligible as a
    # baseline only when its own result passes every numerical check.
    return sorted(loaded, key=lambda a: a.name != "cpp_scalar")


bench.build_arms = scalar_reference
raise SystemExit(bench.main())
