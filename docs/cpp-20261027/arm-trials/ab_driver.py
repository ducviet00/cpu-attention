import importlib.util
import json
import os
import sys
import threading

root = os.environ["ARM_ROOT"]
sys.path[:0] = [root, os.path.join(root, "tests")]
import bench_cpp as bench
from cpu_attention import _kernel

states = {}
for name, filename in (("new", "_C_cpp.so"), ("old", "_C_cpp_old.so")):
    spec = importlib.util.spec_from_file_location(
        "cpu_attention._C", os.path.join(root, "cpu_attention", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states[name] = ({"cpp": module}, {}, threading.local(), threading.local())

original = _kernel.set_backend


def select(name):
    built, bound, plans, paged_plans = states["old" if name == "mojo" else "new"]
    _kernel._BUILTS = built
    _kernel._BOUND = bound
    _kernel._CPP_PLANS = plans
    _kernel._PAGED_PLANS = paged_plans
    original("cpp")


_kernel.set_backend = select
bench.main()
if "--json" in sys.argv:
    path = sys.argv[sys.argv.index("--json") + 1]
    with open(path) as f:
        rows = json.load(f)
    for row in rows:
        row["comparison"] = "cpp=new, mojo=old C++; ctrl=duplicate new"
        row["new/old"] = row["cpp/mojo"]
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
