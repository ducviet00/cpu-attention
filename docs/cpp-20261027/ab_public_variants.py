"""Use the public four-arm harness for saved old/new C++ libraries."""
import importlib.util
import json
import os
import sys
import threading

root = os.environ["ATTN_ROOT"]
sys.path[:0] = [root, os.path.join(root, "tests")]
import bench_cpp as bench
from cpu_attention import _kernel as K

states = {}
for name, path in (("old", os.environ["OLD_LIB"]),
                   ("new", os.environ["NEW_LIB"])):
    spec = importlib.util.spec_from_file_location("cpu_attention._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    states[name] = ({"cpp": mod}, {}, threading.local(), threading.local())

select_backend = K.set_backend


def select(name):
    K._BUILTS, K._BOUND, K._CPP_PLANS, K._PAGED_PLANS = states[
        "old" if name == "mojo" else "new"]
    select_backend("cpp")


K.set_backend = select
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
