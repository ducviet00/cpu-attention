import importlib.util
import json
import os
import sys
import threading

root = os.getcwd()
sys.path[:0] = [root, os.path.join(root, "tests")]
import torch
import bench_cpp as bench
import cpu_attention as attn
from cpu_attention import _kernel

torch.set_num_threads(8)
torch.manual_seed(7)
states = []
for filename in ("_C_cpp_old.so", "_C_cpp.so"):
    spec = importlib.util.spec_from_file_location(
        "cpu_attention._C", os.path.join(root, "cpu_attention", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states.append(({"cpp": module}, {}, threading.local(), threading.local()))


def select(i):
    built, bound, plans, paged = states[i]
    _kernel._BUILTS = built
    _kernel._BOUND = bound
    _kernel._CPP_PLANS = plans
    _kernel._PAGED_PLANS = paged
    _kernel.set_backend("cpp")


rows = []
for dtype in (torch.bfloat16, torch.float32):
    for seq, heads, dim in bench.DENSE:
        q, k, v = [torch.randn(1, seq, heads, dim, dtype=dtype) for _ in range(3)]
        out = []
        for i in range(2):
            select(i)
            out.append(attn.flash_attn_func(q, k, v, causal=True))
        rows.append(dict(kind="dense", dtype=str(dtype), seq=seq, heads=heads,
                         dim=dim, equal=torch.equal(*out),
                         max_diff=(out[0].float()-out[1].float()).abs().max().item()))
    for batch, ctx in bench.PAGED:
        q, kn, vn, ka, va, kc, vc, slots, bt, lens = bench.paged_case(
            batch, ctx, 32, dtype)
        for op in ("self", "cross"):
            out = []
            for i in range(2):
                select(i)
                bench.stage(ka, va, kc, vc, 32, batch, ctx, "cpp", None, "cpp")
                out.append(attn.self_attention(q, kn, vn, kc, vc, slots, bt,
                                               lens, 0.125) if op == "self" else
                           attn.cross_attention(q, kc, vc, bt, lens, 0.125))
            rows.append(dict(kind="paged", dtype=str(dtype), batch=batch,
                             ctx=ctx, op=op, equal=torch.equal(*out),
                             max_diff=(out[0].float()-out[1].float()).abs().max().item()))
with open("compare-outputs.json", "w") as f:
    json.dump(rows, f, indent=1)
print(sum(r["equal"] for r in rows), "/", len(rows), "bit-identical")
assert all(r["equal"] for r in rows)
