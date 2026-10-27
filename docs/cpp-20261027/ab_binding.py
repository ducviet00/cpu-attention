"""Measure checked FASTCALL against tuple parsing, with identical kernels."""
import importlib.util
import json
import sys
import time

import torch

sys.path.insert(0, "/home/ducviet00/cpu-attention")
sys.path.insert(0, "/home/ducviet00/cpu-attention/audit")
import audit_common as C
import cpu_attention as attn
from cpu_attention import _kernel as K


def load(path):
    spec = importlib.util.spec_from_file_location("variant._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


torch.set_num_threads(8)
torch.manual_seed(0)
mods = {"old": load("/tmp/opencode/cpp-rqt10-20261027.so"),
        "new": load("/tmp/opencode/cpp-fastcall-rqt10-20261027.so")}
K.set_backend("cpp")
rows = []
for dtype in (torch.float32, torch.bfloat16):
    q = torch.randn(1, 16, 64, dtype=dtype)
    cache = torch.zeros(8, 16, 32, 64, dtype=dtype)
    vc = torch.zeros_like(cache)
    k = torch.randn(256, 16, 64, dtype=dtype)
    v = torch.randn_like(k)
    bt = torch.arange(8, dtype=torch.int32).view(1, -1)
    lens = torch.tensor([256], dtype=torch.int32)
    attn.reshape_and_cache(k, v, cache, vc, torch.arange(256))
    arms = {}
    for name, mod in mods.items():
        K._BUILTS["cpp"] = mod
        K._BOUND.pop(("cpp", "paged_make_plan"), None)
        K._BOUND.pop(("cpp", "paged_forward"), None)
        K._PAGED_PLANS.p = {}
        attn.cross_attention(q, cache, vc, bt, lens, .125)
        plans = K._PAGED_PLANS.p
        def run(plans=plans):
            K._PAGED_PLANS.p = plans
            attn.cross_attention(q, cache, vc, bt, lens, .125)
        def block(run=run):
            for _ in range(40):
                run()
        arms[name] = block
    arms["ctrl"] = arms["new"]
    samples = {n: [x / 40 for x in xs] for n, xs in C.time_arms(arms, 101).items()}
    stats = {n: C.stats(xs) for n, xs in samples.items()}
    row = dict(dtype=str(dtype), samples_us=samples, stats=stats,
               ratio=stats["new"]["min"] / stats["old"]["min"],
               control=stats["ctrl"]["min"] / stats["new"]["min"])
    rows.append(row)
    print(dtype, row["ratio"], row["control"], stats, flush=True)
with open("/tmp/opencode/binding-ab-20261027.json", "w") as f:
    json.dump(rows, f, indent=2)
