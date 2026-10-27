"""Compare saved C++ builds in one process, without changing the live library.

The control is a *third module*: a byte-identical copy of the old build, loaded
like the others. Reusing the new module's closure (the 2026-10-27 version) hides
module-level effects -- two identical builds read 0.93-1.00 "new/old" there,
because each module owns its workspace and where that lands matters. Modules are
loaded in a random order per run; `ctrl/old` is the noise floor of `new/old`.
"""
import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, "/home/ducviet00/cpu-attention/audit")
import audit_common as C


def load(path):
    spec = importlib.util.spec_from_file_location("variant._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--rows", nargs="+", default=["128:8:64", "512:8:64", "1024:8:64", "512:32:128", "1024:16:128", "2048:16:128", "4096:16:128"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    ctrl_path = os.path.join(tempfile.mkdtemp(), "ctrl_C_cpp.so")
    shutil.copyfile(args.old, ctrl_path)
    paths = {"old": args.old, "new": args.new, "ctrl": ctrl_path}
    order = list(paths)
    random.shuffle(order)
    mods = {name: load(paths[name]) for name in order}
    print("load order", order, flush=True)
    C.burn()
    rows = []
    for shape in args.rows:
        seq, heads, hd = map(int, shape.split(":"))
        dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        q = torch.randn(1, seq, heads, hd, dtype=dt)
        k, v = torch.randn_like(q), torch.randn_like(q)
        plans, outputs, arms = {}, {}, {}
        for name, mod in mods.items():
            p = plans[name] = torch.empty(24, dtype=torch.int64)
            mod.dense_make_plan(p.data_ptr(), 1, seq, seq, heads, heads, hd,
                                int(dt == torch.bfloat16), 1, hd ** -.5,
                                *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
                                *q.stride()[:3], 8)
            forward = mod.dense_forward
            addr = p.data_ptr()
            def run(name=name, forward=forward, addr=addr):
                out = torch.empty_like(q, memory_format=torch.contiguous_format)
                forward(addr, q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr())
                outputs[name] = out
            arms[name] = run
        ref = C.ref_attention(q, k, v, True, dtype=torch.float32)
        errors = {}
        for name in mods:
            arms[name]()
            errors[name] = (outputs[name].float() - ref).abs().max().item()
            assert errors[name] <= (2e-4 if dt == torch.float32 else .03), errors
        samples = C.time_arms(arms, args.rounds)
        stats = {n: C.stats(xs) for n, xs in samples.items()}
        row = dict(shape=shape, dtype=args.dtype, stats=stats, samples_us=samples,
                   errors=errors, loadavg=os.getloadavg(),
                   ratio=stats["new"]["min"] / stats["old"]["min"],
                   control=stats["ctrl"]["min"] / stats["old"]["min"],
                   load_order=order)
        rows.append(row)
        print(shape, "new/old", round(row["ratio"], 4), "ctrl/old", round(row["control"], 4), flush=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
