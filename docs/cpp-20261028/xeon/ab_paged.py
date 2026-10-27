"""Old-vs-new C++ .so A/B for the paged path in one process: five rotated arms
(old, new, a duplicate of each, vLLM VEC with its own staged cache).

    ab_paged.py --old a.so --new b.so --cases 1:4096,8:256 --dtype fp32
"""
import argparse
import importlib.util
import itertools
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, ROOT + "/audit", ROOT + "/tests"]
import torch  # noqa: E402
import audit_common as C  # noqa: E402
import bench_cpp as B  # noqa: E402
import cpu_attention as A  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--new", required=True)
p.add_argument("--cases", default="1:4096,8:256")
p.add_argument("--dtype", default="fp32")
p.add_argument("--rounds", type=int, default=121)
p.add_argument("--vllm", default=os.path.expanduser(
    "~/vllm_cpu_build/vllm_cpu_attn_bench.so"))
p.add_argument("--json", required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(0)


def load(path):
    spec = importlib.util.spec_from_file_location("cpu_attention._C", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mods = {"old": load(a.old), "new": load(a.new)}
vllm = C.load_so(a.vllm)
dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
block = 32
rows = []
C.burn()
for case in a.cases.split(","):
    batch, ctx = map(int, case.split(":"))
    (q, k_new, v_new, k_all, v_all, kc, vc, slots, blocks,
     lens) = B.paged_case(batch, ctx, block, dt)
    kcc, vcc = torch.zeros_like(kc), torch.zeros_like(vc)
    B.stage(k_all, v_all, kcc, vcc, block, batch, ctx, "cpp", None, "cpp")
    B.stage(k_all, v_all, kc, vc, block, batch, ctx, "vec", vllm, None)
    qsl = torch.arange(batch + 1, dtype=torch.int32)
    meta = vllm.get_scheduler_metadata(batch, B.HEADS, B.HEADS, B.DIM, lens, dt,
                                       qsl, True, -1, "vec", False, None, "auto")
    vout = torch.empty_like(q)
    scale = B.DIM ** -0.5
    for op in ("self", "cross"):
        outs = {}

        def arm(name, op=op):
            def f():
                K.set_backend("cpp")
                K._BOUND[("cpp", "paged_call")] = mods[name].paged_call
                K._BOUND[("cpp", "paged_register")] = mods[name].paged_register
                if op == "self":
                    outs[name] = A.self_attention(q, k_new, v_new, kcc, vcc,
                                                  slots, blocks, lens, scale)
                else:
                    outs[name] = A.cross_attention(q, kcc, vcc, blocks, lens,
                                                   scale)
            return f

        def vf(op=op):
            if op == "self":
                vllm.cpu_attn_reshape_and_cache(k_new, v_new, kc, vc, slots,
                                                "vec")
            vllm.cpu_attention_with_kv_cache(
                q, kc, vc, vout, qsl, lens, scale, True, None, -1, blocks, 0.0,
                meta, None, None, 1.0, 1.0, "auto")

        arms = {"old": arm("old"), "new": arm("new"), "ctrl": arm("new"),
                "oldctrl": arm("old"), "vllm": vf}
        C.time_arms(arms, 3)
        orders = list(itertools.permutations(arms))
        random.Random(0).shuffle(orders)
        samples = {n: [] for n in arms}
        for r in range(a.rounds):
            got = C.time_arms({n: arms[n] for n in orders[r % len(orders)]}, 1,
                              warmup=0)
            for n, xs in got.items():
                samples[n].extend(xs)
        st = {n: C.stats(x) for n, x in samples.items()}
        ref = C.ref_decode(q, k_all.transpose(1, 2), v_all.transpose(1, 2))
        err = {n: (outs[n].float() - ref).abs().max().item() for n in mods}
        row = dict(case=case, op=op, dtype=a.dtype, stats=st, samples_us=samples,
                   err=err, new_old=st["new"]["min"] / st["old"]["min"],
                   new_old_median=st["new"]["median"] / st["old"]["median"],
                   ctrl_new=st["ctrl"]["min"] / st["new"]["min"],
                   oldctrl_old=st["oldctrl"]["min"] / st["old"]["min"])
        rows.append(row)
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"{op:5s} b={batch} ctx={ctx} {a.dtype} new/old {row['new_old']:.3f} "
              f"(med {row['new_old_median']:.3f}) ctrl/new {row['ctrl_new']:.3f} "
              f"oldctrl/old {row['oldctrl_old']:.3f} new/vllm "
              f"{st['new']['min'] / st['vllm']['min']:.3f} old/vllm "
              f"{st['old']['min'] / st['vllm']['min']:.3f} | new {st['new']['min']:.1f}"
              f" old {st['old']['min']:.1f} vllm {st['vllm']['min']:.1f} "
              f"err {err['old']:.1e}/{err['new']:.1e}", flush=True)
