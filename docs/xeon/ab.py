"""Paired-process A/B for the C++ dense kernel.

A variant is selected by environment, so two variants cannot share a process
(``env_int`` caches). This runs the two configurations as *alternating
processes* -- A B B A A B ... -- and every process also times the Mojo arm,
which neither variant touches, so each process yields a drift-free
``cpp/mojo``. The report is the median over processes of that ratio, plus the
same for the raw time. ``ctrl`` (a duplicate ``cpp`` arm in the same process)
is the floor.

    python docs/xeon/ab.py --shape 1024:8:64 --dtype fp32 --envs A=,B=NEW

``--envs`` is a comma-separated list of ``name=k1=v1;k2=v2``.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "audit"))

import audit_common as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402


def inner(shape, dtype, rounds):
    seq, heads, dim = (int(x) for x in shape.split(":"))
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    q = torch.randn(1, seq, heads, dim, dtype=dt)
    k = torch.randn(1, seq, heads, dim, dtype=dt)
    v = torch.randn(1, seq, heads, dim, dtype=dt)
    scale = dim ** -0.5

    def carm():
        _kernel.set_backend("cpp")
        return attn.flash_attn_func(q, k, v, scale, True)

    def marm():
        _kernel.set_backend("mojo")
        return attn.flash_attn_func(q, k, v, scale, True)

    C.burn()
    s = C.time_arms({"cpp": carm, "ctrl": carm, "mojo": marm}, rounds)
    return {n: C.stats(x) for n, x in s.items()}


def run_cfg(cfg, shape, dtype, rounds):
    env = dict(os.environ)
    env.pop("CPU_ATTENTION_AMX_V", None)
    for kv in cfg["env"].split(";"):
        if kv:
            a, b = kv.split("=", 1)
            env[a] = b
    r = subprocess.run([sys.executable, __file__, "--inner", "--shape", shape,
                        "--dtype", dtype, "--rounds", str(rounds)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1024:8:64")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--envs", default="A=,B=CPU_ATTENTION_AMX_V=1")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfgs = []
    for spec in args.envs.split(","):
        name, _, env = spec.partition("=")
        cfgs.append({"name": name, "env": env})
    got = {c["name"]: [] for c in cfgs}
    # A B B A A B B A ...: each configuration appears equally often in each
    # position, so a monotone drift cannot favour either.
    order = []
    for i in range(args.pairs):
        order += cfgs if i % 2 == 0 else list(reversed(cfgs))
        order += list(reversed(cfgs)) if i % 2 == 0 else cfgs

    for c in order:
        got[c["name"]].append(run_cfg(c, args.shape, args.dtype, args.rounds))

    out = []
    print(f"shape {args.shape} {args.dtype}  rounds={args.rounds} "
          f"n={len(got[cfgs[0]['name']])} per config")
    ref = cfgs[0]["name"]
    for c in cfgs:
        rows = got[c["name"]]
        cpp = statistics.median(r["cpp"]["min"] for r in rows)
        ctrl = statistics.median(r["ctrl"]["min"] for r in rows)
        mojo = statistics.median(r["mojo"]["min"] for r in rows)
        ratio = statistics.median(r["cpp"]["min"] / r["mojo"]["min"] for r in rows)
        floor = statistics.median(r["ctrl"]["min"] / r["cpp"]["min"] for r in rows)
        row = {"cfg": c["name"], "env": c["env"], "cpp": cpp, "ctrl": ctrl,
               "mojo": mojo, "cpp/mojo": ratio, "ctrl/cpp": floor,
               "samples": [r["cpp"]["min"] for r in rows]}
        out.append(row)
        print(f"  {c['name']:6s} cpp {cpp:9.1f} ctrl {ctrl:9.1f} mojo "
              f"{mojo:9.1f}  cpp/mojo {ratio:5.3f}  ctrl/cpp {floor:5.3f}  "
              f"raw {['%.1f' % s for s in row['samples']]}", flush=True)
    if len(cfgs) == 2:
        a, b = out
        pa = statistics.median(r["cpp"]["min"] / r["mojo"]["min"]
                               for r in got[a["cfg"]])
        pb = statistics.median(r["cpp"]["min"] / r["mojo"]["min"]
                               for r in got[b["cfg"]])
        raw = [r["cpp"]["min"] for r in got[b["cfg"]]] and statistics.median(
            r["cpp"]["min"] for r in got[b["cfg"]]) / statistics.median(
                r["cpp"]["min"] for r in got[a["cfg"]])
        print(f"  >> {b['cfg']}/{a['cfg']}: paired {pb / pa:5.3f}  raw {raw:5.3f}"
              f"  (paired <1 means B faster)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"shape": args.shape, "dtype": args.dtype, "rows": out},
                      f, indent=1)
    return 0


if __name__ == "__main__":
    if "--inner" in sys.argv:
        a = sys.argv[sys.argv.index("--shape") + 1]
        d = sys.argv[sys.argv.index("--dtype") + 1]
        r = int(sys.argv[sys.argv.index("--rounds") + 1])
        print(json.dumps(inner(a, d, r)))
    else:
        raise SystemExit(main())
