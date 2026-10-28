"""The benchmark: unified Mojo vs every applicable C++ path, one process.

    scripts/pin_governor.sh uv run python benchmarks/bench.py --manifest full \
        --json reports/ryzen/<rev>.json [--vllm-bf16 X.so --vllm-fp32 Y.so]

Rules (docs/CPP.md §7):

* Every arm is an independently loaded copy of the package (benchmarks/arms.py):
  ``mojo_new`` (this checkout), ``mojo_ctrl`` (a second, independent copy of it:
  the noise floor), ``mojo_old`` (the oracle build at tag ``mojo-oracle-v1``),
  one ``cpp_<isa>`` arm per C++ ISA path applicable on this CPU (oracle-tag glue,
  this checkout's ``_C_cpp.so``), plus vLLM and SDPA as secondary references.
* Arms are interleaved, the order rotated every round; the minimum decides and
  raw samples + median are kept. Per-arm env (the forced C++ ISA) is applied
  outside the timer.
* Each arm stages its own cache through its own ``reshape_and_cache`` from the
  same logical data; repeated append calls rewrite the same slot, so contexts
  never grow. Allocation is charged identically: ours include the pack and the
  output allocation (except ``engine_out`` rows); vLLM's cache is staged and
  its output preallocated outside the timer.
* Every timed arm's output is validated against the gates after timing; an
  invalid arm is excluded from the baseline and reported.
* ``baseline`` is the fastest *valid* C++ arm; ``ratio = mojo_new / baseline``.
  A row is a ``win`` only below ``1 - band`` with ``band = max(0.03, |ctrl/new
  - 1|)``, a ``loss`` above ``1 + band``, and ``unresolved`` in between.
* A C++ arm more than ``--screen-x`` times slower than the fastest C++ arm in a
  3-call screen cannot be the baseline and is not timed further (the screen
  samples are kept in the JSON).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import arms as A  # noqa: E402
import manifest as M  # noqa: E402
import reference as R  # noqa: E402

DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
ARM64 = platform.machine() in ("aarch64", "arm64")


# ---------------------------------------------------------------------------
# environment record
# ---------------------------------------------------------------------------
def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def environment(arms_):
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    mojo = os.path.join(os.path.dirname(sys.executable), "mojo")
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _sh("git", "-C", _ROOT, "rev-parse", "HEAD"),
        "git_status": _sh("git", "-C", _ROOT, "status", "--porcelain",
                          "cpu_attention"),
        "mojo": _sh(mojo, "--version") if os.path.exists(mojo) else "?",
        "torch": torch.__version__, "python": sys.version.split()[0],
        "gcc": _sh("gcc", "--version").split("\n")[0],
        "lscpu": _sh("lscpu"), "uname": _sh("uname", "-a"),
        "governor": open(gov).read().strip() if os.path.exists(gov) else None,
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": torch.get_num_threads(),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("CPU_ATTENTION", "OMP", "KMP", "VLLM"))},
        "arms": {a.name: a.describe() for a in arms_},
    }


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_arms(fns, envs, rounds, calls, warmup=2):
    """Rotated round-robin; per-call microseconds. ``envs[name]()`` runs
    before each sample, outside the timer."""
    names = list(fns)
    for n in names:
        envs[n]()
        for _ in range(warmup):
            fns[n]()
    samples = {n: [] for n in names}
    for r in range(rounds):
        k = r % len(names)
        for n in names[k:] + names[:k]:
            envs[n]()
            fn = fns[n]
            t0 = time.perf_counter_ns()
            for _ in range(calls):
                fn()
            samples[n].append((time.perf_counter_ns() - t0) / 1000.0 / calls)
    return samples


def stats(xs):
    s = sorted(xs)
    return {"min": s[0], "median": s[len(s) // 2], "n": len(s)}


# ---------------------------------------------------------------------------
# flash_attn_func rows
# ---------------------------------------------------------------------------
def flash_inputs(r, dtype):
    b, sq, sk, h, hkv, d = r["b"], r["sq"], r["sk"], r["h"], r["hkv"], r["d"]
    if r["layout"] == "fused":
        assert sq == sk and h == hkv
        buf = torch.randn(b, sq, 3 * h * d, dtype=dtype)
        q, k, v = (buf[..., i * h * d:(i + 1) * h * d].view(b, sq, h, d)
                   for i in range(3))
        return q, k, v
    return (torch.randn(b, sq, h, d, dtype=dtype),
            torch.randn(b, sk, hkv, d, dtype=dtype),
            torch.randn(b, sk, hkv, d, dtype=dtype))


def flash_row(r, arms_, vllm, args):
    dtype = DT[r["dtype"]]
    q, k, v = flash_inputs(r, dtype)
    scale = r["d"] ** -0.5
    causal = r["causal"]
    out = {}
    fns = {}
    for a in arms_:
        def f(a=a):
            out[a.name] = a.pkg.flash_attn_func(q, k, v, scale, causal)
        fns[a.name] = f
    qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    mask = None
    if causal and r["sq"] != r["sk"]:
        mask = (torch.arange(r["sk"])[None, :] <=
                torch.arange(r["sq"])[:, None] + (r["sk"] - r["sq"]))
    gqa = r["h"] != r["hkv"]

    def sdpa():
        out["sdpa"] = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=mask, is_causal=causal and mask is None,
            scale=scale, enable_gqa=gqa).transpose(1, 2)
    fns["sdpa"] = sdpa
    vl = vllm.get(r["dtype"]) if vllm else None
    if vl is not None and r["sq"] == r["sk"] and r["b"] == 1:
        fns["vllm"] = vllm_flash(vl, r, q, k, v, scale, out)
    ref = R.ref_attention(q, k, v, causal, scale=scale)
    tol = 3e-2 if r["dtype"] == "bf16" else 2e-4

    def check(name):
        got = out[name].float()
        err = (got - ref).abs().max().item()
        if not err <= tol and r["dtype"] == "fp32":
            truth = R.ref_attention(q, k, v, causal, scale=scale,
                                    dtype=torch.float64)
            if (ref.double() - truth).abs().max().item() > tol:
                err = (got.double() - truth).abs().max().item()
        return err, err <= tol
    return fns, check, args.rounds, 1


def vllm_flash(vl, r, q, k, v, scale, out):
    mod, isa = vl
    seq, h, hkv, d, dtype = r["sq"], r["h"], r["hkv"], r["d"], q.dtype
    bs = 64
    nb = (seq + bs - 1) // bs
    qf = q.reshape(seq, h, d).contiguous()
    kc = torch.zeros(nb, hkv, bs, d, dtype=dtype)
    vc = torch.zeros_like(kc)
    mod.cpu_attn_reshape_and_cache(k.reshape(seq, hkv, d).contiguous(),
                                   v.reshape(seq, hkv, d).contiguous(), kc, vc,
                                   torch.arange(seq, dtype=torch.int64), isa)
    lens = torch.tensor([seq], dtype=torch.int32)
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    meta = mod.get_scheduler_metadata(1, h, hkv, d, lens, dtype, qsl,
                                      r["causal"], -1, isa, False, None, "auto")
    o = torch.empty_like(qf)
    bt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)

    def f():
        mod.cpu_attention_with_kv_cache(qf, kc, vc, o, qsl, lens, scale,
                                        r["causal"], None, -1, bt, 0.0, meta,
                                        None, None, 1.0, 1.0, "auto")
        out["vllm"] = o.view(1, seq, h, d)
    return f


# ---------------------------------------------------------------------------
# flash_attn_with_kvcache / engine rows
# ---------------------------------------------------------------------------
def paged_inputs(r, dtype, page):
    b, h, hkv, d = r["b"], r["h"], r["hkv"], r["d"]
    ctxs = r["ctx"] if isinstance(r["ctx"], list) else [r["ctx"]] * b
    npages = [(c + page - 1) // page for c in ctxs]
    width = max(npages)
    bt = torch.zeros(b, width, dtype=torch.int32)
    nxt = 0
    for i, n in enumerate(npages):
        bt[i, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32)
        nxt += n
    lens = torch.tensor(ctxs, dtype=torch.int32)
    kall = [torch.randn(c, hkv, d, dtype=dtype) for c in ctxs]
    vall = [torch.randn(c, hkv, d, dtype=dtype) for c in ctxs]
    q = torch.randn(b, h, d, dtype=dtype)
    slots = torch.tensor([int(bt[i, (c - 1) // page]) * page + (c - 1) % page
                          for i, c in enumerate(ctxs)], dtype=torch.int64)
    allslots = torch.cat([bt[i, torch.arange(c) // page].long() * page +
                          torch.arange(c) % page for i, c in enumerate(ctxs)])
    return dict(q=q, kall=kall, vall=vall, bt=bt, lens=lens, slots=slots,
                allslots=allslots, shape=(nxt, hkv, page, d),
                kn=torch.stack([x[-1] for x in kall]),
                vn=torch.stack([x[-1] for x in vall]))


def paged_row(r, arms_, vllm, args, page):
    dtype = DT[r["dtype"]]
    x = paged_inputs(r, dtype, page)
    q, bt, lens, slots = x["q"], x["bt"], x["lens"], x["slots"]
    kn, vn = x["kn"], x["vn"]
    scale = r["d"] ** -0.5
    kflat, vflat = torch.cat(x["kall"]), torch.cat(x["vall"])
    out, fns, caches = {}, {}, {}
    entry, op = r["entry"], r["op"]
    for a in arms_:
        kc = torch.zeros(x["shape"], dtype=dtype)
        vc = torch.zeros_like(kc)
        a.enter()
        a.pkg.reshape_and_cache(kflat, vflat, kc, vc, x["allslots"])
        caches[a.name] = (kc, vc)
        o = torch.empty_like(q)
        p = a.pkg

        if entry == "store":
            def f(p=p, kc=kc, vc=vc):
                p.reshape_and_cache(kflat, vflat, kc, vc, x["allslots"])
        elif entry == "api":
            q4 = q.unsqueeze(1)
            if op == "self":
                k4, v4, pre = kn.unsqueeze(1), vn.unsqueeze(1), lens - 1

                def f(p=p, kc=kc, vc=vc, n=a.name):
                    out[n] = p.flash_attn_with_kvcache(
                        q4, kc, vc, k=k4, v=v4, cache_seqlens=pre,
                        page_table=bt, softmax_scale=scale).squeeze(1)
            else:
                def f(p=p, kc=kc, vc=vc, n=a.name):
                    out[n] = p.flash_attn_with_kvcache(
                        q4, kc, vc, cache_seqlens=lens, page_table=bt,
                        softmax_scale=scale).squeeze(1)
        else:
            dst = o if entry == "engine_out" else None
            if op == "self":
                def f(p=p, kc=kc, vc=vc, n=a.name, dst=dst):
                    out[n] = p.self_attention(q, kn, vn, kc, vc, slots, bt,
                                              lens, scale, out=dst)
            else:
                def f(p=p, kc=kc, vc=vc, n=a.name, dst=dst):
                    out[n] = p.cross_attention(q, kc, vc, bt, lens, scale,
                                               out=dst)
        fns[a.name] = f
    vl = vllm.get(r["dtype"]) if vllm else None
    if vl is not None and entry in ("engine", "engine_out"):
        try:
            fns["vllm"] = vllm_paged(vl, r, x, q, kn, vn, scale, op, out)
        except Exception as e:  # noqa: BLE001
            print(f"  vllm unavailable for this row: {e}", flush=True)

    kref = torch.nn.utils.rnn.pad_sequence(x["kall"], batch_first=True)
    vref = torch.nn.utils.rnn.pad_sequence(x["vall"], batch_first=True)
    ref = R.ref_decode(q, kref.transpose(1, 2), vref.transpose(1, 2),
                       scale=scale, lens=lens)
    peer = {}

    def result(name):
        if entry != "store":
            return out[name].float()
        # A store is validated by attending over what it wrote.
        arm = next(a for a in arms_ if a.name == name)
        arm.enter()
        kc, vc = caches[name]
        return arm.pkg.cross_attention(q, kc, vc, bt, lens, scale).float()

    def check(name):
        got = result(name)
        if r["dtype"] == "fp32":
            err = (got - ref).abs().max().item()
            if err > 1e-6:
                truth = R.ref_decode(q, kref.transpose(1, 2),
                                     vref.transpose(1, 2), dtype=torch.float64,
                                     scale=scale, lens=lens)
                if (ref.double() - truth).abs().max().item() > 1e-6:
                    err = (got.double() - truth).abs().max().item()
            return err, err <= 1e-6
        # bf16: <= 2 bf16 ulps at the row peak against the C++ reference arm
        # (the test_paged_sweep gate); on aarch64 also <= 1e-3 vs the reference.
        unit = ref.abs().max().clamp_min(1e-30).item() * 2 ** -8
        err_ref = (got - ref).abs().max().item()
        err = (got - peer["ref"]).abs().max().item() / unit
        ok = err <= 2.0 and err_ref <= 3e-2 and (err_ref <= 1e-3 or not ARM64)
        return err, ok

    def check_all(names):
        if r["dtype"] == "bf16":
            peer["ref"] = result(args.ref_arm)
        return {n: check(n) for n in names}
    rounds = args.paged_rounds
    return fns, check_all, rounds, args.block_calls


def vllm_paged(vl, r, x, q, kn, vn, scale, op, out):
    mod, isa = vl
    b, h, hkv, d = r["b"], r["h"], r["hkv"], r["d"]
    kc = torch.zeros(x["shape"], dtype=q.dtype)
    vc = torch.zeros_like(kc)
    mod.cpu_attn_reshape_and_cache(torch.cat(x["kall"]), torch.cat(x["vall"]),
                                   kc, vc, x["allslots"], isa)
    qsl = torch.arange(b + 1, dtype=torch.int32)
    meta = mod.get_scheduler_metadata(b, h, hkv, d, x["lens"], q.dtype, qsl,
                                      True, -1, isa, False, None, "auto")
    o = torch.empty_like(q)
    bt, lens, slots = x["bt"], x["lens"], x["slots"]

    def f():
        if op == "self":
            mod.cpu_attn_reshape_and_cache(kn, vn, kc, vc, slots, isa)
        mod.cpu_attention_with_kv_cache(q, kc, vc, o, qsl, lens, scale, True,
                                        None, -1, bt, 0.0, meta, None, None,
                                        1.0, 1.0, "auto")
        out["vllm"] = o
    f()
    return f


# ---------------------------------------------------------------------------
def build_arms(args):
    arms_ = []
    want = args.arms.split(",")
    if "mojo_new" in want:
        arms_.append(A.current("mojo_new", "mojo"))
    if "mojo_ctrl" in want:
        arms_.append(A.current("mojo_ctrl", "mojo"))
    if "mojo_old" in want and os.path.exists(args.oracle):
        arms_.append(A.tagged("mojo_old", args.oracle_tag, "mojo",
                              {"_C.so": args.oracle}))
    cpp = os.path.join(A.PKG, "_C_cpp.so")
    if "cpp" in want:
        isas = sorted(set(A.cpp_isas("bf16")) | set(A.cpp_isas("fp32")),
                      key=["amx", "vec", "neon", "scalar"].index)
        for isa in isas:
            arms_.append(A.tagged(f"cpp_{isa}", args.oracle_tag, "cpp",
                                  {"_C_cpp.so": cpp},
                                  {"CPU_ATTENTION_ISA": isa}))
    return arms_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="full",
                    choices=("full", "historical", "extended"))
    ap.add_argument("--only", default="", help="substring filter on row labels")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "both"), default="both")
    ap.add_argument("--arms", default="mojo_new,mojo_ctrl,mojo_old,cpp,sdpa,vllm")
    ap.add_argument("--oracle", default=os.path.join(
        _ROOT, "cpu_attention", "_mojo", "oracle", "_C_oracle.so"))
    ap.add_argument("--oracle-tag", default="mojo-oracle-v1")
    ap.add_argument("--vllm-bf16", default=os.environ.get("VLLM_REF_BF16", ""),
                    help="vLLM build for bf16 rows (AMX build on AMX boxes)")
    ap.add_argument("--vllm-fp32", default=os.environ.get("VLLM_REF_FP32", ""))
    ap.add_argument("--vllm-isa-bf16", default="vec")
    ap.add_argument("--vllm-isa-fp32", default="vec")
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--paged-rounds", type=int, default=41)
    ap.add_argument("--block-calls", type=int, default=1)
    ap.add_argument("--screen-x", type=float, default=2.0)
    ap.add_argument("--page", type=int, default=16 if ARM64 else 32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    arms_ = build_arms(args)
    names = [a.name for a in arms_]
    cpp_names = [n for n in names if n.startswith("cpp_")]
    args.ref_arm = cpp_names[0] if cpp_names else names[0]
    vllm = {}
    if "vllm" in args.arms:
        for tag in ("bf16", "fp32"):
            path = getattr(args, f"vllm_{tag}")
            mod = R.load_so(path) if path else None
            if mod is not None:
                vllm[tag] = (mod, getattr(args, f"vllm_isa_{tag}"))
    env = environment(arms_)
    env.update(argv=sys.argv, vllm={k: [getattr(args, f"vllm_{k}"), v[1]]
                                    for k, v in vllm.items()})
    print(json.dumps({k: env[k] for k in ("git", "mojo", "torch", "threads",
                                          "governor", "arms")}, indent=1),
          flush=True)
    rows = [r for r in M.rows(args.manifest)
            if args.dtype in ("both", r["dtype"]) and args.only in M.label(r)]
    results = []
    R.burn()
    torch.manual_seed(0)
    for r in rows:
        label = M.label(r)
        if r["kind"] == "flash":
            fns, check, rounds, calls = flash_row(r, arms_, vllm, args)
            checks = None
        else:
            page = r["page"] or args.page
            fns, check_all, rounds, calls = paged_row(r, arms_, vllm, args, page)
            check, checks = None, check_all
        envs = {n: next((a.enter for a in arms_ if a.name == n), A.clear_env)
                for n in fns}
        if "sdpa" not in args.arms:
            fns.pop("sdpa", None)
        # Screen: a C++ arm far slower than the fastest one cannot be the
        # baseline; keep its screen samples and stop timing it.
        screen = time_arms({n: fns[n] for n in cpp_names}, envs, 3, 1, warmup=1) \
            if len(cpp_names) > 1 else {}
        dropped = {}
        if screen:
            best = min(min(s) for s in screen.values())
            for n, s in screen.items():
                if min(s) > args.screen_x * best:
                    dropped[n] = s
                    fns.pop(n)
        samples = time_arms(fns, envs, rounds, calls)
        st = {n: stats(s) for n, s in samples.items()}
        valid = (checks(list(fns)) if checks else
                 {n: check(n) for n in fns})
        cpp_valid = [n for n in fns if n in cpp_names and valid[n][1]]
        row = dict(r, label=label, rounds=rounds, block_calls=calls,
                   min_us={n: s["min"] for n, s in st.items()},
                   median_us={n: s["median"] for n, s in st.items()},
                   samples_us=samples, err={n: v[0] for n, v in valid.items()},
                   valid={n: v[1] for n, v in valid.items()},
                   screened_out=dropped, loadavg=os.getloadavg())
        line = f"{label:48s}"
        if cpp_valid and "mojo_new" in st:
            base = min(cpp_valid, key=lambda n: st[n]["min"])
            ratio = st["mojo_new"]["min"] / st[base]["min"]
            spread = (abs(st["mojo_ctrl"]["min"] / st["mojo_new"]["min"] - 1)
                      if "mojo_ctrl" in st else 0.0)
            band = max(0.03, spread)
            verdict = ("win" if ratio < 1 - band else
                       "loss" if ratio > 1 + band else "unresolved")
            if not valid["mojo_new"][1]:
                verdict = "INVALID"
            row.update(baseline=base, ratio=ratio, ctrl_spread=spread,
                       band=band, verdict=verdict)
            line += (f" new {st['mojo_new']['min']:10.1f} {base} "
                     f"{st[base]['min']:10.1f} ratio {ratio:5.3f} "
                     f"ctrl {spread:5.3f} {verdict}")
        for n in ("mojo_old", "vllm", "sdpa"):
            if n in st and "mojo_new" in st:
                row[f"new/{n}"] = st["mojo_new"]["min"] / st[n]["min"]
                line += f" /{n} {row[f'new/{n}']:5.3f}"
        bad = [n for n, v in valid.items() if not v[1]]
        if bad:
            line += f"  INVALID {bad}"
        print(line, flush=True)
        results.append(row)
        if args.json:  # rewritten every row: a spot preemption keeps the rest
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w") as f:
                json.dump({"environment": env, "rows": results}, f)
    summarize(results)
    return 0


def summarize(rows):
    v = [r.get("verdict") for r in rows if "verdict" in r]
    print(f"\n{len(v)} rows: {v.count('win')} win, {v.count('unresolved')} "
          f"unresolved, {v.count('loss')} loss, {v.count('INVALID')} invalid")
    for r in rows:
        if r.get("verdict") not in ("win", None):
            print(f"  {r['verdict']:10s} {r['label']}  ratio {r['ratio']:.3f} "
                  f"vs {r['baseline']}")


if __name__ == "__main__":
    raise SystemExit(main())
