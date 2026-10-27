"""Saved C++ builds against each other in one process (old/new kernel A/B).

    ab.py --so old=old.so new=new.so --rows dense:2048:8:64:bf16 \
        cross:1:4096:fp32 --rounds 41 --json out.json

Arms are every `--so` plus `ctrl`, a duplicate of the last one. Each arm calls
its own module's plan entry with its own plan buffer, output and (for paged)
its own staged cache copy, rotated per round. Ratios are `x / first`; the
duplicate control's distance from its twin is the noise floor.
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "audit")]
import audit_common as C  # noqa: E402
import cpu_attention as attn  # noqa: E402
from cpu_attention import _kernel  # noqa: E402

HEADS, DIM = 16, 64


def load(name, path):
    spec = importlib.util.spec_from_file_location(f"v_{name}._C", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dense_arms(mods, seq, heads, hd, dt, workers):
    q = torch.randn(1, seq, heads, hd, dtype=dt)
    k, v = torch.randn_like(q), torch.randn_like(q)
    outs, arms, keep = {}, {}, []
    for name, mod in mods.items():
        p = torch.empty(24, dtype=torch.int64)
        keep.append(p)
        mod.dense_make_plan(p.data_ptr(), 1, seq, seq, heads, heads, hd,
                            int(dt == torch.bfloat16), 1, hd ** -.5,
                            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
                            *q.stride()[:3], workers)

        def run(name=name, fwd=mod.dense_forward, addr=p.data_ptr()):
            out = torch.empty_like(q, memory_format=torch.contiguous_format)
            fwd(addr, q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr())
            outs[name] = out
        arms[name] = run
    ref = C.ref_attention(q, k, v, True, dtype=torch.float32)
    tol = 2e-4 if dt == torch.float32 else 3e-2
    return arms, outs, ref, tol, keep


def paged_arms(mods, op, batch, ctx, block, dt, epoch=0):
    pages = (ctx + block - 1) // block
    blocks = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    slots = blocks[:, (ctx - 1) // block].long() * block + (ctx - 1) % block
    lens = torch.full((batch,), ctx, dtype=torch.int32)
    q = torch.randn(batch, HEADS, DIM, dtype=dt)
    k_all = torch.randn(batch, ctx, HEADS, DIM, dtype=dt)
    v_all = torch.randn(batch, ctx, HEADS, DIM, dtype=dt)
    k_new, v_new = k_all[:, ctx - 1].contiguous(), v_all[:, ctx - 1].contiguous()
    b = torch.arange(batch).repeat_interleave(ctx)
    t = torch.arange(ctx).repeat(batch)
    all_slots = (b * pages * block + t).to(torch.int64)
    _kernel.set_backend("cpp")
    outs, arms, keep = {}, {}, []
    workers = _kernel._workers(batch * HEADS, pages)
    do_store = op == "self"
    # Allocation order rotates with the epoch: which arm's 32 MB cache lands
    # where in physical memory moved a same-binary arm 3-12% on the b=1/4096
    # rows, always in favour of the first-allocated arm.
    order = list(mods.items())
    order = order[epoch % len(order):] + order[:epoch % len(order)]
    for name, mod in order:
        kc = torch.zeros(batch * pages, HEADS, block, DIM, dtype=dt)
        vc = torch.zeros_like(kc)
        attn.reshape_and_cache(k_all.reshape(-1, HEADS, DIM),
                               v_all.reshape(-1, HEADS, DIM), kc, vc, all_slots)
        out = torch.empty_like(q)
        keep += [kc, vc, out]
        kn, vn, sl = (k_new, v_new, slots) if do_store else (None, None, None)
        if hasattr(mod, "paged_call"):
            # The torch-aware entry: same call the public API makes.
            mod.paged_register(q, kc, blocks, do_store, workers)

            def run(name=name, call=mod.paged_call, kc=kc, vc=vc, out=out):
                call(q, kn, vn, kc, vc, sl, blocks, lens, DIM ** -.5, out)
                outs[name] = out
        else:
            p = torch.empty(12, dtype=torch.int64)
            keep.append(p)
            mod.paged_make_plan(p.data_ptr(), batch, HEADS, HEADS, DIM, block,
                                pages, HEADS * DIM, int(dt == torch.bfloat16),
                                0.0, workers, int(do_store))

            def run(name=name, fwd=mod.paged_forward, addr=p.data_ptr(), kc=kc,
                    vc=vc, out=out):
                fwd(addr, q.data_ptr(), k_new.data_ptr() if do_store else 0,
                    v_new.data_ptr() if do_store else 0, kc.data_ptr(),
                    vc.data_ptr(), out.data_ptr(),
                    slots.data_ptr() if do_store else 0, blocks.data_ptr(),
                    lens.data_ptr(), q.stride(0),
                    k_new.stride(0) if do_store else 0, DIM ** -.5)
                outs[name] = out
        arms[name] = run
    ref = C.ref_decode(q, k_all.transpose(1, 2), v_all.transpose(1, 2))
    tol = 1e-6 if dt == torch.float32 else 3e-2
    return arms, outs, ref, tol, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    mods = {}
    for spec in args.so:
        name, path = spec.split("=", 1)
        mods[name] = load(name, path)
    names = list(mods)
    # The control is a third arm with its own plan, output and staged cache:
    # sharing its twin's 32 MB paged cache would hand both an L3 reuse the
    # other arms do not get (measured: a same-binary arm 15% slower).
    mods["ctrl"] = mods[names[-1]]
    torch.manual_seed(0)
    C.burn()
    rows = []
    for row in args.rows:
        kind, *rest = row.split(":")
        dt = torch.float32 if rest[-1] == "fp32" else torch.bfloat16
        if kind == "dense":
            seq, heads, hd = map(int, rest[:3])
            arms, outs, ref, tol, keep = dense_arms(mods, seq, heads, hd, dt,
                                                    args.threads)
            epochs = [lambda: (arms, outs, ref, tol, keep)]
        else:
            batch, ctx = map(int, rest[:2])
            epochs = [lambda e=e: paged_arms(mods, kind, batch, ctx,
                                             args.block, dt, e)
                      for e in range(len(mods))]
        samples, errors, same = {n: [] for n in mods}, {}, {}
        per = -(-args.rounds // len(epochs))
        for make in epochs:
            arms, outs, ref, tol, keep = make()
            for name in mods:
                arms[name]()
                err = (outs[name].float() - ref).abs().max().item()
                errors[name] = max(errors.get(name, 0.0), err)
            for n in mods:
                same[n] = same.get(n, True) and bool(
                    torch.equal(outs[n], outs[names[0]]))
            for n, xs in C.time_arms(arms, per).items():
                samples[n] += xs
        st = {n: C.stats(x) for n, x in samples.items()}
        base = st[names[0]]
        rec = dict(row=row, stats=st, samples_us=samples, errors=errors,
                   bit_identical_to_first=same, loadavg=os.getloadavg(),
                   threads=args.threads, block=args.block, rounds=args.rounds,
                   ratio_min={n: st[n]["min"] / base["min"] for n in st},
                   ratio_med={n: st[n]["median"] / base["median"] for n in st})
        rows.append(rec)
        line = f"{row:24s} " + " ".join(
            f"{n} {st[n]['min']:9.1f} ({st[n]['median']:9.1f})" for n in st)
        line += "  min/first " + " ".join(
            f"{n}={rec['ratio_min'][n]:.3f}" for n in st if n != names[0])
        line += "  med/first " + " ".join(
            f"{n}={rec['ratio_med'][n]:.3f}" for n in st if n != names[0])
        bad = {n: e for n, e in errors.items() if not e <= tol}
        print(line + f"  err {max(errors.values()):.1e}" +
              (f" FAIL {bad}" if bad else ""), flush=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
