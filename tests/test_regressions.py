"""Regression coverage for both APIs.

    uv run python tests/test_regressions.py [--quick]

Gates are the repository's (README): flash_attn_func fp32 <= 2e-4 and bf16 <=
3e-2 against the naive fp32 reference (fp64 arbitrates a reference that is
itself out of tolerance); paged fp32 <= 1e-6; paged bf16 <= 1e-3 beyond the bf16
output rounding (reference.bf16_excess).

Covers: odd lengths around tile/page boundaries, unequal query/key lengths
(bottom-right causal), fully masked rows, MHA/GQA/MQA, all head dims, causal
and noncausal, independent q/k/v strides, page boundaries with permuted
pages, NaN-poisoned cache slots and padding, negative slots (store skipped),
zero-length contexts, out= guard bands, GQA/MQA store order, changed scale,
worker counts, AMX off/on, workspace eviction, concurrent callers, and two
independently loaded builds in one process. Cases that could crash the
interpreter (memory safety) run in a subprocess.
"""

import os
import subprocess
import sys
import threading

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import reference as R  # noqa: E402
import cpu_attention as A  # noqa: E402
from cpu_attention import _kernel as K  # noqa: E402

QUICK = "--quick" in sys.argv
ODD = [15, 17, 31, 33, 47, 49, 63, 65, 127, 129, 255, 257, 511, 513, 1023,
       1025, 2047, 2049, 4095, 4097]
PAIRS = [(1, 129), (17, 301), (127, 129), (129, 127), (257, 513), (513, 257),
         (1025, 2049), (2049, 1025)]
DTYPES = (torch.float32, torch.bfloat16)
FAILS = []
COUNT = [0]


def tag(dt):
    return "fp32" if dt == torch.float32 else "bf16"


def report(name, ok, detail=""):
    COUNT[0] += 1
    if not ok:
        FAILS.append(f"{name} {detail}")
        print(f"[FAIL] {name} {detail}", flush=True)


def flash_err(q, k, v, causal, scale, got):
    ref = R.ref_attention(q, k, v, causal, scale=scale)
    err = (got.float() - ref).abs().max().item()
    tol = 2e-4 if q.dtype == torch.float32 else 3e-2
    if not err <= tol and q.dtype == torch.float32:
        truth = R.ref_attention(q, k, v, causal, scale=scale, dtype=torch.float64)
        if (ref.double() - truth).abs().max().item() > tol:
            err = (got.double() - truth).abs().max().item()
    return err, tol


def check_flash(name, q, k, v, causal, scale=None):
    got = A.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    s = q.shape[-1] ** -0.5 if scale is None else scale
    err, tol = flash_err(q, k, v, causal, s, got)
    report(name, err <= tol, f"err {err:.3e} > {tol}")
    return got


# ---------------------------------------------------------------------------
# flash_attn_func
# ---------------------------------------------------------------------------
def heads_for(i):
    # Rotate MHA / GQA / MQA across cases.
    return ((4, 4), (4, 2), (4, 1))[i % 3]


def test_odd_lengths():
    lens = ODD[::3] if QUICK else ODD
    i = 0
    for s in lens:
        for d in (32, 64, 128):
            for causal in (True, False):
                for dt in DTYPES:
                    if s > 2049 and (d != 64 or not causal):
                        continue  # the longest lengths at one shape keep it short
                    h, hk = heads_for(i)
                    i += 1
                    q = torch.randn(1, s, h, d, dtype=dt)
                    k = torch.randn(1, s, hk, d, dtype=dt)
                    v = torch.randn(1, s, hk, d, dtype=dt)
                    check_flash(f"odd s={s} d={d} h={h}/{hk} causal={causal} "
                                f"{tag(dt)}", q, k, v, causal)


def test_unequal():
    i = 0
    for sq, sk in PAIRS:
        for d in ((64,) if QUICK else (32, 64, 128)):
            for causal in (True, False):
                for dt in DTYPES:
                    if max(sq, sk) > 1025 and d != 64:
                        continue
                    h, hk = heads_for(i)
                    i += 1
                    q = torch.randn(2, sq, h, d, dtype=dt)
                    k = torch.randn(2, sk, hk, d, dtype=dt)
                    v = torch.randn(2, sk, hk, d, dtype=dt)
                    got = check_flash(f"unequal {sq}x{sk} d={d} h={h}/{hk} "
                                      f"causal={causal} {tag(dt)}",
                                      q, k, v, causal)
                    if causal and sq > sk:
                        # Rows i < sq - sk see no key: exactly zero.
                        dead = got[:, :sq - sk]
                        report(f"masked rows exactly 0 {sq}x{sk} {tag(dt)}",
                               bool((dead == 0).all()),
                               f"max {dead.float().abs().max().item()}")


def test_strides():
    for dt in DTYPES:
        for d in (32, 64, 128):
            s, h, hk = 257, 4, 2
            # q, k, v from three different buffers with three token strides.
            qb = torch.randn(1, s, h * d + 24, dtype=dt)
            kb = torch.randn(1, s, 3 * hk * d, dtype=dt)
            vb = torch.randn(1, s + 7, hk * d + 40, dtype=dt)
            q = qb[..., 8:8 + h * d].unflatten(-1, (h, d))
            k = kb[..., hk * d:2 * hk * d].unflatten(-1, (hk, d))
            v = vb[:, :s, 16:16 + hk * d].unflatten(-1, (hk, d))
            for causal in (True, False):
                check_flash(f"strides d={d} causal={causal} {tag(dt)}",
                            q, k, v, causal)
            # The engine's layout: one fused qkv projection.
            buf = torch.randn(1, s, 3 * h * d, dtype=dt)
            q, k, v = (buf[..., j * h * d:(j + 1) * h * d].unflatten(-1, (h, d))
                       for j in range(3))
            check_flash(f"fused qkv d={d} {tag(dt)}", q, k, v, True)


def test_scale_and_plans():
    for dt in DTYPES:
        q = torch.randn(1, 129, 4, 64, dtype=dt)
        k = torch.randn(1, 129, 4, 64, dtype=dt)
        v = torch.randn(1, 129, 4, 64, dtype=dt)
        for scale in (0.125, 0.3, 0.125, 1.7):  # same geometry, changed scale
            check_flash(f"scale {scale} {tag(dt)}", q, k, v, True, scale)
        # Workspace eviction: a large call, a drop, a small call, the large again.
        big = [torch.randn(1, 2049, 8, 128, dtype=dt) for _ in range(3)]
        check_flash(f"evict big {tag(dt)}", *big, True)
        if hasattr(K, "_drop_workspace"):
            K._drop_workspace()
        check_flash(f"evict small {tag(dt)}", q, k, v, True)
        check_flash(f"evict big again {tag(dt)}", *big, True)


def test_workers():
    q = torch.randn(1, 513, 8, 64)
    k = torch.randn(1, 513, 8, 64)
    v = torch.randn(1, 513, 8, 64)
    threads = torch.get_num_threads()
    for w in (1, 2, 3, 0):
        A.set_num_workers(w)
        check_flash(f"workers={w}", q, k, v, True)
    A.set_num_workers(0)
    for t in (1, max(1, threads // 2), threads):
        torch.set_num_threads(t)
        check_flash(f"torch threads={t}", q, k, v, True)
    torch.set_num_threads(threads)


def test_amx_toggle():
    """AMX on -> off -> on in one process (no-op where there is no AMX)."""
    q = torch.randn(1, 257, 8, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 257, 8, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 257, 8, 64, dtype=torch.bfloat16)
    for state in ("1", "0", "1"):
        os.environ["CPU_ATTENTION_AMX"] = state
        if hasattr(K, "_AMX_ENV"):
            K._AMX_ENV = state != "0"
        check_flash(f"amx toggle {state}", q, k, v, True)
    os.environ.pop("CPU_ATTENTION_AMX", None)
    if hasattr(K, "_AMX_ENV"):
        K._AMX_ENV = True


def test_concurrent():
    cases = []
    for i in range(6):
        dt = DTYPES[i % 2]
        s = (65, 257, 513)[i % 3]
        cases.append([torch.randn(1, s, 4, 64, dtype=dt) for _ in range(3)])
    paged_cases = [paged_case(b=3, ctxs=[33, 129, 300], h=4, hk=2, d=64,
                              page=16, dt=dt, seed=7 + j)
                   for j, dt in enumerate(DTYPES)]
    errs = []

    def work(j):
        try:
            for _ in range(4):
                q, k, v = cases[j]
                got = A.flash_attn_func(q, k, v, causal=True)
                err, tol = flash_err(q, k, v, True, 64 ** -0.5, got)
                if not err <= tol:
                    errs.append(f"flash thread {j}: {err}")
                pc = paged_cases[j % 2]
                got = A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"],
                                        pc["lens"], pc["scale"])
                err = paged_err(pc, got)
                if not err <= paged_tol(pc["q"].dtype):
                    errs.append(f"paged thread {j}: {err}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"thread {j}: {e!r}")
    ts = [threading.Thread(target=work, args=(j,)) for j in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    report("concurrent callers", not errs, "; ".join(errs[:3]))


def test_two_builds():
    """Two independently loaded copies of the package, calls interleaved."""
    import arms
    a = arms.current("reg_a")
    b = arms.current("reg_b")
    q = torch.randn(1, 300, 4, 64)
    k = torch.randn(1, 300, 2, 64)
    v = torch.randn(1, 300, 2, 64)
    pc = paged_case(b=2, ctxs=[100, 37], h=4, hk=2, d=64, page=32,
                    dt=torch.float32, seed=3, pkg=a.pkg)
    pc_b = paged_case(b=2, ctxs=[100, 37], h=4, hk=2, d=64, page=32,
                      dt=torch.float32, seed=3, pkg=b.pkg)
    ok = True
    for _ in range(3):
        for arm, c in ((a, pc), (b, pc_b)):
            got = arm.pkg.flash_attn_func(q, k, v, causal=True)
            err, tol = flash_err(q, k, v, True, 64 ** -0.5, got)
            ok = ok and err <= tol
            got = arm.pkg.cross_attention(c["q"], c["kc"], c["vc"], c["bt"],
                                          c["lens"], c["scale"])
            ok = ok and paged_err(c, got) <= 1e-6
    report("two independently loaded builds", ok)


# ---------------------------------------------------------------------------
# flash_attn_with_kvcache / engine
# ---------------------------------------------------------------------------
def paged_tol(dt):
    # bf16: 1e-3 above the bf16 output-rounding floor (reference.bf16_excess).
    return 1e-6 if dt == torch.float32 else 1e-3


def paged_case(b, ctxs, h, hk, d, page, dt, seed, permute=True, poison=True,
               pkg=None):
    """A cache staged through ``pkg.reshape_and_cache`` with NaN everywhere
    a kernel must not read: unused slots of the last page, unused pages and
    the block table's padding entries (which point at a poisoned page)."""
    pkg = pkg or A
    g = torch.Generator().manual_seed(seed)
    npages = [(c + page - 1) // page for c in ctxs]
    total = sum(npages) + 1                       # + one poisoned pad page
    order = torch.randperm(total, generator=g) if permute else torch.arange(total)
    pad_page = int(order[-1])
    width = max(npages) + 1
    bt = torch.full((b, width), pad_page, dtype=torch.int32)
    nxt = 0
    for i, n in enumerate(npages):
        bt[i, :n] = order[nxt:nxt + n].to(torch.int32)
        nxt += n
    fill = float("nan") if poison else 0.0
    kc = torch.full((total, hk, page, d), fill, dtype=dt)
    vc = torch.full((total, hk, page, d), fill, dtype=dt)
    kall = [torch.randn(c, hk, d, generator=g).to(dt) for c in ctxs]
    vall = [torch.randn(c, hk, d, generator=g).to(dt) for c in ctxs]
    slots = torch.cat([bt[i, torch.arange(c) // page].long() * page +
                       torch.arange(c) % page for i, c in enumerate(ctxs)]) \
        if sum(ctxs) else torch.zeros(0, dtype=torch.int64)
    if sum(ctxs):
        pkg.reshape_and_cache(torch.cat(kall), torch.cat(vall), kc, vc, slots)
    q = torch.randn(b, h, d, generator=g).to(dt)
    lens = torch.tensor(ctxs, dtype=torch.int32)
    return dict(q=q, kc=kc, vc=vc, bt=bt, lens=lens, kall=kall, vall=vall,
                scale=d ** -0.5, page=page, hk=hk, d=d)


def paged_ref(pc, dtype=torch.float32):
    d, hk = pc["d"], pc["hk"]
    width = max([len(x) for x in pc["kall"]] + [1])
    kk = torch.zeros(len(pc["kall"]), width, hk, d)
    vv = torch.zeros_like(kk)
    for i, (k, v) in enumerate(zip(pc["kall"], pc["vall"])):
        kk[i, :len(k)] = k.float()
        vv[i, :len(v)] = v.float()
    return R.ref_decode(pc["q"], kk.transpose(1, 2).to(dtype),
                        vv.transpose(1, 2).to(dtype), dtype=dtype,
                        scale=pc["scale"], lens=pc["lens"])


def paged_err(pc, got):
    ref = paged_ref(pc)
    if pc["q"].dtype == torch.bfloat16:
        return R.bf16_excess(got, ref)
    err = (got.float() - ref).abs().max().item()
    if pc["q"].dtype == torch.float32 and err > 1e-6:
        truth = paged_ref(pc, torch.float64)
        if (ref.double() - truth).abs().max().item() > 1e-6:
            err = (got.double() - truth).abs().max().item()
    return err


def test_paged_boundaries():
    i = 0
    for page in (16, 32, 48, 64):
        for d in (32, 64, 128):
            for dt in DTYPES:
                h, hk = heads_for(i)
                i += 1
                ctxs = [page - 1, page, page + 1, 2 * page + 1, 1, 255, 257]
                pc = paged_case(len(ctxs), ctxs, h, hk, d, page, dt, seed=i)
                got = A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"],
                                        pc["lens"], pc["scale"])
                err = paged_err(pc, got)
                report(f"paged poisoned page={page} d={d} h={h}/{hk} {tag(dt)}",
                       err <= paged_tol(dt), f"err {err:.3e}")


def test_paged_store_order():
    """self_attention stores the new token of every kv head (GQA/MQA) before
    any query head of its group reads it."""
    for h, hk in ((8, 8), (8, 2), (8, 1)):
        for dt in DTYPES:
            ctxs = [31, 64, 129]
            full = paged_case(3, ctxs, h, hk, 64, 32, dt, seed=11)
            # Same data, but the last token arrives through self_attention.
            pre = paged_case(3, [c - 1 for c in ctxs], h, hk, 64, 32, dt,
                             seed=11)
            kc, vc, bt = pre["kc"], pre["vc"], full["bt"]
            # Rebuild pre's cache in full's page order so slots line up.
            kc = torch.full_like(full["kc"], float("nan"))
            vc = torch.full_like(full["vc"], float("nan"))
            page = 32
            k_old = torch.cat([k[:-1] for k in full["kall"]])
            v_old = torch.cat([v[:-1] for v in full["vall"]])
            slots_old = torch.cat([bt[i, torch.arange(c - 1) // page].long() * page
                                   + torch.arange(c - 1) % page
                                   for i, c in enumerate(ctxs)])
            A.reshape_and_cache(k_old, v_old, kc, vc, slots_old)
            kn = torch.stack([k[-1] for k in full["kall"]])
            vn = torch.stack([v[-1] for v in full["vall"]])
            slot = torch.tensor([int(bt[i, (c - 1) // page]) * page + (c - 1) % page
                                 for i, c in enumerate(ctxs)])
            got = A.self_attention(full["q"], kn, vn, kc, vc, slot, bt,
                                   full["lens"], full["scale"])
            err = paged_err(full, got)
            report(f"store order h={h}/{hk} {tag(dt)}", err <= paged_tol(dt),
                   f"err {err:.3e}")
            # And the public API with cache_seqlens = pre-append length.
            kc2 = torch.full_like(full["kc"], float("nan"))
            vc2 = torch.full_like(full["vc"], float("nan"))
            A.reshape_and_cache(k_old, v_old, kc2, vc2, slots_old)
            got = A.flash_attn_with_kvcache(
                full["q"].unsqueeze(1), kc2, vc2, k=kn.unsqueeze(1),
                v=vn.unsqueeze(1), cache_seqlens=full["lens"] - 1,
                page_table=bt, softmax_scale=full["scale"]).squeeze(1)
            err = paged_err(full, got)
            report(f"api append h={h}/{hk} {tag(dt)}", err <= paged_tol(dt),
                   f"err {err:.3e}")


def test_paged_out_guard_and_scale():
    for dt in DTYPES:
        pc = paged_case(4, [17, 64, 300, 1025], 8, 2, 64, 32, dt, seed=5)
        b, h, d = 4, 8, 64
        guard = torch.full((b * h * d + 64,), float("nan"), dtype=dt)
        out = guard[32:32 + b * h * d].view(b, h, d)
        A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"], pc["lens"],
                          pc["scale"], out=out)
        ok = bool(torch.isnan(guard[:32]).all() and torch.isnan(guard[-32:]).all())
        report(f"out= guard bands {tag(dt)}", ok)
        err = paged_err(pc, out)
        report(f"out= result {tag(dt)}", err <= paged_tol(dt), f"err {err:.3e}")
        # Changed scale catches a stale plan (a gross error). The scales stay in
        # the range the 1e-6 fp32 gate is defined for: at 0.9 the logits reach
        # ~30 and the fp32 *reference* is 6.4e-6 from fp64 (the kernels
        # 1.7-2.6e-6), so no fp32 kernel can meet 1e-6 there.
        for scale in (0.05, 0.125, 0.05, 0.1):
            pc["scale"] = scale
            got = A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"],
                                    pc["lens"], scale)
            err = paged_err(pc, got)
            report(f"paged scale {scale} {tag(dt)}", err <= paged_tol(dt),
                   f"err {err:.3e}")


def test_paged_workers():
    for w in (1, 2, 3, 0):
        A.set_num_workers(w)
        pc = paged_case(5, [1, 33, 64, 500, 129], 8, 8, 64, 16, torch.float32,
                        seed=9)
        got = A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"],
                                pc["lens"], pc["scale"])
        report(f"paged workers={w}", paged_err(pc, got) <= 1e-6)
    A.set_num_workers(0)


_SUB = r"""
import sys, torch
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
import test_regressions as T, cpu_attention as A
case = sys.argv[1]
for dt in (torch.float32, torch.bfloat16):
    if case == "neg_slot":
        # A negative slot skips that sequence's store (vLLM semantics); its
        # attention still runs over the existing cache.
        pc = T.paged_case(2, [40, 16], 4, 2, 64, 16, dt, seed=1)
        kc, vc = pc["kc"].clone(), pc["vc"].clone()
        kn = torch.randn(2, 2, 64).to(dt); vn = torch.randn(2, 2, 64).to(dt)
        got = A.self_attention(pc["q"], kn, vn, kc, vc, torch.tensor([-1, -1]),
                               pc["bt"], pc["lens"], pc["scale"])
        same = torch.equal(kc.nan_to_num(7), pc["kc"].nan_to_num(7)) and \
            torch.equal(vc.nan_to_num(7), pc["vc"].nan_to_num(7))
        err = T.paged_err(pc, got)
        print("RESULT", same and err <= T.paged_tol(dt), "cache unchanged", same,
              "err", err)
    if case == "len0":
        pc = T.paged_case(3, [40, 0, 5], 4, 4, 64, 16, dt, seed=2)
        got = A.cross_attention(pc["q"], pc["kc"], pc["vc"], pc["bt"],
                                pc["lens"], pc["scale"])
        zero = bool((got[1] == 0).all())
        err = T.paged_err(pc, got)
        print("RESULT", zero and err <= T.paged_tol(dt), "zero row", zero,
              "err", err)
"""


def test_memory_safety():
    code = _SUB.format(root=_ROOT, tests=os.path.join(_ROOT, "tests"))
    for case in ("neg_slot", "len0"):
        p = subprocess.run([sys.executable, "-c", code, case],
                           capture_output=True, text=True, timeout=300)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
        ok = p.returncode == 0 and len(lines) == 2 and \
            all(ln.split()[1] == "True" for ln in lines)
        report(f"{case} (subprocess)", ok,
               f"rc={p.returncode} {lines} {p.stderr.strip()[-200:]}")


def main():
    torch.manual_seed(0)
    print(f"tier {K.module().tier()}{' quick' if QUICK else ''}", flush=True)
    for t in (test_memory_safety, test_odd_lengths, test_unequal, test_strides,
              test_scale_and_plans, test_workers, test_amx_toggle,
              test_paged_boundaries, test_paged_store_order,
              test_paged_out_guard_and_scale, test_paged_workers,
              test_concurrent, test_two_builds):
        before = len(FAILS)
        t()
        print(f"[{'OK' if len(FAILS) == before else 'FAIL'}] {t.__name__}",
              flush=True)
    print(f"regressions: {COUNT[0] - len(FAILS)}/{COUNT[0]} passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
