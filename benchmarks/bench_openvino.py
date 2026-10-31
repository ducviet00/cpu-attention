"""Cold-cache Python-API comparison with OpenVINO (install openvino==2026.4.1).

Run with OMP_WAIT_POLICY=PASSIVE; on a local CPU use scripts/pin_governor.sh.
Raw JSON contains every sample, runtime graph and float64 numerical checks.
"""

import argparse
import json
import os
import platform
import sys

import torch

from bench import (DT, FLASH, PAGED, PAGES, ROOT, best, cpu_attention, label,
                   machine_type, measure, sh, _kernel)
import ov_attention as ovbench

sys.path.insert(0, os.path.join(ROOT, "tests"))
from reference import attention, check


def validate(outputs, q, kv, causal=False):
    """Bound reference memory by checking one sequence and 64 query rows at a time."""
    errors = {name: dict(max_abs=0.0, strict_error=0.0) for name in outputs}
    for i, (k, v) in enumerate(kv):
        for start in range(0, q.shape[1], 64):
            end = min(start + 64, q.shape[1])
            stop = end if causal else len(k)
            ref = attention(q[i:i + 1, start:end], k[None, :stop], v[None, :stop],
                            q.shape[-1] ** -0.5, causal)
            for name, output in outputs.items():
                got = output[i:i + 1, start:end]
                assert torch.isfinite(got).all(), name
                err = (got.double() - ref).abs().max().item()
                strict, limit = check(got, ref, q.dtype, paged=q.shape[1] == 1)
                errors[name]["max_abs"] = max(errors[name]["max_abs"], err)
                errors[name]["strict_error"] = max(errors[name]["strict_error"], strict)
                errors[name]["strict_limit"] = limit
    for name, e in errors.items():
        e["strict_pass"] = e["strict_error"] <= e["strict_limit"]
        # Common comparison tolerance; retain each library's stricter result too.
        assert e["max_abs"] <= (0.03 if q.dtype == torch.bfloat16 else 2e-4), (name, e)
    return errors


def row(kind, spec, dtype, threads):
    b, s, h, d, flag = spec
    outputs, runners, fns, unsupported = {}, {}, {}, {}
    if kind == "flash":
        q, k, v = (torch.randn(b, s, h, d, dtype=dtype) for _ in range(3))
        kv = [(k[i], v[i]) for i in range(b)]
        fns["ours"] = lambda: outputs.__setitem__("ours", cpu_attention.flash_attn_func(
            q, k, v, causal=flag))
        for dynamic in (False, True):
            arm = "openvino " + ("dynamic" if dynamic else "static")
            runners[arm] = ovbench.flash(q, k, v, flag, threads, dynamic=dynamic)
    else:
        lens = s if isinstance(s, list) else [s] * b
        staged = [n - int(flag) for n in lens]
        kv = [(torch.randn(n, h, d, dtype=dtype), torch.randn(n, h, d, dtype=dtype))
              for n in staged]
        q, kn, vn = (torch.randn(b, 1, h, d, dtype=dtype) for _ in range(3))
        before = torch.tensor(staged, dtype=torch.int32)
        for page in PAGES:
            counts = [(n + page - 1) // page for n in lens]
            bt = torch.zeros(b, max(counts), dtype=torch.int32)
            for i, n in enumerate(counts):
                bt[i, :n] = torch.arange(sum(counts[:i]), sum(counts[:i]) + n)
            kc = torch.zeros(sum(counts), h, page, d, dtype=dtype)
            vc = torch.zeros_like(kc)
            for i, (k, v) in enumerate(kv):
                cpu_attention.flash_attn_with_kvcache(q[i:i + 1], kc, vc, k=k[None], v=v[None],
                                                     cache_seqlens=0, page_table=bt[i:i + 1])
            arm = f"ours p{page}"
            extra = dict(k=kn, v=vn) if flag else {}
            fns[arm] = lambda kc=kc, vc=vc, bt=bt, arm=arm, extra=extra: outputs.__setitem__(
                arm, cpu_attention.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=before,
                                                          page_table=bt, **extra))
        try:
            runners["openvino p32"] = ovbench.paged(q, kv, kn, vn, lens, flag, threads)
        except RuntimeError as e:
            if platform.machine() != "aarch64" or dtype != torch.bfloat16:
                raise
            unsupported["openvino p32"] = str(e)
        if flag:
            kv = [(torch.cat([k, kn[i]]), torch.cat([v, vn[i]])) for i, (k, v) in enumerate(kv)]
    for name, runner in runners.items():
        fns[name] = runner.run
        outputs[name] = runner.output
    # Verify before timing, not just against the other implementation.
    for fn in fns.values():
        fn()
    errors = validate(outputs, q, kv, kind == "flash" and flag)
    return fns, outputs, runners, errors, unsupported


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--dtype", choices=("both", "fp32", "bf16"), default="both")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    threads = args.threads or _kernel.physical_cores()
    torch.set_num_threads(threads)
    flush_buf = torch.zeros(2 * _kernel.llc_bytes() // 4, dtype=torch.int32)
    env = dict(machine=machine_type(), cpu=sh("lscpu").split("Model name:")[-1].split("\n")[0].strip(),
               threads=threads, git=os.getenv("CPU_ATTENTION_BENCH_GIT") or
               sh("git", "-C", ROOT, "rev-parse", "--short", "HEAD"),
               torch=torch.__version__, openvino=ovbench.ov.__version__, python=sys.version,
               llc_bytes=_kernel.llc_bytes(), omp_wait_policy=os.getenv("OMP_WAIT_POLICY"),
               governor=sh("cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
               mojo=sh(os.path.join(os.path.dirname(sys.executable), "mojo"), "--version"),
               vllm={})
    result = dict(environment=env, rows=[])
    print(json.dumps(env), flush=True)
    for kind, specs in (("flash", FLASH), ("paged", PAGED)):
        for spec in specs:
            for dt, dtype in DT.items():
                name = label(kind, spec)
                if args.only not in name or args.dtype not in ("both", dt):
                    continue
                torch.manual_seed(0)
                fns, outputs, runners, errors, unsupported = row(kind, spec, dtype, threads)
                timing = measure(fns, lambda: flush_buf.add_(1))
                ours, ref = best(timing, "ours"), best(timing, "openvino")
                data = dict(kind=kind, shape=spec, dtype=dt, ours=ours, ref=ref, timing=timing,
                            validation=errors, unsupported=unsupported,
                            runtime={n: r.runtime for n, r in runners.items()},
                            properties={n: {k: str(r.compiled.get_property(k)) for k in
                                ("INFERENCE_NUM_THREADS", "NUM_STREAMS", "INFERENCE_PRECISION_HINT")}
                                for n, r in runners.items()})
                if ref:
                    data["ratio"] = timing[ours]["median"] / timing[ref]["median"]
                result["rows"].append(data)
                print(f"{kind} {name} {dt}: " + " / ".join(
                    f"{n} {t['median']:.1f}us" for n, t in timing.items()) +
                    (f" ratio={data['ratio']:.2f}" if ref else " OpenVINO unsupported"), flush=True)
                with open(args.json, "w") as f:
                    json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
