"""Markdown tables from benchmarks/bench.py JSON files, one file per machine.

    python benchmarks/report.py --summary a.json b.json ...   # README: medians and ratios
    python benchmarks/report.py --full a.json b.json ...      # BENCHMARK.md: every arm
"""

import argparse
import json
from pathlib import Path


def label(kind, shape):
    """A row's shape in SDPA's legend: N batch, L query length, S key/value length,
    H heads (MHA: query and key/value heads alike), E head dim. Prefill rows mark
    `causal`; decode rows store the new token's K/V in the call unless marked
    `no append`."""
    n, s, h, e, flag = shape
    if kind == "flash":
        return f"N={n} L={s} S={s} H={h} E={e}" + (" causal" if flag else "")
    s = f"{min(s)}..{max(s)}" if isinstance(s, list) else s
    return f"N={n} L=1 S={s} H={h} E={e}" + ("" if flag else " no append")


def us(t):
    return f"{t:,.0f}" if t >= 100 else f"{t:.1f}"


def spread(t):
    return f"{us(t['median'])} ({us(t['min'])}-{us(t['max'])})"


def summary(runs):
    """Per kind and dtype: one row per shape, one column per machine."""
    machines = [r["environment"]["machine"] for r in runs]
    names = list(dict.fromkeys((x["kind"], label(x["kind"], x["shape"])) for r in runs
                               for x in r["rows"]))
    out = []
    openvino = all("openvino" in r["environment"] for r in runs)
    for kind, ref in (("flash", "OpenVINO SDPA" if openvino else "SDPA"),
                      ("paged", "OpenVINO PagedAttention" if openvino else "vLLM")):
        for dt in ("bf16", "fp32"):
            title = "Prefill, `flash_attn_func`" if kind == "flash" else \
                "Paged decode, `flash_attn_with_kvcache`"
            out += [f"**{title}, {dt}**: ours / {ref} µs, ratio\n",
                    "| shape | " + " | ".join(machines) + " |",
                    "|---|" + "---:|" * len(machines)]
            for k, name in names:
                if k != kind:
                    continue
                cells = []
                for r in runs:
                    row = next((x for x in r["rows"] if label(x["kind"], x["shape"]) == name and
                                x["dtype"] == dt), None)
                    if row is None or not row["ref"]:
                        cells.append("-")
                        continue
                    t = row["timing"]
                    mark = "ᵖ¹⁶" if row["ours"].endswith("p16") else ""
                    cells.append(f"{us(t[row['ours']]['median'])}{mark} / "
                                 f"{us(t[row['ref']]['median'])} **{row['ratio']:.2f}**")
                out.append(f"| {name} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def full(runs):
    """Per machine: every arm's median (min-max) in µs."""
    out = []
    for r in runs:
        e = r["environment"]
        backend = f"OpenVINO {e['openvino']}" if "openvino" in e else (
            f"vLLM ISA bf16 `{e['vllm'].get('bf16', ['', '-'])[1]}`, "
            f"fp32 `{e['vllm'].get('fp32', ['', '-'])[1]}`")
        out += [f"### {e['machine']}\n",
                f"{e['cpu']}, {e['threads']} threads, `{e['git']}`, torch {e['torch']}, "
                f"{e['mojo'] or 'prebuilt kernel (compiler unavailable)'}; {backend}.\n"]
        if "openvino" in e:
            checks = [v for row in r["rows"] for v in row["validation"].values()]
            out += [f"Float64 validation: {sum(v['strict_pass'] for v in checks)}/{len(checks)} "
                    "timed arms also pass the README's stricter per-operation limits.\n"]
            for row in r["rows"]:
                for arm, error in row["unsupported"].items():
                    out.append(f"- Unsupported: {label(row['kind'], row['shape'])}, "
                               f"{row['dtype']}, {arm}: `{error.strip().splitlines()[-1]}`")
                for arm, v in row["validation"].items():
                    if not v["strict_pass"]:
                        out.append(f"- {label(row['kind'], row['shape'])}, {row['dtype']}, "
                                   f"{arm}: error {v['strict_error']:.3g} exceeds strict limit "
                                   f"{v['strict_limit']:.3g} (passes the common comparison tolerance).")
            out.append("")
        for kind in ("flash", "paged"):
            rows = [x for x in r["rows"] if x["kind"] == kind]
            arms = list(dict.fromkeys(a for x in rows for a in x["timing"]))
            out += ["| shape | dtype | " + " | ".join(arms) + " | ours / ref |",
                    "|---|---|" + "---:|" * (len(arms) + 1)]
            for x in rows:
                cells = [spread(x["timing"][a]) if a in x["timing"] else "-" for a in arms]
                ratio = f"{x['ratio']:.2f}" if x.get("ratio") else "-"
                out.append(f"| {label(x['kind'], x['shape'])} | {x['dtype']} | " + " | ".join(cells) +
                           f" | {ratio} |")
            out.append("")
    return "\n".join(out)


def write_openvino(runs):
    """Refresh only the OpenVINO sections, preserving published torch/vLLM results."""
    root = Path(__file__).resolve().parents[1]
    intro = ("## OpenVINO comparison\n\n"
             "OpenVINO 2026.4.1 CPU execution, compared in the same process on identical "
             "inputs. Ratios are **ours / OpenVINO** (below 1 favors ours); ᵖ¹⁶ marks "
             "our winning page size. The existing torch/vLLM measurements above are unchanged.\n\n")
    method = ("`benchmarks/bench_openvino.py` reuses the shapes and timer from `bench.py`: "
              "3 warmups, 10 rotated/interleaved samples, a write sweep of twice LLC "
              "before every call, threads = physical cores. The reported median is the "
              "upper-middle sample, matching the existing `bench.py` convention. Ryzen's governor is pinned "
              "to performance and restored afterwards. `OMP_WAIT_POLICY=PASSIVE` avoids "
              "idle OpenMP workers competing with OpenVINO's separate CPU worker pool.\n\n"
              "OpenVINO runs synchronous `InferRequest.infer(share_outputs=True)` on a "
              "compiled one-attention graph, one stream, explicit f32/bf16 inference and "
              "cache precision, no quantization, CPU pinning and hyperthreading disabled. "
              "Torch-backed input/output buffers are bound once (bf16 is shared as raw "
              "16-bit storage), so Python/NumPy conversion, compilation, initial cache "
              "staging, and metadata construction are outside timing. OpenVINO reuses "
              "its output; ours includes output allocation, as in the existing vLLM "
              "comparison. Dense BLHD/BHLD transposes are inside the compiled graph, "
              "not pre-packed outside the timer. Paged caches are staged through each "
              "library's own writer; append rows rewrite the same final slot on every "
              "call; read-only rows use `write_kv_cache=false`. OpenVINO requires page "
              "32; ours runs both 16 and 32. SDPA runs both static and dynamic-sequence "
              "graphs and reports OpenVINO's faster arm per shape.\n\n"
              "Every row is checked against eager float64 attention before timing "
              "(64 query rows at a time to bound memory). Both outputs must be finite "
              "and within 2e-4 fp32 / 3e-2 bf16 max absolute error; raw JSON additionally "
              "records the README's stricter per-operation limits, all samples, actual "
              "runtime graphs/precisions, and unsupported-operation diagnostics. "
              "Static SDPA may execute as fused CPU subgraphs rather than a node named "
              "SDPA; the runtime graph is the authority. These are API latency results, "
              "not isolated kernel times.\n\n"
              "**Axion precision limitation:** bf16 labels describe input/output storage. "
              "OpenVINO's Axion SDPA subgraphs execute in fp32 (conversions are timed), "
              "unlike our native bf16 path. Its paged f32 fallback did not preserve "
              "writes to shared bf16 caches: the initial attempt failed float64 validation "
              "and a cache-mutation diagnostic is retained with the raw results. Those "
              "paged bf16 timings are excluded, not replaced with larger fp32 caches. "
              "Paged fp32 is compared normally.\n\n"
              "Rebuild the kernel from the current checkout with Mojo 1.2.0 using "
              "`python scripts/build.py` and run the supported-tier tests first; do not "
              "reuse an older shared object. Reproduce with "
              "`OMP_WAIT_POLICY=PASSIVE scripts/pin_governor.sh python "
              "benchmarks/bench_openvino.py --json /path/outside/repo/result.json`, then "
              "`python benchmarks/report.py --write-openvino <machine.json> ...` to "
              "regenerate both sections. Install the optional `openvino==2026.4.1` "
              "package to run this benchmark and `tests/test_benchmark_openvino.py`.\n\n")
    for relative, content in (("README.md", intro + summary(runs)),
                              ("benchmarks/BENCHMARK.md", intro + method + full(runs))):
        path = root / relative
        text = path.read_text()
        start, end = "<!-- OPENVINO RESULTS START -->", "<!-- OPENVINO RESULTS END -->"
        block = start + "\n" + content + "\n" + end
        if start in text:
            left, rest = text.split(start, 1)
            _, right = rest.split(end, 1)
            text = left + block + right
        elif relative == "README.md":
            text = text.replace("## Getting the best performance", block + "\n\n## Getting the best performance", 1)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        path.write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--write-openvino", action="store_true")
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in a.files]
    if a.write_openvino:
        write_openvino(runs)
    else:
        print(full(runs) if a.full else summary(runs))
