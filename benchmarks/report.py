"""Markdown tables from benchmarks/bench.py JSON files, one file per machine.

    python benchmarks/report.py --summary a.json b.json ...   # README: medians and ratios
    python benchmarks/report.py --full a.json b.json ...      # BENCHMARK.md: every arm
"""

import argparse
import json


def us(t):
    return f"{t:,.0f}" if t >= 100 else f"{t:.1f}"


def spread(t):
    return f"{us(t['median'])} ({us(t['min'])}-{us(t['max'])})"


def summary(runs):
    """Per kind and dtype: one row per shape, one column per machine."""
    machines = [r["environment"]["machine"] for r in runs]
    names = list(dict.fromkeys((x["kind"], x["name"]) for r in runs for x in r["rows"]))
    out = []
    for kind, ref in (("flash", "SDPA"), ("paged", "vLLM")):
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
                    row = next((x for x in r["rows"] if x["name"] == name and
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
        out += [f"### {e['machine']}\n",
                f"{e['cpu']}, {e['threads']} threads, `{e['git']}`, torch {e['torch']}, "
                f"{e['mojo']}; vLLM ISA bf16 `{e['vllm'].get('bf16', ['', '-'])[1]}`, "
                f"fp32 `{e['vllm'].get('fp32', ['', '-'])[1]}`.\n"]
        for kind in ("flash", "paged"):
            rows = [x for x in r["rows"] if x["kind"] == kind]
            arms = list(dict.fromkeys(a for x in rows for a in x["timing"]))
            out += ["| shape | dtype | " + " | ".join(arms) + " | ours / ref |",
                    "|---|---|" + "---:|" * (len(arms) + 1)]
            for x in rows:
                cells = [spread(x["timing"][a]) if a in x["timing"] else "-" for a in arms]
                ratio = f"{x['ratio']:.2f}" if x.get("ratio") else "-"
                out.append(f"| {x['name']} | {x['dtype']} | " + " | ".join(cells) +
                           f" | {ratio} |")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in a.files]
    print(full(runs) if a.full else summary(runs))
