"""Render the C++ acceptance scorecard from complete, unedited benchmark JSON."""

import argparse
import json
from pathlib import Path

DENSE = {(s, h, d) for s, h, d in (
    (128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
    (4096, 8, 64), (256, 16, 128), (1024, 16, 128), (2048, 16, 128),
    (4096, 16, 128), (512, 32, 128), (1024, 8, 32))}
PAGED = {(b, n) for b, n in ((1, 256), (8, 256), (8, 1024),
                            (32, 1024), (1, 4096))}


def identity(row):
    if row["kind"] == "dense":
        return ("dense", row["dtype"], row["seq"], row["heads"], row["dim"])
    return ("paged", row["dtype"], row["op"], row["batch"], row["ctx"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", action="append", required=True,
                    help="label=file.json[,file.json] (one complete 42-row box)")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="fail unless every box has >=90%% clear wins and no loss >4%%")
    args = ap.parse_args()
    expected = {(kind, dt, *shape) for dt in ("bf16", "fp32")
                for kind, shapes in (("dense", DENSE), ("paged", {
                    (op, b, n) for op in ("self", "cross") for b, n in PAGED}))
                for shape in shapes}
    text = ["# C++ acceptance scorecard", "",
            "Generated from raw JSON. Ratios are C++ / baseline; below 1 wins.",
            "Clear win: `<0.97` of **both** baselines; noise band: `0.97–1.04`.",
            "Times are µs, **minimum (median)**. `ctrl/cpp` is the duplicate arm.", ""]
    accepted = True
    for spec in args.box:
        label, files = spec.split("=", 1)
        paths = [Path(p) for p in files.split(",")]
        rows = [r for p in paths for r in json.loads(p.read_text())]
        keys = [identity(r) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"{label}: expected exactly 22 dense + 20 paged rows")
        for r in rows:
            for field in ("cpp_us", "cpp_med", "mojo_us", "mojo_med",
                          "vllm_us", "vllm_med", "ctrl_us", "ctrl_med"):
                if not r.get(field, 0) > 0:
                    raise ValueError(f"{label}: missing/invalid {field}: {identity(r)}")
            if any(len(r.get("samples_us", {}).get(arm, [])) < 41
                   for arm in ("cpp", "ctrl", "mojo", "vllm")):
                raise ValueError(f"{label}: fewer than 41 raw samples: {identity(r)}")
        ratios = [r["cpp_us"] / min(r["mojo_us"], r["vllm_us"]) for r in rows]
        wins = sum(v < 0.97 for v in ratios)
        noise = sum(0.97 <= v <= 1.04 for v in ratios)
        losses = sum(v > 1.04 for v in ratios)
        passed = wins >= 0.9 * len(rows) and losses == 0
        accepted &= passed
        text += [f"## {label}", "",
                 f"**{wins}/42 clear wins; {noise} noise-band rows; {losses} losses. "
                 f"Acceptance: {'PASS' if passed else 'NOT MET'}.**", "",
                 "Sources: " + ", ".join(f"`{p.name}`" for p in paths), "",
                 "| row | dtype | C++ | Mojo | vLLM | c/best | ctrl/cpp |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for r, ratio in zip(rows, ratios):
            name = (f"dense {r['seq']}:{r['heads']}:{r['dim']}" if r["kind"] == "dense"
                    else f"{r['op']} b={r['batch']} ctx={r['ctx']}")
            times = [f"{r[be + '_us']:.1f} ({r[be + '_med']:.1f})"
                     for be in ("cpp", "mojo", "vllm")]
            text.append(f"| {name} | {r['dtype']} | {' | '.join(times)} | "
                        f"{ratio:.3f} | {r['ctrl_us'] / r['cpp_us']:.3f} |")
        text.append("")
    rendered = "\n".join(text) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return int(args.check and not accepted)


if __name__ == "__main__":
    raise SystemExit(main())
