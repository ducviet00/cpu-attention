"""The 22-row dense matrix, one process per row (see tests/bench_row.py).

On the laptop a single-process sweep loses 20-50% to throttling; on the servers
`tests/bench_cpp.py` is the right tool. This one is for the primary box and for
any run whose absolute numbers must be trustworthy.

    python tests/bench_matrix.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
        --rounds 21 --json /tmp/matrix.json
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DENSE = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
         (4096, 8, 64), (256, 16, 128), (1024, 16, 128), (2048, 16, 128),
         (4096, 16, 128), (512, 32, 128), (1024, 8, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--rounds", type=int, default=41)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cooldown", type=float, default=1.0,
                    help="seconds between rows, so the core can come out of "
                         "its sustained-AVX-512 clock")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    for dtype in ("bf16", "fp32"):
        for seq, heads, dim in DENSE:
            cmd = [sys.executable, os.path.join(HERE, "bench_row.py"),
                   "--seq", str(seq), "--heads", str(heads), "--dim",
                   str(dim), "--dtype", dtype, "--rounds", str(args.rounds),
                   "--isa", args.isa]
            if args.vllm:
                cmd += ["--vllm", args.vllm]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode:
                raise RuntimeError(f"row failed ({cmd}):\n{out.stdout}\n{out.stderr}")
            line = out.stdout.strip().splitlines()
            print(line[-1] if line else out.stderr[-400:], flush=True)
            rows.append(_row_from(out.stdout))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(rows, f, indent=1)
            subprocess.run(["sleep", str(args.cooldown)])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)

    def band(key):
        v = [r[key] for r in rows if key in r]
        return f"{min(v):.3f}-{max(v):.3f}" if v else "-"

    wins_m = sum(1 for r in rows if r.get("cpp/mojo", 9) < 1.0)
    wins_v = sum(1 for r in rows if r.get("cpp/vllm", 9) < 1.0)
    print(f"\ndense: cpp/mojo {band('cpp/mojo')}  ({wins_m}/{len(rows)} wins)")
    print(f"dense: cpp/vllm {band('cpp/vllm')}  ({wins_v}/{len(rows)} wins)")
    print(f"ctrl floor: {band('ctrl/mojo')}")


def _row_from(stdout):
    """Recover the JSON row from a child's stdout (it prints one line)."""
    for line in stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {}


if __name__ == "__main__":
    main()
