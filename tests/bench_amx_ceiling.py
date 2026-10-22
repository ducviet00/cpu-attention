"""Build and run the ceiling probe (`bench_amx_ceiling.mojo`), interpret it.

    .venv/bin/python tests/bench_amx_ceiling.py

Absolute numbers per core (dots/s, ns/logit), so a separate process per
measurement is fine here: nothing is being compared across arms.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cpu_attention", "_mojo", "src")
TESTS = os.path.join(ROOT, "cpu_attention", "_mojo", "tests")
BUILD = os.path.join(ROOT, "cpu_attention", "_mojo", "build")

DOT_FLOPS = 16 * 16 * 32 * 2  # one tdpbf16ps: 16x16 out, 32 contraction
# units produced by one iteration, per thread
UNITS = {
    0: ("dot", 16 * 64),
    1: ("dot", 16 * 64),
    2: ("dot", 16 * 64),
    3: ("dot", 16 * 64),
    4: ("load", 4 * 64),
    5: ("dot", 4 * 64),
    7: ("dot", 64),
    9: ("logit", 32 * 256),
    10: ("logit", 32 * 256),
    11: ("logit", 32 * 256),
    12: ("logit", 32 * 256),
    13: ("logit", 32 * 256),
    14: ("logit", 32 * 128),
    15: ("logit", 32 * 1024),
    16: ("logit", 32 * 128),
    20: ("logit", 8 * 1024),
    30: ("mix", 64),
    31: ("mix", 64),
    32: ("mix", 64),
}
MODES = {
    0: "dots only (no operands)",
    1: "16 dots + 16 loads, fixed L1 lines",
    2: "16 dots + 16 loads, B walking 32 KB",
    3: "mode 2 + 4 tile stores (logits)",
    4: "16 tile loads, no dots",
    5: "A in tiles: 4 dots + 4 loads + 2 stores / 32 keys",
    7: "1 dependent dot chain (latency)",
    9: "softmax 32x256, 1 row in flight",
    10: "softmax 32x256, 2 rows in flight",
    11: "softmax 32x256, 4 rows in flight",
    12: "mode 11 + running max",
    13: "mode 11, degree-2 poly",
    14: "mode 11 at kt=128 (L1-resident)",
    15: "mode 11 at kt=1024 (L2)",
    16: "mode 12 at kt=128",
    20: "kernel-shaped QK loop (8 chunks, 4 stores)",
    30: "4 dots + 8 exp2 per group (overlap test)",
    31: "8 exp2 alone",
    32: "4 dots + 96 vector FMA (overlap test)",
}


def libgomp_flags() -> list[str]:
    """Same lookup as scripts/build.py: the standalone binary links libgomp."""
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(os.path.join(lib, "libgomp.so.1")):
            return ["-Xlinker", "-l:libgomp.so.1", "-Xlinker", f"-L{lib}"]
    except ImportError:
        pass
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64"):
        if os.path.exists(os.path.join(d, "libgomp.so")):
            return ["-Xlinker", "-lgomp"]
    return []


def build() -> str:
    mojo = shutil.which("mojo") or os.path.join(ROOT, ".venv", "bin", "mojo")
    if not os.path.exists(mojo):
        raise SystemExit("mojo not found: run `uv sync --extra build`")
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, "bench_amx_ceiling")
    subprocess.run(
        [mojo, "build", "-I", SRC, os.path.join(TESTS, "bench_amx_ceiling.mojo"),
         "-o", out, *libgomp_flags()],
        check=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4, help="OMP_NUM_THREADS cap")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    exe = os.path.join(BUILD, "bench_amx_ceiling")
    if not args.no_build:
        exe = build()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    res = subprocess.run([exe], capture_output=True, text=True, env=env)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-3000:])
        return res.returncode
    rows = []
    for line in res.stdout.splitlines():
        m = re.match(r"RESULT mode=\s*(\d+)\s+threads=\s*(\d+)\s+iters=\s*(\d+)"
                     r"\s+ns=\s*(\d+)\s+kt=\s*(\d+)", line)
        if m:
            rows.append(tuple(int(x) for x in m.groups()))
    print("\nPer core. `dots/cycle` assumes the AMX pipe runs at the nominal "
          "2.3 GHz;\nthe absolute ns/dot is the number the kernel is judged "
          "against.")
    print(f"{'mode':>4} {'thr':>4} {'ns/unit':>9} {'units/ns':>9} "
          f"{'TFLOP/s/core':>13}  what")
    for mode, threads, iters, ns, kt in rows:
        unit, per_iter = UNITS[mode]
        total = per_iter * iters
        ns_unit = ns / total
        if mode in (30, 31, 32):
            print(f"{mode:>4} {threads:>4} {ns_unit:>9.4f} {1 / ns_unit:>9.1f} "
                  f"{'-':>13}  {MODES[mode]}")
            continue
        if mode == 20:
            # ns per logit, and the equivalent ns per dot (64 logits per dot)
            print(f"{mode:>4} {threads:>4} {ns_unit:>9.4f} {1 / ns_unit:>9.1f} "
                  f"{ns_unit * 64:13.2f}  {MODES[mode]}  A stride {kt} B")
            continue
        if unit == "dot":
            tfl = f"{DOT_FLOPS / ns_unit / 1e9:13.2f}"
        elif unit == "logit":
            tfl = f"{'-':>13}"
        else:
            tfl = f"{'-':>13}"
        print(f"{mode:>4} {threads:>4} {ns_unit:>9.4f} {1 / ns_unit:>9.1f} "
              f"{tfl}  {MODES[mode]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
