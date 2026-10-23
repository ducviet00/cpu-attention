"""A/B an env knob in alternating processes, and judge on the median of mins.

Short calls on the laptop are dominated by the power state, so two in-process
arms can disagree by 30% while the same arm repeats within 2% on the servers.
Alternating whole processes and taking the median of the per-process minima
divides that out, at the cost of wall time.

    python tests/ab_env.py --knob CPU_ATTENTION_VDIRECT --a 0 --b 1 \
        --rows 512:8:64:fp32 2048:16:128:fp32 --rounds 15

Each row is measured in a fresh process per arm; the arms alternate
A B A B A B so a monotonic drift cannot favour either one.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def one(row, knob, val, rounds, vllm, isa, threads):
    seq, heads, dim, dtype = row.split(":")
    env = dict(os.environ)
    if knob:
        env[knob] = str(val)
    cmd = [sys.executable, os.path.join(HERE, "bench_row.py"),
           "--seq", seq, "--heads", heads, "--dim", dim, "--dtype", dtype,
           "--rounds", str(rounds), "--isa", isa]
    if vllm:
        cmd += ["--vllm", vllm]
    if threads:
        cmd += ["--threads", str(threads)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"no JSON from {row} {knob}={val}:\n{out.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="")
    ap.add_argument("--a", default="0")
    ap.add_argument("--b", default="1")
    ap.add_argument("--rows", nargs="+", required=True,
                    help="seq:heads:dim:dtype")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--vllm", default=os.environ.get("VLLM_REF", ""))
    ap.add_argument("--isa", default=os.environ.get("VLLM_ISA", "vec"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    print(f"{'row':>20} {'a min':>9} {'b min':>9} {'a/b':>6}  per-pair", flush=True)
    for row in args.rows:
        aa, bb, pairs = [], [], []
        for _ in range(args.pairs):
            ra = one(row, args.knob, args.a, args.rounds, args.vllm, args.isa,
                     args.threads)
            rb = one(row, args.knob, args.b, args.rounds, args.vllm, args.isa,
                     args.threads)
            aa.append(ra["cpp_us"])
            bb.append(rb["cpp_us"])
            pairs.append(f"{ra['cpp_us'] / rb['cpp_us']:.3f}")
        a, b = statistics.median(aa), statistics.median(bb)
        print(f"{row:>20} {a:9.1f} {b:9.1f} {a / b:6.3f}  {' '.join(pairs)}",
              flush=True)


if __name__ == "__main__":
    main()
