"""Dense rows: the 2026-10-27 Axion matrix against a bench_cpp JSON (c/best)."""
import json
import sys

old = {(r["seq"], r["heads"], r["dim"], r["dtype"]): r
       for r in json.load(open(sys.argv[1])) if r["kind"] == "dense"}


def best(x):
    return min(x["mojo_us"], x.get("vllm_us", 1e9))


for r in json.load(open(sys.argv[2])):
    k = (r["seq"], r["heads"], r["dim"], r["dtype"])
    o = old[k]
    print(f"| {k[0]}:{k[1]}:{k[2]} | {k[3]} | {o['cpp_us']:.1f} "
          f"({o['cpp_med']:.1f}) | {o['cpp_us'] / best(o):.3f} | "
          f"{r['cpp_us']:.1f} ({r['cpp_med']:.1f}) | {r['mojo_us']:.1f} | "
          f"{r.get('vllm_us', 0):.1f} | {r['cpp_us'] / best(r):.3f} | "
          f"{r['ctrl_us'] / r['cpp_us']:.3f} |")
