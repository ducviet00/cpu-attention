# AGENTS.md: working on cpu-attention

Read [`README.md`](README.md) for what the library is and the current numbers.
**The README (typical shapes) and `benchmarks/BENCHMARK.md` (every arm, every
machine) are the single source of truth for numbers**: when a measurement
changes, regenerate both with `benchmarks/report.py` in the same commit, and have
`docs/*.md` and this file point at them instead of repeating a figure. Keep raw
JSON and logs outside the tree. Then read
[`docs/Mojo.md`](docs/Mojo.md) before touching `cpu_attention/_mojo/`. It ends
with a falsification ledger: do not re-run a dead idea without a new mechanism.

## Layout

```
cpu_attention/              the Python API (flash-attn's functions + reshape_and_cache), loader, worker policy
cpu_attention/_mojo/        the Mojo kernel -> _C_<tier>.so (scripts/build.py)
examples/ops.py             an engine's per-layer calls over the internal plumbing (fast-encdec's)
tests/                      pytest against tests/reference.py (numpy float64, explicit loops):
                            test_flash, test_kvcache, test_api (rejections)
benchmarks/bench.py         the benchmark: typical shapes, ours vs torch SDPA (prefill) and vLLM
                            (paged decode), at the Python API, cold caches, median of 10
benchmarks/report.py        README and BENCHMARK.md tables from bench JSON files
benchmarks/BENCHMARK.md     how the benchmark measures, and every machine's full results
scripts/build.py            builds one Mojo shared object per ISA tier; the loader picks the best
scripts/pin_governor.sh     pin the CPU governor to performance for a run on a local machine, restore it after
scripts/build_vllm_ref.sh   builds the pinned vLLM csrc/cpu reference the benchmark compares against
scripts/box_provision.sh    prepares a fresh benchmark VM (toolchain, python3-dev, uv, ~/cpu-attention venv)
scripts/box_verify.sh       on a box: the tests at every tier on a branch, then the benchmark
```

## Rules

- **The API is flash-attn's.** `cpu_attention` exports flash-attn's functions
  plus `reshape_and_cache` (vLLM's signature: the cache layout is ours, so its
  writer must be public); engine conveniences belong in `examples/`.
- **The references are torch SDPA and vLLM.** Prefill is judged against
  `F.scaled_dot_product_attention`, paged decode against vLLM's CPU attention at
  the commit pinned in `scripts/build_vllm_ref.sh` (AMX for x86 bf16 on AMX
  machines, VEC for other x86 cases, NEON on aarch64), both at the Python API
  (`benchmarks/BENCHMARK.md`).
- **One implementation across CPUs is the highest objective**, then latency
  (the model is Modular's unified matmul). Each op has one driver; a target
  differs only by compile-time constants (`Decode`, `FlashPolicy` members derived
  from `simd_width_of` / `CompilationTarget`) or a trait-plugged kernel
  (`PVKernel`, `FlashPolicy`) where the hardware or the numerics really differ.
  A refactor that removes a per-target path may cost at most about 1% of
  latency: measure on c4-, c4a- and c4d-standard-*, record the cost in the ledger, and win
  it back next.
- **No `-march=native`.** One shared object per tier, each for an explicit
  target (`scripts/build.py`); the loader picks the best tier the CPU reports.
- **Run on torch's OpenMP pool.** No private thread pool, and workers capped at
  physical cores.
- **Correctness first.** Every kernel change must pass `uv run --extra test pytest`
  at the best tier and with `CPU_ATTENTION_ISA=` each lower tier the CPU supports
  (`scripts/box_verify.sh` runs them all). Tests compare against the float64
  numpy reference only (never SDPA or vLLM); limits are in the README. Keep them
  concise: a new case is a parametrize entry, not a new script.
- **Measure the way `benchmarks/bench.py` does** (`benchmarks/BENCHMARK.md`):
  the Python API, cold caches before every call, 3 warmup + 10 samples, arms
  interleaved, the median with min and max.
  - Compare two builds only inside one process (`--compare <package dir>`); these
    machines drift 5-15% between processes.
  - On a machine with frequency scaling (a laptop or desktop) pin the governor
    to `performance` for a run and restore it after; the GCP VMs have none.
  - A change that does not beat the spread of its samples by 3-5% is reverted
    and added to the ledger.
- **Profile before changing a loop**: `perf stat` / `perf record` on one row
  (`benchmarks/bench.py --only`); c4d-standard-* exposes no PMU.
- **Code style.** Concise; comments explain *why*, with the measured number. No
  dead experiments in the tree.
