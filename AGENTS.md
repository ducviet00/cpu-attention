# AGENTS.md: working on cpu-attention

Read [`README.md`](README.md) for what the library is and the current numbers.
**The README is the single source of truth for numbers**: when a measurement
changes, update the README table in the same commit, and have `docs/*.md` and
this file point at it instead of repeating a figure. There is no `reports/`
directory: keep raw JSON and logs outside the tree, and put results only in
`README.md`, `AGENTS.md` and `docs/Mojo.md`. Then read
[`docs/Mojo.md`](docs/Mojo.md) before touching `cpu_attention/_mojo/`. It ends
with a falsification ledger: do not re-run a dead idea without a new mechanism.

## Layout

```
cpu_attention/              Python API (flash-attn names + engine aliases), loader, worker policy
cpu_attention/_mojo/        the Mojo kernel -> _C_<tier>.so (scripts/build.py)
tests/                      correctness: test_dense, test_flash_api, fuzz_dense, test_paged_sweep,
                            test_regressions (odd/unequal lengths, poisoned caches, strides, scale,
                            workers, concurrency, ...), test_amx, test_benchmark_protocol (harness
                            validation and controls); reference.py = naive references
benchmarks/manifest.py      the fixed 170-row performance manifest (do not edit it to change a score)
benchmarks/bench.py         the benchmark: Mojo vs torch SDPA (flash rows) and vLLM (paged rows), one process
benchmarks/arms.py          independently loaded copies of the package (the controls)
scripts/build.py            builds one Mojo shared object per ISA tier; the loader picks the best
scripts/pin_governor.sh     pin the CPU governor to performance for a run on a local machine, restore it after
scripts/build_vllm_ref.sh   builds the pinned vLLM csrc/cpu reference the benchmark compares against
scripts/box_provision.sh    prepares a fresh benchmark VM (toolchain, python3-dev, uv, ~/cpu-attention venv)
scripts/box_verify.sh       on a box: every gate on a branch, then the full-manifest A/B against a base
scripts/ab_summary.py       verdicts by row family, new/previous medians and losses of a bench JSON
```

## Rules

- **The references are torch SDPA and vLLM.** `flash_attn_func` rows are judged
  against `F.scaled_dot_product_attention`, paged rows against vLLM's CPU
  attention at the commit pinned in `scripts/build_vllm_ref.sh` (AMX for x86
  bf16 on AMX boxes, VEC for other x86 rows, NEON on aarch64). The acceptance
  bar is a clear per-row win (`benchmarks/bench.py` docstring).
- **One implementation across CPUs is the highest objective**, then latency
  (the model is Modular's unified matmul). Each op has one driver; a target
  differs only by compile-time constants (`Decode`, `FlashPolicy` members derived
  from `simd_width_of` / `CompilationTarget`) or a trait-plugged kernel
  (`PVKernel`, `FlashPolicy`) where the hardware or the numerics really differ.
  A refactor that removes a per-target path may cost at most about 1% of
  latency: measure on all three boxes, record the cost in the ledger, and win
  it back next.
- **No `-march=native`.** One shared object per tier, each for an explicit
  target (`scripts/build.py`); the loader picks the best tier the CPU reports.
- **Run on torch's OpenMP pool.** No private thread pool, and workers capped at
  physical cores.
- **Correctness first.** Every kernel change must pass all of these, at the
  best tier and with `CPU_ATTENTION_ISA=` each lower tier the CPU supports:
  - `tests/test_dense.py`
  - `tests/test_flash_api.py`
  - `tests/fuzz_dense.py --cases 80 --max-seq 900`
  - `tests/test_paged_sweep.py`
  - `tests/test_regressions.py` (and `tests/test_amx.py --dtype both` on AMX CPUs;
    an unavailable required kernel is a failure, never a skip)
  - fast-encdec's `tests/test_cpu_attention.py` (the engine passes strided q/k/v
    chunks of a fused projection)

  The gates are in the README.
- **Measure the way `benchmarks/bench.py` does.**
  - One process, interleaved rotated arms, the minimum, and a duplicate control.
  - When comparing two builds, the control is a *third module* (a copy of the
    old build).
  - On a machine with frequency scaling (a laptop or desktop) pin the governor
    to `performance` for a run and restore it after; the GCP VMs have none.
  - A change that does not beat the control spread by 3-5% is reverted and
    added to the ledger.
  - Rows bound by memory placement swing 20-30% with allocation order: repeat
    them with `--load-seed` before reading anything into one process.
- **Profile before changing a loop**: `perf stat` / `perf record` on one row
  (`benchmarks/bench.py --only`); the C4D exposes no PMU.
- **Code style.** Concise; comments explain *why*, with the measured number. No
  dead experiments in the tree.
