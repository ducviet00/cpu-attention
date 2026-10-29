# AGENTS.md: working on cpu-attention

Read [`README.md`](README.md) for what the library is and the current numbers.
**The README is the single source of truth for numbers**: when a measurement
changes, update the README table in the same commit, and have `docs/*.md` and
this file point at it instead of repeating a figure. There is no `reports/`
directory: keep raw JSON and logs outside the tree, and put results only in
`README.md`, `AGENTS.md`, `docs/Mojo.md` and `docs/CPP.md`. Then read [`docs/CPP.md`](docs/CPP.md) before touching `cpu_attention/csrc/`,
or [`docs/Mojo.md`](docs/Mojo.md) before touching `cpu_attention/_mojo/`. Both
docs end with a falsification ledger: do not re-run a dead idea without a new
mechanism.

## Layout

```
cpu_attention/              Python API (flash-attn names + engine aliases), loader, worker policy
cpu_attention/csrc/         C++ backend  -> _C_cpp.so  (scripts/build_cpp.py)
cpu_attention/_mojo/        Mojo backend -> _C.so      (scripts/build.py)
tests/                      correctness: test_dense, test_flash_api, fuzz_dense, test_paged_sweep,
                            test_regressions (odd/unequal lengths, poisoned caches, strides, scale,
                            workers, concurrency, ...), test_amx, test_benchmark_protocol (harness
                            validation and controls); reference.py = naive references
benchmarks/manifest.py      the fixed 170-row performance manifest (do not edit it to change a score)
benchmarks/bench.py         the benchmark: unified Mojo vs every C++ ISA path vs vLLM vs SDPA, one process
benchmarks/arms.py          independently loaded copies of the package (the controls)
scripts/build.py            builds one Mojo shared object per ISA tier; the loader picks the best
scripts/pin_governor.sh     pin the CPU governor to performance for a run on a local machine, restore it after
scripts/build_vllm_ref.sh   builds the pinned vLLM csrc/cpu reference the benchmark compares against
```

## Rules

- **The Mojo backend is the default and the one being optimized.** The C++
  backend is the performance competitor: its compute code stays unchanged. The
  acceptance bar is a clear per-row win of Mojo over the *fastest numerically
  valid* C++ path (every ISA the CPU supports), judged as in `docs/Mojo.md` §7.
  The previous Mojo build is kept as the oracle at git tag `mojo-oracle-v1.1`.
- **File ownership in the C++ backend.**
  - The shared core is `cpu_attn_impl.hpp`, the entry points
    (`cpu_attn.cpp/.hpp`) and `binding.cpp`.
  - ISA work stays in its own policy file (`cpu_attn_vec.hpp`,
    `cpu_attn_amx.hpp`, `cpu_attn_neon.hpp`).
  - A policy reaches the driver through a detected optional hook, not a new
    branch in the driver.
- **No `-march=native`.** C++ keeps its per-TU ISA flags and runtime dispatch;
  Mojo builds one shared object per tier for an explicit target
  (`scripts/build.py`) and the loader picks the best tier the CPU reports.
- **Run on torch's OpenMP pool.** No private thread pool, and workers capped at
  physical cores.
- **Correctness first.** Every kernel change must pass all of these, with
  `CPU_ATTENTION_BACKEND=mojo` (and `cpp` for the competitor), and for Mojo also
  with `CPU_ATTENTION_ISA=` each lower tier the CPU supports:
  - `tests/test_dense.py`
  - `tests/test_flash_api.py`
  - `tests/fuzz_dense.py --cases 80 --max-seq 900`
  - `tests/test_paged_sweep.py`
  - `tests/test_regressions.py` (and `tests/test_amx.py --dtype both` on AMX CPUs;
    an unavailable required kernel is a failure, never a skip)
  - fast-encdec's `tests/test_cpu_attention.py` (the engine passes strided q/k/v
    chunks of a fused projection)

  The gates are in the README.
- **Measure the way `docs/CPP.md` §7 says.**
  - One process, interleaved rotated arms, the minimum, and a duplicate control.
  - When comparing two builds, the control is a *third module* (a copy of the
    old build).
  - On a machine with frequency scaling (a laptop or desktop) pin the governor
    to `performance` for a run and restore it after; the GCP VMs have none.
  - Compare against the right vLLM ISA: AMX for x86 bf16 on AMX boxes, VEC for
    other x86 rows, NEON on aarch64.
  - A change that does not beat the control spread by 3-5% is reverted and
    added to the ledger.
  - Rows bound by memory placement swing 20-30% with allocation order: repeat
    them with `--load-seed` before reading anything into one process.
- **Profile before changing a loop.**
  - The in-situ phase ablation knobs are `CPU_ATTENTION_{VEC,NEON,AMX,PAGED}_MODE`.
  - Then `perf stat` / `perf record`.
- **Code style.** Concise; comments explain *why*, with the measured number. No
  dead experiments in the tree.
