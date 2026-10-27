# AGENTS.md: working on cpu-attention

Read [`README.md`](README.md) for what the library is and the current numbers.
Then read [`docs/CPP.md`](docs/CPP.md) before touching `cpu_attention/csrc/`,
or [`docs/Mojo.md`](docs/Mojo.md) before touching `cpu_attention/_mojo/`. Both
docs end with a falsification ledger: do not re-run a dead idea without a new
mechanism.

## Layout

```
cpu_attention/              Python API (flash-attn names + engine aliases), loader, worker policy
cpu_attention/csrc/         C++ backend  -> _C_cpp.so  (scripts/build_cpp.py)
cpu_attention/_mojo/        Mojo backend -> _C.so      (scripts/build.py)
tests/                      correctness: test_dense, test_flash_api, fuzz_dense, test_paged_sweep, test_amx;
                            reference.py = naive fp32 references + timing helpers
benchmarks/bench.py         the benchmark: C++ vs Mojo vs vLLM vs SDPA, one process
scripts/build_vllm_ref.sh   builds the pinned vLLM csrc/cpu reference the benchmark compares against
```

## Rules

- **The Mojo backend is the oracle and the default.** Keep it building and
  passing. Flip the default to C++ only when every box meets the acceptance bar
  in `docs/CPP.md` §1.
- **File ownership in the C++ backend.**
  - The shared core is `cpu_attn_impl.hpp`, the entry points
    (`cpu_attn.cpp/.hpp`) and `binding.cpp`.
  - ISA work stays in its own policy file (`cpu_attn_vec.hpp`,
    `cpu_attn_amx.hpp`, `cpu_attn_neon.hpp`).
  - A policy reaches the driver through a detected optional hook, not a new
    branch in the driver.
- **No `-march=native`.** Keep the per-TU ISA flags and runtime dispatch.
- **Run on torch's OpenMP pool.** No private thread pool, and workers capped at
  physical cores.
- **Correctness first.** Every kernel change must pass all of these, with
  `CPU_ATTENTION_BACKEND=cpp` (and without it for Mojo):
  - `tests/test_dense.py`
  - `tests/test_flash_api.py`
  - `tests/fuzz_dense.py --cases 80 --max-seq 900`
  - `tests/test_paged_sweep.py`
  - fast-encdec's `tests/test_cpu_attention.py` (the engine passes strided q/k/v
    chunks of a fused projection)

  The gates are in the README.
- **Measure the way `docs/CPP.md` §7 says.**
  - One process, interleaved rotated arms, the minimum, and a duplicate control.
  - When comparing two builds, the control is a *third module* (a copy of the
    old build).
  - Pin the Ryzen governor to `performance` for a run and restore it after.
  - Compare against the right vLLM ISA: AMX for x86 bf16 on AMX boxes, VEC for
    other x86 rows, NEON on aarch64.
  - A change that does not beat the control spread by 3-5% is reverted and
    added to the ledger.
- **Profile before changing a loop.**
  - The in-situ phase ablation knobs are `CPU_ATTENTION_{VEC,NEON,AMX,PAGED}_MODE`.
  - Then `perf stat` / `perf record`.
- **Code style.** Concise; comments explain *why*, with the measured number. No
  dead experiments in the tree.
