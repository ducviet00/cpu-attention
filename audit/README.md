# Independent audit harness

These scripts produced `docs/AUDIT.md`. They deliberately do **not** import
`tests/bench_vs_refs.py`: the point is to be able to disagree with the repo's
own measurement code.

| file | what it does |
|---|---|
| `audit_common.py` | naive blockwise fp32 attention reference, round-robin timing, statistics |
| `audit_dense.py` | the 22-row dense matrix: ours vs vLLM vs SDPA, min/median, error vs the naive reference |
| `audit_paged.py` | the decoder-step matrix (heads 16, dim 64): self/cross vs vLLM, each kernel on its own cache layout |
| `audit_paged_block.py` | the same paged matrix, but per-arm best-of-40 blocks (the laptop method; power-state-robust) |
| `audit_amx_ab.py` | x86 only: AMX on/off, v1/v2, per-head vs phase-split (`mode & 512`), in one process |
| `audit_qt.py` | ARM bf16 rows-per-work-item sweep (6/8/12/16) |
| `audit_qt_fp32.py` | ARM fp32 sweep (legal values are multiples of 4) |

Requirements: the repo's `.venv` (torch + the built `_C.so`) and, for the
vLLM columns, `~/vllm_cpu_build/vllm_cpu_attn_bench.so` (+ `_amx.so`), built by
`scripts/box_sync.sh … --full` and `scripts/verify.sh`.

## Typical runs

```bash
# ARM
.venv/bin/python audit/audit_dense.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa neon --dtypes bf16,fp32 --rounds 41 --json /tmp/dense-arm.json
.venv/bin/python audit/audit_paged.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa neon --block 16 --dtypes bf16,fp32 --rounds 15 --json /tmp/paged-arm.json

# Ryzen (laptop): the paged column uses the block method; see the note below.
.venv/bin/python audit/audit_dense.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa vec --dtypes bf16,fp32 --rounds 11 --json /tmp/dense-ryzen.json
.venv/bin/python audit/audit_paged_block.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa vec --block 32 --dtypes bf16,fp32 --json /tmp/paged-ryzen.json

# x86, VEC reference
.venv/bin/python audit/audit_dense.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa vec --dtypes bf16,fp32 --rounds 41 --json /tmp/dense-vec.json
# x86, AMX reference (bf16 only; vLLM's AMX path rejects fp32)
.venv/bin/python audit/audit_dense.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench_amx.so \
    --isa amx --dtypes bf16 --rounds 41 --json /tmp/dense-amx.json
```

`--rounds 41` is what the contested rows need; 9–11 is enough for a band on the
rows that are not near 1.00. The duplicate `ctrl` arm is the noise floor: on the
audit boxes it read 0.97–1.04 (m9g) and 0.92–1.04 (m8i), so anything inside that
is not resolvable.

## Laptop vs server

`audit_paged.py` interleaves single calls; on a laptop whose governor is
`powersave` the clock state changes between calls and the median (and even the
min) is dominated by the power state. `audit_paged_block.py` times 40
consecutive calls per arm after 8 warmups, alternating blocks, which is the
method the old laptop table used and is what the README's Ryzen paged column
reports. The interleaved file (`paged-perf.json`) is kept for comparison; the
server boxes do not need it.

## Two traps this harness exists to avoid

* **The vLLM page layouts are not ours.** On bf16 the two implementations pack
  K/V differently on both ISAs (ours pairs 2 dims / 2 positions for
  `dpbf16ps`/`bfdot`; vLLM groups 4 for AMX/`bfmmla`). `audit_paged.py` stages
  a separate cache with each implementation's own `reshape_and_cache`; reading
  one kernel's cache with the other silently returns garbage (max abs diff
  0.26–1.28) while still timing normally.
* **SDPA is not a reference, it is a second approximation.** The repo harness's
  bf16 gate compares two bf16 kernels; it fired at 1024:8:64 on m8i (3.12e-2)
  while both kernels were within 9e-3 of the naive fp32 reference. Always keep
  the `d(ours)`/`d(vllm)`/`d(sdpa)` columns from `audit_dense.py` beside the
  timing.

## Known harness limitation

`audit_amx_ab.py --which onoff` toggles `_kernel._AMX_ENV` between calls. On
`ff2bfdd` that can SIGSEGV after a few shapes: `_drop_workspace()` frees the
AMX workspace but does not clear `_AMX_PLANS`, so a later AMX call uses a freed
pointer (`_kernel.py:844`). Run `--which onoff` one shape per process, or use
the `v1v2`/`phase` arms (which are clean) until the plan-cache invalidation is
fixed.
