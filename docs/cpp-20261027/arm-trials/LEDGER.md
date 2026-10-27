# NEON ISA campaign, 2026-10-27

Hardware: existing Google Axion c4a-standard-8, 8 physical Neoverse-V2 cores,
64 KiB L1D/core, 2 MiB L2/core, 80 MiB L3. **Not Graviton.** No VM was created
or removed. SSH endpoint `ducviet00@34.87.120.65`.

Source: `a1d9fdf42c3dc9cfbcadea5ffc45d8fa9ed16168`; independent local worktree
`/tmp/opencode/cpu-attention-arm-isa1`, branch `cpp-arm-20261027-isa1`.
Only `cpu_attention/csrc/cpu_attn_neon.hpp` is changed. Remote isolated directory
`/home/ducviet00/cpu-attention-arm-isa1-20261027`; previous snapshots untouched.
The venv is a symlink to the existing `~/cpu-attention/.venv`; the oracle is a
private copied `_C.so`, SHA256
`25987fce987a72a8245654bbafc2c7f6615c612bf38955d2be1503c3dc9af312`.
No Mojo source changes or default changes. No local CPU benchmarks.

The remote vLLM tree has no `.git`. Its cpu_attn.cpp, cpu_attn_neon.hpp,
cpu_attn_impl.hpp, cpu_isa.cpp and utils.cpp were downloaded from GitHub at
`db3b0f0` and compared byte-for-byte: all MATCH. The reused NEON reference SO:
`~/vllm_cpu_build/vllm_cpu_attn_bench.so`, SHA256
`8d6da4cc469faacf603e128281a9ceae1a57b637642caff537738892f9fe8e13`.

## Profiling before loop changes

`perf stat -e cycles,instructions true` fails: hardware events unsupported by
the VM. No hardware-cycle or IPC claims are made. Software `cpu-clock:u`
sampling works. bf16 S=1024/H=8/D=64: QK 33.46%, PV 26.36%, shared driver
including inlined softmax 27.06%; OpenMP barrier symbols ~4.8%. QK disassembly
shows ten live row maxima and repeated stack vector traffic. PV samples cluster
on loads; V dimension-pair addresses are separated by the whole padded sequence.
Paged fp32 b1/4096: attend_head 83.16%, OpenMP barriers ~9.4%; hot QK loads have
runtime page-stride address arithmetic. Annotated text is archived beside JSON.

Commands (from remote isolated directory):

```bash
OMP_PROC_BIND=spread CPU_ATTENTION_ISA=neon CPU_ATTENTION_BACKEND=cpp \
perf record -e cpu-clock:u -F 499 -o profile-bf16.data -- .venv/bin/python -c \
'import torch,cpu_attention as a; torch.set_num_threads(8); q,k,v=[torch.randn(1,1024,8,64,dtype=torch.bfloat16) for _ in range(3)]; [a.flash_attn_func(q,k,v,causal=True) for i in range(1800)]'
OMP_PROC_BIND=spread CPU_ATTENTION_ISA=neon CPU_ATTENTION_BACKEND=cpp \
perf record -e cpu-clock:u -F 499 -o profile-paged-fp32.data -- \
.venv/bin/python profile_paged.py
perf report -i profile-bf16.data --stdio --no-children --percent-limit 1
```

## Experiments and failures

All ratios are **new/old**, lower is faster. Dense trials have 61 interleaved
rounds; paged trials have 121, with duplicate new control. `ab_driver.py` invokes
the existing corrected `tests/bench_cpp.py` and switches independent extension,
bound-entry, dense-plan and paged-plan caches. Its `mojo` label means **old C++**,
not the Mojo oracle; `comparison` and `new/old` fields make this explicit.
`final-matrix.json` uses the unmodified harness with the actual Mojo oracle.

| Trial | Result | Decision |
|---|---|---|
| Move bf16 mask/max outside QK, following the existing fp32 path | 0.980 short; 0.990–0.995 D=32/64; 1.005–1.016 D=128 long | Reverted completely: saved registers did not repay the extra logits pass. |
| 64-key blocked bf16 V, initially on all head dimensions | Combined with previous trial: 0.950 at 4096:16:128, 0.972 D=32, but D=64 long 1.003–1.008 | Investigated separately before retaining. |
| V blocking alone, original QK restored | 0.946 at 4096:16:128, 0.979 D=32, D=64 long 1.007–1.013 | Retain only D=32/128; D=64 uses its original unblocked V. |
| Page-32 compile-time fp32 paged specialization; arithmetic and cache layout unchanged | Corrected 121-round run: 0.810–0.987, long b1 0.846–0.872, b32 0.914–0.925 | Retained; other page sizes use general fallback. |
| Same page-32 specialization for bf16 paged | 0.724–0.978; b8/1024 0.724–0.738, b32 0.901–0.903 | Retained. |

The first paged A/B driver did not isolate `_PAGED_PLANS`; closures could capture
the wrong extension. Its raw file is explicitly named
`trial3-invalid-shared-paged-cache.json` and **is not evidence**. The corrected
121-round rerun is `trial3-ab.json`; dense trials were unaffected. A first paged
profile command also had a shell quoting error and produced no valid profile;
the archived `profile_paged.py` invocation is the corrected one.

## Build and gates

Compiler: Ubuntu GCC 15.2.0-16ubuntu1. Standard build, no `-march=native`:
`-O3 -fPIC -std=c++20 -fopenmp -Wall -Wextra -Wno-unused-parameter`, and
`-march=armv8.2-a+bf16+i8mm` for neon.cpp only.

Every trial ran dense 82 checks, flash API, dense fuzz 80/max900 + 24 stray-read
probes, and strict paged 504 checks before A/B. The unchanged strict reference
precision resolution is reported, not hidden: HD=128/page=32/ctx=33 has naive
fp32 error 1.831e-6 against fp64, while C++ is 2.496e-7 against fp64. Strict
paged worst other fp32 error is 8.05e-7; worst bf16-to-Mojo is 1.92 ulp.

```bash
.venv/bin/python scripts/build_cpp.py --test
CPU_ATTENTION_BACKEND=cpp CPU_ATTENTION_ISA=neon .venv/bin/python \
    tests/fuzz_dense.py --cases 80 --max-seq 900
CPU_ATTENTION_BACKEND=cpp CPU_ATTENTION_ISA=neon .venv/bin/python \
    tests/test_paged_sweep.py

# Old SO was saved immediately after building untouched a1d9fdf.
ARM_ROOT=$PWD OMP_PROC_BIND=spread CPU_ATTENTION_ISA=neon \
    .venv/bin/python ab_driver.py --isa neon --threads 8 --rounds 61 \
    --paged-rounds 121 --json final-oldnew-ab.json

# Both C++ and Mojo allocate output, vLLM follows the established staged-cache
# and preallocated-output convention. Includes all 22 dense + 20 paged rows.
OMP_PROC_BIND=spread CPU_ATTENTION_ISA=neon .venv/bin/python tests/bench_cpp.py \
    --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa neon --threads 8 \
    --rounds 61 --paged-rounds 121 --json final-matrix.json
```

The final commands above started on the original `a1d9fdf` Python/shared headers;
the primary's later `90d705c` FASTCALL changes are **not** included. The primary
should merge the ISA-only commit and rerun its integrated final gates/matrix.
