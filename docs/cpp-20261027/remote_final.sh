#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cpu-attention-primary-20261027"
mkdir -p results-final
export CPU_ATTENTION_BACKEND=cpp OMP_PROC_BIND=spread OMP_PLACES=cores
if [[ $(uname -m) == aarch64 ]]; then THREADS=8; else THREADS=4; fi
export OMP_NUM_THREADS=$THREADS
.venv/bin/python scripts/build_cpp.py --test > results-final/build-test.log 2>&1
.venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900 > results-final/fuzz.log 2>&1
.venv/bin/python tests/test_paged_sweep.py > results-final/paged-parity.log 2>&1
CPU_ATTENTION_BACKEND=mojo .venv/bin/python tests/test_dense.py > results-final/mojo-dense.log 2>&1
CPU_ATTENTION_BACKEND=mojo .venv/bin/python tests/test_flash_api.py > results-final/mojo-flash.log 2>&1
if [[ $(uname -m) == aarch64 ]]; then
  .venv/bin/python tests/bench_cpp.py --threads "$THREADS" \
    --vllm "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --isa neon \
    --block 16 --rounds 61 --paged-rounds 121 --json results-final/matrix-page16.json \
    > results-final/matrix-page16.log 2>&1
else
  for dtype in bf16 fp32; do
    if [[ $dtype == bf16 ]]; then ISA=amx; REF=vllm_cpu_attn_bench_amx.so;
    else ISA=vec; REF=vllm_cpu_attn_bench.so; fi
    .venv/bin/python tests/bench_cpp.py --threads "$THREADS" --dtype "$dtype" \
      --vllm "$HOME/vllm_cpu_build/$REF" --isa "$ISA" --rounds 61 \
      --paged-rounds 121 --json "results-final/$dtype-$ISA.json" \
      > "results-final/$dtype-$ISA.log" 2>&1
  done
fi
sha256sum cpu_attention/_C.so cpu_attention/_C_cpp.so cpu_attention/csrc/*.hpp \
  "$HOME"/vllm_cpu_build/*.so > results-final/sha256.txt
lscpu > results-final/lscpu.txt
printf 'Final checkpoint validation and matrix complete\n'
