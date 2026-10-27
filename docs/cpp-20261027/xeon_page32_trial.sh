#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cpu-attention-primary-20261027"
cp cpu_attention/_C_cpp.so cpu_attention/_C_cpp_page32_old.so
cp cpu_attn_vec.hpp cpu_attention/csrc/cpu_attn_vec.hpp
export CPU_ATTENTION_BACKEND=cpp OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
.venv/bin/python scripts/build_cpp.py --test > results/page32-build-test.log 2>&1
.venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900 > results/page32-fuzz.log 2>&1
.venv/bin/python tests/test_paged_sweep.py > results/page32-parity.log 2>&1
ATTN_ROOT="$PWD" OLD_LIB="$PWD/cpu_attention/_C_cpp_page32_old.so" \
  NEW_LIB="$PWD/cpu_attention/_C_cpp.so" .venv/bin/python ab_public_variants.py \
  --isa amx --threads 4 --only paged --block 32 --paged-rounds 121 \
  --json results/page32-oldnew.json > results/page32-oldnew.log 2>&1
for dtype in bf16 fp32; do
  if [[ $dtype == bf16 ]]; then ISA=amx; REF=vllm_cpu_attn_bench_amx.so;
  else ISA=vec; REF=vllm_cpu_attn_bench.so; fi
  .venv/bin/python tests/bench_cpp.py --threads 4 --dtype "$dtype" \
    --vllm "$HOME/vllm_cpu_build/$REF" --isa "$ISA" --only paged \
    --block 32 --paged-rounds 121 --json "results/page32-$dtype-final.json" \
    > "results/page32-$dtype-final.log" 2>&1
done
printf 'Xeon page32 trial complete\n'
