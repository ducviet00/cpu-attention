#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cpu-attention-primary-20261027"
cp cpu_attn_neon.hpp cpu_attention/csrc/cpu_attn_neon.hpp
export CPU_ATTENTION_BACKEND=cpp OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
.venv/bin/python scripts/build_cpp.py --test > results/page16-build-test.log 2>&1
.venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900 > results/page16-fuzz.log 2>&1
.venv/bin/python tests/test_paged_sweep.py > results/page16-parity.log 2>&1
ATTN_ROOT="$PWD" OLD_LIB="$PWD/cpu_attention/_C_cpp_page16_old.so" \
  NEW_LIB="$PWD/cpu_attention/_C_cpp.so" .venv/bin/python ab_public_variants.py \
  --isa neon --threads 8 --only paged --block 16 --paged-rounds 121 \
  --json results/page16-oldnew.json > results/page16-oldnew.log 2>&1
.venv/bin/python tests/bench_cpp.py --threads 8 \
  --vllm "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --isa neon \
  --only paged --block 16 --paged-rounds 121 --json results/page16-final.json \
  > results/page16-final.log 2>&1
printf 'Page16 trial complete\n'
