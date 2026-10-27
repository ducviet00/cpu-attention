#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cpu-attention-primary-20261027"
cp cpu_attention/_C_cpp.so cpu_attention/_C_cpp_pg16_old.so
cp neon-pg16-trial.hpp cpu_attention/csrc/cpu_attn_neon.hpp
export CPU_ATTENTION_BACKEND=cpp OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
.venv/bin/python scripts/build_cpp.py --test > results/pg16-build-test.log 2>&1
.venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900 > results/pg16-fuzz.log 2>&1
.venv/bin/python tests/test_paged_sweep.py > results/pg16-parity.log 2>&1
ATTN_ROOT="$PWD" OLD_LIB="$PWD/cpu_attention/_C_cpp_pg16_old.so" \
  NEW_LIB="$PWD/cpu_attention/_C_cpp.so" .venv/bin/python ab_public_variants.py \
  --isa neon --threads 8 --only paged --dtype bf16 --block 16 --paged-rounds 121 \
  --json results/pg16-oldnew.json > results/pg16-oldnew.log 2>&1
printf 'PG16 trial complete\n'
