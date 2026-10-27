#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cpu-attention-primary-20261027"
cp cpu_attention/_C_cpp.so cpu_attention/_C_cpp_softmax4_old.so
cp amx-softmax8-trial.hpp cpu_attention/csrc/cpu_attn_amx.hpp
export CPU_ATTENTION_BACKEND=cpp OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
.venv/bin/python scripts/build_cpp.py --test > results/softmax8-build-test.log 2>&1
.venv/bin/python tests/fuzz_dense.py --cases 80 --max-seq 900 > results/softmax8-fuzz.log 2>&1
.venv/bin/python "$HOME/amx-results/ab_headers.py" \
  --old "$PWD/cpu_attention/_C_cpp_softmax4_old.so" \
  --new "$PWD/cpu_attention/_C_cpp.so" --rounds 120 \
  --json "$PWD/results/softmax8-oldnew.json" > results/softmax8-oldnew.log 2>&1
printf 'Softmax8 trial complete\n'
