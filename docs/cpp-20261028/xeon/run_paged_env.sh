#!/usr/bin/env bash
# Paged env-knob screen: one bench_cpp.py process per setting (vLLM VEC fp32 and
# Mojo are the in-process references).  VAR=... VALS="..." run_paged_env.sh
set -eo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores PYTHONPATH=$PWD
for v in $VALS; do
  echo "== $VAR=$v"
  env "$VAR=$v" .venv/bin/python tests/bench_cpp.py --threads 4 --dtype "${DT:-fp32}" \
    --isa vec --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --only paged \
    --paged-rounds "${ROUNDS:-41}" 2>&1 | grep "^paged" | grep -E "${CASES:-.}"
done
