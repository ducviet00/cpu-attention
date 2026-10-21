#!/bin/bash
# The reps=0 half of the perf matrix: what the process costs before any call.
#
#   sudo bash scripts/prof/x86/perf_baseline.sh > /tmp/base.csv
#
# `perf stat` wraps the whole process, so the matrix totals include ~3.2 G cycles
# of Python + torch start-up (and the per-shape tensor construction).  Running the
# same driver with `--reps 0 --warm 0` measures exactly that, for every
# (variant, shape, dtype); `analyze.py` divides the difference by `warm + reps`.
# Verified variant-independent: 3.18/3.19 G cycles for ours/vllm at 2048:8:64.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$PWD/.venv/bin/python}
# read before any `sudo`: the child runs as root, so ~ is not the same home
VLLM=${VLLM:-$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so}
EV="cycles,instructions,ld_blocks.store_forward,l1d.replacement,l2_rqsts.demand_data_rd_miss,mem_inst_retired.all_loads,exe_activity.bound_on_loads,uops_issued.any"

echo "variant,shape,dtype,metric,value"
for s in ${SHAPES:-2048:8:64 1024:16:128 512:8:64 128:8:64}; do
  for d in ${DTYPES:-fp32 bf16}; do
    for v in ${VARIANTS:-ours vllm sdpa}; do
      out=$(sudo perf stat -x, -e "$EV" taskset -c 3 \
            env OMP_NUM_THREADS=1 VLLM_REF="$VLLM" "$PY" scripts/prof/x86/run_one.py \
            --variant "$v" --shape "$s" --dtype "$d" --reps 0 --warm 0 2>&1)
      echo "$out" | awk -F, -v v="base_$v" -v s="$s" -v d="$d" '
        $1 ~ /^[0-9]+$/ && $3 != "" {gsub(/ /,"",$3); printf "%s,%s,%s,%s,%s\n", v, s, d, $3, $1}'
    done
  done
done
