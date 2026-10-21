#!/bin/bash
# Hardware-counter matrix: cycles / IPC / D1 / L2 / branch misses / store-forward
# stalls for one variant at a time, single-threaded on core 3.
#
#   sudo bash scripts/prof/x86/perf_matrix.sh > /tmp/perf.csv
#
# One variant per process, so every counter belongs to that variant alone.  Two
# <=8-event groups per case so nothing is multiplexed (a scaled event shows as a
# "(x.xx%)" suffix, which this script flags).
# `perf_event_paranoid=4` on this box, hence sudo; the PMU itself does work here
# (`perf record` does not -- no overflow interrupts).
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$PWD/.venv/bin/python}
# read before any `sudo`: the child runs as root, so ~ is not the same home
VLLM=${VLLM:-$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so}
G1="cycles,instructions,branches,branch-misses"
# group 2 is the mechanism: store-forwarding stalls (our P@V re-reads the
# probabilities as a 4-byte load from a 64-byte store written a few instructions
# earlier), L1 fills, L2 misses, load-port-bound cycles, issue slots.
G2="ld_blocks.store_forward,ld_blocks.no_sr,mem_inst_retired.all_loads,l1d.replacement,l2_rqsts.demand_data_rd_miss,exe_activity.bound_on_loads,uops_issued.any"

reps_for() {
  case "$1" in
    4096:8:128) echo 3 ;;
    2048:8:64)  echo 6 ;;
    1024:16:128) echo 6 ;;
    512:8:64)   echo 40 ;;
    128:8:64)   echo 120 ;;
    *)          echo 10 ;;
  esac
}

echo "variant,shape,dtype,metric,value"
run() { # variant shape dtype
  local v=$1 s=$2 d=$3 r out ev
  r=$(reps_for "$s")
  for ev in "$G1" "$G2"; do
    out=$(sudo perf stat -x, -e "$ev" taskset -c 3 \
          env OMP_NUM_THREADS=1 VLLM_REF="$VLLM" "$PY" scripts/prof/x86/run_one.py \
          --variant "$v" --shape "$s" --dtype "$d" --reps "$r" 2>&1)
    echo "$out" | grep -q '%)' && echo "WARN multiplexed $v $s $d $ev" >&2
    echo "$out" | awk -F, -v v="$v" -v s="$s" -v d="$d" '
      $1 ~ /^[0-9]+$/ && $3 != "" {
        gsub(/ /,"",$3)
        printf "%s,%s,%s,%s,%s\n", v, s, d, $3, $1
      }'
    echo "$out" | sed -n 's/.*median_us=\([0-9.]*\).*/'"$v,$s,$d"',median_us,\1/p'
    echo "$out" | grep -E "error|Error|Traceback" >&2
  done
}

SHAPES=${SHAPES:-"2048:8:64 1024:16:128 512:8:64 128:8:64"}
DTYPES=${DTYPES:-"fp32 bf16"}
VARIANTS=${VARIANTS:-"ours vllm sdpa"}
for s in $SHAPES; do
  for d in $DTYPES; do
    for v in $VARIANTS; do
      run "$v" "$s" "$d"
    done
  done
done
echo DONE >&2
