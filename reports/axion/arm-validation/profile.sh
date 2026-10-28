#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
for dtype in bf16 fp32; do
  perf stat -e cycles,instructions,cache-misses -o "$out/perf-$dtype-stat.txt" python "$out/profile_paged.py" "$dtype" > "$out/perf-$dtype-stat.log" 2>&1
  perf record -F 499 -g -o "$out/perf-$dtype.data" -- python "$out/profile_paged.py" "$dtype" > "$out/perf-$dtype-record.log" 2>&1
  perf report -i "$out/perf-$dtype.data" --stdio --no-children --percent-limit 1 > "$out/perf-$dtype-report.txt" 2>&1
done
