#!/usr/bin/env bash
set -u
root=$HOME/arm-validation-afbea06
out=$root/reports/axion/arm-validation/gates
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
exec 9>"$HOME/bench.lock"
flock 9
uv pip install --target "$HOME/arm-validation-engine-afbea06/deps" transformers Pillow ninja > "$out/engine-deps.log" 2>&1 || exit 1
cd "$HOME/arm-validation-engine-afbea06"
export PYTHONPATH=$root:$PWD/deps:$PWD
export OMP_NUM_THREADS=8
for backend in mojo cpp; do
  CPU_ATTENTION_BACKEND=$backend python tests/test_cpu_attention.py > "$out/$backend-fastencdec.log" 2>&1
  echo "$backend fastencdec exit=$?" | tee -a "$out/status.txt"
done
