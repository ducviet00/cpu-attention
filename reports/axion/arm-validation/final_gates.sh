#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
python scripts/build.py --tests > "$out/build-final-standalone.log" 2>&1
echo "standalone exit=$?"
sha256sum cpu_attention/_C*.so > "$out/final-sha256.txt"
for test in test_dense test_flash_api fuzz_dense test_paged_sweep test_regressions; do
  extra=()
  if [ "$test" = fuzz_dense ]; then extra=(--cases 80 --max-seq 900); fi
  CPU_ATTENTION_BACKEND=mojo CPU_ATTENTION_ISA=scalar python "tests/$test.py" "${extra[@]}" > "$out/hilo-gates/lower-$test.log" 2>&1
  echo "lower $test exit=$?"
done
CPU_ATTENTION_BACKEND=cpp CPU_ATTENTION_ISA=scalar python tests/test_regressions.py > "$out/gates/cpp-scalar-regressions.log" 2>&1
echo "cpp scalar regressions exit=$?"
