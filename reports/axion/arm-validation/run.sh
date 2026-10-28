#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
mkdir -p "$out/gates"
date -Is > "$out/source.txt"
git rev-parse HEAD >> "$out/source.txt"
git status --short >> "$out/source.txt"
ps -eo pid,pcpu,args --sort=-pcpu | head -20 > "$out/processes.txt"
python scripts/build.py > "$out/build-mojo.log" 2>&1 || exit 1
python scripts/build_cpp.py > "$out/build-cpp.log" 2>&1 || exit 1
sha256sum cpu_attention/_C*.so >> "$out/source.txt"
for backend in mojo cpp; do
  for test in test_dense test_flash_api fuzz_dense test_paged_sweep test_regressions; do
    extra=()
    if [ "$test" = fuzz_dense ]; then extra=(--cases 80 --max-seq 900); fi
    CPU_ATTENTION_BACKEND=$backend python "tests/$test.py" "${extra[@]}" > "$out/gates/$backend-$test.log" 2>&1
    status=$?
    echo "$backend $test exit=$status" | tee -a "$out/gates/status.txt"
  done
done
python benchmarks/bench.py --manifest full --only paged --arms mojo_new,mojo_ctrl,mojo_old,cpp,vllm --rounds 41 --paged-rounds 41 --threads 8 --vllm-bf16 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-fp32 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-isa-bf16 neon --vllm-isa-fp32 neon --json "$out/paged-full-afbea06.json" > "$out/paged-full-afbea06.log" 2>&1
echo "PAGED_DONE exit=$?"
