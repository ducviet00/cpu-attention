#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
python scripts/build.py --tier=aarch64-bf16 > "$out/build-bfdot.log" 2>&1 || exit 1
sha256sum cpu_attention/_C*.so > "$out/bfdot-sha256.txt"
mkdir -p "$out/bfdot-gates"
for test in test_dense test_flash_api fuzz_dense test_paged_sweep test_regressions; do
  extra=()
  if [ "$test" = fuzz_dense ]; then extra=(--cases 80 --max-seq 900); fi
  CPU_ATTENTION_BACKEND=mojo python "tests/$test.py" "${extra[@]}" > "$out/bfdot-gates/$test.log" 2>&1
  status=$?
  echo "$test exit=$status" | tee -a "$out/bfdot-gates/status.txt"
done
python "$out/compare.py" --manifest full --only paged --arms mojo_new,mojo_ctrl,cpp,vllm --screen-x 100000 --rounds 41 --paged-rounds 41 --threads 8 --vllm-bf16 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-fp32 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-isa-bf16 neon --vllm-isa-fp32 neon --json "$out/paged-full-bfdot.json" > "$out/paged-full-bfdot.log" 2>&1
echo "BFDOT_DONE exit=$?"
