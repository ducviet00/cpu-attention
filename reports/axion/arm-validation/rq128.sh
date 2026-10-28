#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
python "$out/rq128.py" --previous "$out/before" --manifest full --only flash --arms mojo_new,mojo_ctrl,mojo_old,cpp,sdpa,vllm --rounds 41 --paged-rounds 41 --threads 8 --vllm-bf16 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-fp32 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-isa-bf16 neon --vllm-isa-fp32 neon --json "$out/flash-full-rq128.json" > "$out/flash-full-rq128.log" 2>&1
echo "RQ128_DONE exit=$?"
