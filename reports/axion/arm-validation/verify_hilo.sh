#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
python "$out/safe_reference.py" --previous "$out/before" --manifest full --only paged --arms mojo_new,mojo_ctrl,mojo_old,cpp,vllm --rounds 41 --paged-rounds 41 --threads 8 --vllm-bf16 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-fp32 "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" --vllm-isa-bf16 neon --vllm-isa-fp32 neon --json "$out/paged-full-hilo-scalar-peer.json" > "$out/paged-full-hilo-scalar-peer.log" 2>&1
echo "SCALAR_PEER_DONE exit=$?"
cd "$HOME/arm-validation-engine-afbea06"
export PYTHONPATH=$HOME/arm-validation-afbea06:$PWD/deps:$PWD
export OMP_NUM_THREADS=8
CPU_ATTENTION_BACKEND=mojo python tests/test_cpu_attention.py > "$HOME/arm-validation-afbea06/$out/hilo-gates/fastencdec.log" 2>&1
echo "HILO_ENGINE_DONE exit=$?"
