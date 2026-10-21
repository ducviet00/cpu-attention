#!/usr/bin/env bash
# Provision a benchmark box: toolchain, repo, venv, kernel build, vLLM reference.
# usage: scripts/setup_box.sh <arch>   (arch = arm64 | x86_64)
set -euo pipefail
ARCH="$1"
export PATH="$HOME/.local/bin:$PATH"

echo "== apt =="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build rsync python3.14-dev >/dev/null

echo "== uv =="
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
uv --version

echo "== repo =="
cd "$HOME/cpu-attention"
uv sync --extra build 2>&1 | tail -2
export PATH="$HOME/cpu-attention/.venv/bin:$PATH"
.venv/bin/python scripts/build.py 2>&1 | tail -1

echo "== parity =="
.venv/bin/python tests/test_dense.py 2>&1 | tail -1
.venv/bin/python tests/test_flash_api.py 2>&1 | tail -1

echo "== vllm reference build =="
cd "$HOME/vllm_cpu_build"
T="$HOME/cpu-attention/.venv/lib/python3.14/site-packages/torch"
if [ "$ARCH" = "arm64" ]; then
  FLAGS="-DARM_BF16_SUPPORT -march=armv8.2-a+bf16"
else
  FLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2"
fi
g++ -O3 -fPIC -shared -std=c++20 -fopenmp $FLAGS -DVLLM_NUMA_DISABLED \
  -DTORCH_EXTENSION_NAME=vllm_cpu_attn_bench -Icsrc \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -I/usr/include/python3.14 \
  csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp \
  csrc/cpu/minimal_binding.cpp \
  -o vllm_cpu_attn_bench.so -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python \
  || echo "WARNING: vLLM reference build failed (set -e disabled for it)"
cd "$HOME/cpu-attention"
echo "vllm ref: $(ls -la "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" | awk '{print $5}') bytes, \
$(objdump -d vllm_cpu_attn_bench.so | grep -c bfdot || true) bfdot, \
$(objdump -d vllm_cpu_attn_bench.so | grep -c bfmmla || true) bfmmla, \
$(objdump -d vllm_cpu_attn_bench.so | grep -c vdpbf16ps || true) vdpbf16ps"

echo "== isa =="
grep -o -m1 'avx512_bf16\|amx_bf16\|bf16' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
"$HOME/cpu-attention/.venv/bin/python" -c "import torch; print('amx:', torch.cpu._is_amx_tile_supported(), 'threads:', torch.get_num_threads())"
echo "== profilers =="
sudo apt-get install -y -qq valgrind >/dev/null 2>&1 && valgrind --version
sudo apt-get install -y -qq linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1
perf --version 2>/dev/null || echo "perf not installed"
cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null
(sudo perf stat -e cycles true) 2>&1 | tail -3
for p in llvm-mca llvm-mca-19 llvm-mca-20 llvm-mca-21; do command -v $p && $p --version | head -2; done
echo "== ready =="
touch "$HOME/setup.done"
