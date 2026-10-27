#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/cpu-attention"
mkdir -p "$HOME/vllm_cpu_build"
cd "$HOME/vllm_cpu_build"
[ -e csrc ] || ln -s "$HOME/vllm/csrc" csrc
cp "$ROOT/scripts/vllm_ref/minimal_binding.cpp" .
python3 csrc/cpu/generate_cpu_attn_dispatch.py
T=$("$ROOT/.venv/bin/python" -c 'import torch,os;print(os.path.dirname(torch.__file__))')
PY=$("$ROOT/.venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
FLAGS=(-O3 -fPIC -std=c++20 -fopenmp -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2 -DVLLM_NUMA_DISABLED -Icsrc -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PY")
for variant in vec amx; do
  NAME=vllm_cpu_attn_bench
  EXTRA=()
  if [ "$variant" = amx ]; then
    NAME=vllm_cpu_attn_bench_amx
    EXTRA=(-DCPU_CAPABILITY_AMXBF16 -mamx-tile -mamx-bf16 -mamx-int8)
  fi
  for source in csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp minimal_binding.cpp; do
    g++ "${FLAGS[@]}" "${EXTRA[@]}" -DTORCH_EXTENSION_NAME="$NAME" -c "$source" -o "${variant}-$(basename "$source").o"
  done
  g++ -shared -fopenmp "$variant-"*.o -o "$NAME.so" -L"$T/lib" -Wl,-rpath,"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python
  objdump -d "$NAME.so" | grep -c tdpbf16ps || true
done
