#!/usr/bin/env bash
# One-shot setup for a bench box that has no checkout yet.
#   scripts/box_setup_cpp.sh <host> <arch: x86_64|aarch64> [--vllm-amx]
set -euo pipefail

HOST="${1:?host}"
ARCH="${2:?arch}"
AMXFLAGS="${3:-}"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$KEY" "$HOST")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== rsync =="
rsync -az --delete \
  --exclude '.venv' --exclude '*.so' --exclude '__pycache__' \
  --exclude 'build' --exclude '.ruff_cache' --exclude '.git' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY" \
  "$ROOT/" "$HOST:cpu-attention/"

if [ "$ARCH" = "x86_64" ]; then
  echo "== rsync vllm csrc =="
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $KEY" \
    "$HOME/vllm/csrc/" "$HOST:vllm/csrc/"
fi

echo "== remote build =="
"${SSH[@]}" "bash -s -- $ARCH $AMXFLAGS" <<'EOS'
set -eo pipefail
ARCH="$1"
AMXFLAGS="${2:-}"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cpu-attention"

if [ ! -x .venv/bin/python ]; then
  echo "-- uv sync --"
  uv sync --extra build 2>&1 | tail -3
fi

echo "-- python --"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "threads", torch.get_num_threads())
try:
    import mojo
    print("mojo ok")
except Exception as e:
    print("mojo MISSING", e)
PY

echo "-- build mojo --"
if [ -f scripts/build.py ]; then
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/build.py 2>&1 | tail -3
fi

echo "-- build cpp --"
.venv/bin/python scripts/build_cpp.py 2>&1 | tail -2

echo "-- vllm ref --"
if [ ! -f "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" ]; then
  mkdir -p "$HOME/vllm_cpu_build"
  cd "$HOME/vllm_cpu_build"
  [ -e csrc ] || ln -s "$HOME/vllm/csrc" csrc
  cp "$HOME/cpu-attention/scripts/vllm_ref/minimal_binding.cpp" .
  python3 csrc/cpu/generate_cpu_attn_dispatch.py
  PV=$("$HOME/cpu-attention/.venv/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  T="$HOME/cpu-attention/.venv/lib/python$PV/site-packages/torch"
  PY=$("$HOME/cpu-attention/.venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
  if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    FLAGS="-DARM_BF16_SUPPORT -march=armv8.2-a+bf16+i8mm"
  else
    FLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2 $AMXFLAGS"
  fi
  g++ -O3 -fPIC -shared -std=c++20 -fopenmp $FLAGS -DVLLM_NUMA_DISABLED \
    -DTORCH_EXTENSION_NAME=vllm_cpu_attn_bench -Icsrc \
    -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PY" \
    csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp \
    minimal_binding.cpp \
    -o vllm_cpu_attn_bench.so -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python
fi
SO=$(ls "$HOME/vllm_cpu_build"/*.so)
echo "vllm ref: $(ls -la "$SO" | awk '{print $5}') bytes, bfdot=$(objdump -d "$SO" | grep -c bfdot || true) bfmmla=$(objdump -d "$SO" | grep -c bfmmla || true) tdpbf16ps=$(objdump -d "$SO" | grep -c tdpbf16ps || true)"

cd "$HOME/cpu-attention"
echo "-- isa --"
.venv/bin/python -c "import cpu_attention as a; print('cpp isa', a._kernel._entry('isa')())" 2>/dev/null || true
echo "-- parity cpp --"
CPU_ATTENTION_BACKEND=cpp .venv/bin/python tests/test_dense.py 2>&1 | tail -2
CPU_ATTENTION_BACKEND=cpp .venv/bin/python tests/test_flash_api.py 2>&1 | tail -2
echo "-- parity mojo --"
.venv/bin/python tests/test_dense.py 2>&1 | tail -2
.venv/bin/python tests/test_flash_api.py 2>&1 | tail -2
echo "-- cpp .so --"
objdump -d cpu_attention/_C_cpp.so | grep -cE "bfdot|bfmmla|vdpbf16ps|tdpbf16ps" || true
touch "$HOME/setup.done"
echo "SETUP OK"
EOS
