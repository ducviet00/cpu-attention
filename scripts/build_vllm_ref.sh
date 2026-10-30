#!/usr/bin/env bash
# Build vLLM's csrc/cpu attention kernel as a standalone torch extension, the
# baseline `benchmarks/bench.py --vllm` compares against.
#
#   scripts/build_vllm_ref.sh            # -> ~/vllm_cpu_build/vllm_cpu_attn_bench.so
#                                        #    (+ vllm_cpu_attn_bench_amx.so on x86)
#
# vLLM is pinned so every box builds the same kernel; bump deliberately. The
# VEC build is the fp32 baseline on x86 and the NEON build is the only one on
# aarch64. The AMX build is the bf16 baseline on an AMX box (`--isa amx`).
set -euo pipefail

VLLM_COMMIT="db3b0f0fae4bc98492b3863685e541e62bc59576"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="$REPO/.venv/bin/python"
OUT="$HOME/vllm_cpu_build"

# Only the pinned commit: a full clone of vLLM stalled on the GCP boxes.
if [ "$(git -C "$HOME/vllm" rev-parse HEAD 2>/dev/null)" != "$VLLM_COMMIT" ]; then
  rm -rf "$HOME/vllm" && git init -q "$HOME/vllm"
  git -C "$HOME/vllm" fetch -q --depth 1 https://github.com/vllm-project/vllm "$VLLM_COMMIT"
  git -C "$HOME/vllm" checkout -q FETCH_HEAD
fi

mkdir -p "$OUT" && cd "$OUT"
ln -sfn "$HOME/vllm/csrc" csrc
cp "$REPO/scripts/vllm_ref/minimal_binding.cpp" .
python3 csrc/cpu/generate_cpu_attn_dispatch.py

T=$("$PYBIN" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
PY=$("$PYBIN" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

build() {  # build <extension name> <flags...>
  local name="$1"; shift
  g++ -O3 -fPIC -shared -std=c++20 -fopenmp "$@" -DVLLM_NUMA_DISABLED \
    -DTORCH_EXTENSION_NAME="$name" -Icsrc \
    -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PY" \
    csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp \
    minimal_binding.cpp \
    -o "$name.so" -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python \
    -Wl,-rpath,"$T/lib"
  echo "built $OUT/$name.so"
}

if [ "$(uname -m)" = "aarch64" ]; then
  build vllm_cpu_attn_bench -DARM_BF16_SUPPORT -march=armv8.2-a+bf16+i8mm
else
  VEC=(-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni
       -mfma -mavx2)
  build vllm_cpu_attn_bench "${VEC[@]}"
  build vllm_cpu_attn_bench_amx "${VEC[@]}" -DCPU_CAPABILITY_AMXBF16 \
    -mamx-tile -mamx-bf16 -mamx-int8
fi
