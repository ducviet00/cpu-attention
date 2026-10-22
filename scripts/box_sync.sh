#!/usr/bin/env bash
# Sync a local tree to a bench box and (re)build the kernel there.
#
#   scripts/box_sync.sh <arch> <host> [local_dir] [--full]
#
#   arch        arm64 | x86_64        (only used for the vLLM reference flags)
#   host        ubuntu@<ip>
#   local_dir   worktree to push (default: this repo)
#   --full      also (re)create the venv and the vLLM reference .so
#
# The first call on a fresh box needs --full. Later calls are incremental: it
# rsyncs source only and rebuilds _C.so (~20 s), which is what an agent does
# after every kernel edit.
set -euo pipefail

ARCH="${1:?usage: box_sync.sh <arm64|x86_64> <host> [local_dir] [--full]}"
HOST="${2:?host required}"
LOCAL="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FULL="${4:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-cpu-attention}"
SSH=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST")

echo "== sync $LOCAL -> $HOST:~/$REMOTE_DIR =="
rsync -az --delete \
  --exclude '.venv' --exclude '_C.so' --exclude '__pycache__' \
  --exclude '_mojo/build' --exclude '.ruff_cache' --exclude '.git' \
  -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
  "$LOCAL/" "$HOST:$REMOTE_DIR/"

if [ "$FULL" = "--full" ]; then
  echo "== toolchain =="
  "${SSH[@]}" 'bash -s' <<'EOS'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build rsync python3.14-dev pkg-config >/dev/null
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
$HOME/.local/bin/uv --version
EOS
  echo "== venv =="
  "${SSH[@]}" "bash -lc 'cd ~/$REMOTE_DIR && ~/.local/bin/uv sync --extra build 2>&1 | tail -3'"
fi

echo "== build =="
# `set -o pipefail` so a failed build is not hidden by `tail`'s exit status.
"${SSH[@]}" "bash -lc 'set -o pipefail; cd ~/$REMOTE_DIR && PATH=\$PWD/.venv/bin:\$PATH .venv/bin/python scripts/build.py 2>&1 | tail -2'"

if [ "$FULL" = "--full" ]; then
  echo "== vllm reference =="
  "${SSH[@]}" "bash -s -- $ARCH" <<'EOS'
set -e
ARCH="$1"
# Pinned so both boxes build the same csrc/cpu; bump deliberately.
VLLM_COMMIT="db3b0f0fae4bc98492b3863685e541e62bc59576"
if [ ! -d "$HOME/vllm/.git" ]; then
  git clone https://github.com/vllm-project/vllm "$HOME/vllm" >/dev/null 2>&1
fi
cd "$HOME/vllm" && git checkout -q "$VLLM_COMMIT"
mkdir -p "$HOME/vllm_cpu_build"
cd "$HOME/vllm_cpu_build"
ln -sfn "$HOME/vllm/csrc" csrc
cp "$HOME/cpu-attention/scripts/vllm_ref/minimal_binding.cpp" .
python3 csrc/cpu/generate_cpu_attn_dispatch.py
T="$HOME/cpu-attention/.venv/lib/python3.14/site-packages/torch"
PY=$(ls -d /usr/include/python3.14 2>/dev/null || ls -d $HOME/cpu-attention/.venv/include/python3.14)
if [ "$ARCH" = "arm64" ]; then
  FLAGS="-DARM_BF16_SUPPORT -march=armv8.2-a+bf16+i8mm"
else
  FLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2"
fi
g++ -O3 -fPIC -shared -std=c++20 -fopenmp $FLAGS -DVLLM_NUMA_DISABLED \
  -DTORCH_EXTENSION_NAME=vllm_cpu_attn_bench -Icsrc \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PY" \
  csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp \
  minimal_binding.cpp \
  -o vllm_cpu_attn_bench.so -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python
echo "vllm ref ok: $(ls -la vllm_cpu_attn_bench.so | awk '{print $5}') bytes"
EOS
fi

echo "== parity =="
"${SSH[@]}" "bash -lc 'cd ~/$REMOTE_DIR && .venv/bin/python tests/test_dense.py 2>&1 | tail -2; .venv/bin/python tests/test_flash_api.py 2>&1 | tail -2'"
