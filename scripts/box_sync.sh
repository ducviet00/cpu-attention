#!/usr/bin/env bash
# rsync this tree to a bench box, rebuild _C.so there, run both parity suites.
#   scripts/box_sync.sh x86_64 ubuntu@3.27.45.233 /path/to/worktree
set -euo pipefail
TARGET_TRIPLE=${1:?triple}
HOST=${2:?host}
SRC=${3:?source dir}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REMOTE=${REMOTE:-cpu-attention}

RS=(rsync -az --delete --exclude '.venv' --exclude '.git' --exclude '__pycache__'
    --exclude '*.egg-info' --exclude 'build' --exclude '*.so' --exclude '.ruff_cache')
"${RS[@]}" -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$SRC/" "$HOST:~/$REMOTE/"

ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST" "
  set -e
  cd ~/$REMOTE
  # extra C sources (AMX shim etc.) are built by scripts/build.py when present
  PATH=\$HOME/$REMOTE/.venv/bin:\$PATH .venv/bin/python scripts/build.py
"
