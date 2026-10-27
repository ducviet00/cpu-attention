#!/usr/bin/env bash
# push.sh <variant-name>: sync this worktree, build _C_cpp.so on the Axion box
# under the bench lock, and keep a copy as so/<variant-name>.so.
set -euo pipefail
H=ducviet00@34.87.120.65
W=$(cd "$(dirname "$0")/../../.." && pwd)
rsync -az --delete --exclude .venv --exclude _C.so --exclude __pycache__ \
  --exclude _mojo/build --exclude .ruff_cache --exclude .git --exclude _C_cpp.so \
  --exclude so/ "$W/" "$H:cpu-attention-arm-agent/"
ssh "$H" "cd ~/cpu-attention-arm-agent && mkdir -p so && flock ~/bench.lock -c '.venv/bin/python scripts/build_cpp.py' 2>&1 | tail -1 && cp cpu_attention/_C_cpp.so so/$1.so"
