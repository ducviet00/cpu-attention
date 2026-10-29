#!/usr/bin/env bash
# box_provision.sh [REPO [REF]]: prepare a fresh benchmark VM (the GCP Ubuntu images are bare).
#   toolchain and python3-dev (the C++ binding needs Python.h), uv, and ~/cpu-attention
#   with its venv: scripts/build_vllm_ref.sh links the vLLM reference against that torch.
# REPO: a git URL or a bundle path (default: origin on GitHub); REF: the branch to check
# out (a bundle's HEAD may name no branch).
set -uo pipefail
REPO=${1:-https://github.com/ducviet00/cpu-attention.git}; REF=${2:-}
sudo -n apt-get update -qq >/dev/null 2>&1
sudo -n apt-get install -y -qq build-essential git python3-dev >/dev/null 2>&1 && echo APT_OK || echo APT_FAIL
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uv --version
cd ~ && rm -rf cpu-attention && git clone -q ${REF:+-b "$REF"} "$REPO" cpu-attention && cd cpu-attention &&
  uv sync --extra build >/dev/null 2>&1 && echo SYNC_OK || echo SYNC_FAIL
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core\(s\)"
