#!/usr/bin/env bash
# On the box: build _C_cpp.so from this tree and keep a named copy for A/Bs.
#   build_variant.sh <name>   ->  ~/xa_so/<name>.so
set -eo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p ~/xa_so
.venv/bin/python scripts/build_cpp.py 2>&1 | tail -1
cp cpu_attention/_C_cpp.so ~/xa_so/"$1".so
