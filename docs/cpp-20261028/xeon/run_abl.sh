#!/usr/bin/env bash
# Phase ablation sweep on the box (one process per mode), under the bench lock.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
ROWS="${ROWS:-128:8:64:fp32,128:8:64:bf16,1024:8:64:fp32,1024:8:32:fp32,256:16:128:fp32}"
for m in ${MODES:-0 1 2 4 8 16 32}; do
  .venv/bin/python docs/cpp-20261028/xeon/abl.py --mode "$m" --rows "$ROWS" \
    --rounds "${ROUNDS:-41}" 2>&1 | grep '"row"'
done
