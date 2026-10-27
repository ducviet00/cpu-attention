#!/usr/bin/env bash
# Env-knob screen: one process per setting, Mojo as the in-process reference.
#   VAR=CPU_ATTENTION_KT VALS="256 512" ROWS=... run_env.sh
set -eo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
ROWS="${ROWS:-128:8:64:fp32,1024:8:64:fp32,1024:8:32:fp32,256:16:128:fp32,2048:8:64:fp32}"
for v in $VALS; do
  echo "== $VAR=$v"
  env "$VAR=$v" .venv/bin/python docs/cpp-20261028/xeon/abl.py --mode 0 --rows "$ROWS" \
    --rounds "${ROUNDS:-41}" 2>&1 | grep "^#"
done
