#!/usr/bin/env bash
# On the box: in-process old/new A/B of two saved builds (ab_so.py), under the lock.
#   ab.sh <old> <new> <json-name> [shapes] [dtype] [rounds]
set -eo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
mkdir -p ~/xa_out
.venv/bin/python docs/cpp-20261028/xeon/ab_so.py --old ~/xa_so/"$1".so \
  --new ~/xa_so/"$2".so --json ~/xa_out/"$3".json \
  --shapes "${4:-128:8:64,1024:8:64,1024:8:32,2048:8:64,256:16:128}" \
  --dtype "${5:-both}" --rounds "${6:-60}" 2>&1 | grep -v Warn | grep -v _conversion
