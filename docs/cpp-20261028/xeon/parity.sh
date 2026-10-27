#!/usr/bin/env bash
# On the box: the C++ parity gates for this tree (PYTHONPATH pins this checkout:
# the shared venv's editable install points at another tree).
#   parity.sh [dense|paged|all]
set -eo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores
export PYTHONPATH=$PWD CPU_ATTENTION_BACKEND=cpp
f() { .venv/bin/python "$@" 2>&1 | grep -v -i -E "numpy|_conversion" | tail -2; }
case "${1:-dense}" in
  dense|all)
    f tests/test_dense.py
    f tests/test_flash_api.py
    f tests/fuzz_dense.py --cases 80 --max-seq 900
    f tests/test_amx.py ;;&
  paged|all)
    f tests/test_paged_sweep.py ;;
esac
