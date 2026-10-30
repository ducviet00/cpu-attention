#!/usr/bin/env bash
# box_verify.sh BRANCH "ISAS" [bench.py args...]: the tests on BRANCH at the best tier and at
# each lower tier in ISAS (e.g. "vec avx2 scalar" or "neon scalar"), then the benchmark.
#   REPO     git URL or bundle to clone from (default: ~/cpu-attention's origin)
#   NOBENCH  1 = tests only
# Results go to ~/verify-BRANCH.out/{tests-*.log,bench.json,bench.log}.
# Launch it outside the ssh session (`sudo systemd-run --uid=$USER --setenv=HOME=$HOME
# --working-directory=$HOME bash ...`): when the last session logs out, systemd stops the
# user's processes, nohup and setsid included.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
BR=$1; ISAS=$2; shift 2
REPO=${REPO:-$(git -C ~/cpu-attention remote get-url origin)}
D=$HOME/verify-$BR; O=$D.out
rm -rf "$D" "$O"; mkdir -p "$O"
git clone -q -b "$BR" "$REPO" "$D" && cd "$D" || exit 1
uv sync --extra build --extra test >/dev/null 2>&1 || { echo SYNC_FAIL; exit 1; }
uv run --extra build python scripts/build.py > "$O/build.log" 2>&1 && echo BUILD_OK || { echo BUILD_FAIL; tail -20 "$O/build.log"; exit 1; }
echo "HEAD $(git rev-parse --short HEAD)"
for isa in best $ISAS; do
  [ "$isa" = best ] && unset CPU_ATTENTION_ISA || export CPU_ATTENTION_ISA=$isa
  uv run --extra test pytest -q > "$O/tests-$isa.log" 2>&1 && echo "PASS $isa $(tail -1 "$O/tests-$isa.log")" || echo "FAIL $isa $(tail -3 "$O/tests-$isa.log" | tr '\n' ' ')"
done
unset CPU_ATTENTION_ISA
echo TESTS_DONE
[ "${NOBENCH:-}" = 1 ] && exit 0
uv run python benchmarks/bench.py --json "$O/bench.json" "$@" > "$O/bench.log" 2>&1 && echo BENCH_DONE || { echo BENCH_FAIL; tail -20 "$O/bench.log"; }
