#!/usr/bin/env bash
# box_verify.sh BRANCH BASE "ISAS" [bench.py args...]: the AGENTS.md gates on BRANCH (at the
# best tier and at each lower tier in ISAS, e.g. "vec avx2 scalar" or "neon scalar"), then
# the full manifest against BASE as the independent previous build, with SDPA and vLLM.
#   REPO     git URL or bundle to clone from (default: ~/cpu-attention's origin)
#   THREADS  bench threads (default: physical cores)
#   NOBENCH  1 = gates only
# Results go to ~/verify-BRANCH.out/{gates/*.log,full.json,full.log}; summarize the JSON
# with scripts/ab_summary.py.
# Launch it outside the ssh session (`sudo systemd-run --uid=$USER --setenv=HOME=$HOME
# --working-directory=$HOME bash ...`): when the last session logs out, systemd stops the
# user's processes, nohup and setsid included.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
BR=$1; BASE_REF=$2; ISAS=$3; shift 3
REPO=${REPO:-$(git -C ~/cpu-attention remote get-url origin)}
THREADS=${THREADS:-$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)}
D=$HOME/verify-$BR; BASE=$HOME/verify-base; G=$D.out/gates
rm -rf "$D" "$BASE"; mkdir -p "$G"
git clone -q -b "$BR" "$REPO" "$D" && git clone -q -b "$BASE_REF" "$REPO" "$BASE" || exit 1
cd "$BASE" && uv sync --extra build >/dev/null 2>&1 && uv run --extra build python scripts/build.py > "$G/base-build.log" 2>&1 && echo BASE_BUILD_OK || echo BASE_BUILD_FAIL
cd "$D" && uv sync --extra build >/dev/null 2>&1 || { echo SYNC_FAIL; exit 1; }
uv run --extra build python scripts/build.py > "$G/build.log" 2>&1 && echo BUILD_OK || { echo BUILD_FAIL; tail -20 "$G/build.log"; exit 1; }
echo "HEAD $(git rev-parse --short HEAD) THREADS $THREADS"
run() { local tag=$1; shift; "$@" > "$G/$tag.log" 2>&1 && echo "PASS $tag $(tail -1 "$G/$tag.log" | cut -c1-100)" || echo "FAIL $tag $(tail -3 "$G/$tag.log" | tr '\n' ' ' | cut -c1-300)"; }
gates() {
  run "$1-flash_api" uv run python tests/test_flash_api.py
  run "$1-dense" uv run python tests/test_dense.py
  run "$1-sweep" uv run python tests/test_paged_sweep.py
  run "$1-fuzz" uv run python tests/fuzz_dense.py --cases 80 --max-seq 900
  run "$1-reg" uv run python tests/test_regressions.py
}
gates best
grep -q amx_tile /proc/cpuinfo && run amx uv run python tests/test_amx.py --dtype both
for isa in $ISAS; do CPU_ATTENTION_ISA=$isa gates "$isa"; done
run protocol uv run python tests/test_benchmark_protocol.py
echo GATES_DONE
[ "${NOBENCH:-}" = 1 ] && exit 0
uv run python benchmarks/bench.py --manifest full --threads "$THREADS" --rounds 61 --paged-rounds 121 \
  --arms mojo_new,mojo_ctrl,sdpa,vllm --previous "$BASE/cpu_attention" "$@" \
  --json "$D.out/full.json" > "$D.out/full.log" 2>&1 && echo BENCH_DONE || { echo BENCH_FAIL; tail -20 "$D.out/full.log"; }
