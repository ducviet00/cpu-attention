#!/usr/bin/env bash
# Authoritative verification of one tree on one box: sync, build from scratch,
# parity, then the full 22-row matrix against both vLLM references.
#
#   RUNS=2 scripts/verify.sh <arch> <host> <local_tree> [label]
#
#   arch    arm64 | x86_64   (picks --isa neon / the vLLM AMX build)
#   host    ubuntu@<ip>
#   label   name for the JSON files (default: the tree's directory name)
#   RUNS    how many times to run the matrix (default 1). Rows at 1.00-1.04
#           answer differently on consecutive runs -- this campaign published one
#           0.98 that re-measured at 1.14-1.21 -- so a claim about a row near
#           the line needs RUNS=2 and `scripts/band.py`. Both runs happen in the
#           same process image and box, back to back, so they share the drift.
#   REMOTE_DIR  where the tree lands on the box (default cpu-attention). Set it
#           per agent: the box is shared, and only one thing may time at a time.
#
# Every remote step takes the box flock (~/bench.lock): a build or a parity run
# from another agent beside a timing run moves rows by 1.7x.
#
# Writes ~/verify-<label>-<isa>.json on the box and prints the scorecard. On x86
# it also builds and times vLLM's AMX variant, because that is the ISA vLLM
# would pick on an AMX machine and it is the honest bar for bf16 (their AMX path
# is bf16-only; fp32 has no AMX kernel, so VEC is their best there).
#
# Nothing here is trusted to be incremental: the .so is rebuilt from scratch, so
# a stale build cannot pass for a fresh one.
set -euo pipefail

ARCH="${1:?usage: verify.sh <arch64|x86_64> <host> <tree> [label]}"
HOST="${2:?host required}"
TREE="${3:?tree required}"
LABEL="${4:-$(basename "$TREE")}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-cpu-attention}"
SSH=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST")

# Run a shell script on the box under the measurement lock. The script arrives
# on stdin, so no quoting survives into the remote shell.
remote_locked() {
  "${SSH[@]}" "flock ~/bench.lock -c 'bash -s'"
}

echo "== sync =="
REMOTE_DIR="$REMOTE_DIR" scripts/box_sync.sh "$ARCH" "$HOST" "$TREE"

echo "== clean build =="
remote_locked <<EOS
set -eo pipefail
cd ~/$REMOTE_DIR
rm -rf cpu_attention/_mojo/build cpu_attention/_C.so
PATH=\$PWD/.venv/bin:\$PATH .venv/bin/python scripts/build.py 2>&1 | tail -1
EOS

echo "== parity =="
remote_locked <<EOS
set -e
cd ~/$REMOTE_DIR
.venv/bin/python tests/test_dense.py 2>&1 | tail -1
.venv/bin/python tests/test_flash_api.py 2>&1 | tail -1
EOS
# Randomised parity, including the plan cache's hit/invalidate paths: a fixed
# shape list cannot cover the per-dtype and per-head-dim weight layouts crossed
# with GQA, odd lengths and strided inputs, and a stale plan corrupts silently.
remote_locked <<EOS
set -e
cd ~/$REMOTE_DIR
.venv/bin/python tests/fuzz_dense.py --cases 40 --max-seq 700 2>&1 | tail -1
EOS
# The AMX kernel has its own parity harness (shape grid, both causal modes,
# compared against SDPA *and* against the Mojo kernel) plus a byte-level pack
# self-test; it reports "skipped" on a host without AMX.
remote_locked <<EOS
cd ~/$REMOTE_DIR
.venv/bin/python tests/test_amx.py 2>&1 | tail -2
EOS

if [ "$ARCH" = "arm64" ]; then
  ISAS="neon"
else
  # Build vLLM's AMX variant too: same sources, torch's AMXBF16 capability
  # macro, and the tile instructions enabled.
  "${SSH[@]}" 'bash -s' <<'EOS'
set -e
cd "$HOME/vllm_cpu_build"
T="$HOME/cpu-attention/.venv/lib/python3.14/site-packages/torch"
PY=$(ls -d /usr/include/python3.14 2>/dev/null || ls -d "$HOME/cpu-attention/.venv/include/python3.14")
g++ -O3 -fPIC -shared -std=c++20 -fopenmp \
  -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2 \
  -DCPU_CAPABILITY_AMXBF16 -mamx-tile -mamx-bf16 -mamx-int8 \
  -DVLLM_NUMA_DISABLED -DTORCH_EXTENSION_NAME=vllm_cpu_attn_bench_amx -Icsrc \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PY" \
  csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/utils.cpp minimal_binding.cpp \
  -o vllm_cpu_attn_bench_amx.so -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python \
  2>/dev/null || echo "WARNING: AMX reference build failed"
ls -la vllm_cpu_attn_bench_amx.so 2>/dev/null | awk '{print "amx ref:", $5, "bytes"}'
EOS
  ISAS="vec amx"
fi

RUNS="${RUNS:-1}"
for ISA in $ISAS; do
  REF="~/vllm_cpu_build/vllm_cpu_attn_bench.so"
  [ "$ISA" = "amx" ] && REF="~/vllm_cpu_build/vllm_cpu_attn_bench_amx.so"
  # vLLM's AMX path is bf16-only (`TileGemm224` is specialised for a BFloat16
  # query; an fp32 query trips TORCH_CHECK and aborts the process), so the amx
  # matrix runs bf16 only. fp32 has no AMX kernel anywhere, which is why VEC is
  # that column's honest baseline.
  DT=""
  [ "$ISA" = "amx" ] && DT="--dtype bf16"
  for R in $(seq 1 "$RUNS"); do
    SUF="$ISA"; [ "$RUNS" -gt 1 ] && SUF="$ISA-r$R"
    echo "== matrix, isa=$ISA$DT run $R/$RUNS =="
    remote_locked <<EOS
# The box must be *idle*, not merely ours: a foreign build inflates a row by
# 1.7x and a foreign python by 25%. Wait for a load average under 0.3 (a
# single-threaded parent costs ~0.1), then refuse to publish if something else
# was running while we timed.
wait_idle() {
  for _ in \$(seq 1 240); do
    L=\$(cut -d' ' -f1 /proc/loadavg)
    awk "BEGIN{exit !(\$L <= 0.3)}" && return 0
    sleep 5
  done
  echo "!! box still loaded (\$L) after 20 min" >&2
  return 1
}
foreign_hogs() {
  ps -eo pcpu,comm --sort=-pcpu --no-headers | head -3
}
wait_idle || exit 1
echo "box idle: load \$(cut -d' ' -f1 /proc/loadavg)"
cd ~/$REMOTE_DIR
.venv/bin/python tests/bench_vs_refs.py --vllm $REF --isa $ISA $DT --rounds ${ROUNDS:-9} --json ~/verify-$LABEL-$SUF.json 2>/dev/null | tail -3
EOS
    echo "load after run $R: $(ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$HOST" "cut -d' ' -f1-3 /proc/loadavg; ps -eo pcpu,comm --sort=-pcpu --no-headers | head -3 | tr '\n' ' '")"
  done
done

echo "== scorecard (local) =="
mkdir -p /tmp/verify
for ISA in $ISAS; do
  if [ "$RUNS" -gt 1 ]; then
    for R in $(seq 1 "$RUNS"); do
      scp -q -i "$SSH_KEY" "$HOST:~/verify-$LABEL-$ISA-r$R.json" \
        "/tmp/verify/$LABEL-$ISA-r$R.json"
    done
    # band: per-row min..max over the runs, and count a row as a win only if it
    # is below 1 in every one of them
    python3 "$(dirname "${BASH_SOURCE[0]}")/band.py" \
      /tmp/verify/"$LABEL"-"$ISA"-r*.json
  else
    scp -q -i "$SSH_KEY" "$HOST:~/verify-$LABEL-$ISA.json" \
      "/tmp/verify/$LABEL-$ISA.json"
    python3 "$(dirname "${BASH_SOURCE[0]}")/scorecard.py" \
      /tmp/verify/$LABEL-$ISA.json
  fi
done
