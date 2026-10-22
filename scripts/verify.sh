#!/usr/bin/env bash
# Authoritative verification of one tree on one box: sync, build from scratch,
# parity, then the full 22-row matrix against both vLLM references.
#
#   scripts/verify.sh <arch> <host> <local_tree> [label]
#
#   arch    arm64 | x86_64   (picks --isa neon / the vLLM AMX build)
#   host    ubuntu@<ip>
#   label   name for the JSON files (default: the tree's directory name)
#
# Writes ~/verify-<label>-<isa>.json on the box and prints the scorecard. On x86
# it also builds and times vLLM's AMX variant, because that is the ISA vLLM
# would pick on an AMX machine and it is the honest bar for bf16 (their AMX path
# is bf16-only; fp32 has no AMX kernel, so VEC is their best there).
#
# Nothing here is trusted to be incremental: the .so is rebuilt from scratch, so
# a stale build cannot pass for a fresh one.
set -euo pipefail

ARCH="${1:?usage: verify.sh <arm64|x86_64> <host> <tree> [label]}"
HOST="${2:?host required}"
TREE="${3:?tree required}"
LABEL="${4:-$(basename "$TREE")}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST")

echo "== sync =="
scripts/box_sync.sh "$ARCH" "$HOST" "$TREE"

echo "== clean build =="
"${SSH[@]}" 'bash -lc "cd ~/cpu-attention && rm -rf cpu_attention/_mojo/build cpu_attention/_C.so && PATH=\$PWD/.venv/bin:\$PATH .venv/bin/python scripts/build.py 2>&1 | tail -1"'

echo "== parity =="
"${SSH[@]}" 'bash -lc "cd ~/cpu-attention && .venv/bin/python tests/test_dense.py 2>&1 | tail -1 && .venv/bin/python tests/test_flash_api.py 2>&1 | tail -1"'
# Randomised parity, including the plan cache's hit/invalidate paths: a fixed
# shape list cannot cover the per-dtype and per-head-dim weight layouts crossed
# with GQA, odd lengths and strided inputs, and a stale plan corrupts silently.
"${SSH[@]}" 'bash -lc "cd ~/cpu-attention && .venv/bin/python tests/fuzz_dense.py --cases 40 --max-seq 700 2>&1 | tail -1"'
# The AMX kernel has its own parity harness (shape grid, both causal modes,
# compared against SDPA *and* against the Mojo kernel) plus a byte-level pack
# self-test; it reports "skipped" on a host without AMX.
"${SSH[@]}" 'bash -lc "cd ~/cpu-attention && .venv/bin/python tests/test_amx.py 2>&1 | tail -2"'

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
  csrc/cpu/cpu_attn.cpp csrc/cpu/cpu_isa.cpp csrc/cpu/utils.cpp minimal_binding.cpp \
  -o vllm_cpu_attn_bench_amx.so -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python \
  2>/dev/null || echo "WARNING: AMX reference build failed"
ls -la vllm_cpu_attn_bench_amx.so 2>/dev/null | awk '{print "amx ref:", $5, "bytes"}'
EOS
  ISAS="vec amx"
fi

for ISA in $ISAS; do
  # Single-quoted: the tilde must reach the remote shell unexpanded.
  REF='~/vllm_cpu_build/vllm_cpu_attn_bench.so'
  [ "$ISA" = "amx" ] && REF='~/vllm_cpu_build/vllm_cpu_attn_bench_amx.so'
  # vLLM's AMX path is bf16-only (`TileGemm224` is specialised for a BFloat16
  # query; an fp32 query trips TORCH_CHECK and aborts the process), so the amx
  # matrix runs bf16 only. fp32 has no AMX kernel anywhere, which is why VEC is
  # that column's honest baseline.
  DT=""
  [ "$ISA" = "amx" ] && DT="--dtype bf16"
  echo "== matrix, isa=$ISA$DT =="
  "${SSH[@]}" "bash -lc 'cd ~/cpu-attention && .venv/bin/python tests/bench_vs_refs.py --vllm $REF --isa $ISA $DT --rounds ${ROUNDS:-9} --json ~/verify-$LABEL-$ISA.json 2>/dev/null | tail -3'"
done

echo "== scorecard (local) =="
mkdir -p /tmp/verify
for ISA in $ISAS; do
  scp -q -i "$SSH_KEY" "$HOST:~/verify-$LABEL-$ISA.json" "/tmp/verify/$LABEL-$ISA.json"
done
# scorecard.py is stdlib-only and this script is often run from a worktree that
# has no .venv of its own.
ls /tmp/verify/"$LABEL"-*.json | xargs python3 "$(dirname "${BASH_SOURCE[0]}")/scorecard.py"
