#!/usr/bin/env bash
# Final verification on a bench box: build, parity, full matrix (61 dense / 121
# paged rounds, rotated arms, duplicate control) against Mojo and the vLLM build
# for the ISA the box actually uses. Run under ~/bench.lock (scripts/box_run.sh).
set -euo pipefail
cd "$HOME/cpu-attention-final-20261028"
OUT=results-final; mkdir -p $OUT
# The shared venv's editable install points at ~/cpu-attention: pin this tree.
export PYTHONPATH=$PWD CPU_ATTENTION_BACKEND=cpp OMP_PROC_BIND=spread OMP_PLACES=cores
if [[ $(uname -m) == aarch64 ]]; then THREADS=8; else THREADS=4; fi
export OMP_NUM_THREADS=$THREADS
PY=.venv/bin/python
$PY scripts/build_cpp.py --test > $OUT/build-test.log 2>&1
$PY tests/fuzz_dense.py --cases 80 --max-seq 900 > $OUT/fuzz.log 2>&1
$PY tests/test_paged_sweep.py > $OUT/paged-parity.log 2>&1
CPU_ATTENTION_BACKEND=mojo $PY tests/test_dense.py > $OUT/mojo-dense.log 2>&1
CPU_ATTENTION_BACKEND=mojo $PY tests/test_flash_api.py > $OUT/mojo-flash.log 2>&1
if [[ $(uname -m) == aarch64 ]]; then
  $PY tests/bench_cpp.py --threads $THREADS --vllm "$HOME/vllm_cpu_build/vllm_cpu_attn_bench.so" \
    --isa neon --block 16 --rounds 61 --paged-rounds 121 --json $OUT/matrix.json > $OUT/matrix.log 2>&1
else
  for dtype in bf16 fp32; do
    if [[ $dtype == bf16 ]]; then ISA=amx; REF=vllm_cpu_attn_bench_amx.so; else ISA=vec; REF=vllm_cpu_attn_bench.so; fi
    $PY tests/bench_cpp.py --threads $THREADS --dtype $dtype --vllm "$HOME/vllm_cpu_build/$REF" \
      --isa $ISA --rounds 61 --paged-rounds 121 --json $OUT/$dtype-$ISA.json > $OUT/$dtype-$ISA.log 2>&1
  done
fi
sha256sum cpu_attention/_C.so cpu_attention/_C_cpp.so cpu_attention/csrc/*.hpp "$HOME"/vllm_cpu_build/*.so > $OUT/sha256.txt
lscpu > $OUT/lscpu.txt
echo VERIFY_DONE
