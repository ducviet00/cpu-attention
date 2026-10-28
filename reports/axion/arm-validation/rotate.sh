#!/usr/bin/env bash
set -u
cd /home/ducviet00/arm-validation-afbea06
export PATH=$HOME/agent-arm/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$PWD
exec 9>"$HOME/bench.lock"
flock 9
out=reports/axion/arm-validation
ps -eo pid,pcpu,args --sort=-pcpu | head -15 > "$out/processes-rotated.txt"
python "$out/rotated_packages.py" --previous "$out/before" --manifest full --only paged --arms mojo_new,mojo_ctrl,cpp --rounds 41 --paged-rounds 41 --threads 8 --json "$out/paged-full-hilo-rotated.json" > "$out/paged-full-hilo-rotated.log" 2>&1
echo "ROTATED_DONE exit=$?"
