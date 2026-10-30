#!/usr/bin/env bash
# Run a command with every CPU's governor (and EPP, where present) pinned to
# `performance`, then restore each CPU's ORIGINAL values, also on error/Ctrl-C.
#
#   scripts/pin_governor.sh python benchmarks/bench.py ...
#
# On boxes without cpufreq control (GCP VMs) it just runs the command.
set -u
saved=()
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor \
         /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference; do
  [ -w "$f" ] || sudo -n test -w "$f" 2>/dev/null || continue
  saved+=("$f=$(cat "$f")")
done
restore() {
  for kv in "${saved[@]}"; do
    echo "${kv#*=}" | sudo -n tee "${kv%%=*}" >/dev/null
  done
  [ ${#saved[@]} -gt 0 ] && echo "pin_governor: restored ${#saved[@]} settings" >&2
}
trap restore EXIT INT TERM
for kv in "${saved[@]}"; do
  echo performance | sudo -n tee "${kv%%=*}" >/dev/null
done
[ ${#saved[@]} -gt 0 ] && echo "pin_governor: pinned ${#saved[@]} settings to performance" >&2
"$@"
