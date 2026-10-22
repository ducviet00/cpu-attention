#!/usr/bin/env bash
# Wait for the box to be idle, take the bench lock, run a command, and report
# the load average on both sides so a contended run can be spotted.
#
#   scripts/box_idle.sh <host> '<command>'
set -euo pipefail
HOST="${1:?usage: box_idle.sh <host> '<command>'}"
CMD="${2:?command required}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST" \
  "for i in \$(seq 1 120); do
     l=\$(cut -d' ' -f1 /proc/loadavg); ok=\$(awk -v l=\$l 'BEGIN{print (l<0.8)?1:0}');
     if [ \$ok = 1 ]; then n=\$((\${n:-0}+1)); else n=0; fi;
     if [ \${n:-0} -ge 6 ]; then break; fi; sleep 2; done;
   echo \"load before: \$(cat /proc/loadavg)\";
   flock --timeout 3600 ~/bench.lock -c $(printf '%q' "$CMD");
   echo \"load after: \$(cat /proc/loadavg)\""
