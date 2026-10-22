#!/usr/bin/env bash
# Wait for the shared box's 1-minute load average to stay under 0.8, then run.
# The flock in box_run.sh serialises agents that use it; this catches everything
# else (an agent's build, a verify run from a session that did not take the
# lock). A concurrent 2-core job inflates our 8-thread timings by ~25%.
set -euo pipefail
HOST="${1:?usage: quiet.sh <host> '<command>'}"
CMD="${2:?command required}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST" \
  "for i in \$(seq 1 240); do l=\$(cut -d' ' -f1 /proc/loadavg); \
     ok=\$(awk -v l=\$l 'BEGIN{print (l<0.8)?1:0}'); \
     if [ \$ok = 1 ]; then n=\$((\${n:-0}+1)); else n=0; fi; \
     if [ \${n:-0} -ge 6 ]; then break; fi; sleep 2; done; \
   echo \"load before: \$(cat /proc/loadavg)\"; \
   flock --timeout 3600 ~/bench.lock -c $(printf '%q' "$CMD"); \
   echo \"load after: \$(cat /proc/loadavg)\""
