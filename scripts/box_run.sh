#!/usr/bin/env bash
# Run one command on a bench box under an advisory lock, so two agents (or one
# agent and the integrator) can share the single measurement box safely.
#
#   scripts/box_run.sh <host> '<command>'
#   scripts/box_run.sh ubuntu@13.210.162.215 'cd ~/arm-work && .venv/bin/python tests/bench_vs_refs.py ...'
#
# Why: these boxes are the bottleneck. The rule is "one measurement at a time"
# -- a `mojo build` running beside a sweep turned a 0.59 o/sdpa row into 1.04 --
# but the *only* way to enforce it across sessions is to serialise on the box
# itself. `flock` on ~/bench.lock does that: whoever holds it measures, everyone
# else waits. Builds take the lock too, because a build is 4 cores of compiler.
#
# The lock is released when the command exits, including on failure. A command
# that hangs holds the lock, so pass a timeout in the command if that matters
# (`timeout 900 ...`).
set -euo pipefail

HOST="${1:?usage: box_run.sh <host> '<command>'}"
CMD="${2:?command required}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -i "$SSH_KEY" "$HOST" \
  "flock --timeout 3600 ~/bench.lock -c $(printf '%q' "$CMD")"
