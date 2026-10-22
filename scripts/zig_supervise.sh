#!/usr/bin/env bash
# Optional supervisor for ENGINE=zig. It watches both rank containers and follows the exit codes of the engine
# (docs/ZIG.md, "Exit codes"). Run it on the head, after ./start.sh, for example in tmux or in a systemd service.
#   scripts/zig_supervise.sh
# Rules, from the exit codes of the two ranks:
#   rank 0 exits 0                       a clean stop (./stop.sh or a signal): the supervisor ends with 0
#   a rank exits 75 (a peer fault)       ./stop.sh, then ./start.sh. The new pair is good only if ./start.sh ends with 0:
#                                        it ends with a real completion. At most ZIG_MAX_RESTARTS restarts in a row.
#   a rank exits 137 (killed)            no restart; the supervisor ends with 1. Check the memory first (docs/MEMORY.md).
#   any other code (for example 1)       no restart; the supervisor ends with 1. Read the logs: do not restart in a loop.
# Settings (environment): ZIG_POLL_S (default 10), ZIG_GRACE_S (default 30: how long to wait for the second rank to
# exit), ZIG_MAX_RESTARTS (default 3), ZIG_STABLE_S (default 600: after this time up, the restart count goes back to 0).
set -uo pipefail
# shellcheck source=zig_common.sh
. "$(dirname "$0")/zig_common.sh"

POLL_S=${ZIG_POLL_S:-10}
GRACE_S=${ZIG_GRACE_S:-30}
MAX_RESTARTS=${ZIG_MAX_RESTARTS:-3}
STABLE_S=${ZIG_STABLE_S:-600}

# The state of one rank: "running", "exited CODE" or "unknown" (no answer from ssh: that is no proof of an exit).
state() {  # $1 = head or worker
  local out
  if [ "$1" = head ]; then
    out=$(docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' tfv41-r0 2>/dev/null) || { echo "exited 1"; return; }
  else
    out=$(wssh "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' tfv41-r1" 2>/dev/null) || { echo unknown; return; }
  fi
  case "$out" in
    true*) echo running ;;
    *) echo "exited ${out#* }" ;;
  esac
}

restarts=0
up_since=$SECONDS
log "watching tfv41-r0 (head) and tfv41-r1 (worker); at most $MAX_RESTARTS restarts in a row"
while true; do
  s0=$(state head)
  s1=$(state worker)
  case "$s0:$s1" in
    running:running | running:unknown)
      [ $((SECONDS - up_since)) -lt "$STABLE_S" ] || restarts=0
      sleep "$POLL_S"
      continue
      ;;
  esac
  # A rank has exited. Wait for the other rank to exit too (it exits 75 within 2 to 8 s after a peer fault).
  waited=0
  while [ "$waited" -lt "$GRACE_S" ] && { [ "$s0" = running ] || [ "$s1" = running ]; }; do
    sleep "$POLL_S"
    waited=$((waited + POLL_S))
    s0=$(state head)
    s1=$(state worker)
  done
  c0=${s0#exited }; c1=${s1#exited }      # "running" and "unknown" stay as they are
  log "exit codes: rank 0 $c0, rank 1 $c1"
  if [ "$c0" = 0 ]; then
    log "rank 0 stopped clean: no restart"
    exit 0
  fi
  if [ "$c0" = 137 ] || [ "$c1" = 137 ]; then
    log "a rank was killed (137): no restart. Check docker inspect -f '{{.State.OOMKilled}}' and the memory (docs/MEMORY.md)."
    exit 1
  fi
  bad=0
  for c in "$c0" "$c1"; do
    case "$c" in 0 | 75 | running | unknown) ;; *) bad=1 ;; esac
  done
  if [ "$bad" = 1 ]; then
    log "a rank exited with an error code that is not 75: no restart. Read the logs; do not restart in a loop."
    exit 1
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "$restarts restarts in a row: give up. Read the logs."
    exit 1
  fi
  restarts=$((restarts + 1))
  log "peer fault (exit code 75): restart $restarts of $MAX_RESTARTS"
  "$ROOT/stop.sh" || true
  if ! "$ROOT/start.sh"; then
    log "the restart did not end with a good first completion: give up. Read the logs."
    exit 1
  fi
  up_since=$SECONDS
done
