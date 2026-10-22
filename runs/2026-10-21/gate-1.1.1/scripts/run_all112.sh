#!/usr/bin/env bash
# Orchestrator on the head for the dsv41-1.1.1 gate. Usage: run_all111.sh <marker>
# The worker holder (flock on the worker, marker <marker>) is started first. This script waits until it holds the worker lock, then takes
# the head lock with the same marker, runs gate112.sh, and releases both locks and all rel111 containers on exit.
M=$1
R=~/rel111; L=$R/logs; W=10.100.80.2
log() { echo "[run_all111 $(date -u +%FT%TZ)] $*"; }
release() {
  log "release: stop containers, remove markers"
  (cd "$R" && IMAGE=tensorfold-dsv41:cb4ceaff2b8c ./tfv41-g.sh stop > "$L/final-stop.log" 2>&1)
  rm -f /tmp/$M.run /tmp/rel111.mem /tmp/rel111.arm /tmp/rel111.sent /tmp/rel111.trip /tmp/rel111.stop
  ssh -n -o BatchMode=yes $W "rm -f /tmp/$M.run /tmp/rel111.mem"
  sleep 8
  log "markers: head [$(ls /tmp/$M.* /tmp/rel111.* 2>/dev/null | tr '\n' ' ')] worker [$(ssh -n $W "ls /tmp/$M.* /tmp/rel111.* 2>/dev/null" | tr '\n' ' ')]"
  log "containers: head [$(docker ps -a --format '{{.Names}}' | grep rel111 | tr '\n' ' ')] worker [$(ssh -n $W "docker ps -a --format '{{.Names}}'" | grep rel111 | tr '\n' ' ')]"
}
trap release EXIT
log "waiting for the worker lock (marker $M)"
until ssh -n -o BatchMode=yes -o ConnectTimeout=10 $W "test -e /tmp/$M.held"; do sleep 10; done
log "worker lock held"
touch /tmp/$M.run
nohup setsid bash -c "while ! flock -w 10800 /tmp/spark-gpu.lock bash -c 'touch /tmp/$M.held; while [ -e /tmp/$M.run ]; do sleep 3; done; rm -f /tmp/$M.held'; do [ -e /tmp/$M.run ] || exit 0; done" > "$L/head-hold.log" 2>&1 < /dev/null &
until [ -e /tmp/$M.held ]; do sleep 5; done
log "head lock held"
ORDER=cb "$R/gate112.sh" > "$L/gate112-run.log" 2>&1
log "gate112 exit $?"
tail -n 80 "$L/gate112-run.log"
