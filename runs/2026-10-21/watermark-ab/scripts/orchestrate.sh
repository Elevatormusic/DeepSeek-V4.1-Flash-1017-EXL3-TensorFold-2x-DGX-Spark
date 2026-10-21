#!/usr/bin/env bash
# orchestrate.sh MARKER: watermark A/B test on both DGX Spark nodes (runs on the head).
# 1. Queue on the GPU lock: the worker first, then the head, with the unique marker MARKER.
# 2. For each setting B, V1, V2, V3, B2 (B2 = the baseline again): set the two vm values on both nodes at runtime,
#    start the engine (fresh), run the stall test twice, then the 4-stream long run once. Sample at 1 s.
# 3. On exit, error or abort: stop the test containers, restore the saved vm values, verify, free both locks.
# Only vm.min_free_kbytes and vm.watermark_scale_factor are changed (setwm.sh / restore.sh).
set -u
M=${1:?marker}
R=$HOME/wmab; L=$R/logs; W=10.100.80.2
export IMAGE=${IMAGE:-tensorfold-dsv41:cb4ceaff2b8c}
export GIDMODE=${GIDMODE:-auto}
LA=$R/tfv41-w.sh
HEAD_MIN_KIB=$((1280 * 1024))          # stop a setting when head MemAvailable is under 1.25 GiB
WINDOW_LIMIT=${WINDOW_LIMIT:-6600}     # at most 1 h 50 min of lock time
DEADMAN_S=${DEADMAN_S:-9900}
PY="python3 -I"
mkdir -p "$L"
# shellcheck disable=SC1091
. "$R/orig.env"
SETTINGS="${SETTINGS:-B:$ORIG_MIN:$ORIG_SCALE V1:1048576:25 V2:524288:50 V3:524288:25 B2:$ORIG_MIN:$ORIG_SCALE}"
REF_SLOW=${REF_SLOW:-0}; REF_MAX=${REF_MAX:-0}

log() { echo "[wmab $(date -u +%H:%M:%SZ)] $*"; }
wrun() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$W" "$@"; }
now() { date +%s.%N; }
T_LOCK=0

release() {
  trap - EXIT
  log "release: stop test containers, restore the vm values, free the locks"
  rm -f /tmp/wmab.arm /tmp/wmab.mem; wrun "rm -f /tmp/wmab.arm /tmp/wmab.mem"
  "$LA" stop > "$L/final-stop.log" 2>&1
  local ok=1
  "$R/restore.sh" || ok=0
  wrun "$R/restore.sh" || ok=0
  [ $ok = 1 ] && log "RESTORE_OK both nodes read back the saved values" || log "RESTORE_FAILED: run $R/restore.sh on both nodes"
  rm -f /tmp/wmab.watch; wrun "rm -f /tmp/wmab.watch"
  pkill -f "wmab-deadma[n]-$M" 2> /dev/null; wrun "pkill -f 'wmab-deadma[n]-$M'" 2> /dev/null
  rm -f "/tmp/$M.run" /tmp/wmab.arm /tmp/wmab.trip; wrun "rm -f /tmp/$M.run /tmp/wmab.arm /tmp/wmab.trip"
  sleep 8
  log "markers: head [$(ls /tmp/$M.* /tmp/wmab.* 2> /dev/null | tr '\n' ' ')] worker [$(wrun "ls /tmp/$M.* /tmp/wmab.* 2> /dev/null" | tr '\n' ' ')]"
  log "containers: head [$(docker ps -a --format '{{.Names}}' | grep wmab | tr '\n' ' ')] worker [$(wrun "docker ps -a --format '{{.Names}}'" | grep wmab | tr '\n' ' ')]"
  log "vm values: head $(cat /proc/sys/vm/min_free_kbytes)/$(cat /proc/sys/vm/watermark_scale_factor) worker $(wrun 'echo $(cat /proc/sys/vm/min_free_kbytes)/$(cat /proc/sys/vm/watermark_scale_factor)')"
}
trap release EXIT
trap 'log "signal: abort"; exit 130' INT TERM HUP

apply_both() {  # $1 = min_free_kbytes, $2 = watermark_scale_factor. 0 only when both nodes read back the values.
  "$R/setwm.sh" "$1" "$2" || return 1
  wrun "$R/setwm.sh $1 $2" || return 1
  [ "$(cat /proc/sys/vm/min_free_kbytes)" = "$1" ] && [ "$(cat /proc/sys/vm/watermark_scale_factor)" = "$2" ] || return 1
  wrun "[ \$(cat /proc/sys/vm/min_free_kbytes) = $1 ] && [ \$(cat /proc/sys/vm/watermark_scale_factor) = $2 ]"
}
restore_both() { "$R/restore.sh"; wrun "$R/restore.sh"; }

# ---- 1. lock ----
if [ "$(wrun "cat $R/orig.env")" != "$(cat "$R/orig.env")" ]; then log "orig.env differs between the nodes: stop"; exit 1; fi
log "queue: worker lock (marker $M)"
wrun "setsid nohup $R/hold.sh $M > /tmp/$M.holder.log 2>&1 < /dev/null &"
n=0
until wrun "test -e /tmp/$M.held"; do sleep 10; n=$((n + 1)); [ $((n % 60)) = 0 ] && log "still waiting for the worker lock ($((n / 6)) min)"; done
log "worker lock held; queue: head lock"
setsid nohup "$R/hold.sh" "$M" > "/tmp/$M.holder.log" 2>&1 < /dev/null &
n=0
until [ -e "/tmp/$M.held" ]; do sleep 5; n=$((n + 1)); [ $((n % 120)) = 0 ] && log "still waiting for the head lock ($((n / 12)) min)"; done
T_LOCK=$(date +%s)
log "both locks held at $(date -u +%T)Z"

# ---- 2. guards ----
for c in $(docker ps --format '{{.Names}}'); do case "$c" in zigdsv41*|dsv41-*|rel11*|tfv41*) log "head still runs $c: stop"; exit 1;; esac; done
wrun "docker ps --format '{{.Names}}'" | grep -qE '^(zigdsv41|dsv41-|rel11|tfv41)' && { log "worker still runs another engine: stop"; exit 1; }
{ echo "kernel $(uname -r)"; echo "head: $(free -m | awk '/Mem:/{print "free " $4 ", cache " $6 ", avail " $7}')"; for k in vm.swappiness vm.compaction_proactiveness vm.extfrag_threshold vm.min_free_kbytes vm.watermark_scale_factor; do echo "head $k=$(cat /proc/sys/${k//./\/})"; done
  echo "head thp $(cat /sys/kernel/mm/transparent_hugepage/enabled)"; echo "worker thp $(wrun 'cat /sys/kernel/mm/transparent_hugepage/enabled')"
  echo "image $IMAGE gid $GIDMODE"; } > "$L/env.txt"
wrun "setsid nohup bash -c 'sleep $DEADMAN_S; $R/restore.sh; : wmab-deadman-$M' > $R/deadman.log 2>&1 < /dev/null &"
setsid nohup bash -c "sleep $DEADMAN_S; $R/restore.sh; : wmab-deadman-$M" > "$R/deadman.log" 2>&1 < /dev/null &
touch /tmp/wmab.watch; wrun "touch /tmp/wmab.watch"
setsid nohup "$R/watch.sh" head "$HEAD_MIN_KIB" wmab-r0 >> "$L/watch-head.log" 2>&1 < /dev/null &
wrun "mkdir -p $R/out; setsid nohup $R/watch.sh worker 0 wmab-r1 >> $R/out/watch-worker.log 2>&1 < /dev/null &"

win() { printf '%s\t%s\t%s\t%s\n' "$2" "$3" "$4" "$5" >> "$1/windows.tsv"; }   # dir name start end rc

check_trip() {  # 0 if a watchdog tripped
  [ -e /tmp/wmab.trip ] || wrun "test -e /tmp/wmab.trip"
}

run_setting() {  # $1 name, $2 min_free_kbytes, $3 watermark_scale_factor
  local NAME=$1 MIN=$2 SCALE=$3 SD=$L/$1 CHECK=1 ABORT=0 ta tb q slow mx rc p PF1 T0 SINCE
  case "$NAME" in B|B2) CHECK=0;; esac
  mkdir -p "$SD"; : > "$SD/windows.tsv"; : > "$SD/notes.txt"; rm -f "$SD/trip.txt"
  log "=== $NAME: min_free_kbytes=$MIN watermark_scale_factor=$SCALE ($(( $(date +%s) - T_LOCK )) s into the window)"
  rm -f /tmp/wmab.arm /tmp/wmab.trip; wrun "rm -f /tmp/wmab.arm /tmp/wmab.trip"
  touch /tmp/wmab.mem; wrun "touch /tmp/wmab.mem"
  setsid nohup "$R/sample.sh" "$SD/mem-head.csv" > /dev/null 2>&1 < /dev/null &
  wrun "setsid nohup $R/sample.sh $R/out/mem-$NAME.csv > /dev/null 2>&1 < /dev/null &"
  sleep 4
  T0=$(now); SINCE=$(date '+%Y-%m-%d %H:%M:%S')
  if ! apply_both "$MIN" "$SCALE" | tee -a "$SD/notes.txt"; then log "FATAL: could not set or read back the values"; exit 1; fi
  sleep 8
  "$R/zone.sh" > "$SD/zone-head.txt"; wrun "$R/zone.sh" > "$SD/zone-worker.txt"
  log "$NAME zones: head $(cat "$SD/zone-head.txt") | worker $(cat "$SD/zone-worker.txt")"
  ta=$(now); sleep 6; tb=$(now); win "$SD" idle0 "$ta" "$tb" 0
  SPILL_TAG=$NAME "$LA" start > "$SD/start.log" 2>&1
  if ! grep -q 'serving' "$SD/start.log"; then
    log "$NAME: engine did not start"; echo "engine did not start" >> "$SD/notes.txt"; ABORT=2
  else
    date +%s.%N > /tmp/wmab.arm; wrun 'date +%s.%N > /tmp/wmab.arm'
    sleep 30
    ta=$(now); sleep 20; tb=$(now); win "$SD" idle "$ta" "$tb" 0
    log "$NAME idle (engine loaded): head $(awk -F, -v a="$ta" -v b="$tb" 'NR>1 && $1>=a && $1<=b {s+=$2; n++} END {if (n) printf "%d MiB", s/n/1024}' "$SD/mem-head.csv")"
    PF1=$(now)
    for p in 1 2; do
      ta=$(now)
      "$R/pfrun.sh" "$SD" "p$p"; rc=$?
      tb=$(now); win "$SD" "pf$p" "$ta" "$tb" "$rc"
      docker logs -t wmab-r0 > "$SD/live-r0.log" 2>&1
      q=$($PY "$R/analyze.py" quick "$SD/live-r0.log" "$ta" "$tb")
      log "$NAME stall test pass $p: $q (client rc $rc)"; echo "pass $p: $q rc $rc" >> "$SD/notes.txt"
      if check_trip; then ABORT=1; echo "tripped in pass $p" >> "$SD/notes.txt"; break; fi
      slow=$(echo "$q" | sed -n 's/.*slow=\([0-9]*\).*/\1/p'); mx=$(echo "$q" | sed -n 's/.*max=\([0-9.]*\).*/\1/p')
      if [ "$NAME" = B ]; then
        [ "${slow:-0}" -gt "$REF_SLOW" ] && REF_SLOW=$slow
        awk -v a="${mx:-0}" -v b="$REF_MAX" 'BEGIN {exit !(a > b)}' && REF_MAX=$mx
      fi
      if [ $CHECK = 1 ]; then
        if awk -v s="${slow:-0}" -v rs="$REF_SLOW" -v m="${mx:-0}" -v rm="$REF_MAX" \
             'BEGIN {ts = 3 * rs; if (ts < 8) ts = 8; tm = 2 * rm; if (tm < 6) tm = 6; exit !(s >= ts || m >= tm)}'; then
          ABORT=1; log "$NAME: stalls clearly worse than B (slow $slow, max $mx; B ref slow $REF_SLOW, max $REF_MAX): stop this setting"
          echo "stopped: stalls clearly worse than B (slow $slow, max $mx; B ref slow $REF_SLOW max $REF_MAX)" >> "$SD/notes.txt"; break
        fi
      fi
    done
    if [ $ABORT = 0 ]; then
      ta=$(now)
      ( cd "$R/v41" && timeout 5400 python3 -I -u "$R/v41/capdrive.py" long --lens 20000,50000,100000,200000 --gen 1500 --seed 7 > "$SD/long4.log" 2>&1; echo "LONG_EXIT=$?" >> "$SD/long4.log" )
      tb=$(now); win "$SD" long "$ta" "$tb" "$(sed -n 's/^LONG_EXIT=//p' "$SD/long4.log")"
      log "$NAME long run: $(grep -a 'elapsed' "$SD/long4.log" | tail -n 1) $(tail -n 1 "$SD/long4.log")"
      check_trip && { ABORT=1; echo "tripped in the long run" >> "$SD/notes.txt"; }
    fi
    win "$SD" all "$PF1" "$(now)" 0
  fi
  # ---- collect ----
  rm -f /tmp/wmab.arm; wrun "rm -f /tmp/wmab.arm"
  { cat /tmp/wmab.trip 2> /dev/null; wrun "cat /tmp/wmab.trip 2> /dev/null"; } > "$SD/trip.txt"
  sleep 3
  rm -f /tmp/wmab.mem; wrun "rm -f /tmp/wmab.mem"; sleep 3
  scp -q -o BatchMode=yes "$W:$R/out/mem-$NAME.csv" "$SD/mem-worker.csv"
  journalctl -u earlyoom --since "$SINCE" --no-pager 2> /dev/null | grep -a -E 'sending SIG|low memory' | cut -c1-220 > "$SD/earlyoom-head.txt"
  wrun "journalctl -u earlyoom --since '$SINCE' --no-pager 2> /dev/null | grep -a -E 'sending SIG|low memory' | cut -c1-220" > "$SD/earlyoom-worker.txt"
  SAVE_AS="$SD/rank" "$LA" stop > "$SD/stop.log" 2>&1
  $PY "$R/analyze.py" setting "$SD" > "$SD/summary.txt" 2>&1
  if [ -s "$SD/trip.txt" ] || [ $ABORT != 0 ]; then
    log "$NAME stopped early (abort $ABORT, trip: $(tr '\n' ' ' < "$SD/trip.txt")): put B back"
    restore_both
  fi
  log "$NAME done"
}

# ---- 3. run ----
for s in $SETTINGS; do
  IFS=: read -r name min scale <<< "$s"
  if [ $(( $(date +%s) - T_LOCK )) -gt $((WINDOW_LIMIT - 900)) ]; then log "window too short for $name and later: stop here"; break; fi
  run_setting "$name" "$min" "$scale"
done
log "all settings done"
