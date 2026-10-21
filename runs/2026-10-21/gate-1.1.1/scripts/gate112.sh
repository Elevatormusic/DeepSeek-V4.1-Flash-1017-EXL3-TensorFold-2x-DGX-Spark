#!/usr/bin/env bash
# dsv41-1.1.1 gate, run on the head inside one GPU lock window (both locks are already held by run_all111.sh).
# Step b: baseline 1.1.0 image fc9200a44463 (GID index pinned to 3).  Step c: candidate cb4ceaff2b8c (GID index unset: auto-detect).
# Same order in both steps, no restart inside a step:
#   start, smoke, suite (recipe main), grammar check, decode bench, cold 32K prefill (B and D), live checks,
#   long 4-stream run (20K/50K/100K/200K), then (candidate only) SIGTERM during a prefill.
# MemAvailable is sampled every 1 s on both nodes. The head watchdog stops both containers if the head falls under 1.25 GiB.
set -u
R=~/rel111; V=$R/v41; L=$R/logs; W=10.100.80.2
BASE=fc9200a44463; CAND=cb4ceaff2b8c
MIN_HEAD_KIB=$((1280 * 1024))
LA=$R/tfv41-g.sh
log() { echo "[gate111 $(date -u +%H:%M:%SZ)] $*"; }
idle() {
  echo "head: $(awk '/MemAvailable/{printf "%d MiB", $2/1024}' /proc/meminfo) | worker: $(ssh -n $W "awk '/MemAvailable/{printf \"%d MiB\", \$2/1024}' /proc/meminfo")"
}
memstart() {  # $1 = label
  touch /tmp/rel111.mem; ssh -n $W 'touch /tmp/rel111.mem'
  ( while [ -e /tmp/rel111.mem ]; do
      kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
      echo "$(date -u +%T) $kib"
      if [ -e /tmp/rel111.arm ] && [ "$kib" -lt "$MIN_HEAD_KIB" ] && [ ! -e /tmp/rel111.trip ]; then
        touch /tmp/rel111.trip /tmp/rel111.stop
        echo "TRIP $(date -u +%T) head MemAvailable $kib KiB: stopping the rel111 containers" >> "$L/trip.log"
        docker stop -t 30 rel111-r0 > /dev/null 2>&1 &
        ssh -n $W 'docker stop -t 30 rel111-r1 > /dev/null 2>&1' &
        pkill -f 'capdrive.py|measure_bd.py|v41_suite.py|v41_bench.py|v41_grammar_check.py|live_checks.py|prefill_req.py'
      fi
      sleep 1
    done ) > "$L/mem-$1-head.log" 2>&1 &
  ssh -n $W 'while [ -e /tmp/rel111.mem ]; do echo "$(date -u +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 1; done' > "$L/mem-$1-worker.log" 2>&1 &
}
memstop() { rm -f /tmp/rel111.mem /tmp/rel111.arm; ssh -n $W 'rm -f /tmp/rel111.mem'; sleep 3; }
minwin() {  # $1 = label, $2 = window start HH:MM:SS, $3 = window end (optional)
  for n in head worker; do
    echo "  $n: $(awk -v s="$2" -v e="${3:-99:99:99}" '$1 >= s && $1 <= e { if (m == "" || $2 < m) { m = $2; t = $1 } } END { printf "lowest %d MiB at %s", m / 1024, t }' "$L/mem-$1-$n.log")"
  done
}
earlyoom_lines() {  # $1 = since
  echo "earlyoom head:"; journalctl -u earlyoom --since "$1" --no-pager 2>&1 | grep -a -E "sending SIG|low memory" | cut -c1-200
  echo "earlyoom worker:"; ssh -n $W "journalctl -u earlyoom --since '$1' --no-pager 2>&1 | grep -a -E 'sending SIG|low memory' | cut -c1-200"
}
tripped() { [ -e /tmp/rel111.trip ]; }
prebuild() {  # $1 = tag: build the CUDA extensions on both nodes before any weights load
  local T=$1 CACHE=$HOME/tfv41-cache
  local cmd="mkdir -p $CACHE && find $CACHE/torch_extensions-$T -maxdepth 2 -name lock -type f -print -delete 2>/dev/null; docker run --rm --name rel111-prebuild --gpus all --cgroup-parent tensorfold.slice --device /dev/infiniband -v $CACHE:/cache -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions-$T -e CUDA_CACHE_PATH=/cache/nv --entrypoint python3 tensorfold-dsv41:$T -c 'import time; t=time.time()
from tensorfold.cuda import rdma
from tensorfold.cuda.exl3 import experts, linear, prefetch
for m in (rdma, linear, experts, prefetch):
    m._ext()
print(\"extensions ready in\", round(time.time()-t), \"s\")'"
  bash -c "$cmd" > "$L/prebuild-$T-head.log" 2>&1 &
  local hp=$! wrc=0 hrc=0
  ssh -n -o BatchMode=yes $W "$cmd" > "$L/prebuild-$T-worker.log" 2>&1 || wrc=$?
  wait "$hp" || hrc=$?
  log "prebuild $T: head rc=$hrc worker rc=$wrc; $(tail -n 1 "$L/prebuild-$T-head.log") / $(tail -n 1 "$L/prebuild-$T-worker.log")"
  [ "$hrc" = 0 ] && [ "$wrc" = 0 ]
}
gid_report() {  # $1 = label: RDMA and NCCL lines of both rank logs
  { echo "head:"; docker logs rel111-r0 2>&1 | grep -a -E "RDMA GID index|NCCL settings by default|decode gathers over RDMA|gathers stay on NCCL|RDMA gather unavailable" | cut -c1-240
    echo "worker:"; ssh -n $W "docker logs rel111-r1 2>&1" | grep -a -E "RDMA GID index|NCCL settings by default|decode gathers over RDMA|gathers stay on NCCL|RDMA gather unavailable" | cut -c1-240; } | tee "$L/$1-gid.log"
}
logsummary() {  # $1 = prefix
  docker logs rel111-r0 > "$L/$1-r0-live.log" 2>&1; ssh -n $W "docker logs rel111-r1" > "$L/$1-r1-live.log" 2>&1
  log "$1 admission waits: r0=$(grep -ac 'admission waits' "$L/$1-r0-live.log") r1=$(grep -ac 'admission waits' "$L/$1-r1-live.log"); late captures: r0=$(grep -ac 'captured while serving' "$L/$1-r0-live.log") r1=$(grep -ac 'captured while serving' "$L/$1-r1-live.log"); WARNING: r0=$(grep -ac WARNING "$L/$1-r0-live.log") r1=$(grep -ac WARNING "$L/$1-r1-live.log"); terminate: r0=$(grep -ac 'terminate called' "$L/$1-r0-live.log") r1=$(grep -ac 'terminate called' "$L/$1-r1-live.log"); Traceback/ERROR: r0=$(grep -acE 'Traceback|ERROR' "$L/$1-r0-live.log") r1=$(grep -acE 'Traceback|ERROR' "$L/$1-r1-live.log")"
}

sigterm_test() {  # candidate: SIGTERM to rank 0 during a prefill
  rm -f /tmp/rel111.arm
  rm -f /tmp/rel111.sent; log "SIGTERM test: starting a 100K-token prefill"
  ( cd "$V" && timeout 600 python3 -I -u prefill_req.py 100000 > "$L/c-sigterm-client.log" 2>&1 ) &
  local cp=$!
  local k=0
  while [ ! -e /tmp/rel111.sent ] && [ $k -lt 120 ]; do sleep 1; k=$((k + 1)); done
  sleep 10
  if kill -0 "$cp" 2> /dev/null; then log "SIGTERM test: request sent $(( $(date +%s) - $(stat -c %Y /tmp/rel111.sent) )) s ago and not finished: rank 0 is in the prefill"; else log "SIGTERM test: client already ended before the signal (NOT a valid prefill test)"; fi
  docker logs --tail 3 rel111-r0 2>&1 | cut -c1-200 | tee "$L/c-sigterm-before.log"
  local t0; t0=$(date +%s.%N)
  docker kill --signal=TERM rel111-r0 > /dev/null 2>&1
  local i=0
  while [ "$(docker inspect -f '{{.State.Running}}' rel111-r0 2>/dev/null)" = "true" ] && [ $i -lt 120 ]; do sleep 0.5; i=$((i + 1)); done
  local t1; t1=$(date +%s.%N)
  local code; code=$(docker inspect -f '{{.State.ExitCode}}' rel111-r0 2>/dev/null)
  docker logs rel111-r0 > "$L/c-sigterm-r0.log" 2>&1
  ssh -n $W "docker logs rel111-r1" > "$L/c-sigterm-r1.log" 2>&1
  kill "$cp" 2> /dev/null; wait "$cp" 2> /dev/null
  log "SIGTERM test: rank 0 exit code $code after $(echo "$t1 - $t0" | bc) s; 'terminate called' r0=$(grep -ac 'terminate called' "$L/c-sigterm-r0.log") r1=$(grep -ac 'terminate called' "$L/c-sigterm-r1.log"); worker running: $(ssh -n $W "docker inspect -f '{{.State.Running}} exit={{.State.ExitCode}}' rel111-r1" 2>&1)"
  log "SIGTERM test: last r0 lines:"; tail -n 6 "$L/c-sigterm-r0.log" | cut -c1-220
}

step() {  # $1 = label (b or c), $2 = tag, $3 = GID mode, $4 = sigterm test (yes/no)
  local P=$1 TAG=$2 GM=$3 SIG=$4 T0 LS LSJ LE ML
  export IMAGE=tensorfold-dsv41:$TAG GIDMODE=$GM
  rm -f /tmp/rel111.trip /tmp/rel111.stop /tmp/rel111.arm
  export V41_BASE_URL=http://172.17.0.1:8888/v1 V41_ROUTER_URL=http://172.17.0.1:8888/v1 V41_ROUTER_API_KEY=direct-engine-no-key DGX_ROUTER_API_KEY=direct-engine-no-key
  log "STEP $P ($TAG, GID mode $GM): idle memory before the start: $(idle)" | tee -a "$L/idle-$P.log"
  "$LA" stop > "$L/$P-stop-pre.log" 2>&1
  prebuild "$TAG" || { log "prebuild failed for $TAG"; return 1; }
  T0=$(date -u +%T); ML=$P-full
  memstart "$ML"
  "$LA" start > "$L/$P-start.log" 2>&1; echo "START_EXIT=$?" >> "$L/$P-start.log"
  tail -n 2 "$L/$P-start.log"
  if ! grep -q serving "$L/$P-start.log"; then
    memstop; log "$P not serving"; gid_report "$P-failed"
    if [ "$GM" = auto ]; then
      "$LA" stop > "$L/$P-stop-fail.log" 2>&1
      log "auto GID start failed: item 7 FAILS; retrying with the pinned index so that the other checks can run"
      GIDMODE=pinned; GM=pinned; export GIDMODE
      T0=$(date -u +%T); ML=$P-full2; memstart "$ML"
      "$LA" start > "$L/$P-start2.log" 2>&1; echo "START_EXIT=$?" >> "$L/$P-start2.log"
      grep -q serving "$L/$P-start2.log" || { memstop; log "$P not serving after the retry"; return 1; }
    else
      return 1
    fi
  fi
  gid_report "$P"
  touch /tmp/rel111.arm
  log "$P idle memory after the start: $(idle)" | tee -a "$L/idle-$P.log"
  cd "$V"
  python3 -I smoke.py > "$L/$P-smoke.log" 2>&1; echo "SMOKE_EXIT=$?" >> "$L/$P-smoke.log"
  log "$P smoke: $(tail -n 1 "$L/$P-smoke.log")"
  timeout 3600 python3 -I -u v41_suite.py > "$L/$P-suite-$TAG.log" 2>&1; echo "SUITE_EXIT=$?" >> "$L/$P-suite-$TAG.log"
  cp suite-results.json "$L/$P-suite-results-$TAG.json"
  log "$P suite: $(grep -ac '^\[PASS\]' "$L/$P-suite-$TAG.log") pass, $(grep -ac '^\[FAIL\]' "$L/$P-suite-$TAG.log") fail: $(tail -n 3 "$L/$P-suite-$TAG.log" | tr '\n' ' ')"
  tripped && { log "$P TRIPPED (head memory) after the suite"; memstop; return 1; }
  timeout 1800 python3 -I -u v41_grammar_check.py 2 > "$L/$P-grammar-$TAG.log" 2>&1; echo "CHECK_EXIT=$?" >> "$L/$P-grammar-$TAG.log"
  log "$P grammar: $(tail -n 2 "$L/$P-grammar-$TAG.log" | tr '\n' ' ')"
  timeout 3600 python3 -I -u v41_bench.py > "$L/$P-bench-$TAG.log" 2>&1; echo "BENCH_EXIT=$?" >> "$L/$P-bench-$TAG.log"
  cp bench-results.json "$L/$P-bench-results-$TAG.json"
  log "$P bench done: $(tail -n 1 "$L/$P-bench-$TAG.log")"
  tripped && { log "$P TRIPPED (head memory) after the bench"; memstop; return 1; }
  # cold 32K prefill: a copy of the pf32 kit (B random natural text, D kit_bench words), 3 interleaved runs each
  cd $R/pf32; rm -f results-$P.jsonl
  timeout 1800 python3 -I -u measure_bd.py prompts.json results-$P.jsonl > "$L/$P-prefill-$TAG.log" 2>&1; echo "MEASURE_EXIT=$?" >> "$L/$P-prefill-$TAG.log"
  cp results-$P.jsonl "$L/$P-prefill-results-$TAG.jsonl"
  log "$P cold prefill: $(grep -a -E '^ *(32000|32768) ' "$L/$P-prefill-$TAG.log" | tr '\n' ' ')"
  cd "$V"
  tripped && { log "$P TRIPPED (head memory) after the prefill"; memstop; return 1; }
  # live checks of the new fixes (run on both images; the baseline is expected to fail some)
  timeout 1800 python3 -I -u live_checks.py "$L/$P-live-checks-$TAG.json" > "$L/$P-live-$TAG.log" 2>&1; echo "LIVE_EXIT=$?" >> "$L/$P-live-$TAG.log"
  log "$P live checks: $(grep -a LIVE_CHECKS "$L/$P-live-$TAG.log")"
  log "$P idle memory before the long run: $(idle)" | tee -a "$L/idle-$P.log"
  LS=$(date -u +%T); LSJ=$(date -u +"%Y-%m-%d %H:%M:%S")
  log "$P long run start $LS"
  timeout 5400 python3 -I -u capdrive.py long --lens 20000,50000,100000,200000 --gen 1500 --seed 7 > "$L/$P-long4-$TAG.log" 2>&1; echo "LONG_EXIT=$?" >> "$L/$P-long4-$TAG.log"
  LE=$(date -u +%T)
  cat "$L/$P-long4-$TAG.log"
  log "$P long run window $LS - $LE; lowest MemAvailable in the long run:"; minwin "$ML" "$LS" "$LE" | tee "$L/$P-long-minmem.log"
  log "$P lowest MemAvailable in the whole step so far:"; minwin "$ML" "$T0"
  earlyoom_lines "$LSJ" | tee "$L/$P-earlyoom.log"
  tripped && { log "$P TRIPPED (head memory) in or after the long run: $(cat "$L/trip.log" 2> /dev/null | tail -n 1)"; logsummary "$P"; memstop; return 1; }
  logsummary "$P"
  curl -s -m 10 --interface 127.0.0.1 http://172.17.0.1:8888/v1/models > /dev/null && log "$P server alive after the long run" || log "$P SERVER DEAD after the long run"
  docker ps --format '{{.Names}} {{.Status}}' | grep rel111; ssh -n $W "docker ps --format '{{.Names}} {{.Status}}'" | grep rel111
  if [ "$SIG" = yes ]; then
    if curl -s -m 10 --interface 127.0.0.1 http://172.17.0.1:8888/v1/models > /dev/null; then sigterm_test; else log "SIGTERM test skipped: engine not serving"; fi
  fi
  memstop
  cd "$R"; "$LA" stop > "$L/$P-stop.log" 2>&1
  return 0
}

cd "$R" || exit 1
log "GATE START. images: head $(docker image inspect tensorfold-dsv41:$BASE --format '{{.Id}}' | cut -c8-19) / $(docker image inspect tensorfold-dsv41:$CAND --format '{{.Id}}' | cut -c8-19); worker $(ssh -n $W "docker image inspect tensorfold-dsv41:$BASE --format '{{.Id}}'" | cut -c8-19) / $(ssh -n $W "docker image inspect tensorfold-dsv41:$CAND --format '{{.Id}}'" | cut -c8-19)"
if [ "${ORDER:-bc}" = cb ]; then
  step e $CAND auto yes
  log "step e finished"
  step d $BASE pinned no
  log "step d finished"
else
  step b $BASE pinned no
  log "step b finished"
  step c $CAND auto yes
  log "step c finished"
fi
log "GATE111_DONE"
