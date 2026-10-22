#!/usr/bin/env bash
# watch.sh ROLE MIN_AVAIL_KIB CONTAINER: safety watchdog for one node. Runs while /tmp/wmab.watch exists.
# It acts only while /tmp/wmab.arm exists (it holds the epoch of the arming).
# It stops CONTAINER (a container of this test) and writes /tmp/wmab.trip when:
#   - MemAvailable is under MIN_AVAIL_KIB (head only; the worker passes 0), or
#   - earlyoom logged "sending SIG..." or "low memory" since the arming.
role=${1:?role}; minkib=${2:-0}; ctr=${3:?container}
n=0
while [ -e /tmp/wmab.watch ]; do
  if [ -e /tmp/wmab.arm ] && [ ! -e /tmp/wmab.trip ]; then
    reason=""
    kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$minkib" -gt 0 ] && [ "$kib" -lt "$minkib" ]; then reason="MemAvailable $kib KiB under $minkib KiB"; fi
    n=$((n + 1))
    if [ -z "$reason" ] && [ $((n % 2)) = 0 ]; then
      since=$(date -d "@$(cut -d. -f1 /tmp/wmab.arm)" '+%Y-%m-%d %H:%M:%S')
      if journalctl -u earlyoom --since "$since" --no-pager 2> /dev/null | grep -a -q -E 'sending SIG|low memory'; then reason="earlyoom logged a stop"; fi
    fi
    if [ -n "$reason" ]; then
      echo "TRIP $(date -u +%T) $role: $reason" | tee /tmp/wmab.trip
      docker stop -t 30 "$ctr" > /dev/null 2>&1
      pkill -f 'wmab/v41/' 2> /dev/null   # only the clients of this test (their scripts live in ~/wmab/v41)
    fi
  fi
  sleep 1
done
