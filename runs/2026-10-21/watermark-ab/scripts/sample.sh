#!/usr/bin/env bash
# sample.sh OUTFILE: once per second, until /tmp/wmab.mem is removed, write one CSV row:
#   epoch, MemAvailable, MemFree, Cached (KiB), vmstat counters, allocstall sum, kswapd0 and kcompactd0 CPU ticks.
out=${1:?out file}
KEYS="compact_stall compact_fail compact_success compact_daemon_wake compact_migrate_scanned compact_free_scanned pgscan_kswapd pgscan_direct pgsteal_kswapd pgsteal_direct pageoutrun kswapd_low_wmark_hit_quickly kswapd_high_wmark_hit_quickly pswpin pswpout pgmajfault pgmigrate_success thp_fault_alloc"
kp=$(pgrep -x kswapd0 | head -n 1); cp=$(pgrep -x kcompactd0 | head -n 1)
echo "epoch,MemAvailable,MemFree,Cached,$(echo "$KEYS" | tr ' ' ','),allocstall,kswapd_ticks,kcompactd_ticks" > "$out"
while [ -e /tmp/wmab.mem ]; do
  now=$(date +%s.%N)
  kt=$(awk '{print $14 + $15}' "/proc/${kp:-0}/stat" 2> /dev/null); kc=$(awk '{print $14 + $15}' "/proc/${cp:-0}/stat" 2> /dev/null)
  awk -v now="$now" -v kt="${kt:-0}" -v kc="${kc:-0}" -v keys="$KEYS" '
    BEGIN { n = split(keys, K, " ") }
    FNR == NR { if ($1 == "MemAvailable:") ma = $2; if ($1 == "MemFree:") mf = $2; if ($1 == "Cached:") ca = $2; next }
    { v[$1] = $2; if ($1 ~ /^allocstall/) as += $2 }
    END { printf "%s,%s,%s,%s", now, ma, mf, ca; for (i = 1; i <= n; i++) printf ",%s", v[K[i]]; printf ",%s,%s,%s\n", as, kt, kc }
  ' /proc/meminfo /proc/vmstat >> "$out"
  sleep 1
done
