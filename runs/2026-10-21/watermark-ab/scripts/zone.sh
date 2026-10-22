#!/usr/bin/env bash
# zone.sh: sum of the zone watermarks and the reserve, in MiB, from /proc/zoneinfo.
#   totalreserve = per zone (high watermark + largest lowmem_reserve), as the kernel counts it for MemAvailable.
ps=$(getconf PAGESIZE)
awk -v ps="$ps" '
  /^Node/ { z = $2 "/" $4 }
  /^ +min / { mn += $2 }
  /^ +low / { lo += $2 }
  /^ +high / { hi += $2; zhi[z] = $2 }
  /protection:/ { s = $0; gsub(/[^0-9 ]/, " ", s); n = split(s, a, " "); m = 0; for (i = 1; i <= n; i++) if (a[i] + 0 > m) m = a[i] + 0; prot[z] = m }
  END { for (k in zhi) tr += zhi[k] + prot[k];
        printf "min_MiB=%.0f low_MiB=%.0f high_MiB=%.0f totalreserve_MiB=%.0f\n", mn * ps / 1048576, lo * ps / 1048576, hi * ps / 1048576, tr * ps / 1048576 }
' /proc/zoneinfo
