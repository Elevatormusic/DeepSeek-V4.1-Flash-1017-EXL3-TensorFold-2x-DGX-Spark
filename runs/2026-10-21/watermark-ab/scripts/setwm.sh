#!/usr/bin/env bash
# setwm.sh MIN_KIB SCALE: set vm.min_free_kbytes and vm.watermark_scale_factor at runtime. Nothing else is changed.
# The user approved only these two values, with "sysctl -w", for the watermark A/B test.
set -u
min=${1:-}; scale=${2:-}
case "$min" in ''|*[!0-9]*) echo "bad min"; exit 2;; esac
case "$scale" in ''|*[!0-9]*) echo "bad scale"; exit 2;; esac
[ "$min" -ge 262144 ] && [ "$min" -le 2097152 ] || { echo "min out of range"; exit 2; }
[ "$scale" -ge 10 ] && [ "$scale" -le 100 ] || { echo "scale out of range"; exit 2; }
sudo -n sysctl -w vm.min_free_kbytes="$min" vm.watermark_scale_factor="$scale" > /dev/null || exit 1
echo "$(hostname) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes) watermark_scale_factor=$(cat /proc/sys/vm/watermark_scale_factor)"
