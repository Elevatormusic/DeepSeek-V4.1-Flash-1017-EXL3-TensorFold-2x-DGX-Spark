#!/usr/bin/env bash
# restore.sh: put vm.min_free_kbytes and vm.watermark_scale_factor back to the values saved in orig.env.
# Safe to run many times. Exit code 0 only when both values read back equal to the saved values.
D=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
. "$D/orig.env" || { echo "no orig.env"; exit 3; }
sudo -n sysctl -w vm.min_free_kbytes="$ORIG_MIN" vm.watermark_scale_factor="$ORIG_SCALE" > /dev/null
now_min=$(cat /proc/sys/vm/min_free_kbytes); now_scale=$(cat /proc/sys/vm/watermark_scale_factor)
echo "$(hostname) restore: min_free_kbytes=$now_min (saved $ORIG_MIN) watermark_scale_factor=$now_scale (saved $ORIG_SCALE)"
[ "$now_min" = "$ORIG_MIN" ] && [ "$now_scale" = "$ORIG_SCALE" ]
