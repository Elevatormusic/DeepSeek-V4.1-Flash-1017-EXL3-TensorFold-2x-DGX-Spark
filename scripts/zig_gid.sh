#!/usr/bin/env bash
# Print the RoCE v2 GID index of port 1 of one RDMA device: the first RoCE v2 entry that holds an IPv4-mapped address.
#   scripts/zig_gid.sh [DEVICE]        (default device: rocep1s0f1)
# The Zig engine reads this index at each start (TF_RDMA_GID_INDEX). After a link flap the index can move.
# Exit code 1 and no output when the device has no such entry.
# SYSFS is only for the offline tests.
D=${1:-rocep1s0f1}
BASE=${SYSFS:-/sys}/class/infiniband/$D/ports/1
for g in "$BASE"/gids/*; do
  [ -e "$g" ] || continue
  n=${g##*/}
  t=$(cat "$BASE/gid_attrs/types/$n" 2>/dev/null)
  v=$(cat "$g" 2>/dev/null)
  case "$t" in
    *v2*)
      case "$v" in
        0000:0000:0000:0000:0000:ffff:*) echo "$n"; exit 0 ;;
      esac
      ;;
  esac
done
exit 1
