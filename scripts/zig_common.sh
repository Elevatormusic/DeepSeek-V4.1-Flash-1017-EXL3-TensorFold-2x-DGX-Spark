# Shared by the Zig scripts (zig_start.sh, zig_inputs.sh, zig_build_image.sh, zig_supervise.sh). Source it; do not run it.
# It reads .env through common.sh, then sets the defaults and the paths of the Zig engine.
# shellcheck shell=bash
# shellcheck disable=SC2034  # the scripts that source this file use these variables
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ZIG_IMAGE=${ZIG_IMAGE:-tensorfold-dsv41-zig:d58d25cc}
ZIG_RANK_PORT=${ZIG_RANK_PORT:-29620}
ZIG_RDMA_GID_INDEX=${ZIG_RDMA_GID_INDEX:-auto}
# Three folders in CACHE_DIR, on both nodes. The names show the pack: a new pack gets new names.
ZIG_KIT=$CACHE_DIR/zig-kit                       # the kernel kit (aot/, cubins/, vision/, ...), 0.6 GiB
ZIG_DATA=$CACHE_DIR/zig-data                     # dsv41_token_map.json
ZIG_RANK_CACHE=$CACHE_DIR/zig-rank-cache-28      # one weight file for each node, 96 GiB
# The engine keeps 110 GiB of MemAvailable as the start condition (docs/ZIG.md). Both nodes need the same.
ZIG_NEED_GIB=110
# Context and pool, and the streams of the lanes. Only 1,048,576 tokens and 4 streams have a test (docs/ZIG.md).
# shellcheck disable=SC2034  # used by the scripts that source this file
ZIG_FIRST_DEVICE=${RDMA_DEVICES%%,*}

# wrun keeps stdin open (wssh closes it), so a script can go to the worker on stdin.
wrun() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

# The RoCE v2 GID index of this node ($1 = side: head or worker). A number in ZIG_RDMA_GID_INDEX wins.
zig_gid() {
  if [ "$ZIG_RDMA_GID_INDEX" != auto ]; then echo "$ZIG_RDMA_GID_INDEX"; return 0; fi
  if [ "$1" = head ]; then
    bash "$ROOT/scripts/zig_gid.sh" "$ZIG_FIRST_DEVICE"
  else
    wrun bash -s -- "$ZIG_FIRST_DEVICE" < "$ROOT/scripts/zig_gid.sh"
  fi
}
