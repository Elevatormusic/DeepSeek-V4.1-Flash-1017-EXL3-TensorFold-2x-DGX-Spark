#!/usr/bin/env bash
# Compiles the engine's CUDA extensions into $CACHE_DIR/torch_extensions on the head and the worker, before any weights
# load. start.sh runs it first. When they are already built for this image, it takes about 20 s.
# Why: the engine compiles them at its first start after a new image, but only after the weights (~99 GiB) are in
# memory. The compiler then took the node below 1 GiB available, and earlyoom (scripts/host_tuning.sh) stopped the
# rank. With nothing else loaded, the compile has the memory it needs (~10-15 min the first time).
set -euo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

cg=${CGROUP_PARENT-tensorfold.slice}
build_cmd() {
  cat <<EOF
docker run --rm --gpus all ${cg:+--cgroup-parent $cg} --device /dev/infiniband \
  -v $CACHE_DIR:/cache -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e CUDA_CACHE_PATH=/cache/nv \
  --entrypoint python3 $IMAGE -c 'from tensorfold.cuda import rdma
from tensorfold.cuda.exl3 import experts, linear, prefetch
for m in (rdma, linear, experts, prefetch):
    m._ext()'
EOF
}

log "CUDA extensions: building on both nodes if needed (the first time: ~10-15 min)"
mkdir -p "$CACHE_DIR" "$LOGS"
wssh "mkdir -p '$CACHE_DIR'"
# A build that was stopped (earlyoom, Ctrl-C) leaves its lock file, and the next build then waits on it forever. Here
# no rank runs yet (start.sh checked that), so a lock is such a leftover: remove it.
unlock="find '$CACHE_DIR/torch_extensions' -maxdepth 2 -name lock -type f -print -delete 2>/dev/null || true"
for left in $(bash -c "$unlock") $(wssh "$unlock"); do log "removed the leftover build lock $left"; done
bash -c "$(build_cmd)" > "$LOGS/build-extensions-head.log" 2>&1 &
head_pid=$!
wrc=0
wssh "$(build_cmd)" > "$LOGS/build-extensions-worker.log" 2>&1 || wrc=$?
hrc=0
wait "$head_pid" || hrc=$?
if [ "$hrc" != 0 ] || [ "$wrc" != 0 ]; then
  log "CUDA extension build failed (head $hrc, worker $wrc): see $LOGS/build-extensions-*.log"
  exit 1
fi
log "CUDA extensions ready on both nodes"
