#!/usr/bin/env bash
# Build the Zig engine image on the head at the pinned fork commit (ZIG_TF_COMMIT), copy it to the worker, and check that
# both nodes hold the same image ID and the label tf.ref.
#   scripts/zig_build_image.sh
# Steps: (1) the base image tensorfold-dsv41:$ZIG_BASE_COMMIT, built from docker/Dockerfile when it is missing; (2) the
# Zig image, from docker/Dockerfile.zig; (3) docker save | docker load to the worker; (4) the checks.
# The compile needs about 40 GiB of free memory on the head: stop the server first (./stop.sh). Nothing in this script
# touches a running engine; it refuses to run when a rank container exists.
set -euo pipefail
# shellcheck source=zig_common.sh
. "$(dirname "$0")/zig_common.sh"

BASE_IMAGE=tensorfold-dsv41:$ZIG_BASE_COMMIT
[[ "$ZIG_TF_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ZIG_TF_COMMIT must be a full 40-character commit in .env" >&2; exit 2; }
[ "$ZIG_IMAGE" = "tensorfold-dsv41-zig:${ZIG_TF_COMMIT:0:8}" ] \
  || { echo "ZIG_IMAGE ($ZIG_IMAGE) must be tensorfold-dsv41-zig:${ZIG_TF_COMMIT:0:8} (the first 8 characters of ZIG_TF_COMMIT)" >&2; exit 2; }
if docker ps -a --format '{{.Names}}' | grep -q '^tfv41-r'; then
  echo "a rank container exists on the head: run ./stop.sh first (the compile needs the memory)" >&2; exit 1
fi
avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' "${MEMINFO_FILE:-/proc/meminfo}" 2>/dev/null || true)
[ "${avail:-0}" -ge 40 ] || { echo "only ${avail:-0} GiB MemAvailable on the head, need 40 for the compile" >&2; exit 1; }

if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  log "base image $BASE_IMAGE is here"
else
  log "building the base image $BASE_IMAGE from $TF_REPO @ $ZIG_BASE_COMMIT (the NGC base pull is ~20 GB the first time)"
  docker build --build-arg TF_REPO="$TF_REPO" --build-arg TF_COMMIT="$ZIG_BASE_COMMIT" -t "$BASE_IMAGE" "$ROOT/docker"
fi

log "building $ZIG_IMAGE from $TF_REPO @ $ZIG_TF_COMMIT (Zig compile, sm_121)"
docker build -f "$ROOT/docker/Dockerfile.zig" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg TF_REPO="$TF_REPO" \
  --build-arg TF_REF="$ZIG_TF_COMMIT" --build-arg JOBS="${JOBS:-12}" -t "$ZIG_IMAGE" "$ROOT/docker"

log "copying $ZIG_IMAGE to the worker ($WORKER_SSH)"
docker save "$ZIG_IMAGE" | ssh -o BatchMode=yes "$WORKER_SSH" docker load

head_id=$(docker image inspect -f '{{.Id}}' "$ZIG_IMAGE")
worker_id=$(wssh "docker image inspect -f '{{.Id}}' '$ZIG_IMAGE'")
[ "$head_id" = "$worker_id" ] || { log "image IDs differ: head $head_id, worker $worker_id"; exit 1; }
ref=$(docker image inspect -f '{{index .Config.Labels "tf.ref"}}' "$ZIG_IMAGE")
[ "$ref" = "$ZIG_TF_COMMIT" ] || { log "label tf.ref is $ref, not $ZIG_TF_COMMIT"; exit 1; }
log "both nodes hold $ZIG_IMAGE ($head_id, tf.ref $ref)"
