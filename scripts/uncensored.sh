#!/usr/bin/env bash
# Use the BF16 option with the standard start, stop, status and verify commands.
set -euo pipefail

case "${1:-}" in
  -h|--help)
    echo "usage: scripts/uncensored.sh start|stop|status|verify|build"
    echo "Copy .env.uncensored.example to .env.uncensored first. See docs/UNCENSORED.md."
    exit 0 ;;
  start|stop|status|verify|build) action=$1 ;;
  *) echo "usage: scripts/uncensored.sh start|stop|status|verify|build" >&2; exit 2 ;;
esac
[ "$#" = 1 ] || { echo "use one command only" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TF_ENV_FILE="$ROOT/.env.uncensored"
# shellcheck source=common.sh
. "$ROOT/scripts/common.sh"

if [ "$action" = start ] || [ "$action" = build ]; then
  [[ "$TF_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    log "TF_COMMIT must be a full 40-character lowercase commit SHA"; exit 2;
  }
  image_tag=${IMAGE##*:}
  [[ "$IMAGE" = *:* && "$image_tag" =~ ^[0-9a-f]{12,40}$ && "$TF_COMMIT" = "$image_tag"* ]] || {
    log "IMAGE tag must be a 12-to-40-character prefix of TF_COMMIT"; exit 2;
  }
fi

# Existing start helpers pass these values through a remote shell.
for value in "$MODEL_VIEW" "$CACHE_DIR" "$IMAGE"; do
  [[ "$value" =~ ^[a-zA-Z0-9_./:@-]+$ ]] || {
    log "model paths, cache paths and image names must have no spaces or shell characters"; exit 2;
  }
done

if [ "$action" = build ]; then
  exec bash "$ROOT/docker/build.sh"
fi

# Hash the override before start. Use its full SHA256 as the cache identity.
# Stop, status and verify use the receipt; they must work if a weight file is damaged.
UNCENSORED_CACHE_ID=$(python3 -I - "$MODEL_VIEW/override-receipt.json" <<'PY'
import json
import re
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    digest = json.load(stream)["override_sha256"]
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("invalid override SHA256 in the receipt")
print(digest)
PY
)
export UNCENSORED_CACHE_ID
# Read the settings again so that the cache path includes the verified identity.
# shellcheck source=common.sh
. "$ROOT/scripts/common.sh"

[[ "$CACHE_DIR" = */"$UNCENSORED_CACHE_ID" ]] || {
  log "CACHE_DIR must end with the full UNCENSORED_CACHE_ID"; exit 2;
}

gpu_guard() {
  local ids id gpu runtime
  ids=$(docker ps -q) || {
    echo "cannot list running Docker containers" >&2; return 1;
  }
  for id in $ids; do
    gpu=$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{.Count}}{{end}}' "$id") || {
      echo "cannot inspect GPU requests for Docker container $id" >&2; return 1;
    }
    runtime=$(docker inspect --format '{{.HostConfig.Runtime}}' "$id") || {
      echo "cannot inspect runtime for Docker container $id" >&2; return 1;
    }
    if [ -n "$gpu" ] || [ "$runtime" = nvidia ]; then
      echo "a GPU container is running; stop it with its normal stop command first" >&2
      return 1
    fi
  done
}

if [ "$action" = start ]; then
  checked=$(python3 "$ROOT/scripts/prepare_uncensored.py" --check-view "$MODEL_VIEW")
  [ "$checked" = "$UNCENSORED_CACHE_ID" ] || { log "override identity changed"; exit 1; }
  worker_hash=$(wssh "sha256sum '$MODEL_VIEW/model/override-dealign-bf16.safetensors'" | cut -d ' ' -f1)
  [ "$worker_hash" = "$UNCENSORED_CACHE_ID" ] || { log "worker override differs"; exit 1; }
  gpu_guard
  wssh "$(declare -f gpu_guard); gpu_guard"
fi

exec bash "$ROOT/$action.sh"
