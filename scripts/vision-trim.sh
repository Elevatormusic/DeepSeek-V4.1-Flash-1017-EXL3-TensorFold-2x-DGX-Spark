#!/usr/bin/env bash
# Listen on VISION_TRIM_PORT (8001) and forward to the engine on PORT from .env.
# systemd runs this. The engine stays on PORT so start.sh's port check is unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/.env" ]; then
  echo "no $ROOT/.env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$ROOT/.env"
set +a
export VISION_TRIM_UPSTREAM="http://127.0.0.1:${PORT:?PORT is unset}"
export PORT="${VISION_TRIM_PORT:-8001}"
export BIND_HOST="${VISION_TRIM_BIND:-127.0.0.1}"
export VISION_MAX_IMAGES="${VISION_MAX_IMAGES:?set VISION_MAX_IMAGES in .env}"
exec python3 "$ROOT/scripts/vision_trim.py"
