#!/usr/bin/env bash
# Both rank containers and the API.
set -uo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"

docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}' | grep tfv41 || echo "head: no rank container"
wssh "docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}'" | grep tfv41 || echo "worker: no rank container"
if healthy; then echo "API: serving $SERVED_NAME at $API"; margin_check; else echo "API: not serving"; fi
