#!/usr/bin/env bash
# Stop and remove both rank containers. Their logs go to logs/ first; the caches stay on disk.
# The exit code of each rank goes to the log line. With ENGINE=zig, a clean stop gives 0 on rank 0, and 0 or 75 on
# rank 1 (docs/ZIG.md, "Exit codes"). Rank 0 stops first, as the Zig engine needs.
set -uo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"

mkdir -p "$LOGS"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker logs tfv41-r0 > "$LOGS/r0-$stamp.log" 2>&1 || true
wssh "docker logs tfv41-r1" > "$LOGS/r1-$stamp.log" 2>&1 || true
docker stop -t 60 tfv41-r0 >/dev/null 2>&1
code0=$(docker inspect -f '{{.State.ExitCode}}' tfv41-r0 2>/dev/null || echo "none")
docker rm tfv41-r0 >/dev/null 2>&1
code1=$(wssh "docker stop -t 60 tfv41-r1 >/dev/null 2>&1; docker inspect -f '{{.State.ExitCode}}' tfv41-r1 2>/dev/null || echo none; docker rm tfv41-r1 >/dev/null 2>&1")
log "stopped (exit code rank 0: $code0, rank 1: ${code1:-none}; logs in $LOGS/*-$stamp.log)"
