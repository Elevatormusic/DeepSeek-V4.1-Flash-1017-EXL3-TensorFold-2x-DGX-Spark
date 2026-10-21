# Shared by start.sh, stop.sh, status.sh and verify.sh: reads .env and defines the helpers. Source it; do not run it.
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/.env" ]; then
  echo "no $ROOT/.env: copy .env.example to .env and edit it" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$ROOT/.env"
set +a

API="http://${BIND_HOST/0.0.0.0/127.0.0.1}:${PORT}/v1"
# shellcheck disable=SC2034  # the scripts that source this file use STORE_PORT and LOGS
STORE_PORT=29551                 # TensorFold's rank-0 rendezvous port (--master-port default)
# shellcheck disable=SC2034
LOGS="$ROOT/logs"

log() { echo "[tfv41 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
auth() { [ -n "${API_KEY_FILE:-}" ] && printf 'Authorization: Bearer %s' "$(grep -v '^#' "$API_KEY_FILE" | head -1 | sed 's/^.*: *//')"; }
healthy() { curl -fsS -m 10 -H "$(auth)" "$API/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; }

# Head memory margin after a start (docs/MEMORY.md). The 1.1.0 gate idled at 2.9 GiB MemAvailable on the head, and
# earlyoom stops the engine at 1.0 GiB. Prints one line, [ ok ] or [warn]. It never fails: a thin margin is a warning.
# MEMINFO_FILE is only for the offline tests.
margin_check() {
  local kb gib hint=""
  kb=$(awk '/^MemAvailable:/ {print $2; exit}' "${MEMINFO_FILE:-/proc/meminfo}" 2>/dev/null)
  gib=$(awk -v k="${kb:-0}" 'BEGIN { printf "%.2f", k / 1048576 }')
  [ "${VISION:-1}" = 0 ] || hint="set VISION=0 in .env if you send no images, "
  if [ -z "$kb" ]; then
    echo "[warn] head: cannot read MemAvailable"
  elif [ "$kb" -lt 2621440 ]; then         # 2.5 GiB
    echo "[warn] head: MemAvailable is $gib GiB, under 2.5 GiB (earlyoom stops the engine at 1.0 GiB)." \
      "Close browsers and desktop programs on the head, stop other containers on it, ${hint}set PREFILL_CHUNK=1024" \
      "or PARALLEL=2 in .env, and run scripts/host_tuning.sh if you did not. See docs/MEMORY.md."
  else
    echo "[ ok ] head: MemAvailable is $gib GiB (earlyoom stops the engine at 1.0 GiB)"
  fi
}
