#!/usr/bin/env bash
# Read fault facts on both nodes. Review the output before you share it.
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
echo '=== head: host fault facts'
python3 "$ROOT/scripts/host_fault_report.py" || exit 1
echo '=== worker: host fault facts'
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" 'python3 -' < "$ROOT/scripts/host_fault_report.py"; then
  echo 'worker report unavailable: SSH or the remote collector failed' >&2
  exit 1
fi
echo 'Review the complete output for private data before you share it. No fault cause is established.'
