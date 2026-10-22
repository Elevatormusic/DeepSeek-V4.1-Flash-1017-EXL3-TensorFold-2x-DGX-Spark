#!/usr/bin/env bash
# Collects the facts that a memory report needs, on both nodes, into one text file. Attach the file to an issue.
#   scripts/oom_report.sh [OUTPUT_FILE]       (run it on the head; default: logs/oom-report-<UTC time>.txt)
# It only reads. It changes no setting and sends nothing. It never prints a key or a token:
#   - It prints the settings of .env from a fixed list of names, not the whole file. API_KEY_FILE shows only "set".
#   - A final filter removes keys in known formats, the rest of each line from a word that names a credential (key,
#     token, secret, password, Authorization, Bearer, cookie), home paths and the host names of the two nodes.
# Read the file before you post it. docs/MEMORY.md explains how to read the numbers.
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

out=${1:-$LOGS/oom-report-$(date -u +%Y%m%dT%H%M%SZ).txt}
# the script has no set -e: each write that the report needs stops it with an error when it fails
# fd 3 keeps the terminal's stderr: the report block below sends its own stderr into the report file
exec 3>&2
fail() { echo "error: $*" >&3; exit 1; }
[ -d "$out" ] && fail "$out is a folder; give a file name"
mkdir -p "$LOGS" "$(dirname "$out")" || fail "cannot make the folder for $out"
head_name=$(hostname 2>/dev/null || true)
worker_name=$(wssh hostname 2>/dev/null || true)

section() { printf '\n=== %s\n' "$1"; }

# The facts of one node. It runs here for the head and over ssh for the worker (bash -s).
# $1 = engine image of .env. Single quotes: nothing in this text expands on the head.
node_script() {
  cat <<'NODE'
image=$1
sub() { printf '\n--- %s\n' "$1"; }
sub "time and kernel"
echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ), kernel $(uname -r)"
sub "NVIDIA driver"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
[ -n "$drv" ] || drv=$(cat /sys/module/nvidia/version 2>/dev/null)
echo "driver ${drv:-unknown}"
sub "free -m"
free -m 2>&1
sub "/proc/meminfo (kB)"
grep -E '^(MemTotal|MemFree|MemAvailable|AnonPages|Cached|Shmem|Buffers|Active\(file\)|Inactive\(file\)|SReclaimable|SwapTotal|SwapFree):' /proc/meminfo
sub "kernel memory settings"
for k in min_free_kbytes watermark_scale_factor watermark_boost_factor swappiness compaction_proactiveness; do
  echo "vm.$k = $(cat /proc/sys/vm/$k 2>/dev/null || echo n/a)"
done
echo "transparent_hugepage = $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
ps=$(getconf PAGESIZE 2>/dev/null || echo 4096)
awk -v ps="$ps" '$1 == "low" {low += $2} $1 == "high" {high += $2}
  END {printf "sum of the zone low watermarks: %.2f GiB\nsum of the zone high watermarks: %.2f GiB\n", low * ps / 1073741824, high * ps / 1073741824}' /proc/zoneinfo 2>/dev/null
sub "swap"
swapon --show 2>&1 | head -5
echo "tensorfold.slice MemorySwapMax = $(systemctl show -p MemorySwapMax --value tensorfold.slice 2>/dev/null || echo n/a)"
sub "earlyoom"
echo "service: $(systemctl is-active earlyoom 2>&1 | head -1)"
grep -a '^EARLYOOM_ARGS' /etc/default/earlyoom 2>/dev/null || echo "no EARLYOOM_ARGS line"
sub "earlyoom journal, last 24 h (status and stop lines, newest last)"
if j=$(journalctl -u earlyoom --since "24 hours ago" --no-pager -o short-iso 2>&1); then
  printf '%s\n' "$j" | grep -a -E 'mem avail|sending|SIG|kill|terminat' | tail -n 100 || true
  printf '%s\n' "$j" | grep -a -q -E 'mem avail|sending|SIG|kill|terminat' || echo "(no earlyoom lines in the last 24 h)"
else
  echo "journalctl failed: $(printf '%s\n' "$j" | head -3)"
fi
sub "kernel log, NVRM lines (last 20)"
nv=$(dmesg 2>&1 | grep -a -E 'NVRM|^dmesg:' | tail -n 20)
case "$nv" in
  "") echo "(no NVRM lines)" ;;
  dmesg:*) echo "dmesg not readable: $nv" ;;
  *) printf '%s\n' "$nv" ;;
esac
sub "engine image on this node"
docker image inspect "$image" --format 'id {{.Id}}, created {{.Created}}' 2>&1 | head -3
docker ps -a --filter name=tfv41 --format '{{.Names}} | {{.Image}} | {{.Status}}' 2>&1 | head -5
NODE
}

# The last 50 lines of an engine log that mention memory, earlyoom or terminate. It reads the log as a stream: a
# long-lived rank can have a log of hundreds of MiB, and this script runs while memory is short.
log_filter() { grep -a -i -E 'memory|earlyoom|terminate|mem avail|NCCL error' | tail -n 50; }

# One rank. $1 = rank (0 or 1). Without the container (./stop.sh removes it), it reads the newest log that ./stop.sh
# saved in logs/.
engine_lines() {
  local rank=$1 src="docker logs tfv41-r$1" f rc lines
  # at most 50 lines, so they stay in memory (no temporary file that can fail). rc = the status of docker logs
  if [ "$rank" = 0 ]; then
    lines=$(docker logs tfv41-r0 2>&1 | log_filter; exit "${PIPESTATUS[0]}"); rc=$?
  else
    lines=$(wssh "docker logs tfv41-r1" 2>&1 | log_filter; exit "${PIPESTATUS[0]}"); rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    f=$(ls -t "$LOGS"/r"$rank"-*.log 2>/dev/null | head -1)
    if [ -n "$f" ]; then lines=$(log_filter < "$f"); src="saved log ${f##*/}"; else lines=""; src="no container and no saved log"; fi
  fi
  echo
  echo "--- rank $rank engine log ($src): last 50 lines with memory, earlyoom or terminate"
  [ -z "$lines" ] || printf '%s\n' "$lines"
}

report=$({
  echo "OOM report of the DeepSeek-V4.1 TensorFold recipe"
  echo "made $(date -u +%Y-%m-%dT%H:%M:%SZ) by scripts/oom_report.sh. Read docs/MEMORY.md to read the numbers."

  section "recipe"
  echo "repo commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "IMAGE=${IMAGE:-}"
  echo "TF_COMMIT=${TF_COMMIT:-}"

  section ".env settings (a fixed list of names)"
  for k in PARALLEL CONTEXT SPILL_GIB PREFILL_CHUNK VISION MEM_FLOOR_GIB MEM_CAP MTP_DRAFTS THINKING \
      DROP_UNOFFERED_TOOL_CALLS CGROUP_PARENT RESTART_POLICY PREFLIGHT; do
    if [ -n "${!k+x}" ]; then echo "$k=${!k}"; else echo "$k=(not set)"; fi
  done
  if [ -n "${API_KEY_FILE:-}" ]; then echo "auth file: set (the file is not read)"; else echo "auth file: not set"; fi

  section "head node"
  node_script | bash -s -- "${IMAGE:-}" 2>&1
  engine_lines 0

  section "worker node"
  if worker_out=$(node_script | ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" bash -s -- "${IMAGE:-}" 2>&1); then
    printf '%s\n' "$worker_out"
  else
    echo "ssh to the worker failed, so the worker facts are missing:"
    printf '%s\n' "$worker_out" | head -3
  fi
  engine_lines 1
} 2>&1)

# Final filter. Fixed strings (node names, WORKER_SSH, $HOME) go through a sed pattern with escaped characters.
esc() { printf '%s' "$1" | sed 's/[][\.*^$/|&(){}+?]/\\&/g'; }
filter=(-E
  -e 's/(hf_|sk-|ghp_|gho_|ghs_|xox[bp]-|github_pat_)[A-Za-z0-9_-]{8,}/[removed]/g'
  # Basic is a common word, so it counts only with a value of base64 text after it
  -e 's/(^|[^a-z])basic[[:space:]]+[A-Za-z0-9+\/=]{8,}.*$/\1[removed]/Ig'
  # from the first word that names a credential to the end of the line. A value can be quoted, escaped, nested
  # in JSON or follow any Authorization scheme, so a rule for each value form always misses one. "token" counts
  # only as a whole word part (token, hf_token, token-budget), not in tokens or tokenizer.
  -e 's/(api[_-]?key|secret|passw(or)?d|authori[sz]ation|bearer|cookie|credential|token([^a-z]|$)).*$/[removed]/Ig'
  # "key" alone or as the last part of a name (--key, --access-key, ssh_key), not in keys or monkey
  -e 's/(^|[^a-z])key([^a-z]|$).*$/\1[removed]/Ig'
  # a camel-case name part (privateKey, accessKeyId, apiToken, clientSecret): case-sensitive, after a lower-case
  # letter or a digit, so KeyError and Tokenizer stay
  -e 's/([a-z0-9])(Key|Token|Secret|Passw|Credential|Cookie|Auth)([^a-z]|$).*$/\1[removed]/g'
  -e 's#/home/[A-Za-z0-9._-]+#/home/<user>#g'
  -e 's#/Users/[A-Za-z0-9._-]+#/Users/<user>#g')
[ -n "${HOME:-}" ] && [ "$HOME" != / ] && filter+=(-e "s/$(esc "$HOME")/~/g")
[ -n "${WORKER_SSH:-}" ] && filter+=(-e "s/$(esc "$WORKER_SSH")/<worker>/g")
[ "${#worker_name}" -ge 3 ] && filter+=(-e "s/$(esc "$worker_name")/worker/g")
[ "${#head_name}" -ge 3 ] && filter+=(-e "s/$(esc "$head_name")/head/g")
if ! printf '%s\n' "$report" | sed "${filter[@]}" > "$out"; then
  rm -f "$out"                     # a part of the report (for example on a full disk) is not a report
  fail "cannot write the report to $out"
fi

echo "report written to $out"
echo "Read it, then attach it to your issue. The filter removes keys and tokens, but check it for anything else that is private."
