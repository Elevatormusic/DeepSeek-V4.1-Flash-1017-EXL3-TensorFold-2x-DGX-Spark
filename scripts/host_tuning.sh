#!/usr/bin/env bash
# Tunes host memory for the engine on both nodes. Run it once (it asks for your sudo password on each node).
# Why: the engine holds about 102 GiB of the 128 GiB. Under load the kernel swaps, reclaims and compacts memory, and
# long prompts then read 10-30% slower with 1-4 s stalls (runs/2026-10-18/). When memory runs out, the head can also
# stop (no ssh) until you cut its power. The engine's RUNBOOK.md asks for the swap part.
#   scripts/host_tuning.sh          apply on the head and the worker (you can run it again; the result is the same)
#   scripts/host_tuning.sh --undo   remove the changes on both nodes and put back the sysctl values from before
#                                   the first run (kept in /var/lib/tensorfold-host-tuning/sysctl.before)
# It checks both nodes before it changes either one. If a step fails on a node, that node goes back to its state
# before the first run; if the worker fails, the head goes back too, so the two nodes stay alike.
# Changes on each node:
#   1. /etc/systemd/system/tensorfold.slice with MemorySwapMax=0. start.sh puts both rank containers in this slice.
#   2. /etc/sysctl.d/99-tensorfold.conf (kernel defaults in brackets):
#        vm.swappiness = 10               [60]   swap other programs less
#        vm.compaction_proactiveness = 0  [20]   no background compaction while a prompt runs
#        vm.extfrag_threshold = 500       [500]  the kernel compacts memory before it reclaims it
#        vm.min_free_kbytes = 1048576     [set at boot]  keep 1 GiB free for the kernel
#        vm.watermark_scale_factor = 25   [10]   free memory in the background before programs must wait
#      Measured with 2 bench runs each (248 prompt chunks): background compaction on gave 46 slow chunks (over
#      1.3 s), the first version of this script (extfrag_threshold 1000, kernel watermarks) gave 5, these values
#      gave 2 (both at 1.31 s) and 128K prompts at 1,986 and 1,981 tokens/s (runs/2026-10-18/INDEX.md).
#      Scale 25 in place of 50 shows no stall rise and reads higher in MemAvailable (about 0.6 GiB), because the
#      kernel keeps less in reserve (2 complete runs each; runs/2026-10-21/watermark-ab/). It adds no free memory.
#   3. earlyoom (required; the script stops if it is missing) stops the engine, not the node, when memory runs
#      out. It sends SIGTERM to the process named "tensorfold" below 1 GiB available and SIGKILL below 0.5 GiB.
#      It ignores swap, because the engine cannot use swap. The engine stops its own growth at 1.5 GiB available
#      (TF_DS_MEM_FLOOR_GIB in start.sh), and a 1M-token prompt took the head down to 1.7 GiB, so a healthy engine is not
#      stopped. Your old file stays as /etc/default/earlyoom.pre-tensorfold.
#   4. Transparent huge pages off (enabled and defrag set to "never"), now and at each boot
#      (/etc/systemd/system/tensorfold-thp.service). The GPU reads host memory through the SMMU, so when the kernel
#      moves pages to make a huge page (compaction), the GPU waits. A Docker healthcheck that started Python every
#      10 s, and khugepaged, made 2-5 s compactions that matched the slowest prompt chunks. With THP off, 3 bench
#      runs gave 1 slow chunk (1.31 s) instead of 10, and 32K-128K prompts ran within 1% of each other
#      (runs/2026-10-18/INDEX.md). The engine does not use huge pages. --undo puts back the modes from before.
set -euo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

node_script() {
  cat <<'EOF'
set -euo pipefail
eo=/etc/default/earlyoom
conf=/etc/sysctl.d/99-tensorfold.conf
slice=/etc/systemd/system/tensorfold.slice
before=/var/lib/tensorfold-host-tuning/sysctl.before
eostate=/var/lib/tensorfold-host-tuning/earlyoom.before
thp=/sys/kernel/mm/transparent_hugepage
thpunit=/etc/systemd/system/tensorfold-thp.service
thpstate=/var/lib/tensorfold-host-tuning/thp.before
keys="vm.swappiness vm.compaction_proactiveness vm.extfrag_threshold vm.min_free_kbytes vm.watermark_scale_factor"
mode() { local m; m=$(sed -n 's/.*\[\(.*\)\].*/\1/p' "$1"); echo "${m:-$(cat "$1")}"; }  # the [selected] word

undo() {  # puts the node back to its state before the first run
  if [ -f "$thpunit" ]; then systemctl disable -q tensorfold-thp.service || true; fi
  rm -f "$slice" "$conf" "$thpunit"
  systemctl daemon-reload
  if [ -f "$thp/enabled" ]; then
    thp_enabled=madvise; thp_defrag=madvise              # the DGX OS defaults, when nothing was saved
    if [ -f "$thpstate" ]; then read -r thp_enabled thp_defrag < "$thpstate"; fi
    echo "$thp_enabled" > "$thp/enabled"; echo "$thp_defrag" > "$thp/defrag"
  fi
  rm -f "$thpstate"
  if [ -f "$before" ]; then
    sysctl -q -p "$before"
    rm -f "$before"
    note="sysctl values from before the first run are back"
  else
    # No saved values (a run of an older version of this script): kernel defaults, then the other sysctl files.
    sysctl -q -w vm.swappiness=60 vm.compaction_proactiveness=20 vm.extfrag_threshold=500 vm.watermark_scale_factor=10
    sysctl -q --system || true
    note="no saved sysctl values: kernel defaults set, vm.min_free_kbytes goes back at the next boot"
  fi
  if [ -f "$eo.pre-tensorfold" ]; then
    mv "$eo.pre-tensorfold" "$eo"
    was_enabled=enabled; was_active=active
    if [ -f "$eostate" ]; then read -r was_enabled was_active < "$eostate"; fi
    if [ "$was_enabled" = disabled ]; then systemctl disable -q earlyoom || true; fi
    if [ "$was_active" = active ]; then systemctl restart earlyoom || true; else systemctl stop earlyoom || true; fi
  fi
  rm -f "$eostate"
}

case "${1:-}" in
  --undo) undo; echo "$(hostname): undone ($note)"; exit 0 ;;
  --check|"") ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac
# earlyoom is the part that keeps the node reachable when memory runs out: stop before any change if it is missing.
if [ ! -f "$eo" ] || ! command -v earlyoom >/dev/null; then
  echo "$(hostname): earlyoom is not installed. Install it (sudo apt install earlyoom), then run this script again." >&2
  exit 1
fi
for k in $keys; do sysctl -n "$k" >/dev/null; done
if [ "${1:-}" = "--check" ]; then echo "$(hostname): ready"; exit 0; fi

# Keep the live values before the first run, so --undo can put them back. With $conf present they are ours.
if [ ! -f "$conf" ] && [ ! -f "$before" ]; then
  mkdir -p "${before%/*}"
  for k in $keys; do echo "$k = $(sysctl -n "$k")"; done > "$before.new"
  mv "$before.new" "$before"
fi
# Also keep earlyoom's boot and run state before we first take it over, so --undo can put that back too.
if [ ! -f "$eo.pre-tensorfold" ] && [ ! -f "$eostate" ]; then
  mkdir -p "${eostate%/*}"
  echo "$(systemctl is-enabled earlyoom 2>/dev/null || true) $(systemctl is-active earlyoom 2>/dev/null || true)" > "$eostate"
fi
# And the huge page modes, the same way.
if [ -f "$thp/enabled" ] && [ ! -f "$thpunit" ] && [ ! -f "$thpstate" ]; then
  mkdir -p "${thpstate%/*}"
  echo "$(mode "$thp/enabled") $(mode "$thp/defrag")" > "$thpstate"
fi
# From here on, a failure puts the node back to its state before the first run, so it is never left half tuned.
rollback() {
  local rc=$?
  [ "$rc" = 0 ] && return
  set +e; trap - EXIT
  undo >/dev/null 2>&1
  echo "$(hostname): setup failed; the node is back to its state before the first run" >&2
  exit "$rc"
}
trap rollback EXIT
printf '[Unit]\nDescription=TensorFold rank containers (no swap)\n\n[Slice]\nMemorySwapMax=0\n' > "$slice"
systemctl daemon-reload
systemctl start tensorfold.slice
printf '%s\n' "vm.swappiness = 10" "vm.compaction_proactiveness = 0" "vm.extfrag_threshold = 500" \
  "vm.min_free_kbytes = 1048576" "vm.watermark_scale_factor = 25" > "$conf"
sysctl -q -p "$conf"
if [ -f "$thp/enabled" ]; then
  printf '%s\n' "[Unit]" "Description=TensorFold: transparent huge pages off (compaction stalls the GB10 GPU)" \
    "DefaultDependencies=no" "After=sysinit.target" "Before=docker.service" "" "[Service]" "Type=oneshot" \
    "RemainAfterExit=yes" "ExecStart=/bin/sh -c 'echo never > $thp/enabled; echo never > $thp/defrag'" "" \
    "[Install]" "WantedBy=multi-user.target" > "$thpunit"
  systemctl daemon-reload
  systemctl enable -q tensorfold-thp.service
  echo never > "$thp/enabled"; echo never > "$thp/defrag"
fi
# -M: SIGTERM / SIGKILL limits in KiB of MemAvailable. -s 100,100: both signals also when swap is free. -p: earlyoom
# runs first. --prefer matches rank 0 (tensorfold-native) and rank 1 (tf-dsv41-lanes). No quotes: the init script passes these words to earlyoom as they are (systemd removes quotes).
args="-r 3600 -p -M 1048576,524288 -s 100,100 --prefer ^(tensorfold|tf-dsv41) --avoid ^(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager|gdm|Xorg|nvidia)"
[ -f "$eo.pre-tensorfold" ] || cp "$eo" "$eo.pre-tensorfold"
{ grep -v '^EARLYOOM_ARGS=' "$eo.pre-tensorfold" || true; printf 'EARLYOOM_ARGS="%s"\n' "$args"; } > "$eo.new"
mv "$eo.new" "$eo"
systemctl enable -q earlyoom
systemctl restart earlyoom
sleep 1
if ! systemctl is-active -q earlyoom; then
  echo "$(hostname): earlyoom did not start with the new settings (journalctl -u earlyoom shows why)." >&2
  exit 1
fi
trap - EXIT
echo "$(hostname): swappiness $(cat /proc/sys/vm/swappiness), compaction_proactiveness" \
  "$(cat /proc/sys/vm/compaction_proactiveness), min_free_kbytes $(cat /proc/sys/vm/min_free_kbytes)," \
  "tensorfold.slice swap $(systemctl show -p MemorySwapMax --value tensorfold.slice)," \
  "huge pages $( [ -f "$thp/enabled" ] && mode "$thp/enabled" || echo n/a)," \
  "earlyoom $(systemctl is-active earlyoom 2>/dev/null || true)"
EOF
}

case "${1:-}" in ""|--undo) ;; *) echo "usage: scripts/host_tuning.sh [--undo]" >&2; exit 2 ;; esac
# Root runs the script from mktemp files (random names, owner-only), never from a fixed path in /tmp.
tmp=$(mktemp)
rtmp=""
cleanup() { rm -f "$tmp"; [ -z "$rtmp" ] || ssh -n -o BatchMode=yes "$WORKER_SSH" "rm -f '$rtmp'" || true; }
trap cleanup EXIT
node_script > "$tmp"
rtmp=$(ssh -n -o BatchMode=yes "$WORKER_SSH" mktemp)
scp -q "$tmp" "$WORKER_SSH:$rtmp"
if [ -z "${1:-}" ]; then
  log "check both nodes before any change"
  sudo bash "$tmp" --check
  ssh -t "$WORKER_SSH" "sudo bash '$rtmp' --check"
fi
log "head"
sudo bash "$tmp" "${1:-}"
log "worker ($WORKER_SSH)"
if ! ssh -t "$WORKER_SSH" "sudo bash '$rtmp' ${1:-}"; then
  [ -z "${1:-}" ] || exit 1
  log "the worker failed: the head goes back too, so both nodes stay alike"
  sudo bash "$tmp" --undo
  exit 1
fi
log "done. Restart the server (./stop.sh, then ./start.sh): the engine sizes its memory at start."
