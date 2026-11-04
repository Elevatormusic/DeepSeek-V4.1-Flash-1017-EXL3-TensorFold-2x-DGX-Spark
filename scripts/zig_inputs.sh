#!/usr/bin/env bash
# Make the run-time inputs of the Zig engine on both nodes. Run it on the head, after scripts/download.sh,
# scripts/patch_template.py and scripts/zig_build_image.sh.
#   scripts/zig_inputs.sh [phase ...]
# With no argument it runs all phases in this order. Each phase is safe to run again.
#   kit         the kernel kit (0.6 GiB) at a fixed commit into $CACHE_DIR/zig-kit on the head, checked with
#               scripts/verify_kit.sh before anything from it runs, then copied to the worker and checked there
#   token-map   Engram's token map ($CACHE_DIR/zig-data/dsv41_token_map.json), made in the Zig image from the tokenizer of
#               the view, on the head; then copied to the worker
#   rank-cache  the weight file of each rank ($CACHE_DIR/zig-rank-cache-28, 96 GiB on each node), made in the Zig image on
#               both nodes at the same time. No GPU, no network. The server must not run.
# REBUILD=1 removes an old weight file of the same rank first (rank-cache phase) and makes the token map again.
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h | --help) usage; exit 0 ;; esac

# shellcheck source=zig_common.sh
. "$(dirname "$0")/zig_common.sh"

PHASES=(kit token-map rank-cache)
# The kernel kit is the kit of the TensorFold 1.0 (Zig) engine for DeepSeek-V4.1-Flash. Its folder kit/ holds all files,
# and MANIFEST lists the sha256 of each file except itself. Three pins make the kit safe to use:
#   1. the download is fixed to one commit of the Hugging Face repo (KIT_REVISION), so the files cannot change;
#   2. the sha256 of MANIFEST is the one of the tests (KIT_MANIFEST_SHA256);
#   3. scripts/verify_kit.sh (a file of this repo) checks every file against MANIFEST, and that no other file is in the
#      folder. Nothing from the kit runs, on the head or on the worker, before checks 2 and 3 pass. The kit has its own
#      tools/verify_kit.sh; this script never runs it.
KIT_REPO=bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10
KIT_REVISION=4996441a82d95b9a7853b01a1255e1dafc926387
KIT_MANIFEST_SHA256=8c6dcb712a7557a8aeb794d68daf85b07befae7bb1bdfa470223e957718f2845
# The token map of the tests. make_token_map.py made it from the tokenizer of the 1017 checkpoint (tokenizers package).
TOKEN_MAP_SHA256=de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28

fail() { log "ERROR: $*"; exit 1; }
find_hf() {
  local c
  for c in hf huggingface-cli "$HOME/.local/bin/hf"; do
    if command -v "$c" > /dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}
need_view() {
  [ -f "$MODEL_VIEW/model/config.json" ] && [ -f "$MODEL_VIEW/model/chat_template.jinja" ] \
    || fail "no view in $MODEL_VIEW: run scripts/download.sh and scripts/patch_template.py first"
}
need_image() {
  docker image inspect "$ZIG_IMAGE" > /dev/null 2>&1 || fail "no image $ZIG_IMAGE on the head: run scripts/zig_build_image.sh"
  [ "$(docker image inspect -f '{{.Id}}' "$ZIG_IMAGE")" = "$(wssh "docker image inspect -f '{{.Id}}' '$ZIG_IMAGE'" 2> /dev/null)" ] \
    || fail "the worker holds another $ZIG_IMAGE or none: run scripts/zig_build_image.sh"
}
no_engine() {
  if docker ps -a --format '{{.Names}}' | grep -q '^tfv41-r' || wssh "docker ps -a --format '{{.Names}}'" | grep -q '^tfv41-r'; then
    fail "a rank container exists: run ./stop.sh first"
  fi
}

# Pins 2 and 3 for the folder $1 on the head. Nothing in the folder runs.
check_kit() {
  local sum
  [ -f "$1/MANIFEST" ] || fail "kit: $1 has no MANIFEST"
  sum=$(sha256sum "$1/MANIFEST" | cut -d' ' -f1)
  [ "$sum" = "$KIT_MANIFEST_SHA256" ] || fail "kit: MANIFEST has sha256 $sum, expected $KIT_MANIFEST_SHA256. Delete $1 and run the phase again."
  bash "$ROOT/scripts/verify_kit.sh" "$1" > /dev/null || {
    bash "$ROOT/scripts/verify_kit.sh" "$1" >&2 || true
    fail "kit: $1 does not match its MANIFEST (see the lines above)"
  }
}

phase_kit() {
  local hf tmp
  hf=$(find_hf) || fail "kit: the hf command is missing. Install it with: pipx install 'huggingface_hub[cli]'"
  mkdir -p "$ZIG_KIT"
  if [ ! -f "$ZIG_KIT/MANIFEST" ]; then
    tmp=$(mktemp -d "$CACHE_DIR/.zig-kit-dl.XXXXXX")
    log "kit: download the folder kit/ of $KIT_REPO at commit ${KIT_REVISION:0:12} (0.6 GiB)"
    "$hf" download "$KIT_REPO" --revision "$KIT_REVISION" --include 'kit/*' --local-dir "$tmp" > /dev/null       || { rm -rf "$tmp"; fail "kit: the download failed"; }
    if ! (check_kit "$tmp/kit"); then rm -rf "$tmp"; exit 1; fi
    cp -a "$tmp/kit/." "$ZIG_KIT/"
    rm -rf "$tmp"
  fi
  check_kit "$ZIG_KIT"
  log "kit: the head holds a kit that matches its MANIFEST"
  wssh "mkdir -p '$ZIG_KIT'"
  rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "$ZIG_KIT/" "$WORKER_SSH:$ZIG_KIT/"
  # The checker goes to the worker on stdin. The worker never runs a file from the kit.
  wrun bash -s -- "$ZIG_KIT" < "$ROOT/scripts/verify_kit.sh" > /dev/null || fail "kit: the kit on the worker does not match its MANIFEST"
  log "kit: the worker holds a kit that matches its MANIFEST"
}

phase_token_map() {
  local sum
  need_view
  need_image
  mkdir -p "$ZIG_DATA"
  if [ -s "$ZIG_DATA/dsv41_token_map.json" ] && [ "${REBUILD:-0}" != 1 ]; then
    log "token-map: $ZIG_DATA/dsv41_token_map.json is here (REBUILD=1 makes it again)"
  else
    log "token-map: make it in $ZIG_IMAGE from the tokenizer of the view"
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODEL_VIEW/model:/model:ro" -v "$ZIG_DATA:/data" --entrypoint python3 "$ZIG_IMAGE" \
      /opt/tensorfold/tools/dsv41-zig/inputs/make_token_map.py /model /data/dsv41_token_map.json
  fi
  sum=$(sha256sum "$ZIG_DATA/dsv41_token_map.json" | cut -d' ' -f1)
  if [ "$sum" = "$TOKEN_MAP_SHA256" ]; then
    log "token-map: the file is the one of the tests (sha256 ${sum:0:12})"
  else
    log "token-map: WARNING: sha256 $sum differs from the file of the tests ($TOKEN_MAP_SHA256). A tokenizers version with other rules can cause this."
  fi
  wssh "mkdir -p '$ZIG_DATA'"
  rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "$ZIG_DATA/dsv41_token_map.json" "$WORKER_SSH:$ZIG_DATA/"
  [ "$(sha256sum "$ZIG_DATA/dsv41_token_map.json" | cut -d' ' -f1)" = "$(wssh "sha256sum '$ZIG_DATA/dsv41_token_map.json'" | cut -d' ' -f1)" ] \
    || fail "token-map: the file on the worker differs"
  log "token-map: the same file is on both nodes"
}

# The command of one rank, as text. No network, 24 GiB of memory, 16 CPUs, and the user of the shell that runs it, so the
# files belong to you. The $(id ...) parts are for the node that runs the command.
rank_cache_cmd() {  # $1 = rank
  echo "docker run --rm --name zig-rankcache --network none --memory 24g --cpus 16 --user \$(id -u):\$(id -g)" \
    "-v $MODEL_VIEW/model:/model:ro -v $ZIG_RANK_CACHE:/out --entrypoint tf-dsv41-rank-cache $ZIG_IMAGE /model /out $1 2"
}

phase_rank_cache() {
  local rc0=0 rc1=0 pid
  need_view
  need_image
  no_engine
  mkdir -p "$ZIG_RANK_CACHE"
  wssh "mkdir -p '$ZIG_RANK_CACHE'"
  if [ "${REBUILD:-0}" = 1 ]; then
    rm -f "$ZIG_RANK_CACHE"/rank0of2-*.bin
    wssh "rm -f '$ZIG_RANK_CACHE'/rank1of2-*.bin"
  fi
  if compgen -G "$ZIG_RANK_CACHE/rank0of2-*.bin" > /dev/null && wssh "compgen -G '$ZIG_RANK_CACHE/rank1of2-*.bin'" > /dev/null; then
    log "rank-cache: both weight files are here (REBUILD=1 makes them again)"
    return 0
  fi
  log "rank-cache: make the weight file of each rank, on both nodes at the same time (96 GiB each)"
  wssh "$(rank_cache_cmd 1)" > "$LOGS/rank-cache-r1.log" 2>&1 &
  pid=$!
  bash -c "$(rank_cache_cmd 0)" > "$LOGS/rank-cache-r0.log" 2>&1 || rc0=$?
  wait "$pid" || rc1=$?
  log "rank-cache: rank 0 exit code $rc0, rank 1 exit code $rc1 (logs: $LOGS/rank-cache-r*.log)"
  [ "$rc0" = 0 ] && [ "$rc1" = 0 ] || fail "rank-cache: a rank failed"
  ls -la "$ZIG_RANK_CACHE"
  wssh "ls -la '$ZIG_RANK_CACHE'"
}

want=()
for a in "$@"; do
  case "$a" in
    kit | token-map | rank-cache) want+=("$a") ;;
    all) want=("${PHASES[@]}") ;;
    *) echo "unknown phase: $a" >&2; usage >&2; exit 2 ;;
  esac
done
[ ${#want[@]} -gt 0 ] || want=("${PHASES[@]}")
mkdir -p "$CACHE_DIR" "$LOGS"
for p in "${PHASES[@]}"; do
  for w in "${want[@]}"; do
    [ "$w" = "$p" ] && "phase_${p//-/_}"
  done
done
log "done: ${want[*]}"
