#!/usr/bin/env bash
# Get the model files and build the model view. Run it on the head node.
#   scripts/download.sh [phase ...]
# With no argument, it runs all phases in this order. Each phase is safe to run again:
# it resumes a download, or it skips work that is done.
#   download  the EXL3 weights at a pinned revision, with the hf command. ENGINE=zig (the default in .env.example):
#             the 2.8 bpw pack, 47 files, 190 GiB. ENGINE=python (the rollback): the 2.9 bpw pack, 43 shards, 196 GiB.
#   engram    Engram shards 47 and 48 (189 GiB in total) and two small files, with resumable curl
#   verify    sha256 check of the files against runs/2026-10-22/zig-28/weights-2.8bpw.sha256 (ENGINE=zig, 51 files)
#             or runs/2026-10-17/weights.sha256 (ENGINE=python, 53 files). It reads about 380 GiB.
#   view      hard-link view in $MODEL_VIEW (model/ and engram/); the chat template comes next
#   worker    copy the view to the worker with rsync, then check the hashes there
# The files go to $DOWNLOAD_DIR first (default: $MODEL_VIEW.src). Keep it on the same disk as
# $MODEL_VIEW, because the view uses hard links. The repos are public. HF_TOKEN is optional.
set -euo pipefail

PHASES=(download engram verify view worker)

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in -h | --help) usage; exit 0 ;; esac

# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

# --- Pinned inputs. The sizes of the 2.9 bpw pack come from runs/2026-10-17/PROVENANCE.md, section 2. ---
# The pack follows the engine: ENGINE=zig (the default) gets the 2.8 bpw pack, ENGINE=python (the rollback) the 2.9 bpw pack.
case "${ENGINE:-zig}" in
  zig)
    PACK=2.8
    EXL3_REPO=jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw     # MIT licence
    EXL3_REV=859396591c340195526aaf61f94b4ad5323a5e18
    EXL3_SHARDS=42            # *.safetensors in model/: 40 layer files, native and head-mtp
    EXL3_BYTES=203793355048   # model/ in the view: 47 files (46 files of the pack and its LICENSE)
    SUMS="$ROOT/runs/2026-10-22/zig-28/weights-2.8bpw.sha256"
    ;;
  python)
    PACK=2.9
    EXL3_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
    EXL3_REV=afc7aa0aab98ee735c0f54e0a122019af1ac08f1
    EXL3_SHARDS=43
    EXL3_BYTES=210818691539   # model/ in the view, without the two template files
    SUMS="$ROOT/runs/2026-10-17/weights.sha256"
    ;;
  *) echo "ENGINE must be zig or python" >&2; exit 1 ;;
esac
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_FILES=(            # "name size in bytes": the four files of the base repo that the view needs
  "model-00047-of-00048.safetensors 101535150936"
  "model-00048-of-00048.safetensors 101537926640"
  "model.safetensors.index.json 7470294"
  "config.json 3311"
)
SPARE_BYTES=$((10 * 1024 * 1024 * 1024))   # keep 10 GiB free after the work

ENGRAM_BYTES=0
for e in "${ENGRAM_FILES[@]}"; do ENGRAM_BYTES=$((ENGRAM_BYTES + ${e##* })); done

: "${MODEL_VIEW:?set MODEL_VIEW in .env}"
MODEL_VIEW=${MODEL_VIEW%/}
DL_DIR=${DOWNLOAD_DIR:-$MODEL_VIEW.src}
DL_DIR=${DL_DIR%/}

# The Xet path used too much memory on the Sparks. Use plain HTTP. You can override each setting.
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_ENDPOINT="${HF_ENDPOINT%/}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}" HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

# --- Helpers ---
fail() { log "ERROR: $*"; exit 1; }
wrun() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }   # stdin stays open (wssh closes it)
gib_up() { echo $((($1 + 1073741823) >> 30)); }
gib_down() { echo $(($1 >> 30)); }
fsize() { if [ -f "$1" ]; then stat -c %s "$1"; else echo 0; fi; }
dir_bytes() { if [ -d "$1" ]; then du -sb "$1" | cut -f1; else echo 0; fi; }

need_cmd() {  # $@ = commands that must exist
  local c
  for c in "$@"; do
    command -v "$c" > /dev/null 2>&1 || fail "the command $c is missing. Install it, then run this script again."
  done
}

find_hf() {  # print the name of the Hugging Face command line tool
  local c
  for c in hf huggingface-cli "$HOME/.local/bin/hf"; do
    if command -v "$c" > /dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

free_bytes() {  # $1 = path. Print the free bytes on the disk that holds it (or its nearest parent).
  local p=$1
  while [ ! -d "$p" ]; do p=$(dirname "$p"); done
  df -B1 --output=avail "$p" | tail -1 | tr -dc 0-9
}

left_to_get() {  # $1 = total bytes of a folder when complete, $2 = the folder. Print the bytes still missing.
  local used
  used=$(dir_bytes "$2")
  if [ "$used" -ge "$1" ]; then echo 0; else echo $(($1 - used)); fi
}

check_disk() {  # $1 = label, $2 = free bytes, $3 = bytes still to write
  local free=$2 need=$3
  if [ "$free" -lt $((need + SPARE_BYTES)) ]; then
    fail "$1: not enough free disk space. Need $(gib_up $((need + SPARE_BYTES))) GiB (10 GiB spare included). Found $(gib_down "$free") GiB."
  fi
  log "$1: $(gib_down "$free") GiB free, $(gib_up "$need") GiB still to write. OK."
}

curl_get() {  # $1 = url, $2 = output file. curl resumes at the size of the output file.
  # The token goes to curl on stdin, so it never shows in the process list. curl sends it to the first host only.
  if [ -n "${HF_TOKEN:-}" ]; then printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; fi |
    curl -sSfL -K - -C - --connect-timeout 30 --speed-limit 1024 --speed-time 120 -o "$2" "$1"
}

fetch() {  # $1 = file name in the base repo, $2 = size in bytes. Download to name.part, then rename.
  local name=$1 want=$2 dst="$DL_DIR/engram/$1" attempt have
  local part="$dst.part" url="$HF_ENDPOINT/$BASE_REPO/resolve/$BASE_REV/$1"
  if [ "$(fsize "$dst")" -eq "$want" ]; then
    log "engram: $name is complete"
    return 0
  fi
  for attempt in $(seq 1 20); do
    have=$(fsize "$part")
    if [ "$have" -eq "$want" ]; then break; fi
    if [ "$have" -gt "$want" ]; then
      log "engram: $name.part is larger than $want bytes. Delete it and start again."
      rm -f "$part"
      have=0
    fi
    log "engram: $name from byte $have (attempt $attempt of 20)"
    if ! curl_get "$url" "$part"; then
      log "engram: $name stopped. Retry in 30 s."
      sleep 30
    fi
  done
  if [ "$(fsize "$part")" -ne "$want" ]; then
    log "engram: $name has $(fsize "$part") bytes, expected $want"
    return 1
  fi
  mv "$part" "$dst"
  log "engram: $name is complete"
}

link_file() {  # $1 = source, $2 = link name. Make a hard link. Skip it when the link is right already.
  local src
  src=$(readlink -f "$1")
  if [ "$src" -ef "$2" ]; then return 0; fi
  ln -f -- "$src" "$2" ||
    fail "view: cannot link $src. Hard links work on one disk only. Put DOWNLOAD_DIR on the disk of MODEL_VIEW."
}

# --- Phases ---
check_head_setup() {  # Run once, before a download. Fail now, not after 200 GiB.
  local need=0 anchor=$MODEL_VIEW
  mkdir -p "$DL_DIR" "$(dirname "$MODEL_VIEW")"
  if [ ! -d "$anchor" ]; then anchor=$(dirname "$MODEL_VIEW"); fi
  if [ "$(stat -c %d "$DL_DIR")" != "$(stat -c %d "$anchor")" ]; then
    fail "$DL_DIR and $anchor are on different disks. Hard links need one disk. Set DOWNLOAD_DIR in .env to a folder on the disk of MODEL_VIEW."
  fi
  if has download; then need=$((need + $(left_to_get "$EXL3_BYTES" "$DL_DIR/model"))); fi
  if has engram; then need=$((need + $(left_to_get "$ENGRAM_BYTES" "$DL_DIR/engram"))); fi
  check_disk "disk on the head" "$(free_bytes "$DL_DIR")" "$need"
}

phase_download() {
  local hf attempt files filter
  hf=$(find_hf) || fail "download: the hf command is missing. Install it with: pipx install 'huggingface_hub[cli]'"
  mkdir -p "$DL_DIR/model"
  log "download: $EXL3_REPO at ${EXL3_REV:0:8}, about $(gib_up "$EXL3_BYTES") GiB, into $DL_DIR/model"
  # The 2.9 bpw pack: the whole repo. The 2.8 bpw pack: only the files of the hash list. The repo also holds a chat
  # template and a README, which the view does not use (the template comes from scripts/patch_template.py).
  files=()
  filter=(--exclude 'assets/*')
  if [ "$PACK" = 2.8 ]; then
    [ -f "$SUMS" ] || fail "download: $SUMS is missing"
    filter=()                 # the hf command takes file names or filters, not both
    mapfile -t files < <(awk '$2 ~ /^model\// { sub(/^model\//, "", $2); print $2 }' "$SUMS")
  fi
  for attempt in 1 2 3 4 5; do
    if "$hf" download "$EXL3_REPO" "${files[@]}" --revision "$EXL3_REV" "${filter[@]}" --local-dir "$DL_DIR/model" > /dev/null; then
      log "download: done"
      return 0
    fi
    log "download: attempt $attempt of 5 failed. Retry in 60 s."
    sleep 60
  done
  fail "download: failed 5 times. Run the phase again to resume."
}

phase_engram() {
  # The hf command refuses files over 50 GB without hf_xet. Use resumable curl on the resolve URL.
  # The URL redirects to the CDN, and the CDN accepts byte ranges.
  local e pids=() p bad=0
  need_cmd curl
  mkdir -p "$DL_DIR/engram"
  log "engram: $BASE_REPO at ${BASE_REV:0:8}, about $(gib_up "$ENGRAM_BYTES") GiB, into $DL_DIR/engram"
  log "engram: to see the progress, run: ls -l '$DL_DIR/engram'"
  for e in "${ENGRAM_FILES[@]}"; do
    fetch "${e%% *}" "${e##* }" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [ "$bad" = 0 ] || fail "engram: a download failed. Run the phase again to resume."
  log "engram: done"
}

phase_verify() {
  local root
  [ -f "$SUMS" ] || fail "verify: $SUMS is missing"
  if [ -d "$DL_DIR/model" ] && [ -d "$DL_DIR/engram" ]; then
    root=$DL_DIR
  elif [ -d "$MODEL_VIEW/model" ] && [ -d "$MODEL_VIEW/engram" ]; then
    root=$MODEL_VIEW
  else
    fail "verify: no files in $DL_DIR or $MODEL_VIEW. Run the download and engram phases first."
  fi
  log "verify: check $(wc -l < "$SUMS") files in $root against $(basename "$SUMS")"
  log "verify: this reads about $(gib_up $((EXL3_BYTES + ENGRAM_BYTES))) GiB with one thread and takes a while. One line shows for each finished file."
  (cd "$root" && sha256sum -c "$SUMS") ||
    fail "verify: a file is missing or damaged (see FAILED above). Delete it, then run the download and engram phases again."
  log "verify: all files match"
}

phase_view() {
  local e name f n
  for e in "${ENGRAM_FILES[@]}"; do
    [ -f "$DL_DIR/engram/${e%% *}" ] || fail "view: ${e%% *} is missing in $DL_DIR/engram. Run the engram phase first."
  done
  compgen -G "$DL_DIR/model/model-*.safetensors" > /dev/null ||
    fail "view: no weights in $DL_DIR/model. Run the download phase first."
  mkdir -p "$MODEL_VIEW/model" "$MODEL_VIEW/engram"
  log "view: link the files into $MODEL_VIEW (hard links, no copy)"
  for f in "$DL_DIR"/model/*.json "$DL_DIR"/model/*.safetensors "$DL_DIR"/model/LICENSE; do
    [ -e "$f" ] || continue
    link_file "$f" "$MODEL_VIEW/model/$(basename "$f")"
  done
  for e in "${ENGRAM_FILES[@]}"; do
    name=${e%% *}
    link_file "$DL_DIR/engram/$name" "$MODEL_VIEW/engram/$name"
  done
  if [ "$PACK" = 2.8 ]; then pattern='*.safetensors'; else pattern='model-*-of-*.safetensors'; fi
  n=$(find "$MODEL_VIEW/model" -maxdepth 1 -name "$pattern" | wc -l)
  [ "$n" -eq "$EXL3_SHARDS" ] || fail "view: $n weight files in $MODEL_VIEW/model, expected $EXL3_SHARDS"
  log "view: done. $(find "$MODEL_VIEW" -type f | wc -l) files, $n weight shards."
  log "view: $DL_DIR and the view share the data. Deleting $DL_DIR frees no disk space."
  if [ ! -f "$MODEL_VIEW/model/chat_template.jinja" ]; then
    log "view: the chat template comes next. Run: $ROOT/scripts/patch_template.py"
  fi
}

phase_worker() {
  local q have free need
  need_cmd rsync ssh
  [ -f "$SUMS" ] || fail "worker: $SUMS is missing"
  [ -d "$MODEL_VIEW/model" ] && [ -d "$MODEL_VIEW/engram" ] || fail "worker: no view on this node. Run the view phase first."
  wssh true 2> /dev/null || fail "worker: ssh to $WORKER_SSH failed. Check WORKER_SSH in .env."
  q=$(printf '%q' "$MODEL_VIEW")
  wssh "mkdir -p $q"
  have=$(wssh "du -sb $q | cut -f1")
  free=$(wssh "df -B1 --output=avail $q | tail -1 | tr -dc 0-9")
  need=$((EXL3_BYTES + ENGRAM_BYTES))
  if [ "$have" -ge "$need" ]; then need=0; else need=$((need - have)); fi
  check_disk "disk on the worker" "$free" "$need"
  # Each file shows under its final name only when it is complete: a partial file waits in .rsync-partial.
  # A second run skips the finished files and resumes the others. It deletes nothing on the worker.
  log "worker: copy $MODEL_VIEW to $WORKER_SSH"
  rsync -a --partial-dir=.rsync-partial --info=progress2,stats1 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "$MODEL_VIEW/" "$WORKER_SSH:$MODEL_VIEW/"
  log "worker: sha256 check on the worker. It reads about $(gib_up $((EXL3_BYTES + ENGRAM_BYTES))) GiB there and takes a while."
  wrun "cd $q && sha256sum -c -" < "$SUMS" ||
    fail "worker: a file is missing or damaged on the worker (see FAILED above). Delete it on the worker, then run the worker phase again."
  log "worker: view OK on $WORKER_SSH"
  if [ -f "$MODEL_VIEW/model/chat_template.jinja" ]; then
    log "worker: chat_template.jinja was copied with the view"
  else
    log "worker: the chat template comes next. Run: $ROOT/scripts/patch_template.py --worker"
  fi
}

# --- Main ---
want=()
has() {  # $1 = phase. True when the user asked for it.
  local p
  for p in "${want[@]}"; do
    if [ "$p" = "$1" ]; then return 0; fi
  done
  return 1
}
valid_phase() {  # $1 = word from the command line
  local p
  for p in "${PHASES[@]}" all; do
    if [ "$p" = "$1" ]; then return 0; fi
  done
  return 1
}

if [ $# -eq 0 ]; then
  want=("${PHASES[@]}")
else
  for a in "$@"; do
    valid_phase "$a" || { echo "unknown phase: $a" >&2; usage >&2; exit 2; }
    if [ "$a" = all ]; then want=("${PHASES[@]}"); else want+=("$a"); fi
  done
fi

if has download || has engram; then check_head_setup; fi
for p in "${PHASES[@]}"; do
  if has "$p"; then "phase_$p"; fi
done
log "done: ${want[*]}"
