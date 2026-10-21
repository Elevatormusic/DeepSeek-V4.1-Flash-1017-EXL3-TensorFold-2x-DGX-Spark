#!/usr/bin/env bash
# Opt-in display-backed DeepSeek variant with SSD spill. The ordinary start.sh and .env stay unchanged.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./start-display-spill.sh [--build|--check|--build-only|--restore-drm]
  no argument    use the saved image, prepare DRM, probe both nodes, then start
  --build        explicitly build/copy an updated variant, then prepare and start
  --check        report DRM state on both nodes without changing anything
  --build-only   build/copy the variant without DRM changes, probes, or startup
  --restore-drm  restore runtime DRM defaults on both nodes; do not start a server

Stop both ranks with ./stop.sh before building, preparing, or restoring DRM.
The head may prompt for sudo; the worker needs passwordless sudo for preparation.
No display manager is stopped and no boot configuration is changed.
EOF
}

action=${1:-start}
case "$action" in
  --help|-h) usage; exit 0 ;;
  start|--build|--check|--build-only|--restore-drm) ;;
  *) usage >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { usage >&2; exit 2; }
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"
stage="$ROOT/experiments/display-spill"
probe_stage="$ROOT/experiments/display-kv"
helper="$ROOT/scripts/display_drm.sh"
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10)

if [ "$action" = --check ]; then
  log "head DRM state (allocation probes, not parameter files, prove readiness)"
  bash "$helper" check
  log "worker DRM state"
  ssh "${ssh_args[@]}" "$WORKER_SSH" 'bash -s -- check' < "$helper"
  exit 0
fi

# Serialize this launcher; never silently stop an existing model or reload under it.
exec 9>"$ROOT/.display-kv.lock"   # the same lock as start-display-kv.sh: one display cutover at a time
flock -n 9 || { log "another display launcher (display-KV or display-spill) is active"; exit 1; }
head_names=$(docker ps --format '{{.Names}}')
worker_names=$(wssh "docker ps --format '{{.Names}}'")
for name in $head_names $worker_names; do
  case "$name" in
    tfv41-r*) log "REFUSING: $name is running; use ./stop.sh first"; exit 1 ;;
  esac
done

prepare_drm() {
  local mode=$1 remote_rc=0 head_rc=0
  # Preauthorize before changing either node; sudo -v alone need not be allowed remotely.
  wssh 'sudo -n true' || { log "worker needs passwordless sudo for DRM preparation"; return 1; }
  if [ "$(id -u)" != 0 ]; then
    # sudo -v asks for a password even with NOPASSWD and needs a TTY: try the non-interactive check first.
    sudo -n true 2>/dev/null || sudo -v || { log "head sudo authorization failed; no DRM changes made"; return 1; }
  fi
  log "worker: runtime DRM $mode"
  ssh "${ssh_args[@]}" "$WORKER_SSH" "sudo -n bash -s -- $mode" < "$helper" || remote_rc=$?
  if [ "$remote_rc" != 0 ] && [ "$mode" = on ]; then
    return "$remote_rc"
  fi
  log "head: runtime DRM $mode"
  if [ "$(id -u)" = 0 ]; then
    bash "$helper" "$mode" || head_rc=$?
  else
    sudo -n bash "$helper" "$mode" || head_rc=$?
  fi
  [ "$remote_rc" = 0 ] && [ "$head_rc" = 0 ]
}

if [ "$action" = --restore-drm ]; then
  prepare_drm off
  log "DRM defaults restored; ./start.sh starts the ordinary engine"
  exit 0
fi

image_state="$ROOT/.display-spill-image"
if [ "$action" = --build ] || [ "$action" = --build-only ]; then
  base_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
  fingerprint=$({
    printf '%s\n' "$base_id"
    sha256sum "$stage/Dockerfile" "$stage/apply.py" "$stage/compose.py" "$stage/overlay/"*.py "$probe_stage/apply.py" "$probe_stage/probe.py" "$probe_stage/overlay/"*.py
  } | sha256sum)
  fingerprint=${fingerprint%% *}
  variant="tensorfold-dsv41:display-spill-${fingerprint:0:12}"
  existing=$(docker image inspect "$variant" --format '{{index .Config.Labels "org.tensorfold.display-spill.source"}}' 2>/dev/null || true)
  if [ "$existing" != "$fingerprint" ]; then
    log "building $variant from $IMAGE (pinned source is checked inside the build)"
    docker build -f "$stage/Dockerfile" --build-arg "BASE_IMAGE=$IMAGE" \
      --label "org.tensorfold.display-spill.source=$fingerprint" -t "$variant" "$ROOT"
  fi
  variant_id=$(docker image inspect "$variant" --format '{{.Id}}')
  worker_id=$(wssh "docker image inspect '$variant' --format '{{.Id}}' 2>/dev/null || true")
  if [ "$variant_id" != "$worker_id" ]; then
    log "copying $variant to $WORKER_SSH"
    docker save "$variant" | ssh "${ssh_args[@]}" "$WORKER_SSH" docker load
  fi
  worker_id=$(wssh "docker image inspect '$variant' --format '{{.Id}}'")
  [ "$variant_id" = "$worker_id" ] || { log "variant image IDs differ between nodes"; exit 1; }
  printf '%s %s\n' "$variant" "$variant_id" > "$image_state.tmp"
  mv "$image_state.tmp" "$image_state"
  [ "$action" != --build-only ] || { log "image prepared on both nodes and saved: $variant"; exit 0; }
else
  if [ ! -f "$image_state" ]; then
    log "no saved display-spill image; run ./start-display-spill.sh --build (or --build-only) first"
    exit 1
  fi
  read -r variant variant_id extra < "$image_state"
  if [[ ! $variant =~ ^tensorfold-dsv41:display-spill-[a-f0-9]{12}$ || ! $variant_id =~ ^sha256:[a-f0-9]{64}$ || -n ${extra:-} ]]; then
    log "invalid saved display-spill image; run ./start-display-spill.sh --build-only"
    exit 1
  fi
  head_id=$(docker image inspect "$variant" --format '{{.Id}}' 2>/dev/null || true)
  worker_id=$(wssh "docker image inspect '$variant' --format '{{.Id}}' 2>/dev/null || true")
  if [ "$head_id" != "$variant_id" ] || [ "$worker_id" != "$variant_id" ]; then
    log "saved image missing or changed on a node; run ./start-display-spill.sh --build-only to prepare it"
    exit 1
  fi
  log "using saved display-spill image $variant; no build or image transfer requested"
fi

launch_started=0
cleanup() {
  local rc=$1
  trap - ERR INT TERM
  set +e
  if [ "$launch_started" = 1 ]; then
    log "display-spill startup failed; stopping any ranks belonging to $variant"
    if [ "$(docker inspect tfv41-r0 --format '{{.Image}}' 2>/dev/null || true)" = "$variant_id" ]; then
      docker stop -t 60 tfv41-r0 >/dev/null 2>&1 || true
    fi
    wssh "if [ \"\$(docker inspect tfv41-r1 --format '{{.Image}}' 2>/dev/null || true)\" = '$variant_id' ]; then docker stop -t 60 tfv41-r1 >/dev/null 2>&1 || true; fi" || true
  fi
  log "cutover failed; attempting to restore DRM defaults on both nodes"
  if ! prepare_drm off; then
    log "automatic DRM restore was refused or failed; inspect state with --check"
    log "after all GPU/DRM users stop: ./start-display-spill.sh --restore-drm"
  fi
  log "no ordinary-memory fallback was started; ./start.sh starts the ordinary engine"
  exit "$rc"
}
trap 'cleanup $?' ERR
trap 'cleanup 130' INT
trap 'cleanup 143' TERM
prepare_drm on
log "probing the entire 1792 MiB display reservation on the head"
docker run --rm --gpus all --network none --device /dev/dri --ulimit memlock=-1 \
  --cap-add IPC_LOCK --entrypoint python3 "$variant" /opt/display-kv/probe.py --confirm-idle
log "probing the entire 1792 MiB display reservation on the worker"
wssh "docker run --rm --gpus all --network none --device /dev/dri --ulimit memlock=-1 --cap-add IPC_LOCK --entrypoint python3 '$variant' /opt/display-kv/probe.py --confirm-idle"

bundle="$probe_stage/bundle-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python3 "$probe_stage/prepare.py" "$bundle" --image "$variant" --spill
log "starting display-KV; launch snapshot and logs: $bundle"
launch_started=1
bash "$bundle/start.sh"
trap - ERR INT TERM
log "display-spill is serving; ./verify.sh checks inference and ./stop.sh stops both ranks"
