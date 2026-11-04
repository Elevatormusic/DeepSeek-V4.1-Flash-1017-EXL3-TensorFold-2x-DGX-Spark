# Shared by the display launchers. Keep their base separate from the ordinary engine.
# shellcheck shell=bash
DISPLAY_BASE_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46
DISPLAY_BASE_IMAGE=tensorfold-dsv41:07aa02585f64

display_base() {
  if ! docker image inspect "$DISPLAY_BASE_IMAGE" >/dev/null 2>&1; then
    log "building the pinned display base $DISPLAY_BASE_IMAGE"
    docker build --build-arg "TF_REPO=$TF_REPO" --build-arg "TF_COMMIT=$DISPLAY_BASE_COMMIT" \
      -t "$DISPLAY_BASE_IMAGE" "$ROOT/docker"
  fi
}
