#!/usr/bin/env bash
# Deploy a new engine image (ENGINE=python only) with an automatic rollback: switch IMAGE in .env, restart both ranks, then a live
# structured-output request, the 32-test suite and the --parallel grammar check. Any failure puts OLD back.
#   scripts/deploy_gate.sh NEW_TAG OLD_TAG        (tags of tensorfold-dsv41, both already on both nodes)
#   SKIP_BENCH=1 scripts/deploy_gate.sh ...       (skip the concurrency benchmark at the end)
set -u
NEW=$1 OLD=$2
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
# This gate swaps IMAGE, the Python engine image. It does not know ZIG_IMAGE (docs/ZIG.md, "Update the engine").
if [ "${ENGINE:-zig}" != python ]; then
  echo "deploy_gate.sh works with ENGINE=python only (the rollback). Set ENGINE=python in .env to use it." >&2
  exit 2
fi
OUT="$ROOT/runs/$(date -u +%Y-%m-%d)"
mkdir -p "$OUT"
export V41_BASE_URL="$API" V41_MODEL="$SERVED_NAME"
# The suite exits 1 on any failed test. Without a router key the 2 router tests cannot pass, so skip that group then.
SUITE_GROUPS=""
[ -n "${V41_ROUTER_API_KEY:-${DGX_ROUTER_API_KEY:-}}" ] || SUITE_GROUPS="speed tools schema reasoning format long_context"

swap() {  # $1 = tag: set it in .env, restart both ranks
  sed -i "s/^IMAGE=.*/IMAGE=tensorfold-dsv41:$1/" "$ROOT/.env"
  (cd "$ROOT" && ./stop.sh && ./start.sh) > "$OUT/gate-$1-start.log" 2>&1
}
rollback() { log "ROLLBACK to $OLD: $*"; swap "$OLD"; log "ROLLED_BACK"; exit 1; }

swap "$NEW" || rollback "start failed"
log "serving $NEW"
(cd "$ROOT" && ./verify.sh) > "$OUT/verify-$NEW.log" 2>&1 || rollback "verify failed (see $OUT/verify-$NEW.log)"
log "VERIFY_OK"
cd "$OUT" || exit 1
# shellcheck disable=SC2086
python3 -I -u "$ROOT/tests/v41_suite.py" $SUITE_GROUPS > "suite-$NEW.log" 2>&1; echo "SUITE_EXIT=$?" >> "suite-$NEW.log"
log "$(grep -ac '^\[PASS' "suite-$NEW.log") pass, $(grep -ac '^\[FAIL' "suite-$NEW.log") fail"
grep -q "SUITE_EXIT=0" "suite-$NEW.log" || rollback "suite failed"
python3 -I -u "$ROOT/tests/v41_grammar_check.py" 2 > "grammar-check-$NEW.log" 2>&1
grep -q "passed$" "grammar-check-$NEW.log" && tail -1 "grammar-check-$NEW.log" | grep -q "^\([0-9]*\)/\1 passed$" \
  || rollback "grammar check failed"
log "$(tail -1 "grammar-check-$NEW.log")"
if [ "${SKIP_BENCH:-0}" = 1 ]; then log "GATE_DONE (bench skipped)"; exit 0; fi
python3 -I -u "$ROOT/tests/v41_bench.py" > "bench-$NEW.log" 2>&1
log "GATE_DONE ($OUT)"
