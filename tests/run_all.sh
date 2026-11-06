#!/usr/bin/env bash
# Run the full test set against the running server. Every output goes to runs/<UTC date>/.
#   tests/run_all.sh              verify.sh, the suite, the grammar check, the conformance check, the differential check
#   BENCH=1 tests/run_all.sh      also the concurrency and prefix-cache benchmark
#   HERMES=1 tests/run_all.sh     also the Hermes agent gate (needs HERMES_DIR and a router; see tests/README.md)
# The settings come from .env (through scripts/common.sh) and from the variables in tests/README.md.
# The script prints one PASS, FAIL or SKIP line per step. A SKIP does not fail the run.
# The exit code is 1 when any step fails, and 0 when none fails.
set -uo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"
case "${ENGINE:-zig}" in
  zig)
    # shellcheck source=scripts/zig_common.sh
    . "$ROOT/scripts/zig_common.sh"
    IMAGE=$ZIG_IMAGE
    ;;
  python) ;;
  *) echo "ENGINE must be zig or python" >&2; exit 1 ;;
esac

DAY=$(date -u +%F)
STAMP=$(date -u +%H%M%SZ)
TAG=${IMAGE##*:}
OUT="$ROOT/runs/$DAY"
P="$OUT/runall-$TAG-$STAMP"                 # the start of every output file name of this run
FIXTURE_DIR=${FIXTURE_DIR:-$ROOT/runs/2026-10-17/encoding-fixtures}
LOCAL_GROUPS="speed tools schema reasoning format long_context"   # every suite group except "router"
mkdir -p "$OUT"
SUMMARY="$P-summary.txt"
: > "$SUMMARY"
FAILS=0

# The tests read these two variables. The tests send no API key to the engine.
export V41_BASE_URL="$API" V41_MODEL="$SERVED_NAME"

say() { echo "$*" | tee -a "$SUMMARY"; }
pass() { say "[PASS] $1: $2"; }
fail() { say "[FAIL] $1: $2"; FAILS=$((FAILS + 1)); }
skip() { say "[SKIP] $1: $2"; }

# True when the last line that matches $2 shows equal counts (for example 30/30). $2 has the passed count in group 1
# and the total in group 2, and it matches the whole line.
counts_match() {
  local line a b
  line=$(grep -aE "$2" "$1" | tail -1)
  [ -n "$line" ] || return 1
  a=$(sed -E "s|$2|\1|" <<<"$line")
  b=$(sed -E "s|$2|\2|" <<<"$line")
  [ "$a" = "$b" ] && [ "$a" != 0 ]
}

# Print one PASS or FAIL line for a step. A step passes only when both checks pass:
# the exit code is 0, and the log has no FAIL line and has a summary line with equal counts.
#   $1 = step name   $2 = exit code   $3 = log file   $4 = start time in seconds   $5 = status of the summary check
report() {
  local why="" n last
  [ "$2" = 0 ] || why="exit code $2"
  n=$(grep -aEc '^(\[FAIL\]|FAIL )' "$3")
  [ "$n" = 0 ] || why="${why:+$why; }$n FAIL line(s)"
  [ "$5" = 0 ] || why="${why:+$why; }no summary line with equal counts"
  last=$(grep -a . "$3" | tail -1 | sed -E 's/ -> .*//' | cut -c1-90)
  if [ -z "$why" ]; then
    pass "$1" "$last ($((SECONDS - $4)) s)"
  else
    fail "$1" "$why (log: ${3#"$ROOT/"})"
  fi
}

# Run a Python check that imports the tensorfold package. Use the host Python when it has the package.
# Otherwise use the engine image. Add docker options (for example --gpus all) in CHECK_DOCKER_ARGS.
#   $1 = script name in tests/   $2... = script arguments as /model, /fixtures, /encoding, /tests paths
tf_python() {
  local script=$1
  shift
  # shellcheck disable=SC2086
  docker run --rm --network none -e HF_HUB_OFFLINE=1 ${CHECK_DOCKER_ARGS:-} --entrypoint python3 \
    -v "$ROOT/tests:/tests:ro" -v "$MODEL_VIEW/model:/model:ro" -v "$FIXTURE_DIR:/fixtures:ro" \
    ${ENCODING_DIR:+-v "$ENCODING_DIR:/encoding:ro"} \
    "$IMAGE" -I -u "/tests/$script" "$@"
}
have_tensorfold() { python3 -I -c 'import tensorfold' >/dev/null 2>&1; }
have_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

say "run_all: server $API, model $SERVED_NAME, image tag $TAG, output runs/$DAY"
[ -z "${API_KEY_FILE:-}" ] || say "WARN: API_KEY_FILE is set. The tests send no API key, so the engine tests get HTTP 401."

if ! healthy; then
  fail "server" "no answer at $API/models for model $SERVED_NAME. Start it with ./start.sh and try again."
  say "1 step failed. Summary: ${SUMMARY#"$ROOT/"}"
  exit 1
fi

# 1. verify.sh: five real requests.
t=$SECONDS
log="$P-verify.log"
(cd "$ROOT" && ./verify.sh) > "$log" 2>&1
rc=$?
counts_match "$log" '^([0-9]+)/([0-9]+) passed.*$'
report "verify" "$rc" "$log" "$t" "$?"

# 2. The suite. The router group needs a router key; without it, run the other groups only.
SUITE_ARGS=""
if [ -z "${V41_ROUTER_API_KEY:-${DGX_ROUTER_API_KEY:-}}" ]; then
  SUITE_ARGS=$LOCAL_GROUPS
  skip "suite/router" "no V41_ROUTER_API_KEY, so the 2 router tests do not run (see tests/README.md)"
fi
t=$SECONDS
log="$P-suite.log"
rm -f "$ROOT/tests/suite-results.json"      # the suite writes this file next to itself; remove an old one
# shellcheck disable=SC2086
python3 -I -u "$ROOT/tests/v41_suite.py" $SUITE_ARGS > "$log" 2>&1
rc=$?
[ ! -f "$ROOT/tests/suite-results.json" ] || mv "$ROOT/tests/suite-results.json" "$P-suite-results.json"
counts_match "$log" '^([0-9]+)/([0-9]+) passed in .*$'
report "suite" "$rc" "$log" "$t" "$?"

# 3. The grammar check: grammar and plain requests together, under --parallel.
t=$SECONDS
log="$P-grammar-check.log"
python3 -I -u "$ROOT/tests/v41_grammar_check.py" "${GRAMMAR_ROUNDS:-2}" > "$log" 2>&1
rc=$?
counts_match "$log" '^([0-9]+)/([0-9]+) passed$'
report "grammar-check" "$rc" "$log" "$t" "$?"

# 4. The conformance check: DeepSeek's golden fixtures. It needs the tensorfold package (host Python or engine image).
if [ "${ENGINE:-zig}" = zig ]; then
  skip "conformance" "this test checks the Python prompt path; Zig prompt conformance needs a native test"
elif [ ! -d "$FIXTURE_DIR" ]; then
  skip "conformance" "no fixture folder $FIXTURE_DIR (set FIXTURE_DIR)"
elif have_tensorfold || have_image; then
  t=$SECONDS
  log="$P-conformance.log"
  if have_tensorfold; then
    python3 -I -u "$ROOT/tests/v41_conformance.py" "$MODEL_VIEW/model" "$FIXTURE_DIR" > "$log" 2>&1
  else
    tf_python v41_conformance.py /model /fixtures > "$log" 2>&1
  fi
  rc=$?
  counts_match "$log" '^([0-9]+)/([0-9]+) cases match$'
  report "conformance" "$rc" "$log" "$t" "$?"
else
  skip "conformance" "the tensorfold package is not on this host and the image $IMAGE is not here"
fi

# 5. The differential check: our prompts against DeepSeek's encoding.py. It needs encoding.py (ENCODING_DIR).
if [ "${ENGINE:-zig}" = zig ]; then
  skip "differential" "this test checks the Python prompt path; Zig prompt conformance needs a native test"
elif [ -z "${ENCODING_DIR:-}" ] || [ ! -f "${ENCODING_DIR}/encoding.py" ]; then
  skip "differential" "ENCODING_DIR is not set or has no encoding.py (see tests/README.md for the download)"
elif have_tensorfold || have_image; then
  t=$SECONDS
  log="$P-differential.log"
  if have_tensorfold; then
    python3 -I -u "$ROOT/tests/v41_differential.py" "$MODEL_VIEW/model" "$ENCODING_DIR" "$ROOT/tests" > "$log" 2>&1
  else
    tf_python v41_differential.py /model /encoding /tests > "$log" 2>&1
  fi
  rc=$?
  counts_match "$log" '^([0-9]+)/([0-9]+) match .*$'
  report "differential" "$rc" "$log" "$t" "$?"
else
  skip "differential" "the tensorfold package is not on this host and the image $IMAGE is not here"
fi

# 6. The benchmark (BENCH=1). It writes bench-results.json to its working folder, so it runs in a new folder.
if [ "${BENCH:-0}" = 1 ]; then
  t=$SECONDS
  log="$P-bench.log"
  work=$(mktemp -d)
  (cd "$work" && python3 -I -u "$ROOT/tests/v41_bench.py") > "$log" 2>&1
  rc=$?
  [ ! -f "$work/bench-results.json" ] || mv "$work/bench-results.json" "$P-bench-results.json"
  rmdir "$work" 2>/dev/null
  grep -aq '^\[bench\] done' "$log"
  report "bench" "$rc" "$log" "$t" "$?"
else
  skip "bench" "set BENCH=1 to run it (about 3 minutes)"
fi

# 7. The Hermes agent gate (HERMES=1). It talks to the router that the Hermes install points at.
if [ "${HERMES:-0}" = 1 ]; then
  HERMES_PYTHON=${HERMES_PYTHON:-${HERMES_DIR:-}/hermes-agent/venv/bin/python}
  if [ -z "${HERMES_DIR:-}" ]; then
    fail "hermes" "HERMES=1 needs HERMES_DIR (the Hermes install folder)"
  elif [ ! -x "$HERMES_PYTHON" ]; then
    fail "hermes" "no Python at $HERMES_PYTHON (set HERMES_PYTHON)"
  else
    t=$SECONDS
    log="$P-hermes.txt"
    "$HERMES_PYTHON" -I -u "$ROOT/tests/hermes_thinking_modes.py" --report "$P-hermes.json" > "$log" 2>&1
    rc=$?
    counts_match "$log" '^([0-9]+)/([0-9]+) passed$'
    report "hermes" "$rc" "$log" "$t" "$?"
  fi
else
  skip "hermes" "set HERMES=1 and HERMES_DIR to run it (needs the Hermes install and its router)"
fi

say ""
if [ "$FAILS" = 0 ]; then
  say "run_all: no step failed. Summary: ${SUMMARY#"$ROOT/"}"
  exit 0
fi
say "run_all: $FAILS step(s) failed. Summary: ${SUMMARY#"$ROOT/"}"
exit 1
