#!/usr/bin/env bash
# tests/host_tuning_test.sh [SCRIPT]: runs the node script of host_tuning.sh against stub sysctl/systemctl and temp paths.
# Checks: the first apply saves the live values and the huge page modes; a second apply keeps them; --undo restores them and removes the
# file; --undo with no saved file falls back to kernel defaults plus "sysctl --system"; a missing or dead earlyoom
# stops the apply, a failed apply rolls the node back, --check changes nothing, and --undo puts back the boot and
# run state of earlyoom.
set -euo pipefail
src=${1:-$(dirname "$0")/../scripts/host_tuning.sh}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/etc/systemd/system" "$T/etc/sysctl.d" "$T/etc/default" "$T/var/lib" "$T/thp"
printf 'always [madvise] never\n' > "$T/thp/enabled"
printf 'always defer defer+madvise [madvise] never\n' > "$T/thp/defrag"
printf 'EARLYOOM_ARGS="-r 3600"\n' > "$T/etc/default/earlyoom"
printf '%s\n' "vm.swappiness=33" "vm.compaction_proactiveness=7" "vm.extfrag_threshold=444" \
  "vm.min_free_kbytes=67584" "vm.watermark_scale_factor=12" > "$T/state"
cat > "$T/bin/sysctl" <<STUB
#!/usr/bin/env bash
st="$T/state"; log="$T/log"
set_kv() { k=\${1%%=*}; v=\${1#*=}; k=\${k// /}; v=\${v// /}; grep -v "^\$k=" "\$st" > "\$st.n" || true; echo "\$k=\$v" >> "\$st.n"; mv "\$st.n" "\$st"; }
case "\$*" in
  "-n "*) grep "^\$2=" "\$st" | cut -d= -f2 ;;
  "-q -w "*) shift 2; for kv in "\$@"; do set_kv "\$kv"; done ;;
  "-q -p "*) while IFS= read -r l; do [ -n "\$l" ] && set_kv "\$(echo "\$l" | tr -d ' ')"; done < "\$3" ;;
  "-q --system") echo "sysctl --system" >> "\$log" ;;
  *) echo "unexpected sysctl \$*" >&2; exit 9 ;;
esac
STUB
cat > "$T/bin/systemctl" <<STUB
#!/usr/bin/env bash
case "\$*" in
  "show -p MemorySwapMax --value"*) echo 0 ;;
  "is-enabled earlyoom") if [ -f "$T/eo-enabled" ]; then echo enabled; else echo disabled; exit 1; fi ;;
  "is-active"*) [ -f "$T/inactive" ] && exit 3; if [ -f "$T/eo-active" ]; then echo active; else echo inactive; exit 3; fi ;;
  "enable -q earlyoom") touch "$T/eo-enabled"; echo "systemctl \$*" >> "$T/log" ;;
  "disable -q earlyoom") rm -f "$T/eo-enabled"; echo "systemctl \$*" >> "$T/log" ;;
  "restart earlyoom") [ -f "$T/restartfail" ] && exit 1; touch "$T/eo-active"; echo "systemctl \$*" >> "$T/log" ;;
  "stop earlyoom") rm -f "$T/eo-active"; echo "systemctl \$*" >> "$T/log" ;;
  *) echo "systemctl \$*" >> "$T/log" ;;
esac
STUB
touch "$T/eo-enabled" "$T/eo-active"
printf '#!/bin/sh\n' > "$T/bin/earlyoom"
chmod +x "$T/bin/sysctl" "$T/bin/systemctl" "$T/bin/earlyoom"
awk '/^  cat <<'"'"'EOF'"'"'$/ {on=1; next} on && /^EOF$/ {exit} on' "$src" \
  | sed -e "s#/etc/systemd/system#$T/etc/systemd/system#g" -e "s#/etc/sysctl.d#$T/etc/sysctl.d#g" \
        -e "s#/var/lib/#$T/var/lib/#g" -e "s#/etc/default/earlyoom#$T/etc/default/earlyoom#g" \
        -e "s#/sys/kernel/mm/transparent_hugepage#$T/thp#g" > "$T/node.sh"
run() { PATH="$T/bin:$PATH" bash "$T/node.sh" "$@" >/dev/null; }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; cat "$T/state"; exit 1; fi; }
b="$T/var/lib/tensorfold-host-tuning/sysctl.before"

run
check "first apply saves the live values" "grep -q 'vm.swappiness = 33' '$b' && grep -q 'vm.min_free_kbytes = 67584' '$b'"
check "first apply sets the tuned values" "grep -q '^vm.swappiness=10$' '$T/state' && grep -q '^vm.min_free_kbytes=1048576$' '$T/state' && grep -q '^vm.watermark_scale_factor=25$' '$T/state'"
args_line=$(grep '^EARLYOOM_ARGS=' "$T/etc/default/earlyoom")
case "$args_line" in *"'"*) noq=no ;; *"--prefer ^(tensorfold|tf-dsv41) --avoid ^("*) noq=yes ;; *) noq=no ;; esac
check "first apply writes the earlyoom args without quotes" "[ $noq = yes ]"
tb="$T/var/lib/tensorfold-host-tuning/thp.before"
tu="$T/etc/systemd/system/tensorfold-thp.service"
check "first apply turns huge pages off now and at boot" "grep -qx never '$T/thp/enabled' && grep -qx never '$T/thp/defrag' \
  && grep -q '^ExecStart=.*never' '$tu' && grep -q 'systemctl enable -q tensorfold-thp.service' '$T/log'"
check "first apply saves the huge page modes" "grep -qx 'madvise madvise' '$tb'"
run
check "second apply keeps the saved values" "grep -q 'vm.swappiness = 33' '$b' && grep -qx 'madvise madvise' '$tb'"
run --undo
check "undo restores the huge page modes and removes the unit" "grep -qx madvise '$T/thp/enabled' \
  && grep -qx madvise '$T/thp/defrag' && [ ! -f '$tu' ] && [ ! -f '$tb' ] \
  && grep -q 'systemctl disable -q tensorfold-thp.service' '$T/log'"
check "undo restores all five values" "grep -q '^vm.swappiness=33$' '$T/state' && grep -q '^vm.compaction_proactiveness=7$' '$T/state' && grep -q '^vm.extfrag_threshold=444$' '$T/state' && grep -q '^vm.min_free_kbytes=67584$' '$T/state' && grep -q '^vm.watermark_scale_factor=12$' '$T/state'"
check "undo removes the saved file and the conf" "[ ! -f '$b' ] && [ ! -f '$T/etc/sysctl.d/99-tensorfold.conf' ]"
check "undo restores the earlyoom file" "grep -q '^EARLYOOM_ARGS=\"-r 3600\"$' '$T/etc/default/earlyoom' && [ ! -f '$T/etc/default/earlyoom.pre-tensorfold' ]"
run
rm -f "$b"
run --undo
check "undo without a saved file sets kernel defaults" "grep -q '^vm.swappiness=60$' '$T/state' && grep -q '^vm.watermark_scale_factor=10$' '$T/state'"
check "undo without a saved file reloads the other sysctl files" "grep -q 'sysctl --system' '$T/log'"
check "apply enables earlyoom" "grep -q 'systemctl enable -q earlyoom' '$T/log'"

mv "$T/etc/default/earlyoom" "$T/eo.keep"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply without earlyoom must fail"; exit 1; fi
check "apply without earlyoom stops with a message" "grep -q 'earlyoom is not installed' '$T/err'"
check "apply without earlyoom changes nothing" "[ ! -f '$T/etc/sysctl.d/99-tensorfold.conf' ] && [ ! -f '$b' ]"
mv "$T/eo.keep" "$T/etc/default/earlyoom"
rolled_back="[ ! -f '$T/etc/sysctl.d/99-tensorfold.conf' ] && [ ! -f '$T/etc/systemd/system/tensorfold.slice' ] \
  && [ ! -f '$b' ] && grep -q '^vm.swappiness=60$' '$T/state' && grep -q '^vm.watermark_scale_factor=10$' '$T/state' \
  && grep -q '^EARLYOOM_ARGS=\"-r 3600\"$' '$T/etc/default/earlyoom' && [ ! -f '$T/etc/default/earlyoom.pre-tensorfold' ] \
  && grep -qx madvise '$T/thp/enabled' && [ ! -f '$T/etc/systemd/system/tensorfold-thp.service' ]"
touch "$T/inactive"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply with a dead earlyoom must fail"; exit 1; fi
check "apply fails when earlyoom does not start" "grep -q 'earlyoom did not start' '$T/err'"
check "that failure rolls the node back" "$rolled_back && grep -q 'back to its state before the first run' '$T/err'"
rm -f "$T/inactive"
touch "$T/restartfail"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply with a failed restart must fail"; exit 1; fi
check "a failed earlyoom restart rolls the node back" "$rolled_back"
rm -f "$T/restartfail"
out=$(PATH="$T/bin:$PATH" bash "$T/node.sh" --check)
check "--check passes on a ready node and changes nothing" "echo '$out' | grep -q ready && [ ! -f '$T/etc/sysctl.d/99-tensorfold.conf' ] && [ ! -f '$b' ]"
mv "$T/etc/default/earlyoom" "$T/eo.keep"
if PATH="$T/bin:$PATH" bash "$T/node.sh" --check >/dev/null 2>&1; then echo "FAIL --check without earlyoom must fail"; exit 1; fi
echo "PASS --check fails without earlyoom"
mv "$T/eo.keep" "$T/etc/default/earlyoom"
e="$T/var/lib/tensorfold-host-tuning/earlyoom.before"
rm -f "$T/eo-enabled" "$T/eo-active"
run
check "apply enables and starts a disabled, stopped earlyoom" "[ -f '$T/eo-enabled' ] && [ -f '$T/eo-active' ] && grep -q '^disabled inactive$' '$e'"
run --undo
check "undo disables and stops it again" "[ ! -f '$T/eo-enabled' ] && [ ! -f '$T/eo-active' ] && [ ! -f '$e' ]"
echo "all checks passed"
