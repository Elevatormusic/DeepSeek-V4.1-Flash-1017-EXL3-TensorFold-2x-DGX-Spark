#!/usr/bin/env bash
# Checks before a start, on both nodes. It refuses (exit 1) on the faults that otherwise show as a hang or a crash
# minutes into the start: missing files, different images, a bad RoCE GID, no memory, no disk, a busy port.
#   scripts/preflight.sh          (start.sh runs it first; PREFLIGHT=0 ./start.sh skips it)
# ENGINE=zig (the default) checks the Zig engine: the 2.8 bpw view, the Zig image, the kernel kit, the token map and the
# rank cache. ENGINE=python (the rollback) checks the Python engine and the 2.9 bpw view. The host checks are the same.
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
ENGINE=${ENGINE:-zig}

bad=0
wrun() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }   # (stdin open: wssh closes it)
ok() { echo "[ ok ] $*"; }
no() { echo "[FAIL] $*"; bad=1; }

wssh true 2>/dev/null && ok "ssh to the worker ($WORKER_SSH)" || { no "ssh to the worker ($WORKER_SSH) failed"; exit 1; }
ping -c 2 -W 2 "$WORKER_IP" >/dev/null 2>&1 && ok "the worker answers on the CX7 link ($WORKER_IP)" \
  || no "no answer from $WORKER_IP: check the cable and docs/NETWORK.md"

# What differs between the engines: the image, the number of weight files, the GID setting, the rank cache folder,
# the fix for an image on one node only, and the memory a node needs (below).
case "$ENGINE" in
  zig)
    # shellcheck source=zig_common.sh
    . "$(dirname "$0")/zig_common.sh"
    PF_IMAGE=$ZIG_IMAGE PF_SHARDS=42 PF_SHARDS_LABEL="42 weight files" PF_GID=$ZIG_RDMA_GID_INDEX
    PF_RANK_CACHE=$ZIG_RANK_CACHE PF_BUILD="scripts/zig_build_image.sh" PF_NEED_DISK=110
    ;;
  python)
    PF_IMAGE=$IMAGE PF_SHARDS=43 PF_SHARDS_LABEL="43 weight shards" PF_GID=$GID_INDEX
    PF_RANK_CACHE=$CACHE_DIR/tf-ds-rank PF_BUILD="docker/build.sh" PF_NEED_DISK=110
    ;;
  *) echo "ENGINE must be zig or python" >&2; exit 1 ;;
esac

# MemAvailable that a node needs before the start, in GiB (docs/MEMORY.md). From the 1.1.0 gate: the head had
# 113,838 MiB (111.2 GiB) and the worker 112,687 MiB (110.0 GiB) free before the start. The engine then held 108.3 GiB on
# the head and 106.5 GiB on the worker, and the head idled at 2.9 GiB. The worker value is 1 GiB under its gate value
# because the worker idles 0.7 GiB higher than the head. Rank 0 alone holds the vision tower (0.95 GiB), so the head
# needs 1 GiB less with VISION=0. The values are for the default CONTEXT and PARALLEL.
# The Zig engine (ENGINE=zig) has no vision setting. The tests of its pair began with 110 GiB free on both nodes.
HEAD_NEED_GIB=111
WORKER_NEED_GIB=109
[ "${VISION:-1}" = 0 ] && HEAD_NEED_GIB=$((HEAD_NEED_GIB - 1))
if [ "$ENGINE" = zig ]; then HEAD_NEED_GIB=$ZIG_NEED_GIB WORKER_NEED_GIB=$ZIG_NEED_GIB; fi

# Extra lines for the node script of the Zig engine: the kernel kit, the token map and the weight file of this rank.
# They expand on the head, in the node script below. $1 = rank of the node.
zig_node_checks() {
  cat <<EOZ
[ -f "$ZIG_KIT/MANIFEST" ] && [ -d "$ZIG_KIT/aot" ] || echo "FAIL no kernel kit in $ZIG_KIT: run scripts/zig_inputs.sh kit"
[ -f "$ZIG_DATA/dsv41_token_map.json" ] || echo "FAIL no $ZIG_DATA/dsv41_token_map.json: run scripts/zig_inputs.sh token-map"
compgen -G "$ZIG_RANK_CACHE/rank${1}of2-*.bin" >/dev/null && echo "OK rank cache of rank $1" \\
  || echo "FAIL no rank cache file for rank $1 in $ZIG_RANK_CACHE: run scripts/zig_inputs.sh rank-cache"
EOZ
}

check_node() {  # $1 = label, $2 = command runner ("" for here, wssh for the worker), $3 = GiB of MemAvailable needed
  local who=$1 run=$2 mem_need=$3 out hint="" rank=0
  [ "$who" = worker ] && rank=1
  [ "$who" = head ] && [ "${VISION:-1}" != 0 ] && [ "$ENGINE" = python ] && hint=", set VISION=0 in .env if you send no images"
  out=$($run bash -s <<EOF
set -u
# The tested driver is 580.178.04 (README). Issue #24 reports node hangs on 580.173.02 (a CX7 PCIe completion
# timeout, then an SMMU stall); the cause is not confirmed, so an older driver is a warning, not a failure.
drv=\$(cat /sys/module/nvidia/version 2>/dev/null)
if [ -z "\$drv" ]; then echo "WARN cannot read the NVIDIA driver version"
elif [ "\$(printf '%s\n' 580.178.04 "\$drv" | sort -V | head -1)" = 580.178.04 ]; then
  echo "OK driver \$drv, kernel \$(uname -r)"
else
  echo "WARN driver \$drv is older than the tested 580.178.04 (kernel \$(uname -r)): update it, see issue #24"
fi
for d in "$MODEL_VIEW/model" "$MODEL_VIEW/engram"; do [ -d "\$d" ] || echo "FAIL missing \$d"; done
[ -f "$MODEL_VIEW/model/chat_template.jinja" ] || echo "FAIL missing $MODEL_VIEW/model/chat_template.jinja"
n=\$(ls "$MODEL_VIEW"/model/*.safetensors 2>/dev/null | wc -l); [ "\$n" -ge $PF_SHARDS ] || echo "FAIL \$n of $PF_SHARDS_LABEL"
docker image inspect "$PF_IMAGE" --format '{{.Id}}' 2>/dev/null | sed 's/^/IMAGE /' || echo "FAIL no image $PF_IMAGE"
$( [ "$ENGINE" = zig ] && zig_node_checks "$rank" )
# Opt in with PREFLIGHT_NCCL_BOOTSTRAP=1 in .env (off by default: it starts one container on each node, and no live
# run shows it is safe for every setup). This checks NCCL's socket bootstrap only, not the RDMA data path. It
# allocates no GPU memory. The next line expands here, on the head, before the script runs on the node.
if [ "${PREFLIGHT_NCCL_BOOTSTRAP:-0}" = 1 ]; then
if bootstrap=\$(timeout --kill-after=5s 20s docker run --rm --init --network host -e NCCL_DEBUG=WARN -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    --entrypoint python3 "$PF_IMAGE" -c 'import ctypes; lib = ctypes.CDLL("libnccl.so.2"); uid = ctypes.create_string_buffer(128); raise SystemExit(lib.ncclGetUniqueId(ctypes.byref(uid)))' 2>&1); then
  echo "OK NCCL socket bootstrap ($NCCL_SOCKET_IFNAME)"
else
  echo "FAIL NCCL socket bootstrap ($NCCL_SOCKET_IFNAME): \$(echo "\$bootstrap" | tr '\\n' ' '); check NCCL_SOCKET_IFNAME in .env"
fi
fi
for dev in \$(echo "$RDMA_DEVICES" | tr ',' ' '); do
  # The engine's RDMA gather sends 4096-byte packets. RoCE gets that MTU only when the netdev MTU is ~4200 or more
  # (9000 is usual). At the default 1500, the gather fails with "transport retry counter exceeded" (issue #15).
  nd=\$(ls /sys/class/infiniband/\$dev/device/net 2>/dev/null | head -1)
  mtu=\$(cat /sys/class/net/\$nd/mtu 2>/dev/null || echo 0)
  am=\$(ibv_devinfo -d \$dev -i 1 2>/dev/null | awk '/active_mtu/ {print \$2; exit}')
  if [ -n "\$am" ] && [ "\$am" != 4096 ]; then
    echo "FAIL \$dev RoCE MTU is \$am, not 4096: set MTU 9000 on \${nd:-its netdev} (docs/NETWORK.md)"
  elif [ -z "\$am" ] && [ "\$mtu" -lt 4200 ]; then
    echo "FAIL \$dev netdev \${nd:-?} has MTU \$mtu, need 9000: see docs/NETWORK.md"
  else
    echo "OK \$dev RoCE MTU 4096 (\$nd MTU \$mtu)"
  fi
  if [ "$PF_GID" = "-1" ]; then echo "OK gid auto (\$dev)"; continue; fi
  if [ "$PF_GID" = "auto" ]; then   # the Zig engine reads the index at each start: a RoCE v2 GID with an IPv4 address must exist
    found=""
    for gf in /sys/class/infiniband/\$dev/ports/1/gids/*; do
      gn=\${gf##*/}; gt=\$(cat /sys/class/infiniband/\$dev/ports/1/gid_attrs/types/\$gn 2>/dev/null)
      case "\$gt:\$(cat "\$gf" 2>/dev/null)" in *v2:0000:0000:0000:0000:0000:ffff:*) found=\$gn; break;; esac
    done
    [ -n "\$found" ] && echo "OK \$dev GID \$found is RoCE v2 with an IPv4 address (auto)" \\
      || echo "FAIL \$dev has no RoCE v2 GID with an IPv4 address: see docs/NETWORK.md"
    continue
  fi
  g=\$(cat /sys/class/infiniband/\$dev/ports/1/gids/$PF_GID 2>/dev/null)
  t=\$(cat /sys/class/infiniband/\$dev/ports/1/gid_attrs/types/$PF_GID 2>/dev/null)
  case "\$g" in ""|0000:0000:0000:0000:0000:0000:0000:0000) echo "FAIL \$dev GID $PF_GID is empty: see docs/NETWORK.md";;
    *) [ "\$t" = "RoCE v2" ] && echo "OK \$dev GID $PF_GID \$g (\$t)" || echo "FAIL \$dev GID $PF_GID is \$t, not RoCE v2";; esac
done
avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo); [ "\$avail" -ge $mem_need ] \
  && echo "OK \$avail GiB memory available (need $mem_need)" \
  || echo "FAIL only \$avail GiB memory available, need $mem_need: close browsers, desktop programs and other containers on this node$hint, and run scripts/host_tuning.sh if you did not (docs/MEMORY.md)"
mkdir -p "$CACHE_DIR"; free=\$(df -BG --output=avail "$CACHE_DIR" | tail -1 | tr -dc 0-9)
[ -d "$PF_RANK_CACHE" ] && need=5 || need=$PF_NEED_DISK
[ "\$free" -ge "\$need" ] && echo "OK \$free GiB free for the cache" || echo "FAIL \$free GiB free in $CACHE_DIR (need \$need)"
[ -z "${CGROUP_PARENT-tensorfold.slice}" ] || [ "\$(systemctl show -p MemorySwapMax --value ${CGROUP_PARENT-tensorfold.slice} 2>/dev/null)" = 0 ] \
  && echo "OK no swap for the engine" || echo "FAIL ${CGROUP_PARENT-tensorfold.slice} has swap or is missing: run scripts/host_tuning.sh"
diff=""
for kv in vm.swappiness=10 vm.compaction_proactiveness=0 vm.extfrag_threshold=500 vm.min_free_kbytes=1048576 \
    vm.watermark_scale_factor=25; do
  k=\${kv%%=*}; [ "\$(cat /proc/sys/\$(echo "\$k" | tr . /) 2>/dev/null)" = "\${kv#*=}" ] || diff="\$diff \$k"
done
[ -z "\$diff" ] && echo "OK memory settings of scripts/host_tuning.sh" \
  || echo "FAIL memory settings differ (\${diff# }): run scripts/host_tuning.sh"
grep -q '\[never\]' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null \
  && echo "OK transparent huge pages off" \
  || echo "FAIL transparent huge pages are on (their compaction stalls the GPU): run scripts/host_tuning.sh"
systemctl is-active -q earlyoom && grep -qF -- "--prefer ^(tensorfold|tf-dsv41) " /etc/default/earlyoom 2>/dev/null \
  && echo "OK earlyoom stops the engine before the node runs out of memory" \
  || echo "FAIL earlyoom is not running with the settings of scripts/host_tuning.sh: run it again"
docker ps --format '{{.Names}}' | grep -q '^tfv41-r' && echo "FAIL a tfv41 rank already runs: ./stop.sh first" || true
EOF
)
  while IFS= read -r line; do
    case "$line" in
      FAIL*) no "$who: ${line#FAIL }" ;;
      OK*) ok "$who: ${line#OK }" ;;
      WARN*) echo "[warn] $who: ${line#WARN }" ;;
      IMAGE*) echo "${line#IMAGE }" > "/tmp/tfv41-image-$who" ;;
    esac
  done <<< "$out"
}
check_node head "" "$HEAD_NEED_GIB"
check_node worker wrun "$WORKER_NEED_GIB"
if [ -s /tmp/tfv41-image-head ] && cmp -s /tmp/tfv41-image-head /tmp/tfv41-image-worker; then
  ok "both nodes hold $PF_IMAGE ($(cut -c8-19 /tmp/tfv41-image-head))"
else
  no "the nodes hold different images (or none) for $PF_IMAGE: run $PF_BUILD"
fi
rm -f /tmp/tfv41-image-head /tmp/tfv41-image-worker
ss -ltn | grep -q ":$PORT " && no "port $PORT is in use on the head" || ok "port $PORT is free"
[ "$bad" = 0 ] && echo "preflight: all checks passed" || { echo "preflight: fix the FAIL lines, then start again"; exit 1; }
