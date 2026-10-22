#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 on the Zig engine with the 2.8 bpw pack, TP=2 across both Sparks. ./start.sh runs this
# script when ENGINE=zig (the default). Order: preflight, rank 1 on the worker, rank 0 on the head, wait until the API
# answers, then one real completion. Only a real completion proves that both nodes work; /health alone does not.
#   scripts/zig_start.sh
# Settings come from .env (docs/ZIG.md). PREFLIGHT=0 skips the preflight.
# Exit codes: 0 serving; 1 a fault (a rank exited, no reply, a wrong reply); 2 a wrong setting.
# After a start that fails, read docs/ZIG.md ("Exit codes") before you start again.
set -euo pipefail
# shellcheck source=zig_common.sh
. "$(dirname "$0")/zig_common.sh"

case "${PARALLEL}" in
  1 | 2 | 3 | 4) ;;
  *) echo "PARALLEL must be 1, 2, 3 or 4 for ENGINE=zig (the tests ran 4 streams at most)" >&2; exit 2 ;;
esac
case "${THINKING:-0}" in
  0 | 1) ;;
  *) echo "THINKING must be 0 or 1" >&2; exit 2 ;;
esac
case "${REASONING_EFFORT:-}" in
  "" | low | medium | high | xhigh) ;;
  *) echo "REASONING_EFFORT must be low, medium, high or xhigh" >&2; exit 2 ;;
esac
case "$ZIG_RDMA_GID_INDEX" in
  auto | [0-9] | [0-9][0-9]) ;;
  *) echo "ZIG_RDMA_GID_INDEX must be auto or a number" >&2; exit 2 ;;
esac

if [ "${PREFLIGHT:-1}" = 1 ]; then
  "$ROOT/scripts/preflight.sh" || exit 1
fi
mkdir -p "$LOGS"

# The arguments of one docker run, in three parts. The paths in the container are the same on both ranks.
# TF_DS_EXIT_ON_FAULT=1 must stay on: it gives the exit code 75 that scripts/zig_supervise.sh reads.
common_args() {  # $1 = rank, $2 = RoCE v2 GID index of that node
  local key=""
  [ -n "${API_KEY_FILE:-}" ] && [ "$1" = 0 ] && key="-v $API_KEY_FILE:/run/keys:ro"
  cat <<EOF
-d --name tfv41-r$1 --restart ${RESTART_POLICY:-no} $( [ -n "${CGROUP_PARENT-tensorfold.slice}" ] && echo "--cgroup-parent ${CGROUP_PARENT-tensorfold.slice}" ) --gpus all --ipc=host --network host --oom-score-adj 1000
--ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband
-v $MODEL_VIEW/model:/model:ro -v $MODEL_VIEW/engram:/engram:ro -v $ZIG_RANK_CACHE:/rank-cache:ro
-v $ZIG_KIT:/kit:ro -v $ZIG_DATA:/data:ro $key
-e TF_DS_REPLAY_FLOOR=1 -e TF_DS_STREAMS=$PARALLEL -e HF_HUB_OFFLINE=1 -e TF_DS_EXIT_ON_FAULT=1
-e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
-e TF_RDMA_DEVICES=$RDMA_DEVICES -e TF_RDMA_GID_INDEX=$2
EOF
}

rank0_args() {
  cat <<EOF
-e TF_TP_WORLD=2 -e TF_TP_MASTER=$HEAD_IP -e TF_TP_PORT=$ZIG_RANK_PORT
-e TF_DS_RANK_CACHE=/rank-cache -e TF_DS_KIT=/kit -e TENSORFOLD_CUDA_KERNELS=/kit/aot
-e TF_DS_ENGRAM=/engram -e TF_DS_TOKEN_MAP=/data/dsv41_token_map.json
-e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=1 -e TF_DS_L2_PREFETCH=1 -e TF_DS_ENGRAM_AIO=1
$ZIG_IMAGE
tensorfold-native serve /model --backend cuda --host $BIND_HOST --port $PORT
--name $SERVED_NAME $( [ -n "${SERVED_ALIAS:-}" ] && echo "--alias $SERVED_ALIAS" )
$( [ -n "${API_KEY_FILE:-}" ] && echo "--api-key-file /run/keys" )
--context $CONTEXT --parallel $PARALLEL --max-tokens 32768
--temperature $TEMPERATURE --top-p $TOP_P --top-k 20 $( [ "${THINKING:-0}" = 1 ] && echo --thinking || echo --no-thinking )
$( [ -n "${REASONING_EFFORT:-}" ] && echo "--reasoning-effort $REASONING_EFFORT" )
EOF
}

rank1_args() {
  cat <<EOF
$ZIG_IMAGE
tf-dsv41-lanes /model /rank-cache 1 2 $HEAD_IP $ZIG_RANK_PORT /kit
--engram /engram --token-map /data/dsv41_token_map.json
--pool $CONTEXT --streams $PARALLEL --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1
--rdma $RDMA_DEVICES
EOF
}

# The whole docker run line of a rank, on one line.
run_args() {  # $1 = rank, $2 = GID index
  { common_args "$1" "$2"; "rank$1_args"; } | tr '\n' ' '
}

head_gid=$(zig_gid head) || { log "no RoCE v2 GID with an IPv4 address on $ZIG_FIRST_DEVICE of the head (docs/NETWORK.md)"; exit 1; }
worker_gid=$(zig_gid worker) || { log "no RoCE v2 GID with an IPv4 address on $ZIG_FIRST_DEVICE of the worker (docs/NETWORK.md)"; exit 1; }

docker rm -f tfv41-r0 >/dev/null 2>&1 || true
wssh "docker rm -f tfv41-r1 >/dev/null 2>&1 || true"

# Rank 1 first. It retries its connection to rank 0 until rank 0 is up.
log "starting rank 1 on the worker ($WORKER_SSH, GID index $worker_gid, $ZIG_IMAGE)"
wssh "docker run $(run_args 1 "$worker_gid")" >/dev/null
log "starting rank 0 on the head (GID index $head_gid)"
# shellcheck disable=SC2046
docker run $(run_args 0 "$head_gid") >/dev/null

# A rank that exits tells why with its exit code. The meaning of each code is in docs/ZIG.md.
explain_exit() {  # $1 = side, $2 = container exit code
  case "$2" in
    75) log "$1 rank exited with 75: a peer fault. Run ./stop.sh, then ./start.sh (the optional scripts/zig_supervise.sh does this)." ;;
    137) log "$1 rank exited with 137: it was killed (OOM kill, earlyoom or an operator). Check memory (docs/MEMORY.md)." ;;
    *) log "$1 rank exited with $2: read the log above. Do not restart in a loop." ;;
  esac
}
exited() {  # $1 = side. Print the exit code and return 0 when the rank container does not run. Return 1 when it runs.
  local out
  if [ "$1" = head ]; then
    out=$(docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' tfv41-r0 2>/dev/null) || out="false 1"
  else
    # No answer from ssh is no proof that the rank exited.
    out=$(wssh "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' tfv41-r1" 2>/dev/null) || return 1
  fi
  case "$out" in
    true*) return 1 ;;
    *) echo "${out#* }"; return 0 ;;
  esac
}

log "waiting for $API/models (a start takes 26 to 34 s when the rank caches exist)"
ready=0
for _ in $(seq 1 120); do
  if healthy; then ready=1; break; fi
  if code=$(exited head); then
    log "tfv41-r0 exited"; docker logs --tail 60 tfv41-r0; explain_exit head "$code"; exit 1
  fi
  if code=$(exited worker); then
    log "tfv41-r1 exited"; wssh "docker logs --tail 60 tfv41-r1"; explain_exit worker "$code"; exit 1
  fi
  sleep "${ZIG_POLL_S:-5}"
done
[ "$ready" = 1 ] || { log "not ready after 10 min: docker logs tfv41-r0"; exit 1; }

# The first real completion. A 200 from /models is not proof that TP=2 works; a correct reply is.
log "first completion (17 times 23)"
if ! python3 -I - "$API" "$SERVED_NAME" "$(auth)" <<'PY'
import json, sys, urllib.request

api, model, auth = sys.argv[1], sys.argv[2], sys.argv[3]
headers = {"Content-Type": "application/json", **({"Authorization": auth.split(": ", 1)[1]} if auth else {})}
body = {"model": model, "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]}
try:
    req = urllib.request.Request(api + "/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        status, reply = r.status, json.load(r)
    choice = reply["choices"][0]
    content, finish = (choice["message"].get("content") or "").strip(), choice.get("finish_reason")
except Exception as exc:  # noqa: BLE001
    print(f"first completion failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
ok = status == 200 and content == "391" and finish == "stop"
print(f"first completion: HTTP {status}, content {content!r}, finish_reason {finish!r}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
then
  log "the first completion failed: the pair is not good. Read docker logs tfv41-r0 and, on the worker, docker logs tfv41-r1"
  exit 1
fi
log "serving $SERVED_NAME at $API"
margin_check
