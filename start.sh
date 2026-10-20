#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 on TensorFold, TP=2 across both Sparks: rank 0 on this node (the head), wait for its
# rendezvous port, then rank 1 on the worker, then wait until the API answers. First start: ~15 min (rank caches);
# later starts: ~2-4 min.
set -euo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"

if [ "${PREFLIGHT:-1}" = 1 ]; then
  "$ROOT/scripts/preflight.sh" || exit 1
fi
mkdir -p "$CACHE_DIR" "$LOGS"
wssh "mkdir -p '$CACHE_DIR'"
"$ROOT/scripts/build_extensions.sh" || exit 1    # before the weights load (see the script)

run_args() {  # $1 = rank. Every TF_DS_* setting below is explained in docs/WHAT-I-CHANGED.md
  local r=$1 key=""
  [ -n "${API_KEY_FILE:-}" ] && [ "$r" = 0 ] && key="-v $API_KEY_FILE:/run/keys:ro"   # rank 0 serves HTTP
  cat <<EOF
-d --name tfv41-r$r --restart ${RESTART_POLICY:-no} $( [ -n "${CGROUP_PARENT-tensorfold.slice}" ] && echo "--cgroup-parent ${CGROUP_PARENT-tensorfold.slice}" ) --gpus all --network host --ipc host --shm-size 32g --oom-score-adj 1000
--ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_PTRACE
-v $MODEL_VIEW/model:/model:ro -v $MODEL_VIEW/engram:/engram:ro -v $CACHE_DIR:/cache $key
-e TF_DS_REPLAY=1 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_PREFILL_CHUNK=${TF_DS_PREFILL_CHUNK:-2048}
-e TF_DS_RANK_CACHE=/cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32
-e TF_DS_WARM_LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,131,514,1024,2113 -e TF_DS_ENGRAM=/engram
-e TF_DS_SPILL_GIB=${SPILL_GIB:-64} -e TF_DS_SPILL_DIR=/cache/tf-ds-spill
-e TF_DS_ENGRAM_DONTNEED=1 -e TF_DS_MEM_FLOOR_GIB=${MEM_FLOOR_GIB:-1.5}
-e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv
-e TF_RDMA_DEVICES=$RDMA_DEVICES -e TF_RDMA_GID_INDEX=$GID_INDEX
-e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_IB_GID_INDEX=$GID_INDEX -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e HF_HUB_OFFLINE=1
--entrypoint tensorfold $IMAGE serve /model --tp 2 --rank $r --master $HEAD_IP
--host $BIND_HOST --port $PORT --name $SERVED_NAME $( [ -n "${SERVED_ALIAS:-}" ] && echo "--alias $SERVED_ALIAS" )
$( [ -n "$key" ] && echo "--api-key-file /run/keys" )
--context $CONTEXT $( [ "${VISION:-1}" = 1 ] && echo --vision ) --parallel $PARALLEL --mtp-drafts $MTP_DRAFTS
--temperature $TEMPERATURE --top-p $TOP_P --no-thinking --no-update-check
EOF
}

log "starting rank 0 on the head ($IMAGE)"
docker rm -f tfv41-r0 >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run $(run_args 0 | tr '\n' ' ') >/dev/null
for _ in $(seq 1 120); do ss -ltn | grep -q ":$STORE_PORT " && break; sleep 2; done
ss -ltn | grep -q ":$STORE_PORT " || { log "rank 0 never opened port $STORE_PORT"; docker logs --tail 40 tfv41-r0; exit 1; }

log "starting rank 1 on the worker ($WORKER_SSH)"
wssh "docker rm -f tfv41-r1 >/dev/null 2>&1 || true"
wssh "docker run $(run_args 1 | tr '\n' ' ')" >/dev/null

log "waiting for $API/models (the first start writes the rank caches: slow)"
for _ in $(seq 1 360); do
  healthy && { log "serving $SERVED_NAME at $API"; exit 0; }
  docker ps --format '{{.Names}}' | grep -q '^tfv41-r0$' || { log "tfv41-r0 exited"; docker logs --tail 60 tfv41-r0; exit 1; }
  wssh "docker ps --format '{{.Names}}'" | grep -q '^tfv41-r1$' || { log "tfv41-r1 exited"; wssh "docker logs --tail 60 tfv41-r1"; exit 1; }
  sleep 10
done
log "not ready after 60 min"; exit 1
