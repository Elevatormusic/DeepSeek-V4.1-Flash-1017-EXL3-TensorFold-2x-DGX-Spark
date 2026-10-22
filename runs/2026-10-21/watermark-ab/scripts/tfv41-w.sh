#!/usr/bin/env bash
# Launcher for the watermark A/B test. Copy of the rel111 launcher (production settings: graphs on, SPILL_GIB=64,
# --parallel 4, 1M context). Containers wmab-r0 (head) and wmab-r1 (worker). Chunk log on (TF_DS_CHUNK_LOG=1).
# It does not touch the production unit or the router.
#   IMAGE=tensorfold-dsv41:<tag> GIDMODE=pinned|auto SPILL_TAG=<name> tfv41-w.sh start|stop|status
set -euo pipefail

IMAGE=${IMAGE:?set IMAGE}
TAG=${IMAGE##*:}
GIDMODE=${GIDMODE:-auto}
WORKER=10.100.80.2
MASTER=10.100.80.1
VIEW=/home/<user>/.cache/huggingface/model-views/deepseek-v41-1017-exl3-2.9bpw-afc7aa0a
CACHE=/home/<user>/tfv41-cache
NAME=DeepSeek-v4.1-Flash-1017-TF
ALIAS=DeepSeek-v4.1-Flash-1017-EXL3
API=http://172.17.0.1:8888/v1
LOG=/home/<user>/wmab/logs
SPILL_GIB=${SPILL_GIB:-64}
SPILL_DIR=${SPILL_DIR:-/cache/tf-ds-spill-wmab-${SPILL_TAG:-x}}
WARM=${WARM:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,131,514,1024,2113}
DRAFTS=${DRAFTS:-5}
case "$GIDMODE" in
  pinned) GIDARGS="-e TF_RDMA_GID_INDEX=3 -e NCCL_IB_GID_INDEX=3" ;;
  auto) GIDARGS="" ;;
  *) echo "GIDMODE must be pinned or auto" >&2; exit 2 ;;
esac

log() { echo "[tfv41w $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }
healthy() { curl --interface 127.0.0.1 -fsS -m 10 "$API/models" 2>/dev/null | grep -q "\"$NAME\""; }

run_args() {  # $1 = rank
  local r=$1
  cat <<EOF
-d --name wmab-r$r --cgroup-parent tensorfold.slice ${EXTRA_ARGS:-} --gpus all --network host --ipc host --shm-size 32g --oom-score-adj 1000
--ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_PTRACE
-v $VIEW/model:/model:ro -v $VIEW/engram:/engram:ro -v $CACHE:/cache
-e TF_DS_REPLAY=1 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_PREFILL_CHUNK=${CHUNK:-2048}
-e TF_DS_RANK_CACHE=/cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32
-e TF_DS_WARM_LENGTHS=$WARM -e TF_DS_ENGRAM=/engram
-e TF_DS_SPILL_GIB=$SPILL_GIB -e TF_DS_SPILL_DIR=$SPILL_DIR
-e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv -e TF_DS_CHUNK_LOG=${CHUNK_LOG:-1} -e TF_DS_TIMING=${TIMING:-0} -e TF_DS_ENGRAM_DONTNEED=${DONTNEED:-1} -e TF_DS_MEM_FLOOR_GIB=${MEM_FLOOR:-1.5} -e TF_DS_ENGRAM_THREADS=${ETHREADS:-128}
-e TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1 $GIDARGS
-e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions-$TAG -e HF_HUB_OFFLINE=1
--entrypoint tensorfold $IMAGE serve /model --tp 2 --rank $r --master $MASTER
--host 172.17.0.1 --port 8888 --name $NAME --alias $ALIAS
--context ${CONTEXT:-1048576} --vision --parallel 4 --mtp-drafts $DRAFTS --temperature 1.0 --top-p 1.0 ${THINK_ARGS:---no-thinking} ${SERVE_EXTRA:-} --no-update-check
EOF
}

case "${1:-status}" in
  start)
    mkdir -p "$CACHE" "$LOG"; wssh "mkdir -p $CACHE"
    if docker ps --format '{{.Names}}' | grep -qE '^(dsv41-1017|glm53|sparkrun_|zigdsv41|rel11|tfv41)'; then
      log "another model is running on the head; stop it first"; exit 1
    fi
    log "starting rank 0 (head), image $IMAGE, GID mode $GIDMODE"
    docker rm -f wmab-r0 >/dev/null 2>&1 || true
    # shellcheck disable=SC2046
    docker run $(run_args 0 | tr '\n' ' ') >/dev/null
    for i in $(seq 1 120); do ss -ltn | grep -q ':29551 ' && break; sleep 2; done
    ss -ltn | grep -q ':29551 ' || { log "rank 0 store port 29551 never listened"; docker logs --tail 40 wmab-r0; exit 1; }
    log "starting rank 1 (worker)"
    wssh "docker rm -f wmab-r1 >/dev/null 2>&1 || true"
    wssh "docker run $(run_args 1 | tr '\n' ' ')" >/dev/null
    log "waiting for $API/models"
    for i in $(seq 1 360); do
      healthy && { log "serving $NAME"; exit 0; }
      docker ps --format '{{.Names}}' | grep -q '^wmab-r0$' || { log "wmab-r0 exited"; docker logs --tail 60 wmab-r0; exit 1; }
      wssh "docker ps --format '{{.Names}}'" | grep -q '^wmab-r1$' || { log "wmab-r1 exited"; wssh "docker logs --tail 60 wmab-r1"; exit 1; }
      sleep 10
    done
    log "not ready after 60 min"; exit 1
    ;;
  stop)   # SAVE_AS=<path prefix>: save both rank logs (with timestamps) as <prefix>-r0.log and <prefix>-r1.log before the stop
    if [ -n "${SAVE_AS:-}" ]; then
      docker logs -t wmab-r0 > "${SAVE_AS}-r0.log" 2>&1 || true
      wssh "docker logs -t wmab-r1" > "${SAVE_AS}-r1.log" 2>&1 || true
    fi
    docker stop -t 60 wmab-r0 >/dev/null 2>&1 || true; docker rm wmab-r0 >/dev/null 2>&1 || true
    wssh "docker stop -t 60 wmab-r1 >/dev/null 2>&1; docker rm wmab-r1 >/dev/null 2>&1" || true
    log "stopped"
    ;;
  status)
    docker ps -a --format '{{.Names}} {{.Status}}' | grep wmab || echo "no head rank"
    wssh "docker ps -a --format '{{.Names}} {{.Status}}'" | grep wmab || echo "no worker rank"
    healthy && echo "API: serving $NAME" || echo "API: not serving"
    ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
