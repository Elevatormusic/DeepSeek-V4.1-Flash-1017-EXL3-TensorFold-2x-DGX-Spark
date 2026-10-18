#!/usr/bin/env bash
# topkab3.sh: production off; GPU tests and the selection timing; both ranks on the new code (mounted over the
# image's package, files dated before the built extensions); 3-pass bench and gate (benchgate2.sh topk8c); production
# back.
set -u
D=$HOME/deploy/tensorfold-v41-20261017
IMG=tensorfold-dsv41:07aa02585f64
M="-v /tmp/tfsrc2/tensorfold:/opt/tensorfold/src/tensorfold:ro"
back() { echo "== production $(date -u +%T)"; cd "$D" && ./tfv41.sh stop >/dev/null 2>&1; ./tfv41.sh start 2>&1 | tail -1; }
cd "$D" && ./tfv41.sh stop >/dev/null 2>&1
echo "== tests $(date -u +%T)"
docker run --rm --gpus all -v /tmp/tfsrc2:/w -v "$HOME/tfv41-cache/pytest-tmp:/ptmp" -e PYTHONPATH=/w:/ptmp/pt \
  -e TRITON_CACHE_DIR=/ptmp/triton -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh $IMG -c \
  "cd /w && python3 -m pytest -q -p no:cacheprovider --basetemp=/ptmp/bt tests/test_deepseek_v41_prompt_tile.py tests/test_deepseek_v41_spill_direct.py tests/test_deepseek_v41_spill.py > /ptmp/pytest.out 2>&1; s=\$?; tail -4 /ptmp/pytest.out; [ \$s = 0 ] && python3 /w/topkbench.py; exit \$s"
if [ $? != 0 ]; then echo "== tests failed: no A/B"; back; exit 1; fi
sleep 20
echo "== engines $(date -u +%T)"
CHUNK_LOG=1 PROF_MOUNT0="$M" PROF_MOUNT1="$M" timeout 1500 ./tfv41-prof.sh start 2>&1 | tail -1
docker logs tfv41-r0 2>&1 | grep -a "allocator ceiling\|building CUDA"
ssh -n 10.100.80.2 "docker logs tfv41-r1 2>&1 | grep -a 'allocator ceiling\|building CUDA'"
if docker logs tfv41-r0 2>&1 | grep -q "building CUDA"; then echo "== extensions rebuilt: no A/B"; back; exit 1; fi
bash /tmp/benchgate2.sh topk8c
back
