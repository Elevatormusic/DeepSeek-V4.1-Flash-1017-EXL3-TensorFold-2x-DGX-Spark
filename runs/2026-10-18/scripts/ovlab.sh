#!/usr/bin/env bash
# ovlab.sh: production off; GPU tests; both ranks on the gather-overlap code (mounted over image 3c45bdaeab7d, its
# own spill dir so the long-prompt check computes fresh); long-prompt greedy check; 3-pass bench and gate
# (benchgate2.sh ovl); production back.
set -u
D=$HOME/deploy/tensorfold-v41-20261017
IMG=tensorfold-dsv41:3c45bdaeab7d
M="-v /tmp/tfsrc2/tensorfold:/opt/tensorfold/src/tensorfold:ro -e TF_DS_SPILL_DIR=/cache/tf-ds-spill-ab"
back() { echo "== production $(date -u +%T)"; cd "$D" && ./tfv41.sh stop >/dev/null 2>&1; ./tfv41.sh start 2>&1 | tail -1; }
cd "$D" && ./tfv41.sh stop >/dev/null 2>&1
echo "== tests $(date -u +%T)"
docker run --rm --gpus all -v /tmp/tfsrc2:/w -v "$HOME/tfv41-cache/pytest-tmp:/ptmp" -e PYTHONPATH=/w:/ptmp/pt \
  -e TRITON_CACHE_DIR=/ptmp/triton -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh $IMG -c \
  "cd /w && python3 -m pytest -q -p no:cacheprovider --basetemp=/ptmp/bt tests/test_deepseek_v41_gather_overlap.py tests/test_deepseek_v41_prompt_tile.py tests/test_deepseek_v41_spill_direct.py > /ptmp/pytest.out 2>&1; s=\$?; tail -6 /ptmp/pytest.out; exit \$s"
if [ $? != 0 ]; then echo "== tests failed: no A/B"; back; exit 1; fi
sleep 20
echo "== engines $(date -u +%T)"
rm -rf "$HOME/tfv41-cache/tf-ds-spill-ab"; ssh -n 10.100.80.2 'rm -rf $HOME/tfv41-cache/tf-ds-spill-ab'
IMAGE=$IMG CHUNK_LOG=1 PROF_MOUNT0="$M" PROF_MOUNT1="$M" timeout 1500 ./tfv41-prof.sh start 2>&1 | tail -1
docker logs tfv41-r0 2>&1 | grep -a "allocator ceiling\|building CUDA"
ssh -n 10.100.80.2 "docker logs tfv41-r1 2>&1 | grep -a 'allocator ceiling\|building CUDA'"
if docker logs tfv41-r0 2>&1 | grep -q "building CUDA"; then echo "== extensions rebuilt: no A/B"; back; exit 1; fi
echo "== long-prompt greedy check $(date -u +%T)"
for s in 9101 9102; do python3 -I /tmp/longgreedy.py $s /tmp/lg-on-$s.json; done
for s in 9101 9102; do cmp -s <(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['text'])" /tmp/lg-off-$s.json) \
  <(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['text'])" /tmp/lg-on-$s.json) && echo "seed $s same" || echo "seed $s DIFFERENT"; done
bash /tmp/benchgate2.sh ovl
back
