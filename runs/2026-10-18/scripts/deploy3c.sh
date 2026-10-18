#!/usr/bin/env bash
# deploy3c.sh: production to image 3c45bdaeab7d (fork PR #7 head). Prebuild the CUDA extensions with the engine
# stopped, start, smoke, gate (suite + grammar check), one prefill pass. Any failure: back to 07aa02585f64.
set -u
NEW=3c45bdaeab7d OLD=07aa02585f64 W=10.100.80.2
TF=$HOME/deploy/tensorfold-v41-20261017 V=$HOME/deploy/v41-1017-20261017
log() { echo "[deploy $(date -u +%T)] $*"; }
setimg() { sed -i "9s|^IMAGE=\${IMAGE:-tensorfold-dsv41:[0-9a-f]*}.*|IMAGE=\${IMAGE:-tensorfold-dsv41:$1}   # $2|" "$TF/tfv41.sh"; }
rollback() {
  log "ROLLBACK to $OLD: $*"
  setimg $OLD "PRs 1-4 + 6 (O_DIRECT spill)"
  cd "$TF" && ./tfv41.sh stop >/dev/null 2>&1; ./tfv41.sh start 2>&1 | tail -1
  exit 1
}
cp "$TF/tfv41.sh" "$TF/tfv41.sh.bak-pre-$NEW-$(date -u +%Y%m%dT%H%M%SZ)"
cd "$TF" && ./tfv41.sh stop >/dev/null 2>&1
log "prebuild extensions"
bash /tmp/prebuild.sh tensorfold-dsv41:$NEW "$HOME/tfv41-cache" 2>&1 | tail -1 || rollback "prebuild head"
scp -q /tmp/prebuild.sh $W:/tmp/prebuild.sh
ssh -n $W "bash /tmp/prebuild.sh tensorfold-dsv41:$NEW \$HOME/tfv41-cache 2>&1 | tail -1" || rollback "prebuild worker"
setimg $NEW "PRs 1-4, 6 and 7 (O_DIRECT spill, 8-key indexer tiles)"
grep -n "^IMAGE" "$TF/tfv41.sh"
./tfv41.sh start 2>&1 | tail -1
docker inspect tfv41-r0 --format '{{.Config.Image}}' | grep -q $NEW || rollback "rank 0 not on $NEW"
docker logs tfv41-r0 2>&1 | grep -a "allocator ceiling\|building CUDA"
ssh -n $W "docker inspect tfv41-r1 --format '{{.Config.Image}}'; docker logs tfv41-r1 2>&1 | grep -a 'allocator ceiling\|building CUDA'"
cd "$V"
python3 -I - <<'EOF' || rollback "smoke failed"
import http.client, json, sys
def post(body):
    c = http.client.HTTPConnection("172.17.0.1", 8888, timeout=120, source_address=("127.0.0.1", 0))
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "x", "max_tokens": 200, **body}),
              {"Content-Type": "application/json"})
    r = c.getresponse(); d = json.loads(r.read() or b"{}"); return r.status, d
schema = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {"type": "object",
          "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": False}}}
s, d = post({"messages": [{"role": "user", "content": "Title for: fixing nginx 502"}], "response_format": schema,
             "chat_template_kwargs": {"enable_thinking": False}})
ok = s == 200 and isinstance(json.loads(d["choices"][0]["message"]["content"]).get("title"), str)
s2, d2 = post({"messages": [{"role": "user", "content": "Say hi."}], "chat_template_kwargs": {"enable_thinking": False}})
print("smoke", s, s2, flush=True)
sys.exit(0 if ok and s2 == 200 else 1)
EOF
log "SMOKE_OK"
set -a; . "$HOME/dgx-lan-model-router/.env"; set +a
timeout 3600 python3 -I -u v41_suite.py > suite-$NEW.log 2>&1
tail -1 suite-$NEW.log; grep -a "^\[FAIL\]" suite-$NEW.log | head -5
grep -aq "^32/32 passed" suite-$NEW.log || rollback "suite failed"
timeout 1800 python3 -I -u v41_grammar_check.py 2 > grammar-check-$NEW.log 2>&1
tail -1 grammar-check-$NEW.log
grep -aq "^14/14 passed" grammar-check-$NEW.log || rollback "grammar check failed"
log "GATE_OK"
cd "$V/kitbench-fd23" && ./pf.sh deploy3c 2>&1 | grep -v warm
curl -s -m 10 --interface 127.0.0.1 http://172.17.0.1:8888/v1/models > /dev/null && log "DONE serving $NEW"
