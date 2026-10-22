#!/usr/bin/env python3
"""One long completion request (prompt of about N tokens, max_tokens 1): used to keep rank 0 busy in a prefill. Usage: prefill_req.py N"""
import http.client
import json
import sys

sys.path.insert(0, "$HOME/rel111/v41")
import capdrive  # noqa: E402  (binds the source address to 127.0.0.1)

n = int(sys.argv[1])
text = capdrive.prompt_of(n, 4242)
print("prompt ready", flush=True)
open("/tmp/rel111.sent", "w").close()
c = http.client.HTTPConnection("172.17.0.1", 8888, timeout=900, source_address=("127.0.0.1", 0))
c.request("POST", "/v1/completions", json.dumps({"model": "DeepSeek-v4.1-Flash-1017-EXL3", "prompt": text, "max_tokens": 1}),
          {"Content-Type": "application/json"})
r = c.getresponse()
print("status", r.status, r.read()[:200], flush=True)
