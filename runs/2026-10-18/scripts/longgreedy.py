"""longgreedy.py SEED OUT: one ~30K-token prompt (made from SEED), greedy decode of 256 tokens, thinking off. Saves the
reply and its sha256 to OUT (JSON). The same SEED on two servers that each compute the prompt fresh shows whether
their prompt processing gives the same tokens."""
import hashlib
import http.client
import json
import random
import sys
import time

seed, out = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
words = ["river", "cobalt", "amber", "falcon", "granite", "willow", "copper", "harbor", "meadow", "ember", "glacier",
         "lantern", "orchid", "summit", "thistle", "velvet", "canyon", "saffron", "beacon", "juniper"]
lines = [f"Record {i:04d}: the code for {rng.choice(words)}-{rng.choice(words)} is {rng.randint(1000, 9999)}."
         for i in range(2400)]
ask = sorted(rng.sample(range(2400), 6))
prompt = ("Here is a list of records.\n\n" + "\n".join(lines) +
          f"\n\nGive the code of records {', '.join(f'{i:04d}' for i in ask)}, one per line, then say which of them "
          "is the largest and explain in two sentences how you found it.")
body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.0,
        "top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
c = http.client.HTTPConnection("172.17.0.1", 8888, timeout=900, source_address=("127.0.0.1", 0))
t = time.time()
c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
r = c.getresponse()
d = json.loads(r.read())
text = d["choices"][0]["message"]["content"]
res = {"seed": seed, "status": r.status, "seconds": round(time.time() - t, 1), "usage": d.get("usage"),
       "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
json.dump(res, open(out, "w"), indent=1)
print(seed, r.status, res["seconds"], "s", (d.get("usage") or {}).get("prompt_tokens"), "prompt tokens",
      (d.get("usage") or {}).get("prompt_tokens_details"), res["sha256"][:16])
