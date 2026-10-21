#!/usr/bin/env python3
"""Cold-prefill measurement: repeated-sentence prompt (A), random natural text (B), random token ids (C),
and the kit_bench random-word prompt (D, the method behind the README number).

Run on the head: python3 -I -u measure.py prompts.json results.jsonl
Every measured request starts with a unique salt line, so the prefix cache cannot hit (cached_tokens is checked).
max_tokens 1, temperature 0, thinking off. TTFT = time to the first streamed piece. Prefill tok/s = prompt_tokens / TTFT.
Safety: a request waits while head MemAvailable is under 2 GiB (idle gate); the run aborts if it falls under 1.25 GiB.
"""
import http.client
import json
import random
import statistics
import sys
import threading
import time
import uuid

BASE_HOST, BASE_PORT = "172.17.0.1", 8888
MODEL = "DeepSeek-v4.1-Flash-1017-EXL3"
PROMPTS, OUT = sys.argv[1], sys.argv[2]
IDLE_GATE_MIB, ABORT_MIB = 1792, 1280   # coordinator rule 2026-10-21: stop if head MemAvailable falls under 1.25 GiB
EPOCH = int(__import__("os").environ.get("EPOCH", "1"))   # engine start number (a clean restart after a crash raises it)

WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine theory market garden window letter music answer bridge island harbor").split()


def memavail_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024


class Mon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.low = 10**9
        self.abort = False
        self.stop = False

    def run(self):
        while not self.stop:
            m = memavail_mib()
            self.low = min(self.low, m)
            if m < ABORT_MIB:
                self.abort = True
            time.sleep(0.25)


MON = Mon()


def conn():
    return http.client.HTTPConnection(BASE_HOST, BASE_PORT, timeout=1800, source_address=("127.0.0.1", 0))


def post_json(path, body):
    c = conn()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return json.loads(data)


def stream(path, body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    c = conn()
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    ttft, usage, total = None, None, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta") or {}
            piece = d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ch.get("text")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    c.close()
    return ttft, total, usage


def wait_idle():
    if __import__("os").path.exists("/tmp/pf32.stop"):
        raise RuntimeError("stop file /tmp/pf32.stop (earlyoom event): stop")
    t = time.time()
    while memavail_mib() < IDLE_GATE_MIB:
        if time.time() - t > 120:
            raise RuntimeError(f"head MemAvailable stayed under {IDLE_GATE_MIB} MiB for 120 s: stop")
        time.sleep(2)


def salt():
    return f"Run {uuid.uuid4().hex}.\n"


def kit_ids(length):
    rng = random.Random(int(time.time() * 1000) ^ (length << 4))
    s = " ".join(f"{rng.randrange(10**9)}" for _ in range(8))
    words = [rng.choice(WORDS) for _ in range(int(length * 0.9) + 64)]
    text = f"Session {s}. Read the following notes carefully.\n" + " ".join(words)
    ids = post_json("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": True})["tokens"]
    while len(ids) < length:
        text += " " + " ".join(rng.choice(WORDS) for _ in range(length - len(ids) + 64))
        ids = post_json("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": True})["tokens"]
    return ids[:length]


def chat_body(text):
    return {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}


ROWS = []


def ALL():
    return [json.loads(ln) for ln in open(OUT)]


DONE = set()


def one(kind, size, rep, text=None, ids=None, salted=True, label=None):
    step = f"{label or kind}-{size}-{rep}"
    if step in DONE and not step.startswith("warm"):
        print("skip (done before restart):", step, flush=True)
        return None
    wait_idle()
    MON.low = memavail_mib()
    if kind == "D":
        ttft, total, u = stream("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0})
    else:
        t = (salt() + text) if salted else text
        if kind in ("B", "C", "W"):
            t += "\nReply with OK."
        ttft, total, u = stream("/v1/chat/completions", chat_body(t))
    pt = u["prompt_tokens"]
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    row = {"step": step, "epoch": EPOCH, "kind": label or kind, "size": size, "rep": rep, "prompt_tokens": pt, "cached_tokens": cached,
           "ttft_s": round(ttft, 3), "total_s": round(total, 3), "prefill_tps": round(pt / ttft, 1),
           "net_tps": round((pt - cached) / ttft, 1), "head_min_mib": MON.low, "head_idle_mib": memavail_mib(),
           "unix": time.time()}
    ROWS.append(row)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if MON.abort:
        raise RuntimeError(f"head MemAvailable fell under {ABORT_MIB} MiB: abort")
    time.sleep(5)
    return row


def main():
    P = json.load(open(PROMPTS, encoding="utf-8"))
    try:
        for ln in open(OUT):
            DONE.add(json.loads(ln)["step"])
    except FileNotFoundError:
        pass
    MON.start()
    print("start head MemAvailable MiB:", memavail_mib(), flush=True)
    # 1. the suite order on the fresh engine, as v41_suite.py runs it: 8K then 32K, exact text, no salt (cached_tokens shows reuse)
    one("A", 8000, 0, P["A_exact_8k"], salted=False, label="A-exact(suite order)")
    one("A", 32000, 0, P["A_exact_32k"], salted=False, label="A-exact(suite order)")
    # 0. warm-up (not counted)
    one("W", 8000, 0, P["W8"], label="warm8")
    one("W", 32000, 0, P["W32"], label="warm32")
    # 2. 32K, three rounds, interleaved A B C D
    for r in range(3):
        one("A", 32000, r, P["A_exact_32k"], label="A-salted")
        one("B", 32000, r, P["B32"][r])
        one("C", 32000, r, P["C32"][r])
        one("D", 32768, r, ids=kit_ids(32768), label="D-kitbench")
    # 3. 8K, three rounds, interleaved
    for r in range(3):
        one("A", 8000, r, P["A_exact_8k"], label="A-salted")
        one("B", 8000, r, P["B8"][r])
        one("C", 8000, r, P["C8"][r])
        one("D", 8192, r, ids=kit_ids(8192), label="D-kitbench")
    # summary
    ROWS[:0] = [json.loads(ln) for ln in open(OUT)][:0]
    print("\nSUMMARY (median [min-max])", flush=True)
    for size_key in (32000, 32768, 8000, 8192):
        for kind in ("A-salted", "B", "C", "D-kitbench"):
            rs = [x for x in ALL() if x["kind"] == kind and x["size"] == size_key and x["rep"] is not None]
            if len(rs) < 2:
                continue
            tt = [x["ttft_s"] for x in rs]
            tp = [x["prefill_tps"] for x in rs]
            print(f"{size_key:>6} {kind:11} n={len(rs)} pt={sorted(set(x['prompt_tokens'] for x in rs))} cached={[x['cached_tokens'] for x in rs]} "
                  f"TTFT {statistics.median(tt):.2f} [{min(tt):.2f}-{max(tt):.2f}] s  "
                  f"tok/s {statistics.median(tp):.0f} [{min(tp):.0f}-{max(tp):.0f}]", flush=True)
    print("head MemAvailable lowest MiB over the run:", min(x["head_min_mib"] for x in ALL()), flush=True)
    MON.stop = True


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("MEASURE_ERROR", type(e).__name__, e, flush=True)
        sys.exit(2)
