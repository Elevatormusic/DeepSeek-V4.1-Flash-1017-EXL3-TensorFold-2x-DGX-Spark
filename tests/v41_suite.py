#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 EXL3 serving test suite (stdlib only).

Groups: speed (decode prose/code, prefill), tools (agentic path), schema
(structured output), reasoning, long context, router path.
Writes suite-results.json next to this script and prints a summary.
"""
import http.client
import json
import os
import random
import re
import sys
import time
import urllib.request
import uuid

# Environment overrides. The defaults are the values of the original 2x DGX Spark setup.
#   V41_BASE_URL        raw engine API (only 127.0.0.1 may connect)
#   V41_MODEL           model id sent in every request to the raw engine API
#   V41_ROUTER_URL      optional router in front of the engine
#   V41_ROUTER_API_KEY  key for the router (DGX_ROUTER_API_KEY is also read)
# With no router key, router/tools_stream fails with a message and router/default does not run.
BASE = os.environ.get("V41_BASE_URL", "http://172.17.0.1:8888/v1")
ROUTER = os.environ.get("V41_ROUTER_URL", "http://127.0.0.1:8443/v1")
MODEL = os.environ.get("V41_MODEL", "DeepSeek-v4.1-Flash-1017-EXL3")
ROUTER_KEY = os.environ.get("V41_ROUTER_API_KEY") or os.environ.get("DGX_ROUTER_API_KEY", "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suite-results.json")
ONLY = set(sys.argv[1:])


class _LoopbackConn(http.client.HTTPConnection):
    """The raw-port guard admits only loopback sources."""

    def __init__(self, *a, **kw):
        kw["source_address"] = ("127.0.0.1", 0)
        super().__init__(*a, **kw)


class _LoopbackHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_LoopbackConn, req)


OPENER = urllib.request.build_opener(_LoopbackHandler)
RESULTS = []


def _req(url, body, key=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)


def chat(body, base=BASE, key=None, timeout=900):
    body = {"model": MODEL if base == BASE else "dgx-model", **body}
    opener = OPENER if base == BASE else urllib.request.build_opener()
    t0 = time.time()
    with opener.open(_req(base + "/chat/completions", body, key), timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def chat_stream(body, base=BASE, key=None, timeout=900):
    """Return (content, reasoning, tool_call_deltas, usage, ttft, total)."""
    body = {"model": MODEL if base == BASE else "dgx-model", **body, "stream": True,
            "stream_options": {"include_usage": True}}
    opener = OPENER if base == BASE else urllib.request.build_opener()
    t0 = time.time()
    ttft = None
    content, reasoning, tools, usage = [], [], [], None
    with opener.open(_req(base + "/chat/completions", body, key), timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                dlt = c.get("delta", {})
                piece = dlt.get("content") or dlt.get("reasoning") or dlt.get("reasoning_content")
                if (piece or dlt.get("tool_calls")) and ttft is None:
                    ttft = time.time() - t0
                if dlt.get("content"):
                    content.append(dlt["content"])
                if dlt.get("reasoning") or dlt.get("reasoning_content"):
                    reasoning.append(dlt.get("reasoning") or dlt.get("reasoning_content"))
                if dlt.get("tool_calls"):
                    tools.extend(dlt["tool_calls"])
    return "".join(content), "".join(reasoning), tools, usage, ttft, time.time() - t0


def record(group, name, ok, **info):
    RESULTS.append({"group": group, "name": name, "pass": bool(ok), **info})
    extra = " ".join(f"{k}={v}" for k, v in info.items() if k != "detail")
    print(f"[{'PASS' if ok else 'FAIL'}] {group}/{name} {extra}", flush=True)
    if not ok and info.get("detail"):
        print(f"       detail: {str(info['detail'])[:400]}", flush=True)


def run(group, name, fn):
    if ONLY and group not in ONLY:
        return
    try:
        fn()
    except Exception as e:  # a crashed test is a failed test
        record(group, name, False, detail=f"{type(e).__name__}: {e}")


def msg(d):
    return d["choices"][0]["message"]


def off():
    return {"chat_template_kwargs": {"thinking": False}}


def think(effort="high"):
    return {"chat_template_kwargs": {"thinking": True, "reasoning_effort": effort}}


# ------------------------------------------------------------------ schema --
def validate(inst, schema, path="$"):
    """Small JSON-Schema subset validator: type, required, properties,
    additionalProperties false, enum, items, minItems, maxItems, minimum, maximum, pattern."""
    errs = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
             "null": type(None)}
    if t:
        allowed = t if isinstance(t, list) else [t]
        if not any(isinstance(inst, types[x]) and not (x in ("integer", "number") and isinstance(inst, bool))
                   for x in allowed):
            return [f"{path}: expected {t}, got {type(inst).__name__}"]
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in enum")
    if isinstance(inst, dict):
        for k in schema.get("required", []):
            if k not in inst:
                errs.append(f"{path}: missing {k}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: extra key {k}" for k in inst if k not in props]
        for k, v in inst.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, v in enumerate(inst):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            errs.append(f"{path}: below minimum")
        if "maximum" in schema and inst > schema["maximum"]:
            errs.append(f"{path}: above maximum")
    if isinstance(inst, str) and "pattern" in schema and not re.fullmatch(schema["pattern"], inst):
        errs.append(f"{path}: {inst!r} does not match {schema['pattern']}")
    return errs


# ------------------------------------------------------------------- speed --
PROSE = ["Write a 400-word short story about a lighthouse keeper who finds a message in a bottle.",
         "Explain how the immune system distinguishes self from non-self, for a curious high-school student."]
CODE = ["Write a complete Python module implementing an LRU cache class with get/put, type hints, docstrings, "
        "and unittest tests.",
        "Write a TypeScript function that parses an ISO-8601 duration string (like P3DT4H5M) into seconds, "
        "with error handling and examples."]


def speed_decode(kind, prompts, mode):
    def fn():
        rates, ttfts = [], []
        for p in prompts:
            extra = off() if mode == "off" else think()
            c, r, _, u, ttft, total = chat_stream({"messages": [{"role": "user", "content": salted(p)}],
                                                   "max_tokens": 700, "temperature": 1.0, "top_p": 1.0, **extra})
            n = u["completion_tokens"]
            rates.append((n - 1) / max(total - ttft, 1e-6))
            ttfts.append(ttft)
        avg = sum(rates) / len(rates)
        record("speed", f"decode_{kind}_{mode}", avg > 5, tok_s=round(avg, 1),
               per_prompt=[round(x, 1) for x in rates], ttft_s=[round(x, 2) for x in ttfts])
    return fn


def salted(text):
    """Put a unique first line before a prompt, so that no prompt shares a prefix with an earlier prompt.
    The prefix cache then cannot give a hit, and a speed row counts only work that the server did."""
    return f"Run {uuid.uuid4().hex}.\n{text}"


def varied_prompt(target_tokens, rng=None):
    """Return unique, varied English text of about target_tokens tokens (stdlib only).
    The text is shuffled sentences of the Python language reference, with no repeated sentence, behind a unique
    first line. A repeated sentence is not a fair prefill test: the 8K prompt then shares a prefix with the
    32K prompt (a prefix-cache hit), and repeated text prefills faster than varied text."""
    import pydoc_data.topics
    rng = rng or random.Random()
    sents = set()
    for text in pydoc_data.topics.topics.values():
        for para in re.split(r"\n\s*\n", text):
            if re.search(r"(^|\n) {4,}\S|>>>|::|[{}=#]", para):  # skip code blocks and markup
                continue
            sents.update(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(para.split()))
                         if 7 <= len(s.split()) <= 55 and s.endswith("."))
    sents = sorted(sents)
    rng.shuffle(sents)
    want = target_tokens * 4.7  # about 4.7 characters per token for this text
    out, chars = [], 0
    for s in sents:
        if chars >= want:
            break
        out.append(s)
        chars += len(s) + 1
    if chars < want:
        raise RuntimeError("not enough varied text for this prompt size")
    return salted(" ".join(out))


def cached_tokens(usage):
    """Return cached_tokens from a usage dict, or None when the server does not report it."""
    details = (usage or {}).get("prompt_tokens_details")
    if not isinstance(details, dict) or details.get("cached_tokens") is None:
        return None
    return int(details["cached_tokens"])


def prefill_row(usage, ttft):
    """Return (ok, info) for a prefill row. The row fails when the server reports a prefix-cache hit.
    prefill_tok_s counts only tokens that the server did not take from the prefix cache."""
    pt = int(usage["prompt_tokens"])
    cached = cached_tokens(usage)
    if not ttft:
        return False, {"prompt_tokens": pt, "cached_tokens": cached, "detail": "no first token was streamed"}
    info = {"prompt_tokens": pt, "cached_tokens": cached, "ttft_s": round(ttft, 2),
            "prefill_tok_s": round((pt - (cached or 0)) / ttft, 0)}
    if cached is None:
        info["note"] = "server does not report cached_tokens: a prefix-cache hit cannot be ruled out"
    elif cached > 0:
        info["detail"] = f"prefix-cache hit of {cached} tokens: this row is not a cold prefill"
    return not cached, info


def speed_prefill(target_tokens):
    def fn():
        text = varied_prompt(target_tokens)
        c, r, _, u, ttft, total = chat_stream({"messages": [{"role": "user", "content": text + "\nReply with OK."}],
                                               "max_tokens": 8, "temperature": 0, **off()})
        ok, info = prefill_row(u, ttft)
        record("speed", f"prefill_{target_tokens // 1000}k", ok, **info)
    return fn


# ------------------------------------------------------------------- tools --
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city", "unit"], "additionalProperties": False}}}
TIME_TOOL = {"type": "function", "function": {"name": "get_time", "description": "Current local time for a timezone.",
             "parameters": {"type": "object", "properties": {"timezone": {"type": "string",
                            "description": "IANA name, e.g. Europe/Paris"}}, "required": ["timezone"]}}}
CREATE_EVENT = {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event.",
                "parameters": {"type": "object", "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO-8601 datetime"},
                    "duration_minutes": {"type": "integer", "minimum": 1},
                    "attendees": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "email": {"type": "string"}},
                        "required": ["name", "email"]}},
                    "reminder": {"type": "object", "properties": {"minutes_before": {"type": "integer"}},
                                 "required": ["minutes_before"]}},
                    "required": ["title", "start", "duration_minutes", "attendees"]}}}


def calls(d):
    return msg(d).get("tool_calls") or []


def args_of(call):
    return json.loads(call["function"]["arguments"])


def tool_single(mode):
    def fn():
        extra = off() if mode == "off" else think()
        d, dt = chat({"messages": [{"role": "user", "content": "What's the weather in Tokyo in celsius?"}],
                      "tools": [WEATHER, TIME_TOOL], "max_tokens": 2048, "temperature": 1.0, **extra})
        cs = calls(d)
        ok = len(cs) >= 1 and cs[0]["function"]["name"] == "get_weather" and not validate(
            args_of(cs[0]), WEATHER["function"]["parameters"]) and "tokyo" in args_of(cs[0])["city"].lower()
        record("tools", f"single_call_{mode}", ok, s=round(dt, 1), finish=d["choices"][0]["finish_reason"],
               detail=json.dumps(cs)[:300] or msg(d).get("content"))
    return fn


def tool_no_call():
    d, dt = chat({"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                  "tools": [WEATHER, TIME_TOOL], "max_tokens": 512, "temperature": 1.0, **off()})
    ok = not calls(d) and "paris" in (msg(d).get("content") or "").lower()
    record("tools", "no_call_when_not_needed", ok, s=round(dt, 1), detail=msg(d))


def tool_required():
    d, dt = chat({"messages": [{"role": "user", "content": "Hello there!"}], "tools": [WEATHER, TIME_TOOL],
                  "tool_choice": "required", "max_tokens": 512, "temperature": 1.0, **off()})
    cs = calls(d)
    ok = len(cs) >= 1 and cs[0]["function"]["name"] in ("get_weather", "get_time")
    record("tools", "tool_choice_required", ok, s=round(dt, 1), detail=msg(d))


def tool_named():
    d, dt = chat({"messages": [{"role": "user", "content": "I'm flying to Paris tomorrow."}],
                  "tools": [WEATHER, TIME_TOOL],
                  "tool_choice": {"type": "function", "function": {"name": "get_time"}},
                  "max_tokens": 512, "temperature": 1.0, **off()})
    cs = calls(d)
    ok = len(cs) == 1 and cs[0]["function"]["name"] == "get_time" and "timezone" in args_of(cs[0])
    record("tools", "tool_choice_named", ok, s=round(dt, 1), detail=msg(d))


def tool_nested():
    prompt = ("Book a 45-minute meeting titled 'Q4 roadmap' on 2026-10-20 at 14:00 with Ana (ana@example.com) "
              "and Raj (raj@example.com), reminder 10 minutes before.")
    d, dt = chat({"messages": [{"role": "user", "content": prompt}], "tools": [CREATE_EVENT],
                  "max_tokens": 2048, "temperature": 1.0, **off()})
    cs = calls(d)
    a = args_of(cs[0]) if cs else {}
    errs = validate(a, CREATE_EVENT["function"]["parameters"]) if cs else ["no call"]
    ok = (not errs and a.get("duration_minutes") == 45 and len(a.get("attendees", [])) == 2
          and a.get("reminder", {}).get("minutes_before") == 10 and a.get("start", "").startswith("2026-10-20T14:00"))
    record("tools", "nested_arguments", ok, s=round(dt, 1), detail=errs or a)


def tool_parallel():
    d, dt = chat({"messages": [{"role": "user", "content":
                  "Get the weather in celsius for both Oslo and Lima. Call the tool for each city."}],
                  "tools": [WEATHER], "max_tokens": 2048, "temperature": 1.0, **off()})
    cs = calls(d)
    cities = sorted(args_of(c)["city"].lower() for c in cs)
    ok = len(cs) == 2 and any("oslo" in c for c in cities) and any("lima" in c for c in cities)
    record("tools", "parallel_calls_one_turn", ok, s=round(dt, 1), n_calls=len(cs), detail=cities)


def tool_round_trip(mode):
    def fn():
        extra = off() if mode == "off" else think()
        msgs = [{"role": "user", "content": "What's the weather in Berlin in celsius? Then tell me if I need a coat."}]
        d, _ = chat({"messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 1.0, **extra})
        cs = calls(d)
        if not cs:
            return record("tools", f"tool_result_round_trip_{mode}", False, detail=msg(d))
        msgs.append({"role": "assistant", "content": msg(d).get("content") or "", "tool_calls": cs})
        msgs.append({"role": "tool", "tool_call_id": cs[0]["id"],
                     "content": json.dumps({"city": "Berlin", "temp_c": 3, "conditions": "windy, light rain"})})
        d2, dt = chat({"messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 1.0, **extra})
        c = (msg(d2).get("content") or "").lower()
        ok = not calls(d2) and ("3" in c) and ("coat" in c or "jacket" in c)
        record("tools", f"tool_result_round_trip_{mode}", ok, s=round(dt, 1), detail=c[:300])
    return fn


def tool_stream():
    c, r, tools, u, ttft, total = chat_stream({"messages": [{"role": "user", "content":
                                               "Weather in Madrid, fahrenheit please."}],
                                               "tools": [WEATHER], "max_tokens": 1024, "temperature": 1.0, **off()})
    name = "".join(t.get("function", {}).get("name") or "" for t in tools)
    argstr = "".join(t.get("function", {}).get("arguments") or "" for t in tools)
    try:
        a = json.loads(argstr)
    except ValueError:
        a = None
    ok = name == "get_weather" and a is not None and not validate(a, WEATHER["function"]["parameters"])
    record("tools", "streaming_tool_call_deltas", ok, s=round(total, 1), detail={"name": name, "args": argstr})


# Mock file system for the multi-step agent loop.
def agent_loop(mode):
    def fn():
        fs = {"config.json": json.dumps({"service": "ingest", "port": 8041, "workers": 3}),
              "README.md": "Service notes: the port lives in config.json."}
        tools = [
            {"type": "function", "function": {"name": "list_files", "description": "List files in the workspace.",
             "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write a file (overwrites).",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                            "content": {"type": "string"}}, "required": ["path", "content"]}}}]
        msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to do the task, then reply DONE."},
                {"role": "user", "content": "Find the service port in this workspace, then write the port number "
                                            "plus 1 (just the number) to out.txt."}]
        extra = off() if mode == "off" else think()
        t0, steps = time.time(), 0
        for steps in range(1, 11):
            d, _ = chat({"messages": msgs, "tools": tools, "max_tokens": 4096, "temperature": 1.0, **extra})
            m = msg(d)
            cs = m.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": cs} if cs else {})})
            if not cs:
                break
            for c in cs:
                a = args_of(c)
                n = c["function"]["name"]
                if n == "list_files":
                    out = json.dumps(sorted(fs))
                elif n == "read_file":
                    out = fs.get(a.get("path", ""), "ERROR: no such file")
                elif n == "write_file":
                    fs[a["path"]] = a["content"]
                    out = "ok"
                else:
                    out = "ERROR: unknown tool"
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        ok = fs.get("out.txt", "").strip() == "8042"
        record("tools", f"agent_loop_{mode}", ok, steps=steps, s=round(time.time() - t0, 1),
               detail={"out.txt": fs.get("out.txt"), "last": msgs[-1].get("content", "")[:200]})
    return fn


# ------------------------------------------------------------------ schema --
PERSON = {"type": "object", "additionalProperties": False, "required": ["name", "age", "email", "skills", "level"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 130},
                         "email": {"type": "string", "pattern": r"[^@\s]+@[^@\s]+\.[a-z]+"},
                         "skills": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                         "level": {"type": "string", "enum": ["junior", "mid", "senior"]}}}


def schema_strict(mode, n=5):
    def fn():
        good, bad = 0, []
        for i in range(n):
            extra = off() if mode == "off" else think()
            d, _ = chat({"messages": [{"role": "user", "content":
                         "Invent a fictional software engineer profile."}],
                         "response_format": {"type": "json_schema", "json_schema": {
                             "name": "person", "schema": PERSON, "strict": True}},
                         "max_tokens": 2048, "temperature": 1.0, **extra})
            try:
                errs = validate(json.loads(msg(d)["content"]), PERSON)
            except (ValueError, TypeError) as e:
                errs = [f"not JSON: {e}: {str(msg(d).get('content'))[:120]}"]
            good += not errs
            if errs:
                bad.append(errs)
        record("schema", f"json_schema_strict_{mode}", good == n, valid=f"{good}/{n}", detail=bad[:2])
    return fn


def schema_json_object():
    d, _ = chat({"messages": [{"role": "user", "content":
                 "Return a JSON object with keys city, country, population for Lisbon."}],
                 "response_format": {"type": "json_object"}, "max_tokens": 512, "temperature": 1.0, **off()})
    try:
        o = json.loads(msg(d)["content"])
        ok = isinstance(o, dict) and {"city", "country", "population"} <= set(o)
    except (ValueError, TypeError):
        ok, o = False, msg(d).get("content")
    record("schema", "json_object_mode", ok, detail=o)


def schema_choice():
    d, _ = chat({"messages": [{"role": "user", "content": "Sentiment of: 'This update broke everything.'"}],
                 "structured_outputs": {"choice": ["positive", "negative", "neutral"]},
                 "max_tokens": 16, "temperature": 1.0, **off()})
    c = (msg(d).get("content") or "").strip()
    record("schema", "choice_constraint", c == "negative", got=c)


def schema_regex():
    d, _ = chat({"messages": [{"role": "user", "content": "Give me a US phone number for a fictional business."}],
                 "structured_outputs": {"regex": r"\(\d{3}\) \d{3}-\d{4}"},
                 "max_tokens": 32, "temperature": 1.0, **off()})
    c = (msg(d).get("content") or "").strip()
    record("schema", "regex_constraint", bool(re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", c)), got=c)


def schema_prompt_only():
    """No constraint: JSON from instructions alone (what clients without response_format get)."""
    good = 0
    for _ in range(5):
        d, _ = chat({"messages": [{"role": "user", "content":
                     "Reply with ONLY a JSON object (no prose, no code fence) with keys "
                     "\"title\" (string), \"tags\" (array of 3 strings), \"rating\" (integer 1-5) for the movie Alien."}],
                     "max_tokens": 512, "temperature": 1.0, **off()})
        try:
            o = json.loads(msg(d)["content"].strip())
            good += isinstance(o.get("tags"), list) and len(o["tags"]) == 3 and 1 <= int(o["rating"]) <= 5
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    record("schema", "json_from_prompt_only", good >= 4, valid=f"{good}/5")


# --------------------------------------------------------------- reasoning --
def reasoning_modes():
    q = "If a train leaves at 14:37 and arrives at 17:12, how many minutes is the trip? Reply with the number."
    out = {}
    for label, extra in [("off", off()), ("low", think("low")), ("high", think("high")), ("max", think("max"))]:
        d, dt = chat({"messages": [{"role": "user", "content": q}], "max_tokens": 8192, "temperature": 1.0, **extra})
        m = msg(d)
        r = m.get("reasoning") or m.get("reasoning_content") or ""
        out[label] = {"reasoning_chars": len(r), "answer_ok": "155" in (m.get("content") or ""),
                      "tokens": d["usage"]["completion_tokens"], "s": round(dt, 1)}
    ok = (out["off"]["reasoning_chars"] == 0 and out["high"]["reasoning_chars"] > 0
          and all(v["answer_ok"] for v in out.values()))
    record("reasoning", "thinking_modes", ok, detail=out)
    print("       " + json.dumps(out), flush=True)


def reasoning_no_leak():
    d, _ = chat({"messages": [{"role": "user", "content": "Is 221 prime? Answer yes or no, then one sentence."}],
                 "max_tokens": 4096, "temperature": 1.0, **think()})
    c = msg(d).get("content") or ""
    ok = "</think>" not in c and "<think>" not in c and "no" in c.lower()
    record("reasoning", "no_think_tag_leak", ok, detail=c[:200])


def stop_sequence():
    d, _ = chat({"messages": [{"role": "user", "content": "Count from 1 to 20 separated by spaces."}],
                 "stop": [" 7"], "max_tokens": 128, "temperature": 0, **off()})
    c = msg(d).get("content") or ""
    record("format", "stop_sequence", "7" not in c.split() and "6" in c, got=c)


def multilingual():
    d, _ = chat({"messages": [{"role": "user", "content": "Translate to Japanese and French: 'The library opens at nine.'"}],
                 "max_tokens": 256, "temperature": 1.0, **off()})
    c = msg(d).get("content") or ""
    ok = bool(re.search(r"[぀-ヿ一-鿿]", c)) and ("biblioth" in c.lower())
    record("format", "multilingual_utf8", ok, detail=c[:200])


# ------------------------------------------------------------ long context --
def needle(target_tokens):
    def fn():
        hay = ("In the archive, clerks filed weather logs, ship manifests, and tax rolls in no special order. ")
        n = max(1, target_tokens // 20)
        parts = [hay] * n
        parts.insert(int(n * 0.63), "The secret launch code is ORCHID-7731. ")
        prompt = "".join(parts) + "\nWhat is the secret launch code? Reply with the code only."
        d, dt = chat({"messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0, **off()},
                     timeout=1800)
        c = msg(d).get("content") or ""
        pt = d["usage"]["prompt_tokens"]
        record("long_context", f"needle_{target_tokens // 1000}k", "ORCHID-7731" in c, prompt_tokens=pt,
               s=round(dt, 1), got=c.strip()[:40])
    return fn


# ------------------------------------------------------------ router path --
def router_tools_stream():
    if not ROUTER_KEY:
        return record("router", "chat_tools_stream", False, detail="V41_ROUTER_API_KEY not set")
    c, r, tools, u, ttft, total = chat_stream({"messages": [{"role": "user", "content": "Weather in Rome, celsius."}],
                                               "tools": [WEATHER], "max_tokens": 2048}, base=ROUTER, key=ROUTER_KEY)
    argstr = "".join(t.get("function", {}).get("arguments") or "" for t in tools)
    try:
        ok = "rome" in json.loads(argstr)["city"].lower()
    except (ValueError, KeyError):
        ok = False
    record("router", "chat_tools_stream", ok, s=round(total, 1), detail=argstr)


def router_default_thinking():
    if not ROUTER_KEY:
        return
    d, dt = chat({"messages": [{"role": "user", "content": "What is 12*13? Reply with the number."}],
                  "max_tokens": 4096}, base=ROUTER, key=ROUTER_KEY)
    m = msg(d)
    record("router", "default_request_hermes_shape", "156" in (m.get("content") or ""), s=round(dt, 1),
           reasoning_chars=len(m.get("reasoning_content") or m.get("reasoning") or ""))


TESTS = [
    ("speed", "decode_prose_off", speed_decode("prose", PROSE, "off")),
    ("speed", "decode_code_off", speed_decode("code", CODE, "off")),
    ("speed", "decode_prose_think", speed_decode("prose", PROSE[:1], "think")),
    ("speed", "decode_code_think", speed_decode("code", CODE[:1], "think")),
    ("speed", "prefill_8k", speed_prefill(8000)),
    ("speed", "prefill_32k", speed_prefill(32000)),
    ("tools", "single_off", tool_single("off")),
    ("tools", "single_think", tool_single("think")),
    ("tools", "no_call", tool_no_call),
    ("tools", "required", tool_required),
    ("tools", "named", tool_named),
    ("tools", "nested", tool_nested),
    ("tools", "parallel", tool_parallel),
    ("tools", "round_trip_off", tool_round_trip("off")),
    ("tools", "round_trip_think", tool_round_trip("think")),
    ("tools", "stream", tool_stream),
    ("tools", "agent_off", agent_loop("off")),
    ("tools", "agent_think", agent_loop("think")),
    ("schema", "strict_off", schema_strict("off")),
    ("schema", "strict_think", schema_strict("think", 3)),
    ("schema", "json_object", schema_json_object),
    ("schema", "choice", schema_choice),
    ("schema", "regex", schema_regex),
    ("schema", "prompt_only", schema_prompt_only),
    ("reasoning", "modes", reasoning_modes),
    ("reasoning", "no_leak", reasoning_no_leak),
    ("format", "stop", stop_sequence),
    ("format", "multilingual", multilingual),
    ("long_context", "needle_64k", needle(64000)),
    ("long_context", "needle_200k", needle(200000)),
    ("router", "tools_stream", router_tools_stream),
    ("router", "default", router_default_thinking),
]

if __name__ == "__main__":
    t0 = time.time()
    for g, n, f in TESTS:
        run(g, n, f)
    json.dump(RESULTS, open(OUT, "w"), indent=1)
    p = sum(r["pass"] for r in RESULTS)
    print(f"\n{p}/{len(RESULTS)} passed in {time.time() - t0:.0f}s -> {OUT}")
    sys.exit(0 if RESULTS and p == len(RESULTS) else 1)
