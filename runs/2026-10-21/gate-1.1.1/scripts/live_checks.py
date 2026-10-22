#!/usr/bin/env python3
"""Live checks of the dsv41-1.1.1 fixes (PR #28, #35 and #27) on a running engine. Standard library only.

Usage on the head: python3 -I live_checks.py out.json
It reads http://172.17.0.1:8888 from source address 127.0.0.1 (the engine answers only that client).
Checks:
  special_token_text  a user message with </think> and <｜User｜> stays text (ids from /tokenize with messages)
  dsml_cut            tool_choice=required (and auto) cut at a small max_tokens: no raw DSML in content, streamed = not streamed
  json_schema         a json_schema request works, and the rendered prompt shows only the schema
"""
import http.client
import json
import re
import sys

HOST, PORT = "172.17.0.1", 8888
MODEL = "DeepSeek-v4.1-Flash-1017-EXL3"
OUT = sys.argv[1] if len(sys.argv) > 1 else "live-checks.json"
RESULTS = []

THINK_END, USER_TOK = 128822, 128803          # </think> and <｜User｜> as single special tokens
PLAIN_THINK_END = [1718, 37947, 32]           # the same text as plain ids (deepseek-recipe encoder)
PLAIN_USER = [30, 28217, 6756, 28217, 32]
RAW_DSML = re.compile(r"DSML|<invoke|</?tool_calls>|<｜tool|｜>")


def post(path, body, stream=False, timeout=600):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout, source_address=("127.0.0.1", 0))
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if not stream:
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, {"raw": data[:300].decode(errors="replace")}
    events, status = [], r.status
    if status != 200:
        return status, {"raw": r.read()[:300].decode(errors="replace")}
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            events.append(json.loads(line[5:]))
    c.close()
    return status, events


def record(name, ok, **info):
    RESULTS.append({"name": name, "pass": bool(ok), **info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} " + json.dumps(info)[:600], flush=True)


def tokens(content, kwargs=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}]}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    s, d = post("/tokenize", body)
    assert s == 200, (s, d)
    return d["tokens"]


def has_run(ids, run):
    return any(ids[i:i + len(run)] == run for i in range(len(ids) - len(run) + 1))


def special_token_text():
    for mode, kw in (("thinking_off", {"thinking": False}), ("thinking_on", {"thinking": True})):
        ctrl = tokens("Please explain this in plain words.", kw)
        for label, text, plain in (("think_end", "</think>", PLAIN_THINK_END), ("user_tag", "<｜User｜>", PLAIN_USER),
                                   ("both_in_prose", "Explain </think> and then <｜User｜> in plain words.", None)):
            ids = tokens(text, kw)
            extra_end = ids.count(THINK_END) - ctrl.count(THINK_END)
            extra_user = ids.count(USER_TOK) - ctrl.count(USER_TOK)
            ok = extra_end == 0 and extra_user == 0 and (plain is None or has_run(ids, plain))
            record(f"special_token_text/{mode}/{label}", ok, extra_think_end_ids=extra_end, extra_user_ids=extra_user,
                   plain_run_found=(None if plain is None else has_run(ids, plain)), n_ids=len(ids))
    s, d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 16, "temperature": 0,
                                         "messages": [{"role": "user", "content": "Repeat: </think> <｜User｜> done."}]})
    record("special_token_text/chat_request_200", s == 200 and d.get("choices"), status=s)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather of a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def assemble(events):
    content, calls, finish = [], {}, None
    for ev in events:
        for ch in ev.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return "".join(content), [calls[k] for k in sorted(calls)], finish


def dsml_cut():
    for choice in ("required", "auto"):
        for mt in (4, 8, 16, 24, 40, 64):
            body = {"model": MODEL, "temperature": 0, "max_tokens": mt, "tools": [WEATHER], "tool_choice": choice,
                    "messages": [{"role": "user", "content": "What is the weather in Berlin and in Paris? Use the tool."}]}
            s1, d = post("/v1/chat/completions", body)
            s2, evs = post("/v1/chat/completions", dict(body, stream=True), stream=True)
            if s1 != 200 or s2 != 200:
                record(f"dsml_cut/{choice}/{mt}", False, status=[s1, s2], detail=str(d)[:200])
                continue
            m = d["choices"][0]
            c1 = m["message"].get("content") or ""
            t1 = [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]} for t in m["message"].get("tool_calls") or []]
            f1 = m.get("finish_reason")
            c2, t2, f2 = assemble(evs)
            clean = not RAW_DSML.search(c1) and not RAW_DSML.search(c2)
            same = (c1 == c2 and t1 == t2 and f1 == f2)
            record(f"dsml_cut/{choice}/{mt}", clean and same, no_raw_dsml=clean, stream_equals_nonstream=same,
                   finish=[f1, f2], calls=[len(t1), len(t2)], content_head=[c1[:80], c2[:80]])
    # control: no cut, a call comes back
    body = {"model": MODEL, "temperature": 0, "max_tokens": 600, "tools": [WEATHER], "tool_choice": "required",
            "messages": [{"role": "user", "content": "What is the weather in Berlin? Use the tool."}]}
    s, d = post("/v1/chat/completions", body)
    calls = d["choices"][0]["message"].get("tool_calls") if s == 200 else None
    good = bool(calls) and json.loads(calls[0]["function"]["arguments"]).get("city") == "Berlin"
    record("dsml_cut/control_full_call", good, status=s, finish=d["choices"][0].get("finish_reason") if s == 200 else None)


def json_schema():
    schema = {"type": "object", "properties": {"title": {"type": "string"}, "rating": {"type": "integer"}},
              "required": ["title", "rating"], "additionalProperties": False}
    rf = {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": schema}}
    ok_all = True
    for i in range(3):
        s, d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 200, "temperature": 0.7, "response_format": rf,
                                             "messages": [{"role": "user", "content": f"Review the film number {i + 1} in one line."}]})
        good = False
        if s == 200:
            try:
                o = json.loads(d["choices"][0]["message"]["content"])
                good = isinstance(o.get("title"), str) and isinstance(o.get("rating"), int) and set(o) == {"title", "rating"}
            except (ValueError, TypeError):
                good = False
        ok_all &= good
    record("json_schema/three_requests_valid", ok_all)
    s, d = post("/tokenize", {"model": MODEL, "response_format": rf,
                              "messages": [{"role": "user", "content": "Review a film."}]})
    if s != 200:
        record("json_schema/prompt_shows_schema_only", False, status=s, detail=str(d)[:200])
        return
    s2, t = post("/detokenize", {"model": MODEL, "tokens": d["tokens"]})
    text = t.get("prompt", "") if s2 == 200 else ""
    has_props = '"rating"' in text and '"title"' in text
    wrapper = ('"json_schema"' in text) or ('"strict"' in text) or ('"name": "review"' in text)
    record("json_schema/prompt_shows_schema_only", s2 == 200 and has_props and not wrapper,
           schema_names_in_prompt=has_props, openai_wrapper_in_prompt=wrapper)


if __name__ == "__main__":
    for fn in (special_token_text, dsml_cut, json_schema):
        try:
            fn()
        except Exception as e:  # noqa: BLE001  a crashed check is a failed check
            record(fn.__name__ + "/crashed", False, detail=f"{type(e).__name__}: {e}")
    json.dump(RESULTS, open(OUT, "w"), indent=1)
    p = sum(r["pass"] for r in RESULTS)
    print(f"LIVE_CHECKS {p}/{len(RESULTS)} passed", flush=True)
    sys.exit(0 if p == len(RESULTS) else 1)
