# Zig engine release fault campaign: test plan and results

Task (2026-10-21, from the main session; the user's words: "lets switch over to our zig engine and run a bunch of
tests to check for faults before we release it"). Engine: the Zig TensorFold line, dsv41 family, DeepSeek-V4.1-Flash-1017
EXL3 2.9 bpw, TP=2 on the two GB10 nodes. Nothing is released, tagged or merged by this campaign.

## Engine under test

- Ref: `Elevatormusic/TensorFold` `dsv41-zig-dev` @ `a3c2aa01` = `dsv41-zig` + PRs #29-#33 (top_k 0, RDMA GID
  setting, structured output, DSML tool calls with the cut containment and the string-value grammar, fault isolation).
- One image for the whole campaign: `tensorfold-dsv41-zig:a3c2aa01`, image ID `sha256:5852b192e685...` on the
  worker. The head gets the same image by `docker save | docker load` (not the thin rebuild of the earlier windows,
  whose ID differs). Each window logs the image ID on both nodes and stops if they differ.
- Launch: as the earlier windows (`zigwin.sh`), on 127.0.0.1:8899 (never the router's 8888), `--parallel 4`,
  `--context 1048576`, sampling defaults temperature 1.0, top_p 1.0, top_k 20, thinking off,
  `TF_DS_EXIT_ON_FAULT=1`, `TF_RDMA_GID_INDEX` from the port's RoCE v2 IPv4 GID at each start. NCCL GID detection
  stays automatic (no `NCCL_IB_GID_INDEX`). `--oom-score-adj 1000`, cgroup parent `tensorfold.slice`.
- Python control: the Python 1.1.1 engine, image `tensorfold-dsv41:cb4ceaff2b8c` (the same ID on both nodes), as a
  test launch with its own container names (`zrel-py-r0/r1`) on 127.0.0.1:8898, own spill directory. The production
  unit `dgx-v41-1017-serving`, `tfv41.sh` and the router are not touched.

## Window rules

- Worker lock first (`flock /tmp/spark-gpu.lock`), then the head lock by a holder with a unique token
  (`/tmp/zig-lock.run.<token>`). At most 2 h a window (most are under 1 h), a gap of at least 15 min between my
  windows. Before a start: no other containers (router excepted) and at least 110 GiB free on both nodes.
- Memory guard on both nodes: kill only my container when MemAvailable is under 1,280 MiB. Stop rule for my
  clients: stop sending when head MemAvailable is under 1.25 GiB. earlyoom acts at 1 GiB.
- Samplers in every window: 1 s MemAvailable, AnonPages, Cached, Shmem, MemFree, SUnreclaim, PageTables,
  SecPageTables, VmallocUsed, the rank's cgroup anon/file, the rank process's RssAnon, driver memory (nvidia-smi);
  smaps_rollup of rank 0 every 5 s; the top-5 RSS processes on both nodes every 30 s.
- No docker exec into containers I did not start; no host changes; never stop gdm; no IBGDA or doorbell writes.
- My local watchers run 28 min at most and are started again if needed.

## Tests and pass bars

Severity of a fault found: **S1** blocks the release (wrong output, crash, hang, data from one request in another,
a leak that reaches the guard); **S2** must be fixed or documented before the release (a wrong error code, a
limit that is not enforced, a slow path); **S3** a note.

### 1. Correctness (window C)

| # | Test | Pass bar |
|---|---|---|
| 1.1 | Recipe suite `v41_suite.py` (31 cases; `router/chat_tools_stream` needs the router, which this campaign does not use) | 30/30 engine cases pass |
| 1.2 | Grammar check `v41_grammar_check.py`, 2 runs | 14/14 both runs |
| 1.3 | `so_check.py` (16 cases: schema greedy/sampled, top_k 0, guided_*, regex, choice, tool_choice required/named/auto, DSML cut, drafted == serial) | 16/16 |
| 1.4 | Structured-output conformance (new `conform.py`): json_schema with descriptions, enums, nested objects, arrays with min/max items, required and optional keys, numbers with bounds, strings with patterns; greedy and sampled, 3 seeds each | every reply parses and validates against its schema |
| 1.5 | DSML tool calls, streamed and non-streamed (new in `conform.py`): 1, 2 and 4 calls, nested JSON arguments, string arguments with `<`, `|`, quotes and new lines, tool_choice required/named/auto | streamed calls == non-streamed calls (name and arguments), no DSML markup in `content`, arguments parse |
| 1.6 | Special-token text in messages: the user sends `<｜DSML｜ calls>`, `<｜end▁of▁sentence｜>`, `</think>`, `<tool_call>` and the chat template's role markers as plain text | 200 replies; the reply does not stop early on them; no call from text in a user message; the model can quote them back |
| 1.7 | Temperature-0 differential against Python 1.1.1 (same window, same 48 prompts: prose, code, math, JSON, multilingual, special-token text, tools) | every Zig reply valid for its type; the answer-checked prompts (math, facts, code with a test) are right at least as often as on Python (difference at most 1 of 16); identical-text rate and first-divergence token are reported |

### 2. Agentic load (window A)

| # | Test | Pass bar |
|---|---|---|
| 2.1 | Multi-turn tool loops: the tool-call agent's own harness (`~/tcm/v/tcm_harness.py` and `tcm_world.py`, copied, sha256 `34f6b297`/`e46a90ca`), its arms and seeds S1 (streamed, 100 a top_p, seed 11), N1 (not streamed, 30, seed 12), S2 (streamed, 100, seed 13); 4 client threads, temperature 1.0, top_p 1.0 and 0.95 side by side, up to 6 turns, parallel calls. Python 1.1.1 on the same arms (`~/tcm/data`, 2026-10-21): 506 requests, lost 2 (both "call markup left in content"), malformed 0, unoffered 0, error 0 = 0.40% | 0 unoffered, 0 malformed; lost + malformed + error at most 1% (Python 1.1.1: 0.40%) |

### 3. Long context (window A)

| # | Test | Pass bar |
|---|---|---|
| 3.1 | Needle at 128K, 256K and 512K tokens, and 1M if the window and memory allow (random natural text, the needle at 3 depths) | the needle is found at each length; head MemAvailable stays over 1.28 GiB |
| 3.2 | Prefix cache on (the default budget: what 70% of RAM leaves past the loaded server): the same 32K and 128K prompts twice | the second request reports `cached_tokens` > 0 and gives the same greedy answer |
| 3.3 | Prefix cache off (`--prompt-cache-gib 0`, a restart): the same prompt twice | `cached_tokens` 0 both times; the same greedy answer |

### 4. Robustness (window R)

| # | Test | Pass bar |
|---|---|---|
| 4.1 | Client disconnect mid-stream (after 5 chunks) and during prefill of a 64K prompt; 20 times | the server stays up; the stream's slot is free (the next 4 parallel requests start at once); no error lines but the cancel |
| 4.2 | Burst above capacity: 8 and 16 requests at once with `--parallel 4` | every request completes or gets a clear 429/503; none hangs (all done in 10 min); later requests are fine |
| 4.3 | Bad requests: broken JSON, an invalid JSON schema, an unknown `tool_choice`, `max_tokens` over the context, `max_tokens` 0 and negative, empty `messages`, a wrong role, a huge `n`, a 2 MB string, a tool with no name | each gets a 4xx with a message, none a 5xx or a crash; the server serves normally after the set |
| 4.4 | SIGTERM to rank 0 during a 256K prefill, and during decode | rank 0 exits within 30 s with a clean code (0 or 143), rank 1 exits too (75 or 0); no core; a restart is ready |
| 4.5 | Forced fault, exit 75: stop rank 1 during decode | rank 0's live requests get an error reply (no hang), rank 0 exits 75 after the grace; a restart is ready and serves |
| 4.6 | A restart right after a stop (no wait) | ready in under 2 min; the first reply is correct |
| 4.7 | Rank 1 dies when idle: stop rank 1 | rank 0 exits (75) without a hang; restart works |

### 5. Soak (windows S1, S2)

| # | Test | Pass bar |
|---|---|---|
| 5.1 | Full mix (`soak.py`: plain, JSON schema, tool_choice required, top_k 0, streamed, 16K needle; keyed references alone and under load), 4 client threads, about 60 min a window, 2 windows | 0 fails, 0 reference mismatches, 0 round faults, no guard kill, no earlyoom; head low point over 1.28 GiB; rank 0 anon and driver memory flat in the last 30 min (under 100 MiB growth) |

### 6. Performance sanity (window C, same window as Python)

| # | Test | Pass bar |
|---|---|---|
| 6.1 | Cold 32K prefill (the coordinator's `pf32` kit: B natural text, D kit_bench words; `cached_tokens` 0 checked), median of 3, both engines | Zig at least 95% of Python on each prompt type |
| 6.2 | Decode at 1 stream (kit_bench set a, median of 3) and 4 streams, both engines | Zig at least 95% of Python at 1 and 4 streams |

## Windows

| Window | Content | Length |
|---|---|---|
| C | Python 1.1.1 start, differential prompts, cold 32K, decode 1/4; stop; Zig start, the same, then 1.1-1.6 | about 60 min |
| A | Zig: agentic loops (2.1), long context with the cache on (3.1, 3.2); restart with the cache off (3.3) | about 60 min |
| R | Zig: robustness 4.1-4.7 (each restart about 35 s) | about 45 min |
| S1, S2 | Zig: soak 5.1 | about 70 min each |

## Results

**Change of target (2026-10-22, the user's decision via the main session):** the campaign runs on the EXL3 2.8 bpw
pack, the new default. Engine `dsv41-zig-dev` @ `32894bad` (= `a3c2aa01` + PR #38, the 2.8 pack loader), image
`tensorfold-dsv41-zig:32894bad`, image ID `sha256:89812ac5…` on both nodes. Zig view `~/zigdsv41/view-28` (the pack
without the Python dtype override), rank cache `~/zigdsv41/rank-cache-28`. Python control: 1.1.1 (`cb4ceaff2b8c`) on
the same pack (the coordinator's view with its dtype override, rank cache `tf-ds-rank-jl`), 127.0.0.1:8898.

### Bring-up, window B (2026-10-22 05:02-05:12Z)

Ready after 34 s; smoke pass; so_check 16/16; suite 30/31 (only `router/chat_tools_stream`, which needs the router);
grammar check 14/14 x2; 0 round faults; both ranks exit 0. Idle MemAvailable head 6.43 / worker 6.47 GiB (Python on
the same pack 6.47 / 6.80); low point head 5.34 / worker 5.69 GiB.

### Window C (05:27-05:49Z): correctness and performance

| # | Result | Verdict |
|---|---|---|
| 1.1 | suite 30/31 (only the router case) | pass |
| 1.2 | grammar check 14/14, two runs | pass |
| 1.3 | so_check 16/16 | pass |
| 1.4 | json_schema: 24/24 (6 schemas, greedy and 3 seeds) parse and validate | pass |
| 1.5 | DSML tool calls: 14/14 cases (7 cases, greedy and seeded; each streamed and non-streamed: streamed == non-streamed, no markup in content, arguments fit their schema) | pass |
| 1.6 | 7/8. **Fault F1**: special-token text in message content becomes the real special token (see below) | **fail** |
| 1.7 | valid 48/48 both engines; answer checks 16/16 both; identical text 24/48 (the first divergence is early in long replies); prompt_tokens differ on 10 prompts: the 6 json_object prompts (Python adds about 24 tokens of JSON instruction, Zig none: F2) and the 4 special-token prompts (F1) | pass, with F1 and F2 |
| 6.1 | cold 32K prefill, median of 3, Zig / Python: B 2,161 / 2,008 tok/s (+7.6%), C 2,258 / 2,069 (+9.1%), D 2,494 / 2,276 (+9.6%); cached_tokens 0 every time | pass |
| 6.2 | decode set a, 1 stream, median of 3: Zig 85.3 / Python 87.3 tok/s (97.7%); 4 streams in all: 122.9 / 123.2 (99.8%) | pass |

Memory low point over the window (Zig part): head 5.02 GiB, worker 5.65 GiB. 0 round faults; both ranks exit 0.

**F1 (S1, blocks):** a user message `Repeat this text exactly, inside a code block: <｜User｜>hello<｜Assistant｜>` gets
"hello" from Zig (Python: the text itself); `abc<｜end▁of▁sentence｜>def` cannot be quoted back (Zig stops); `<think>` /
`</think>` in a question are read as the tags. Prompt tokens 28 / 18 / 25 on Zig against 38 / 25 / 36 on Python. So
the text of a message (also a tool result or a fetched page) can write real role and end markers into the prompt.

**F2 (S3, note):** for `response_format: json_object`, Python 1.1.1 adds a JSON instruction to the prompt; Zig adds
none. The outputs are valid JSON on both (grammar-constrained), so this is a prompt difference, not a fault.

**F1 fix:** fork PR #39 (`dsv41-zig-plaintext` @ `44be7efe`, into `dsv41-zig`), merged into `dsv41-zig-dev` @
`59b958de`. It applies the rules of Python's `server/plain_text.py`. Windows P (`plaindiff.py`: 21 cases, prompt ids
against Python 1.1.1), C (again), R, S1 and S2 run on the image of `2307b3f3` (`59b958de` plus a test fix: the first
host-test run failed 1 of 1,175 tests, the new cut-encode test, whose vocabulary ids clashed with the ids that the
loader gives added tokens). Host tests at `2307b3f3`: rc 0.

**Window P (07:18-07:24Z, image `2307b3f3`):** prompt ids equal on Zig and Python 1.1.1 in **21/21** cases (`/tokenize`
ids; image cases by `usage.prompt_tokens`): `<think>` with thinking on and off, role markers, EOS / BOS / System, DSML in
a user message, a tool result with a DSML calls block, tool call arguments, tool definitions, a system message,
grounding in assistant content and reasoning (kept) and in a user turn (plain), the image placeholder as text, a
spelling split across two text parts, full-width look-alikes, marker look-alikes, no spelling, and three image requests
(a placeholder that the client wrote next to an image: 400 on both). Smoke pass; both ranks exit 0.

### Window C again (07:38-08:01Z, image `2307b3f3`, with the F1 fix)

| # | Result | Verdict |
|---|---|---|
| 1.1 | suite 30/31 (only the router case) | pass |
| 1.2 | grammar check 14/14, two runs | pass |
| 1.3 | so_check: all pass | pass |
| 1.4-1.6 | conform 46/46: json_schema, DSML calls, and 1.6 8/8 (think tags, DSML in user and system, EOS in a tool result, EOS quoted back: `abc<｜end▁of▁sentence｜>def`) | pass: **F1 fixed** |
| 1.7 | valid 48/48 both; answer checks 16/16 both; identical text 25/48; prompt_tokens differ only on the 6 json_object prompts (F2) | pass, with F2 |
| P | plaindiff 21/21 equal | pass |
| 6.1 | cold 32K prefill, median of 3, Zig / Python: B 2,161 / 2,004 tok/s (+7.8%), C 2,242 / 2,066 (+8.5%), D 2,487 / 2,268 (+9.7%); cached_tokens 0 | pass |
| 6.2 | decode set a (code), 1 stream: 85.3 / 87.1 tok/s (97.9%); prose 57.7 / 55.0; structured 104.9 / 107.9 (97.2%); 4 streams in all: 123.0 / 123.3 (99.8%) | pass |

Memory low point (Zig part): head 5.00 GiB, worker 5.13 GiB. 0 round faults; both ranks exit 0.

### Window A (06:04-06:49Z, image `32894bad`): agentic load and long context

| # | Result | Verdict |
|---|---|---|
| 2.1 | tcm arms S1 (213 requests), N1 (78), S2 (214): 505 requests; lost 0, malformed 0, unoffered 0, error 0 = 0.00% (top_p 1.0: 258, top_p 0.95: 247). Python 1.1.1: 0.40% | pass |
| 3.1 | needle found 7/7: 130,838 tokens (3 depths), 261,751 (2), 523,577, 1,038,664 | pass |
| 3.2 | prefix cache on: 32K twice, `cached_tokens` 0 then 30,720, 13.9 s then 1.0 s; 128K twice, 0 then 129,024, 57.9 s then 1.2 s; same answers | pass |
| 3.3 | prefix cache off (restart, `--prompt-cache-gib 0`): `cached_tokens` 0 and 0 at 32K and 128K; same answers; the same times both requests | pass |

Memory low point: head 3.25 GiB, worker 3.23 GiB (over the 1.28 GiB bar). 0 round faults; both ranks exit 0 at
both stops. Window A ran before the F1 fix; the fix changes only the prompt ids of a request whose text spells a
special token.

### Window R (08:16-08:33Z, image `2307b3f3`): robustness, first run

| # | Result | Verdict |
|---|---|---|
| 4.1 | 20 disconnects (10 streams after 5 chunks, 10 during a 64K prefill); then 4 at once: TTFT 0.95-1.16 s, all done in 1.7 s | pass |
| 4.2 | burst 8: 8 done (TTFT 0.08-4.62 s, 7.0 s); burst 16: 16 done (TTFT up to 9.98 s, 13.6 s); 0 refused, none hangs | pass |
| 4.3 | 23 bad requests, 0 5xx, 0 crashes; 18 get a 4xx; 5 get 200: unknown `tool_choice`, `max_tokens` 0, `max_tokens` -5, `temperature` -1, `top_p` 2 | **F4** (see below) |
| 4.4a, 4.4b, 4.5 | **not tested**: the `bg` launcher held its ssh session, so each signal came after the background requests had ended (the server was idle). Fixed in `rel.sh`; window R runs again after V | rerun |
| 4.6 | restart right after a stop: 33 s to a correct reply | pass |
| 4.7 | SIGKILL to rank 1 when idle: rank 0 still running after 180 s (seen twice: in 4.7 and in the idle 4.5) | **F3** (see below) |

Memory low point: head 5.30 GiB, worker 5.38 GiB. 0 round faults.

**Second run (11:46-12:06Z):** the `bg` fix (`< /dev/null`) was not enough: with `a && b &` the remote shell keeps
the ssh session open until `b` ends, so 4.4a, 4.4b and 4.5 again got their signals after the requests ended (fixed
with a subshell, `( ... & )`; a test of the three forms on the head: 7 s, 0 s, 1 s for a 6 s job). Third run queued.
What the second run did show:

- 4.7, now with a request: rank 1 killed when idle; after 60 s rank 0 still runs; **`/health` 200**; the next request
  gets **no reply in 120 s** (client timeout); rank 0 logs no round fault and still runs 4 min later.
- 4.3 on Python 1.1.1: the same 5 cases get 200 there too (unknown `tool_choice`, `max_tokens` 0, `max_tokens` -5,
  `temperature` -1, `top_p` 2); the other 18 get 400. Zig matches Python.
- 4.1, 4.2, 4.6 pass again (restart 34 s).

**F3 (S1, blocks):** when rank 1 dies while rank 0 is idle, rank 0 stays up, `/health` says 200, and the next request
hangs (no reply, no error, no exit 75). A supervisor that trusts `/health` keeps a broken pair, and every client waits
until its own timeout. The third run saves rank 0's thread states during the hang.

**F4 (S3, does not block):** the five invalid fields are served, not refused, on Zig and on Python 1.1.1 alike. The 4.3
bar asks for a 4xx; this is a known difference against the bar, not a regression.

**Third run (12:21-12:43Z), the signals now during the requests:**

| # | Result | Verdict |
|---|---|---|
| 4.4a | SIGTERM to rank 0 45 s into a 256K prefill: rank 0 exits 0 in 5 s, rank 1 exits 75; the request ends `cancelled` at 48 s | pass |
| 4.4b | SIGTERM to rank 0 20 s into a 4-stream decode: rank 0 exits 0 in 4 s, rank 1 exits 0; the 4 streams end `cancelled` | pass |
| 4.5 | SIGKILL to rank 1 20 s into a 4-stream decode: rank 0 still runs after 180 s; the 4 streams never end (no result in 10 min) | **fail: F3** |
| 4.6 | restart right after a stop: 33 s to a reply | pass |
| 4.7 | SIGKILL to rank 1 when idle: `/health` 200; the next request: no reply in 120 s; rank 0 still runs 3 min later. Rank 0's threads during the hang: 3 running, 68 in `futex_do_wait`, 2 in `ib_uverbs_event_read`, 7 in `poll_schedule_timeout`, none in a TCP read | **fail: F3** |

F3 so far: idle death 3 of 3 runs hang; death during decode 1 of 1 valid runs hang. The round waits on the RDMA ring's
arrival flags; no error completion came, so the ring's abort word (which every GPU wait reads) was never set.

**F3 fix:** fork PR #40 (`dsv41-zig-peerwatch` @ `a6dd2397`, into `dsv41-zig`), merged into `dsv41-zig-dev` @
`2b2c7fbd`. A watch thread on each rank polls the TCP links for the peer's end (`POLLRDHUP`, `HUP`, `ERR`); on a peer
gone without a done frame it aborts the RDMA ring (a live round then fails and its requests get the error) and exits 75
after the grace. A TCP keepalive (10 s, 3 probes 3 s apart) covers a node that vanishes. Three Codex review rounds
added: the follower keeps its watch on an error (so its exit 75 stays), the watch stops when a start fails, it aborts
every ring (TP4 too), and a follower that ends on an error releases its rings from the watch before `Model.close`
frees them. Window R runs again on the image of `dsv41-zig-dev` @ `92f3ea1b`, with a new case 4.5b (SIGKILL to rank 0
during decode: rank 1 must exit), after the coordinator's paired A/B on the GPU pair.

**Fourth run (14:08-14:20Z, image `87cf5678` = `dsv41-zig-dev` with PR #40 @ `0e92350a`):** host tests rc 0 first.

| # | Result | Verdict |
|---|---|---|
| 4.4a | SIGTERM to rank 0 45 s into a 256K prefill: rank 0 exits 0 in 6 s, rank 1 exits 75; the request ends `cancelled` at 48 s | pass |
| 4.4b | SIGTERM to rank 0 20 s into a 4-stream decode: rank 0 exits 0 in 5 s, rank 1 exits 0; the 4 streams end `cancelled` | pass |
| 4.5 | SIGKILL to rank 1 20 s into a 4-stream decode: rank 0 has 1 round fault, the 4 streams end with the error `PeerDown` (`server_error`) at 21 s, rank 0 exits **75 in 7 s** | pass: **F3 fixed** |
| 4.5b | SIGKILL to rank 0 20 s into a 4-stream decode: rank 1 exits **75 in 1 s**; the 4 streams lose their connection (the server process is gone) | pass |
| 4.6 | restart right after a stop: 30 s to a reply | pass |
| 4.7 | SIGKILL to rank 1 when idle: rank 0 exits **75 in 8 s** with no request | pass: **F3 fixed** |

After each case the next start serves the smoke prompt (26-34 s to ready). The Python control (`robust.py bad` on
1.1.1) again serves the same 5 invalid fields with 200 (F4). Memory low point: head 5,671 MiB, worker 5,740 MiB.
Codex then found one more teardown path (rank 0 of `tf-dsv41-lanes` that ends on an error stopped the watch before
its done frame, so the exit 75 was lost); fixed in `bbb861b7` (dev `fefb8c1d`). That fix changes the order of rank 0's
clean stop (the done frame first, then the watch stops). Its build and host tests are queued behind another job on
the worker; 4.4a, 4.4b and 4.6 cover that path on hardware and need one short rerun (about 12 min).

**Fifth run (17:19-17:31Z, image `952398c5` = dev with PR #40 @ `5a339278`, the one teardown `Watch.end`):** on the
coordinator's request, the watch, the rings and the exit 75 now have one teardown for every path (PR #40 description).
Host tests rc 0 first.

| # | Result | Verdict |
|---|---|---|
| 4.4a | SIGTERM to rank 0 in a 256K prefill: rank 0 exits 0 in 6 s, rank 1 exits 75; the request ends `cancelled` at 48 s | pass |
| 4.4b | SIGTERM to rank 0 during a 4-stream decode: rank 0 exits 0 in 4 s, rank 1 exits 0; the 4 streams end `cancelled` | pass |
| 4.5 | SIGKILL to rank 1 during decode: 1 round fault, the 4 streams end with `PeerDown`, rank 0 exits 75 in 7 s | pass |
| 4.5b | SIGKILL to rank 0 during decode: rank 1 exits 75 in 2 s | pass |
| 4.6 | restart right after the stop: 33 s to a reply | pass |
| 4.7 | SIGKILL to rank 1 when idle: rank 0 exits 75 in 8 s | pass |

Every next start serves the smoke prompt. Python control: the same 5 invalid fields get 200 (F4). Memory low point:
head 5,546 MiB, worker 5,485 MiB.

### Window S1 (08:48-09:54Z, image `2307b3f3`): soak

3,903 s, 4 client threads: ok 2,487, fail 1, reference mismatches 0 (23 reference checks under load); by kind (ok /
fail): tools 498/0, schema 528/1, plain 466/0, needle 264/0, top_k 0 498/0, stream 233/0. 0 round faults, 0 earlyoom
kills, both ranks exit 0. The one fail: a JSON-schema request (`max_tokens` 200, temperature 1.0) where the model
wrote a whole description inside the `city` string and reached the 200-token limit before the object closed (rank 0:
`tokens=200 finish=length`). This is the soak's token limit, not an engine fault: the output stayed inside the grammar.
Head MemAvailable: 6.36 GiB at the start, 4.4-4.5 GiB from 09:07 to the end (flat), low point 4.28 GiB. Rank 0
anonymous memory over the last 30 min: 1,698 to 1,724 MiB (+26 MiB). Verdict: **pass** (the fail is a test limit).
From S2 on, `soak.py` also records each fail's `finish_reason`.

### Window S2 (10:09-11:15Z, image `2307b3f3`): soak

3,905 s: ok 2,498, fail 1, reference mismatches 0 (23 checks under load); tools 500/0, schema 529/1, plain 469/0,
needle 265/0, top_k 0 500/0, stream 235/0. 0 round faults, 0 earlyoom kills, both ranks exit 0. The one fail is the
same request as in S1 (the soak's seeds repeat: the same "Elias Thorne ... Port Kestrel" reply at t = 1,564 s),
`finish_reason: length`: the same seed gives the same reply in two runs, so the case is deterministic. Head
MemAvailable 4.56-4.72 GiB from 10:40 to the end (flat), low point 4.36 GiB. Rank 0 anonymous memory over the last 30
min: 1,697 to 1,721 MiB (+24 MiB). Verdict: **pass** (the fail is a test limit).

### Window V (11:30-11:31Z, image `2307b3f3`, 2.9 bpw pack): the other pack still serves

Ready after 26 s (the 2.9 view and its own rank cache `~/zigdsv41/rank-cache`); smoke pass; so_check all pass; 0
round faults; both ranks exit 0. Verdict: **pass**.

## Release go/no-go

| Fault | Severity | State | Blocks |
|---|---|---|---|
| F1: special-token text in a message became the real token | S1 | fixed (PR #39, Codex 👍); P 21/21, C 1.6 8/8 | no |
| F3: a rank whose peer died stayed up and hung requests | S1 | fixed (PR #40, one teardown `Watch.end`); R on `952398c5`: 4.5, 4.5b, 4.7 exit 75 in 2-8 s, clean stops pass | no, after Codex 👍 on #40 |
| F2: `json_object` gets no JSON instruction in the prompt (Python adds one) | S3 | known difference; output valid JSON on both | no |
| F4: 5 invalid fields served with 200 (`tool_choice` unknown, `max_tokens` 0 / -5, `temperature` -1, `top_p` 2) | S3 | known difference; Python 1.1.1 does the same | no |
| Soak: 1 schema request per soak ends `finish_reason: length` at 200 tokens | none | a limit of the soak test (deterministic seed), not an engine fault | no |

Passed with no fault: B, A (tcm 505 requests 0.00% lost; needle 7/7 up to 1,038,664 tokens; prefix cache on/off), C
(suite, grammar, so_check, conform 46/46, prefill +7.8-9.7% against Python, decode 97.2-99.8% of Python), S1 and S2
(about 2,500 requests each, 0 reference mismatches, memory flat), V (2.9 pack serves).

Memory against Python 1.1.1 on the 2.8 pack: idle MemAvailable head 6.43 / worker 6.47 GiB on Zig, 6.47 / 6.80 GiB on
Python; low points head 5.00-5.34 GiB in B/C/R, 3.25 GiB at the 1M needle (A), 4.28-4.36 GiB in the soaks.

**Verdict: go** for Zig + 2.8 (image of `dsv41-zig-dev` @ `952398c5`) as the default when Codex gives PR #40 its 👍.
The R rerun on that image passed. Rollback: Python 1.1.1 + 2.9.
