# TensorFold dsv41-1.1.1 release gate: report

Date: 2026-10-21 (nodes, UTC 19:13 to 22:16).
Candidate: fork branch `deepseek-v41-tp2`, commit `cb4ceaff2b8cc0583a1377ce4329bbb1291ced0e`. Image `tensorfold-dsv41:cb4ceaff2b8c`, image ID `a760ff885c96`, same ID on the head and the worker. Git HEAD in the image is the full commit. Pip pins equal the baseline image (`transformers 5.20.2`, `xgrammar 0.2.8`, same NGC torch).
Baseline: `tensorfold-dsv41:fc9200a44463` (release 1.1.0, same git tree as `730d41abd8f4`).
The production unit, the production launcher and the router were not touched.

## Verdict

**One check of the pass bar is not met, by a small margin: the lowest head MemAvailable in the long 4-stream run (item 5).** All other checks pass. The miss is smaller than the run-to-run spread of the baseline itself (see "Head memory"). The release was published with a known-issue line about the head memory margin.

| Check | Result |
| --- | --- |
| 1. Recipe suite (recipe main, with the PR #38 fix), direct to the engine | 32/32 on both images, in both windows (same test set) |
| 2. Grammar check (`v41_grammar_check.py 2`) | 14/14 on both images, in both windows |
| 3. Decode benches (1 and 4 streams, prose and code) | within noise (table below) |
| 4. Cold 32K prefill (B random natural text, D kit_bench words, 3 interleaved runs each, `cached_tokens` 0 in all) | equal or better (table below) |
| 5. Long 4-stream run, identical tokens, 0 errors | PASS for tokens and errors (4 of 4 hashes equal in all four runs). Lowest head MemAvailable: candidate LOWER by 227 MiB (window 1) and 100 MiB (window 2). **Not met as written.** |
| 6a. `</think>` and `<｜User｜>` in a user message stay text | PASS (6/6 id checks: no extra special id, the plain id run is present) |
| 6b. tool_choice required/auto cut at small max_tokens: no raw DSML in content, stream = not stream | PASS (12/12 cuts, plus a full-call control) |
| 6c. json_schema request works; the prompt shows only the schema | PASS (3/3 valid; no OpenAI wrapper in the rendered prompt) |
| 6d. SIGTERM to rank 0 during a prefill | PASS: exit code 0 after 2.6 s, no `terminate called` in either rank log (window 2: request in the prefill for 11 s before the signal) |
| 7. GID | PASS: started with `NCCL_IB_GID_INDEX` and `TF_RDMA_GID_INDEX` both unset. Detected index 3 on both ports of both nodes (`RDMA GID index of ... port 1: 3 (detected)`). `decode gathers over RDMA writes` is printed, and `gathers stay on NCCL` is not. NCCL cuMem is off by default. Baseline ran with the pinned index 3. |
| Safety | No watchdog trip (limit 1.25 GiB), no earlyoom event on either node, 0 late graph captures, 0 WARNING lines, 0 admission waits |

The baseline on the live checks: 5/22 (the 17 fails are the bugs that the PRs fix: 6 special-token ids, 10 raw DSML cuts, 1 json_schema wrapper). So the live checks do detect the old behaviour.

## Method

- Two windows (the same launcher, same order, production settings: graphs on, `SPILL_GIB=64`, `--parallel 4`, 1M context, chunk 2048, thinking off, a fresh spill folder per start).
  - Window 1: baseline, then candidate (19:13 to 19:50 UTC).
  - Window 2: candidate, then baseline (21:42 to 22:16 UTC). Window 2 repeats window 1 with the order swapped, because the head memory result of window 1 was unclear.
- A first try of window 2 (20:09 UTC) was stopped after the candidate step. Its spill folder was reused from window 1, so the prompts hit the prefix cache (`cached=49152`). Those logs are not used and not kept here. The launcher now makes a fresh spill folder at each start.
- Order in each step: start, smoke, suite, grammar check, bench, cold 32K prefill, live checks, long 4-stream run (20K/50K/100K/200K, 1,500 new tokens, seed 7), then (candidate only) the SIGTERM test. MemAvailable sampled every 1 s on both nodes.
- Watchdog: stops both containers if head MemAvailable is under 1.25 GiB. It did not trip.
- Cleanup: the rank containers and the lock markers were removed, and my spill folders were deleted with a throwaway container of the same image. No `docker exec`, no `gdm` stop, no host change, no IBGDA code.

## Decode (bench, tok/s)

| Case | Baseline W1 / W2 | Candidate W1 / W2 |
| --- | --- | --- |
| Prose x1, per stream | 53.4 / 53.4 | 54.8 / 54.7 |
| Code x1, per stream | 101.4 / 101.3 | 99.9 / 99.4 |
| Prose x4, per stream (aggregate) | 27.3 (101.1) / 27.1 (100.3) | 27.2 (101.8) / 27.0 (101.2) |
| Code x4, per stream (aggregate) | 40.3 (147.7) / 40.1 (147.3) | 40.2 (148.9) / 40.0 (148.0) |
| Prefix cache, 24K prompt cold / warm TTFT, s | 11.37 / 1.11 and 11.21 / 1.31 | 11.14 / 1.07 and 11.21 / 1.17 |

The suite decode rows (thinking off, prose / code) are 57.8 / 96.3 and 56.4 / 90.1 (baseline), 58.0 / 95.4 and 59.0 / 98.3 (candidate). The differences are inside the noise (about 2%).

## Cold 32K prefill (median of 3, `cached_tokens` 0, production settings)

| Prompt | Baseline W1 / W2 | Candidate W1 / W2 |
| --- | --- | --- |
| B random natural text, TTFT s, tok/s | 17.20 s, 1,862 [1,822-1,937] / 16.42 s, 1,950 | 16.29 s, 1,966 [1,966-1,967] / 16.35 s, 1,959 [1,945-1,961] |
| D kit_bench words (32,768 tokens), TTFT s, tok/s | 15.45 s, 2,121 [2,116-2,220] / 14.71 s, 2,228 | 14.56 s, 2,250 [2,248-2,258] / 14.67 s, 2,234 [2,201-2,246] |

The suite prefill rows (salted, varied text, `cached_tokens` 0): 32K 1,875 and 1,913 tok/s (baseline), 1,971 and 1,963 tok/s (candidate).

## Head memory (long 4-stream run)

| | Baseline W1 / W2 | Candidate W1 / W2 |
| --- | --- | --- |
| Head MemAvailable before the run, MiB | 3,703 / 3,130 | 3,085 / 2,820 |
| Lowest head MemAvailable in the run, MiB | **1,848** / **1,684** | **1,621** / **1,584** |
| Fall from the start of the run, MiB | 1,855 / 1,446 | 1,464 / 1,236 |
| Lowest worker MemAvailable in the run, MiB | 2,037 / 2,391 | 2,660 / 2,371 |
| Lowest head MemAvailable in the whole step before the SIGTERM test, MiB | 1,848 / 1,548 | 1,621 / 1,572 |

- Lowest head value: the candidate is 227 MiB lower in window 1 and 100 MiB lower in window 2, so the pass bar "not worse" is not met as written.
- In both windows the candidate started the run with less free memory (618 and 310 MiB less), and its fall from that level is smaller. The free memory after a start varies by about 300 to 600 MiB from start to start on both images (candidate after start: 3,181 and 3,471 MiB; baseline: 3,740 and 3,152 MiB).
- A real cost exists: the candidate holds about 0.14 to 0.2 GiB more on the head. The boot log shows `reserved +644 MiB` for the `--parallel` warm-up (baseline +508 MiB) and an allocator hold of 103.1 GiB (baseline 102.9 GiB). Probable source: the second draft graph of PR #25 (coupled drafts).
- The baseline itself spans 1,450 to 1,848 MiB across windows (1,654 and 1,450 MiB in the 1.1.0 gate), more than the candidate gap. The candidate lows (1,584 and 1,621 MiB) stay 0.58 GiB over the 1 GiB earlyoom limit and 0.3 GiB over the 1.25 GiB stop limit.
- The SIGTERM test of window 1 shows a head low of 1,423 MiB at 19:50:08. That is the 100K-token prefill of the test itself (the baseline did not run it). Do not compare it.

## Long run details

- Token hashes, all four runs: `d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6` (the same as the 1.1.0 gate). 0 errors. 256 to 262 s.
- Decode per stream (20K, 50K, 100K, 200K): baseline 19.6, 30.0, 41.5, 115.3 (W1) and 19.5, 30.3, 41.6, 115.8 (W2); candidate 20.5, 31.6, 42.2, 115.0 (W1) and 20.5, 31.3, 42.1, 115.6 (W2). TTFT 9.1 s (candidate) against 9.1 to 10.4 s (baseline) at 20K, and 242 to 244 s against 245 to 248 s at 200K.

## SIGTERM details

- Window 1: a 100K-token request was sent, SIGTERM came 14 s later. Rank 0 exited with code 0 in 3.2 s. The client saw a closed connection. The send time was not recorded, so I treat window 1 as probable evidence only.
- Window 2: the script waited for the "request sent" marker, waited 10 s, checked that the client still ran (it did: 11 s into a prefill of about 46 s), then sent SIGTERM. Rank 0 exited with code 0 in 2.6 s. `terminate called` appears 0 times in the rank 0 log and 0 times in the rank 1 log.
- Rank 1 (worker) then ends with exit code 1 and the line `RuntimeError: rank 0's step link closed`. This is the expected result of the head leaving, not an abort.
- No baseline SIGTERM control was run.

## Files

- `data/`: all logs and results of the gate. Window 1 files have the prefix `b-` (baseline) and `c-` (candidate); window 2 files have `e-` (candidate) and `d-` (baseline). `gate111-run.log` and `gate112-run.log` are the run logs of the two windows. `mem-*-full-head.log` and `mem-*-full-worker.log` hold `MemAvailable` in kB each second.
- `scripts/`: the launcher (`tfv41-g.sh`), the window script (`gate112.sh`, with the order switch of window 2), the lock wrapper (`run_all112.sh`), `live_checks.py`, `prefill_req.py` and `measure_bd.py` (a copy of the cold-prefill kit with only the B and D prompts).
- Suite: recipe main `4f351af` (PR #38 included), sent straight to the engine (the router URL set to the engine URL, a dummy key; the two router tests then run against the engine, as in the 1.1.0 gate). The grammar and bench scripts are unchanged copies of recipe main.
- Host names and home paths are removed from these files.
