# Runs of 21 October 2026 (UTC): release gate of engine `dsv41-1.1.0`

Same pair of DGX Spark as `runs/2026-10-18/`. The gate ran in a lock window from 02:17 to 02:49 UTC (the node clock). Each step started a fresh engine, then ran the same checks in the same order, with the whole suite sent straight to the engine (`v41_suite.py`):

1. start, then a structured-output smoke test;
2. `v41_suite.py` (32 tests);
3. `v41_grammar_check.py 2` (14 checks);
4. `v41_bench.py` (1 to 4 streams, prefix cache);
5. a long run: 4 streams at the same time with prompts of 20K, 50K, 100K and 200K tokens, 1,500 new tokens each, with `MemAvailable` read each second on both nodes.

Settings, equal in both steps (production settings of the pair): `CONTEXT=1048576`, `PARALLEL=4`, `SPILL_GIB=64` (a fresh spill folder in each step), decode graphs on, no restart inside a step. The gate report does not list the other settings. The data are one pass each.

## Release gate of engine 1.1.0

| Step | Engine image | Engine source |
|---|---|---|
| Baseline (1.0.x production image) | `tensorfold-dsv41:3c45bdaeab7d` | the 1.0.0 code, `07aa02585f64`, plus the prompt indexer top-k change (fork PR #7) |
| 1.1.0 | `tensorfold-dsv41:fc9200a44463` | the same git tree as `730d41abd8f4`, the commit of release [dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) that the recipe now pins |

The 1.1.0 step ran the image `fc9200a44463`, the head of fork PR #24. Release commit `730d41abd8f4` is the merge of that pull request and has an identical git tree, so the engine code is the same. The tool-call check for `TF_DROP_UNOFFERED_TOOL_CALLS=1` ran in an earlier window on the candidate `e2b282c50724` (see section 2 of the gate report).

| Check | Baseline `3c45bdaeab7d` | 1.1.0 `fc9200a44463` |
|---|---|---|
| Suite (`v41_suite.py`) | 32/32 | 32/32 |
| Grammar check, `--parallel` 2 | 14/14 | 14/14 |
| Decode, 1 stream, prose / code | 52.7 / 100.4 tok/s | 51.9 / 99.7 tok/s |
| Decode, 4 streams, prose, per stream / aggregate | 25.3 / 96.1 tok/s | 26.8 / 99.3 tok/s (+5.9% / +3.3%) |
| Decode, 4 streams, code, per stream / aggregate | 37.1 / 141.5 tok/s | 39.7 / 145.7 tok/s (+7.0% / +3.0%) |
| Prefill, 32K prompt, cold (no prefix-cache hit), random text | not measured with this method | 1,963 tok/s, TTFT 16.32 s (median of 3) |
| Prefix cache, 24K prompt, cold / warm TTFT | 11.43 s / 1.10 s | 11.18 s / 1.05 s |
| Long run, errors | 0 | 0 |
| Long run, lowest head `MemAvailable` | 1,654 MiB | 1,450 MiB |
| Long run, lowest worker `MemAvailable` | 2,146 MiB | 2,507 MiB |
| Late graph captures while serving | not counted | 0 |
| earlyoom events | none | none |

- The token hashes of the 4 long prompts are the same in both steps (`d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6`).
- The 1-stream decode difference is inside the run-to-run noise. The 4-stream gain is mostly from fork PR #20 (drafts of all streams ranked in one round). The image also holds fork PRs #8 and #24, so the gain is not from #20 alone.
- The long runs started with different head `MemAvailable` (2,113 MiB for the baseline and 3,190 MiB for 1.1.0), so the two lowest values are not strictly comparable. The head margin against the 1 GiB earlyoom limit is small in both steps. A first window on the candidate `e2b282c50724` (no admission fix yet) had one long run in which earlyoom stopped the head engine; three repeats and both steps of this window did not. The gate report gives the details. The admission fix (fork PR #24) was added after that run.
- The prefill and decode rows are single passes. They are release-gate checks, not an A/B with a median of 3 runs.
- **Correction (2026-10-21).** The first version of the 32K prefill row showed 2,717 tok/s (baseline, TTFT 11.78 s) and 2,751 tok/s (1.1.0, TTFT 11.63 s). Those numbers came from a prefix-cache hit of 6,144 tokens (3 chunks of 2,048): `v41_suite.py` sent an 8K prompt with a repeated sentence just before the 32K prompt with the same sentence, and it did not log `cached_tokens`. The old prefill comparison is withdrawn. Counted on the 25,857 tokens that the server prefilled, the 1.1.0 value was 2,218 tok/s. A cold 32K prompt on 1.1.0 (median of 3, `cached_tokens` 0, production settings) gives: repeated sentence 2,228 tok/s (TTFT 14.37 s); random natural text 1,963 tok/s (TTFT 16.32 s, range 1,961-1,969); random token ids 2,004 tok/s; kit_bench method (random words from a list of about 150 words, 32,768 tokens) 2,235 tok/s (TTFT 14.66 s). A small-vocabulary prompt prefills about 11-14% faster than varied text. The baseline `3c45bdaeab7d` was not measured with a cold prompt, so no 1.1.0 against baseline prefill difference is claimed. The data are in the section "Cold 32K prefill check of 1.1.0" below and in `prefill-32k/`. The cold side of the 24K prefix-cache row is not affected: `data/f-bench-fc9200a44463.log` and `data/b-bench-3c45bdaeab7d.log` show `cached_tokens` 0 for it. The raw files in `gate-1.1.0/` are unchanged: the suite logs, the gate report and the suite result files still show the old values, and they must be read with this correction.

## Cold 32K prefill check of 1.1.0 (correction data)

Same pair, image `tensorfold-dsv41:fc9200a44463`, one fresh engine start with production settings (graphs on, `SPILL_GIB=64`, `--parallel 4`, 1M context, prefill chunk 2048, thinking off). Every request: `max_tokens` 1, temperature 0, streamed, from loopback. TTFT is the time to the first streamed piece; prefill tok/s is `prompt_tokens` / TTFT. Every measured prompt starts with a unique `Run <32 hex characters>.` line, and `cached_tokens` was 0 for all 12 of them. 3 rounds, prompt types interleaved. Raw rows: `prefill-32k/results.jsonl`.

| Prompt (32K) | `prompt_tokens` | TTFT median [range], s | Prefill median [range], tok/s |
|---|---|---|---|
| Repeated sentence (the text of the old suite, salted) | 32,023-32,025 | 14.37 [14.32-15.39] | 2,228 [2,080-2,236] |
| Random natural text (shuffled sentences of the Python language reference and library docstrings, no repeats) | 32,029-32,030 | 16.32 [16.27-16.33] | 1,963 [1,961-1,969] |
| Random token ids (decoded) | 32,026-32,029 | 15.99 [15.76-16.21] | 2,004 [1,975-2,033] |
| kit_bench method (random words from a list of about 150 words, 32,768 tokens) | 32,768 | 14.66 [14.61-14.72] | 2,235 [2,227-2,243] |

The cache case, on a fresh engine, in the old suite order (exact text, no salt):

| Request | `prompt_tokens` | `cached_tokens` | TTFT, s | Prefill, tok/s | Without cached tokens, tok/s |
|---|---|---|---|---|---|
| 8K repeated prompt (first request after the start) | 8,001 | 0 | 4.68 | 1,711 | 1,711 |
| 32K repeated prompt (right after the 8K prompt) | 32,001 | 6,144 | 11.66 | 2,745 | 2,218 |

- The prefix cache keeps whole chunks of 2,048 tokens: the 32K request reused 3 chunks (19.2% of the prompt). The 2,751 tok/s of the gate is this case (within 0.2% of 2,745).
- Small-vocabulary prompts (the repeated sentence and the 150-word list) prefill about 11-14% faster than varied text, with no cache hit in either. The cause is not tested.
- The run stopped after the third kit_bench request because the head `MemAvailable` fell to 1,234 MiB, which is under the 1,280 MiB stop rule of that window. No earlyoom event and no crash. The 8K rounds were not run.
- Not tested: the baseline `3c45bdaeab7d` with a cold prompt, and a cold 8K row.

Files:

| File | Content |
|---|---|
| `prefill-32k/results.jsonl` | the 16 requests of the cold 32K prefill check (one JSON row per request, with `prompt_tokens`, `cached_tokens`, `ttft_s`) |
| `prefill-32k/measure.log` | the console output of that check, including the stop of the run by the memory rule after the third kit_bench request |
| `prefill-32k/measure.py`, `prefill-32k/build_prompts.py` | the client and the prompt builder of that check (the builder needs the `tokenizer.json` of the model; the built prompts are not saved) |
| `gate-1.1.0/gate-report.md` | the gate report (host names, home paths and router details removed) |
| `gate-1.1.0/data/b-*` | baseline step: suite, grammar check, bench, long run, lowest memory, earlyoom check |
| `gate-1.1.0/data/f-*` | 1.1.0 step: the same files |
| `gate-1.1.0/data/idle-b.log`, `idle-f.log` | idle `MemAvailable` before the start, after the start and before the long run |
| `gate-1.1.0/data/mem-b-full-*`, `mem-f-full-*` | `MemAvailable` in kB each second, head and worker, over the whole step |

## Release gate of engine 1.1.1

Same pair of DGX Spark. Two lock windows on 21 October 2026 (UTC 19:13 to 19:50 and 21:42 to 22:16). Window 2 repeated window 1 with the step order swapped (candidate first), to check the head memory result. Each step started a fresh engine with a new prompt-spill folder, then ran the same checks in the same order. Settings, equal in all steps (production settings): `CONTEXT=1048576`, `PARALLEL=4`, `SPILL_GIB=64`, decode graphs on.

1. start, then a structured-output smoke test;
2. `v41_suite.py` (32 tests, recipe main with the cache-safe prefill rows), sent straight to the engine;
3. `v41_grammar_check.py 2` (14 checks);
4. `v41_bench.py` (1 to 4 streams, prefix cache);
5. cold 32K prefill: 3 interleaved runs each of random natural text (B) and kit_bench words (D), `cached_tokens` 0;
6. live checks of the new fixes (`live_checks.py`, 22 checks);
7. a long run: 4 streams at the same time with prompts of 20K, 50K, 100K and 200K tokens, 1,500 new tokens each, with `MemAvailable` read each second on both nodes;
8. (candidate only) SIGTERM to rank 0 during a 100K-token prefill.

| Step | Engine image | Engine source |
|---|---|---|
| Baseline (release 1.1.0) | `tensorfold-dsv41:fc9200a44463` | the same git tree as `730d41abd8f4`, the commit of release [dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) |
| 1.1.1 | `tensorfold-dsv41:cb4ceaff2b8c` | the commit of release [dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1) that the recipe now pins |

Each cell shows window 1 / window 2.

| Check | Baseline 1.1.0 | 1.1.1 |
|---|---|---|
| Suite (`v41_suite.py`) | 32/32, 32/32 | 32/32, 32/32 |
| Grammar check | 14/14, 14/14 | 14/14, 14/14 |
| Decode, 1 stream, prose / code | 53.4 / 101.4 and 53.4 / 101.3 tok/s | 54.8 / 99.9 and 54.7 / 99.4 tok/s |
| Decode, 4 streams, prose, per stream / aggregate | 27.3 / 101.1 and 27.1 / 100.3 tok/s | 27.2 / 101.8 and 27.0 / 101.2 tok/s |
| Decode, 4 streams, code, per stream / aggregate | 40.3 / 147.7 and 40.1 / 147.3 tok/s | 40.2 / 148.9 and 40.0 / 148.0 tok/s |
| Prefill, 32K, cold, random natural text, median of 3 | 1,862 and 1,950 tok/s | 1,966 and 1,959 tok/s |
| Prefill, 32,768 tokens, cold, kit_bench words, median of 3 | 2,121 and 2,228 tok/s | 2,250 and 2,234 tok/s |
| Prefix cache, 24K prompt, cold / warm TTFT | 11.37 s / 1.11 s and 11.21 s / 1.31 s | 11.14 s / 1.07 s and 11.21 s / 1.17 s |
| Live checks of the new fixes | 5/22 and 5/22 | 22/22 and 22/22 |
| Long run, errors | 0 and 0 | 0 and 0 |
| Long run, lowest head `MemAvailable` | 1,848 and 1,684 MiB | 1,621 and 1,584 MiB |
| Long run, lowest worker `MemAvailable` | 2,037 and 2,391 MiB | 2,660 and 2,371 MiB |
| earlyoom events, late graph captures | none, 0 | none, 0 |

- The token hashes of the 4 long prompts are the same in all four runs (`d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6`).
- Decode and prefill differences are inside the noise (about 2%). Every prefill value is cold: `cached_tokens` was 0 for all of them.
- **Head memory.** The lowest head `MemAvailable` of 1.1.1 is 227 MiB (window 1) and 100 MiB (window 2) below 1.1.0. The long runs started with less free memory on 1.1.1 (3,085 and 2,820 MiB, against 3,703 and 3,130 MiB), and the fall from that level is smaller. The image holds about 0.14 GiB more on the head (`reserved +644 MiB` against `+508 MiB` in the `--parallel` warm-up; probable source: the second draft graph of fork PR #25). The low point stays 0.58 GiB over the 1 GiB earlyoom limit. The 1.1.0 baseline alone spans 1,450 to 1,848 MiB across windows. The release notes list this as a known issue.
- RDMA: with `NCCL_IB_GID_INDEX` and `TF_RDMA_GID_INDEX` unset, the 1.1.1 engine detects index 3 on both ports of both nodes and uses the RDMA gather (`data/e-gid.log`). 1.1.0 ran with the index pinned to 3.
- SIGTERM to rank 0, 11 s into a 100K-token prefill (window 2): exit code 0 after 2.6 s, no `terminate called` in either rank log (`data/e-sigterm-*.log`; the window 1 SIGTERM run is in `data/gate111-run.log` and in the stop-time rank logs `data/r0-*` and `data/r1-*`).
- A first try of window 2 was stopped after its candidate step, because its prompt-spill folder came from window 1 and gave prefix-cache hits. Its logs are not used and not kept.

Files:

| File | Content |
|---|---|
| `gate-1.1.1/gate-report.md` | the gate report (host names and home paths removed) |
| `gate-1.1.1/data/b-*`, `c-*` | window 1: baseline and 1.1.1 steps (suite, grammar check, bench, cold prefill, live checks, long run, lowest memory, earlyoom check, rank logs) |
| `gate-1.1.1/data/d-*`, `e-*` | window 2: baseline and 1.1.1 steps, the same files |
| `gate-1.1.1/data/mem-*-full-*` | `MemAvailable` in kB each second, head and worker, over the whole step |
| `gate-1.1.1/scripts/` | the launcher, the window scripts and the new test scripts |

## Watermark A/B: `vm.watermark_scale_factor` 50 against 25

Same pair, image `tensorfold-dsv41:cb4ceaff2b8c` (1.1.1 candidate), production settings (graphs on, `SPILL_GIB=64`, `--parallel 4`, 1M context, prefill chunk 2048). Two windows (20:23 to 21:15 and 22:16 to 22:43 UTC, node clock). Each run set only `vm.min_free_kbytes` and `vm.watermark_scale_factor` at runtime on both nodes, started a fresh engine, ran the stall test twice (248 prefill chunks of 2,048 tokens, cold random prompts) and the long 4-stream run once (20K, 50K, 100K and 200K tokens). `MemAvailable`, `/proc/vmstat` and the CPU time of kswapd and kcompactd were sampled each second on both nodes. A watchdog stopped the test containers under 1.25 GiB of head `MemAvailable`.

| Setting | `vm.min_free_kbytes` | `vm.watermark_scale_factor` | Runs | Lowest head `MemAvailable`, long run (mean) | Gain against B | Slow chunks (over 1.3 s) per 248 | Max chunk |
|---|---|---|---|---|---|---|---|
| B (the value before this change) | 1 GiB | 50 | 2 | 1,516 MiB | 0 | 4 and 0 | 2.04 s |
| V1 (the new value) | 1 GiB | 25 | 2 | 1,994 MiB | +477 MiB (formula +934 MiB) | 1 and 0 | 1.31 s |
| V2 | 512 MiB | 50 | 2 | 1,982 MiB (V2r only) | +466 MiB | 0 of 151 (stopped) and 1 | 1.72 s |
| V3 | 512 MiB | 25 | 1 | 3,114 MiB | +1,598 MiB | 4 | 2.54 s |

- V1 is not worse than B on stalls. The median chunk is 0.94 s in all runs, and B itself has 0 to 4 slow chunks. This run does not show that V1 is faster. It has 2 runs for each side, under the 3 of the A/B rule in CONTRIBUTING.md, so it makes no speed claim.
- The gain is a lower kernel reserve, not new free memory. The engine holds the same memory in all runs.
- V3 is not safe: on the head, 4,195 compaction stalls in the stall test (B: 2 and 0), 25,437 swap-in pages and 24,018 major faults in the long run (B: 72 and 109).
- V2 is irregular: its first run fell to 1,192 MiB on the head and the watchdog stopped it; its repeat held 1,982 MiB.
- 0 request errors and no earlyoom line in any run. The long-run token hashes equal those of the 1.1.0 gate.
- `host_tuning.sh` and `preflight.sh` now use 25. A node that ran the old script fails the preflight memory check until it runs `scripts/host_tuning.sh` again.

Details, per-run tables, limits and the file list: [watermark-ab/RESULT.md](watermark-ab/RESULT.md). The tables come from `watermark-ab/summarize.py` (`watermark-ab/summary-table.txt`); the raw files are in `watermark-ab/<run>/` (runs `B`, `B2`, `V1`, `V1r`, `V2`, `V2r`, `V3`). Host names and home paths are replaced in all files.
