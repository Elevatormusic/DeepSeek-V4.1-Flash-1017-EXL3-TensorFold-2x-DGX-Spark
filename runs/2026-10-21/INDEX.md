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
- **Correction (2026-10-21).** The first version of the 32K prefill row showed 2,717 tok/s (baseline, TTFT 11.78 s) and 2,751 tok/s (1.1.0, TTFT 11.63 s). Those numbers came from a prefix-cache hit of 6,144 tokens (3 chunks of 2,048): `v41_suite.py` sent an 8K prompt with a repeated sentence just before the 32K prompt with the same sentence, and it did not log `cached_tokens`. The old prefill comparison is withdrawn. Counted on the 25,857 tokens that the server prefilled, the 1.1.0 value was 2,218 tok/s. A cold 32K prompt on 1.1.0 (median of 3, `cached_tokens` 0, production settings) gives: repeated sentence 2,228 tok/s (TTFT 14.37 s); random natural text 1,963 tok/s (TTFT 16.32 s, range 1,961-1,969); random token ids 2,004 tok/s; kit_bench method (random words from a list of about 150 words, 32,768 tokens) 2,235 tok/s (TTFT 14.66 s). A small-vocabulary prompt prefills about 11-14% faster than varied text. The baseline `3c45bdaeab7d` was not measured with a cold prompt, so no 1.1.0 against baseline prefill difference is claimed. The cold side of the 24K prefix-cache row is not affected: `data/f-bench-fc9200a44463.log` and `data/b-bench-3c45bdaeab7d.log` show `cached_tokens` 0 for it. The raw files in `gate-1.1.0/` are unchanged: the suite logs, the gate report and the suite result files still show the old values, and they must be read with this correction.

Files:

| File | Content |
|---|---|
| `gate-1.1.0/gate-report.md` | the gate report (host names, home paths and router details removed) |
| `gate-1.1.0/data/b-*` | baseline step: suite, grammar check, bench, long run, lowest memory, earlyoom check |
| `gate-1.1.0/data/f-*` | 1.1.0 step: the same files |
| `gate-1.1.0/data/idle-b.log`, `idle-f.log` | idle `MemAvailable` before the start, after the start and before the long run |
| `gate-1.1.0/data/mem-b-full-*`, `mem-f-full-*` | `MemAvailable` in kB each second, head and worker, over the whole step |
