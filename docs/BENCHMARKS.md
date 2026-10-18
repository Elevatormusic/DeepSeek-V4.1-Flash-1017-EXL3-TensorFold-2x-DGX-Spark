# Benchmarks

This page gives the method and every number behind the speed claims in the README.
All numbers come from the files in [`runs/2026-10-17/`](../runs/2026-10-17/). The page states no number that a run did not produce.
A number that the page derives from run numbers (a ratio, a median, a rate) is marked "derived".

## Summary

Both engines ran on the same two DGX Spark, with the same weight files, on the same day (2026-10-17), with the same test suite ([`tests/v41_suite.py`](../tests/v41_suite.py)).

- vLLM baseline: the MiaAI-Lab kit. One run (run 01).
- TensorFold: six runs (02, 03, 04, 05, 09, 11). Each run used a different engine build, not a repeat of one build.

| Single stream | vLLM kit | TensorFold, median | TensorFold, min - max | Speedup, median | Speedup, min - max |
|---|---:|---:|---:|---:|---:|
| Decode, prose (tok/s) | 27.0 | 56.8 | 52.5 - 57.7 | 2.10x | 1.94x - 2.14x |
| Decode, code (tok/s) | 39.9 | 92.2 | 86.0 - 93.3 | 2.31x | 2.16x - 2.34x |
| Decode, prose, thinking on (tok/s) | 25.7 | 54.2 | 47.2 - 58.5 | 2.11x | 1.84x - 2.28x |
| Decode, code, thinking on (tok/s) | 34.5 | 71.6 | 59.6 - 77.8 | 2.08x | 1.73x - 2.26x |
| Prefill, 32K prompt (tok/s) | 925 | 2,100 | 1,597 - 2,260 | 2.27x | 1.73x - 2.44x |
| Prefill, 8K prompt, first long prompt (tok/s) | 789 | 562 | 427 - 958 | 0.71x | 0.54x - 1.21x |
| Needle in 64K, total time (s) | 80.4 | 35.9 | 34.0 - 40.0 | 2.24x | 2.01x - 2.36x |
| Needle in 200K, total time (s) | 216.9 | 109.8 | 103.6 - 126.9 | 1.98x | 1.71x - 2.09x |

How to read the table:

- "Median" and "min - max" are the median and the range of the six TensorFold runs. The medians are derived.
- "Speedup" is the TensorFold value divided by the vLLM value for a rate. It is the vLLM value divided by the TensorFold value for a time. The ratios are derived.
- The speedup range uses the best and the worst TensorFold run against the one vLLM run. The vLLM run has no spread of its own, so the true range is wider.
- The 8K row is slower on TensorFold. See "Prefill" below.

## What was the same and what differed

| Item | vLLM kit (run 01) | TensorFold (runs 02 - 11) |
|---|---|---|
| Hardware | The same two DGX Spark, TP=2 over the ConnectX-7 link | The same |
| Weights | The same model view (EXL3 2.9 bpw, 1017 Engram tables) | The same |
| Suite | `tests/v41_suite.py` | The same file |
| Date | 2026-10-17, about 18:46Z to 18:56Z | 2026-10-17, 20:14Z to 23:03Z |
| Speculative decoding | DSpark, 3 draft tokens (`DSPARK_TOKENS=3`) | DSpark, up to 5 draft tokens (`--mtp-drafts 5`) |
| Concurrency setting | `MAX_NUM_SEQS=1` | `--parallel 4` |
| Longest context set | `MAX_MODEL_LEN=262144` | `--context 1048576` |
| Prompt chunk | `MAX_NUM_BATCHED_TOKENS=1536` | `TF_DS_PREFILL_CHUNK=2048` |
| Chat template | The kit template, unpatched | The kit template; patched from run 04 on (inferred from file times, see `PROVENANCE.md`, section 3) |

The engine settings are the settings that each stack was run with. They are not tuned to be equal.
The suite sets `temperature=1.0` and `top_p=1.0` for the decode tests on both engines.

## Method

### Tools

- `tests/v41_suite.py` made the single-stream numbers. It uses only the Python standard library.
- `tests/v41_bench.py` made the concurrency and prefix-cache numbers. It uses only the Python standard library.
- No other benchmark tool was used.
- After the runs, the test files got the settings `V41_BASE_URL` and `V41_MODEL` and a non-zero exit code on failure. The requests did not change.

### How each number is counted

All token counts come from the `usage` field of the server reply. The suite never counts tokens in the client.

| Number | Definition | Source in the code |
|---|---|---|
| Decode speed (tok/s) | `(completion_tokens - 1) / (T_end - T_first)`. `T_first` is the time of the first streamed chunk that has content, reasoning or a tool call. `T_end` is the time when the stream ends. | `speed_decode`, `chat_stream` in `v41_suite.py` |
| Decode, one table row | The mean of the per-prompt decode speeds. Thinking off: 2 prompts. Thinking on: 1 prompt. | `speed_decode` |
| Prefill speed (tok/s) | `prompt_tokens / TTFT`. TTFT is the time from the request to the first streamed token. | `speed_prefill` |
| Needle time (s) | The wall time of one non-streamed request. It covers the prefill, up to 32 generated tokens and the HTTP round trip. | `needle` |
| Aggregate speed (tok/s) | All completion tokens of all streams divided by the wall time of the batch. The wall time includes the TTFT. | `concurrency` in `v41_bench.py` |
| Per-stream speed (tok/s) | The mean over the streams of `(completion_tokens - 1) / (T_end - T_first)`. | `stream` in `v41_bench.py` |

### Prompts and settings

- **Prose and code are reported apart.** Speculative decoding drafts code better than prose, so one mixed number would hide this.
  - Prose prompts: a 400-word short story, and an explanation of the immune system.
  - Code prompts: a Python LRU cache module with tests, and a TypeScript ISO-8601 duration parser.
- **Thinking off and thinking on are reported apart.** The suite sends `chat_template_kwargs.thinking` as false or true. Thinking on uses `reasoning_effort` "high".
- **Decode tests:** `max_tokens=700`, `temperature=1.0`, `top_p=1.0`, streamed.
- **Prefill tests:** `max_tokens=8`, `temperature=0`, thinking off. The prompt repeats one sentence until it reaches 8,000 or 32,000 tokens. The measured prompts have 8,001 and 32,001 tokens.
- **Needle tests:** `max_tokens=32`, `temperature=0`, thinking off. The prompt repeats one sentence. It hides the text "The secret launch code is ORCHID-7731." at 63 percent of the depth. The prompts have 64,030 and 200,030 tokens. A run passes when the answer contains `ORCHID-7731`. All seven runs passed both needle tests.
- **Concurrency benchmark:** 1, 2, 3 and 4 streams, prose and code, thinking off, `max_tokens=512`, `ignore_eos=true`. So each stream makes exactly 512 tokens.
- **Prefix-cache benchmark:** a document of about 24,200 tokens, then a second question about the same document, then a new document of the same size.

### Warm-up

- The suite sends no warm-up request. Nothing is dropped from the results.
- TensorFold also warms up when it starts (`TF_DS_WARM_LENGTHS`). The TTFT of the decode tests is 0.1 to 0.3 s, so they show no warm-up cost.
- The first long prompt after a start still pays a one-time cost (about 10 s, see [OPERATIONS.md](OPERATIONS.md)). The 8K prefill row is that first long prompt, so it includes this cost.
- The benchmark sends one 8-token request before it measures. The first prose request still has a TTFT of 7.26 s in both benchmark files. The files do not say why.

### Spread

- A single cell can move by 15 percent or more between runs. The decode tests use temperature 1.0, so the generated text and the speculative acceptance differ between runs.
- The thinking-on rows use one prompt per run. They have the widest spread.
- The tables show the spread. Do not read the median alone.

## Results

### The runs

| Run | Engine | Image | Suite result | Suite time |
|---|---|---|---:|---:|
| 01 | vLLM, MiaAI kit `6f7d1590` | `sha256:2f0cf3ad...` | 32/32 | 602 s |
| 02 | TensorFold | `469171515cd3` | 25/32 | 291 s |
| 03 | TensorFold | `4a413d9c732b` | 27/32 | 294 s |
| 04 | TensorFold | `90c2d59c8179` | 28/32 | 278 s |
| 05 | TensorFold | `7f80c3d267da` | 32/32 | 285 s |
| 09 | TensorFold | `d5c9d3cdf4fa` | 32/32 | 324 s |
| 11 | TensorFold | `fd2365042f56` (the image of 17 October; the recipe now builds `3c45bdaeab7d`) | 32/32 | 295 s |

- The failed tests in runs 02, 03 and 04 are tool-choice and structured-output tests, not speed tests. Their speed rows are valid.
- Runs 02, 03 and 04 used engine builds from before the structured-output work. See [`INDEX.md`](../runs/2026-10-17/INDEX.md) for the failed test names.

### Single stream, every run

Values are from the `*_suite-results_*.json` files.

| Metric | vLLM 01 | TF 02 | TF 03 | TF 04 | TF 05 | TF 09 | TF 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode, prose (tok/s) | 27.0 | 56.7 | 52.5 | 57.1 | 56.0 | 57.7 | 56.9 |
| Decode, code (tok/s) | 39.9 | 86.0 | 91.7 | 92.5 | 93.3 | 92.5 | 91.8 |
| Decode, prose, thinking on (tok/s) | 25.7 | 58.5 | 51.5 | 47.2 | 53.6 | 54.8 | 54.7 |
| Decode, code, thinking on (tok/s) | 34.5 | 77.8 | 70.9 | 72.3 | 77.2 | 59.6 | 63.7 |
| Prefill, 8K prompt (tok/s) | 789 | 603 | 551 | 573 | 958 | 427 | 490 |
| Prefill, 32K prompt (tok/s) | 925 | 2,260 | 2,116 | 2,131 | 1,597 | 1,601 | 2,084 |
| Needle in 64K (s) | 80.4 | 38.6 | 40.0 | 35.5 | 34.0 | 36.3 | 35.3 |
| Needle in 200K (s) | 216.9 | 126.9 | 110.6 | 108.9 | 103.6 | 125.2 | 108.5 |

Notes on the table:

- The per-prompt decode values are in the `per_prompt` field of the JSON files. For vLLM the two prose prompts gave 24.7 and 29.3 tok/s. For run 11 they gave 53.4 and 60.4 tok/s.
- The thinking-on code row fell from 77.8 and 77.2 tok/s (runs 02 and 05) to 59.6 and 63.7 tok/s (runs 09 and 11). Each value is one sample. The files do not show if this is noise or a cost of the later commits. This was not tested.
- vLLM has one run, so its spread is unknown.

### Prefill

| Prompt tokens | vLLM TTFT | vLLM speed | TensorFold TTFT, min - max | TensorFold speed, median (derived) | TensorFold speed, min - max |
|---|---:|---:|---:|---:|---:|
| 8,001 | 10.13 s | 789 tok/s | 8.36 - 18.74 s | 562 tok/s | 427 - 958 tok/s |
| 32,001 | 34.6 s | 925 tok/s | 14.16 - 20.04 s | 2,100 tok/s | 1,597 - 2,260 tok/s |

- **The 8K row is the first long prompt after a start.** On TensorFold, its TTFT (8.4 to 18.7 s) is close to the TTFT of the 32K prompt (14.2 to 20.0 s). The one-time warm-up dominates it. On vLLM, the 8K and 32K rates are close (789 and 925 tok/s).
- **The prompts share a start.** The 8K prompt and the 32K prompt repeat the same sentence, so the 32K prompt starts with the text of the 8K prompt. Both engines have a prefix cache. The suite does not record `cached_tokens`. So the files cannot show how much of the 32K prompt came from the cache. Check the `cached=` field in the engine log of each request.
- **The 200K needle prompt shares its first part with the 64K needle prompt**, for the same reason.
- **Prefill rate at 64K and 200K, derived.** The needle tests give prompt tokens divided by total time. For 200,030 tokens: vLLM 922 tok/s, TensorFold 1,576 to 1,931 tok/s (median 1,823). For 64,030 tokens: vLLM 796 tok/s, TensorFold 1,601 to 1,883 tok/s (median 1,784). These rates include up to 32 generated tokens and the HTTP round trip.

### Concurrency (TensorFold only)

The run used `v41_bench.py`: thinking off, 512 tokens per stream. Per-stream speed and TTFT are means over the streams.

Image `fd2365042f56` of 17 October (run 13). The recipe now builds `3c45bdaeab7d`; its runs are in [runs/2026-10-18](../runs/2026-10-18/INDEX.md):

| Streams | Prose per stream | Prose aggregate | Prose TTFT | Code per stream | Code aggregate | Code TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 51.0 tok/s | 29.6 tok/s * | 7.26 s * | 95.8 tok/s | 94.1 tok/s | 0.11 s |
| C2 | 37.7 tok/s | 73.5 tok/s | 0.14 s | 57.2 tok/s | 106.6 tok/s | 0.15 s |
| C3 | 29.7 tok/s | 82.3 tok/s | 0.18 s | 43.5 tok/s | 124.3 tok/s | 0.29 s |
| C4 | 24.9 tok/s | 94.4 tok/s | 0.29 s | 36.8 tok/s | 140.3 tok/s | 0.27 s |

Earlier image `7f80c3d267da` (run 07):

| Streams | Prose per stream | Prose aggregate | Prose TTFT | Code per stream | Code aggregate | Code TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 46.9 tok/s | 28.2 tok/s * | 7.26 s * | 82.9 tok/s | 81.6 tok/s | 0.11 s |
| C2 | 37.7 tok/s | 73.5 tok/s | 0.14 s | 61.8 tok/s | 116.4 tok/s | 0.16 s |
| C3 | 29.9 tok/s | 84.3 tok/s | 0.18 s | 44.3 tok/s | 127.2 tok/s | 0.21 s |
| C4 | 25.1 tok/s | 95.7 tok/s | 0.22 s | 34.6 tok/s | 132.3 tok/s | 0.27 s |

- \* The first prose request has a TTFT of 7.26 s in both files. The aggregate speed counts this time, so the prose C1 aggregate is low. The per-stream speed does not count the TTFT.
- At four streams the per-stream speed is 49 percent (prose) and 38 percent (code) of the one-stream value (run 13, derived). The aggregate speed is 94.4 and 140.3 tok/s.
- The two benchmark runs differ by up to 16 percent in the same cell (code C1 per stream: 82.9 and 95.8 tok/s). Use these tables with that spread in mind.
- **No vLLM concurrency baseline exists.** The vLLM kit ran with `MAX_NUM_SEQS=1`, so it serves one request at a time. This repo did not measure vLLM at 2 to 4 streams.
- The benchmark does not compare the content of a reply under load with the reply of the same request alone.

### Prefix cache (TensorFold only)

| Step | Prompt tokens | Cached tokens | TTFT, run 13 | TTFT, run 07 |
|---|---:|---:|---:|---:|
| Cold, document A, question 1 | 24,209 | 0 | 12.76 s | 13.02 s |
| Warm, document A, question 2 | 24,210 | 22,528 | 1.07 s | 1.87 s |
| Cold, document B (same size) | 24,374 | 0 | 12.98 s | 19.07 s |

- 22,528 tokens are 11 chunks of 2,048 tokens. That is 93 percent of the prompt (derived).
- The warm question is 11.9 times faster than the cold one in run 13, and 7.0 times faster in run 07 (derived, against document A).
- The cold TTFT of document B in run 07 (19.07 s) is 46 percent above document A (13.02 s). The files do not explain this. In run 13 both are near 13 s.
- The vLLM kit was not run through this check.

### KV cache size per token

Not measured. The files do not hold the KV pool size in tokens, so the bytes per token cannot be derived.
The vLLM run set `KV_CACHE_MEMORY_BYTES=2684354560` (2.5 GiB). The number of tokens that pool held was not recorded.
The engine logs are not in this repo.

### Output quality

This repo did not measure perplexity or an accuracy score.
The suite checks that answers are correct on a few fixed tasks (arithmetic, tool arguments, schema validity, the needle).
The model card of the weights (`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`, revision `c178a012`) reports perplexity for the weights: code 1.9094 and prose 6.4520, against 1.8749 and 6.4277 for the unquantized model.
Those numbers are from the card, for the weights. They are not from TensorFold or vLLM output.

## Hardware and software

All values are from [`PROVENANCE.md`](../runs/2026-10-17/PROVENANCE.md).

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, one per node, two nodes |
| Driver, CUDA | 580.178.04, CUDA 13.0 |
| Kernel, OS | 7.0.0-1019-nvidia, Ubuntu 24.04.5 LTS |
| Docker | 29.6.2 |
| CPU, memory | 20 cores (Cortex-X925 and Cortex-A725, aarch64), about 121 GiB unified memory per node |
| Link | QSFP cable, ConnectX-7, 200 Gb/s, MTU 9000, RoCE v2, GID index 3 |
| Weights | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` at `afc7aa0aab98ee735c0f54e0a122019af1ac08f1` (43 shards) |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`, shards 47 and 48 |
| vLLM baseline | MiaAI-Lab kit at `6f7d1590ad49a2b8995188e45d7b9db31e677452`, image `sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce` |
| vLLM settings | TP=2, `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=1536`, `GPU_MEM_UTIL=0.88`, `KV_BLOCK_SIZE=64`, `KV_CACHE_MEMORY_BYTES=2684354560`, `SPEC_METHOD=dspark`, `DSPARK_TOKENS=3`, `PREFIX_CACHE_RETENTION_INTERVAL=4096` |
| TensorFold | Fork [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) at `fd2365042f56` (the image measured on 17 October; the recipe now builds `3c45bdaeab7d`), 0.6.6 base, xgrammar 0.2.8 |
| TensorFold settings | `--tp 2 --context 1048576 --vision --parallel 4 --mtp-drafts 5 --temperature 1.0 --top-p 1.0`, `TF_DS_REPLAY=1`, `TF_DS_REPLAY_FLOOR=1`, `TF_DS_PREFILL_CHUNK=2048` |
| Base image, torch | `nvcr.io/nvidia/pytorch:26.07-py3`, torch `2.13.0a0+9186a08b2c.nv26.7.59513937`, Python 3.12 |

## Limits of these numbers

- One pair of DGX Spark. One day. One vLLM run.
- The six TensorFold runs are six engine builds. They show the spread over builds, not over repeats of one build.
- The GPU clock state and the link bandwidth were not recorded before each run. A slow GPU state can lower a result by a large factor.
- The prefix cache can change the 32K prefill and the 200K needle rows (see "Prefill").
- The thinking-on rows are single samples.
- Decode with temperature 1.0 depends on the generated text. A different prompt gives a different speed.
- More limits: [KNOWN-LIMITS.md](KNOWN-LIMITS.md).

## Repeat the measurement

Run the full test set against your server:

```bash
tests/run_all.sh
```

Add the benchmark:

```bash
BENCH=1 tests/run_all.sh
```

Each run writes its files to `runs/<date>/`. [`tests/README.md`](../tests/README.md) lists the settings.
