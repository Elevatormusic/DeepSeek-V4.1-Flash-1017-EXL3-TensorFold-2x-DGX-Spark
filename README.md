# DeepSeek V4.1 Flash on two DGX Spark

**Run DeepSeek's V4.1 Flash model on two NVIDIA DGX Spark computers, twice as fast as the standard setup, and ready for AI agents.**

Also on Hugging Face: [Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster](https://huggingface.co/Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster).

Measured on 17 October 2026 on one pair of DGX Spark. Every number on this page comes from a saved run in [`runs/`](runs/2026-10-17/INDEX.md).

---

## At a glance

| | Standard setup (vLLM) | This recipe | |
|---|---:|---:|---|
| Writing speed, text | 27 tokens/s | **57 tokens/s** | **2.1× faster** |
| Writing speed, code | 40 tokens/s | **92 tokens/s** | **2.3× faster** |
| Find one fact in a 64,000-token document | 80 s | **36 s** | **2.2× faster** |

A *token* is a piece of a word. 100 tokens are about 75 English words.

Both setups ran on the same two computers, with the same model files and the same prompts, on the same day.

---

## What you get

- **Twice the speed.** Replies come about twice as fast as with the standard vLLM setup.
- **Ready for AI agents.** The model can call tools when you tell it to, and it can reply in an exact format, such as JSON that matches your schema.
- **Four conversations at once.** Up to 4 requests share the two computers. Together they reach 140 tokens/s on code.
- **It remembers long documents.** Ask a second question about the same long document and the reply starts in about 1 second instead of 13. After a restart, it starts in about 3 seconds.
- **It keeps running.** One bad request ends only that request, not the whole server.
- **A very long memory.** The model can read up to 1 million tokens in one conversation.

---

## Before you start

You need:

- **Two NVIDIA DGX Spark**, joined by one QSFP cable between their ConnectX-7 network ports. NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide shows how to cable them, set fixed addresses, and set up ssh.
- **About 500 GB of free disk on each computer.** The model files use 385 GB. The first start uses 110 GB more.
- **About 100 GB of free memory on each computer.** Close other AI programs first.
- **A network setup** with fixed addresses on the cable link and ssh from the first computer to the second. Do NVIDIA's guide first, then read [our network notes](docs/NETWORK.md).
- **Software on the head:** Docker with GPU support (DGX OS has it), Python 3, `curl`, `rsync`, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`). The worker needs Docker and `rsync`.
- **Time.** Most of the first setup is the 400 GB download. It takes a few hours on a fast internet link.

In this guide, the *head* is the computer where you type the commands. The *worker* is the second computer.

---

## Set up

Run each command on the head, in this folder.

**1. Make your settings file.** Then open `.env` and type your addresses and folder paths.

```bash
cp .env.example .env
```

**2. Download the model.** This step gets 400 GB, checks every file, and copies it to the worker.

```bash
scripts/download.sh
```

**3. Prepare the chat template.** This step gets the template and applies a small fix.

```bash
scripts/patch_template.py --worker
```

**4. Build the engine.** This step builds the server software and copies it to the worker.

```bash
docker/build.sh
```

**5. Start the server.** It checks your setup first. The first start takes about 15 minutes. Later starts take 2 to 4 minutes.

```bash
./start.sh
```

**6. Check that it works.** This sends five real requests and shows the result of each.

```bash
./verify.sh
```

Your server is ready at `http://127.0.0.1:8888/v1`. It works with any app that supports the OpenAI API. The model name is `DeepSeek-V4.1-Flash-1017`.

---

## Connect your app

Use these three settings in any app that supports a custom OpenAI endpoint:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8888/v1` |
| Model | `DeepSeek-V4.1-Flash-1017` |
| API key | any text (or your key, if you set one) |

Thinking is off by default. Turn it on per request with `"reasoning_effort": "high"`.

[The connection guide](docs/CONNECT.md) covers thinking levels, tool calls, exact formats, and use from other computers.

---

## Everyday use

| To do this | Run this |
|---|---|
| See if the server is running | `./status.sh` |
| Stop the server | `./stop.sh` |
| Run all the tests | `tests/run_all.sh` |
| Install a new engine version safely | `scripts/deploy_gate.sh <new> <old>` |

If something goes wrong, read [Operations](docs/OPERATIONS.md). It covers restarts, logs, and what to avoid on DGX Spark.

---

## What I changed to make it work

The fast engine for this model was written by [bertholomus](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10). It was fast, but agent apps could not use it fully. I added what they need, with no loss of speed.

1. **Tool calls on demand.** Apps can now tell the model that it *must* call a tool, or call one specific tool. Agent apps use this to plan their steps.
2. **Prompts exactly as DeepSeek writes them.** Every prompt now matches DeepSeek's own reference, byte for byte.
3. **Replies in an exact format.** The model can now follow a JSON schema, a pattern, or a list of choices. This works while it uses its speed tricks, and for 4 requests at once.
4. **One error no longer stops the server.** Before, one failed request stopped all requests until a manual restart. Now it ends only that request.

[Read the full story](docs/WHAT-I-CHANGED.md), with the reason for each change. The engine code is in [my fork](https://github.com/Elevatormusic/TensorFold), in pull requests [#1](https://github.com/Elevatormusic/TensorFold/pull/1), [#2](https://github.com/Elevatormusic/TensorFold/pull/2) and [#3](https://github.com/Elevatormusic/TensorFold/pull/3).

---

## Good to know

- **The first long question after a start is slower.** It takes about 6 seconds more. Later questions do not.
- **Long documents load slower than bertholomus reports.** With 32,000 tokens or more, this setup reads text 30 to 37% slower than their published numbers. The cause is not known yet.
- **The standard setup was tested one request at a time only.** So the "four at once" numbers have no vLLM comparison.
- **Living Weights is not available.** That TensorFold feature works only with a different model on one computer.

[All known limits](docs/KNOWN-LIMITS.md)

---

## Go back to the standard setup

1. Stop this server:

```bash
./stop.sh
```

2. Start the [MiaAI-Lab vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) as its guide tells you. It uses the same model files. Run only one server at a time.

---

## For experts

<details>
<summary><b>Full speed results</b></summary>

#### This recipe vs the vLLM kit (same computers, same day, same prompts)

vLLM: 1 run. This recipe: 6 runs (6 engine builds of the same day, the last is the released one). The table shows the middle value and the range.

| Single request | vLLM kit | This recipe (middle, range) | Faster by |
|---|---:|---:|---:|
| Writing, text | 27.0 tok/s | **56.8** (52.5-57.7) tok/s | **2.1×** |
| Writing, code | 39.9 tok/s | **92.2** (86.0-93.3) tok/s | **2.3×** |
| Writing, text, thinking on | 25.7 tok/s | **54.2** (47.2-58.5) tok/s | **2.1×** |
| Writing, code, thinking on | 34.5 tok/s | **71.6** (59.6-77.8) tok/s | **2.1×** |
| Find a fact in 64K tokens | 80.4 s | **35.9** (34.0-40.0) s | **2.2×** |
| Read a 32K prompt (\*) | 925 tok/s | 2,100 (1,597-2,260) tok/s | 2.3× |
| Find a fact in 200K tokens (\*) | 216.9 s | 109.8 (103.6-126.9) s | 2.0× |
| Read an 8K prompt, first after a start | 789 tok/s | 562 (427-958) tok/s | 0.7× |

- Writing speed counts the tokens the server reports, per second of writing. Text and code are shown apart, because code drafts better with speculative decoding.
- (\*) These two rows can include cache hits, on both engines. The suite's 32K prompt starts with its 8K prompt, and the 200K prompt starts like the 64K one.
- The 8K row is slow because it is the first long prompt after each start. A longer warm-up list does not fix it ([tuning notes](runs/2026-10-17/17_tuning_warmup-spill-drafts/NOTES.md)). The likely cause is cold reads of the Engram tables from disk.
- Both engines use speculative decoding (vLLM: DSpark, 3 drafts; this recipe: DSpark, 5 drafts; 4 drafts measured slower).

#### Same test client as bertholomus

bertholomus's own benchmark client ([`kit_bench.py`](runs/2026-10-17/16_kit-bench-bertholomus_tensorfold-fd2365042f56/)) ran against this server with their settings. Their numbers are from [their README](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10): original V4.1 Flash weights, their computers. This recipe uses the 1017 weights.

| kit_bench row | Their Zig engine v0.6.0 | Their Python engine v0.5.1 | **This recipe** |
|---|---:|---:|---:|
| Writing, set a: code / text / structured | 89.8 / 59.0 / 106.6 | 89.4 / 57.7 / 107.2 | 84.7 / 58.7 / 103.8 |
| Writing, set b: code / text / structured | 103.1 / 64.1 / 143.2 | 101.5 / 63.4 / 143.4 | 95.1 / 57.7 / 140.6 |
| 2 / 4 requests at once, total | 83.1 / 118.9 | 79.8 / 113.0 | 81.5 / 116.7 |
| 4 requests for 90 s, total | 142.0 | 125.3 | 120.0 |
| Read a prompt, no cache: 8K / 32K / 64K / 128K | 2,433 / 2,495 / 2,461 / 2,354 | 2,387 / 2,421 / 2,363 / 2,242 | 2,190 / 1,707 / 1,457 / 1,420 |
| Writing after a 128K prompt | 100.1 | 49.8 | 61.8 (52-73, 3 runs) |

Their Zig engine does not have the agent features above (forced tool calls and exact formats return an error there).

#### Several requests at once (released engine)

| Requests | Text, total | Code, total | Code, each |
|---:|---:|---:|---:|
| 1 | 29.6 tok/s (includes a 7.3 s first-request wait) | 94.1 tok/s | 95.8 tok/s |
| 2 | 73.5 tok/s | 106.6 tok/s | 57.2 tok/s |
| 3 | 82.3 tok/s | 124.3 tok/s | 43.5 tok/s |
| 4 | **94.4 tok/s** | **140.3 tok/s** | 36.8 tok/s |

- **Cache.** A second question about the same 24K-token document: first token after **12.8 s cold, 1.1 s warm** (22,528 of 24,210 tokens from the cache).
- **After a restart** (`SPILL_GIB=64`, on by default): an 18K-token document came back in **2.7 s** instead of 16.8 s.

Method, definitions and every run: [Benchmarks](docs/BENCHMARKS.md).

</details>

<details>
<summary><b>Exact versions</b></summary>

| What | Version |
|---|---|
| Model weights | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `afc7aa0a` (43 files) |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, files 47 and 48 |
| Engine | [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) @ `fd2365042f56` (TensorFold 0.6.6 + bertholomus's V4.1 engine + PRs #1-#3) |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` |
| Chat template | MiaAI kit `files/chat_template.jinja` @ `6f7d1590` + [our patch](runs/2026-10-17/template-patch.diff) |
| vLLM comparison | MiaAI kit @ `6f7d1590`, image `sha256:2f0cf3ad...` |
| Tested on | DGX OS (Ubuntu 24.04.5), driver 580.178.04, CUDA 13.0, Docker 29.6.2 |

Every checksum, file size and computer detail: [PROVENANCE](runs/2026-10-17/PROVENANCE.md).

</details>

<details>
<summary><b>What is in this folder</b></summary>

| Path | What |
|---|---|
| `start.sh`, `stop.sh`, `status.sh`, `verify.sh` | run the server |
| `scripts/` | download, template fix, setup checks, safe engine install (`_raw_*`: the scripts as they ran on the tested pair) |
| `docker/` | the engine build |
| `tests/` | the 32 agent tests, the format check, the benchmark, the prompt tests, the Hermes test ([guide](tests/README.md)) |
| `runs/2026-10-17/` | every log and result behind this page |
| `docs/` | how to connect your app, the full story, benchmarks, network, operations, known limits |

</details>

---

## Credits and licence

This recipe stands on the work of DeepSeek (the model), LibertAIDAI (the compressed weights), turboderp (ExLlamaV3), ashhart and the TensorFold contributors, **bertholomus (the fast V4.1 engine this recipe runs)**, willjriley and Capicua25x (engine fixes), MiaAI-Lab (the vLLM kit and the template) and tonyd2wild (the benchmark method). [Full credits](CREDITS.md)

The scripts, tests and documents here are Apache-2.0 ([LICENSE](LICENSE)). The model, the template and the engine keep their own licences.
