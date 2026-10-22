# What I changed to make it work

The starting point was the MiaAI-Lab vLLM kit for the same weights on the same two Sparks (decode 27 tok/s prose, 40 tok/s code). bertholomus's DeepSeek-V4.1 TP2 engine for TensorFold ([their recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)) was about 2x faster, but it was not ready for agents: on the first run it passed only 25 of the 32 serving tests. This page lists every change that took it to 32/32, in order, with the reason for each.

All engine changes are in the public fork [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) as pull requests:
- [#1](https://github.com/Elevatormusic/TensorFold/pull/1): tool_choice and the V4.1 encoding.
- [#2](https://github.com/Elevatormusic/TensorFold/pull/2): structured output.
- [#3](https://github.com/Elevatormusic/TensorFold/pull/3): fault isolation.
- [#4](https://github.com/Elevatormusic/TensorFold/pull/4): faster long prefills (the Engram page drop).
- [#6](https://github.com/Elevatormusic/TensorFold/pull/6): steady prefill with the prefix-cache spill on (spill files written with `O_DIRECT`, past the page cache).

Release 1.0.0 of the engine was built from the commit at the head of #6, `07aa02585f64`, which contains all five and their review fixes. The serving results of 17 October were measured on `fd2365042f56` (the head of #3); #4 and the review fixes do not change decoding.

The image in this repo is now built from the commit of [release dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1), `cb4ceaff2b8c`. It adds more fork pull requests ([section 7](#7-engine-release-110) and [section 8](#8-engine-release-111)).

## Test score at each step

| Step | Change | Suite (32 tests) |
|---|---|---|
| 0 | bertholomus's V4.1 engine, Python line, with willjriley's fixes, as found (branch `deepseek-v41-tp2` @ `469171515cd3`) | 25/32 |
| 1 | DSML tool_choice gate | 27/32 |
| 2 | V4.1 encoding fixes + chat template patch | 28/32 |
| 3 | Structured output on the speculative TP=2 decoder | 32/32 |
| 4 | Live-deploy fixes, Codex review fixes, fault isolation | 32/32 (kept) |

The suite covers:
- speed;
- 12 tool-calling tests (parallel calls, required and named tool_choice, nested arguments, streaming deltas, 4-step agent loops);
- 6 structured-output tests;
- thinking levels;
- 64K and 200K needle-in-a-haystack.

The suite is [`tests/v41_suite.py`](../tests/v41_suite.py), and the run logs are in [`runs/2026-10-17/`](../runs/2026-10-17/).

## 1. Forced tool calls on DeepSeek's DSML format (PR #1)

- **Problem.** `tool_choice: "required"` and `tool_choice: {"function": ...}` returned HTTP 400. TensorFold's call gate forces a tool call by steering the first tokens of the answer, but it only knew single-token openers such as `<tool_call>`. DeepSeek-V4.1 writes a call as `<｜DSML｜ calls>`: a plain `<`, then one special token.
- **Change.** The gate learned a "head" (text that comes before the opener token). It reads the call format from the chat template's own rendering of a probe call. The gate only accepts the spaced V4.1 form, because that is the only form the reply parser turns into `tool_calls`.
- **Result.** Required and named tool_choice pass. These are what agent frameworks use for planning steps.

## 2. Prompts byte-identical to DeepSeek's own encoder (PR #1 + template patch)

- **Problem.** DeepSeek publishes a reference encoder (`encoding/encoding.py` in the model repo). TensorFold's prompts differed from it in three places:
  - A second leading system message (Codex's developer turn) was merged into the first. DeepSeek keeps it as its own `<｜System｜>` turn.
  - `response_format: json_object` was refused. DeepSeek's encoding serves it as a "## Response Format" block in the system turn.
  - A request with tools and no system message rendered the tools block without the empty system turn and the blank line that the reference writes.
- **Change.**
  - Leading system messages stay separate on this model (`merge_leading = False`).
  - json_object becomes the template's response-format input.
  - The chat template from the MiaAI kit gets a small patch (1 hunk, +6/-2 lines) for the tools-without-system case and for json_object without a system message. [`scripts/patch_template.py`](../scripts/patch_template.py) downloads the template at a pinned commit and applies the patch; the template is AGPL-3.0, so this repo does not copy it.
- **Proof.**
  - `tests/v41_differential.py`: 12/12 API-shaped requests render byte-identical to DeepSeek's `encoding.py`.
  - `tests/v41_conformance.py`: 3/3 of DeepSeek's golden fixtures that use API roles.

## 3. Structured output on the speculative, two-node decoder (PR #2)

- **Problem.** The V4.1 engine refused every grammar: json_schema, regex, choice and EBNF. Agents need them. Hermes, for example, titles every session with a strict json_schema, so those calls failed.
- **Why it is hard.** DSpark speculative decoding verifies a chain of up to 5 drafted tokens per step. Also, each of the two GPUs samples its own copy of the gathered logits ("keyed sampling", no broadcast), so both ranks must hold the same grammar state at every step.
- **Change.**
  - Rank 0 packs the request's grammar and sends it to rank 1 with the request, in the serial header and in the `--parallel` admission op. Both ranks compile it from the same tokenizer.
  - Each round, the grammar cuts the drafts it cannot accept before the forward pass. It then masks the verify rows, and both ranks follow the tokens the round keeps.
  - Thinking stays free until `</think>`.
  - The reply stops at the grammar's own stop token, also with `ignore_eos`.
  - A grammar that fails ends only its own stream.
- **Result.** Strict json_schema (thinking on and off), json_object, regex and choice all pass, also 4 at a time under `--parallel` (`tests/v41_grammar_check.py`: 14/14). Single-stream speed is unchanged when no grammar is set.

## 4. What the live deploy found (PR #2, PR #3)

- **Inference-mode bug.** The first grammar build passed every unit test, then broke the whole lane on the first live json_schema request: "Inplace update to inference tensor outside InferenceMode". The model makes its logits under `torch.inference_mode()`, but the `--parallel` rounds run under `torch.no_grad()`. The mask now runs inside inference mode, and a regression test reproduces the exact error.
- **One bad request took the server down.** That bug showed a bigger problem: any exception in a `--parallel` step aborted NCCL, and the server then failed every request until a manual restart. PR #3 changes this:
  - A fault in one stream's own host work (its grammar, its sampling, its reply callback) ends only that stream; the other streams continue.
  - A device fault (CUDA error, out of memory) still breaks the lane, because the CUDA context may be bad.
  - A broken lane now ends the process with exit code 75 after 5 s, so a supervisor can restart it.
  - If only one rank ends a stream, both ranks still fail that step together instead of hanging.
- **Deploy gate.** Every image in this repo went through [`scripts/deploy_gate.sh`](../scripts/deploy_gate.sh), which rolls back on its own if any check fails:
  1. deploy;
  2. a live json_schema request and a plain request;
  3. the 32-test suite;
  4. the `--parallel` grammar check.
- **Review.** The OpenAI Codex review bot found 8 issues across PRs #1-#3 in two rounds, and 2 more in a third round (#2: a stream whose grammar failed could send one unmasked token; #3: ranks that differ only in draft depth could pass the round check). All are fixed, each with a test.

## 5. Launcher settings (start.sh)

| Setting | Value | Why |
|---|---|---|
| `--parallel` | 4 | 4 requests on the lane at once (bertholomus's release gates check that each reply equals its solo run; this repo does not repeat that test) |
| `--mtp-drafts` | 5 | DSpark draft depth; the engine verifies up to 5 drafted tokens per step |
| `--context` | 1048576 | the model's full 1M window; the KV pool fits on 2 x 128 GB |
| `--temperature` / `--top-p` | 1.0 / 1.0 | DeepSeek's sampling recommendation for V4.1 |
| `TF_DS_REPLAY=1` | on | decoder sliding-window bounded replay: the decoder layers run only over the last 128 prompt tokens (DeepSeek's deployment mode); the needle tests pass with it at 64K and 200K |
| `TF_DS_REPLAY_FLOOR=1` | on | a short prompt chunk whose window reaches below the replay cut reads its keys by the gather path, so a reply never depends on the slot's previous request |
| `TF_DS_PREFILL_CHUNK=2048` | 2048 | prompt chunk size (default 512); 4096 measured 6-15% slower (runs/2026-10-18), and the prefix cache keeps prompts at these boundaries |
| `TF_DS_RANK_CACHE` | `/cache/tf-ds-rank` | each rank writes its weight shard to one file on the first start; later starts read it back (~2 min instead of ~15) |
| `TF_DS_RANK_CACHE_READERS=32` | 32 | reader threads for that file (default 8); NVMe Gen5 reads at ~11.7 GB/s with 32 |
| `TF_DS_WARM_LENGTHS` | `1,2,...,17,33,131,514,1024,2113` | prompt lengths for the warm-up, so no request pays a Triton kernel build (prompts of 3-16 tokens each build their own kernel) |
| `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` | under `/cache` | compiled kernels survive a restart: short prompts take 0.1 s on first use instead of 2.6-6.1 s |
| `TF_DS_SPILL_GIB=64` | 64 | kept prompts are also written to disk, so a prompt survives a restart (PR #6 writes them with `O_DIRECT`, so the writes do not start page compaction) |
| `TF_DS_ENGRAM_DONTNEED=1` | on | PR #4: drops the Engram tables' cached pages after each prompt chunk, so long prefills do not push the node into reclaim and compaction stalls |
| `TF_DROP_UNOFFERED_TOOL_CALLS` | 1 (`DROP_UNOFFERED_TOOL_CALLS` in `.env`) | release 1.1.0 (fork PR #15): a call to a tool that the request did not offer is dropped (or kept as text), also while the reply streams. The engine has this off; the recipe turns it on. `0` in `.env` turns it off. See [OPERATIONS.md](OPERATIONS.md) |
| `TF_DS_MEM_FLOOR_GIB=1.5` | 1.5 | after the warm-up, the engine grows only while 1.5 GiB of host memory stays free. The default (2.0) left 1.2-1.4 GiB of headroom on some starts, and the second chunk of an 8K prompt then failed with CUDA out of memory. 1.5 stays above earlyoom's 1 GiB limit |
| `--oom-score-adj 1000` | docker flag | if a node runs out of memory, the kernel and earlyoom stop the engine first, not sshd or the systemd manager |
| `--cgroup-parent tensorfold.slice` | docker flag | the slice from `scripts/host_tuning.sh`, where the engine cannot swap |
| `TF_DS_ENGRAM=/engram` | path | DeepSeek's 1017 Engram tables (shards 47/48), read from NVMe at random offsets |
| `TF_RDMA_*`, `NCCL_IB_*` | CX7 ports | TP=2 runs on both ConnectX-7 ports with RoCE v2 (see [NETWORK.md](NETWORK.md)) |

## 6. Hermes release gate

- **What it is.** Before publishing, the image had to pass Hermes Agent's own client code against the live server, through a router. The harness is [`tests/hermes_thinking_modes.py`](../tests/hermes_thinking_modes.py).
- **What it covers.**
  - 8 thinking levels;
  - streaming and non-streaming;
  - a math task and a multi-call tool task;
  - 3 session titles with strict json_schema.
- **Result.** 35/35.

## 7. Engine release 1.1.0

The engine is now [release dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) (`730d41abd8f4`). The release notes give every change, the results and the known issues. This list gives only what changes for a user of this recipe:

- **No decode graph is captured while serving** (fork PR #19). Users with long parallel sessions saw `cudaErrorStreamCaptureInvalidated` or a CUDA allocation failure during graph capture. The warm-up now captures every graph that the window can use.
- **Lower host-memory peak on the head** (fork PRs #21 and #24), for long prompts and images. Recipe issue [#21](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/21) stays open: the head still has a small margin under 4 long streams ([OPERATIONS.md](OPERATIONS.md)).
- **Draft ranking across streams** (fork PR #20): the engine ranks the drafts of all streams in one round. In the gate run, decode with 4 streams is 3.0 to 7.0% faster than on the previous image. One stream is not faster. See [runs/2026-10-21](../runs/2026-10-21/INDEX.md).
- **Opt-in `TF_DROP_UNOFFERED_TOOL_CALLS`** (fork PR #15), on in this recipe (see the table in section 5).
- Smaller fixes: the RDMA path MTU follows the port (PR #9), overlapping image requests no longer crash the lane (PR #13), and `--vision-max-images` applies to V4.1 image input (PR #14). The prompt indexer top-k change (PR #18) is now in the pinned branch.
- Not used by the recipe: dense BF16 overrides (PR #8).

The settings of the new engine switches are in [OPERATIONS.md](OPERATIONS.md#engine-switches-of-release-110).

## 8. Engine release 1.1.1

The engine is now [release dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1) (`cb4ceaff2b8c`). The release notes give every change, the results and the known issues. This list gives only what changes for a user of this recipe:

- **Host setup of NCCL and RDMA** (fork PR #22). NCCL cuMem is off by default. This is a probable cause of the `NV_ERR_NO_MEMORY` reports. The RDMA GID index is read from sysfs when `GID_INDEX` is `-1`. The page cache of the weight files is released after the weight load. `.env.example` still sets `GID_INDEX=3`, the tested value.
- **Tool calls are contained** (fork PRs #23 and #35). A DSML call counts only when its block is complete. A reply that stops inside a DSML block (`max_tokens`) no longer shows the raw block in `content`, and the stream agrees with the whole reply.
- **Special-token text stays text** (fork PR #28, security). `</think>`, `<｜User｜>` and the other special-token spellings in message text are plain text. A client cannot end the thinking or forge a turn with them.
- **Clean stop after SIGTERM** (fork PR #34). Rank 0 exits with code 0 and without `terminate called` when earlyoom or `docker stop` ends it during a prefill.
- **`json_schema` prompt** (fork PR #27, by hyungraekim-bagelcode). The prompt shows the schema, not the OpenAI wrapper.
- **Coupled drafts** (fork PR #25) and **L2 prefetch at once** (fork PR #26). Output tokens do not change. In the gate runs, decode and cold prefill are equal to 1.1.0 within the noise ([runs/2026-10-21](../runs/2026-10-21/INDEX.md#release-gate-of-engine-111)).
- Known issue: the new image holds about 0.14 GiB more on the head, and the lowest head `MemAvailable` in the long run was 1.58 to 1.62 GiB (1.1.0: 1.68 to 1.85 GiB). See [OPERATIONS.md](OPERATIONS.md#engine-switches-of-release-111).

## 9. The Zig engine as the default (22 October 2026)

The Zig engine is bertholomus's second engine for this model. The fork branch `dsv41-zig-dev` ports to it the changes that this page lists for the Python engine, in these pull requests of [the fork](https://github.com/Elevatormusic/TensorFold): `top_k` 0 (#29), the RDMA GID setting (#30), structured output (#31), forced DSML tool calls (#32), fault isolation with exit code 75 (#33), the loader for the 2.8 bpw pack (#38), special-token text as plain text (#39), and a peer watch that ends a rank whose peer died (#40). The release fault campaign found two faults that blocked a release (F1 and F3); both are fixed ([runs/2026-10-22](../runs/2026-10-22/INDEX.md#zig-engine-and-the-28-bpw-pack)).

This recipe changed as follows: `ENGINE` in `.env` picks the engine (`zig` is the default); the Zig scripts and `docker/Dockerfile.zig` are new; the Python engine and the 2.9 bpw pack stay as the rollback ([ROLLBACK.md](ROLLBACK.md)). See [ZIG.md](ZIG.md).
