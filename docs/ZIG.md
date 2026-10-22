# The Zig engine with the 2.8 bpw pack (the default)

Since 22 October 2026, this recipe serves DeepSeek-V4.1-Flash-1017 with the Zig engine and the EXL3 2.8 bpw pack. The earlier setup (the Python engine 1.1.1 and the 2.9 bpw pack) stays as the rollback. [ROLLBACK.md](ROLLBACK.md) tells you how to use it.

The pair is the same: two NVIDIA DGX Spark, TP=2, joined by one QSFP cable. The commands are the same: `./start.sh`, `./stop.sh`, `./status.sh` and `./verify.sh`. `ENGINE=zig` in `.env` selects the Zig engine. This is the value in `.env.example`.

## What changes for you

| | Zig engine (default) | Python engine (rollback) |
|---|---|---|
| Engine | [TensorFold fork](https://github.com/Elevatormusic/TensorFold), branch `dsv41-zig-dev`, commit `d58d25ccd6dbd624814cc2ccc848f8e5ba84580c` | the same fork, release `dsv41-1.1.1` (`cb4ceaff2b8c`) |
| Weights | `jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` (MIT, 47 files, 190 GiB) | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` (196 GiB) |
| Image | `tensorfold-dsv41-zig:d58d25cc` | `tensorfold-dsv41:cb4ceaff2b8c` |
| Free memory at idle | 6.43 GiB (head), 6.47 GiB (worker) | 6.47 GiB, 6.80 GiB on the same pack; about 3 GiB less on the 2.9 bpw pack |
| Decode, 1 stream, code | 85.3 tok/s | 87.1 tok/s |
| Prefill, cold 32K prompt | 2,161 to 2,487 tok/s | 2,004 to 2,268 tok/s |
| Tool calls, 505 requests | 0.00% lost or malformed | 0.40% lost (506 requests) |
| Exit code of a broken pair | 75 on both ranks, 2 to 8 s after the fault | 75 after 5 s |
| Prompts on disk after a restart (`SPILL_GIB`) | not set up | yes |

The numbers are in [Measured numbers](#measured-numbers). The Zig engine and the Python engine decode at nearly the same speed. The Zig engine prefills 7.8 to 9.7% faster. The 2.8 bpw pack leaves about 3 GiB more free memory on each node.

Credit: the 2.8 bpw pack is the work of the author of `jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`. The Zig engine and its kernel kit are the work of bertholomus. See [CREDITS.md](../CREDITS.md).

## Test status (read this first)

- The full release tests ran on earlier images of the same branch. The release image `d58d25cc` adds two checks: a `TCP_USER_TIMEOUT` of 20 s on the rank links, and a check of every RDMA ring after a synchronization. On 2026-10-22 (20:03-20:29Z) it passed the host tests (`zig build test-cuda-host`), the 6 robustness cases (a surviving rank exits with 75 in 2-8 s; clean stops and a restart in 33 s pass) and a decode check: 85.5 tok/s at 1 stream and 123.1 tok/s at 4 streams (median of 3), +0.3 % and +0.1 % against the earlier image. Memory low point: head 5,537 MiB, worker 5,393 MiB.
- The scripts of this recipe for the Zig engine (`scripts/zig_*.sh`, `docker/Dockerfile.zig`) have offline tests only (`tests/test_zig_engine.py`). Nobody has run them on a pair yet. They use the same commands as the scripts of the tests.
- [Gaps](#gaps) lists what a public user cannot do yet, and what no test covers.

## Requirements

- Two DGX Spark, with the network setup of [NETWORK.md](NETWORK.md) (MTU 9000 on the CX7 ports).
- About 500 GB of free disk on each node: the pack 190 GiB, the Engram tables 189 GiB, the rank cache 96 GiB, the kernel kit 0.6 GiB, and the images. A node that also holds the rollback setup needs about 1 TB.
- 110 GiB of free memory (`MemAvailable`) on each node before the start. `scripts/preflight.sh` checks this. Close other AI programs, browsers and desktop programs first.
- Docker with GPU support, Python 3, `curl`, `rsync` and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`) on the head. The worker needs Docker and `rsync`.

## Set up

Run each command on the head, in this folder.

1. **Make your settings file.** Open `.env` and type your addresses and folder paths. `ENGINE=zig` is already there.

   ```bash
   cp .env.example .env
   ```

2. **Download the pack.** This step gets 380 GiB (the 2.8 bpw pack and the Engram tables), checks every file against [`weights-2.8bpw.sha256`](../runs/2026-10-22/zig-28/weights-2.8bpw.sha256), and copies the view to the worker.

   ```bash
   scripts/download.sh
   ```

3. **Prepare the chat template.** The pack has its own template file. The view does not use it. The tested template is the template of the MiaAI-Lab kit with our patch ([scripts/patch_template.py](../scripts/patch_template.py)).

   ```bash
   scripts/patch_template.py --worker
   ```

4. **Build the engine image.** This step builds the base image (the Python engine image of commit `3c45bdaeab7d`) if it is missing, then the Zig image, and copies it to the worker. The compile needs about 40 GiB of free memory on the head. See [The image](#the-image).

   ```bash
   scripts/zig_build_image.sh
   ```

5. **Tune the memory settings (once).** Same as for the Python engine. It asks for your password on each node. `scripts/host_tuning.sh --undo` removes it.

   ```bash
   scripts/host_tuning.sh
   ```

6. **Make the inputs of the engine.** This step gets the kernel kit (at a fixed commit, checked with `scripts/verify_kit.sh` before anything from it runs), makes the token map, and makes the weight file of each rank (96 GiB on each node). The server must not run.

   ```bash
   scripts/zig_inputs.sh
   ```

7. **Start the server.** It checks your setup, starts rank 1 on the worker, then rank 0 on the head, and sends one real request. A start takes 26 to 34 s when the rank caches exist.

   ```bash
   ./start.sh
   ```

8. **Check that it works.** This sends five real requests and shows the result of each.

   ```bash
   ./verify.sh
   ```

Your server is ready at `http://127.0.0.1:8888/v1`. The model name is `DeepSeek-V4.1-Flash-1017`. [CONNECT.md](CONNECT.md) tells you how to connect your app. The port, the model name and the other API settings are the same as for the Python engine.

## Files and folders

| Path | What | Made by |
|---|---|---|
| `$MODEL_VIEW/model/` | the pack (46 files, hard links), its `LICENSE`, and our chat template | `scripts/download.sh`, `scripts/patch_template.py` |
| `$MODEL_VIEW/engram/` | the Engram tables of the 1017 checkpoint (shards 47 and 48) | `scripts/download.sh` |
| `$CACHE_DIR/zig-kit/` | the kernel kit (Triton kernels as binaries, EXL3 kernels, RoPE tables, vision files), 0.6 GiB | `scripts/zig_inputs.sh kit` |
| `$CACHE_DIR/zig-data/dsv41_token_map.json` | Engram's compressed token map | `scripts/zig_inputs.sh token-map` |
| `$CACHE_DIR/zig-rank-cache-28/` | one weight file for each node, in the order that the loader reads (96 GiB) | `scripts/zig_inputs.sh rank-cache` |

The folders exist on both nodes, at the same paths. The rank cache is made once for each view. The name `28` shows the pack. A new pack gets a new name.

The Python engine uses other folders (`tf-ds-rank`, `tf-ds-spill`, `triton`, `nv`, `torch_extensions`). Both setups can live on the same node.

## The image

`docker/Dockerfile.zig` builds three programs from one commit of the fork:

| Program | Path in the image | Role |
|---|---|---|
| `tensorfold-native` | `/opt/tensorfold/native/bin/tensorfold-native` | rank 0 (head): the OpenAI-compatible server |
| `tf-dsv41-lanes` | `/opt/tensorfold/bin/tf-dsv41-lanes` | rank 1 (worker): the follower |
| `tf-dsv41-rank-cache` | `/opt/tensorfold/bin/tf-dsv41-rank-cache` | makes the weight file of a rank |

The build uses Zig 0.17.0 (downloaded from ziglang.org and checked against a sha256 in the Dockerfile) with `-Dsm=121 -Doptimize=ReleaseFast`, and xgrammar v0.2.8 for structured output. xgrammar is fetched by the full commit `97787376faee` (the commit of the tag v0.2.8; the build checks it), with its submodules at the commits that this tree records. The image has the labels `tf.engine=zig` and `tf.ref=<full commit>`. `scripts/zig_build_image.sh` checks the label and checks that both nodes hold the same image ID.

| Item | Value |
|---|---|
| Image tag | `tensorfold-dsv41-zig:d58d25cc` |
| Image ID | `sha256:9bdfda7fe538d64774e7ffdcc8ac2d061b54fe244ab4c51e778f21a38b3d7d51` (the same on both nodes) |
| Source | `Elevatormusic/TensorFold`, branch `dsv41-zig-dev`, commit `d58d25ccd6dbd624814cc2ccc848f8e5ba84580c` |
| Base image | `tensorfold-dsv41:3c45bdaeab7d` (NGC PyTorch 26.07: CUDA 13, NCCL, libibverbs; the recipe builds it from `docker/Dockerfile`) |

The branch `dsv41-zig-dev` is the branch `dsv41-zig` plus three pull requests of the fork that are open at this time: #38 (the loader for the 2.8 bpw pack), #39 (special-token text in a message is plain text) and #40 (a rank whose peer dies ends with exit code 75). The [fork documentation](https://github.com/Elevatormusic/TensorFold/blob/dsv41-zig-dev/docs/recipes/deepseek-v41-2x-spark.md) has the engine side.

## The launch

`scripts/zig_start.sh` runs two `docker run` commands. The container names are `tfv41-r0` (head) and `tfv41-r1` (worker), as for the Python engine, so `./stop.sh` and `./status.sh` work for both. Both ranks mount the view, the rank cache, the kernel kit and the token map read-only. The main settings:

| Setting | Value | Where it comes from |
|---|---|---|
| Rank 1 starts first | it retries its connection to rank 0 until rank 0 is up | the tests |
| Rank 0 rendezvous | `HEAD_IP`, port `ZIG_RANK_PORT` (29620) | `.env` |
| Server | `BIND_HOST`:`PORT`, `--context CONTEXT`, `--parallel PARALLEL` | `.env` |
| Streams, pool | `TF_DS_STREAMS` and `--streams` = `PARALLEL`, `--pool` = `CONTEXT` | `.env` |
| Sampling | `--temperature`, `--top-p` from `.env`, `--top-k 20`, `--max-tokens 32768` | the tests |
| Thinking | `--no-thinking`, or `--thinking` when `THINKING=1` | `.env` |
| RoCE v2 GID | `TF_RDMA_GID_INDEX`, read from each node at each start (`ZIG_RDMA_GID_INDEX=auto`) | `scripts/zig_gid.sh` |
| NCCL GID | automatic: the script does not set `NCCL_IB_GID_INDEX` | the tests |
| Memory | `--oom-score-adj 1000`, `--cgroup-parent tensorfold.slice` | `scripts/host_tuning.sh` |
| `TF_DS_EXIT_ON_FAULT=1` | on both ranks: it gives the exit code 75 | the tests |

Only `CONTEXT=1048576` and `PARALLEL=4` have a test. `PARALLEL` can be 1 to 4. Other values of `CONTEXT` have no test.

These `.env` settings do not apply to the Zig engine: `MTP_DRAFTS`, `PREFILL_CHUNK`, `SPILL_GIB`, `MEM_FLOOR_GIB`, `MEM_CAP`, `VISION`, `DROP_UNOFFERED_TOOL_CALLS`, `GID_INDEX` and `TF_COMMIT`/`IMAGE`. The Zig engine has its own drafting, and no vision, spill or memory-floor setting in this launch. To change an engine switch, add `-e NAME=value` to `common_args` in `scripts/zig_start.sh`.

## Check that it works

`./start.sh` does these checks and stops with exit code 1 if one fails:

1. The API answers: `curl -fsS -m 5 http://127.0.0.1:8888/v1/models` gives 200.
2. A real completion works. A 200 from `/v1/models` or `/health` is not proof that TP=2 works. Only a real completion is. The script sends "What is 17 times 23? Answer with the number only." at temperature 0. The pass condition: HTTP 200, `content` is `391`, `finish_reason` is `stop`.

`./verify.sh` sends five more requests (plain, thinking, `json_schema`, forced tool call, regex).

## Stop and restart

```bash
./stop.sh
```

`./stop.sh` stops rank 0 first, then rank 1, saves both logs to `logs/`, and prints the exit code of each rank. Rank 0 exits 0 in 4 to 6 s and sends its done frame. Rank 1 then exits 0. It exits 75 when the stop came during a prefill, because rank 1 sees the link end before the done frame. Live requests end with `finish_reason` `cancelled`.

To restart, run `./stop.sh`, then `./start.sh`. A restart right after a stop takes 30 to 34 s to a correct reply. A pair is good again only after a real completion. `./start.sh` ends with one.

## Exit codes

| Exit | Meaning | Action |
|---|---|---|
| 0 | clean stop | none |
| 75 | a peer fault: the other rank died, its link closed, a keepalive or the `TCP_USER_TIMEOUT` (20 s) ended the link, or a round failed (its requests got the error first). It comes 2 to 8 s after the death of the peer, idle or in decode. | stop and remove both containers, then start the pair again |
| 137 | killed (SIGKILL: an OOM kill, earlyoom or an operator) | check `docker inspect -f '{{.State.OOMKilled}}'` and the memory ([MEMORY.md](MEMORY.md)), then restart the pair |
| other (for example 1) | an error while the peer is up, for example a start that fails (a wrong setting or a missing file) | read the log and fix the cause. Do not restart in a loop. |

Either rank can exit 75. The other rank then exits 75 too, because it sees the link end. `/health` can say 200 while the pair is broken in an older engine build: the build `d58d25cc` has the peer watch that ends the process instead.

`TF_DS_EXIT_ON_FAULT=1` must stay on for both ranks.

### The supervisor (optional, not tested on hardware)

The default way to run the engine is the manual start and stop: [The launch](#the-launch) and [Stop and restart](#stop-and-restart) (`./start.sh`, `./stop.sh`). After an exit code 75, you run `./stop.sh && ./start.sh` yourself.

`scripts/zig_supervise.sh` is an optional helper that does this for you. It has offline tests only: nobody has run it on a pair. It follows the table above. Run it on the head after `./start.sh`, for example in `tmux`:

```bash
scripts/zig_supervise.sh
```

| It sees | It does |
|---|---|
| rank 0 exits 0 | ends with 0, no restart |
| a rank exits 75 | `./stop.sh`, then `./start.sh`. The new pair counts as good only if `./start.sh` ends with 0. At most 3 restarts in a row (`ZIG_MAX_RESTARTS`). |
| a rank exits 137 | no restart, ends with 1 |
| any other code | no restart, ends with 1 |

Keep `RESTART_POLICY=no` in `.env`. The docker restart policy restarts one container alone, and the pair needs both ranks to restart together.

## Memory

The weights are 3 GiB smaller per node than the 2.9 bpw pack, so the nodes keep more free memory. The numbers are `MemAvailable`:

| Point | Head | Worker |
|---|---|---|
| Idle after 120 s, Zig | 6.43 GiB | 6.47 GiB |
| Idle, Python engine, same pack | 6.47 GiB | 6.80 GiB |
| Lowest, suite and 4-stream decode | 5.00 to 5.54 GiB | 5.13 to 5.69 GiB |
| Lowest, needle of 1,038,664 tokens | 3.25 GiB | 3.23 GiB |
| Lowest, soaks of 65 min at 4 client threads | 4.28 to 4.36 GiB | |

Rank 0 anonymous memory grew by 24 to 26 MiB in the last 30 min of each soak. earlyoom (set up by `scripts/host_tuning.sh`) stops the engine at 1 GiB. The tests also ran a small memory guard next to each rank. The guard is a test tool and is not part of this recipe.

[MEMORY.md](MEMORY.md) explains earlyoom and how to read a stop.

## Measured numbers

All Zig numbers come from [`runs/2026-10-22/zig-28/zig-release-faults.md`](../runs/2026-10-22/zig-28/zig-release-faults.md). The 2.8 bpw pack against the 2.9 bpw pack (both on the Python engine) is in [`pack-2.8bpw-test.md`](../runs/2026-10-22/zig-28/pack-2.8bpw-test.md). One pair of DGX Spark, 22 October 2026 (the node clock, UTC).

**Decode** (median of 3, Zig and Python 1.1.1 on the same pack):

| Workload | Zig | Python | Zig / Python |
|---|---:|---:|---:|
| 1 stream, code set | 85.3 tok/s | 87.1 tok/s | 97.9% |
| 1 stream, prose | 57.7 tok/s | 55.0 tok/s | |
| 1 stream, structured output | 104.9 tok/s | 107.9 tok/s | 97.2% |
| 4 streams, all together | 123.0 tok/s | 123.3 tok/s | 99.8% |

**Prefill** (cold 32K prompt, `cached_tokens` 0, median of 3):

| Prompt set | Zig | Python | Change |
|---|---:|---:|---:|
| B (natural text) | 2,161 tok/s | 2,004 tok/s | +7.8% |
| C | 2,242 tok/s | 2,066 tok/s | +8.5% |
| D (kit_bench words) | 2,487 tok/s | 2,268 tok/s | +9.7% |

**Prefix cache.** A repeated 32K prompt: 13.9 s, then 1.0 s (`cached_tokens` 30,720). A repeated 128K prompt: 57.9 s, then 1.2 s (`cached_tokens` 129,024).

**Long context.** The needle was found at 130,838, 261,751, 523,577 and 1,038,664 tokens (7 of 7).

**Agent load.** Tool loops with parallel calls, 505 requests, 4 client threads: 0 lost, 0 malformed, 0 calls to a tool that the request did not offer, 0 errors. The Python engine lost 2 of 506 (0.40%).

**Soaks.** Two soaks of 65 min, about 2,500 requests each (plain, JSON schema, forced tool call, `top_k` 0, streamed, 16K needle): 0 reference mismatches, 0 round faults, no earlyoom stop. In each soak, one schema request ended with `finish_reason` `length` at its 200-token limit. This is a limit of the test, and the output stayed inside the grammar.

**Checks.** Recipe suite 30 of 31 (the one case that needs a router is not run), grammar check 14 of 14, `so_check` 16 of 16, structured-output and DSML conformance 46 of 46, answer checks 16 of 16.

**The pack.** In the A/B on the Python engine, the 2.8 bpw pack gave quality and speed equal to the 2.9 bpw pack in our tests (suite 32/32, grammar 14/14, 515 and 530 tool-call requests with 0 failures). The MTP acceptance at depth 1 was 2 to 3 points lower (74.5% against 71.7%). Nobody has measured it on the Zig engine yet.

## Known differences from the Python engine

- **`response_format: json_object`.** The Python engine adds a JSON instruction to the prompt (about 24 tokens). The Zig engine adds none. The output is valid JSON on both, because both constrain it with a grammar. Only the prompt-token counts differ.
- **Invalid fields.** Five invalid fields get HTTP 200, not a 4xx, on both engines: an unknown `tool_choice`, `max_tokens` 0, `max_tokens` -5, `temperature` -1, `top_p` 2. The other 18 bad requests get a 4xx on both.
- **Text.** For the same 48 prompts at temperature 0, 25 replies are identical. In the others the first difference comes early in long replies. All 48 replies are valid and 16 of 16 answer checks pass on both engines.
- **Special-token text in a message.** The Zig engine reads it as plain text, as the Python engine does (the prompt ids are equal in 21 of 21 cases).
- **Spill.** The Python engine writes cached prompts to disk (`SPILL_GIB`). This launch of the Zig engine sets no spill.

## Update the engine

1. Set `ZIG_TF_COMMIT` (the full commit) and `ZIG_IMAGE` (`tensorfold-dsv41-zig:` and the first 8 characters of the commit) in `.env`.
2. Run `./stop.sh`, then `scripts/zig_build_image.sh`, then `scripts/zig_inputs.sh`, then `./start.sh`.
3. Run `./verify.sh` and `tests/run_all.sh`.

`scripts/deploy_gate.sh` works for the Python engine only. A new Zig image needs a new gate run: save its output under `runs/YYYY-MM-DD/` ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Gaps

These are things that a public user cannot do, or that no test covers, at this time:

1. **The image is not built from this recipe yet.** The tested images came from the same Dockerfile and the same commands on the worker. The scripts of this recipe have offline tests only. Run `scripts/zig_build_image.sh` once and save the output before you trust it. Its result is the single place for the image ID (see [The image](#the-image)).
2. **The base image can differ.** The recipe builds `tensorfold-dsv41:3c45bdaeab7d` from `docker/Dockerfile` with `pip`. A new build can pick other package versions than the image of the tests. The Zig engine uses the CUDA tools, NCCL and `libibverbs` of the NGC base. It does not run the Python engine. Nobody has tested a build on the bare NGC image.
3. **The kernel kit comes from a third party.** `scripts/zig_inputs.sh kit` downloads the folder `kit/` of `bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10` (mixed licences: see `KIT-LICENSES.md` in the kit). The kit README names this place. The script fixes the download to one commit of that repo (`4996441a82d9`; the `MANIFEST` of that commit has the sha256 `8c6dcb71`, the file that the tests used, read over HTTPS on 22 October 2026). The script then checks, before anything from the kit runs: (1) the sha256 of `MANIFEST`; (2) every file of the kit against `MANIFEST`, and that the folder holds no file that `MANIFEST` does not list, with `scripts/verify_kit.sh` of this repo. It never runs `tools/verify_kit.sh` of the kit. It sends `scripts/verify_kit.sh` to the worker on stdin, so the worker also runs only the checker of this repo. This recipe has not downloaded the other kit files from that commit yet; the offline tests use a small kit of their own.
4. **The token map** is made by `make_token_map.py` of the fork. The file it made from our tokenizer has the same sha256 as the file of the tests (`de40bcdd71f2`). A different `tokenizers` version can give another file. The script warns in that case.
5. **earlyoom and rank 1.** `scripts/host_tuning.sh` sets `--prefer ^(tensorfold|tf-dsv41)`. It matches rank 0 (`tensorfold-native`) and rank 1 (`tf-dsv41-lanes`); the containers also run with `--oom-score-adj 1000`. A node that ran an older `scripts/host_tuning.sh` has `--prefer ^tensorfold` and must run the script again (`scripts/preflight.sh` checks this). The tests saw no earlyoom stop of rank 1 and did not test this choice on hardware.
6. **Not measured on the Zig engine:** MTP acceptance by depth; a prompt cache that survives a restart; image answers (3 image requests were compared for prompt tokens only); `PARALLEL` below 4; `CONTEXT` other than 1,048,576; a KL measurement against the native release.
7. **One pair.** All results come from one pair of DGX Spark.
