# Operations

> **This page is about the Python engine, the rollback (`ENGINE=python`).** The default engine is the Zig engine: read [ZIG.md](ZIG.md) for its start, stop, exit codes and restart rules. `./start.sh`, `./stop.sh`, `./status.sh` and `./verify.sh` work for both engines. The settings that this page names (`MTP_DRAFTS`, `PREFILL_CHUNK`, `VISION`, `MEM_FLOOR_GIB`, `SPILL_GIB`) and the first-start times apply to the Python engine only.

## Start, stop, check

```bash
./start.sh
```

`start.sh` first runs `scripts/build_extensions.sh`. It compiles the CUDA extensions on both nodes before the weights load, and removes lock files that a stopped compile left. A compile while the weights fill the memory can make earlyoom stop the engine; a left lock file makes the next start wait forever. Logs: `logs/build-extensions-head.log` and `logs/build-extensions-worker.log`.

```bash
./status.sh
```

```bash
./verify.sh
```

```bash
./stop.sh
```

- **Start times.** The first start writes each rank's weight cache to `CACHE_DIR` (~15 min). Later starts read it back and come up in ~2-4 min.
- **Short prompts are fast from the first request.** Compiled kernels are kept in `CACHE_DIR/triton` and `CACHE_DIR/nv`, and the warm-up covers prompts of 1 to 17 tokens. Before this, a 3-16 token prompt took 2.6-6.1 s the first time after each start; now it takes 0.1 s.
- **First long prompt.** The first long prompt after a start (8K+ tokens) takes ~6 s longer than later ones. The README's 8K row includes this. A longer warm-up list does not remove it; the likely cause is cold Engram reads from NVMe.
- **Prompts survive a restart.** With `SPILL_GIB` above 0 (default 64), kept prompts also go to `CACHE_DIR/tf-ds-spill` on each node. After a restart, a conversation continues from disk (an 18K-token prompt: 2.7 s instead of 16.8 s). Old entries go after 7 days or past the size limit.

## Prefill configuration

Set `PREFILL_CHUNK` in `.env` to select the tokens per prefill chunk on both ranks.
Use a positive integer, for example `2048` or `512`. The default stays `2048`.
An absent or empty value also uses `2048`. Other values stop `start.sh` before
preflight, extension builds or container commands. Restart the server after a change.
This setting does not change the other serving defaults. No performance result is
claimed for a different chunk size.

## Engine switches of release 1.1.0

The engine release [dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) adds the settings below. The release notes explain each change.
`start.sh` sets only `DROP_UNOFFERED_TOOL_CALLS`, from `.env`. The other settings keep the engine default, which is the tested value.
To change one, add `-e NAME=value` to `run_args` in `start.sh`. This function builds the arguments of both ranks, so one line gives both ranks the same value. Then restart the server.

**Tool calls to tools that the request did not offer.** `DROP_UNOFFERED_TOOL_CALLS=1` (the default in `.env.example`) makes the engine drop a call to a tool that the request did not offer. It does this also while the reply streams. Set `DROP_UNOFFERED_TOOL_CALLS=0` in `.env` to turn it off, then restart. The engine itself has this off by default. The recipe turns it on. Only `0` and `1` are valid.

**Decode graphs (fork PR #19).** The engine no longer captures a decode graph while it serves. The warm-up captures every graph that the context window can use. If a graph is still captured late, the log shows a `WARNING` line with `captured while serving`. The gate run has none ([runs/2026-10-21/INDEX.md](../runs/2026-10-21/INDEX.md)).

| Setting | Default | Effect |
|---|---|---|
| `TF_DS_ROUND_ROWS` | `1,2,3,4,6,8,12,16` | Row sizes of the round graphs for contexts up to `TF_DS_FINE_CONTEXT`. A round runs in the graph of the next size up. `0`: a graph for every row count (more graph memory). |
| `TF_DS_ROUND_ROWS_LONG` | `1,2,4,8,16` | The same, for longer contexts. |
| `TF_DS_FINE_CONTEXT` | `32768` | The context length (tokens) where the two lists change. |
| `TF_DS_CAPTURE_MODE` | `thread_local` | `global` selects the capture mode of release 1.0.0. |

**Host memory (fork PRs #21 and #24).** These settings lower the host-memory peak of the head. See [#21](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/21).

| Setting | Default | Effect |
|---|---|---|
| `TF_DS_ENGRAM_BLOCK` | `512` | The cross-rank sum of an Engram layer runs this many rows at a time. `0`: the whole chunk (a higher peak, same tokens). |
| `TF_DS_TRIM_MIB` | `256` | When no stream is filling, free torch blocks over this size go back to the node. `-1`: off. |
| `TF_DS_ADMIT_GIB` | `1` | With streams live, a new request waits while free memory is under `MEM_FLOOR_GIB` plus this value. The log shows `admission waits` when it first waits. `0`: off. |

Draft ranking (fork PR #20): `TF_DS_DEPTH_POLICY=rank` is the default. The engine ranks the drafts of all streams in one round by how likely each draft token is to survive. A different value (for example `conf`) chooses the draft depth of each stream alone. Output tokens do not change.

**Little free memory on the head.** The 1.1.0 release notes give a known issue: with 4 long streams, the lowest `MemAvailable` on the head was 1.45 GiB in the gate run, against the 1 GiB earlyoom limit. If the head has less free memory at idle (for example because other processes use it), set `PREFILL_CHUNK=1024` or `PARALLEL=2` in `.env`, then run `./stop.sh && ./start.sh`. No run in this repo measures these two values on 1.1.0.

## Engine switches of release 1.1.1

The engine release [dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1) adds the settings below. The release notes explain each change. `start.sh` does not set them: the engine default is the tested value. To change one, add `-e NAME=value` to `run_args` in `start.sh`, then restart the server.

| Setting | Default | Effect |
|---|---|---|
| `TF_DS_DRAFT_COUPLED` | `1` | The drafts of a sampled stream follow the stream's own keyed draw (fork PR #25). Output tokens do not change. `0`: greedy drafts for every stream, and no second draft graph. |
| `TF_DS_DRAFT_COUPLED_SCALE` | `0.5` | Drafter temperature as a part of the stream temperature, for coupled drafts. |
| `TF_EXL3_L2_DELAY_NS` | `0` | The decode L2 prefetch issues at once (fork PR #26). `3000` restores the wait of 1.1.0. |
| `TF_SHUTDOWN_GRACE_S` | `3` | Seconds that the server waits for the current round after SIGTERM or Ctrl-C (fork PR #34). The exit code is 0. |
| `GID_INDEX` (`.env`) | `3` | `-1` makes the engine read the RoCE v2 IPv4 GID index of each port from sysfs (fork PR #22). The log shows `RDMA GID index of ... (detected)`. A value of 0 or more fixes the index. |

**Known issue: head memory margin.** In the long run of the 1.1.1 gate (4 streams with 20K, 50K, 100K and 200K prompts), the lowest head `MemAvailable` was 1.58 to 1.62 GiB, against 1.68 to 1.85 GiB for 1.1.0. The new image holds about 0.14 GiB more on the head (the second draft graph of fork PR #25). The rest of the gap came from less free memory at the start of each window. There was no earlyoom stop, and the low point stayed 0.58 GiB over the 1 GiB earlyoom limit. With less free memory at idle, use the low-memory settings (`PREFILL_CHUNK=1024` or `PARALLEL=2`; see "Little free memory on the head" above). `TF_DS_DRAFT_COUPLED=0` gives back part of the extra memory (not measured). Run record: [runs/2026-10-21](../runs/2026-10-21/INDEX.md#release-gate-of-engine-111).

## When the lane breaks

The engine keeps serving when one request fails:
- A fault in one request's own work (its grammar, its sampling, its client going away) ends only that request.
- A device fault (CUDA error, out of memory) or a lost rank breaks the lane. NCCL is aborted, and the lane cannot serve again in that process.

A broken lane ends both rank processes with **exit code 75** after 5 s (`TF_DS_EXIT_ON_BREAK=1`, the default). The 5 s give the live requests time to get their error.

- **Restart by hand.** Run:

```bash
./stop.sh && ./start.sh
```

- **Restart on its own.** Use a supervisor, with these rules:
  - Both ranks must restart together, rank 0 first, because rank 1 connects to rank 0's port 29551.
  - The simplest supervisor is a systemd unit on the head that runs `./start.sh`, with `ExecStop=./stop.sh` and `Restart=on-failure`, and that checks `./status.sh`.
  - `RESTART_POLICY=on-failure` in `.env` (docker's own restart) restarts each container alone. It works when rank 0 comes back first, but no tool enforces that order.
- **Out of memory.** With `scripts/host_tuning.sh`, earlyoom stops the engine when a node has less than 1 GiB available, so the node stays reachable by ssh. `journalctl -u earlyoom` shows when it did. Then restart by hand as above. [Memory](MEMORY.md) explains how to read the journal, and `terminate called without an active exception` in the log of rank 0 (a SIGTERM from earlyoom when the journal shows one at the same time, else a crash).
- **Out of memory under load.** Some users see earlyoom stop the engine with several long requests at the same time, or with many images in one agent session ([#21](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/21)).
  Short tests often do not show it. earlyoom runs on each node, so watch both. Run this on the head while the load runs. It shows the head first, then the worker:

  ```bash
  . ./.env; watch -n 1 "grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes $WORKER_SSH grep MemAvailable /proc/meminfo"
  ```

  In the traces in [runs/2026-10-20/INDEX.md](../runs/2026-10-20/INDEX.md), image requests made the head run out first. These traces ran on a validation image (`deepseek-v41-tp2` at 434301e with fork PRs #13, #14 and #15), before fork PRs #21 and #24, which the shipped `cb4ceaff2b8c` has. The values on the shipped image can differ. The head runs the vision tower and decodes the images. The worker went down to about 2.0 GiB.

  | Load | `PARALLEL` | Head lowest `MemAvailable` | Result |
  |---|---|---|---|
  | 4 long prompts, then 4 requests with 8 images each at the same time | 4 | 1,009 MiB (earlyoom reading) | earlyoom stopped the engine |
  | 4 requests with 8 images each, on a new engine | 4 | 1,379 MiB | all passed |
  | 4 long prompts (loaded from the spill cache), then 4 requests with 8 images each | 2 | 1,356 MiB | all passed |
  | then 4 requests with 16 images each (needs `--vision-max-images`; the recipe allows 8) | 2 | 901 MiB (earlyoom reading) | earlyoom stopped the engine |

  Each image request also had random-word text: about 34.5K prompt tokens in all with 8 images. With 8 images in each request (the recipe's limit), `PARALLEL=2` passed the load that failed with `PARALLEL=4`. This is not a clean A/B: in the `PARALLEL=2` run the long prompts came from the spill cache and were not filled again. With 16 images in each request, `PARALLEL=2` also failed.

  If `MemAvailable` on either node goes near 1 GiB (1,048,576 kB):
  1. Set `PARALLEL=2` in `.env`, then run `./stop.sh && ./start.sh`. Each request still gets the full context. This is measured only on the validation image above.
  2. Or set `PREFILL_CHUNK=1024` in `.env` and restart (the 1.1.0 release notes advise this for a head with little free memory). It is not measured for memory in this repo.
  3. Send fewer images at the same time. Keep the images in each request low (the server accepts at most 8), and do not send many image requests at the same time. For an agent, send screenshots one at a time, not as a batch.
  4. If you do not need image input, set `VISION=0` in `.env`, then run `./stop.sh && ./start.sh`. The vision tower then does not load on the head. The engine log gives its size as 0.95 GiB on rank 0 ([runs/2026-10-20/oom/run2-engine-vision.txt](../runs/2026-10-20/oom/run2-engine-vision.txt)). A run with `VISION=0` is not measured yet.
  5. Do not raise `MEM_FLOOR_GIB`. With 2.0, an 8K prompt failed with CUDA out of memory, which breaks the lane ([runs/2026-10-18/INDEX.md](../runs/2026-10-18/INDEX.md)).
  6. Do not turn off earlyoom or lower its limit. Without it, a node that runs out of memory can stop answering ssh until a power cycle.

  Issue #21 tracks the engine fix. For an issue, run `scripts/oom_report.sh` and attach the file ([Memory](MEMORY.md#report-a-stop)).
- **The whole node stops (no ssh, no ping above the link).** One report ([#24](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/24)) shows this kernel log on the head just before the hang:

  ```text
  mlx5_core 0002:01:00.0: PCIe Bus Error: severity=Uncorrectable (Non-Fatal) ...
  mlx5_core 0002:01:00.0:    [14] CmpltTO
  arm-smmu-v3 arm-smmu-v3.0.auto: CMD_SYNC timeout at ...
  ```

  Only a power cycle recovered the node. The cause is not known. That setup used driver 580.173.02. This recipe is tested on 580.178.04 and kernel 7.0.0-1019-nvidia, and our pair has no such event in 20 days of this load. `scripts/preflight.sh` shows a `[warn]` line for a driver older than 580.178.04.
  1. Update the driver to 580.178.04 or newer on both nodes before you use this recipe.
  2. After the power cycle, if the node stops at a `grub>` prompt, the file system needs a check. Start the node from recovery media and run `fsck -f` on the root partition.
  3. Save `journalctl -k -b -1` from the node that stopped, and add it to #24 with the driver, kernel and CX7 firmware (`ethtool -i <netdev>`).
- **Boot.** Do not let a second server also start at boot (a vLLM unit, another model). Both servers want port 8888 and ~108 GiB per node, and the second one fails or runs a node out of memory.

## Things that hurt on DGX Spark

- **Do not `docker exec` into a rank container while it serves.** The exec process joins the container's cgroup, and its memory counts against the rank.
- **Do not stop the display manager on a Spark** (`systemctl stop gdm`). On the measured pair it powered the node off.
- **`/health` 200 is not proof that TP=2 works.** Only a real completion proves it; `./verify.sh` sends five.
- **Memory.** The engine holds about 108 GiB on the head and 106 GiB on the worker, for its half of the weights, the KV pool and the 1M context. Preflight needs 111 GiB `MemAvailable` on the head and 109 GiB on the worker. Close other GPU programs, browsers and desktop programs before you start. [Memory](MEMORY.md) explains why the head runs out first.

## Logs

- `docker logs tfv41-r0` (head) and `ssh <worker> docker logs tfv41-r1`.
- `./stop.sh` saves both logs to `logs/` first.
- Each request prints one line with its numbers:

```text
[tensorfold] done req-... prompt=... cached=... tokens=... tok/s=... ttft=...s rounds=... accepted=.../...
```

## Update the engine

1. Set `TF_COMMIT` and `IMAGE` in `.env`.
2. Build the image and copy it to the worker:

```bash
./docker/build.sh
```

3. Deploy with rollback:

```bash
./scripts/deploy_gate.sh <new-tag> <old-tag>
```

`scripts/deploy_gate.sh` deploys the new image, runs a live structured-output request, the 32-test suite and the `--parallel` grammar check, and goes back to the old image if any step fails.
