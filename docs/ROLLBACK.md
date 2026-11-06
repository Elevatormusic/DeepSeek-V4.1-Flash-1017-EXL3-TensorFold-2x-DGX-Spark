# Rollback: the Python engine 1.1.1 with the 2.9 bpw pack

Until 22 October 2026, this recipe served DeepSeek-V4.1-Flash-1017 with the Python engine (release [dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1), commit `cb4ceaff2b8c`) and the EXL3 2.9 bpw pack. That setup still works and is the rollback of the [Zig engine](ZIG.md). Nothing was removed. `ENGINE=python` in `.env` selects it.

Use the rollback when the Zig engine gives a fault that the Python engine does not give, or when you need a feature that only the Python engine has:

- cached prompts on disk after a restart (`SPILL_GIB`);
- the experiments in `experiments/` (cache metrics, display KV, memory and top-k, tool output);
- the engine switches of [OPERATIONS.md](OPERATIONS.md) (draft depth `MTP_DRAFTS`, `PREFILL_CHUNK`, `VISION`, `MEM_FLOOR_GIB`).

The rollback has a cost. The 2.9 bpw pack leaves about 3 GiB less free memory on each node at idle (see [ZIG.md](ZIG.md#memory)). The Python engine prefills 7.8 to 9.7% slower on the same pack.

## Switch from the Zig engine to the rollback

Run each command on the head, in this folder.

1. Stop the server.

   ```bash
   ./stop.sh
   ```

2. Change two lines in `.env`. Every other line stays.

   ```text
   ENGINE=python
   MODEL_VIEW=$HOME/models/deepseek-v41-1017-exl3-2.9bpw
   ```

3. Get the 2.9 bpw pack, if it is not on both nodes. This step downloads 385 GiB, checks every file, and copies the view to the worker. Skip steps 3 and 4 if the pack and the template are already there.

   ```bash
   scripts/download.sh
   scripts/patch_template.py --worker
   ```

4. Build the Python engine image, if it is not on both nodes.

   ```bash
   docker/build.sh
   ```

5. Start the server. The first start writes the rank caches (about 15 min). Later starts take 2 to 4 min.

   ```bash
   ./start.sh
   ```

6. Check that it works.

   ```bash
   ./verify.sh
   ```

To go back to the Zig engine, set `ENGINE=zig` and `MODEL_VIEW=$HOME/models/deepseek-v41-1017-exl3-2.8bpw`, then run `./stop.sh && ./start.sh`.

Both setups use the container names `tfv41-r0` and `tfv41-r1` and the port from `.env`. Run one at a time. The setups use different cache folders in `CACHE_DIR`, so the rank caches of one do not touch the other.

## Disk

A node that holds both setups needs about 1 TB: each setup has its own view and its own copy of the Engram tables (189 GiB each). The Engram tables have the same bytes in both. If disk is tight, make the files in `$MODEL_VIEW/engram/` of the second view hard links to the files of the first view (`ln`, on one disk). The sha256 of those four files is the same in both lists (`runs/2026-10-17/weights.sha256` and `runs/2026-10-22/zig-28/weights-2.8bpw.sha256`).

## Set up the rollback from the start

You need the same hardware and network as for the Zig engine ([ZIG.md](ZIG.md#requirements)) and these changes:

- About 650 GB of free disk on each node. The model files use 414 GB (385 GiB). The first start needs 110 GiB more for the cache. With the default `SPILL_GIB=64`, cached prompts use up to 64 GiB more. With `SPILL_GIB=0`, about 550 GB is enough.
- About 111 GiB of free memory on the head and 109 GiB on the worker (`MemAvailable` before the start). `scripts/preflight.sh` checks this. The engine holds about 108 GiB on the head. See [MEMORY.md](MEMORY.md).

Steps, on the head, in this folder (with `ENGINE=python` and the 2.9 `MODEL_VIEW` in `.env`):

1. `cp .env.example .env`, then change the two lines above and the addresses and paths.
2. `scripts/download.sh`
3. `scripts/patch_template.py --worker`
4. `docker/build.sh` (builds the engine from [my TensorFold fork](https://github.com/Elevatormusic/TensorFold) at `TF_COMMIT` and copies it to the worker)
5. `scripts/host_tuning.sh` (once)
6. `./start.sh`
7. `./verify.sh`

## What changes in `.env`

| Setting | Zig engine | Python engine (rollback) |
|---|---|---|
| `ENGINE` | `zig` | `python` |
| `MODEL_VIEW` | `$HOME/models/deepseek-v41-1017-exl3-2.8bpw` | `$HOME/models/deepseek-v41-1017-exl3-2.9bpw` |
| Image | `ZIG_IMAGE`, `ZIG_TF_COMMIT` | `IMAGE`, `TF_COMMIT` |
| RoCE v2 GID | `ZIG_RDMA_GID_INDEX` (`auto`: read at each start) | `GID_INDEX` (3, or -1 for automatic) |
| `MTP_DRAFTS`, `PREFILL_CHUNK`, `SPILL_GIB`, `MEM_FLOOR_GIB`, `MEM_CAP`, `VISION`, `DROP_UNOFFERED_TOOL_CALLS` | not used | used |
| Other settings (addresses, `PORT`, `CONTEXT`, `PARALLEL`, `THINKING`, sampling, `API_KEY_FILE`, `CGROUP_PARENT`) | used | used |

## Where the rollback is documented

| Topic | File |
|---|---|
| Start, stop, check; engine switches of 1.1.0 and 1.1.1; when the lane breaks (exit code 75 after 5 s); update the engine | [OPERATIONS.md](OPERATIONS.md) |
| Why each change was made to the engine | [WHAT-I-CHANGED.md](WHAT-I-CHANGED.md) |
| Prompt spill to disk | [SPILL.md](SPILL.md) |
| Speed results of the Python engine and the vLLM comparison | [BENCHMARKS.md](BENCHMARKS.md), [runs/2026-10-17](../runs/2026-10-17/INDEX.md) |
| Exact versions and checksums of the 2.9 bpw pack | [runs/2026-10-17/PROVENANCE.md](../runs/2026-10-17/PROVENANCE.md) |

The scripts `scripts/build_extensions.sh` and `scripts/deploy_gate.sh`, and the experiments, work with `ENGINE=python` only. `docker/build.sh` builds the Python engine image at `TF_COMMIT`. It works with both engines.
