# Memory

The head node (rank 0) is the first node to run out of memory. earlyoom then stops the engine. This page explains why, how to read the stop, and which settings give the head more room.

> **The Zig engine (the default)** needs 110 GiB of `MemAvailable` on both nodes at the start. At idle the head has 6.4 GiB free; the 2.8 bpw pack leaves about 3 GiB more than the 2.9 bpw pack. The numbers, and the earlyoom gap for rank 1, are in [ZIG.md](ZIG.md#memory). The rest of this page was measured on the Python engine (the rollback), but the cause (the head is the tight node) is the same.

## What to do

1. Run `scripts/host_tuning.sh` once (it needs sudo on both nodes). Restart the server after it.
2. Start the server. `./start.sh` and `./status.sh` print the `MemAvailable` of the head. `[ ok ]` means 2.5 GiB or more. `[warn]` means less.
3. On a `[warn]`, close browsers and desktop programs on the head. Stop other containers on it. Do not run `docker build` or `docker load` while the engine serves. Do not stop the display manager (`systemctl stop gdm` powers the node off).
4. If the margin stays small, change one setting at a time in `.env`, then run `./stop.sh && ./start.sh`:
   1. `VISION=0` if you send no images (+0.95 GiB at idle).
   2. `PREFILL_CHUNK=1024`.
   3. `PARALLEL=2`.
5. If the engine still stops, run `scripts/oom_report.sh` on the head and attach the file to an issue ([Report a stop](#report-a-stop)).

## Why the head is the tight node

A DGX Spark has one pool of memory for the CPU and the GPU (GB10 unified memory). The engine weights, the KV pool, the page cache, the desktop and every other program take from the same pool. There is no separate GPU memory. So the `MemAvailable` line of `/proc/meminfo` is also the GPU headroom.

Both nodes hold half of the model. Rank 0 holds more:

| Part | Size | Where it comes from |
|---|---|---|
| Engine, head | about 108 GiB (`MemAvailable` 113,838 MiB before the start, 2,941 MiB after) | gate of 1.1.0, [idle-f.log](../runs/2026-10-21/gate-1.1.0/data/idle-f.log) |
| Engine, worker | about 106 GiB (112,687 MiB before, 3,669 MiB after) | the same file |
| Vision tower, rank 0 only (`VISION=1`) | 0.95 GiB | engine log, [run2-engine-vision.txt](../runs/2026-10-20/oom/run2-engine-vision.txt) |
| HTTP server, tokenizer, grammar compiler, image decoder, rank 0 only | not measured one by one. The head idles 0.7 GiB lower than the worker. | gate of 1.1.0 |
| Desktop session, browsers, routers, dashboards, `docker build`, `docker load` | depends on your node | |

Any program that you run on the head takes memory from the margin. In the gate, the head fell to 1,450 MiB with 4 long prompts at the same time. earlyoom stops the engine at 1,024 MiB. The margin is 426 MiB. A browser on the head can use more than that.

### The numbers of one gate run

All values are `MemAvailable` in MiB, from the 1.1.0 gate ([gate report](../runs/2026-10-21/gate-1.1.0/gate-report.md)). One pass.

| Point | Head | Worker |
|---|---|---|
| Before the start | 113,838 | 112,687 |
| After the start (idle) | 2,941 | 3,669 |
| Lowest, 4 streams at the same time (prompts of 20K, 50K, 100K and 200K tokens) | 1,450 | 2,507 |
| New requests wait (`MEM_FLOOR_GIB` 1.5 plus `TF_DS_ADMIT_GIB` 1; only while a stream is live) | 2,560 | 2,560 |
| Engine floor `MEM_FLOOR_GIB` 1.5 | 1,536 | 1,536 |
| earlyoom SIGTERM / SIGKILL | 1,024 / 512 | 1,024 / 512 |

Real agent traffic can be heavier than this load. The longest prompt of the gate has 200K tokens. A stop needs only 2 to 4 long prompts (more than 20K tokens each) that start in the same second while the head idles under about 3.2 GiB.

### What `MemAvailable` leaves out

The kernel keeps a reserve and does not count it as available. `scripts/host_tuning.sh` makes this reserve larger (`vm.min_free_kbytes` 1 GiB, `vm.watermark_scale_factor` 25) to prevent compaction stalls. earlyoom reads `MemAvailable`, so it sees less memory than `MemFree` plus the page cache.

On our pair, the kernel formula gives the same value as `/proc/meminfo` (to the kB) on both nodes. This read was made with the earlier value `vm.watermark_scale_factor` 50, with these parts ([data](../runs/2026-10-21/INDEX.md#kernel-memory-reserve-of-the-pair-read-only-check)):

| Part | Size |
|---|---|
| Reserve that is subtracted from `MemFree` (high watermarks 2.22 GiB plus the lowmem reserve 0.47 GiB) | 2.69 GiB |
| Cut from the page cache (the sum of the low watermarks), when the page cache is over 3.22 GiB | 1.61 GiB |
| Cut from the reclaimable slab (half of it, at most the sum of the low watermarks) | 0.25 GiB (head), 0.42 GiB (worker) |
| Hidden in all | 4.55 GiB (head), 4.71 GiB (worker) |

With the kernel defaults, the same formula gives 0.79 GiB for the first part. Scale 50 added 1.90 GiB to it.

`host_tuning.sh` now sets `vm.watermark_scale_factor` 25. By the formula, with the page cache of the loaded engine, this lowers the reserve by about 0.62 GiB, so `MemAvailable` reads higher for the same real state. In the watermark A/B the head low point rose by 0.47 GiB, with no rise in slow prefill chunks in 2 complete runs per side. The 3-run rule was not met: the third run of each side stopped early at the head memory rule ([runs/2026-10-21](../runs/2026-10-21/INDEX.md#watermark-ab-vmwatermark_scale_factor-50-against-25), [runs/2026-10-22](../runs/2026-10-22/INDEX.md)). It adds no free memory: earlyoom and the engine floor now act at less real free memory.

Keep `vm.min_free_kbytes` at 1 GiB. With 512 MiB the head showed compaction stalls and swap (variant V3 of the A/B). These values prevent the stalls of [runs/2026-10-18](../runs/2026-10-18/INDEX.md). Do not change them without the stall test.

**Upgrade:** a node that ran the earlier `host_tuning.sh` still has 50. The preflight memory check fails with "memory settings differ (vm.watermark_scale_factor)" until you run `scripts/host_tuning.sh` again on the head. It is safe to repeat. Restart the server after it.

## How earlyoom works here

`scripts/host_tuning.sh` starts earlyoom with these arguments (in `/etc/default/earlyoom`):

```text
-r 3600 -p -M 1048576,524288 -s 100,100 --prefer ^(tensorfold|tf-dsv41) --avoid ^(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager|gdm|Xorg|nvidia)
```

- `-M 1048576,524288`: earlyoom sends SIGTERM when `MemAvailable` is under 1 GiB, and SIGKILL when it is under 512 MiB. The values are in KiB.
- `--prefer ^(tensorfold|tf-dsv41)`: earlyoom stops the engine process first. The pattern matches rank 0 of the Zig engine (`tensorfold-native`) and of the Python engine, and rank 1 of the Zig engine (`tf-dsv41-lanes`). The containers also run with `--oom-score-adj 1000`, so the kernel OOM killer also picks them first.
- `--avoid ...`: earlyoom does not stop ssh, Docker, the network manager or the desktop. The node stays reachable.
- `-s 100,100`: the swap does not change this. The engine runs in `tensorfold.slice`, which has no swap.
- `-r 3600`: earlyoom prints a status line each hour.

Earlyoom runs on each node. Read the journal on the node that stopped (on the head, then on the worker with ssh):

```bash
journalctl -u earlyoom --since "24 hours ago" --no-pager | grep -a -E "mem avail|sending"
```

A stop shows two lines:

```text
earlyoom[<pid>]: mem avail: <MiB> of <MiB> MiB (<n>%), swap free: ...
earlyoom[<pid>]: sending SIGTERM to process <pid> uid <n> "tensorfold": oom_score <n>, VmRSS <MiB> MiB, cmdline ...
```

The first number is the `MemAvailable` that earlyoom read. It sampled it about once each second. The lowest point can be a little lower than the logged value. The stops in the traces of this repo were at 901 and 1,009 MiB ([OPERATIONS.md](OPERATIONS.md#when-the-lane-breaks)).

Rank 1 shows `NCCL error 4` after the head stops. This is a result of the stop, not a second fault.

## "terminate called without an active exception"

The C++ runtime prints this line in the log of rank 0 (`docker logs tfv41-r0`) when the process ends with no exception to report. The line alone does not tell you the cause. An uncaught exception, a call to `std::terminate` or a thread that is not joined can also print it.

Before you call it a memory stop, find a stop event within a few seconds of the line:

- an earlyoom `sending SIGTERM` line in the journal (`journalctl -u earlyoom`), or
- a `docker stop` of the container (yours, or from a script).

If you find one, rank 0 got SIGTERM while it processed a prompt, and the container exits with code 134. All cases in the traces of this repo were of this type. If you find no stop event, treat it as an engine crash: open an issue with the output of `scripts/oom_report.sh` and the full log of rank 0.

For a SIGTERM, engine fork [PR #34](https://github.com/Elevatormusic/TensorFold/pull/34) makes rank 0 end with exit code 0 and without this line. Release 1.1.1 (the pinned image `cb4ceaff2b8c`) has that change. Release 1.1.0 (`730d41abd8f4`) and older still print the line. The memory cause stays: earlyoom still stops the engine. Use the journal to find out why.

## Settings for a head with little memory

Change one setting at a time. Restart after each change.

| Setting | Effect | Measured? |
|---|---|---|
| `VISION=0` | The vision tower does not load on rank 0, and image input is off. | +0.95 GiB at idle (engine log). No run of the whole recipe with `VISION=0` is in this repo. |
| `PREFILL_CHUNK=1024` | A smaller prefill chunk gives a lower peak. The estimate for the chunk transient is 0.84 GiB at 2048 and about 0.42 GiB at 1024. Prefill is slower with a smaller chunk. | Not measured on 1.1.0, for memory or for prefill speed. |
| `PARALLEL=2` | Fewer streams at the same time. Each request still gets the full context. | Only on a validation image, with images: [OPERATIONS.md](OPERATIONS.md#when-the-lane-breaks). |
| Send images one request at a time | Rank 0 decodes the images of a request before the request waits for its turn. | Four requests with 8 images each at the same time: lowest head `MemAvailable` 1,379 MiB on a new engine ([OPERATIONS.md](OPERATIONS.md#when-the-lane-breaks)). |

Do not do these:

- Do not raise `MEM_FLOOR_GIB` above 2.0. With 2.0, an 8K prompt failed with CUDA out of memory, and that breaks the lane ([runs/2026-10-18](../runs/2026-10-18/INDEX.md)).
- Do not turn off earlyoom or lower its limit. Without it, a node that runs out of memory can stop answering ssh.
- Do not use `PREFLIGHT=0` to get past the memory check. It also turns off the check of the earlyoom settings.

`PREFILL_CHUNK` and `PARALLEL` keep their defaults (2048 and 4). We have not measured 1024 and 2 on 1.1.0 for memory. Tell us what you see.

## The checks in the scripts

`scripts/preflight.sh` checks `MemAvailable` before the start:

| Node | Needed (GiB) | Where the value comes from |
|---|---|---|
| Head | 111 (110 with `VISION=0`) | `MemAvailable` before the start in the gate: 113,838 MiB = 111.2 GiB. The vision tower is 0.95 GiB. |
| Worker | 109 | The gate value is 110.0 GiB. The worker idles 0.7 GiB higher than the head, so the value is 1 GiB lower. |

The values are for the default `CONTEXT` and `PARALLEL`. A node with a desktop session or a browser can be under these values. The message names the fix: close the programs, set `VISION=0`, and run `scripts/host_tuning.sh`. A node under the value can start and then idle with no margin, or stop while the weights load. The old limit was 100 GiB. The README said "about 100 GB".

After the API answers, `./start.sh` and `./status.sh` read `MemAvailable` on the head. Under 2.5 GiB, they print a `[warn]` line with the same steps. The check never stops a start. The gate idle value was 2.9 GiB.

## Report a stop

Run this on the head, after the stop:

```bash
scripts/oom_report.sh
```

It writes `logs/oom-report-<UTC time>.txt` and prints the path. Attach the file to the issue form ([bug report](../.github/ISSUE_TEMPLATE/bug_report.yml) or [performance report](../.github/ISSUE_TEMPLATE/performance_report.yml)). The script only reads. For each node, it collects:

- `free -m` and these lines of `/proc/meminfo`: `MemAvailable`, `AnonPages`, `Cached`, `Shmem`;
- the earlyoom lines of the journal of the last 24 hours;
- `vm.min_free_kbytes`, `vm.watermark_scale_factor` and the sum of the zone watermarks;
- the engine image (tag, id) and the TensorFold commit of `.env`;
- the `.env` settings `PARALLEL`, `CONTEXT`, `SPILL_GIB`, `PREFILL_CHUNK`, `VISION` and a few more, from a fixed list;
- the last 50 lines of the engine log that mention memory, earlyoom or terminate;
- the NVIDIA driver version and the `NVRM` lines of the kernel log.

The script never reads or prints a key. It removes token-like text, home paths and the host names of the two nodes from the file. Read the file before you post it.

To see how the margin changes while a load runs, run this on the head. It shows the head first, then the worker:

```bash
. ./.env; watch -n 1 "grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes $WORKER_SSH grep MemAvailable /proc/meminfo"
```
