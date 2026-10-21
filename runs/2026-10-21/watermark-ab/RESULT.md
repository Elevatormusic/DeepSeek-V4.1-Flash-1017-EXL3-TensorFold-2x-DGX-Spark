# Watermark A/B: `vm.watermark_scale_factor` 50 against 25 (and two other settings)

Date: 2026-10-21 (UTC, node clock). Same pair of DGX Spark as the other runs of this day. This run is the evidence for the change of `vm.watermark_scale_factor` from 50 to 25 in `scripts/host_tuning.sh`.

## Short result

- **V1 (`vm.min_free_kbytes` 1 GiB, `vm.watermark_scale_factor` 25) is not worse than the baseline B (1 GiB, 50) on stalls.** In two stall tests of 248 prefill chunks each, V1 has 1 and 0 slow chunks (over 1.3 s). B has 4 and 0. The baseline itself varies from 0 to 4 slow chunks, so this run shows "not worse", not "better".
- **V1 raises the head `MemAvailable` by about 0.47 GiB** (477 MiB, lowest value in the long 4-stream run; mean of two runs against the mean of two B runs). The kernel formula gives +934 MiB (0.91 GiB). The measured gain is about half of it, because the free memory of the node differs between two starts by about 0.4 GiB.
- **This is a lower reserve, not more free memory.** The engine holds the same memory in every run. With V1 the kernel keeps about 0.9 GiB less in reserve, so `MemAvailable` reads higher for the same real state. earlyoom (1 GiB limit) and the engine floor (1.5 GiB) now act at less real free memory. This lowers false stops. It does not prevent a real out-of-memory case.
- **V3 (512 MiB, 25) is not safe.** It gives +1.6 GiB, but the head shows memory pressure: 4,195 compaction stalls in the stall test (B: 2 and 0), 25,437 swap-in pages and 24,018 major faults in the long run (B: 72 and 109), and a chunk of 2.54 s.
- **V2 (512 MiB, 50) is irregular.** Its first run reached 1,192 MiB on the head and the watchdog stopped it (limit 1,280 MiB). Its repeat reached 1,982 MiB. It gives the same measured gain as V1 (+466 MiB) with a less regular low point.
- The median chunk time is 0.94 s in every run. Prefill tok/s at 8K, 32K, 64K and 128K is within 2 percent of B. No request failed. earlyoom logged no line.

## Settings

| Name | `vm.min_free_kbytes` | `vm.watermark_scale_factor` | Sum of zone reserve (`totalreserve`) | Sum of low watermarks | Gain by the kernel formula |
|---|---|---|---|---|---|
| B (the `host_tuning.sh` value before this change) | 1,048,576 (1 GiB) | 50 | 2,754 MiB | 1,647 MiB | 0 |
| V1 | 1,048,576 (1 GiB) | 25 | 2,131 MiB | 1,336 MiB | +934 MiB |
| V2 | 524,288 (512 MiB) | 50 | 2,242 MiB | 1,135 MiB | +1,024 MiB |
| V3 | 524,288 (512 MiB) | 25 | 1,619 MiB | 824 MiB | +1,958 MiB |

The kernel subtracts the `totalreserve` from `MemFree` and cuts the page cache by the sum of the low watermarks (see the kernel-reserve check in [INDEX.md](../INDEX.md)). Gain = drop of the `totalreserve` + drop of the low watermarks. The values are read from `/proc/zoneinfo` (`zone-head.txt`, `zone-worker.txt` in each run folder).

## Method

Equal in all runs (the A/B rule of CONTRIBUTING.md):

| Setting | Value |
|---|---|
| Engine image | `tensorfold-dsv41:cb4ceaff2b8c` (release 1.1.1 candidate), RDMA GID index auto-detected |
| `TF_DS_PREFILL_CHUNK` | 2048 |
| `SPILL_GIB` | 64 (a new spill folder for each run) |
| `CONTEXT` / `PARALLEL` / `MTP_DRAFTS` | 1048576 / 4 / 5 |
| `MEM_FLOOR_GIB` | 1.5 |
| Host tuning | `scripts/host_tuning.sh` applied on both nodes (swappiness 10, compaction_proactiveness 0, extfrag_threshold 500, THP never, earlyoom as in the script). Only the two `vm` values below changed, at runtime with `sysctl -w`. |
| Prompts | Random-word prompts of `kit_bench.py prefill`, a new seed each time: no prefix cache hit (cold). |
| Runs per side | 2 (B and B2; V1 and V1r; V2 stopped and V2r; V3 once). This is under the 3 runs of the A/B rule: see "Limits". |

Steps in each run: set the two values on both nodes; start the engine fresh; wait 30 s; measure idle `MemAvailable` for 20 s; run the stall test twice; run the long 4-stream run once.

- **Stall test:** two passes of `runs/2026-10-18/scripts/pf.sh` (`scripts/pfrun.sh`): an 8K warm prompt, 8K twice, then 32K, 64K and 128K. This is 124 chunks per pass, 248 for two passes. Chunk times come from `TF_DS_CHUNK_LOG=1` on rank 0 (`rank-r0.log`).
- **Long run:** 4 streams at the same time with prompts of 20K, 50K, 100K and 200K tokens, 1,500 new tokens each (`capdrive.py long`, seed 7). The token hashes equal those of the 1.1.0 gate in all runs that finished.
- **Samples at 1 s on both nodes** (`mem-head.csv`, `mem-worker.csv`): `MemAvailable`, `MemFree`, `Cached`, `/proc/vmstat` counters (`compact_*`, `allocstall*`, `pgscan*`, `pgsteal*`, `pswpin`, `pswpout`, `pgmajfault`) and the CPU time of kswapd0 and kcompactd0.
- **Watchdog** (`scripts/watch.sh`): it stops the test containers when the head `MemAvailable` is under 1.25 GiB or when earlyoom logs a stop. It never touches other containers.
- **Order:** B, V1, V2, V3, B2 in the first window (20:23 to 21:15 UTC), then V2r and V1r in the second window (22:16 to 22:43 UTC). Both locks of the pair were held, so no other job ran.
- The two `vm` values were read back after each window. Both nodes ended at 1,048,576 and 50.

## Results

`summary-table.txt` has this table, made by `summarize.py` from the `summary.json` of each run. MemAvailable is in MiB. "Slow" is a chunk over 1.3 s. Compaction columns are `compact_stall/compact_fail/allocstall` on the head.

| Run | Idle MemAvailable head / worker | Lowest head, stall test | Lowest head / worker, long run | Chunks (248): median, p95, max, slow | Long run: max chunk, slow | Prefill tok/s 8K / 32K / 64K / 128K | Head compaction, stall test | Head compaction, long run | Errors |
|---|---|---|---|---|---|---|---|---|---|
| B | 3,467 / 3,469 | 1,451 | 1,347 / 2,698 | 0.94 s, 1.03 s, 2.04 s, 4 | 1.23 s, 0 | 2,197 / 2,268 / 2,178 / 2,093 | 2/1/1 | 6/0/0 | 0 |
| B2 | 3,051 / 3,707 | 1,867 | 1,686 / 2,520 | 0.94 s, 1.03 s, 1.18 s, 0 | 1.14 s, 0 | 2,212 / 2,247 / 2,205 / 2,110 | 0/0/1,928 | 0/0/0 | 0 |
| V1 | 3,835 / 4,103 | 2,340 | 2,228 / 2,636 | 0.94 s, 1.04 s, 1.31 s, 1 | 1.19 s, 0 | 2,200 / 2,254 / 2,182 / 2,110 | 120/15/15 | 497/153/152 | 0 |
| V1r | 3,702 / 4,082 | 1,926 | 1,759 / 3,044 | 0.94 s, 1.04 s, 1.30 s, 0 | 1.39 s, 1 | 2,196 / 2,228 / 2,195 / 2,108 | 0/0/0 | 0/0/0 | 0 |
| V2 (stopped) | 3,598 / 4,211 | 1,192 | no long run | 151 chunks: 0.93 s, 1.04 s, 1.19 s, 0 | no long run | 2,174 / 2,257 / 2,203 / 2,106 | 1/1/1 | no long run | stopped by the watchdog |
| V2r | 3,872 / 3,969 | 2,103 | 1,982 / 2,999 | 0.94 s, 1.03 s, 1.72 s, 1 | 1.15 s, 0 | 2,150 / 2,178 / 2,204 / 2,113 | 46/2/2 | 4/1/1 | 0 |
| V3 | 4,564 / 4,709 | 3,240 | 3,114 / 3,327 | 0.94 s, 1.06 s, 2.54 s, 4 | 2.19 s, 3 | 2,148 / 2,032 / 2,183 / 2,112 | 4,195/443/412 | 1,927/1,189/713 | 0 |

Per setting (mean of the runs; gain against the mean of B and B2):

| Setting | Idle head MemAvailable | Lowest head MemAvailable, long run | Slow chunks | Max chunk |
|---|---|---|---|---|
| B (B, B2) | 3,259 | 1,516 | 4 and 0 of 248 | 2.04 s |
| V1 (V1, V1r) | 3,768 (+510) | 1,994 (+477) | 1 and 0 of 248 | 1.31 s |
| V2 (V2 stopped, V2r) | 3,735 (+476) | 1,982 (+466), V2r only | 0 of 151 and 1 of 248 | 1.72 s |
| V3 | 4,564 (+1,305) | 3,114 (+1,598) | 4 of 248 | 2.54 s |

Extra head numbers for V3 (long run): swap-in 25,437 pages, swap-out 26,960 pages, major faults 24,018. For B the same numbers are 72, 1 and 109 (`B/summary.json`, `V3/summary.json`).

## Findings

1. Chunk times do not separate B, V1 and V2. The median is 0.94 s everywhere. The slow chunks are rare and change from run to run: B gave 4 in one run and 0 in the next.
2. The `/proc/vmstat` compaction counters change a lot between two runs of the same setting (head `allocstall` 1 in B and 1,928 in B2; worker `compact_stall` 0 in V1 and 807 in V1r). They do not match the chunk times. Use the chunk times, the swap-in and the major faults as the stall signal.
3. V3 is the only setting with a clear sign of pressure in the swap and major-fault numbers, and it has the largest chunk (2.54 s).
4. The head `MemAvailable` moves by about 1 GiB from second to second during a prefill (see `mem-head.csv`), also at B. A single low value (V2: 1,192 MiB) can be an excursion, so the means of two runs are the fair figure.

## Limits

- Two runs per setting (V3: one). The A/B rule of CONTRIBUTING.md asks for at least 3 per side. This run makes no speed claim. It makes two claims: V1 is not worse than B on slow chunks in these runs, and V1 reads about 0.47 GiB higher in `MemAvailable`.
- One engine image, one model view, THP off, one pass of the long run for each setting.
- The test ran for about 1 hour 20 minutes in total. It does not cover a slow drift of the page cache over days.
- The gain is a change of the reading. The real free memory is the same.

## Files

| File | Content |
|---|---|
| `summarize.py`, `summary-table.txt` | the script and its output (the tables above) |
| `<run>/summary.json` | the analysis of one run (`scripts/analyze.py`): chunk statistics, memory minima, `/proc/vmstat` deltas, log counts |
| `<run>/mem-head.csv`, `<run>/mem-worker.csv` | the 1 s samples of each node |
| `<run>/rank-r0.log`, `<run>/rank-r1.log` | the engine logs with a time stamp for each line (chunk times) |
| `<run>/pf-p1*.json`, `<run>/pf-p2*.json`, `<run>/pf-p*.log` | `kit_bench.py prefill` results of the two stall-test passes |
| `<run>/long4.log` | the long run (`capdrive.py long`): TTFT, decode tok/s and token hash of each stream |
| `<run>/zone-head.txt`, `<run>/zone-worker.txt` | sums of the zone watermarks and the reserve |
| `<run>/notes.txt`, `<run>/windows.tsv`, `<run>/trip.txt`, `<run>/earlyoom-*.txt`, `<run>/start.log` | notes, the time windows, the watchdog result, the earlyoom lines (empty in all runs), the engine start |
| `window1-orchestrate.log`, `window2-orchestrate.log`, `env.txt` | the console output of the two windows and the node state |
| `scripts/` | the scripts as they ran (home path and host names replaced) |
