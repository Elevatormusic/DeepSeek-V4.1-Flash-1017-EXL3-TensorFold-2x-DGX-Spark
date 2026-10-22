# Runs of 22 October 2026 (UTC)

## Watermark A/B, third window (B3 and V1c)

Same pair, image and settings as the watermark A/B of [runs/2026-10-21](../2026-10-21/INDEX.md#watermark-ab-vmwatermark_scale_factor-50-against-25) (see [RESULT.md](../2026-10-21/watermark-ab/RESULT.md) for the method). The window ran from 00:17 to 00:26 UTC (the node clock), so its files are in this day folder. It was a third run of each side: B3 (`vm.min_free_kbytes` 1 GiB, `vm.watermark_scale_factor` 50) and V1c (1 GiB, 25).

| Run | Idle head `MemAvailable` | Chunks run | Slow chunks (over 1.3 s) | Max chunk | Stop |
|---|---|---|---|---|---|
| B3 (baseline) | 2,843 MiB | 164 of 248 | 2 | 1.51 s | the head `MemAvailable` fell under 1.25 GiB (1,225 MiB) in pass 2 |
| V1c | 3,225 MiB | 21 of 248 | 1 | 1.47 s | the head `MemAvailable` fell under 1.25 GiB (1,256 MiB) in pass 1 |

- The memory rule stopped both runs, the baseline too. Each side still has 2 complete runs in `runs/2026-10-21/`, so the 3-run rule of CONTRIBUTING.md is not met. Neither run shows a stall rise for V1.
- The window began with less free memory on the head than the first two (idle 2,843 MiB for B3, against 3,467 and 3,051 MiB for B and B2).
- The values were read back as 1,048,576 and 50 on both nodes at the end. No earlyoom line.

| File | Content |
|---|---|
| `watermark-ab/B3/`, `watermark-ab/V1c/` | summary.json, 1 s memory samples of both nodes, engine logs with time stamps, the pass logs, the time windows and the stop reason of each run |
| `watermark-ab/window3-orchestrate.log` | the console output of the window |

The tables of all nine runs come from `runs/2026-10-21/watermark-ab/summarize.py`.
