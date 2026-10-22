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

## Zig engine and the 2.8 bpw pack

The Zig engine (fork branch `dsv41-zig-dev`, commit `d58d25ccd6dbd624814cc2ccc848f8e5ba84580c`) with the EXL3 2.8 bpw pack (`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` @ `85939659`) is the default of this recipe. The Python engine 1.1.1 with the 2.9 bpw pack is the rollback. All runs below are from one pair of DGX Spark.

| File | Content |
|---|---|
| [`zig-28/zig-release-faults.md`](zig-28/zig-release-faults.md) | the release fault campaign of the Zig engine: decode and prefill against the Python engine 1.1.1 on the same pack, the agent tests (505 requests), the needle test (1,038,664 tokens), faults F1 and F3 and their fixes, the exit-code contract |
| [`zig-28/pack-2.8bpw-test.md`](zig-28/pack-2.8bpw-test.md) | the 2.8 bpw pack against the 2.9 bpw pack on the Python engine: memory at idle and at the lowest point, decode, MTP acceptance, 530 agent requests, the plain tensors with other dtypes |
| [`zig-28/weights-2.8bpw.sha256`](zig-28/weights-2.8bpw.sha256) | the sha256 of the 47 files of the 2.8 bpw pack (46 view files and the LICENSE) and of the four Engram files (`scripts/download.sh` checks every file against this list) |

Two checks were made when the recipe scripts were written (22 October 2026):

- **Token map.** `tools/dsv41-zig/inputs/make_token_map.py` of the fork, run on the tokenizer of the 1017 checkpoint, made a file with the sha256 `de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28`. This is the sha256 of the file that the tested engine served. `scripts/zig_inputs.sh token-map` compares against this value and warns when it differs.
- **Kernel kit.** The `MANIFEST` of commit `4996441a82d95b9a7853b01a1255e1dafc926387` of `bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10` has the sha256 `8c6dcb712a7557a8aeb794d68daf85b07befae7bb1bdfa470223e957718f2845`, the same as the `MANIFEST` of the kit that the tested nodes loaded. It lists 223 files; the repo folder `kit/` holds those 223 files and `MANIFEST`. Only `MANIFEST` was downloaded for this check. The other kit files were not downloaded for it.

The raw logs of the campaign (the engine logs and the `out/` folders of the runs) stay on the test nodes. They are not in this folder. The two reports above hold the tables that this recipe quotes.
