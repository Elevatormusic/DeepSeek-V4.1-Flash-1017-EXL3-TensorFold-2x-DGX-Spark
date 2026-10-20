# Optional BF16 uncensored weights

This option adds the BF16 matrices from the pinned dealignai pack to a new model view.
The normal `./start.sh` setup keeps its current model and engine.
Use this option only with the BF16 engine change described below.

## Source and scope

The source is [dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED/tree/89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c)
at revision `89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c`.
The base is the recipe's LibertAIDAI EXL3 pack at revision
`afc7aa0aab98ee735c0f54e0a122019af1ac08f1`.
The [tensor manifest](../scripts/uncensored_tensors.json) gives the source shard,
shape and payload SHA256 for each of the 54 matrices:

- Layers 10 through 36: `attn.wo_b.weight`, shape `[5120, 8192]`, BF16.
- Layers 10 through 36: `ffn.shared_experts.w2.weight`, shape `[5120, 2304]`, BF16.

The other weights and the Engram files come from the normal base view.
This is a base view plus the specified matrices.
It is not a full copy or a full byte comparison of the source pack.
This repo publishes no weights.
The source weights use the DeepSeek V4.1 Flash 1017 licence.
See [the source licence](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE).

An 8 bpw override test produced a repetition loop; no decode-speed benefit was established.
BF16 is the documented option.
This statement does not claim a paired decode benchmark or that all BF16 answers are stable.

## Engine requirement

The engine needs BF16 output projections, separate shared experts, override-group
replacement and an in-place TP sum for large partial results.
The default engine revision in `.env.example` does not have these changes.
The separate engine change is [TensorFold PR #8](https://github.com/Elevatormusic/TensorFold/pull/8),
at commit `6c53993796efddcb0a80fafc49fa2d952544eda2`.
That pull request is open and is not merged.
Its exact head has not completed GPU qualification.
The example settings pin that commit and image tag `tensorfold-dsv41:6c53993796ef`.
Do not treat results from an earlier local engine as GPU qualification of this head.

## Prepare the new view

Run each command on the head node, in the recipe folder.
Complete the base setup first, including `scripts/download.sh` and the chat template step.
Run `scripts/download.sh verify` to check the base files before you make hard links.

1. Download only the required source shards at the pinned revision.
   These shards are large. Keep the download on a disk with sufficient free space.
   The preparation script needs local files and uses only the Python standard library.

   ```bash
   hf download dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED \
     --revision 89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c \
     --include 'model-0001[2-9]-of-00043.safetensors' \
       'model-0002[0-9]-of-00043.safetensors' \
       'model-0003[0-8]-of-00043.safetensors' \
     --local-dir "$HOME/models/dealign-source-89af9bf1"
   ```

2. Make the new view. Replace the base path if your `.env` uses a different path.
   Put the output on the same disk as the base view, because it uses hard links.
   The output folder must not exist.

   ```bash
   python3 scripts/prepare_uncensored.py \
     --source-dir "$HOME/models/dealign-source-89af9bf1" \
     --base-view "$HOME/models/deepseek-v41-1017-exl3-2.9bpw" \
     --output-view "$HOME/models/deepseek-v41-1017-dealign-bf16"
   ```

   The script checks dtype, shape, offsets and the SHA256 of each tensor payload.
   It writes `model/override-dealign-bf16.safetensors`, `override-receipt.json`
   and `override.sha256` in the new view.
   The receipt records the SHA256 of the new complete file.
   A different safetensors header can change the complete file hash while the tensor bytes stay identical.
   Do not modify the linked base files through either view.

3. Check the base hashes on the worker with `scripts/download.sh worker`.
   Make a new view on the worker with hard links to its own base files.
   Run this block on the worker. Change the base path if necessary.
   Both folders must be on the same disk. The output folder must not exist.

   ```bash
   BASE_VIEW=$HOME/models/deepseek-v41-1017-exl3-2.9bpw
   BF16_VIEW=$HOME/models/deepseek-v41-1017-dealign-bf16
   test ! -e "$BF16_VIEW" && mkdir "$BF16_VIEW" && \
     cp -al "$BASE_VIEW/model" "$BASE_VIEW/engram" "$BF16_VIEW/"
   ```

   Then run these commands on the head. Transfer only the new override and
   its receipt, not a second copy of all base weights. Replace `<worker>` and
   the absolute destination paths with the worker settings from your `.env`.

   ```bash
   BF16_VIEW=$HOME/models/deepseek-v41-1017-dealign-bf16
   rsync -a --partial-dir=.rsync-partial \
     "$BF16_VIEW/model/override-dealign-bf16.safetensors" \
     '<worker>:/absolute/path/to/deepseek-v41-1017-dealign-bf16/model/'
   rsync -a "$BF16_VIEW/override-receipt.json" "$BF16_VIEW/override.sha256" \
     '<worker>:/absolute/path/to/deepseek-v41-1017-dealign-bf16/'
   ```

   Check the base hashes in the new worker view with the recipe's
   `runs/2026-10-17/weights.sha256` file.
   In the new view on each node, run `sha256sum -c override.sha256`.
   The launcher also compares the full override hash on both nodes before start.

## Settings and start

Copy `.env.uncensored.example` to `.env.uncensored`.
This file reads your normal `.env` for network, authentication and host settings.
Keep the pinned BF16 engine revision and matching image in the new file.
Check that `MODEL_VIEW` is the new view on both nodes.
The paths must have no spaces or shell characters.

```bash
cp .env.uncensored.example .env.uncensored
bash scripts/uncensored.sh build
```

Stop any running model with its normal stop command first.
Disable any external service that automatically restarts it.
Run only one model server at a time.
The launcher refuses a start if either node has a running GPU container.
It does not stop other containers for you.
Do not run `docker exec` in an active rank container.

```bash
bash scripts/uncensored.sh start
bash scripts/uncensored.sh status
bash scripts/uncensored.sh verify
```

The launcher uses the normal rank container names and standard preflight checks.
It selects `.env.uncensored` through `TF_ENV_FILE`.
The option uses a 1,024-token prefill chunk and a 1.5 GiB free-memory floor.
Keep this floor at or above 1.5 GiB.
The 1,048,576-token context setting is a configured limit.
It is not evidence of a completed 1M-token request with these weights.
NVIDIA allocation warnings were observed while checks completed without an OOM kill.
These checks do not prove that every request fits in memory.
The [saved service summary](../runs/2026-10-19/uncensored-bf16/service-check.json)
records the tested local source, settings, and remaining limits. It is not a
qualification receipt for the new engine PR head or this new preparation wrapper.

Each full override SHA256 has its own cache folder under `CACHE_DIR`.
Keep `${UNCENSORED_CACHE_ID:-unverified}` at the end of the cache path in the settings file.
Do not reuse a base rank cache or saved prompt state for different weights.
The cache key does not hash all weight bytes by itself.

To stop this option and return to the base setup:

```bash
bash scripts/uncensored.sh stop
./start.sh
./verify.sh
```

The stop command saves both rank logs through the normal `stop.sh`.
If the view receipt is missing, use `./stop.sh` to stop the standard rank containers.

## Offline checks

`tests/test_prepare_uncensored.py` uses small local tensors.
It checks payload integrity, malformed headers, failed preparation cleanup,
new-view isolation and damage detection.
It does not test model quality or GPU execution.
Run the full offline suite with `tests/run_offline.sh`.
