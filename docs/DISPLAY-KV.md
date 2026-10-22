# Display-backed KV experiment

This is an opt-in display-only variant. The ordinary `start.sh`, `.env` and
base image stay unchanged. Full two-node hardware validation is still pending.
Do not use the offline test results as proof of safe serving or better performance.

## Scope and risks

The installer changes only the pinned TensorFold `model.py`, `engine.py` and
`multi.py`, and adds `display_kv.py`. It checks the source revision, file hashes,
replacement anchors and generated Python before it writes files. It does not
install tool-output guards, memory-policy changes, metrics or a replacement spill tier.

The private launch bundle uses native packed KV, a 1792 MiB display reservation
per node, prefill chunk 512 and `SPILL_GIB=0`. Spill is disabled because this
stage does not include mapped-storage-safe spill. Ordinary launch defaults do
not change. The display pool and each request's context limit are separate.

Display mapping and runtime DRM changes can fail or affect the local display.
Keep both nodes idle before a cutover. Never stop GDM or another display manager.
Never use `docker exec` in serving `tfv41-r0` or `tfv41-r1` containers. Run only
one model server at a time. Keep `BIND_HOST=127.0.0.1` unless network access and
an API key are explicitly configured. Do not change DRM while any GPU or DRM
client is active. The launcher refuses active ranks and incompatible clients.

## Opt-in use

First complete the ordinary recipe setup, including network and host tuning.
The launcher builds from `IMAGE` in `.env`. It checks the pristine source of the 1.0.0 engine `07aa02585f64`, not the engine that `docker/build.sh` builds now (`cb4ceaff2b8c`). For this experiment, build the 1.0.0 base on the head, copy it to the worker, and set `IMAGE=tensorfold-dsv41:07aa02585f64` in `.env` (see the return steps below). Run these commands from the repository root:

```bash
docker build --build-arg TF_COMMIT=07aa02585f64 -t tensorfold-dsv41:07aa02585f64 docker
```

```bash
. ./.env; docker save tensorfold-dsv41:07aa02585f64 | ssh -o BatchMode=yes "$WORKER_SSH" docker load
```

The 1.0.0 engine ignores `DROP_UNOFFERED_TOOL_CALLS`. Use the head node and the repository root. Inspect state without a cutover:

```bash
./start-display-kv.sh --check
```

When the current work has finished, stop both ranks deliberately:

```bash
./stop.sh
./start-display-kv.sh --build-only
./start-display-kv.sh
./verify.sh
```

`--build-only` builds and copies the derived image but does not change DRM or
start a server. `--build` builds, copies and starts in one command. A later
launch uses the saved image and checks its ID on both nodes. The head can ask
for sudo. The worker needs passwordless sudo for runtime DRM preparation.
The launcher probes the full reservation on both nodes before it starts the
private bundle. It does not stop a running server for you.

Private bundles and logs are under `experiments/display-kv/bundle-launch-*`.
They contain a configuration snapshot. Do not publish them or their secrets.
If startup fails, the launcher attempts to restore DRM defaults. It does not
start an ordinary-memory fallback. Inspect `--check` if restore fails.

To return to the ordinary engine after all GPU and DRM clients have stopped, first set `IMAGE=tensorfold-dsv41:cb4ceaff2b8c` in `.env` again (the ordinary engine; `TF_COMMIT` in `.env` should still be `cb4ceaff2b8c`). Then run:

```bash
./stop.sh
./start-display-kv.sh --restore-drm
./start.sh
```

## Measured cost (tester report, not repeated)

A tester measured the combined display build of pull request #6 on one pair of
Spark nodes. That build had display KV and also a patched SSD spill, a tool-output
guard and a memory guard, which this experiment does not include. The maintainers
did not repeat the runs. The saved report, with setup and caveats, is
[`runs/2026-10-19/display-kv-tester-ab.md`](../runs/2026-10-19/display-kv-tester-ab.md).
Credit: Sora39831 for the runs, and hjasond for an early report.

| Measure | Stock lane | Display lane |
|---|---|---|
| Cold prefill, 32K / 128K / about 1M tokens | 2,124 / 1,923 / 1,299 tok/s | 1,680 / 1,534 / 1,092 tok/s |
| Change in prefill tok/s | | -20.9% / -20.2% / -15.9% |
| Decode, one stream | 51 to 62 tok/s | 48 to 59 tok/s |
| Prefix cache hit, 54K prompt, TTFT | 0.73 s | 0.46 s |
| Minimum host MemAvailable, 1M prefill | 1.96 GiB | 2.01 GiB |

- The display lane gives about the same KV positions (2,109,440 against 2,097,152) and no more free host memory. It moves the KV out of host RAM.
- The prefill gap is not isolated. The display bundle fixes the prefill chunk at 512 (it ignores `PREFILL_CHUNK` in `.env`) and the stock lane used 2048. The stock image of the test also carried a local patch, and the runs were sequential with one run per size.
- The SSD spill never evicted in these runs. This experiment keeps `SPILL_GIB=0`.

## Switches of the bundle

The private bundle sets `SPILL_GIB=0` and the prefill chunk 512 and ignores the `.env` values of both. Everything else comes from `.env`, as `./start.sh` reads it.

## Head sudo and source drift

The launcher tries `sudo -n true` on the head first and calls `sudo -v` only when that fails, so a user with passwordless sudo can run it without a terminal. If both fail, it stops before any DRM change. A source-drift error names the file and shows the expected and the found SHA-256 values. A base image with a local patch counts as drift: build a pristine base from the committed Dockerfile first.

## Validation status

Offline tests use temporary fixtures and mocks. Optional source tests need an
explicit `TF_DISPLAY_SOURCE` checkout. See [tests](../tests/README.md).
Full hardware validation of this split is pending: two-node mapping and probe
success, a real reply, `verify.sh` PASS on all five checks, and an ordinary
launch regression check. A health response alone is not sufficient.
No new capacity, TTFT, decode tok/s or prefill performance claim is made.

See [implementation notes](../experiments/display-kv/README.md) and
[attribution](../experiments/display-kv/NOTICE.md).
