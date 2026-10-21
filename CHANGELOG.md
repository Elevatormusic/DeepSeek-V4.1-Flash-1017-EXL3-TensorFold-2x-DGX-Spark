# Changelog

This file lists the notable changes of each release. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow [Semantic Versioning](https://semver.org/):

- **Major**: a change that needs action from users, for example a new engine image, new model files or a changed default.
- **Minor**: a new opt-in feature or setting.
- **Patch**: a fix or a docs change.

Each release is a git tag. To use a release, run `git checkout vX.Y.Z` before `./start.sh`.

## [Unreleased]

- Add an independent opt-in [cache metrics experiment](experiments/cache-metrics/README.md) for health and Prometheus accounting. The ordinary engine stays unchanged.

### Added

- `docs/OPERATIONS.md`: the engine switches of release 1.1.0 (decode graph sizes, capture mode, host-memory settings, draft ranking), with their defaults.
- Add an opt-in display-backed KV launcher with a private bundle, spill disabled and offline safety tests. Full hardware validation is pending. Ordinary launch defaults do not change.
- Add a separate memory/top-k image with bounded key scratch and an opt-in live allocator policy. Ordinary startup keeps its original ceiling. See [Memory and top-k](docs/MEMORY-TOPK.md).
- `PREFILL_CHUNK` in `.env`: set a positive integer for both ranks. The default stays `2048`.
- Opt-in `experiments/display-spill` image: add display-safe SSD spill to display KV. It keeps `SPILL_GIB` from `.env`, drains queued writes on shutdown, and rejects corrupt entries. The ordinary image and defaults do not change. See `docs/SPILL.md`.
- `THINKING` and `REASONING_EFFORT` in `.env`: turn thinking on for requests that do not say, and set their effort. The default stays off. `docs/CONNECT.md` says to set thinking before you compare benchmark scores.
- Opt-in preflight check: set `PREFLIGHT_NCCL_BOOTSTRAP=1` in `.env` and preflight runs a bounded NCCL socket-bootstrap check in the engine image on each node (20 s, then KILL after 5 s more, with an init process in the container so that signals reach the probe). It is off by default, so `./start.sh` behaves as before. It checks that NCCL can create a unique ID with `NCCL_SOCKET_IFNAME`; it does not test the RDMA data path or allocate GPU memory. A failure names the interface and points to `.env`.
- Opt-in `experiments/tool-output` image: reject a complete model tool call that the request did not offer. It returns a safe error (HTTP 422) and does not expose tool arguments. Ordinary prose still streams. The guard is on in that image; `STRICT_TOOL_OUTPUT=0` in `.env` turns it off. The ordinary image and defaults do not change. See `docs/TOOL-OUTPUT.md`.

### Changed

- Correction of the 1.1.0 release gate: the 32K prefill row (2,751 tok/s) had a prefix-cache hit of 6,144 tokens and is withdrawn ([runs/2026-10-21](runs/2026-10-21/INDEX.md)). With no prefix-cache hit, random natural text gives 1,963 tok/s at 32K on 1.1.0. `tests/v41_suite.py` now salts each speed prompt, uses varied text for the prefill rows, logs `cached_tokens` and fails a prefill row on a cache hit. The kit_bench numbers stay, with a note that they use a small-vocabulary prompt.
- New engine image: the recipe now builds TensorFold [release dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) (`730d41abd8f4`; before: `07aa02585f64`). Git does not track `.env`, so `git pull` does not change your pin. In your `.env`, set `TF_COMMIT=730d41abd8f4` and `IMAGE=tensorfold-dsv41:730d41abd8f4`. Then run `docker/build.sh` and `./stop.sh && ./start.sh`. Without the new `.env` values, `docker/build.sh` builds the old engine again. It fixes graph capture while serving (#19) and lowers the host-memory peak of the head (#21, #24). In the gate run, 4-stream decode is 3.0 to 7.0% faster ([runs/2026-10-21](runs/2026-10-21/INDEX.md)). The head memory margin under 4 long streams stays small: set `PREFILL_CHUNK=1024` or `PARALLEL=2` when the head has little free memory ([docs/OPERATIONS.md](docs/OPERATIONS.md#engine-switches-of-release-110)).
- New engine image: the recipe now builds TensorFold [release dsv41-1.1.1](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.1) (`cb4ceaff2b8c`; before: `730d41abd8f4`). Git does not track `.env`, so `git pull` does not change your pin. In your `.env`, set `TF_COMMIT=cb4ceaff2b8c` and `IMAGE=tensorfold-dsv41:cb4ceaff2b8c`. Then run `docker/build.sh` and `./stop.sh && ./start.sh`. The new engine sets NCCL cuMem off, contains DSML tool calls, keeps special-token text as text (security), exits cleanly after SIGTERM, and fixes the `json_schema` prompt. Decode and cold 32K prefill are equal to 1.1.0 within the noise ([runs/2026-10-21](runs/2026-10-21/INDEX.md#release-gate-of-engine-111)). Known issue: the lowest head `MemAvailable` in the long run was 1.58 to 1.62 GiB (1.1.0: 1.68 to 1.85 GiB), and the image holds 0.14 GiB more. Use the low-memory settings when the head has little free memory ([docs/OPERATIONS.md](docs/OPERATIONS.md#engine-switches-of-release-111)).
- New default: `DROP_UNOFFERED_TOOL_CALLS=1` in `.env.example` and `start.sh`. The engine drops a call to a tool that the request did not offer, also while the reply streams. Set `DROP_UNOFFERED_TOOL_CALLS=0` in `.env` to turn it off.
- The three experiments (cache metrics, display-backed KV, memory and top-k) still need the 1.0.0 engine `07aa02585f64` as their base. Their docs now show how to build it.
- `scripts/preflight.sh` shows a `[warn]` line when a node has an NVIDIA driver older than the tested 580.178.04 (#24). `docs/OPERATIONS.md` gives the node-hang log signature from #24 and the recovery steps.
- `docs/OPERATIONS.md`: what to do when earlyoom stops the engine under concurrent load (#21): `PARALLEL=2`, and do not raise `MEM_FLOOR_GIB`.
- `docs/OPERATIONS.md`: the out-of-memory steps now use the measured traces from #21 (`runs/2026-10-20/`). With 8 images in each request, `PARALLEL=2` passed the load that failed with `PARALLEL=4` (not a clean A/B); with 16 it also failed. The steps now also say to send fewer images at the same time, or to set `VISION=0` when you do not need images.

### Fixed

- README: the free disk you need is about 650 GB (the model view is 414 GB, the first start needs 110 GiB more for the cache, and the default prompt spill can use 64 GiB), not 500 GB. With `SPILL_GIB=0`, about 550 GB is enough.
- README: the concurrency columns are now "aggregate throughput" (all completion tokens divided by the batch wall time, TTFT included), not "aggregate decode".
- `docs/CONNECT.md`: the examples show how to send your key when you set `API_KEY_FILE`.
- `./start.sh` stopped with "Permission denied" on a new clone: `scripts/build_extensions.sh` had no executable bit (#15).
- Preflight now checks that RoCE has MTU 4096 on both nodes. At the default MTU 1500, the RDMA gather failed with "transport retry counter exceeded" (#15). `docs/NETWORK.md` tells you to set MTU 9000.

### Repo

- CI: a defaults guard test, a home-path and public-IP scan, workflow checks (actionlint, zizmor) and a secret scan (gitleaks).
- CI: the secret scan refuses a `.gitleaksignore` in the checkout, because gitleaks loads it in any case.
- CI: the secret scan runs from the base branch (`secret-scan.yml`, `pull_request_target`), so a pull request cannot change it.
- AGENTS.md: rules for AI agents that open issues and pull requests, with links to CONTRIBUTING.md, the templates and SECURITY.md.
- Issues: the `issue-labels` workflow adds a type label from the title prefix (`[bug]`, `[feature]`, `[perf]`), also for an issue made without a form. An issue with no prefix gets `needs-triage`.

## [1.0.0] - 2026-10-19

The first release: the setup that the README measured on 17 and 18 October 2026.

### Serving

- DeepSeek-V4.1-Flash-1017 (EXL3, 2.9 bpw) on two DGX Spark with TensorFold, tensor-parallel over the QSFP link.
- About 2x the decode speed of the standard vLLM setup (57 vs 27 tok/s on prose, 92 vs 40 tok/s on code), 1,048,576-token context and 4 concurrent requests.
- Prefill of about 2,000 to 2,200 tok/s up to 64K tokens: fork PRs 1-4, host memory settings and warm-up cache (#3), O_DIRECT prompt spill and transparent huge pages off (#5).
- Prefix cache in memory and on disk (`SPILL_GIB`), so a restart keeps cached prompts.
- Forced tool calls and structured output for AI agents.

### Docs

- README in plain language, connect guide and notes for agents (#1, #2).
- Standard serving terms: prefill, decode, TTFT, tok/s (#4).

### Repo

- Contributing rules, issue forms, pull request template, CI (shellcheck, ruff, offline tests, Markdown links), security policy and code of conduct (#7).
- AI disclosure and label rules for contributors (#8).
- A pull request check for the type and the AI disclosure, and this changelog (#9).

[Unreleased]: https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/releases/tag/v1.0.0
