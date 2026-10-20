# Notes for AI agents

You can set up and run this recipe for a user. Follow these steps in this order. Run every command on the head node, in this folder.

## Setup

1. Check that the two nodes are cabled and reach each other (NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide; [docs/NETWORK.md](docs/NETWORK.md)). `ibdev2netdev` must show the CX7 ports as Up on both nodes. Set MTU 9000 on the CX7 netdevs on both nodes ([docs/NETWORK.md](docs/NETWORK.md#mtu)): at the default MTU 1500, the start fails.
2. `cp .env.example .env`, then fill in the values from `ibdev2netdev`, `ip -br addr` and `show_gids`. Ask the user for paths you cannot find.
3. `scripts/download.sh` (~400 GB; resumable: run it again after a break). If a download needs a Hugging Face login, ask the user to run `hf auth login`. Never type a token yourself.
4. `scripts/patch_template.py --worker`
5. `docker/build.sh`
6. `scripts/host_tuning.sh` (once; it needs sudo on both nodes, so the user types the password). It turns swap off for the engine, sets the kernel memory values, turns transparent huge pages off (a systemd unit keeps this after a reboot) and makes earlyoom stop the engine (not the node) when memory runs out; preflight fails without it. Restart the server after it, because the engine sizes its memory at start.
7. `./start.sh` (it runs `scripts/preflight.sh` first; fix each `[FAIL]` line it prints). The first start takes ~15 min.
8. `./verify.sh` must print PASS for all five checks. Only a real reply proves that both nodes work; `/health` alone does not.

Then give the user the three client settings from [docs/CONNECT.md](docs/CONNECT.md).

## Rules

- **Never `docker exec` into `tfv41-r0` or `tfv41-r1` while they serve.** The exec process counts against the rank's memory.
- **Never stop the display manager** (`systemctl stop gdm`) on a Spark. On the tested pair it powered the node off.
- **Run one model server at a time.** Each one needs port 8888 and ~100 GB of memory on each node.
- **Keep `BIND_HOST=127.0.0.1`** unless the user asks for network access. Then set `API_KEY_FILE` too ([docs/CONNECT.md](docs/CONNECT.md)).
- **If the server stops with exit code 75**, the lane broke on a device fault. Run `./stop.sh && ./start.sh`.
- **If NCCL hangs at start after a cable or reboot event**, the RoCE GID index moved. Set `GID_INDEX=-1` in `.env` and restart ([docs/NETWORK.md](docs/NETWORK.md)).

## Issues and pull requests

If you open an issue or a pull request in this repo, read [CONTRIBUTING.md](CONTRIBUTING.md) first and obey it. The checks below fail or the maintainer closes the work if you do not.

**Issues**

1. Search the open issues first. For a question, use [Discussions](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/discussions), not an issue.
2. Use the issue form that fits: bug report, feature request or performance report. Do not open a blank issue. Fill in every required field.
3. Fill in the last field, "AI disclosure": your model and size, your harness and what you and the user checked. Do not write `none`: an AI wrote it.
4. For a security problem, do not open an issue. Follow [SECURITY.md](SECURITY.md).

**Pull requests**

1. One focused change in each pull request, made on a branch from `main`. Do not mix a refactor with a change in behaviour. Do not change unrelated files.
2. Do not change a protected default ([Default behaviour](CONTRIBUTING.md#default-behaviour)) unless the user asks for it. Then follow the four steps in that section. `tests/test_defaults.py` fails on a silent change.
3. Run the offline tests and the linters ([Run the tests](CONTRIBUTING.md#run-the-tests)). They must pass before you push.
4. Give each performance number a saved run under `runs/YYYY-MM-DD/` ([The evidence rule](CONTRIBUTING.md#the-evidence-rule)). Never write a number that no run shows. A claim that a change makes something faster or slower also needs a fair A/B ([Performance claims need a fair A/B](CONTRIBUTING.md#performance-claims-need-a-fair-ab)).
5. Remove secrets and machine-specific values: home paths with a user name, public IP addresses, host names, keys ([No secrets](CONTRIBUTING.md#no-secrets-and-no-machine-specific-values)). `tests/test_machine_values.py` and gitleaks check this.
6. Use the [pull request template](.github/pull_request_template.md) and fill in every part. Tick exactly one type. Keep the "AI disclosure" block at the bottom, with your model and size, your harness and what you and the user checked. The `pr-template` check fails without them.
7. If users see the change, add one line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
8. Reply to each review comment, or push a fix for it. Do not merge: the maintainer merges.

Write docs and the pull request text in ASD-STE100 Simplified Technical English. Use the standard serving terms: prefill, decode, TTFT, tok/s, median, prefix cache.

## Where to look

| Question | File |
|---|---|
| A start fails | `logs/` (saved by `./stop.sh`), `docker logs tfv41-r0`, [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| How to call the API | [docs/CONNECT.md](docs/CONNECT.md) |
| Run the tests | [tests/README.md](tests/README.md) |
| Open an issue or a pull request | [CONTRIBUTING.md](CONTRIBUTING.md), [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/), [.github/pull_request_template.md](.github/pull_request_template.md) |
| Report a security problem | [SECURITY.md](SECURITY.md) |
| What is not supported | [docs/KNOWN-LIMITS.md](docs/KNOWN-LIMITS.md) |
| Exact versions and checksums | [runs/2026-10-17/PROVENANCE.md](runs/2026-10-17/PROVENANCE.md) |
