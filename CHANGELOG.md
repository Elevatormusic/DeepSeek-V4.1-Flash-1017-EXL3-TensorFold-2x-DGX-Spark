# Changelog

This file lists the notable changes of each release. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow [Semantic Versioning](https://semver.org/):

- **Major**: a change that needs action from users, for example a new engine image, new model files or a changed default.
- **Minor**: a new opt-in feature or setting.
- **Patch**: a fix or a docs change.

Each release is a git tag. To use a release, run `git checkout vX.Y.Z` before `./start.sh`.

## [Unreleased]

### Serving

- Add an optional BF16 uncensored model view, with pinned tensor checks, separate settings and weight-specific caches. The normal model and engine defaults stay the same.

### Fixed

- `./start.sh` stopped with "Permission denied" on a new clone: `scripts/build_extensions.sh` had no executable bit (#15).
- Preflight now checks that RoCE has MTU 4096 on both nodes. At the default MTU 1500, the RDMA gather failed with "transport retry counter exceeded" (#15). `docs/NETWORK.md` tells you to set MTU 9000.

### Repo

- CI: a defaults guard test, a home-path and public-IP scan, workflow checks (actionlint, zizmor) and a secret scan (gitleaks).
- CI: the secret scan refuses a `.gitleaksignore` in the checkout, because gitleaks loads it in any case.
- AGENTS.md: rules for AI agents that open issues and pull requests, with links to CONTRIBUTING.md, the templates and SECURITY.md.

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
