# Contributing

Thank you for your help. This file tells you how to propose a change and what a pull request must show.
Read the [Code of Conduct](CODE_OF_CONDUCT.md) first. To report a security problem, read [SECURITY.md](SECURITY.md). Do not open a public issue for it.

## How to propose a change

1. Look at the [open issues](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues) and pull requests. Make sure that nobody does the same work.
2. For a large change, open an issue first. Tell us the problem and your plan. Large changes include a new serving mode, a new default, a new engine image or a new folder layout.
3. Fork the repo. Make a branch from `main`.
4. Make one focused change. See [Small, focused pull requests](#small-focused-pull-requests).
5. Run the offline tests ([Run the tests](#run-the-tests)). They must pass.
6. If your change touches the serving path, run the live checks on your own pair. Save the output under `runs/YYYY-MM-DD/` ([The evidence rule](#the-evidence-rule)).
7. If users see the change, add one line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
8. Open a pull request. Fill in every part of the template.

For a question, use [Discussions](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/discussions), not an issue.

The maintainer reviews each pull request. An automatic review bot can also comment. Reply to each comment, or push a fix for it.

## Small, focused pull requests

- **One change in each pull request.** For example, a fix to `start.sh` and a new benchmark are two pull requests.
- **Keep the diff small.** A reviewer must be able to read all of it. Split a large change into steps that each work alone.
- **Do not mix a refactor with a change in behaviour.** Put each one in its own pull request.
- **Do not change unrelated files.** Do not reformat files that your change does not need.

## Labels

### Pull requests

Each pull request gets exactly one type label. Tick the type in the template. The `pr-template` check adds the label. It fails if you tick no type, more than one type, or if the AI disclosure block is missing. Edit the pull request text to fix it: the check runs again.

| Label | Use it for |
|---|---|
| `type: fix` | a bug fix, or a fix after review findings |
| `type: feature` | new code: a feature, tool or add-on |
| `type: docs` | docs only: README, guides, roadmap |
| `type: results` | run results or a close-out |
| `type: research` | a study or failure research, with no fix |
| `type: spec` | a plan or spec before the work |

If your change needs two types, it is two pull requests.

### Issues

Use the issue form that fits. The form adds the label.

| Form | Label | Use it for |
|---|---|---|
| Bug report | `bug` | the server does not start, fails a check, gives wrong output or stops |
| Feature request | `enhancement` | a new setting, mode or tool |
| Performance report | `performance` | a speed or memory result, or a regression |

The maintainer can add more labels, for example `good first issue` or `duplicate`.

## Default behaviour

Users run `./start.sh` with the values in `.env.example`. A change to a default changes the server of every user after `git pull`.

These defaults are important. Do not change them silently:

| Setting | Where | Default |
|---|---|---|
| Thinking off for requests that do not ask for it (`--no-thinking`) | `start.sh` | off |
| Bind address (`BIND_HOST`) | `.env.example` | `127.0.0.1` (this node only) |
| Prefill chunk (`TF_DS_PREFILL_CHUNK`) | `.env` (`start.sh` fallback: `2048`) | `2048` |
| Prompt spill to disk (`SPILL_GIB`) | `.env.example` | `64` |
| Engine image (`TF_COMMIT`, `IMAGE`) | `.env.example` | the image in the README |
| Context, concurrency, draft depth (`CONTEXT`, `PARALLEL`, `MTP_DRAFTS`) | `.env.example` | as in the README |

If your pull request changes a default:

1. Say so in the title of the pull request.
2. Tick the default-change box in the template. Write the old value, the new value and the reason.
3. Give the evidence: a saved run that shows the effect ([The evidence rule](#the-evidence-rule)).
4. Update `.env.example`, the README and the doc in `docs/` that describes the setting.

A new feature must be off by default, or it must be in a separate start script. Then a user turns it on.

## The evidence rule

Every performance number in this repo comes from a saved run. This rule applies to the README, `docs/`, the Hugging Face card and the pull request text.

- **Link each number to its run.** The run is a file under `runs/YYYY-MM-DD/`. Use the UTC date of the run.
- **Use one layout.** Put each run in `runs/YYYY-MM-DD/`. You can use subfolders in a day folder. Do not use other layouts (for example `runs/YYYY/MM/`). Do not put files directly in `runs/`.
- **Add the run to the day index.** Each day folder has an `INDEX.md`. Add one line for each new run: what it measured, the image, the settings and the result.
- **Keep `runs/` tracked by git.** Do not add `runs/` or `runs/*` to `.gitignore`. The offline tests fail if git ignores a file in `runs/`.
- **Do not edit old runs.** A run is a record. To correct a run, add a new run and say in the index which run it replaces.
- **Do not change the scripts in `runs/`.** They are the scripts as they ran. CI does not lint them.

### Performance claims need a fair A/B

A claim such as "X makes prefill 10% faster" needs an A/B test. The A side and the B side must have the same settings, except for the change that you test.

Keep these identical on both sides, and write their values in the pull request:

- the engine image id (`IMAGE`, `TF_COMMIT`)
- `TF_DS_PREFILL_CHUNK` (in `.env`; `start.sh` fallback: `2048`)
- `SPILL_GIB`, `CONTEXT`, `PARALLEL`, `MTP_DRAFTS`, `MEM_FLOOR_GIB`
- the host tuning state (`scripts/host_tuning.sh`)
- the prompt lengths, the number of runs and the prefix-cache state (cold or warm)

Give the median and the range of at least 3 runs for each side. Use the standard terms: prefill tok/s, decode tok/s, TTFT, prefix cache. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) gives the method.

If one setting is different between the sides, the result is not a valid A/B. Do the runs again.

## Run the tests

### Offline tests (no server, no GPU)

Run these on any Linux or macOS computer. CI runs the same script.

```bash
tests/run_offline.sh
```

It runs `tests/host_tuning_test.sh` and the Python `unittest` files (`test_*.py` and `*_test.py`) in `tests/` and in each `experiments/*/tests/` folder. Each step prints `PASS`, `FAIL` or `SKIP`.

CI also runs the linters. Run them before you push:

```bash
python3 -m pip install -r .github/requirements-ci.txt
```

```bash
ruff check .
```

```bash
shellcheck -x -S warning $(git ls-files '*.sh' ':!:runs/**' ':!:scripts/_raw_*')
```

If you change a file in `.github/workflows/`, also run:

```bash
actionlint
```

```bash
zizmor .github/workflows/
```

CI also runs [gitleaks](https://github.com/gitleaks/gitleaks) on all commits, to find keys and tokens. Two offline tests guard the repo rules:

- `tests/test_defaults.py` fails if `start.sh` or `.env.example` changes a protected default ([Default behaviour](#default-behaviour)).
- `tests/test_machine_values.py` fails if a tracked file has a home path with a user name or a public IP address.

### Rules for new tests

- **An offline test needs no GPU, no server, no network and no `.env`.** Name it `test_*.py` or `*_test.py`. Use only the Python standard library.
- **A test that needs a GPU, a server or the `tensorfold` package must skip itself.** Use `unittest.skipUnless(...)` with a clear reason. CI runs on GitHub-hosted runners with no GPU, so it shows the skip.
- **A live test (against a running server) has a different name**, for example `v41_*.py`. Add it to `tests/run_all.sh` and to [tests/README.md](tests/README.md).

### Live checks (on two DGX Spark)

If your change touches the serving path, run these on the head node with the server up:

```bash
./verify.sh
```

```bash
tests/run_all.sh
```

`./verify.sh` must print PASS for all five checks. `tests/run_all.sh` saves its output to `runs/<UTC date>/`. [tests/README.md](tests/README.md) tells you what each test checks.

## No secrets and no machine-specific values

- **Do not commit `.env`.** Git ignores it. Put new settings in `.env.example` with a safe default and a comment.
- **Do not commit secrets.** This includes API keys, Hugging Face tokens, router keys, ssh keys and passwords. If you commit a secret by mistake, revoke it at once. Then tell the maintainer.
- **Remove machine-specific values from runs and logs** before you commit them. This includes host names, user names, home paths, public IP addresses and serial numbers. Use the example values of `.env.example` (for example `10.100.80.1`) or a placeholder such as `<head>`.
- **Do not put a real key in a command line.** Read it from a file or from a prompt (see [tests/README.md](tests/README.md)).

## Documents

- Write in [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/): short sentences, one instruction in each sentence, active voice.
- Use the standard serving terms as they are: prefill, decode, TTFT, tok/s, median, prefix cache, structured output.
- If you add a setting, document it in `.env.example` and in the doc in `docs/` that is about it.
- If you add a doc, link it from the README or from another doc. CI checks the local links in all Markdown files.

## AI disclosure

**Pull requests:** if an AI helped, paste this block at the very bottom. If no AI helped, write `AI: none` in it. The template already has the block.

**Issues:** the last field of each issue form asks for the same information. It starts empty and you must fill it in. Write `none` if no AI helped.

```html
<details>
<summary>AI disclosure</summary>

AI: <model and size> via <harness>, e.g. "Qwen3-8B Q4 via Aider" or "Claude Opus 5.5 via Claude Code"
Checked: <what you did yourself, e.g. "read every line, ran tests">

</details>
```

This does not cause a rejection. It tells reviewers how carefully to check your work.
Keep the empty lines after `<summary>` and before `</details>`. Without them, GitHub does not format the text.

## Licence

This repo uses the Apache License 2.0 ([LICENSE](LICENSE)). When you open a pull request, you agree that your contribution uses the same licence. If you add code or data from another project, give its licence and source in [CREDITS.md](CREDITS.md).
