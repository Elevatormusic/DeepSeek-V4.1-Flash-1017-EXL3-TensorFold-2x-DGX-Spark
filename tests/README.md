# Tests

This folder holds the tests that gave the results in [`runs/2026-10-17/`](../runs/2026-10-17/).
Run all of them with one command, or run each test alone.

## Offline tests (no server, no GPU)

`tests/run_offline.sh` runs every offline test. CI runs the same script on each pull request. It needs no server, no GPU and no `.env`.

```bash
tests/run_offline.sh
```

It runs `host_tuning_test.sh` (see below) and the Python `unittest` files (`test_*.py` and `*_test.py`) in `tests/` and in each `experiments/*/tests/` folder.
`test_repo_rules.py` checks the repo rules of [CONTRIBUTING.md](../CONTRIBUTING.md): git tracks `runs/`, each entry in `runs/` is a `YYYY-MM-DD` folder with an `INDEX.md`, and `.env.example` sources cleanly and holds no token.
A test that needs a GPU, a server or the `tensorfold` package skips itself.

`test_prepare_uncensored.py` checks the [optional BF16 view](../docs/UNCENSORED.md) with small local tensor files. It checks atomic publication, index handling and receipt limits. Publication tests need Linux. No downloaded weights are needed.
`test_uncensored_launcher.py` uses shell stubs to check help, invalid commands, worker hash mismatch, Docker query failures, active GPU refusal, engine pins and cache identity. It starts no server.

## Script check (no server needed)

`bash tests/host_tuning_test.sh` runs the node part of `scripts/host_tuning.sh` in a temporary folder, with stub `sysctl` and `systemctl` commands. It changes nothing on the node. It checks that the first run keeps the old sysctl values, that `--undo` puts them back, that a missing or stopped earlyoom stops the setup, that a failed setup puts the node back, and that transparent huge pages go off with a systemd unit and come back on with `--undo`.

## Run everything

Run this on the head node, with the server up:

```bash
tests/run_all.sh
```

The script reads your `.env` through `scripts/common.sh`. It prints one `PASS`, `FAIL` or `SKIP` line per step.
It exits with code 1 when any step fails. A `SKIP` does not fail the run.

| Step | Runs when | Time (measured on the pair) |
|---|---|---|
| `verify.sh` | always | about 1 minute |
| `v41_suite.py` | always (the 2 router tests run only with a router key) | 278 to 324 s |
| `v41_grammar_check.py` | always | about 25 s |
| `v41_conformance.py` | the `tensorfold` package is on the host, or the engine image is on the node | not recorded |
| `v41_differential.py` | `ENCODING_DIR` holds `encoding.py`, and the conditions of the conformance test are met | not recorded |
| `v41_bench.py` | `BENCH=1` | about 2.5 minutes |
| `hermes_thinking_modes.py` | `HERMES=1` | about 2 minutes |

Every output goes to `runs/<UTC date>/`. The file names start with `runall-<image tag>-<time>`:

- `...-summary.txt`: the PASS, FAIL and SKIP lines.
- `...-verify.log`, `...-suite.log`, `...-grammar-check.log`, `...-conformance.log`, `...-differential.log`, `...-bench.log`, `...-hermes.txt`: the console output of each step.
- `...-suite-results.json`, `...-bench-results.json`, `...-hermes.json`: the result files.

A step passes only when both checks pass:

1. The test exits with code 0.
2. The log has no `FAIL` line, and the last summary line shows equal counts (for example `30/30 passed`).

## Settings

The tests run on the head node. They bind their source address to `127.0.0.1`.

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `V41_BASE_URL` | suite, benchmark, grammar check | `http://172.17.0.1:8888/v1` | The engine API. `run_all.sh` sets it from `.env`. |
| `V41_MODEL` | suite, benchmark, grammar check | `DeepSeek-v4.1-Flash-1017-EXL3` | The model id in each request. `run_all.sh` sets it to `SERVED_NAME`. |
| `V41_ROUTER_URL` | suite (`router` group) | `http://127.0.0.1:8443/v1` | A router in front of the engine. It must accept the model id `dgx-model`. |
| `V41_ROUTER_API_KEY` | suite (`router` group) | none | The key for the router. `DGX_ROUTER_API_KEY` also works. |
| `ENCODING_DIR` | `run_all.sh`, differential test | none | A folder that holds DeepSeek's `encoding.py`. See below. |
| `FIXTURE_DIR` | `run_all.sh` | `runs/2026-10-17/encoding-fixtures` | The folder with the conformance fixtures. |
| `BENCH` | `run_all.sh` | `0` | Set to `1` to run the benchmark. |
| `HERMES` | `run_all.sh` | `0` | Set to `1` to run the Hermes gate. |
| `HERMES_DIR` | Hermes gate | none | The Hermes install folder. |
| `HERMES_PYTHON` | `run_all.sh` | `$HERMES_DIR/hermes-agent/venv/bin/python` | The Python of the Hermes install. |
| `HERMES_PROVIDER_KEY_ENV` | Hermes gate | `HERMES_CUSTOM_DGX_SPARK_API_KEY` | The name of the variable that holds the provider key. |
| `GRAMMAR_ROUNDS` | `run_all.sh` | `2` | The number of rounds of the grammar check. |
| `CHECK_DOCKER_ARGS` | `run_all.sh` | none | Extra `docker run` options for the conformance and differential tests (for example `--gpus all`). |

The tests send no API key to the engine. If `.env` sets `API_KEY_FILE`, the engine tests get HTTP 401.
Start the server without `API_KEY_FILE` for the test run.

## What each test checks

### `verify.sh` (in the repo root)

Sends five real requests: a plain reply, a thinking reply, a strict `json_schema` reply, a forced tool call (`tool_choice` required) and a regex reply.
It checks the content of each reply. A health check alone does not prove that TP=2 works.

Run it alone:

```bash
./verify.sh
```

### `v41_suite.py`

The 32-test suite. It uses only the Python standard library. It has seven groups:

| Group | Tests | What it checks |
|---|---:|---|
| `speed` | 6 | Decode speed (prose and code, thinking off and on), prefill speed at 8K and 32K tokens. |
| `tools` | 12 | Tool calls: single, none needed, `tool_choice` required and named, nested arguments, parallel calls, tool-result round trip, streamed deltas, a four-step agent loop. Thinking off and on where the name says so. |
| `schema` | 6 | Structured output: strict `json_schema` (thinking off and on), `json_object`, choice, regex, and JSON from the prompt alone (no constraint). |
| `reasoning` | 2 | The four thinking levels (off, low, high, max) and no `<think>` tag in the answer. |
| `format` | 2 | A stop sequence, and Japanese and French output. |
| `long_context` | 2 | Needle in a haystack at 64K and 200K tokens. |
| `router` | 2 | A streamed tool call, and a default request, both through a router. |

The speed numbers in the results are described in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).
The suite exits with code 1 when any test fails. It writes `suite-results.json` next to the script.

Run all groups:

```bash
python3 -I -u tests/v41_suite.py
```

Run some groups. Name the groups after the command:

```bash
python3 -I -u tests/v41_suite.py tools schema
```

Run every group except `router` (30 tests):

```bash
python3 -I -u tests/v41_suite.py speed tools schema reasoning format long_context
```

### `v41_grammar_check.py`

Sends grammar requests and plain requests at the same time, four at a time, under `--parallel`.
There are seven cases per round: `json_schema` (thinking off and on), `json_object`, regex, choice, plain prose and plain code.
Each grammar reply must match its format. Each plain reply must come back. Nothing may hang.
The default is two rounds (14 requests).

```bash
python3 -I -u tests/v41_grammar_check.py 2
```

### `v41_conformance.py`

Renders the five DeepSeek encoding fixtures through TensorFold's prompt path and compares each prompt with DeepSeek's golden output, byte for byte.
It runs no DeepSeek code. The fixtures are data (`runs/2026-10-17/encoding-fixtures/`).
It imports the `tensorfold` package, so it must run on a host that has the package, or in the engine image.

It needs the patched `chat_template.jinja` in the model folder.

Run it in the engine image, from the repo root. First load your `.env`:

```bash
source ./.env
```

Then start the image:

```bash
docker run --rm --network none --entrypoint python3 -v "$PWD/tests:/tests:ro" -v "$MODEL_VIEW/model:/model:ro" -v "$PWD/runs/2026-10-17/encoding-fixtures:/fixtures:ro" "$IMAGE" -I -u /tests/v41_conformance.py /model /fixtures
```

### `v41_differential.py`

Renders 12 API-shaped conversations (chat, system turns, thinking levels, tools, tool results, `json_object`) with DeepSeek's reference encoder (`encoding.py`) and with TensorFold's prompt path.
Each prompt must match byte for byte. It needs `encoding.py` (see the next section) and the `tensorfold` package.

Run it in the engine image, from the repo root, after `source ./.env`. `ENCODING_DIR` is the folder that holds `encoding.py`:

```bash
docker run --rm --network none --entrypoint python3 -v "$PWD/tests:/tests:ro" -v "$MODEL_VIEW/model:/model:ro" -v "$ENCODING_DIR:/encoding:ro" "$IMAGE" -I -u /tests/v41_differential.py /model /encoding /tests
```

`run_all.sh` runs both tests with the host Python when the host has the `tensorfold` package. Otherwise it uses these docker commands.

### `v41_bench.py`

The concurrency and prefix-cache benchmark. It sends 1, 2, 3 and 4 streams of prose and of code (512 tokens each), then a prefix-cache check with a document of about 24,000 tokens.
It writes `bench-results.json` to the current folder. It does not exit with code 1 on slow results: it reports numbers only.

```bash
python3 -I -u tests/v41_bench.py
```

### `hermes_thinking_modes.py`

The Hermes agent gate. It drives the `AIAgent` of a Hermes Agent install against a live router at eight thinking levels, streamed and not streamed, with a math task and a tool task, plus three session-title requests with a strict `json_schema`.
It runs 35 cases. For each case it checks the reasoning field on the wire, that reasoning came back when it should, the answer, and that no `<think>` tag leaks.
It uses a throwaway `HERMES_HOME`, so your Hermes sessions and memory stay untouched.

It needs a Hermes install (`HERMES_DIR`), the Python of that install, and a router that the Hermes config points at. Run it with the Python of the install:

```bash
"$HERMES_DIR/hermes-agent/venv/bin/python" -I -u tests/hermes_thinking_modes.py --report hermes.json
```

## Tests that need a router or Hermes

| Test | Needs | How to skip it |
|---|---|---|
| Suite group `router` (2 tests) | A router in front of the engine, `V41_ROUTER_URL`, `V41_ROUTER_API_KEY` | `run_all.sh` skips it when no router key is set. Alone: name the other groups. |
| `hermes_thinking_modes.py` | A Hermes Agent install and a router | `run_all.sh` skips it unless `HERMES=1`. |

Without a router key, the `router/tools_stream` test fails with the message `V41_ROUTER_API_KEY not set`, and `router/default` does not run.
This is why a plain run of `v41_suite.py` on a node without a router shows one failure and 31 results.

Set the router key without writing it to a file or to your shell history:

```bash
read -rs V41_ROUTER_API_KEY && export V41_ROUTER_API_KEY
```

The 32/32 results in `runs/2026-10-17/` include the two router tests.

## Get DeepSeek's `encoding.py`

The differential test needs the reference encoder of DeepSeek. This repo does not hold it.

| Item | Value |
|---|---|
| Repo | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Path | `encoding/encoding.py` |
| Licence | MIT (Copyright (c) 2023 DeepSeek) |
| Size | 37,962 bytes |
| sha256 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |

1. Make a folder that holds only this file. The test puts the folder first on its import path.

```bash
mkdir -p "$HOME/ds-encoding"
```

2. Download the file at the pinned revision.

```bash
curl -L -o "$HOME/ds-encoding/encoding.py" https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py
```

3. Check the hash. The command must print `OK`.

```bash
echo "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2  $HOME/ds-encoding/encoding.py" | sha256sum -c -
```

4. Point the tests at the folder.

```bash
export ENCODING_DIR="$HOME/ds-encoding"
```

Without `ENCODING_DIR`, `run_all.sh` prints `SKIP differential` and goes on.
The conformance test does not need `encoding.py`.

## What the tests do not cover

See [docs/KNOWN-LIMITS.md](../docs/KNOWN-LIMITS.md).
