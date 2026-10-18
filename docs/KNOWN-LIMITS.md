# Known limits

This page lists what this recipe does not measure, does not support, and does not ship.
Each item was checked against the files in this repo. The file that supports the item is named.

## Not measured

1. **Image input.**
   - The server starts with `--vision` by default (`VISION=1` in `.env.example`).
   - No test in `tests/`, no check in `verify.sh` and no Hermes case sends an image to the server. The conformance test only renders the image placeholder in a prompt.
   - Not measured: image decode speed, two requests with images at the same time, many images in one request.
2. **vLLM under concurrency.** The vLLM kit ran with `MAX_NUM_SEQS=1` (`runs/2026-10-17/PROVENANCE.md`, section 4). It serves one request at a time. No vLLM result exists for 2 to 4 streams, so the TensorFold concurrency tables have no baseline.
3. **KV cache bytes per token.** The run files do not hold the KV pool size in tokens, for either engine. See [BENCHMARKS.md](BENCHMARKS.md).
4. **Context near 1M tokens.**
   - The server is set to 1,048,576 tokens (`--context 1048576`).
   - Two needle tests reached about 1,035,000 tokens and found the needle (818 s and 830 s to the first token, [runs/2026-10-18](../runs/2026-10-18/INDEX.md)). Each had one needle at 63% depth; the second ran on the final setup.
   - Memory is close to the limit at 1M: the fill took the head down to 1.7 GiB available, and earlyoom stops the engine below 1 GiB. Close other programs on the head before a 1M request.
   - Not measured near 1M: answer quality beyond one fact, decode speed, two long prompts at the same time.
   - The vLLM baseline was set to 262,144 tokens (`MAX_MODEL_LEN`).
5. **Output quality.** The repo has no perplexity or accuracy test. The suite checks a few fixed tasks. No test compares TensorFold output with vLLM output. The perplexity numbers on the model card are for the weights, not for this engine.

More that is not measured:

- **Replies under load.** bertholomus's release gates check that a reply under `--parallel 4` equals its solo reply. No test in this repo repeats that check.
- **Long-prompt prefill is still below bertholomus's published numbers.** With their client, the final setup gives 2,178 / 2,196 / 2,152 / 2,061 tok/s at 8K / 32K / 64K / 128K (median of 3 runs; each range is less than 2.5% wide); their Python engine published 2,387 / 2,421 / 2,363 / 2,242 on other nodes with the original V4.1 weights. On 17 October this recipe gave 2,190 / 1,707 / 1,457 / 1,420. The first long prompt after a start is still slower.
   - Fixed: swap, kernel compaction and Engram page-cache reclaim caused most of the 1.3-3.8 s chunk stalls. The final memory settings made 128K steady (1,851-1,986 instead of 1,202-1,988) ([runs/2026-10-18](../runs/2026-10-18/INDEX.md)).
   - Fixed: a fresh 64K prompt froze the head when memory ran out. Now earlyoom stops the engine first, and the gate's fresh 64K and 200K needles pass.
   - Fixed: some runs were 10-20% slower (one 64K run at 1,836 tok/s; 1 of 3 greedy code replies at 69-71 tok/s instead of 85). The cause was page compaction, which stalls the GPU: the spill writes and transparent huge pages. With the O_DIRECT spill and THP off, prefill ranges are less than 2% wide and 3 of 3 greedy code replies run at 84.7-85.1 tok/s. The fast/slow CPU core idea was tested and was not the cause.
   - Still open: the attention of the ratio-2 layers grows with position (68 ms a chunk at the start, 195 ms at 61K). Their per-section numbers are not published, so this cannot be compared yet.
- **More than four streams.** The benchmark stops at four streams, which is the `--parallel` setting.
- **Prefix-cache use in the speed rows.** The suite does not record `cached_tokens`, so a cache hit in the 32K prefill and the 200K needle rows cannot be ruled out. See "Prefill" in [BENCHMARKS.md](BENCHMARKS.md).
- **GPU clock and link state.** No run file records the GPU clock or the link bandwidth before a run.
- **The conformance and differential tests.** Their outputs are in `runs/2026-10-17/18_*` (3/3 and 12/12), measured on `fd2365042f56` (the image of 17 October).
- **Measured on one pair only.** All results come from one pair of DGX Spark, on 17 and 18 October.

## Not supported and not shipped

1. **No boot unit.** The server starts by hand (`./start.sh`). The repo has no systemd unit. [OPERATIONS.md](OPERATIONS.md) describes the rules for one.
2. **No automatic restart.** A device fault (CUDA error, out of memory) or a lost rank breaks the lane. Both ranks then exit with code 75 after 5 s. They stay down until you run `./stop.sh && ./start.sh` or a supervisor does it.
3. **No API key in the tests.** The test scripts send no `Authorization` header to the engine. With `API_KEY_FILE` set, the engine tests get HTTP 401.
4. **Living Weights is not available.**
   - It is the `--slide` mode of TensorFold. It ships in upstream TensorFold 1.0.3 and later (the Zig engine).
   - It learns only into Nemotron 3.5 Lightning, MLX 4-bit.
   - It needs about 26 GiB of extra GPU memory on a GB10.
   - It rewrites a single-node model folder.
   - This recipe runs the Python 0.6.6 line, with EXL3 weights, across two nodes.
5. **Only TP=2 on two nodes.** The launcher has no setting for one, three or four nodes.

## Costs and requirements

1. **The engine is a fork, pinned at a commit.**
   - The recipe builds [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) at `3c45bdaeab7d`. The upstream project is `ashhart/TensorFold`.
   - The changes are pull requests #1 to #4 on the fork. The files in this repo do not show that upstream accepted them.
2. **The first start is slow and large.** The first start writes the weight cache of each rank to `CACHE_DIR`. This takes about 15 minutes and about 100 GiB per node (`.env.example`; `scripts/preflight.sh` asks for 110 GiB free). Later starts take 2 to 4 minutes.
3. **The first long prompt pays a warm-up.** The first prompt of 8K tokens or more after a start takes about 10 s longer ([OPERATIONS.md](OPERATIONS.md)). The 8K prefill row in [BENCHMARKS.md](BENCHMARKS.md) includes this cost.
4. **The Engram tables are large.** The two 1017 Engram shards are about 101 GB each. The model folder on each node holds about 196 GiB of weights and about 189 GiB of Engram tables (`PROVENANCE.md`, section 1).
5. **The weights revision is not the newest.**
   - The measured revision of `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` is `afc7aa0a`. The files were downloaded before 17:47Z on 2026-10-17.
   - The model card had revision `c178a012` at 18:33Z the same day.
   - This repo did not compare the two revisions. It did not run the tests on the newer one.

## Tests that need other systems

- Two suite tests (group `router`) need a router in front of the engine. Without a router key, `tests/run_all.sh` skips them and runs 30 tests.
- The Hermes agent gate needs a Hermes Agent install and a router. The results in `runs/2026-10-17/` came from the router of the author. The repo does not ship that router.
- The differential test needs DeepSeek's `encoding.py`, which this repo does not hold. See [tests/README.md](../tests/README.md).
