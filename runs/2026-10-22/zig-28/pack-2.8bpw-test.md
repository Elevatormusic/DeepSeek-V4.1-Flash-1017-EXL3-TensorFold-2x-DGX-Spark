# 2.8 bpw EXL3 pack on TensorFold 1.1.1: A/B against our 2.9 bpw pack

Date: 2026-10-22 (UTC). Engine: TensorFold fork release dsv41-1.1.1, image `tensorfold-dsv41:cb4ceaff2b8c` (ID a760ff885c96) on both nodes. Pair: 2x DGX Spark, TP=2, production settings (context 1,048,576, `--parallel 4`, 5 MTP drafts, prefill chunk 2048, spill 64 GiB, memory floor 1.5 GiB).

- **Pack A:** our pack, `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ afc7aa0a (about 196 GiB).
- **Pack B:** `jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` (MIT, 190 GiB, 47 files).

## Result

**Pack B gives 3.0 GiB more free memory a node at the long-run low point.** Quality and speed are equal in our tests. MTP acceptance is 2-3 points lower at depth 1.

| | Pack A (2.9) | Pack B (2.8) | Change |
| --- | ---: | ---: | ---: |
| Idle MemAvailable, median of 140 s, head (2 runs) | 3,444 / 2,922 MiB | 6,469 / 6,568 MiB | **+3.3 GiB** |
| Idle MemAvailable, median of 140 s, worker (2 runs) | 3,122 / 3,263 MiB | 6,799 / 6,343 MiB | **+3.3 GiB** |
| Lowest MemAvailable in the long 4-stream run, head | 1,786 / 1,427 MiB | 4,874 / 4,462 MiB | **+3.0 GiB** |
| Lowest MemAvailable in the long 4-stream run, worker | 2,472 / 1,877 MiB | 5,493 / 4,780 MiB | **+2.9 GiB** |
| earlyoom events, warnings, errors | 0 | 0 | |
| Suite (`v41_suite.py`) | 32 / 32 | 32 / 32 | |
| Grammar check | 14 / 14 | 14 / 14 | |
| Tool-call check (temperature 1.0, top_p 0.95, parallel calls) | 36 / 36 good, 0 lost, 0 malformed | 39 / 39 good, 0 lost, 0 malformed | |

Changes are the means of the two runs a side. The head low point of pack A in run a2 (1,427 MiB) was 0.18 GiB above the 1.25 GiB stop rule. Pack B keeps it above 4.4 GiB.

## Paired tool-call A/B (500+ requests a pack)

Date: 2026-10-22, 12:59-13:53Z, one GPU window. Same engine and settings as above. Harness `tcm_harness.py`: agentic multi-turn tasks with parallel tool calls, temperature 1.0, top_p 0.95, 4 workers. ABBA order. The same seed gives both packs the same task sequence.

| Step | Pack | Seed | Thinking | Requests | Good | Lost / malformed / error | Tool calls | Parallel turns | Head low point |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| ta1 | A (2.9) | 11 | off | 257 | 257 | 0 / 0 / 0 | 531 | 168 | 2,068 MiB |
| tb1 | B (2.8) | 11 | off | 266 | 266 | 0 / 0 / 0 | 531 | 161 | 4,962 MiB |
| tb2 | B (2.8) | 12 | on | 264 | 264 | 0 / 0 / 0 | 628 | 192 | 5,535 MiB |
| ta2 | A (2.9) | 12 | on | 258 | 258 | 0 / 0 / 0 | 611 | 193 | 2,366 MiB |
| **Total A** | | | | **515** | **515** | 0 | 1,142 | 361 | |
| **Total B** | | | | **530** | **530** | 0 | 1,159 | 353 | |

- No failures on either pack. With 0 failures in 515 and 530 requests, the failure rate of each pack is below about 0.6 % (95 % upper bound, rule of three). The test cannot find a smaller difference.
- The harness checks the form of each call: the call is parsed, the tool is offered, and the arguments match the schema. It does not score whether the model picked the best tool or the best argument values.
- No memory trip. Pack B kept the head 2.8-3.1 GiB higher at the low point.

**Decision on compensation work: none now.** The tool-call form shows no gap between the packs, so block-wise tuning or EoRA has no measured problem to fix. A gap in content quality is still possible. Two tests can find it: a KL measurement against the native release (needs an API key on the head) and a scored tool benchmark (tool-eval-bench, needs approval to install).

## Speed

Decode (`v41_bench.py`, 512 tokens a stream, aggregate tok/s):

| Workload | A | B | Change |
| --- | ---: | ---: | ---: |
| Prose, 1 stream | 54.0 | 55.1 | +2.0 % |
| Prose, 4 streams | 101.7 | 103.5 | +1.8 % |
| Code, 1 stream | 97.6 | 94.8 | -2.9 % |
| Code, 4 streams | 148.7 | 152.3 | +2.4 % |

One run a side. The changes are inside the run-to-run noise of this bench (about 3 %).

Prefill, cold (prefix cache 0), median of 3:

| Prompt | A: TTFT / prefill | B: TTFT / prefill |
| --- | --- | --- |
| 32,000 tokens | 16.23 s / 1,973 tok/s | 16.07 s / 1,993 tok/s |
| 32,768 tokens (kit bench) | 14.51 s / 2,258 tok/s | 14.47 s / 2,265 tok/s |
| 24K document, cold / warm (prefix cache hit 22,528) | 11.42 s / 1.04 s | 11.07 s / 1.11 s |

Long 4-stream run (20K, 50K, 100K and 200K prompts, 1,500 new tokens each): TTFT 9.0-251 s on B against 9.1-245 s on A. The output hashes of B are the same in both B runs.

## MTP acceptance

Accepted drafts by depth, from the engine report after the same MTP block (prose, code and thinking at 1 and 4 streams, temperature 1.0):

| Depth | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (run a2) | 74.5 % | 80.3 % | 85.9 % | 89.3 % | 90.3 % |
| B (run b2) | 71.7 % | 78.4 % | 80.6 % | 87.4 % | 91.3 % |

One-stream samples, decode tok/s (accepted / drafted):

- prose: A 57.6 (247/438), B 58.5 (248/440);
- code: A 96.9 (393/494), B 88.5 (371/529);
- thinking: A 71.9 (653/1064), B 69.2 (629/1065).

The MTP layers of pack B are 4-bit. The lower code acceptance in one sample did not show in the decode bench.

## Step 1: compatibility

- **Expert widths:** the 1.1.1 loader reads all 192,506 tensors of pack B. No tensor is missing or bad.
  - The routed experts are EXL3 with a bit width per expert matrix (codebook mul1).
  - Main layers: 2-bit 14,732 matrices, 3-bit 27,328, 4-bit 3,836, 5-bit 184. MTP: all 4-bit (1,152).
  - The universal EXL3 expert kernels of 1.1.1 have tests for mixed widths.
- **Plain tensors (the blocker):** 100 plain tensors use other dtypes than pack A:
  - `ffn.gate.weight` is BF16 (A: F16);
  - `ffn.gate.bias` is F32 (A: F16);
  - `attn.indexer.weights_proj.weight` is BF16 (A: F16);
  - Engram `q_weight` and `k_weight` are BF16 (A: F32);
  - `mtp.2.markov_head.head.weight` is BF16 (A: F16).
  The first start (run b1) failed in the Markov draft kernel: "Both operands must be same dtype. Got bf16 and fp16".
- **Fix with no engine change:** `zz-dtype-override.safetensors` (220 MiB) in the model view of B, on both nodes. It holds these 100 tensors in the dtypes of pack A.
  - The loader reads files with "override" in the name last, so these entries replace the pack's entries.
  - BF16 to F16 and BF16 to F32 are exact, except for a few hundred subnormal values a matrix. F32 to F16 rounds the gate bias, which pack A also stores as F16.
  - The pack files do not change. The rank cache key includes the file names, so the old cache file was replaced.
- **Extra tensors:** pack B also holds 11 plain compressor and indexer weights (layers 2, 8, 14 and 20). Both engines use the EXL3 forms, so the loader ignores them.

## Files

- **Pack (sha-verified, 47/47 files on each node):** `~/models/deepseek-v41-1017-exl3-2.8bpw-jl`, 190 GiB a node. It stays on both nodes.
- **Python view:** `~/models/deepseek-v41-1017-exl3-2.8bpw-jl-view`. It has hard links to the pack, our patched chat template, our 1017 Engram tables and the override file.
- **Python rank cache:** `~/tfv41-cache/tf-ds-rank-jl`, about 96 GiB a node. The production cache `tf-ds-rank` did not change.
- **Logs:** `~/pk28/logs` on the head: run logs, memory samples at 1 s, suite, bench, prefill, tool-call and MTP files.

## Recommendation

**Adopt pack B.** The user decided this on 2026-10-22:

- The Zig engine + pack B becomes the default in the recipe and the fork.
- The Python engine + pack A stays as a documented rollback.

The Zig session has added pack-B support (router and confidence head from BF16 to FP16 at load). Its GPU bring-up and the release fault campaign come next. For the Python engine, pack B needs the override file, or a dtype cast in the loader (a small fork change, about 2 h with tests).

Open points:

1. MTP acceptance is 2-3 points lower at depth 1. Measure it again on the Zig engine.
2. The pack author's fidelity table compares the 2.8 pack (against 1017) with a September 2.9 bpw pack (against September), not with our 1017 2.9 bpw pack. So its tool-call KL ratio (x1.07) does not compare the two packs of this test. The paired tool-call A/B (515 and 530 requests, 0 failures on both packs) shows no gap in call form. A paired KL measurement against the native 1017 release is still open.
3. One decode bench run a side. Repeat it on the Zig engine before the recipe numbers are published.
