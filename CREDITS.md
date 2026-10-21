# Credits

This recipe stands on other people's work. The table lists what each project gives, its licence, and how this repo uses it. This repo copies no file from any project below, except one test fixture (see "Tool-output guard experiment" below); it downloads or builds each one at a pinned version.

| Project | What it gives | Licence | How this repo uses it |
|---|---|---|---|
| [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek | the model, its Engram tables (shards 47/48), and the reference prompt encoder (`encoding/`) | MIT | `scripts/download.sh` downloads the Engram shards at revision `1bae0e10`; `tests/v41_differential.py` compares our prompts with the reference encoder (download it yourself, see tests/README.md) |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | the EXL3 2.9 bpw quantization that this recipe serves (196 GiB) | MIT | downloaded at a pinned revision |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp | the EXL3 format | MIT | through TensorFold's EXL3 kernels |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | the vLLM kit for the same weights on 2x Spark: our baseline, and the chat template | AGPL-3.0 | the baseline numbers come from it at commit `6f7d1590`; `scripts/patch_template.py` downloads its `files/chat_template.jinja` at that commit and applies our small patch (1 hunk) on your machine (the template is not in this repo) |
| [TensorFold](https://github.com/ashhart/TensorFold) by ashhart and contributors | the engine framework | Apache-2.0 (code before 0.6.0: MIT) | the base of the engine image |
| The DeepSeek-V4.1 TP2 engine (`deepseek_v41` family) by [bertholomus](https://github.com/bertholomus) (Bertholomus AI), with later fixes by willjriley, Capicua25x and hyungraekim-bagelcode (the `json_schema` prompt, fork PR #27) | the V4.1 model code, its kernels, DSpark drafting, Engram, the RDMA path between the nodes: all of the speed | Apache-2.0 | the Python engine line of [their recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) (v0.5.1), built from the fork [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) at commit `cb4ceaff2b8c` (release dsv41-1.1.1), which adds the changes in [docs/WHAT-I-CHANGED.md](docs/WHAT-I-CHANGED.md) |
| [xgrammar](https://github.com/mlc-ai/xgrammar) | the grammar engine behind structured output | Apache-2.0 | installed with TensorFold's `grammar` extra |
| [NVIDIA NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) | the CUDA 13 / sm_121 base image | NVIDIA Deep Learning Container License | `docker/Dockerfile` builds on `nvcr.io/nvidia/pytorch:26.07-py3` |
| [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | the benchmark method (usage-based token counts, prose and code apart, concurrency cells) and the report layout | MIT | method only; no file copied |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research | the agent client that the release gate drives | MIT | `tests/hermes_thinking_modes.py` imports your local Hermes install |

The OpenAI Codex review bot reviewed every engine change; its findings and our fixes are on the three fork pull requests.

## Display mapping experiment

The optional `experiments/display-kv/overlay/display_kv.py` adapts Gilesco's display-memory mapping technique and credits coolbho3k's display-reserve allocator. See [the display NOTICE](experiments/display-kv/NOTICE.md) for the source references and license notes. This display-only variant does not include the separate SSD spill port or NVFP4 C toolkit.

**0rand** wrote the display-KV variant. **Sora39831** and **hjasond** tested it on their own nodes; the measured runs of Sora39831 are in `runs/2026-10-19/` ([record](runs/2026-10-19/display-kv-tester-ab.md)).

## Tool-output guard experiment

The optional `experiments/tool-output/` holds two test files taken from TensorFold at revision `07aa02585f64` (Apache-2.0): `tests/fixtures/http.py.txt` is an unchanged copy of `src/tensorfold/cuda/http.py` (used by the installer tests), and `tests/fixtures/parser.py` is a reduced copy of the reply-text functions. See [the NOTICE](experiments/tool-output/NOTICE.md).

## This repo's licence

Apache-2.0 (see [LICENSE](LICENSE)), the same as TensorFold. It covers the scripts, tests and documents in this repo only. The model weights, the chat template and the engine keep their own licences.
