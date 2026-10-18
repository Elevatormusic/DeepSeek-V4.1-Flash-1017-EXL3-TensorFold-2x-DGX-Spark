# Runs of 18 October 2026 (UTC): long prompts and the 1M context

Same pair of DGX Spark as `runs/2026-10-17/`. All prefill rows use bertholomus's `kit_bench.py prefill` (random-word prompts, so no prompt-cache hit; `cached_tokens` is 0 in the server log) through `scripts/src127.py` (it connects from 127.0.0.1). `scripts/pf.sh TAG` runs one 8K warm prompt, then 8K two times, then 32K, 64K and 128K once each. Speed is `prompt_tokens / time to first token`.

## Current setup: indexer selection with 8-key tiles (fork PR #7)

Image `tensorfold-dsv41:3c45bdaeab7d` (fork PRs #1-#4, [#6](https://github.com/Elevatormusic/TensorFold/pull/6) and [#7](https://github.com/Elevatormusic/TensorFold/pull/7)), the same host settings. The model has 8 indexer layers: 2, 8 and 14 score their keys in the score kernel (the keyed path); 20, 24, 28, 32 and 36 score them in fp32 and apply the candidate pool. Before PR #7, both paths used PyTorch's top-k (8 radix passes over the int64 keys) below 32K keys a row, because the tile search with 64-key tiles needs more than k = 512 tiles. PR #7 keeps the largest key of each 8 keys, so the tile search runs from 4K keys and both of its selections fit one sort launch. The selection is the same set.

Selection timing for one row block (`nsys/` kernel table: PyTorch's top-k was 6.2% of the GPU time of a 64K prompt):

| Keys a row | Keyed layers, score + select (ms) | Candidate layers, select (ms) |
|---:|---|---|
| 16,384 | 11.2 → 8.3 | 11.6 → 3.4 |
| 32,768 | 11.2 → 7.5 | 11.5 → 2.6 |
| 65,536 | | 11.3 → 1.9 |

Prefill, 3 passes on the PR #7 code (mounted over image `07aa02585f64`, the same bytes as `3c45bdaeab7d`):

| Run | 8K (both reps) | 32K | 64K | 128K | Slow chunks (> 1.3 s) |
|---|---|---:|---:|---:|---:|
| topk8c, pass 1 | 2,085 / 2,179 | 2,213 | 2,106 | 2,054 | 1 (the first chunk after the start) |
| topk8c, pass 2 | 2,175 / 2,178 | 2,196 | 2,154 | 2,064 | |
| topk8c, pass 3 | 2,174 / 2,178 | 2,192 | 2,152 | 2,061 | |
| **Median** | **2,178** | **2,196** | **2,152** | **2,061** | |
| Before (odirect2, below) | 2,166 | 2,179 | 2,110 | 1,989 | 5 |

Release gate on this code: suite 32/32, grammar check 14/14 (`gate/suite-topk8c.log`, `gate/grammar-check-topk8c.log`). The first try (`topk8`) failed at the first request with a CUDA out-of-memory error: the mounted files were newer than the built CUDA extensions, so the engine rebuilt them at start and had only 2.2 GiB available at warm-up. A run with only the keyed layers changed (`topk8b`) gave 2,192 / 2,124 / 2,050 at 32K / 64K / 128K.

Deploy of the built image `3c45bdaeab7d` (CUDA extensions compiled first, with the engine stopped): suite 32/32 in 244 s, grammar check 14/14 (`gate/suite-3c45bdaeab7d.log`, `gate/grammar-check-3c45bdaeab7d.log`). One prefill pass right after the gate (`topk/deploy3c/`): 8K 2,173 / 2,165, 32K 2,166, 64K 2,136, 128K 2,053.

## Previous setup: compaction stalls removed (06:00-09:15 UTC)

Image `tensorfold-dsv41:07aa02585f64` (fork PRs #1-#4 and [#6](https://github.com/Elevatormusic/TensorFold/pull/6): spill files written with `O_DIRECT`), the host settings below plus transparent huge pages off (`scripts/host_tuning.sh` item 4), `TF_DS_MEM_FLOOR_GIB=1.5`, the prefix-cache spill on (`TF_DS_SPILL_GIB=64`, as the recipe ships).

| Run (3 passes) | 8K (both reps) | 32K | 64K | 128K | Slow chunks (> 1.3 s) |
|---|---|---:|---:|---:|---:|
| odirect2, pass 1 | 2,189 / 2,186 | 2,195 | 2,115 | 1,976 | |
| odirect2, pass 2 | 2,166 / 2,166 | 2,179 | 2,110 | 1,989 | |
| odirect2, pass 3 | 2,163 / 2,165 | 2,174 | 2,110 | 1,991 | |
| **Range** | **2,163-2,189** | **2,174-2,195** | **2,110-2,115** | **1,976-1,991** | 5: 2 in the first prompt after the start (3.16 s, 1.64 s), 3 at 1.31-1.32 s (the last chunk of a 128K prompt is always ~1.3 s) |

Files: `compaction/odirect/` and `compaction/odirect2/` (two starts of the same code; `odirect` had an image without xgrammar, which does not change prefill). Release gate on this setup: suite 32/32 in 249 s, grammar check 14/14.

### What caused the slow chunks

The GPU of a GB10 reads host memory through the SMMU. When the kernel moves pages to make large free blocks (compaction), the GPU waits, even when another process asked for the compaction. `scripts/tracenode.sh` ran `scripts/memtrace.bt` (bpftrace on `mm_compaction_*` and `mm_vmscan_direct_reclaim_*`), `scripts/hwsample.sh` and a thread sampler on both nodes during each run; `scripts/profsum.py` puts the events next to the chunk log.

| Run (3 passes each, `compaction/<run>/`) | Spill | THP | Slow chunks | Compaction in the run (head) | 32K | 64K |
|---|---|---|---:|---|---:|---:|
| `base` (image 92aa710b533d) | buffered | madvise | 19 | 59 s, 55 s of it inside slow chunks (spill writer threads) | 1,746-1,769 | 1,492-1,747 |
| `pin` (+ busiest engine threads on the fast cores) | buffered | madvise | 20 | 4.9 s, none inside slow chunks | 1,944-2,128 | 1,970-2,111 |
| `nospill` (+ pinned) | off | madvise | 10 | 14.1 s by the router healthcheck's `python`, 3.7 s by khugepaged, inside the large slow chunks | 2,192-2,196 | 1,989-2,126 |
| `nothp` (+ pinned) | off | never | 1 (1.31 s) | 0 | 2,186-2,210 | 2,113-2,128 |
| `spillnothp` | buffered | never | 14 | 15.9 s by kcompactd (page-cache folios of the spill writes) | 2,069-2,193 | 2,016-2,052 |
| `odirect2` (current) | `O_DIRECT` | never | 5 (2 in the first prompt, 3 at 1.31-1.32 s) | 0.05 s | 2,174-2,195 | 2,110-2,115 |

- **The spill writes.** `safetensors.save_file` wrote through the page cache, and ext4 asked for order 7-9 folios. Kernel stacks (`compaction/compstack-64k.txt`, `scripts/compstack.bt`): `try_to_compact_pages` < `filemap_alloc_folio` < `ext4_da_write_begin` < `vfs_write`, user stack `safetensors::tensor::serialize_to_file`.
- **Transparent huge pages.** The LiteLLM router container's Docker healthcheck starts `python` every 10 s. Its start-up, and khugepaged, made synchronous compactions of 2.4-4.7 s, each at the same moment as a slow chunk of the same length (`compaction/nospill/mem-head.txt`, `compaction/memstall-psi.txt`). The engine does not use huge pages.
- **Not the cause:** the CPU core type. The busiest engine thread ran on a Cortex-X925 core in every sample (`threads.txt`), and the GPU clock stayed flat. The `pin` run looked better only because its compaction was lower.

### Two start problems found on the way

- **CUDA extensions after a new image.** The engine compiles them at its first start, after the weights (~99 GiB) are loaded. On the worker the compiler took the node below 1 GiB, and earlyoom stopped the rank; the killed build left a lock file, and the next build waited on it. `scripts/build_extensions.sh` (run by `start.sh`) now compiles them before the weights load, on both nodes, and removes leftover locks.
- **Allocator headroom.** After the warm-up, the engine may grow only while `TF_DS_MEM_FLOOR_GIB` stays free. With the default 2.0 and 3.3 GiB available after the warm-up, the second chunk of an 8K prompt failed (CUDA out of memory in the Engram gather, 400 MiB). `start.sh` now sets 1.5, above earlyoom's 1 GiB limit.

### Decode with THP off

`kitbench-decode/decode-{a,b}-nothp.json`, greedy, 3 replies per prompt, same texts as before. The slow decode state is gone: all 3 code replies of set a run at 84.7-85.1 tok/s (before: 69.5 / 71.0 / 85.2 in `decode-a-ht.json`).

| Set | Code | Prose | Structured |
|---|---|---|---|
| a | 84.7 / 85.1 / 84.8 | 58.8 / 58.6 / 58.6 | 103.9 / 104.0 / 103.5 |
| b | 98.3 / 98.6 / 98.7 | 56.8 / 56.8 / 57.0 | 140.0 / 139.5 / 139.7 |

### Kernel profile of a 64K prompt (nsys)

`nsys/`: 6 s in the middle of one cold 64K prompt, both ranks, `scripts/nsys64k.sh`. Nsight Systems on this GB10 records no PyTorch kernels with its default hardware CUDA trace; `--trace=cuda-sw` (software trace) records them. Stats were made with the image's nsys 2026.3. Rank 0, share of GPU kernel time (`nsys/r0-top.txt`; rank 1 is within 1 point):

| Part | Kernels | Share |
|---|---|---:|
| MoE experts (EXL3) | `grouped_mma`, `grouped_rows`, `rot_in` (bf16), `down_combine`, `gateup_epilogue`, `unpack` | 37.7% |
| Prompt-chunk all-gather (NCCL, 2 ranks) | `ncclDevKernel_AllGather_RING_LL` (the engine sets `NCCL_PROTO=Simple`; NCCL names its shared kernel `_LL`) | 12.4% |
| hc (hyper-connections) | `_hc_post`, `_hc_partial`, `_hc_finish` | 11.3% |
| Dense matmuls (Triton) | `_gemm`, `rot_in` | 11.6% |
| Indexer (ratio-2 attention) | `_index_score_rows` | 6.5% |
| Indexer top-k (PyTorch radix select) | `mbtopk::computeBlockDigitCounts`, `gatherTopK` | 6.2% |
| Sparse attention | `_sparse_attn_part` | 5.9% |
| fp32 SIMT matmul (no tensor cores) | `cutlass_80_simt_sgemm_128x256` | 1.4% |

The kernels fill the window (6.22 s of kernel time in 6 s, with the NCCL kernel on its own stream), so the CPU does not hold back prefill. Candidates, largest expected gain first:

1. **Overlap the all-gather with compute.** It is on the critical path (12%). Split each 2048-token chunk in two and gather one half while the other half computes; or gather bf16 partials in place of fp32 (half the bytes; needs a quality check).
2. **Fused top-k for the indexer.** PyTorch's radix top-k takes 6.2%, about as long as the scoring kernel itself. A top-k inside `_index_score_rows` (or a per-row threshold select) removes most of it. This part also grows with position.
3. **Fuse the hc kernels** (11.3% in 3 kernels and ~750 launches in 6 s).
4. **fp32 SIMT matmul to TF32 or bf16 tensor cores** (1.4%, small but simple).
5. **MoE expert kernels** (38%): the largest part, but already hand-written; ncu is the next step there.

## Older setup (image 92aa710b533d)

Image `tensorfold-dsv41:92aa710b533d` (fork PRs #1-#4), `TF_DS_ENGRAM_DONTNEED=1`, `--oom-score-adj 1000`, and the host settings of `scripts/host_tuning.sh`: no swap for the engine, `vm.swappiness=10`, `vm.compaction_proactiveness=0`, `vm.extfrag_threshold=500`, `vm.min_free_kbytes=1048576`, `vm.watermark_scale_factor=50`, and earlyoom set to stop the engine below 1 GiB available.

| Run | 8K (best of 2) | 32K | 64K | 128K | File |
|---|---:|---:|---:|---:|---|
| hw5 | 2,214 | 2,201 | 2,119 | 1,986 | `kitbench-prefill/pf-hw5-*.json` |
| hw6 (right after hw5) | 2,174 | 2,180 | 2,109 | 1,981 | `kitbench-prefill/pf-hw6-*.json` |
| ht1 (after `scripts/host_tuning.sh` and an engine restart) | 2,223 | 2,065 | 2,037 | 1,914 | `kitbench-prefill/pf-ht1-*.json` |
| ht2 | 2,179 | 2,165 | 1,836 | 1,961 | `kitbench-prefill/pf-ht2-*.json` |
| ht3 | 2,177 | 2,100 | 1,889 | 1,851 | `kitbench-prefill/pf-ht3-*.json` |
| **Median** | **2,179** | **2,165** | **2,037** | **1,961** | |

hw5 and hw6 ran with the same host settings, set by hand before the engine restart. The 128K prompt is now steady (1,851-1,986). The 64K prompt still varies (1,836-2,119). No other request reached the server during these runs.

## Release gate (final setup)

- `gate/suite-ht.log`: 32/32 in 264 s. The 64K and 200K needles filled from new (31.1 s and 94.9 s). Before the memory changes, the 64K needle froze the head.
- `gate/grammar-check-ht.log`: 14/14.
- Lowest MemAvailable during the bench and the gate: 2,206 MiB on the head, 2,712 MiB on the worker. During the 1M needle (below): 1,779 MiB on the head. earlyoom acted at 1,536 MiB in these runs (it did not act); after the 1M run the limit went down to 1,024 MiB.

## Decode check (final setup)

`kitbench-decode/`: greedy decode, 3 replies per prompt, same text and same number of steps as on 17 October (`runs/2026-10-17`). Set a: code 84.7 (median of `decode-a-ht2.json`; `decode-a-ht.json` gave 71.0), prose 59.0, structured 104.3. Set b: code 98.8, prose 57.4, structured 139.5. In each set-a run, 1 or 2 of the 3 code replies ran at 69-71 tok/s instead of 85: decode has the same slow state as the prompts. Engine counters for the 512-token replies of set a (`kitbench-decode/r0-done-ht.log`: every rank 0 `done` line of the decode check; `rounds` = decode steps, `accepted` = accepted/drafted tokens; tokens per decode step = (rounds + accepted) / rounds): prose 1.98 (rounds 258, 253 of 436 drafted tokens accepted), code 3.60 (rounds 142, 369 of 542), structured 4.73 (rounds 108, 403 of 502); the slow code replies (69-71 tok/s) had the same 142 steps.

## Memory settings: A/B with 0.5 s samples

`scripts/hwsample.sh` sampled both nodes every 0.5 s (GPU clock, power and throttle reasons; CPU; PSI; reclaim, compaction and swap counters) during two `pf.sh` runs (248 prompt chunks) for each setting. `scripts/hwcorr.py` puts each chunk of the rank 0 chunk log (`logs/r?-chunks-*.log`) next to the samples.

| Setting | Runs | Slow chunks (over 1.3 s) | 128K tok/s |
|---|---|---:|---:|
| First version of `scripts/host_tuning.sh` (`extfrag_threshold=1000`, kernel watermarks) | hw1, hw2 | 5 of 248 | 1,962 and 1,967 |
| Background compaction on (`compaction_proactiveness=20`), `min_free_kbytes=1048576`, `watermark_scale_factor=50` | hw3, hw4 | 46 of 248 | 1,586 and 1,736 |
| Final settings (above) | hw5, hw6 | 2 of 248 (both 1.31 s) | 1,986 and 1,981 |

- The GPU clock stayed at 2,370-2,460 MHz in slow chunks, and no throttle reason was set. The GPU did not slow down.
- With the first version, slow chunks had 57 times (head) and 9 times (worker) the page scanning of normal chunks. With background compaction on, slow chunks had no extra page scanning; that cause is not known.
- The worker's swap-outs came from other processes, not the engine (polkitd held 1.7 GB of swap).

## First version of the host settings

Same image, `vm.extfrag_threshold=1000` and kernel watermarks. The spread was large: slow runs had chunks of 1.5-2.6 s among the normal 0.9-1.1 s ones (`scripts/chunksum.py` over `logs/r0-final-92aa710b533d.log`).

| Run | 8K (best of 2) | 32K | 64K | 128K | File |
|---|---:|---:|---:|---:|---|
| A (same code, mounted test copy) | 2,200 | 2,188 | 2,113 | 1,988 | `kitbench-prefill/pf-dn-async-*.json` |
| B (final image) | 2,160 | 2,180 | 2,113 | 1,202 | `kitbench-prefill/pf-final1-*.json` |
| C (final image, right after B) | 1,690 | 1,865 | 1,879 | 1,540 | `kitbench-prefill/pf-final2-*.json` |
| Median | 2,160 | 2,180 | 2,113 | 1,540 | |

## 1M context

`needle-1m.json` (`scripts/needle_1m.py 1035000 0.63`): a 1,034,753-token prompt (limit 1,048,576), random filler lines, one needle at 63% depth. Found (`ORCHID-8071`), first token after 830.4 s (1,246 tok/s), `cached_tokens` 0. Image `fd2365042f56` with swap off; chunk log in `logs/r0-20261018T015205Z.log`.

`needle-1m-final.json`: the same test on the final setup. A 1,035,020-token prompt, found (`ORCHID-6456`), first token after 817.7 s (1,266 tok/s), `cached_tokens` 0. Lowest MemAvailable during the fill: 1,779 MiB on the head, 2,542 MiB on the worker. That was only 243 MiB above earlyoom's first limit (1,536 MiB), so `scripts/host_tuning.sh` now stops the engine below 1,024 MiB (SIGKILL below 512 MiB). MemAvailable already leaves out the kernel's high watermark (about 2.3 GiB with these settings), so the kernel still has its reserve at those limits.

## How each change was found (in order)

| Step | Change | 32K | 64K | 128K | File |
|---|---|---:|---:|---:|---|
| 0 | 17 October, image `fd2365042f56` (`runs/2026-10-17/16_*`) | 1,707 | 1,457 | 1,420 | |
| 1 | Swap off for the engine, swappiness 10 | 1,872 | 1,805-1,902 | 1,672 | `e5-prefill-*.json`, `e5-counters.txt` |
| 2 | Prompt chunks of 4096 (rejected) | 1,710 | 1,619 | 1,553 | `pf-c4096-*.json` |
| 3 | Context 262K instead of 1M (rejected: no gain) | 1,676 | 1,272 | 1,280 | `pf-ctx262k-*.json` |
| 4 | Compaction down (`compaction_proactiveness=0`, `extfrag_threshold=1000`) | 2,185 | 2,038 | 1,507 | `pf-compact0-*.json` |
| 5 | + Engram page drop on the chunk's thread | 2,082 | 2,009 | 1,903 | `pf-dontneed-*.json` |
| 6 | + spill off (rejected: no gain, spill stays on) | 2,087 | 1,988 | 1,901 | `pf-dn-nospill-*.json` |
| 7 | + page drop on its own thread (fork PR #4) | 2,188 | 2,113 | 1,988 | `pf-dn-async-*.json` |
| 8 | + 32 Engram reader threads (rejected) | 2,100 | 2,050 | 1,953 | `pf-dn-t32-*.json` |
| 9 | + final memory settings (`extfrag_threshold=500`, `min_free_kbytes=1048576`, `watermark_scale_factor=50`), median of 5 runs | 2,165 | 2,037 | 1,961 | `pf-hw5-*` to `pf-ht3-*` |

What the measurements showed:

- **Per-section timing** (`TF_DS_TIMING=1`, both ranks: `logs/r?-20261018T021147Z.log`, `logs/r0-20261018T020727Z.log`). A normal 2048-token chunk: MoE ~370 ms, attention on ratio-1 layers ~172 ms, hc ~113 ms, gather ~105 ms, Engram compute ~52 ms, Engram read wait ~1.7 ms. Attention on ratio-2 layers grows with position (68 ms at the start, 195 ms at 61K). In a slow chunk every section of one rank grows (MoE 614-702 ms), and the other rank waits in the gather.
- **Kernel counters** for one 64K prompt (`scripts/vmd.sh`): with default compaction, 548 direct compaction stalls on the head; after step 4, 7. Without the page drop the worker reclaimed ~2.2M pages a 64K prompt; with it, ~0.5M.
- **Drive** (`scripts/randread.c` on the worker, idle server): 4 KiB random reads at 514K/s buffered (128 threads) and 0.9-1.6M/s with O_DIRECT. A chunk needs ~49K of them, so the drive is not the limit.
- **GPU clock and rails**: 2,360-2,500 MHz under load, no throttle reason set; both RoCE rails carry equal traffic.
