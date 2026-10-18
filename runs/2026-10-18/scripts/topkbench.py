"""Time a prompt chunk's indexer selection: the old path (64-key tiles) against the 8-key tiles, same inputs."""
import sys

import torch

from tensorfold.families.deepseek_v41.cuda import kernels as K

IH, ID, TOPK = 32, 128, 512


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


for n in [int(x) for x in sys.argv[1:]] or [8192, 16384, 24576, 32768, 49152]:
    rows = max(16, min(2048, (1 << 26) // (4 * n)))     # model.py's row block
    q = torch.randn((rows, IH, ID), device="cuda").to(torch.bfloat16)
    codes = torch.randint(0, 256, (n, ID // 2), device="cuda", dtype=torch.int32).to(torch.uint8)
    scales = torch.randint(120, 131, (n, ID // 32), device="cuda", dtype=torch.int32).to(torch.uint8)
    w = torch.randn((rows, IH), device="cuda").to(torch.bfloat16)
    pos = torch.arange(n - rows, n, device="cuda") * 2
    vis = ((pos + 1) // 2).clamp(1, n)
    k = (codes, scales)

    def old():
        keys, tm = K.index_score(q, k, w, vis, n, keys=True, tmax=True)
        return K.topk_select_pruned(keys, tm, TOPK, vis)

    tile = K.prompt_tile(rows, n, TOPK)

    def new():
        keys, tm = K.index_score(q, k, w, vis, n, keys=True, tmax=True, tile=tile)
        return K.topk_select_pruned(keys, tm, TOPK, vis, tile)

    def score_only():
        return K.index_score(q, k, w, vis, n, keys=True, tmax=True, tile=tile)

    same = torch.equal(old(), new())
    to, tn, ts = timed(old), timed(new), timed(score_only)
    print(f"n={n:6d} rows={rows:5d} tile={tile:2d} old {to:7.2f} ms  new {tn:7.2f} ms  (score {ts:6.2f})  "
          f"x{to / tn:4.2f}  same={same}", flush=True)

print("candidate layers (fp32 scores, ratio 1):", flush=True)
for n in [16384, 32768, 49152, 65536]:
    rows = max(16, min(2048, (1 << 26) // (4 * n)))
    score = torch.randn((rows, n), device="cuda")
    keep = (torch.rand((rows, -(-n // 8)), device="cuda") < 0.25).repeat_interleave(8, dim=-1)[:, :n]
    score.masked_fill_(~keep, float("-inf"))
    vis = torch.arange(n - rows, n, device="cuda") + 1
    tile = K.prompt_tile(rows, n, TOPK)

    def old_c():
        top = K.topk_indices(score, TOPK)
        return torch.where(top < vis[:, None], top, -1)

    def new_c():
        keys, tm = K.score_keys(score, tmax=True, tile=tile)
        return K.topk_select_pruned(keys, tm, TOPK, vis, tile)

    same = torch.equal(old_c(), new_c())
    to, tn = timed(old_c), timed(new_c)
    print(f"n={n:6d} rows={rows:5d} tile={tile:2d} old {to:7.2f} ms  new {tn:7.2f} ms  x{to / tn:4.2f}  same={same}",
          flush=True)
