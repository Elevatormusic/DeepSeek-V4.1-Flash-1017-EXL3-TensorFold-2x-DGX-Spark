# Display-backed KV with SSD spill

This is an opt-in dependent image. It adds the display-safe SSD spill tier to
the display-KV experiment. The ordinary `start.sh`, `.env` and base image stay
unchanged. Full two-node hardware validation is still pending.

## Dependency

This image needs the display-KV feature. The composition installer validates the
union of pristine source hashes once, applies the display transform first, then
the spill transform, checks every generated Python file, and only then writes. The spill transform also removes
the guard that makes the display-only image refuse a nonzero `TF_DS_SPILL_GIB`.
Do not run the two standalone installers in sequence on modified source.

## Build and use

Build from the repository root so both feature folders are in the context:

```bash
docker build -f experiments/display-spill/Dockerfile \
  --build-arg BASE_IMAGE=tensorfold-dsv41:07aa02585f64 \
  -t tensorfold-dsv41:display-spill .
```

Stop both ranks before a cutover:

```bash
./stop.sh
./start-display-spill.sh --build-only
./start-display-spill.sh
./verify.sh
```

`--build-only` builds and copies the image without DRM changes, probes or a
server. The launcher refuses active ranks, prepares DRM on both nodes, probes the
full display reservation, and starts a private bundle with `--spill`. It keeps
the `SPILL_GIB` value from `.env`; set it to `0` to disable disk writes.

To return to the ordinary engine after all GPU and DRM clients stop:

```bash
./stop.sh
./start-display-spill.sh --restore-drm
./start.sh
```

Never stop GDM. Never use `docker exec` in a serving rank. Run only one model
server at a time. The lock `.display-kv.lock` serializes only the two display
launchers. The ordinary `./start.sh` is unchanged and does not take it, so do not
run `./start.sh` or `./stop.sh` while a display launcher is active. Keep `BIND_HOST=127.0.0.1` unless network access and an API key
are explicitly configured.

## Spill settings

| Setting | Default | Effect |
|---|---|---|
| `SPILL_GIB` | `64` | Maximum stored prompt bytes per rank for the current build. `0` disables writes. |
| `TF_DS_SPILL_KEEP_BUILDS` | `1` | Old build folders that stay on disk after a change of image, weights or settings (for a rollback). From `0` to `64`. A build folder with no finished file on this rank (for example, from a start that failed) takes no slot and is removed. With two ranks, the kept builds are the newest ones that both ranks have, in rank 0's order, so a rollback finds the same build on both. A link or a FIFO in a rank folder is removed and never opened. A restore also opens its file without following links and without blocking, and refuses anything that is not a regular file, so a file swapped after the start cannot stop the lane. If a rank fails while it evicts entries for a save, all ranks agree their indexes again before they go on. The spill folder itself must not be a symbolic link (use the real path). A file that cannot be removed when a stored prompt is evicted is tried again at the next save, and the save that needed its room is skipped on every rank. A rank folder that is a symbolic link is never followed. Each can hold up to `SPILL_GIB`, so the disk use per rank can reach `(1 + this value) x SPILL_GIB`. Set `0` to remove old folders at start. |
| `TF_DS_SPILL_MIN` | `4096` | Shortest prompt written to disk. |
| `TF_DS_SPILL_MIN_FREE_GIB` | `50` | No write leaves less disk free on any rank. Zero keeps no reserve; a negative value is refused. |
| `TF_SPILL_WRITERS` | `2` | Writer threads per rank. |
| `TF_SPILL_DIRECT` | `1` | Use O_DIRECT where the file system supports it. |
| `TF_SPILL_FLUSH_S` | `60` | Shutdown drain budget. Each rank waits at most the smaller of this value and 55 seconds. |

Put a setting in `.env` to change it. The private bundle passes `TF_DS_SPILL_MIN`, `TF_DS_SPILL_MIN_FREE_GIB`,
`TF_DS_SPILL_KEEP_BUILDS`, `TF_SPILL_WRITERS`, `TF_SPILL_DIRECT` and `TF_SPILL_FLUSH_S` to both ranks, only when
.env sets them. A write starts when a prompt leaves the kept set (write-through); there is no early-write threshold.
The store writes safetensors files with a CRC-32 per 8 MiB, fsyncs, and renames
from `.partial`. Token ids are stored as base64 int64. The header holds a CRC-32 of itself (`hcrc`, format 4), so a
flipped bit in a count, a name or a flag is refused when the file is scanned or read. The declared header length is
checked first (at most 64 MiB, aligned, inside the file), so a flipped bit cannot make a rank read a huge block. A corrupt or short file is dropped. The
table of tensor descriptors is checked before any memory is allocated: gaps, overlaps, negative numbers and
unknown dtypes are refused. The compatibility key holds the NVIDIA driver (read from the proc file, NVML or the CUDA driver API; if none answers, the tier stays off), the cache planes, the fingerprint of the model files and of the Engram files (`TF_DS_ENGRAM`), the draft rings (so a change of `MTP_DRAFTS` starts a new folder) and the
settings that change a cached row. At start, every rank first tells the others if it could prepare the tier (settings, pinned buffers, key); if one rank
failed, the tier is off on every rank, and the same holds if the store fails to start on one rank. Every rank agrees on the index and on each read. A read is used only when it
worked on every rank. If a write failed on one rank only, `find()` keeps the entry, so the read fails on all
ranks and all of them forget it. A failed read is counted once as `load_failed`.
If a rank cannot prepare a save (for example, an encode error), it tells the other ranks in the size exchange, and every rank skips that save together. At start, a smaller `SPILL_GIB` than the last run removes the oldest stored prompts until the rest fit. The image build copies only the named files of the two folders, so a private `bundle-launch-*` folder in the repository root never enters the image. When a save must free disk for the free-disk floor, it first waits (up to 120 s) for the queued writes, because an entry that is still being written frees no bytes until its write ends. The ranks then measure the shortage again and agree on it; if one rank still has queued writes, every rank skips the save. The writer threads start with the store and each reports that its stream and checksum thread are ready, so a failure to start them is part of the start-up agreement. A write that was evicted while queued stays in the queue accounting (inflight limit, free-disk floor and shutdown drain) until it ends. Everything a rank can fail on while it sets up a save (prompt key, encode, entry, CUDA event, free-space probe) happens before the one size exchange, so a failure is shared and the ranks skip the save together. A free-space probe that fails is not taken as an empty disk. After a save evicts entries for the free-disk floor, the ranks measure the free space again and agree that it is enough; a file that could not be removed (counted as `unlink_failed`) stops the save. Each writer starts its checksum thread before it reports ready. At start, a kept old build loses its unfinished `.partial` files. The spill launcher and the display-KV launcher use one lock file, so only one of them can change the display driver at a time. Rank 0 prints the start line only after all ranks agreed that the tier is on, and each rank that did not fail says that another rank could not start it. A reader thread that cannot start is tried again on the next read, and a reader without its resources fails the read, so the ranks fall back together. The private bundle sets `MEM_CAP=0`, because these images do not carry the memory-topk label.

On SIGTERM (docker stop), rank 0 and rank 1 each wait for their queued writes (the shutdown drain). A second SIGTERM
during the drain is ignored, so it does not cut the writes. Rank 1
also drains when rank 0 closes the link first, because `stop.sh` stops rank 0 first.

## Validation limits

Offline tests use CPU torch and temporary directories. They do not prove a real
two-rank write, eviction, restore and correct reply. That live gate is pending and runs in the next maintenance
window (`./verify.sh` and `tests/run_all.sh` with this image on the pair; save the output under `runs/`). Also test restart restore, a pending write during SIGTERM, one-rank
read/write failure, corrupt entries, low disk space, disabled mode, and no
collective hang. Record counters that prove disk writes and loads happened.
No performance claim is made.

## Installer protocol

`apply.py` exposes `REVISION`, `FILES`, `transform(relative, text)`, `overlays()`
and `install(checkout)`. `compose.py` combines it with the display installer.
`FILES` contains only the pristine files this feature changes. The spill module
is a new overlay; an existing file is rejected.

For a disposable source checkout:

```bash
python3 experiments/display-spill/compose.py "$CHECKOUT" \
  experiments/display-kv/apply.py experiments/display-spill/apply.py
```

The composer checks the exact revision, all pristine hashes, overlay absence and
all generated syntax before the first write. It does not provide rollback for an
I/O failure during writes. Use a disposable checkout.
