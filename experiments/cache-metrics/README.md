# Opt-in cache metrics

This experiment adds read-only cache accounting to `/health` and `/metrics`.
It does not change the ordinary engine image or launcher defaults.
It needs no display allocation or spill module.

## Build and select

The experiment overlay needs the pristine release 1.0.0 engine `07aa02585f64`, not the engine that `docker/build.sh` builds now (`cb4ceaff2b8c`). Build the 1.0.0 base first. Run this command from the recipe root; it does not use `.env`:

```bash
docker build --build-arg TF_COMMIT=07aa02585f64 -t tensorfold-dsv41:07aa02585f64 docker
```

Then run this command from the recipe root:

```bash
docker build -t tensorfold-dsv41:07aa02585f64-metrics experiments/cache-metrics
```

Stop the current server before you select another image.
Copy the new image to the worker. `docker/build.sh` cannot do this: it rebuilds the ordinary image first. Use this helper, which copies the tag that you built and checks that both nodes hold the same image ID:

```bash
experiments/cache-metrics/transfer.sh tensorfold-dsv41:07aa02585f64-metrics
```

Set `IMAGE=tensorfold-dsv41:07aa02585f64-metrics` in your local `.env` on the head.
Keep `BIND_HOST=127.0.0.1`. Start only one model server.
Run `./verify.sh`. Check `/health` and `/metrics` on the local API port.
Use the authentication settings in [CONNECT](../../docs/CONNECT.md).
To return to the ordinary engine, stop the server and restore the original `IMAGE`.

This image has not had a live two-rank serving check. The offline tests, including the transfer helper, use fixtures only. Before you call the experiment supported, the maintainer runs `./verify.sh` and `tests/run_all.sh` with this image on the two-node pair and saves the output under `runs/`. Until then, do not use the image on a node that serves production traffic.
No performance claim is made.

## Accounting

All cache fields describe the local rank, not the sum of both ranks.
Pool tokens count reserved positions. Free plus used equals total.
Kept tokens count prefix cache reservations within that pool.
`kept_bytes` counts unique retained snapshot tensor objects outside the pool.
`kv_cache_payload_bytes` counts unique compressed KV and indexer tensor objects.
Tensor views with different Python identities are not storage-deduplicated.
These byte counts exclude weights, other cache planes and allocator overhead.
Reads use tensor metadata. They do not allocate tensors or synchronize the GPU.

Ordinary pools report CUDA backing. Optional `display_arena.size` metadata reports
mapped display capacity instead. This experiment does not create that arena.
A missing arena or spill protocol is valid. A callable spill `info()` can add an
optional health field. This experiment does not export spill Prometheus families.

All new Prometheus accounting families are gauges. Bytes use bytes, positions use
tokens, and occupancy uses a ratio from zero to one.
`kv_pool_usage_ratio` measures reserved pool positions, including kept prompts.
`kv_cache_usage_ratio` measures the maximum live stream length divided by its
request context. Partial prefill uses `filled`, not the full prompt length.
An idle stream ratio can be zero while pool reservations remain nonzero.
Unsupported engines keep their original metrics and do not report invented cache fields.
Scheduler collections can change between reads. The snapshot is not atomic.

## Installer protocol

`apply.py` exposes `REVISION`, `FILES`, `transform(relative, text)`, `overlays()`
and `install(checkout)`. `FILES` contains only the three pristine files that this
experiment changes. Overlay keys are relative target paths.

For a separate disposable source checkout, run:

```bash
python3 experiments/cache-metrics/apply.py "$TF_METRICS_SOURCE"
```

The installer checks the exact git revision and all pristine SHA-256 hashes.
It rejects an existing overlay. It checks all generated Python with `ast.parse`
before the first write. It rejects repeated transformation and changed anchors.
It does not provide rollback for an I/O failure during writes. Use a disposable checkout.

An external composer can validate pristine hashes for all selected experiments,
apply their transforms in order in memory, merge non-conflicting overlays, check
all generated syntax, and then write. Do not call standalone installers in sequence
on modified source. This PR does not add a shared composition framework.

## Offline tests

```bash
tests/run_offline.sh
TF_METRICS_SOURCE="$CHECKOUT" python3 -m unittest discover -s experiments/cache-metrics/tests -v
```

`CHECKOUT` must refer to explicit pristine pinned source. Tests only read it.
Without `TF_METRICS_SOURCE`, source integration tests skip. All other tests use
the Python standard library and need no TensorFold install, server or GPU.
