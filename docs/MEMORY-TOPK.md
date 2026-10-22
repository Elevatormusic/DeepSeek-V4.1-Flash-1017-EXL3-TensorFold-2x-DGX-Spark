# Memory and top-k

This experiment has two separate mechanisms. The ordinary image and startup
policy do not change. See [the bundle](../experiments/memory-topk/README.md) for
the installer contract and offline tests.

## Select the image

The experiment overlay needs the pristine release 1.0.0 engine `07aa02585f64`, not the engine that `docker/build.sh` builds now (`cb4ceaff2b8c`). Build the 1.0.0 base first. Run this command from the recipe root; it does not use `.env`:

```bash
docker build --build-arg TF_COMMIT=07aa02585f64 -t tensorfold-dsv41:07aa02585f64 docker
```

From the repository root, build the separate feature image:

```bash
docker build -f experiments/memory-topk/Dockerfile \
  --build-arg BASE_IMAGE=tensorfold-dsv41:07aa02585f64 \
  -t tensorfold-dsv41:memory-topk .
```

The base must contain pristine TensorFold revision
`07aa02585f64cfb04c60a96de6554c0d4708ed46`. An image with other source edits is
not a supported base. This overlay has no display dependency. Copy the same
image to the worker with the helper below. Do not use `docker/build.sh` for this: it rebuilds
the ordinary image under the same tag first, so both nodes would then lack the overlay and
preflight would still pass. The helper copies the tag that you built and checks that both
nodes hold the same image ID:

```bash
experiments/memory-topk/transfer.sh tensorfold-dsv41:memory-topk
```

Set `IMAGE=tensorfold-dsv41:memory-topk` in `.env` on the head to select it.
The default `IMAGE` and `TF_COMMIT` stay unchanged.

## Bounded top-k scratch

Selecting the feature image enables bounded top-k. It keeps the pinned kernel's
packed-key selection order and ascending output indices. Equal scores select
lower indices first. Adding zero normalizes signed zero. Scores must be float32,
as in the pinned kernel. The overlay rejects other dtypes.

The int64 key tile has a 16 MiB budget. One complete row is the minimum tile, so
a row can exceed that budget. This is not a bound on total peak memory. The
input reshape, score bits, index vector, selected keys, sort, and output also
need storage. No throughput or live peak-memory result is claimed here.

## Old policy and explicit opt-in

| Setting | Default | Effect |
|---|---|---|
| `MEM_CAP` in `.env` | `0` | `1` forwards `TF_DS_MEM_CAP=1` to both ranks |
| `TF_DS_MEM_CAP` in the engine | `0` | `1` selects the new live allocator policy |
| `MEM_FLOOR_GIB` in `.env` | `1.5` | Forwarded as `TF_DS_MEM_FLOOR_GIB` |
| `TF_DS_MEM_FLOOR_GIB` without the launcher | `2.0` | Floor in GiB. Zero disables the ceiling |

With `MEM_CAP` omitted or `0`, the launcher does not add `TF_DS_MEM_CAP`.
The ordinary image keeps its original policy. The feature image also keeps the
original policy unless `TF_DS_MEM_CAP=1`: after warm-up, the engine reads host
`MemAvailable` once and sets an allocator ceiling from the reserved bytes plus
headroom above the floor. That original path does not refresh each chunk.
`MEM_CAP=0` does not remove that original ceiling.

To test the new policy, select the feature image and set `MEM_CAP=1` in `.env`.
`./start.sh` checks the image label `tensorfold.memory-topk` on both nodes and stops before any
command if it is missing, because an image without the overlay would ignore the setting.
Only `0` and `1` are valid. The new policy reads host headroom before and after
each prefill chunk and decode round. It sets the ceiling to the smaller of the
device total and `reserved + max(0, MemAvailable - floor)`. The floor must be
finite and nonnegative. The policy does not raise small positive floors.
When headroom is below the floor plus 0.5 GiB and at least 64 MiB of allocator
cache is idle, it releases idle blocks and reads headroom again. It prints the
last budget when a step fails. It does not send signals or manage host services.

The first unreadable `MemAvailable` read fails initialization of the new policy.
A later unreadable read keeps the last budget and prints a warning. This budget
can become stale. Returning to `MEM_CAP=0` and the ordinary image requires a
normal server restart. Do not run both images at the same time.

## Tester report on the allocator policy

A tester ran the earlier combined display build, where this policy was on by default, on one pair of Spark nodes (saved in [`runs/2026-10-20/display-kv-tester-kv-budget.md`](../runs/2026-10-20/display-kv-tester-kv-budget.md), with the setup and the caveats of the tester). The maintainers did not repeat the runs. With 6 simultaneous 1M-token streams the policy ended in a CUDA out-of-memory error and exit code 75 (a rank was lost). With 3 to 5 streams and the policy off, the tester saw no lane failure, but no memory spike was large enough to make earlyoom act, so that is no proof that the policy is not needed. This is why `MEM_CAP` stays `0` by default: turn it on only for a test, and expect exit 75 near the floor.

## Validation limits

Mock tests do not prove host protection or rank recovery. No automatic SIGTERM
behaviour is assumed. The new allocator policy needs live long-prefill tests
near the floor on both ranks, plus failure and recovery checks. CUDA numerical
equivalence, graph capture, image builds, real replies, and `verify.sh` also need
live validation. No GPU or service action is part of the offline suite.
