# Display KV with SSD spill

Use [the opt-in guide](../../docs/SPILL.md) for build, cutover and settings.
This experiment depends on `experiments/display-kv`. It is not the ordinary recipe.
Full two-node hardware validation is pending.

- `compose.py` validates the union of pristine source hashes once, applies the
  display transform, then the spill transform, and checks all generated Python
  before the first write.
- `apply.py` adds the spill store hooks to `multi.py`, the shutdown drain to
  `app.py` and the shutdown hook to `http.py`. It assumes the display transform
  has already run.
- `overlay/spill_tier.py` is the display-safe store. It writes safetensors with
  CRC-32, fsyncs and renames from `.partial`. It rejects corrupt or short files.
- `tests/` uses CPU torch and temporary directories. Real source composition
  tests run only when `TF_SPILL_SOURCE` names an explicit pinned checkout.

The top-level `start-display-spill.sh` builds and launches the dependent image.
It refuses active ranks, prepares DRM on both nodes, probes the display
reservation, and keeps `SPILL_GIB` from `.env`. Never stop GDM. Never use
`docker exec` in a serving rank.

No new performance claim is made. A real two-rank write, eviction, restore and
correct reply is a merge gate. See [the guide](../../docs/SPILL.md#validation-limits).
