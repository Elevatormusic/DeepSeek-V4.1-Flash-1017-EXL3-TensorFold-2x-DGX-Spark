#!/usr/bin/env python3
"""Create a separate experimental launch bundle. Never starts or stops anything."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMAGE = "tensorfold-dsv41:07aa02585f64-display-kv-stage1"
# Spill tuning values of .env that the spill bundle passes on to the rank containers, only when .env sets them.
FORWARDED_VARIABLES = ("TF_DS_SPILL_MIN", "TF_DS_SPILL_MIN_FREE_GIB", "TF_DS_SPILL_KEEP_BUILDS",
                       "TF_SPILL_WRITERS", "TF_SPILL_DIRECT", "TF_SPILL_FLUSH_S")
FORWARD_SNIPPET = ('$(for v in ' + " ".join(FORWARDED_VARIABLES)
                   + '; do [ -n "${!v:-}" ] && echo "-e $v=${!v}"; done)')


def prepare(destination: Path, image: str = IMAGE, spill: bool = False) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite {destination}")
    if not destination.parent.is_dir():
        raise ValueError("destination parent must already exist")
    start = (ROOT / "start.sh").read_text()
    anchor = '. "$(dirname "$0")/scripts/common.sh"\n'
    if start.count(anchor) != 1:
        raise ValueError("start.sh configuration anchor changed; review the experiment")
    # This guard is unconditional, even if PREFLIGHT=0 was inherited by mistake.
    guard = '''
head_running=$(docker inspect --format '{{.State.Running}}' tfv41-r0 2>/dev/null || true)
worker_running=$(wssh "docker inspect --format '{{.State.Running}}' tfv41-r1 2>/dev/null || true") || exit 1
if [ "$head_running" = true ] || [ "$worker_running" = true ]; then
  log "REFUSING experimental startup: stop both existing ranks after the benchmark finishes"
  exit 1
fi
'''
    start = start.replace(anchor, anchor + guard)
    # Non-forced removal cannot kill a rank started concurrently by another launcher.
    # start.sh has exactly two such commands (rank 0 and rank 1): another count means start.sh changed.
    if start.count("docker rm -f tfv41-") != 2:
        raise ValueError("start.sh docker rm anchors changed; review the experiment")
    start = start.replace("docker rm -f tfv41-", "docker rm tfv41-")
    anchor = "--device /dev/infiniband --cap-add IPC_LOCK"
    if start.count(anchor) != 1:
        raise ValueError("start.sh device anchor changed; review the experiment")
    start = start.replace(anchor, "--device /dev/infiniband --device /dev/dri --cap-add IPC_LOCK")
    anchor = "-e TF_DS_REPLAY=1"
    if start.count(anchor) != 1:
        raise ValueError("start.sh environment anchor changed; review the experiment")
    start = start.replace(anchor, "-e TF_DS_DISPLAY_KV_MIB=1792 -e TF_DS_KV=native " + anchor)
    config = (ROOT / ".env").read_text()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ValueError("invalid experimental image reference")
    # The display-only stage excludes the mapped-storage-safe spill port.
    # The dependent spill image opts in explicitly and keeps the user's SPILL_GIB.
    # Override only the private bundle, never the user's active .env.
    for anchor in ("TF_DS_PREFILL_CHUNK=2048", "TF_DS_PREFILL_CHUNK=${PREFILL_CHUNK:-2048}"):
        if start.count(anchor) == 1:
            start = start.replace(anchor, "TF_DS_PREFILL_CHUNK=512")
            break
    else:
        raise ValueError("start.sh prefill anchor changed; review the experiment")
    # MEM_CAP=1 needs the memory-topk image label, which these images do not have: the bundle starts with it off.
    overrides = [("IMAGE", image), ("PREFILL_CHUNK", "512"), ("MEM_CAP", "0")]
    if not spill:
        overrides.append(("SPILL_GIB", "0"))
    for key, value in overrides:
        config, count = re.subn(rf"(?m)^{key}=.*$", f"{key}={value}", config)
        if count == 0 and key in ("PREFILL_CHUNK", "SPILL_GIB", "MEM_CAP"):
            config = config.rstrip() + f"\n{key}={value}\n"
            count = 1
        if count != 1:
            raise ValueError(f"expected exactly one {key} in .env")
    if spill:
        # Forward the spill tuning values of .env to both ranks.
        anchor = "-e TF_DS_SPILL_GIB=${SPILL_GIB:-64} -e TF_DS_SPILL_DIR=/cache/tf-ds-spill"
        if start.count(anchor) != 1:
            raise ValueError("start.sh spill anchor changed; review the experiment")
        start = start.replace(anchor, anchor + "\n" + FORWARD_SNIPPET)
    # Validate the installer syntax without importing Torch or touching devices.
    ast.parse((HERE / "apply.py").read_text())
    if spill:
        ast.parse((ROOT / "experiments/display-spill/compose.py").read_text())
    destination.mkdir(mode=0o700)
    try:
        shutil.copytree(ROOT / "scripts", destination / "scripts")
        for name in ("stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
            shutil.copy2(ROOT / name, destination / name)
        (destination / "start.sh").write_text(start)
        (destination / "start.sh").chmod(0o755)
        (destination / ".env").write_text(config)
        (destination / ".env").chmod(0o600)
    except BaseException:
        # The directory was exclusively created above; never remove an existing bundle.
        shutil.rmtree(destination)
        raise
    print(f"Prepared {destination}; image {image}; no service actions performed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory with an existing parent")
    parser.add_argument("--image", default=IMAGE, help="experimental image to use in this snapshot")
    parser.add_argument("--spill", action="store_true", help="keep SPILL_GIB for the display-spill image")
    args = parser.parse_args()
    prepare(args.destination, args.image, args.spill)
