#!/usr/bin/env python3
"""Check the 2.8 bpw pack layout and its small manifests. Do not read weight data."""
import hashlib
from pathlib import Path
import sys

MANIFESTS = {
    "config.json": "8716a12d34f798e991bcc8e6689b25c006dd8a9675de5653e685094503c5d864",
    "model.safetensors.index.json": "1bb234f451a63d0640bd700ba11168c4b7697ea23afdd5d120da0c614fadf8ad",
}
WEIGHTS = {f"layer-{i:02d}.safetensors" for i in range(40)} | {"native.safetensors", "head-mtp.safetensors"}


def check(model: Path) -> None:
    actual = {p.name for p in model.glob("*.safetensors") if p.is_file()}
    if actual != WEIGHTS:
        raise ValueError("2.8 bpw pack needs the exact 42 weight files; wrong, missing or extra files")
    for name, digest in MANIFESTS.items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"2.8 bpw pack manifest differs: {name}")


if __name__ == "__main__":
    try:
        check(Path(sys.argv[1]))
    except (OSError, ValueError) as exc:
        print(f"FAIL {exc}: run scripts/download.sh verify view")
        sys.exit(1)
    print("OK 2.8 bpw pack filenames and manifests")
