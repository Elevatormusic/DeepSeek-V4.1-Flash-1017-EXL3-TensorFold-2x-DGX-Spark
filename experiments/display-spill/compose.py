#!/usr/bin/env python3
"""Compose display KV and display-safe SSD spill from pristine pinned source."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compose(checkout: Path, display_path: Path, spill_path: Path) -> None:
    checkout = checkout.resolve()
    display = load(display_path, "display_kv_apply")
    spill = load(spill_path, "display_spill_apply")
    if display.REVISION != spill.REVISION:
        raise ValueError("feature revisions disagree")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != display.REVISION:
        raise ValueError(f"checkout HEAD must be {display.REVISION}")
    expected = {}
    for feature in (display, spill):
        for path, digest in feature.FILES.items():
            path = Path(path)
            if path in expected and expected[path] != digest:
                raise ValueError(f"source fingerprints disagree for {path}")
            expected[path] = digest
    additions = display.overlays() | spill.overlays()
    for path in additions:
        if (checkout / path).exists() or (checkout / path).is_symlink():
            raise ValueError(f"{path}: overlay already exists")
    source = {}
    for path, digest in expected.items():
        raw = (checkout / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision")
        source[path] = raw.decode()
    result = dict(source)
    for feature in (display, spill):
        for path in feature.FILES:
            path = Path(path)
            result[path] = feature.transform(path, result[path])
    for path, text in additions.items():
        if path in result and result[path] != text:
            raise ValueError(f"overlay collision: {path}")
        result[path] = text
    for path, text in result.items():
        ast.parse(text, filename=str(path))
    for path, text in result.items():
        (checkout / path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("display", type=Path)
    parser.add_argument("spill", type=Path)
    args = parser.parse_args()
    compose(args.checkout, args.display, args.spill)


if __name__ == "__main__":
    main()
