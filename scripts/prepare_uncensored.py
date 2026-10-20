#!/usr/bin/env python3
"""Build a separate BF16 override view from local source shards. Use no GPU."""

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "uncensored_tensors.json"
OVERRIDE = "override-dealign-bf16.safetensors"
MAX_HEADER = 64 * 1024 * 1024
CHUNK = 8 * 1024 * 1024


def unique_object(pairs):
    """Reject repeated JSON keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"repeated JSON key: {key}")
        result[key] = value
    return result


def read_header(stream):
    """Read a bounded safetensors header."""
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("missing safetensors length")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= MAX_HEADER:
        raise ValueError("invalid safetensors header length")
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("short safetensors header")
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, 8 + length


def tensor_range(header, name, expected, start, size):
    """Check the dtype, shape, byte count and file bounds."""
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"missing tensor: {name}")
    if entry.get("dtype") != "BF16" or entry.get("shape") != expected["shape"]:
        raise ValueError(f"wrong dtype or shape: {name}")
    offsets = entry.get("data_offsets")
    if (not isinstance(offsets, list) or len(offsets) != 2
            or any(type(value) is not int for value in offsets)):
        raise ValueError(f"invalid offsets: {name}")
    first, last = offsets
    want = 2 * math.prod(expected["shape"])
    if first < 0 or last - first != want or start + last > size:
        raise ValueError(f"invalid tensor range: {name}")
    return start + first, want


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            digest.update(data)
    return digest.hexdigest()


def extract(source, output, manifest):
    """Copy only the pinned BF16 tensor bytes to one new file."""
    tensors = manifest["tensors"]
    header = {}
    offset = 0
    for name, item in sorted(tensors.items()):
        count = 2 * math.prod(item["shape"])
        header[name] = {"dtype": "BF16", "shape": item["shape"],
                        "data_offsets": [offset, offset + count]}
        offset += count
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with output.open("xb") as destination:
        destination.write(struct.pack("<Q", len(encoded)))
        destination.write(encoded)
        previous_file = None
        stream = None
        try:
            for name, item in sorted(tensors.items()):
                filename = item["file"]
                if Path(filename).name != filename:
                    raise ValueError("source file must have a plain name")
                if filename != previous_file:
                    if stream is not None:
                        stream.close()
                    stream = (source / filename).open("rb")
                    source_header, start = read_header(stream)
                    size = os.fstat(stream.fileno()).st_size
                    previous_file = filename
                first, count = tensor_range(source_header, name, item, start, size)
                stream.seek(first)
                digest = hashlib.sha256()
                while count:
                    data = stream.read(min(count, CHUNK))
                    if not data:
                        raise ValueError(f"short tensor payload: {name}")
                    destination.write(data)
                    digest.update(data)
                    count -= len(data)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"tensor SHA256 mismatch: {name}")
                print(f"checked {name}", flush=True)
        finally:
            if stream is not None:
                stream.close()


def check_base(base):
    """Require the base view layout. Run download.sh verify before this step."""
    model = base / "model"
    engram = base / "engram"
    shards = {path.name for path in model.glob("*.safetensors")}
    expected = {f"model-{index:05d}-of-00043.safetensors" for index in range(1, 44)}
    if shards != expected:
        raise ValueError("base view must contain only the 43 base weight shards")
    for path in (model / "config.json", model / "chat_template.jinja",
                 engram / "config.json", engram / "model.safetensors.index.json",
                 engram / "model-00047-of-00048.safetensors",
                 engram / "model-00048-of-00048.safetensors"):
        if not path.is_file():
            raise ValueError(f"missing base file: {path.name}")


def publish_view(temporary, view):
    """Publish the complete view atomically without replacing an existing path."""
    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "atomic view publication requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "renameat2 is required for atomic view publication")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(temporary), at_fdcwd, os.fsencode(view), rename_noreplace):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(view))


def build_view(source, base, view, manifest):
    """Build in a new folder. Publish the folder only after all checks pass."""
    source, base, view = source.resolve(), base.resolve(), view.resolve()
    if view.exists() or view.is_relative_to(base) or view.is_relative_to(source):
        raise ValueError("the output must be a new folder outside the base and source")
    check_base(base)
    view.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{view.name}-", dir=view.parent))
    try:
        omitted = []
        for folder in ("model", "engram"):
            (temporary / folder).mkdir()
            for path in sorted((base / folder).iterdir()):
                if folder == "model" and path.name == "model.safetensors.index.json":
                    omitted.append(f"{folder}/{path.name}")
                    continue
                if path.is_file() and path.suffix in (".json", ".jinja", ".safetensors"):
                    os.link(path.resolve(), temporary / folder / path.name)
        extract(source, temporary / "model" / OVERRIDE, manifest)
        digest = file_sha256(temporary / "model" / OVERRIDE)
        receipt = {"source": manifest["source"], "base": manifest["base"],
                   "base_verification": "not_verified_by_preparation", "omitted_base_files": omitted,
                   "override_sha256": digest, "tensors": manifest["tensors"]}
        (temporary / "override-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        (temporary / "override.sha256").write_text(
            f"{digest}  model/{OVERRIDE}\n", encoding="ascii")
        publish_view(temporary, view)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return digest


def check_view(view, manifest):
    """Check expected pins and override bytes, but do not verify the base bytes."""
    index = view / "model" / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        raise ValueError("model view must not contain model/model.safetensors.index.json")
    receipt = json.loads((view / "override-receipt.json").read_text(encoding="utf-8"))
    for key in ("source", "base", "tensors"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"receipt does not match the pinned {key}")
    digest = file_sha256(view / "model" / OVERRIDE)
    if receipt.get("override_sha256") != digest:
        raise ValueError("override SHA256 does not match the receipt")
    sums = (view / "override.sha256").read_text(encoding="ascii")
    if sums != f"{digest}  model/{OVERRIDE}\n":
        raise ValueError("override.sha256 does not match the receipt")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="local shards from the pinned source")
    parser.add_argument("--base-view", type=Path, help="verified view from download.sh")
    parser.add_argument("--output-view", type=Path, help="new view, on the base view disk")
    parser.add_argument("--check-view", type=Path, help="check an existing view and print its SHA256")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        if args.check_view:
            if any((args.source_dir, args.base_view, args.output_view)):
                parser.error("--check-view cannot be used with preparation options")
            print(check_view(args.check_view, manifest))
        else:
            if not all((args.source_dir, args.base_view, args.output_view)):
                parser.error("set --source-dir, --base-view and --output-view")
            digest = build_view(args.source_dir, args.base_view, args.output_view, manifest)
            print(f"ready: {args.output_view}\noverride SHA256: {digest}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"prepare_uncensored: {error}\n")


if __name__ == "__main__":
    main()
