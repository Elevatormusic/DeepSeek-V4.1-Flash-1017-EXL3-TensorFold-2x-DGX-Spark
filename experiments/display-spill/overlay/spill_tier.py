"""A disk tier for kept prompt states on CUDA engines (``tensorfold serve --spill-gib``): a state that leaves memory
is written to local disk and read back when a later request extends the same tokens, also after a restart.

A family hands over the snapshot object it keeps (``encode``: tensors into the file, plain values into JSON, the class
by import path, as ``engine.prefix_snapshots`` does on MLX) and, for a shared pool, named row tensors read back into
caller-placed views. Nothing here knows a model.

Writes: the DeepSeek hooks queue every kept prompt of ``min_tokens`` or more at once (write-through); ``highwater`` is
kept for the GLM tier's callers and the /health field, and this tier does not use it. A clean shutdown drains the queue
(``shutdown_drain``).
Copies off the device run on a writer's stream; the engine waits only before the rows move or go (``release``). Reads
run on a reader thread into pinned memory (``load_async``), so other streams keep decoding.

Files: safetensors, header padded to 4 KiB, O_DIRECT where it works, written to ``.partial``, fsynced, renamed. A file
whose size does not match its header is dropped, and every read checks a CRC-32 for each 8 MiB. One directory per
compatibility hash (format, build, runtime, weights, drafter, the engine's settings and cache layout) and per rank;
other hashes are pruned at start.

Several ranks: every decision is taken by all ranks in the same order from the index alone (never from how far this
rank's own writes got), so all hold the same index; a save agrees its size and the free-disk floor in one all-gather, a
read is used only when it worked on every rank, and at start only entries every rank holds are kept, in rank 0's
order.

Credits: wojo (Robert Wojciechowski, https://github.com/wojo) wrote this module and its hooks. The idea of NVMe session
state per rank was proposed in MiaAI-Lab's GLM-5.3-Flash vLLM kit pull request #232, authored by gabewillen (the idea
only, no code). The multi-rank design (one directory a rank, rank 0 deciding, an agreement on every read, a reconcile
at start) is that of patch 0250 of jayleaton/glm53-tensorfold-spark, authored by Jay Leaton (Apache-2.0). The file
conventions and flag names are those of TensorFold's MLX spill (pull request #68, authored by gilby) and of issue #155
(authored by raymondkpwong)."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import errno
import hashlib
import importlib
import json
import os
import queue
import re
import stat
import struct
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

FORMAT = "tensorfold-cuda-spill-4"          # 4: a CRC-32 of the whole header (3: token ids as base64 int64; 2: JSON lists)
SHUTDOWN_CAP_S = 55.0                  # most a shutdown drain waits: docker stop -t 60 (stop.sh) kills the process at 60 s
CHUNK = 64 << 20                       # bytes a staging buffer
ALIGN = 4096                           # O_DIRECT offsets, lengths and buffers
MAX_HEADER = 64 << 20                  # the most a header may declare (1M token ids twice, as base64, are 22 MiB)
BLOCK = 8 << 20                        # bytes one CRC-32 covers (CHUNK is a multiple of it)
STAGE = 8 << 20                        # bytes of the device buffer rows a memcpy may not reach pass through
WAIT_S = 60.0                          # a read of an entry still being written waits this long for it (reader thread)
FLOOR_DRAIN_S = 120.0                  # most a save waits for queued writes before it frees disk for the free-disk floor
WRITER_START_S = 60.0                  # most the constructor waits for the writer threads to report
INFLIGHT = 4 << 30                     # bytes of queued writes past which a new one first waits for the oldest

_DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64", torch.int8: "I8",
       torch.uint8: "U8", torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL"}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _DT[getattr(torch, _name)] = _code
_TD = {v: k for k, v in _DT.items()}

_PINNED: dict[str, list[torch.Tensor]] = {}     # staging allocated before the engine sizes its caches


def _up(x: int) -> int:
    return -(-x // ALIGN) * ALIGN


@dataclass
class SpillConfig:
    """``tensorfold serve`` flags of the tier (every rank must be started with the same values)."""

    root: str
    gib: float
    highwater: float = 0.70            # NOT USED by the DeepSeek hooks (write-through); kept for the GLM callers and /health
    min_tokens: int = 8192             # shorter prompts are not written
    min_free_gib: float = 50.0         # no write leaves less disk free (on any rank)
    keep_builds: int = 1
    writers: int = 2                   # writer threads (TF_SPILL_WRITERS), each with two staging buffers
    direct: bool = True                # O_DIRECT where the file system takes it (TF_SPILL_DIRECT=0: buffered)
    flush_s: float = 60.0              # a clean shutdown's budget (TF_SPILL_FLUSH_S)

    def __post_init__(self) -> None:
        if not 0.0 < self.highwater <= 1.0:
            raise ValueError(f"--spill-highwater: a fraction in (0, 1], not {self.highwater}")
        if not self.gib > 0:
            raise ValueError("--spill-gib must be above 0 to turn the tier on")
        if not self.min_free_gib >= 0:
            raise ValueError(f"TF_DS_SPILL_MIN_FREE_GIB: zero or more (0 keeps no reserve), not {self.min_free_gib}")
        if self.writers < 1:
            raise ValueError("TF_SPILL_WRITERS: at least 1")
        if not 0 <= self.keep_builds <= BUILDS_SHARED:
            raise ValueError(f"TF_DS_SPILL_KEEP_BUILDS: from 0 to {BUILDS_SHARED}, not {self.keep_builds}")

    @classmethod
    def from_args(cls, args: Any) -> SpillConfig | None:
        gib = float(getattr(args, "spill_gib", 0.0) or 0.0)
        if gib <= 0:
            return None
        root = str(getattr(args, "snapshot_dir", "") or "")
        if not root or root.lower() == "none":
            raise ValueError("--spill-gib needs --snapshot-dir")
        return cls(root=os.path.expanduser(root), gib=gib,
                   highwater=float(getattr(args, "spill_highwater", 0.70)),
                   min_tokens=int(getattr(args, "spill_min_tokens", 8192)),
                   min_free_gib=float(getattr(args, "spill_min_free_gib", 50.0)),
                   keep_builds=int(getattr(args, "spill_keep_builds", 1)),
                   writers=int(os.environ.get("TF_SPILL_WRITERS", "2")),
                   direct=os.environ.get("TF_SPILL_DIRECT", "1") != "0",
                   flush_s=float(os.environ.get("TF_SPILL_FLUSH_S", "60")))


def _pinned(n: int) -> list[torch.Tensor]:
    pin = torch.cuda.is_available()
    return [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=pin) for _ in range(n)]


BUILDS_SHARED = 64                     # the most builds a rank offers to the joint choice (the top of TF_DS_SPILL_KEEP_BUILDS)


def _build_name_ok(name: str) -> bool:
    return len(name) == len("spill-") + 16 and all(c in "0123456789abcdef" for c in name[len("spill-"):])


def _encode_builds(names: Sequence[str]) -> list[int]:
    """Build folder names (``spill-`` and 16 hex digits) as integers for a gather: a count, then 4 numbers a name."""

    coded = [n[len("spill-"):] for n in names if _build_name_ok(n)][:BUILDS_SHARED]
    row = [len(coded)]
    for hexes in coded:
        row += [int(hexes[i:i + 4], 16) for i in range(0, 16, 4)]
    return row + [0] * (1 + 4 * BUILDS_SHARED - len(row))


def _decode_builds(row: Sequence[int]) -> list[str]:
    return ["spill-" + "".join(f"{int(v):04x}" for v in row[1 + 4 * i:5 + 4 * i]) for i in range(int(row[0]))]


def _stage(device, stream) -> torch.Tensor:
    """A device buffer for ``_by_kernel``, allocated on ``stream`` (the one it is used on)."""

    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        return torch.empty(STAGE, dtype=torch.uint8, device=device)


def _by_kernel(dst: torch.Tensor, src: torch.Tensor, stage: torch.Tensor, stream) -> None:
    """``dst[:] = src`` (flat bytes) through ``stage`` a piece at a time, on ``stream``. For rows that a memcpy may not
    cross, only a kernel (patch 0072's display span: two host registrations, CUDA refuses a copy across them)."""

    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        for i in range(0, src.numel(), stage.numel()):
            k = min(stage.numel(), src.numel() - i)
            if src.device == stage.device:      # off the device: a kernel into the buffer, then a memcpy out
                torch.bitwise_or(src[i:i + k], 0, out=stage[:k])
                dst[i:i + k].copy_(stage[:k], non_blocking=stream is not None)
            else:                               # onto it: a memcpy into the buffer, then a kernel
                stage[:k].copy_(src[i:i + k], non_blocking=stream is not None)
                torch.bitwise_or(stage[:k], 0, out=dst[i:i + k])


def preallocate(cfg: SpillConfig | None) -> None:
    """Pin the staging buffers now, before the engine sizes its caches from the memory left (two a writer, two for
    reads), so the caches account for them."""

    if cfg is None or _PINNED:
        return
    _PINNED["write"] = _pinned(2 * cfg.writers)
    _PINNED["read"] = _pinned(2)


def release_pinned() -> None:
    """Drop the staging buffers of ``preallocate`` (a tier that every rank turned off must not keep them)."""

    _PINNED.clear()


def _driver_id() -> str:
    """The NVIDIA driver, from the first source that answers: the proc file, NVML, or the CUDA driver API (a coarser
    version, but present wherever CUDA runs). Raises ``RuntimeError`` when none answers."""

    import ctypes

    try:
        with open("/proc/driver/nvidia/version") as f:
            line = f.readline().strip()
        if line:
            return line
    except OSError:
        pass
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() == 0:
            try:
                buf = ctypes.create_string_buffer(96)
                if nvml.nvmlSystemGetDriverVersion(buf, 96) == 0 and buf.value:
                    return "nvml " + buf.value.decode()
            finally:
                nvml.nvmlShutdown()
    except (OSError, AttributeError):
        pass
    try:
        version = ctypes.c_int(0)
        if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value:
            return f"cuda-driver-api {version.value}"
    except (OSError, AttributeError):
        pass
    raise RuntimeError("the NVIDIA driver version cannot be read")


def build_id(model_dir, drafter_dir=None) -> str:
    """What stored states depend on beyond the cache layout, as MLX's prefix-snapshot key names it: the weights' (and
    drafter's) paths, the TensorFold version, the sources and kernels of the model's family and of tensorfold.cuda
    (read, not imported: the MLX kernel modules do not import on CUDA), and the runtime whose kernels make the bits
    (torch, CUDA, GPU, driver). Other values: another directory, nothing reused."""

    from pathlib import Path

    from tensorfold import __version__, families

    pkg = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    family = pkg.joinpath(*families.detect(model_dir).module.split(".")[1:])
    for base in (family, pkg / "cuda"):
        for path in sorted(p for p in base.rglob("*") if p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")):
            digest.update(path.relative_to(pkg).as_posix().encode())
            digest.update(path.read_bytes())
    run = f"torch={torch.__version__}|cuda={torch.version.cuda}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        run += f"|gpu={p.name}:{p.major}.{p.minor}"
    try:
        run += f"|driver={_driver_id()}"
    except RuntimeError:
        if torch.cuda.is_available():                 # a driverless key could reuse rows made under the old driver
            raise
    drafter = f"|drafter={Path(drafter_dir).resolve()}" if drafter_dir is not None else ""
    return f"{Path(model_dir).resolve()}{drafter}|tensorfold={__version__}|sources={digest.hexdigest()[:16]}|{run}"


def weights_id(*dirs) -> str:
    """This rank's copy of the weights (and the drafter's): each file's name, size, mtime and inode (rewritten in
    place: a new folder)."""

    from pathlib import Path

    digest = hashlib.sha256()
    for d in (d for d in dirs if d is not None):
        for path in sorted(Path(d).iterdir()):
            if path.suffix in (".safetensors", ".json"):
                st = path.stat()                              # through the hub's links, to the files themselves
                digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}".encode())
    return digest.hexdigest()[:16]


def ids_key(ids: Sequence[int]) -> tuple[int, int]:
    h = hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).digest()
    return int.from_bytes(h[0:4], "big", signed=True), int.from_bytes(h[4:8], "big", signed=True)


def _encode_ids(ids: Sequence[int]) -> str:
    """Token ids for the file header: int64 bytes in base64. A JSON list of 1M ids cost hundreds of ms on the thread
    that queues the write (it stops every decode stream on the rank)."""

    return base64.b64encode(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).decode("ascii")


def _decode_ids(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.int64).copy()


def key_name(key: tuple[int, int], n: int) -> str:
    return f"{key[0] & 0xffffffff:08x}{key[1] & 0xffffffff:08x}_{n}"


# -- objects to files and back -----------------------------------------------------------------------------------------
def _enc(value: Any, name: str, items: list) -> dict:
    if isinstance(value, torch.Tensor):
        items.append((name, value))
        return {"t": name}
    if isinstance(value, np.ndarray):
        items.append((name, torch.from_numpy(np.ascontiguousarray(value))))
        return {"t": name, "np": 1}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"v": value}
    if isinstance(value, (list, tuple)):
        out = {"l": [_enc(v, f"{name}.{i}", items) for i, v in enumerate(value)]}
        if isinstance(value, tuple):
            out["tuple"] = 1
        return out
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _enc(v, f"{name}.{k}", items) for k, v in value.items()}}
    raise TypeError(f"the spill tier cannot store {type(value).__name__} at {name}")


def _numpy_fields(node: Any) -> set[str]:
    """The tensor names of a layer spec that ``_dec`` turns into NumPy arrays (they are read into host memory)."""

    found: set[str] = set()
    if isinstance(node, dict):
        if node.get("np") and "t" in node:
            found.add(node["t"])
        for value in node.values():
            found |= _numpy_fields(value)
    elif isinstance(node, list):
        for value in node:
            found |= _numpy_fields(value)
    return found


def _dec(spec: dict, tensors: dict) -> Any:
    if "t" in spec:
        t = tensors[spec["t"]]
        return t.cpu().numpy() if spec.get("np") else t
    if "v" in spec:
        return spec["v"]
    if "l" in spec:
        out = [_dec(s, tensors) for s in spec["l"]]
        return tuple(out) if spec.get("tuple") else out
    if "d" in spec:
        return {k: _dec(s, tensors) for k, s in spec["d"].items()}
    raise ValueError(f"an unknown field spec {spec}")


def encode(obj: Any, *, skip: Sequence[str] = (),
           materialize: Callable[[Any], None] | None = None) -> tuple[list, dict]:
    """``obj``'s fields as (named tensors, a JSON-able layer): tensors by name, plain values, lists, tuples and string
    dicts of them; the class by import path. ``transient`` (class attribute) and ``skip`` name fields left out;
    ``materialize(obj)`` first (a family attaching rows that still live in its caches)."""

    if materialize is not None:
        materialize(obj)
    cls = type(obj)
    leave = set(getattr(cls, "transient", ())) | set(skip)
    items: list = []
    fields = {k: _enc(v, f"0.{k}", items) for k, v in vars(obj).items() if k not in leave and not k.startswith("_")}
    return items, {"class": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}


def decode(layer: dict, tensors: dict, classes: Sequence[str]) -> Any:
    """The object ``encode`` described (a dataclass's missing fields get their defaults); ``classes``: the only
    ``module:qualname`` paths it may build (nothing else is imported)."""

    if layer["class"] not in classes:
        raise ValueError(f"a stored state of class {layer['class']!r}, not one of {list(classes)}")
    mod, _, qual = layer["class"].partition(":")
    cls: Any = importlib.import_module(mod)
    for part in qual.split("."):
        cls = getattr(cls, part)
    obj = cls.__new__(cls)
    for k, spec in layer["fields"].items():
        setattr(obj, k, _dec(spec, tensors))
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if not hasattr(obj, f.name):
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
                    setattr(obj, f.name, f.default_factory())             # type: ignore[misc]
    return obj


def _dtype_code(t: torch.Tensor) -> str:
    try:
        return _DT[t.dtype]
    except KeyError:
        raise TypeError(f"the spill tier cannot store {t.dtype}") from None


def _crc_placeholder(nbytes: int) -> str:
    """The header's CRC field before the data is hashed: one 8-digit slot a BLOCK, so the header keeps its length."""

    return ",".join(["00000000"] * (-(-nbytes // BLOCK)))


class _Crcs:
    """CRC-32 of each BLOCK of a byte stream fed in order."""

    def __init__(self) -> None:
        self.done: list[int] = []
        self.cur = 0
        self.fill = 0

    def feed(self, mv) -> None:
        while len(mv):
            k = min(BLOCK - self.fill, len(mv))
            self.cur = zlib.crc32(mv[:k], self.cur)
            self.fill += k
            mv = mv[k:]
            if self.fill == BLOCK:
                self.done.append(self.cur)
                self.cur, self.fill = 0, 0

    def hex(self) -> str:
        return ",".join(f"{c:08x}" for c in self.done + ([self.cur] if self.fill else []))


def _header(items: list, meta: dict) -> tuple[bytes, int]:
    """The safetensors header for ``items`` padded so the data starts on a 4 KiB boundary; (bytes incl. its 8-byte
    length, data bytes)."""

    table: dict = {}
    at = 0
    for name, t in items:
        nbytes = t.numel() * t.element_size()
        table[name] = {"dtype": _dtype_code(t), "shape": list(t.shape), "data_offsets": [at, at + nbytes]}
        at += nbytes
    body = json.dumps({"__metadata__": {k: str(v) for k, v in meta.items()}, **table}, separators=(",", ":")).encode()
    total = _up(8 + len(body))
    body += b" " * (total - 8 - len(body))
    return struct.pack("<Q", len(body)) + body, at


@dataclass
class Entry:
    key: tuple[int, int]
    n: int
    name: str
    size: int = 0                      # exact bytes of the file, agreed by every rank (the largest)
    last: int = 0
    ids: Any = None                    # np.int64, keyed ids for a prompt with pictures (rank 0 matches against it)
    head: bool = False
    fields: dict = field(default_factory=dict)       # the stored object's plain fields: a family's test before a read
    bad: bool = False                  # this rank failed to write it (a read fails here; every rank then forgets it)


class _Stored(Exception):
    """The prompt is stored already: skip the save (raised inside the shared preparation step)."""


@dataclass
class Stats:
    written: int = 0
    write_failed: int = 0
    skipped: int = 0
    full: int = 0
    no_space: int = 0
    deferred: int = 0                  # early writes put off (the writers were busy); the engine counts them
    behind: int = 0                    # early writes queued
    unwritten_evicted: int = 0         # states that left memory without being written (too short, off grid, ...)
    unwritten: dict = field(default_factory=dict)     # unwritten_evicted by reason (/health: unwritten_<reason>)
    loaded: int = 0
    load_failed: int = 0
    crc_failed: int = 0                # reads whose data did not match its CRC-32 (the entry goes, a prefill)
    load_unfit: int = 0                # reads without the draft caches the request needs (the entry goes, a prefill)
    pruned: int = 0
    unlink_failed: int = 0             # evicted files that could not be removed (other than already gone)
    save_s: float = 0.0
    write_s: float = 0.0
    load_s: float = 0.0
    copy_wait_s: float = 0.0           # the engine waited for a copy off the device before moving rows
    copy_waits: int = 0
    behind_s: float = 0.0              # the engine choosing early writes (each iteration past the high-water mark)
    written_bytes: int = 0
    loaded_bytes: int = 0
    loaded_tokens: int = 0
    restore_ms: list = field(default_factory=list)


class Job:
    """One entry on its way to disk. ``copied``: every byte left the device (its rows may move); ``done``: on disk or
    given up."""

    def __init__(self, entry: Entry, path: str, header: bytes, items: list, src_event, seq: int = 0) -> None:
        self.entry = entry
        self.key = entry.key
        self.path = path
        self.seq = seq
        self.partial = f"{path}.{seq}.partial"      # its own: a later write of the same prompt never shares it
        self.nbytes = entry.size
        self.header = header
        self.items = items
        self.src_event = src_event
        self.copied = threading.Event()
        self.done = threading.Event()
        self.ok = False


class LoadJob:
    """A read on the reader thread: ``done`` once every byte is on the device (``event`` recorded on the reader's
    stream) or the read failed (``error``)."""

    def __init__(self, entry: Entry, path: str, header: dict, data_at: int, dests: dict) -> None:
        self.entry = entry
        self.path = path
        self.header = header
        self.data_at = data_at
        self.dests = dests             # name -> device tensor the bytes go to
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.event = None
        self.writing: Job | None = None           # the entry's write, still running when the read was asked for
        self.failed_counted = False               # this job is already in ``Stats.load_failed``
        self.t0 = time.perf_counter()
        self.meta = header.get("__metadata__", {}) or {}


_HCRC_RE = re.compile(rb'"hcrc":"([0-9a-f]{8})"')
_HCRC_ZERO = b'"hcrc":"00000000"'


def _check_header_crc(raw: bytes) -> None:
    """Check the CRC-32 that the header holds for itself (``hcrc``, counted with that field set to zeros). The CRCs of
    the data blocks do not cover the header, so a flipped bit in a count, a name or a flag would pass without this."""

    found = _HCRC_RE.findall(raw)
    if len(found) != 1 or zlib.crc32(_HCRC_RE.sub(_HCRC_ZERO, raw)) != int(found[0], 16):
        raise ValueError("the header does not match its CRC-32")


def _open_file(path: str, flags: int = os.O_RDONLY) -> int:
    """Open a stored file for reading, never blocking and never following a link, and check that it is a regular file.
    (A writable host cache could swap the file for a FIFO or a link after the start-up scan.)"""

    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.ESPIPE, "not a regular file", path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_read(path: str):
    """A binary file object on a regular file (see ``_open_file``)."""

    return os.fdopen(_open_file(path), "rb")


def _read_header_file(f) -> tuple[dict, int]:
    prefix = f.read(8)
    if len(prefix) != 8:
        raise ValueError("not a spill file")
    (size,) = struct.unpack("<Q", prefix)
    # The length is not trusted before the CRC is checked: a flipped bit must not make this read a huge block.
    # A real header is padded to ALIGN, holds at most a few tens of MiB (ids as base64), and fits in the file.
    if size > MAX_HEADER or _up(8 + size) != 8 + size or 8 + size > os.fstat(f.fileno()).st_size:
        raise ValueError("not a spill file")
    body = f.read(size)
    _check_header_crc(prefix + body)
    return json.loads(body), 8 + size


def _check_table(header: dict) -> int:
    """Check the tensor descriptors of a spill header and return the end of the data. Each tensor needs a known dtype, a
    shape of non-negative integers, and a span that holds exactly ``prod(shape) * element size`` bytes. The spans must
    follow each other from 0 without a gap or an overlap. A header that fails this is not trusted: the CRC covers only
    the span bytes, so a shape larger than its span would leave part of a new tensor uninitialized."""

    spans = []
    for name, d in header.items():
        if name == "__metadata__":
            continue
        dt = _TD.get(d.get("dtype")) if isinstance(d, dict) else None
        shape, offsets = d.get("shape"), d.get("data_offsets")
        if dt is None or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name}: bad tensor descriptor")
        if any(type(x) is not int or x < 0 for x in list(shape) + offsets) or offsets[1] < offsets[0]:
            raise ValueError(f"{name}: bad shape or offsets")
        count = 1
        for x in shape:
            count *= x
        if count * torch.empty((), dtype=dt).element_size() != offsets[1] - offsets[0]:
            raise ValueError(f"{name}: the shape does not fill the data span")
        spans.append((offsets[0], offsets[1]))
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise ValueError("tensor spans leave a gap or overlap")
        end = stop
    return end


def _parse_header(raw: bytes) -> tuple[dict, int]:
    (size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + size]), 8 + size


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % ALIGN == 0


def _plain(layers: list) -> dict:
    """A stored object's plain fields (numbers, strings, None) from its JSON layer."""

    return {k: s["v"] for k, s in layers[0]["fields"].items() if "v" in s} if layers else {}


class SpillStore:
    """The tier's files on this rank: an index every rank holds alike, writer threads, a reader thread."""

    def __init__(self, cfg: SpillConfig, *, rank: int = 0, world: int = 1, device: Any = "cuda", signature: str = "",
                 model_id: str = "", share: Callable[[list[int] | None], list[int]] | None = None,
                 gather: Callable[[list[int]], list[list[int]]] | None = None, quiet: bool = False,
                 weights: str = "", classes: Sequence[str] = (), spanned: Sequence[str] = ()) -> None:
        self.cfg = cfg
        self.classes = classes                  # the snapshot classes a read may build (decode)
        self.spanned = frozenset(f"rows.{n}" for n in spanned)   # rows (``save``'s names) a memcpy may not reach:
                                                                 # moved by a kernel (``_by_kernel``)
        self.rank, self.world = int(rank), int(world)
        self.device = device
        self.model_id = model_id
        self.share, self.gather = share, gather
        self.cap = int(cfg.gib * (1 << 30))
        shared = hashlib.sha256(f"{FORMAT}|{signature}".encode()).digest()
        own = hashlib.sha256(weights.encode()).digest()       # this rank's copy of the weights
        words = [int.from_bytes(h[i:i + 4], "big", signed=True) for h in (shared, own) for i in range(0, 16, 4)]
        if self.world > 1 and gather is not None:             # one directory name on every rank
            rows = gather(words)
            differ = [r for r, row in enumerate(rows) if list(row[:4]) != list(rows[0][:4])]
            if differ:
                print(f"[tensorfold] spill rank {self.rank}: rank(s) {differ} run another build or settings than "
                      "rank 0, so no rank can resume another's stored prompts until they match", flush=True)
            words = [w for row in rows for w in row[:8]]
        self.compat = hashlib.sha256(np.asarray(words, dtype=np.int32).tobytes()).hexdigest()[:16]
        self.root = os.path.abspath(cfg.root)                 # the spelling that is checked for a link is the one used
        self.top = os.path.join(self.root, f"spill-{self.compat}")
        self.dir = os.path.join(self.top, f"rank{self.rank}")
        self.uid = self.gid = -1
        probe = self.root
        while True:                                           # files belong to --snapshot-dir's owner (or, when it is
            try:                                              # not made yet, the owner of its nearest folder): a
                st = os.stat(probe)                           # server run as root in a container writes them as you
                self.uid, self.gid = st.st_uid, st.st_gid
                break
            except OSError:
                parent = os.path.dirname(probe)
                if parent == probe:
                    break
                probe = parent
        problem = ""
        try:
            self.lock = threading.RLock()
            self.index: dict[tuple, Entry] = {}
            self.pending: dict[tuple, Job] = {}      # the newest job of each prompt
            self.active: set[Job] = set()            # every job not yet done, also one whose entry was evicted
            self.clock = 0
            self.seq = 0
            self.stuck: dict[str, int] = {}          # evicted files that could not be removed: path -> size (tried again)
            self._evictions = [0, 0]                 # evicted entries, and those whose file stayed, since a save began
            self.generation = 0                     # bumped whenever the index changes
            self.stats = Stats()
            self.direct = bool(cfg.direct) and hasattr(os, "O_DIRECT")
            self.jobs: queue.Queue = queue.Queue()
            self.reads: queue.Queue = queue.Queue()
            self._threads: list[threading.Thread] = []
            self._reader: threading.Thread | None = None
            self._read_stage: torch.Tensor | None = None   # the reader's staging buffer, kept from the start
            self._dirty = False
            self._last_dir_sync = 0.0
            self._dir_lock = threading.Lock()
            missing, probe = [], self.root
            while not os.path.isdir(probe) and os.path.dirname(probe) != probe:
                missing.append(probe)                         # folders this start makes: they are the user's too
                probe = os.path.dirname(probe)
            if os.path.islink(self.root):                     # (name the real folder: a link is never followed)
                raise OSError(f"{cfg.root} is a symbolic link; use the folder it points to")
            os.makedirs(self.root, exist_ok=True)
            for d in reversed(missing):
                self._own(d)
            for d in (self.top, self.dir):                    # prompts' token ids: this user's only
                if os.path.islink(d):                         # (a link is never followed: the start would change
                    raise OSError(f"{d} is a symbolic link")  # and scan what it points to)
                os.makedirs(d, mode=0o700, exist_ok=True)
                os.chmod(d, 0o700)
                self._own(d)
            self._scan()
        except Exception as exc:                              # noqa: BLE001  (any failure is shared before the ranks go on)
            problem = repr(exc)
        flags = gather([int(bool(problem))]) if self.world > 1 and gather is not None else [[int(bool(problem))]]
        bad = [r for r, row in enumerate(flags) if row[0]]
        if bad:                                               # every rank stops alike: none waits on another
            raise RuntimeError(f"--snapshot-dir {cfg.root}: rank(s) {bad} cannot use it"
                               + (f": {problem}" if problem else ""))
        pruned_builds = self._prune_builds()                  # (the ranks choose the builds to keep together)
        found = len(self.index)
        dropped = self._reconcile() if self.world > 1 else 0
        before = len(self.index)
        self._make_room(0)                                    # a lower cap than the last run's: the oldest entries go
        dropped += before - len(self.index)
        self._start()                                         # (a failure here is agreed by the caller, like any other)
        self._quiet = quiet
        self._banner = (f"[tensorfold] spill: {self.top} ({cfg.gib:g} GiB a rank, write-through "
                        f"from {cfg.min_tokens} tokens, {cfg.writers} writers"
                        f"{', O_DIRECT' if self.direct else ''}, floor {cfg.min_free_gib:g} GiB free): "
                        f"{found - dropped} stored prompts from before this start, {self._total() / (1 << 30):.1f} GiB"
                        f"{f'; pruned {pruned_builds} older builds' if pruned_builds else ''}"
                        f"{f'; {self.stats.unlink_failed} files could not be removed' if self.stats.unlink_failed else ''}")

    def announce(self) -> None:
        """Rank 0 prints the start line. The caller does this after every rank agreed that the tier is on."""

        if self.rank == 0 and not self._quiet:
            print(self._banner, flush=True)

    # -- housekeeping --------------------------------------------------------------------------------------------------
    def _own(self, path: str) -> None:
        if self.uid >= 0:
            try:
                os.chown(path, self.uid, self.gid)
            except OSError:
                pass

    def _trim_kept_build(self, mine: str) -> None:
        """Remove the unfinished writes of a kept build and its oldest files until it fits the cap. A file that cannot
        be removed is counted and logged, and the next victim goes."""

        if os.path.islink(mine):                              # (never followed: its target is not this tier's)
            return
        try:
            names = os.listdir(mine)
        except (FileNotFoundError, NotADirectoryError):
            return
        except OSError as exc:
            self._unlink_failed(mine, exc, "list")
            return
        finished = []
        for name in names:
            path = os.path.join(mine, name)
            try:
                if name.endswith(".partial"):
                    os.remove(path)
                elif name.endswith(".safetensors"):
                    status = os.lstat(path)
                    if stat.S_ISREG(status.st_mode):
                        finished.append((status.st_mtime, path, status.st_size))
                    else:                                     # a link or a FIFO: removed itself, never followed
                        os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._unlink_failed(path, exc)
        total = sum(size for _, _, size in finished)
        for _, path, size in sorted(finished):
            if total <= self.cap:
                break
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._unlink_failed(path, exc)
                continue
            total -= size

    @staticmethod
    def _holds_entries(mine: str) -> bool | None:
        """Whether a rank folder of an older build has a finished file (None: it cannot be read)."""

        try:
            if os.path.islink(mine):
                return False
            with os.scandir(mine) as found:                   # (a link or a FIFO named like a file is none)
                return any(e.name.endswith(".safetensors") and e.is_file(follow_symlinks=False) for e in found)
        except (FileNotFoundError, NotADirectoryError):
            return False                                      # no rank folder: nothing to keep
        except OSError:
            return None

    def _is_dir(self, path: str) -> bool:
        """Whether ``path`` is a folder. A path that cannot be read is counted and logged (and is not a folder)."""

        try:
            return stat.S_ISDIR(os.lstat(path).st_mode)       # (a link is not a folder: it is never followed)
        except FileNotFoundError:
            return False
        except OSError as exc:
            self._unlink_failed(path, exc, "read")
            return False

    def _remove_tree(self, top: str) -> bool:
        """Remove a folder with everything in it. A path that cannot be removed is counted and logged, the rest goes,
        and the result says whether the folder is gone."""

        done = True

        def walk_failed(exc: OSError) -> None:             # (a folder that cannot be read holds paths that stay)
            nonlocal done
            if not isinstance(exc, FileNotFoundError):
                self._unlink_failed(str(exc.filename or top), exc, "list")
                done = False

        for base, dirs, files in os.walk(top, topdown=False, onerror=walk_failed):
            for name in files + [d for d in dirs if os.path.islink(os.path.join(base, d))]:
                try:
                    os.remove(os.path.join(base, name))
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self._unlink_failed(os.path.join(base, name), exc)
                    done = False
            try:
                os.rmdir(base)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if done:                                   # (a folder that still holds a stuck file is no new failure)
                    self._unlink_failed(base, exc)
                done = False
        return done and not os.path.lexists(top)

    def _unlink_failed(self, path: str, exc: OSError, what: str = "remove") -> None:
        with self.lock:
            self.stats.unlink_failed += 1
        print(f"[tensorfold] spill rank {self.rank}: could not {what} {path}: {exc!r}", flush=True)

    def _survey_builds(self) -> tuple[list[tuple[str, str]], bool]:
        """The other builds that have a finished file on this rank, newest first, as (name, path), and whether the
        survey is complete. A build without a finished file (for example from a start that failed here) takes no
        retention slot: it is removed."""

        complete = True
        try:
            names = [d for d in os.listdir(self.root)
                     if _build_name_ok(d) and os.path.join(self.root, d) != self.top]   # (only this tier's folders)
        except FileNotFoundError:
            return [], True
        except OSError as exc:                                # (older builds may stay: counted and logged)
            self._unlink_failed(self.root, exc, "list")
            return [], False
        dated = []
        for name in names:
            d = os.path.join(self.root, name)
            try:
                status = os.lstat(d)
            except FileNotFoundError:
                continue
            except OSError as exc:                            # (the build stays: counted and logged)
                self._unlink_failed(d, exc, "read")
                complete = False
                continue
            if stat.S_ISDIR(status.st_mode):
                dated.append((status.st_mtime, name, d))
        builds = []
        for _, name, d in sorted(dated, reverse=True):
            mine = os.path.join(d, f"rank{self.rank}")
            holds = self._holds_entries(mine)
            if holds is None:                                 # unreadable: the survey is not complete, the build stays
                complete = False
                builds.append((name, d))
            elif holds:
                builds.append((name, d))
            elif not os.path.islink(mine) and (not self._is_dir(mine) or self._remove_tree(mine)):
                try:
                    os.rmdir(d)
                except OSError:
                    pass
        return builds, complete

    def _retained_builds(self, names: list[str]) -> set[str]:
        """The builds that stay: the newest ``keep_builds`` that every rank has, in rank 0's order. One rank must not
        keep a build that its peer removes (a rollback then finds no entry that all ranks hold)."""

        keep = max(0, self.cfg.keep_builds)
        names = [n for n in names if _build_name_ok(n)]
        if self.world < 2 or self.gather is None:
            return set(names[:keep])
        offered: list[list[str]] = []
        offset = 0

        def merge(rows: list[list[int]]) -> tuple[list[str], bool]:
            if not offered:
                offered.extend([] for _ in rows)
            for rank, row in enumerate(rows):
                offered[rank] += _decode_builds(row)
            common = [n for n in offered[0] if all(n in set(other) for other in offered)]
            return common, any(int(row[0]) == BUILDS_SHARED for row in rows)

        while True:                                            # a window of each rank's newest builds at a time
            rows = self.gather(_encode_builds(names[offset:offset + BUILDS_SHARED]))
            common, more = self._agreed(lambda: merge(rows))
            if len(common) >= keep or not more:
                return set(common[:keep])
            offset += BUILDS_SHARED

    def _prune_builds(self) -> int:
        """Folders of other compatibility hashes (other builds, models, layouts): the newest ``keep_builds`` that all
        ranks have stay (trimmed to the cap), the others go."""

        shared = self.world > 1 and self.gather is not None
        builds, complete = self._agreed(self._survey_builds) if shared else self._survey_builds()
        if shared:
            complete = all(row[0] for row in self.gather([int(complete)]))
        if not complete:                                      # a rank that cannot see its builds: nobody prunes on a guess
            return 0
        retain = self._retained_builds([name for name, _ in builds])

        def prune() -> int:
            gone = 0
            for name, d in builds:
                mine = os.path.join(d, f"rank{self.rank}")
                if name in retain:                            # a kept build: its unfinished writes go, and its
                    self._trim_kept_build(mine)               # oldest files until it fits the current cap
                    continue
                if os.path.islink(mine):                      # the link goes, never what it points to
                    try:
                        os.remove(mine)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        self._unlink_failed(mine, exc)
                elif self._is_dir(mine) and self._remove_tree(mine):
                    gone += 1
                try:
                    os.rmdir(d)                                # empty once every rank pruned its own
                except OSError:
                    pass
            return gone

        return self._agreed(prune) if shared else prune()

    def _total(self) -> int:
        with self.lock:
            return sum(e.size for e in self.index.values())

    def _free_or_zero(self) -> int:
        try:
            return self.disk_free()
        except OSError:
            return 0

    def _short(self, size: int) -> int:
        """Bytes this rank's disk would be below the free-disk floor after a write of ``size`` (queued writes count as
        written)."""

        floor = int(self.cfg.min_free_gib * (1 << 30))
        with self.lock:
            queued = sum(j.nbytes for j in self.active if not j.done.is_set())
        return max(0, floor + size + queued - self.disk_free())

    def disk_free(self) -> int:
        """Free bytes on the store's disk. A failed probe raises ``OSError``: unknown space is not empty space."""

        st = os.statvfs(self.dir)
        return st.f_bavail * st.f_frsize

    # -- the index -----------------------------------------------------------------------------------------------------
    def _scan(self) -> None:
        found = []
        for name in os.listdir(self.dir):
            path = os.path.join(self.dir, name)
            if name.endswith(".partial"):                      # a write cut short: never read
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:                         # its bytes stay: they hold capacity
                    self._unlink_failed(path, exc)
                    try:
                        self.stuck[path] = os.lstat(path).st_size
                    except OSError:
                        pass
                continue
            if name.endswith(".safetensors"):
                try:
                    status = os.lstat(path)
                except FileNotFoundError:
                    continue
                except OSError as exc:                         # (its bytes stay on disk, out of the index)
                    self._unlink_failed(path, exc, "read")
                    continue
                if stat.S_ISREG(status.st_mode):
                    found.append((status.st_mtime, name))
                else:                                          # a link, a FIFO...: never opened (a FIFO would block)
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        self._unlink_failed(path, exc)
        for _, name in sorted(found):                          # oldest first: recency follows the files' age
            path = os.path.join(self.dir, name)
            try:
                with open(path, "rb") as f:
                    header, data_at = _read_header_file(f)
                    meta = header.get("__metadata__") or {}
                    if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
                        raise ValueError("another format")
                    want = data_at + _check_table(header)
                    if os.fstat(f.fileno()).st_size != want:
                        raise ValueError("size does not match the header")
                ids = _decode_ids(meta.get("keyed") or meta["tokens"])
                n = int(meta["n"])
                key = ids_key(ids)
                if len(ids) != n or key_name(key, n) + ".safetensors" != name:
                    raise ValueError("tokens do not match the name")
                e = Entry(key, n, name[:-12], want, ids=ids, head=meta.get("head") == "1",
                          fields=_plain(json.loads(meta.get("layers", "[]"))))
                self.clock += 1
                e.last = self.clock
                self.index[key] = e
            except MemoryError:
                raise                                          # (not the file's fault: the start agreement shares it)
            except Exception:                                  # noqa: BLE001  (unreadable, cut or another format: gone)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:                         # its bytes stay: they hold capacity
                    self._unlink_failed(path, exc)
                    try:
                        self.stuck[path] = os.lstat(path).st_size
                    except OSError:
                        pass

    def _agreed(self, step: Callable[[], Any]) -> Any:
        """Run a local step of the index agreement. A failure on one rank is shared first, so that no rank waits in a
        collective that another rank never enters: every rank raises."""

        failure, result = None, None
        try:
            result = step()
        except Exception as exc:                                  # noqa: BLE001
            failure = exc
        rows = self.gather([int(failure is not None)])
        if failure is not None:
            raise failure
        if any(row[0] for row in rows):
            raise RuntimeError("another rank could not prepare the index agreement")
        return result

    def _reconcile(self) -> int:
        """Every rank: keep the entries every rank holds, in rank 0's recency order, at the largest rank's size.
        Returns how many this rank dropped."""

        if self.share is None or self.gather is None:
            return 0

        def prepare() -> list[int] | None:
            mine = sorted(self.index.values(), key=lambda e: e.last)
            return [v for e in mine for v in (e.key[0], e.key[1], e.n)] if self.rank == 0 else None

        theirs = self.share(self._agreed(prepare))
        keys = self._agreed(lambda: [(theirs[i], theirs[i + 1], theirs[i + 2]) for i in range(0, len(theirs), 3)])
        keep = []
        if keys:
            every = self.gather(self._agreed(lambda: [int((hi, lo) in self.index and self.index[(hi, lo)].n == n)
                                                      for hi, lo, n in keys]))
            keep = self._agreed(lambda: [k for i, k in enumerate(keys) if all(r[i] for r in every)])
        sizes = self.gather(self._agreed(lambda: [self.index[(hi, lo)].size >> 20 for hi, lo, _ in keep])) if keep else []
        want = {(hi, lo) for hi, lo, _ in keep}
        dropped = 0
        for e in list(self.index.values()):
            if e.key not in want:
                self._delete(e)
                dropped += 1
        self.clock = 0
        for i, (hi, lo, _) in enumerate(keep):
            self.clock += 1
            e = self.index[(hi, lo)]
            e.last = self.clock
            e.size = max(e.size, (max(r[i] for r in sizes) + 1) << 20)
        return dropped

    def _delete(self, e: Entry) -> None:
        with self.lock:
            if self.index.get(e.key) is e:
                self.index.pop(e.key, None)
                self.generation += 1
            job = self.pending.get(e.key)
            if job is None or job.entry is not e:             # (one being written: its writer drops the file)
                try:
                    os.remove(os.path.join(self.dir, e.name + ".safetensors"))
                except FileNotFoundError:
                    pass
                except OSError as exc:                         # the bytes stay on disk: save() measures the disk again
                    self.stats.unlink_failed += 1
                    self.stuck[os.path.join(self.dir, e.name + ".safetensors")] = e.size
                    self._evictions[1] += 1
                    print(f"[tensorfold] spill rank {self.rank}: could not remove {e.name}: {exc!r}", flush=True)
            self._evictions[0] += 1
            self.stats.pruned += 1

    def touch(self, e: Entry) -> None:
        with self.lock:
            self.clock += 1
            e.last = self.clock

    def touch_key(self, key: tuple) -> None:
        """Every rank alike: entry ``key`` becomes the most recently used (the clock moves even when it is gone)."""

        with self.lock:
            self.clock += 1
            e = self.index.get(key)
            if e is not None:
                e.last = self.clock

    def forget(self, key: tuple) -> None:
        """Every rank: an entry that failed to read goes (its files are not trusted)."""

        with self.lock:
            e = self.index.get(key)
        if e is not None:
            self._delete(e)

    def has(self, ids: Sequence[int], key: tuple[int, int] | None = None) -> bool:
        """``ids`` is stored (``key``: its ``ids_key``, if known)."""

        key = ids_key(ids) if key is None else key
        with self.lock:
            e = self.index.get(key)
            return e is not None and e.n == len(ids)      # rank-local states (bad, pending) never decide

    def pending_jobs(self) -> int:
        with self.lock:
            return len(self.active)

    def find(self, prompt: Sequence[int], *, grid: int = 0, whole_needs_head: bool = True,
             usable: Callable[[Entry], bool] | None = None) -> Entry | None:
        """The longest stored prompt that ``prompt`` extends (or equals, when its head row was stored). A rank-local
        ``bad`` mark does not hide an entry: every rank must pick the same candidate, and then the failed read
        makes every rank forget it together."""

        arr = np.asarray(prompt, dtype=np.int64)
        best = None
        with self.lock:
            entries = list(self.index.values())
        for e in entries:
            if e.ids is None or e.n > len(arr) or (e.n == len(arr) and whole_needs_head and not e.head):
                continue
            if (best is not None and e.n <= best.n) or (grid and e.n % grid):
                continue
            if usable is not None and not usable(e):
                continue
            if np.array_equal(arr[:e.n], e.ids):
                best = e
        return best

    def _retry_stuck(self) -> int:
        """Try again to remove the evicted files that stayed. Returns the bytes of those that are still there."""

        with self.lock:
            for path in list(self.stuck):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                self.stuck.pop(path, None)
            return sum(self.stuck.values())

    def _make_room(self, extra: int, keep: frozenset = frozenset(), cap: int | None = None) -> bool:
        """Every rank alike: the least recently used entries go until ``extra`` more bytes fit the cap (or ``cap``, when
        the free-disk floor asks for less). The index alone
        decides (an entry still being written here goes too, its writer then drops the file), so every rank drops the
        same entries whatever its own writes' progress. ``keep``'s entries stay (an early write's: copies of prompts
        still in memory, which would otherwise evict each other in turn under a small cap)."""

        with self.lock:
            cap = self.cap if cap is None else min(cap, self.cap)
            total = self._total() + extra
            if total > cap and extra + sum(e.size for e in self.index.values() if e.key in keep) > cap:
                return False                       # it cannot fit beside ``keep``'s entries: nothing goes for it
            while total > cap:
                old = [e for e in self.index.values() if e.key not in keep]
                if not old:
                    break
                victim = min(old, key=lambda e: (e.last, e.key))
                total -= victim.size
                self._delete(victim)
            return total <= cap

    def _throttle(self, size: int) -> None:
        """A queued write holds its tensors (from live rows, the caller's caches too) until it is on disk: past
        ``INFLIGHT`` bytes queued, this rank first waits for its oldest write (only a wait: it decides nothing)."""

        while True:
            with self.lock:
                jobs = sorted((j for j in self.active if not j.done.is_set()), key=lambda j: j.seq)
            held = sum(j.nbytes for j in jobs)
            if not jobs or held + size <= INFLIGHT:
                return
            jobs[0].done.wait()

    # -- writing -------------------------------------------------------------------------------------------------------
    def _start(self) -> None:
        """Start the writer threads and wait until each reports that it is ready (its stream and checksum thread are
        made on the thread). A failure is an error here, so the caller's start-up agreement sees it."""

        if self._threads:
            return
        if self.spanned and self._read_stage is None:      # the reader's buffer too is taken now: a restore must not
            self._read_stage = _stage(self.device, None)   # meet an allocation that fails after the tier is announced
        bufs = _PINNED.get("write") or _pinned(2 * self.cfg.writers)
        if any(not _aligned(b) for b in bufs):
            self.direct = False
        ready: queue.Queue = queue.Queue()
        threads: list[threading.Thread] = []
        try:
            for i in range(self.cfg.writers):
                t = threading.Thread(target=self._writer, args=(bufs[2 * i:2 * i + 2], ready),
                                     name=f"spill-writer-{self.rank}-{i}", daemon=True)
                t.start()
                threads.append(t)
            for _ in threads:
                try:
                    outcome = ready.get(timeout=WRITER_START_S)
                except queue.Empty:
                    raise RuntimeError("a spill writer thread did not report in time") from None
                if outcome is not None:
                    raise RuntimeError(f"a spill writer thread could not start: {outcome!r}") from outcome
        except BaseException:
            for _ in threads:
                self.jobs.put(None)                        # the threads that did start stop
            raise
        self._threads = threads

    def close(self) -> None:
        """Stop the threads of a store that will not be used (a peer turned the tier off) and let go of its buffers."""

        threads, reader = self._threads, self._reader
        self._threads, self._reader = [], None
        for _ in threads:
            self.jobs.put(None)
        if reader is not None:
            self.reads.put(None)
        for thread in threads + ([reader] if reader is not None else []):
            thread.join(5.0)
        self._read_stage = None
        release_pinned()

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued write is on disk (a shutdown). True when the writes finished in time."""

        end = None if timeout is None else time.monotonic() + timeout
        while True:
            with self.lock:
                jobs = [j for j in self.active if not j.done.is_set()]
            if not jobs:
                break
            left = None if end is None else max(0.0, end - time.monotonic())
            if not jobs[0].done.wait(left) and left is not None and left <= 0:
                return False
        self._sync_dirs(force=True)
        return True

    def shutdown_budget(self) -> float:
        """Seconds a shutdown may wait for queued writes: ``flush_s`` (TF_SPILL_FLUSH_S), at most ``SHUTDOWN_CAP_S``."""

        return max(0.0, min(float(self.cfg.flush_s), SHUTDOWN_CAP_S))

    def shutdown_drain(self) -> bool:
        """``drain`` for a clean stop (SIGTERM), within ``shutdown_budget``; logs the result. True when it finished."""

        t0 = time.perf_counter()
        budget = self.shutdown_budget()
        ok = self.drain(budget)
        print(f"[tensorfold] spill rank {self.rank}: shutdown drain {'done' if ok else 'not finished'} in "
              f"{time.perf_counter() - t0:.1f}s (budget {budget:g}s)", flush=True)
        return ok

    def _sync_dirs(self, force: bool = False) -> None:
        with self._dir_lock:
            now = time.monotonic()
            if self._dirty and (force or now - self._last_dir_sync >= 1.0):
                _fsync_dir(self.dir)
                self._dirty = False
                self._last_dir_sync = now

    def _open_write(self, path: str) -> tuple[int, bool]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if self.direct:
            try:
                return os.open(path, flags | os.O_DIRECT, 0o600), True
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                self.direct = False
        return os.open(path, flags, 0o600), False

    @staticmethod
    def _write_all(fd: int, mv: memoryview) -> None:
        while len(mv):
            k = os.write(fd, mv)
            mv = mv[k:]

    @staticmethod
    def _pwrite_all(fd: int, mv: memoryview, at: int) -> None:
        while len(mv):
            k = os.pwrite(fd, mv, at)
            mv, at = mv[k:], at + k

    def _write(self, job: Job, bufs: list[torch.Tensor], stream, hasher: ThreadPoolExecutor,
               staging: torch.Tensor | None = None) -> bool:
        """One entry: header, then every tensor's bytes packed into the two staging buffers in turn (a buffer's copy
        off the device overlaps the other's write), full aligned buffers written, the tail padded then truncated."""

        path = job.partial
        fd, direct = self._open_write(path)
        total = 0
        try:
            if stream is not None and job.src_event is not None:
                stream.wait_event(job.src_event)        # the rows are final where the engine recorded
            cur, fill, ev = 0, 0, [None, None]
            hdr = np.frombuffer(job.header, dtype=np.uint8)
            bufs[cur][:len(hdr)].copy_(torch.from_numpy(hdr.copy()))
            fill = len(hdr)
            pending = None                              # (buffer index, bytes) copied, not yet written
            crcs, skip = _Crcs(), [len(hdr)]            # the data's CRCs (the header in the first buffer is not data)

            def flush(i: int, n: int) -> None:
                nonlocal total
                if ev[i] is not None:
                    ev[i].synchronize()
                    ev[i] = None
                k = _up(n) if direct else n
                mv = memoryview(bufs[i].numpy())
                hashed = hasher.submit(crcs.feed, mv[skip[0]:n])     # beside the write, which reads the same bytes
                skip[0] = 0
                self._write_all(fd, mv[:k])
                hashed.result()
                total += n

            stage = staging
            for name, t in job.items:
                v = t.reshape(-1).view(torch.uint8)
                if name in self.spanned and stage is None:
                    stage = _stage(v.device, stream)
                s, size = 0, v.numel()
                while s < size:
                    k = min(CHUNK - fill, size - s)
                    if name in self.spanned:
                        _by_kernel(bufs[cur][fill:fill + k], v[s:s + k], stage, stream)
                    elif stream is not None and v.is_cuda:
                        with torch.cuda.stream(stream):
                            bufs[cur][fill:fill + k].copy_(v[s:s + k], non_blocking=True)
                    else:
                        bufs[cur][fill:fill + k].copy_(v[s:s + k])
                    fill += k
                    s += k
                    if fill == CHUNK:
                        if stream is not None:
                            e = torch.cuda.Event()
                            e.record(stream)
                            ev[cur] = e
                        if pending is not None:
                            flush(*pending)
                        pending = (cur, CHUNK)
                        cur ^= 1
                        if ev[cur] is not None:
                            ev[cur].synchronize()
                            ev[cur] = None
                        fill = 0
            if stream is not None and fill:
                e = torch.cuda.Event()
                e.record(stream)
                ev[cur] = e
            for i in (0, 1):
                if ev[i] is not None:
                    ev[i].synchronize()
            job.copied.set()                            # every byte is in host memory: the rows may move now
            if pending is not None:
                flush(*pending)
            if fill:
                flush(cur, fill)
            slot = b'"crc32":"' + _crc_placeholder(len(crcs.done) * BLOCK + crcs.fill).encode() + b'"'
            final = job.header.replace(slot, b'"crc32":"' + crcs.hex().encode() + b'"')
            if job.header.count(slot) != 1 or len(final) != len(job.header):
                raise RuntimeError("the header's CRC field is missing or changed length")
            if final.count(_HCRC_ZERO) != 1:
                raise RuntimeError("the header's own CRC field is missing")
            final = final.replace(_HCRC_ZERO, b'"hcrc":"%08x"' % zlib.crc32(final))      # (counted with zeros)
            head = np.frombuffer(final, dtype=np.uint8)
            bufs[0][:len(head)].copy_(torch.from_numpy(head.copy()))
            self._pwrite_all(fd, memoryview(bufs[0].numpy())[:len(head)], 0)
            os.ftruncate(fd, len(job.header) + sum(t.numel() * t.element_size() for _, t in job.items))
            os.fsync(fd)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            os.close(fd)
            fd = -1
            with self.lock:                             # still this prompt's entry: its file (one dropped meanwhile,
                current = self.index.get(job.key) is job.entry       # by the cap or a failed read, leaves none)
                if current:
                    os.replace(path, job.path)
                    self.stuck.pop(job.path, None)          # (the file that stayed is replaced: no longer stuck)
            if not current:
                os.remove(path)
                return False
            self._own(job.path)
            with self._dir_lock:
                self._dirty = True
            self._sync_dirs()
            with self.lock:
                self.stats.written_bytes += total
            return True
        except Exception as exc:                         # noqa: BLE001  (only this prompt is not written)
            self._log(exc)
            self._settle(stream)                         # copies already queued still read the rows: wait for them
            job.copied.set()
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._unlink_failed(path, exc)
            return False

    @staticmethod
    def _settle(stream) -> None:
        """Wait until the copies queued on the writer stream are done (before the rows they read may be released)."""

        if stream is None:
            return
        try:
            stream.synchronize()
        except Exception:                                # noqa: BLE001  (a failed stream: nothing more can be done)
            pass

    def _writer(self, bufs: list[torch.Tensor], ready: queue.Queue | None = None) -> None:
        try:
            stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
            hasher = ThreadPoolExecutor(1, thread_name_prefix=f"spill-crc-w{self.rank}")
            hasher.submit(lambda: None).result()       # (the pool starts its thread on the first task: start it now)
            # The staging buffer of the rows a memcpy may not reach is allocated now, not by the first write: a rank
            # that cannot get it must say so before the tier is announced.
            staging = _stage(self.device, stream) if self.spanned else None
        except Exception as exc:                       # noqa: BLE001  (reported to the constructor)
            if ready is not None:
                ready.put(exc)
            return
        if ready is not None:
            ready.put(None)
        while True:
            job = self.jobs.get()
            if job is None:
                return
            t0 = time.perf_counter()
            try:
                ok = self._write(job, bufs, stream, hasher, staging)
            except Exception as exc:                   # noqa: BLE001  (the writer never dies)
                self._log(exc)
                self._settle(stream)
                ok = False
            job.ok = ok
            self._entry_done(job, ok)
            job.items = None                            # its tensors go now, not when the next write arrives
            job.copied.set()
            job.done.set()
            with self.lock:
                self.active.discard(job)
                self.stats.write_s += time.perf_counter() - t0
            job = None

    def _log(self, exc: BaseException) -> None:
        with self.lock:
            self.stats.write_failed += 1
        print(f"[tensorfold] spill rank {self.rank}: write failed: {exc!r}", flush=True)

    def save(self, ids: Sequence[int], obj: Any = None, rows: Sequence[tuple[str, torch.Tensor]] = (), *,
             skip: Sequence[str] = (), materialize: Callable[[Any], None] | None = None, head: bool = False,
             meta: dict | None = None,
             tokens: Sequence[int] | None = None, keep: frozenset = frozenset(),
             prepare_failure: BaseException | None = None) -> Job | None:
        """Every rank, in the same order: queue prompt ``ids``'s state (``obj``'s fields, see ``encode``) and the
        named ``rows`` (read back into caller views) for writing; one all-gather agrees the size and the free-disk
        floor. Returns the job, or None when skipped (stored or on its way, too short, no room, no disk)."""

        t0 = time.perf_counter()
        n = len(ids)
        if n < self.cfg.min_tokens:
            return None
        stuck = 0
        failure, key, stored = None, None, 0
        try:
            if prepare_failure is not None:            # the caller could not make obj or rows: shared below
                raise prepare_failure
            stuck = self._retry_stuck()                # evicted files that stayed still hold capacity
            key = ids_key(ids)                         # (this allocates too: it is inside the shared failure path)
            with self.lock:
                e = self.index.get(key)
                stored = int(e is not None and e.n == n)   # stored or on its way (every rank alike)
            if stored:
                raise _Stored
            items, layer = encode(obj, skip=skip, materialize=materialize) if obj is not None else ([], None)
            items = [(k, t if t.is_contiguous() else t.contiguous()) for k, t in items]
            for name, t in rows:
                if not t.is_contiguous():
                    raise ValueError(f"rows {name}: not contiguous")
                items.append((f"rows.{name}", t))
            info = {"format": FORMAT, "compat": self.compat, "model": self.model_id, "n": n,
                    "tokens": _encode_ids(ids if tokens is None else tokens),
                    **({} if tokens is None else {"keyed": _encode_ids(ids)}),
                    "layers": json.dumps([layer] if layer else []),
                    "rows": json.dumps([nm for nm, _ in rows]), "head": "1" if head else "0",
                    "saved": f"{time.time():.3f}",
                    "crc32": _crc_placeholder(sum(t.numel() * t.element_size() for _, t in items)), "hcrc": "00000000",
                    **(meta or {})}
            header, data = _header(items, info)
            size = len(header) + data
            short = self._short(size)                  # bytes this rank's disk would be below the floor
            name = key_name(key, n)                    # the rest of the setup fails here too, not after the agreement
            entry = Entry(key, n, name, size, ids=np.asarray(ids, dtype=np.int64), head=head,
                          fields=_plain([layer] if layer else []))
            src = None
            if torch.cuda.is_available():
                src = torch.cuda.Event()
                src.record(torch.cuda.current_stream())
            with self.lock:
                self.seq += 1
                seq = self.seq
            job = Job(entry, os.path.join(self.dir, name + ".safetensors"), header, items, src, seq)
        except _Stored:
            size = short = 0
        except Exception as exc:                     # noqa: BLE001  (a rank-local failure: share it, never skip the gather)
            if self.world < 2 or self.gather is None:
                raise
            failure, size, short = exc, 0, 0
        if self.world > 1 and self.gather is not None:
            got = self.gather([-(-short >> 20), (size >> 20) + 1, int(failure is not None), stored, -(-stuck >> 20)])
            if any(r[2] for r in got):             # one rank could not prepare it: every rank skips this save
                self._log(failure if failure is not None else RuntimeError("another rank could not prepare the save"))
                return None
            stored = max(r[3] for r in got)
            short = max(r[0] for r in got) << 20
            size = max(r[1] for r in got) << 20
            stuck = max(r[4] for r in got) << 20
        if stored:
            with self.lock:
                e = self.index.get(key) if key is not None else None
                if e is not None:
                    self.touch(e)
            self.stats.skipped += 1
            return None
        entry.size = job.nbytes = size
        if size > self.cap:                        # (larger than the cap: it can never fit, so nothing is evicted)
            self.stats.full += 1
            return None
        if short:
            # A queued write still holds its whole file: removing its entry frees no bytes until the writer ends. Wait
            # for the queue first, so the entries that go are files on disk, then measure again and agree on the
            # result. (The index decides who goes, not this wait.) A rank whose queue did not finish stops every rank.
            drained = self.drain(FLOOR_DRAIN_S)
            try:
                short = self._short(size)
            except OSError:                        # the space is unknown: skip this save on every rank
                short, drained = 0, False
            if self.world > 1 and self.gather is not None:
                got = self.gather([-(-short >> 20), int(drained)])
                short = max(r[0] for r in got) << 20
                drained = all(r[1] for r in got)
            if not drained:
                self.stats.no_space += 1
                return None
        # Every rank evicts the same entries (the index decides, and the stuck bytes are agreed), so every rank knows
        # alike whether this phase runs. Its failures are shared in one exchange before any rank goes on.
        self._evictions = [0, 0]
        if short or self._total() + size + stuck > self.cap:
            code, failed = 0, None
            try:
                if short and not self._make_room(size, keep, cap=self._total() + size - short):
                    code = 1                           # below the floor even with every entry outside ``keep`` gone
                elif not self._make_room(size + stuck, keep):
                    code = 2                           # it does not fit the cap
            except Exception as exc:                   # noqa: BLE001  (a rank-local failure: shared below)
                if self.world < 2 or self.gather is None:
                    raise
                code, failed = 3, exc
            report = [code, self._evictions[1]]        # (an eviction whose file stayed freed no room on that rank)
            reports = self.gather(report) if self.world > 1 and self.gather is not None else [report]
            if any(r[0] == 3 for r in reports):
                self._log(failed if failed is not None else RuntimeError("another rank could not make room"))
                # Some victims may have gone on one rank and not on another. Every rank is here: agree the index again,
                # so that no later lookup finds an entry on one rank only (that rank would wait alone in a restore).
                try:
                    self._reconcile()
                except Exception as exc:               # noqa: BLE001  (shared by the agreement: every rank sees it)
                    self._log(exc)
                return None
            code = max(r[0] for r in reports)
            if code == 1:
                self.stats.no_space += 1
                return None
            if code == 2:
                self.stats.full += 1
                return None
            if any(r[1] for r in reports):
                self.stats.no_space += 1
                return None
        if short:
            # Entries went to make room for the floor. Check that their bytes really left the disk (a file that could
            # not be removed does not), and agree: one rank without room stops the save on every rank.
            try:
                room = self._short(size) == 0
            except OSError:
                room = False
            if self.world > 1 and self.gather is not None:
                room = all(r[0] for r in self.gather([int(room)]))
            if not room:
                self.stats.no_space += 1
                return None
        self._throttle(size)
        with self.lock:                                    # the index changes here, on every rank alike
            old = self.index.get(key)
            if old is not None:
                self.index.pop(key, None)
            self.pending[key] = job
            self.active.add(job)
            self.clock += 1
            entry.last = self.clock                        # recency is the queue order (every rank alike)
            self.index[key] = entry
            self.generation += 1
        self.jobs.put(job)
        self.stats.save_s += time.perf_counter() - t0
        return job

    def release(self, job: Job) -> None:
        """The engine is about to move or free the rows ``job`` reads: wait until they are off the device."""

        if job.copied.is_set():
            return
        w0 = time.perf_counter()
        job.copied.wait()
        self.stats.copy_wait_s += time.perf_counter() - w0
        self.stats.copy_waits += 1

    def _entry_done(self, job: Job, ok: bool) -> None:
        with self.lock:
            if self.pending.get(job.key) is job:
                self.pending.pop(job.key, None)
            if ok:
                self.stats.written += 1
                if job.key not in self.index:              # dropped after its rename: the file goes too
                    try:
                        os.remove(job.path)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:                 # its bytes stay: they hold capacity until a later try
                        self._unlink_failed(job.path, exc)
                        self.stuck[job.path] = job.nbytes
            elif self.index.get(job.key) is job.entry:
                job.entry.bad = True                       # stays indexed (every rank alike); a read fails here

    # -- reading -------------------------------------------------------------------------------------------------------
    def load_async(self, key: tuple, dst_for: Callable[[str, list[int], torch.dtype], torch.Tensor | None]) -> LoadJob:
        """A failed start counts once in ``load_failed`` (see ``_load_async``)."""

        try:
            return self._load_async(key, dst_for)
        except Exception:
            with self.lock:
                self.stats.load_failed += 1
            raise

    def _load_async(self, key: tuple,
                    dst_for: Callable[[str, list[int], torch.dtype], torch.Tensor | None]) -> LoadJob:
        """Start reading entry ``key`` on the reader thread. ``dst_for(name, shape, dtype)`` places a tensor (a view the
        caller owns) or returns None for a new one, allocated here on the calling thread's stream. Raises at once when
        this rank has no readable entry; read errors land in the job. An entry still being written is read once its
        write ends, on the reader thread (this one does not wait)."""

        with self.lock:
            e = self.index.get(key)
            job = self.pending.get(key)
        if e is None or e.bad:
            raise FileNotFoundError(f"rank {self.rank} has no readable stored prompt {key}")
        path = os.path.join(self.dir, e.name + ".safetensors")
        writing = job if job is not None and job.entry is e and not job.done.is_set() else None
        if writing is not None:                        # its header is at hand; its CRCs come with the file (``_read``)
            header, data_at = _parse_header(writing.header)
        else:
            with _open_read(path) as f:
                header, data_at = _read_header_file(f)
        meta = header.get("__metadata__") or {}
        if meta.get("format") != FORMAT or meta.get("compat") != self.compat:
            raise ValueError(f"{path}: another format or build")
        _check_table(header)                                  # before any allocation: the header is not trusted yet
        dests = {}
        on_host = _numpy_fields(json.loads(meta.get("layers", "[]")))       # NumPy fields never need the device
        for name, d in header.items():
            if name == "__metadata__":
                continue
            dt, shape = _TD[d["dtype"]], d["shape"]
            dst = dst_for(name, shape, dt)
            if dst is None:
                dst = torch.empty(shape, dtype=dt, device="cpu" if name in on_host else self.device)
            elif list(dst.shape) != list(shape) or dst.dtype != dt:
                raise ValueError(f"{name}: file {shape} {dt}, destination {list(dst.shape)} {dst.dtype}")
            if not dst.is_contiguous():
                raise ValueError(f"{name}: the destination is not contiguous")
            dests[name] = dst
        lj = LoadJob(e, path, header, data_at, dests)
        lj.writing = writing
        if torch.cuda.is_available():
            lj.after = torch.cuda.Event()
            lj.after.record(torch.cuda.current_stream())      # destinations are free to write from here
        else:
            lj.after = None
        if self._reader is None:
            reader = threading.Thread(target=self._read_loop, name=f"spill-reader-{self.rank}", daemon=True)
            reader.start()
            self._reader = reader                          # only a thread that started is kept: the next read tries again
        self.reads.put(lj)
        return lj

    def _read_loop(self) -> None:
        try:
            pins = _PINNED.get("read") or _pinned(2)
            stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
            pool = ThreadPoolExecutor(4, thread_name_prefix=f"spill-crc-r{self.rank}")
            failed = None
        except Exception as exc:                           # noqa: BLE001  (every read then fails: the ranks agree on it)
            pins = stream = pool = None
            failed = exc
        while True:
            lj = self.reads.get()
            if lj is None:                                 # (close)
                return
            try:
                if failed is not None:
                    raise failed
                self._read(lj, pins, stream, pool)
            except BaseException as exc:                   # noqa: BLE001  (the reader never dies)
                lj.error = exc
            lj.done.set()
            lj = None                                      # (the caller holds what it read, not this thread)

    def _read(self, lj: LoadJob, pins: list[torch.Tensor], stream, pool: ThreadPoolExecutor) -> None:
        w, lj.writing = lj.writing, None
        if w is not None:                               # the entry's write first; the file's header has its CRCs
            if not w.done.wait(WAIT_S) or not w.ok:
                raise OSError(f"{lj.path}: its write failed or did not end in time")
            with _open_read(lj.path) as f:
                header, data_at = _read_header_file(f)
            table = {k: v for k, v in header.items() if k != "__metadata__"}
            if data_at != lj.data_at or table != {k: v for k, v in lj.header.items() if k != "__metadata__"}:
                raise ValueError(f"{lj.path}: not the file that was being written")
            lj.header, lj.meta = header, header.get("__metadata__") or {}
        direct = self.direct and all(_aligned(p) for p in pins) and lj.data_at % ALIGN == 0
        flags = os.O_RDONLY
        fd = -1
        if direct:
            try:
                fd = _open_file(lj.path, flags | os.O_DIRECT)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                direct = False
        if fd < 0:
            fd = _open_file(lj.path, flags)
        try:
            if stream is not None and lj.after is not None:
                stream.wait_event(lj.after)
            table = sorted(((k, v) for k, v in lj.header.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            end = max([v["data_offsets"][1] for _, v in table] or [0])
            if os.fstat(fd).st_size != lj.data_at + end:
                raise EOFError(f"{lj.path}: the file is shorter or longer than its header says")
            views = [(v["data_offsets"][0], v["data_offsets"][1], lj.dests[k].reshape(-1).view(torch.uint8),
                      k in self.spanned) for k, v in table]
            stage = self._read_stage if self._read_stage is not None else                 next((_stage(v.device, stream) for _, _, v, spanned in views if spanned), None)
            ev = [None, None]
            hashing: list = [[], []]                    # a pin's block hashes in flight (they read the pin)
            sums: list = []
            flip, s, ti = 0, 0, 0
            while s < end:
                k = min(CHUNK, end - s)
                pin = pins[flip]
                if ev[flip] is not None:
                    ev[flip].synchronize()
                for h in hashing[flip]:
                    h.result()
                want = _up(k) if direct else k
                mv = memoryview(pin.numpy())[:want]
                at = lj.data_at + s
                got = 0
                while got < k:                           # a read may return less than asked, long before the end
                    if hasattr(os, "preadv"):
                        step = os.preadv(fd, [mv[got:]], at + got)
                    else:
                        part = os.pread(fd, want - got, at + got)
                        mv[got:got + len(part)] = part
                        step = len(part)
                    if step <= 0:
                        raise EOFError(f"{lj.path}: short read")
                    got += step
                hashing[flip] = [pool.submit(zlib.crc32, mv[i:min(i + BLOCK, k)]) for i in range(0, k, BLOCK)]
                sums += hashing[flip]
                lo, hi = s, s + k
                while ti < len(views) and views[ti][1] <= lo:
                    ti += 1
                j = ti
                ctx = torch.cuda.stream(stream) if stream is not None else None
                if ctx is not None:
                    ctx.__enter__()
                try:
                    while j < len(views) and views[j][0] < hi:
                        a, b, v, spanned = views[j]
                        x0, x1 = max(a, lo), min(b, hi)
                        if x1 > x0 and spanned:
                            _by_kernel(v[x0 - a:x1 - a], pin[x0 - lo:x1 - lo], stage, stream)
                        elif x1 > x0:
                            v[x0 - a:x1 - a].copy_(pin[x0 - lo:x1 - lo], non_blocking=stream is not None)
                        j += 1
                finally:
                    if ctx is not None:
                        ctx.__exit__(None, None, None)
                if stream is not None:
                    e = torch.cuda.Event()
                    e.record(stream)
                    ev[flip] = e
                flip ^= 1
                s = hi
                self.stats.loaded_bytes += k
            if stream is not None:
                e = torch.cuda.Event()
                e.record(stream)
                lj.event = e
                e.synchronize()
            stored = [int(c, 16) for c in str(lj.meta.get("crc32", "")).split(",") if c]
            seen = [h.result() for h in sums]
            if seen != stored:
                bad = next((i for i, (a, b) in enumerate(zip(seen, stored)) if a != b), min(len(seen), len(stored)))
                self.stats.crc_failed += 1
                raise ValueError(f"{lj.path}: block {bad} does not match its CRC-32")
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        except BaseException:
            if stream is not None:                      # copies still queued write rows that may move or go now
                stream.synchronize()
            raise
        finally:
            os.close(fd)

    def finish_load(self, lj: LoadJob, *, wait: bool = True) -> tuple[Any, dict, dict]:
        """After ``lj`` is done on this rank: (the decoded object or None, its row tensors by name, metadata). The
        caller's stream waits for the copies. Raises the read's error. Counts the restore, or one ``load_failed`` when
        the read or the decode failed (a CRC mismatch also counts in ``crc_failed``)."""

        if wait:
            lj.done.wait()
        try:
            return self._finish_load(lj)
        except Exception:
            if not lj.failed_counted:
                lj.failed_counted = True
                with self.lock:
                    self.stats.load_failed += 1
            raise

    def _finish_load(self, lj: LoadJob) -> tuple[Any, dict, dict]:
        if lj.error is not None:
            raise lj.error
        if lj.event is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_event(lj.event)
        meta = lj.meta
        layers = json.loads(meta.get("layers", "[]"))
        obj = decode(layers[0], lj.dests, self.classes) if layers else None
        rows = {k[5:]: v for k, v in lj.dests.items() if k.startswith("rows.")}
        e = lj.entry
        self.touch(e)
        dt = time.perf_counter() - lj.t0
        self.stats.loaded += 1
        self.stats.loaded_tokens += e.n
        self.stats.load_s += dt
        self.stats.restore_ms.append(dt * 1000.0)
        if len(self.stats.restore_ms) > 4096:
            del self.stats.restore_ms[:2048]
        return obj, rows, meta

    def load(self, key: tuple, dst_for: Callable[[str, list[int], torch.dtype], torch.Tensor | None] = lambda *a: None
             ) -> tuple[Any, dict, dict]:
        """``load_async`` then ``finish_load``: the read on this thread's time (one stream at a time engines)."""

        return self.finish_load(self.load_async(key, dst_for))

    def info(self) -> dict:
        s = self.stats
        with self.lock:
            entries, pending = len(self.index), len(self.active)
        lat = sorted(s.restore_ms)
        p = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: 0.0)
        return {"gib": self.cfg.gib, "highwater": self.cfg.highwater, "writers": self.cfg.writers,
                "direct": self.direct, "entries": entries, "pending": pending,
                "stored_gib": round(self._total() / (1 << 30), 2),
                "stored_bytes": self._total(), "written": s.written, "skipped": s.skipped, "full": s.full,
                "no_space": s.no_space, "deferred": s.deferred, "behind": s.behind,
                "unwritten_evicted": s.unwritten_evicted, "write_failed": s.write_failed, "loaded": s.loaded,
                "load_failed": s.load_failed, "crc_failed": s.crc_failed, "load_unfit": s.load_unfit,
                "pruned": s.pruned, "unlink_failed": s.unlink_failed,
                "save_s": round(s.save_s, 2),
                "write_s": round(s.write_s, 2), "copy_wait_s": round(s.copy_wait_s, 2), "copy_waits": s.copy_waits,
                "behind_s": round(s.behind_s, 2),
                "written_gib": round(s.written_bytes / (1 << 30), 2), "written_bytes": s.written_bytes,
                "loaded_gib": round(s.loaded_bytes / (1 << 30), 2), "loaded_bytes": s.loaded_bytes,
                "load_s": round(s.load_s, 2), "loaded_tokens": s.loaded_tokens, "restore_ms_p50": p(0.5),
                "restore_ms_p95": p(0.95), "disk_free_bytes": self._free_or_zero(),
                **{f"unwritten_{k}": v for k, v in sorted(s.unwritten.items())}}
