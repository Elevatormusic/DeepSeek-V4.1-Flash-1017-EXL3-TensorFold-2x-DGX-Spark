"""CPU-only checks for the reused GLM spill tier (real CPU torch, no GPU)."""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest
import zlib
from dataclasses import dataclass

try:
    import numpy as np
    import torch
except ImportError as exc:      # CI has no torch: the CPU tests of the spill tier skip themselves
    raise unittest.SkipTest(f"needs numpy and torch: {exc}")

MODULE = os.path.join(os.path.dirname(__file__), "..", "overlay", "spill_tier.py")


def load():
    spec = importlib.util.spec_from_file_location("spill_tier_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["spill_tier_test"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Snap:
    ids: list
    ring: torch.Tensor
    raw: torch.Tensor | None = None


@dataclass
class KeptLike:
    eid: int
    base: int
    size: int
    top: int
    host: object
    snaps: dict
    replay: bool
    keys: object
    tick: int


class SpillTierTests(unittest.TestCase):
    def setUp(self):
        self.sp = load()

    def test_module_is_self_contained(self):
        tree = ast.parse(open(MODULE, encoding="utf-8").read())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        allowed = {"__future__", "base64", "contextlib", "ctypes", "dataclasses", "errno", "hashlib", "importlib", "json",
                   "os", "queue", "re", "shutil", "stat", "struct", "threading", "time", "zlib", "collections",
                   "concurrent", "typing", "pathlib", "numpy", "torch", "tensorfold"}
        self.assertEqual(imports - allowed, set())

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=0)
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=1, highwater=0)
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=1, writers=0)
        cfg = self.sp.SpillConfig(root="/tmp", gib=1)
        self.assertEqual(cfg.writers, 2)
        self.assertTrue(cfg.direct)

    def test_ids_key_and_name(self):
        key = self.sp.ids_key([1, 2, 3])
        self.assertEqual(len(key), 2)
        self.assertEqual(self.sp.key_name(key, 3),
                         f"{key[0] & 0xffffffff:08x}{key[1] & 0xffffffff:08x}_3")

    def test_encode_decode_roundtrip(self):
        snap = Snap(ids=[1, 2], ring=torch.arange(6, dtype=torch.uint8).reshape(2, 3))
        items, layer = self.sp.encode(snap, skip=("ids",))
        self.assertEqual(layer["class"], f"{Snap.__module__}:{Snap.__qualname__}")
        tensors = dict(items)
        obj = self.sp.decode(layer, tensors, [f"{Snap.__module__}:{Snap.__qualname__}"])
        self.assertTrue(torch.equal(obj.ring, snap.ring))
        self.assertIsNone(obj.raw)
        self.assertFalse(hasattr(obj, "ids"))

    def test_by_kernel_bounded_copy(self):
        src = torch.arange(1000, dtype=torch.uint8)
        dst = torch.zeros_like(src)
        stage = torch.empty(64, dtype=torch.uint8)
        self.sp._by_kernel(dst, src, stage, None)
        self.assertTrue(torch.equal(dst, src))

    def test_header_roundtrip(self):
        items = [("a", torch.zeros(4, dtype=torch.uint8)), ("b", torch.ones(2, dtype=torch.int32))]
        header, data = self.sp._header(items, {"format": "x"})
        parsed, data_at = self.sp._parse_header(header)
        self.assertEqual(data_at, len(header))
        self.assertEqual(parsed["__metadata__"]["format"], "x")
        self.assertEqual(parsed["a"]["data_offsets"], [0, 4])
        self.assertEqual(parsed["b"]["data_offsets"], [4, 12])
        self.assertEqual(data, 12)

    def test_crcs(self):
        crcs = self.sp._Crcs()
        crcs.feed(b"hello")
        self.assertEqual(crcs.hex(), f"{zlib.crc32(b'hello'):08x}")

    def test_info_has_enginemetrics_fields(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu")
            info = store.info()
            self.assertEqual(info["gib"], 1)
            for key in ("entries", "stored_bytes", "written", "written_bytes", "loaded",
                        "loaded_bytes", "loaded_tokens", "restore_ms_p50", "restore_ms_p95",
                        "write_failed", "crc_failed", "load_failed", "load_unfit", "disk_free_bytes"):
                self.assertIn(key, info)

    def test_kept_like_object_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{KeptLike.__module__}:{KeptLike.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            host = np.arange(100, dtype=np.int32)
            snaps = {"32": (torch.arange(8, dtype=torch.uint8), None),
                     "64": (torch.arange(8, dtype=torch.uint8), torch.arange(4, dtype=torch.uint8))}
            obj = KeptLike(1, 0, 100, 100, host, snaps, False, np.arange(100, dtype=np.int64), 1)
            job = store.save(ids, obj, skip=("disk", "hits"))
            self.assertIsNotNone(job)
            self.assertTrue(store.drain(10))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            loaded, rows, meta = store.finish_load(lj)
            self.assertEqual(loaded.eid, 1)
            self.assertTrue(np.array_equal(loaded.host, host))
            self.assertEqual(set(loaded.snaps), {"32", "64"})
            self.assertTrue(torch.equal(loaded.snaps["32"][0], snaps["32"][0]))
            self.assertIsNone(loaded.snaps["32"][1])
            self.assertTrue(torch.equal(loaded.snaps["64"][1], snaps["64"][1]))
            self.assertTrue(np.array_equal(loaded.keys, obj.keys))

    def test_corrupt_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            ring = torch.arange(100 * 4, dtype=torch.uint8).reshape(100, 4)
            snap = Snap(ids=ids, ring=ring)
            job = store.save(ids, snap, skip=("ids",))
            self.assertIsNotNone(job)
            self.assertTrue(store.drain(10))
            with open(job.path, "r+b") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
                f.seek(-1, os.SEEK_END)
                f.write(bytes([last[0] ^ 0xFF]))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(Exception):
                store.finish_load(lj)
            self.assertEqual(store.info()["crc_failed"], 1)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            ring = torch.arange(100 * 4, dtype=torch.uint8).reshape(100, 4)
            snap = Snap(ids=ids, ring=ring)
            job = store.save(ids, snap, skip=("ids",))
            self.assertIsNotNone(job)
            self.assertTrue(store.drain(10))
            self.assertTrue(store.has(ids))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            obj, rows, meta = store.finish_load(lj)
            self.assertTrue(torch.equal(obj.ring, ring))
            self.assertEqual(store.info()["loaded"], 1)
            self.assertEqual(store.info()["loaded_tokens"], 100)

    def stored(self, root, ids=None, **config):
        """A store with one written entry; returns (store, ids, classes)."""
        cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, **config)
        classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
        store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
        ids = list(range(100)) if ids is None else ids
        snap = Snap(ids=ids, ring=torch.arange(len(ids) * 4, dtype=torch.uint8).reshape(len(ids), 4))
        self.assertIsNotNone(store.save(ids, snap, skip=("ids",)))
        self.assertTrue(store.drain(10))
        return store, ids, classes

    def entry_path(self, store):
        return os.path.join(store.dir, [n for n in os.listdir(store.dir) if n.endswith(".safetensors")][0])

    def flip_in_header(self, path, old: bytes, new: bytes):
        with open(path, "r+b") as f:
            raw = f.read(1 << 16)
            self.assertEqual(raw.count(old), 1, old)
            f.seek(raw.index(old))
            f.write(new)

    def test_a_flipped_header_field_is_refused_by_the_scan_and_by_a_restore(self):
        for old, new in ((b'"head":"0"', b'"head":"1"'), (b'"rows":"[]"', b'"rows":"[x]"'),
                         (b'"n":"100"', b'"n":"101"')):
            with self.subTest(field=old), tempfile.TemporaryDirectory() as root:
                store, ids, classes = self.stored(root)
                path = self.entry_path(store)
                # A restart finds an intact file.
                cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
                self.assertTrue(again.has(ids))
                self.flip_in_header(path, old, new)
                with self.assertRaises(Exception):
                    again.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
                self.assertEqual(again.info()["load_failed"], 1)
                # A restart drops the file.
                third = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
                self.assertFalse(third.has(ids))
                self.assertFalse(os.path.exists(path))

    def test_a_corrupt_header_length_is_refused_before_the_header_is_read(self):
        import struct

        for declared in (512 << 20, (1 << 62), 5, None):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as root:
                store, ids, classes = self.stored(root)
                path = self.entry_path(store)
                with open(path, "r+b") as f:
                    real = struct.unpack("<Q", f.read(8))[0]
                    f.seek(0)
                    f.write(struct.pack("<Q", real + 4096 if declared is None else declared))
                with open(path, "rb") as f:
                    calls = []
                    original = f.read
                    with self.assertRaises(ValueError):
                        class Spy:
                            def __getattr__(self, name):
                                return getattr(f, name)

                            def read(self, n=-1):
                                calls.append(n)
                                return original(n)

                        self.sp._read_header_file(Spy())
                    self.assertTrue(all(n <= 8 for n in calls), calls)       # only the 8-byte length was read
                cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
                self.assertFalse(self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes).has(ids))

    def test_header_without_its_own_crc_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            self.flip_in_header(self.entry_path(store), b'"hcrc":', b'"hxrc":')
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            self.assertFalse(self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes).has(ids))

    def test_missing_file_counts_one_failed_restore(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            os.remove(self.entry_path(store))
            with self.assertRaises(OSError):
                store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            self.assertEqual(store.info()["crc_failed"], 0)
            self.assertEqual(store.info()["loaded"], 0)

    def test_unknown_key_counts_one_failed_restore(self):
        with tempfile.TemporaryDirectory() as root:
            store, _, _ = self.stored(root)
            with self.assertRaises(FileNotFoundError):
                store.load_async((1, 2), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)

    def test_truncated_file_counts_one_failed_restore_not_a_crc_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            path = self.entry_path(store)
            with open(path, "r+b") as f:
                f.truncate(os.path.getsize(path) - 100)
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(Exception):
                store.finish_load(lj)
            with self.assertRaises(Exception):          # the same job again: still one count
                store.finish_load(lj)
            info = store.info()
            self.assertEqual(info["load_failed"], 1)
            self.assertEqual(info["crc_failed"], 0)
            self.assertEqual(info["loaded"], 0)

    def test_crc_mismatch_counts_in_both_counters(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            with open(self.entry_path(store), "r+b") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
                f.seek(-1, os.SEEK_END)
                f.write(bytes([last[0] ^ 0xFF]))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(ValueError):
                store.finish_load(lj)
            self.assertEqual(store.info()["load_failed"], 1)
            self.assertEqual(store.info()["crc_failed"], 1)

    def test_a_rank_local_write_failure_does_not_change_the_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            before = store.find(ids, whole_needs_head=False)
            self.assertIsNotNone(before)
            before.bad = True                      # this rank failed to write it; the other rank did not
            after = store.find(ids, whole_needs_head=False)
            self.assertIs(after, before)           # both ranks must pick it, then the read fails here and both forget
            with self.assertRaises(Exception):
                store.load_async(before.key, lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            store.forget(before.key)
            self.assertIsNone(store.find(ids, whole_needs_head=False))

    def test_a_lower_cap_than_the_last_run_trims_the_stored_prompts_at_start(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            second = list(range(1000, 1100))
            snap = Snap(ids=second, ring=torch.arange(len(second) * 4, dtype=torch.uint8).reshape(len(second), 4))
            self.assertIsNotNone(store.save(second, snap, skip=("ids",)))
            self.assertTrue(store.drain(10))
            one = max(e.size for e in store.index.values())
            self.assertEqual(len(store.index), 2)

            def reopen(gib):
                cfg = self.sp.SpillConfig(root=root, gib=gib, min_free_gib=0, min_tokens=1)
                return self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)

            same = reopen(1)
            self.assertEqual(len(same.index), 2)                          # the same cap keeps both
            tight = reopen(1.5 * one / (1 << 30))                         # room for one: the older entry goes
            self.assertEqual(len(tight.index), 1)
            self.assertTrue(tight.has(second))
            self.assertFalse(tight.has(ids))
            self.assertLessEqual(tight._total(), tight.cap)
            self.assertEqual(len([n for n in os.listdir(tight.dir) if n.endswith(".safetensors")]), 1)
            none = reopen(1e-6)                                           # room for none
            self.assertEqual(len(none.index), 0)
            self.assertEqual([n for n in os.listdir(none.dir) if n.endswith(".safetensors")], [])

    def gathered(self, store, other):
        """Make the store believe it is rank 0 of two; ``other`` is the peer's row. Returns the list of rows it sent."""
        sent = []

        def gather(row):
            sent.append(list(row))
            return [list(row), list(other)]

        store.world, store.gather = 2, gather
        return sent

    def test_a_failure_while_one_rank_prepares_a_save_is_shared_before_the_gather(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            new = list(range(500, 600))
            snap = Snap(ids=new, ring=torch.arange(len(new) * 4, dtype=torch.uint8).reshape(len(new), 4))
            bad = torch.zeros(4, 4, dtype=torch.uint8).t()                # not contiguous
            self.assertIsNone(store.save(new, snap, skip=("ids",), rows=[("r", bad)]))
            self.assertEqual(len(sent), 1)                                # it still joined the gather
            self.assertEqual(sent[0][2], 1)                               # and said that it failed
            self.assertFalse(store.has(new))
            self.assertEqual(len(store.index), 1)

    def test_a_peer_that_failed_to_prepare_a_save_makes_every_rank_skip_it(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            sent = self.gathered(store, [0, 1, 1, 0, 0])
            new = list(range(500, 600))
            snap = Snap(ids=new, ring=torch.arange(len(new) * 4, dtype=torch.uint8).reshape(len(new), 4))
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(sent[0][2], 0)
            self.assertFalse(store.has(new))
            # With the peer well, the same save goes through.
            self.gathered(store, [0, 1, 0, 0, 0])
            self.assertIsNotNone(store.save(new, snap, skip=("ids",)))
            self.assertTrue(store.drain(10))
            self.assertTrue(store.has(new))

    def test_a_save_under_disk_pressure_waits_for_queued_writes_before_it_frees_entries(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            order = []
            store.drain = lambda timeout=None: order.append("drain") or True
            real = store._make_room

            def room(*args, **kwargs):
                order.append("room")
                return real(*args, **kwargs)

            store._make_room = room
            new = list(range(500, 600))
            snap = Snap(ids=new, ring=torch.arange(len(new) * 4, dtype=torch.uint8).reshape(len(new), 4))
            store.disk_free = lambda: 0                                   # below the floor: entries must go
            store.save(new, snap, skip=("ids",))
            self.assertEqual(order[:2], ["drain", "room"])
            order.clear()
            store.disk_free = lambda: 1 << 40                             # no pressure: no wait
            third = list(range(700, 800))
            snap = Snap(ids=third, ring=torch.arange(len(third) * 4, dtype=torch.uint8).reshape(len(third), 4))
            self.assertIsNotNone(store.save(third, snap, skip=("ids",)))
            self.assertNotIn("drain", order)

    def test_a_failed_write_waits_for_the_writer_stream_before_the_rows_are_released(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            order = []

            class Stream:
                def synchronize(self):
                    order.append(("sync", job.copied.is_set()))

            entry = next(iter(store.index.values()))
            # A header without the CRC slots: the write fails after it has started.
            job = self.sp.Job(entry, os.path.join(store.dir, "failed.safetensors"), b"{}", [], None, 99)
            bufs = [torch.zeros(1 << 20, dtype=torch.uint8), torch.zeros(1 << 20, dtype=torch.uint8)]
            hasher = self.sp.ThreadPoolExecutor(1)
            try:
                self.assertFalse(store._write(job, bufs, Stream(), hasher))
            finally:
                hasher.shutdown()
            self.assertEqual(order, [("sync", False)])                    # the stream was settled before "copied"
            self.assertTrue(job.copied.is_set())

    def test_a_state_larger_than_the_cap_evicts_nothing_under_disk_pressure(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            before = set(store.index)
            rooms = []
            real = store._make_room
            store._make_room = lambda *a, **k: rooms.append(a) or real(*a, **k)
            store.cap = 10                                                # smaller than any state
            store.disk_free = lambda: 0                                   # and the disk is below the floor
            new = list(range(500, 600))
            snap = Snap(ids=new, ring=torch.arange(len(new) * 4, dtype=torch.uint8).reshape(len(new), 4))
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(set(store.index), before)                    # nothing was evicted
            self.assertEqual(rooms, [])
            self.assertEqual(store.stats.full, 1)

    def new_state(self, first=500):
        ids = list(range(first, first + 100))
        return ids, Snap(ids=ids, ring=torch.arange(len(ids) * 4, dtype=torch.uint8).reshape(len(ids), 4))

    def test_a_save_that_cannot_drain_its_queue_is_skipped_by_every_rank(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            rooms = []
            real = store._make_room
            store._make_room = lambda *a, **k: rooms.append(a) or real(*a, **k)
            store.drain = lambda timeout=None: False                      # a write is still queued after the wait
            store.disk_free = lambda: 0
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            self.assertEqual(sent, [])
            new, snap = self.new_state()
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(rooms, [])                                   # nothing was evicted
            self.assertEqual(len(store.index), 1)
            self.assertEqual(store.stats.no_space, 1)
            self.assertEqual(sent[-1][1], 0)                              # this rank told the others it did not drain
            # The peer reports a drain that did not finish; this rank's own drain did.
            store.drain = lambda timeout=None: True
            store.world, store.gather = 2, lambda row: [list(row), [0, 0]] if len(row) == 2 else [list(row), [0, 1, 0, 0, 0]]
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(rooms, [])

    def test_the_floor_shortage_is_measured_again_after_the_drain(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            rooms = []
            real = store._make_room
            store._make_room = lambda *a, **k: rooms.append(k.get("cap")) or real(*a, **k)
            readings = iter([0, 1 << 40, 1 << 40, 1 << 40])               # short before the drain, none after it
            store.disk_free = lambda: next(readings, 1 << 40)
            store.drain = lambda timeout=None: True
            new, snap = self.new_state()
            self.assertIsNotNone(store.save(new, snap, skip=("ids",)))
            self.assertNotIn(True, [cap is not None for cap in rooms])    # no eviction for the floor
            self.assertEqual(len(store.index), 2)

    def test_the_writer_threads_start_with_the_store_and_a_start_failure_is_an_error_of_the_constructor(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            self.assertEqual(len(store._threads), store.cfg.writers)      # started before the first save
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.threading.Thread, "start", side_effect=RuntimeError("no thread")):
                with self.assertRaises(RuntimeError):
                    self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)

    def test_a_writer_thread_that_cannot_make_its_resources_fails_the_constructor(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp, "ThreadPoolExecutor", side_effect=RuntimeError("no checksum thread")):
                with self.assertRaises(RuntimeError) as caught:
                    self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertIn("could not start", str(caught.exception))

    def test_an_evicted_write_that_is_still_queued_stays_in_the_accounting(self):
        import threading

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[f"{Snap.__module__}:{Snap.__qualname__}"],
                                       quiet=True)
            gate = threading.Event()
            store._write = lambda job, bufs, stream, hasher, staging=None: (gate.wait(10), False)[1]   # a write that does not end
            ids, snap = self.new_state()
            first = store.save(ids, snap, skip=("ids",))
            store._delete(store.index[first.key])                                     # evicted while queued
            second = store.save(ids, snap, skip=("ids",))                             # the same prompt again
            self.assertIsNotNone(second)
            self.assertEqual(store.pending_jobs(), 2)                                 # both jobs are counted
            self.assertFalse(store.drain(0.2))                                        # and a drain waits for both
            gate.set()
            self.assertTrue(store.drain(10))
            self.assertEqual(store.pending_jobs(), 0)

    def test_a_reader_thread_that_did_not_start_is_tried_again_and_a_failed_reader_fails_reads(self):
        import threading
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            with mock.patch.object(self.sp.threading.Thread, "start", side_effect=RuntimeError("no thread")):
                with self.assertRaises(RuntimeError):
                    store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            self.assertIsNone(store._reader)                              # not kept: the next read starts it again
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            obj, rows, meta = store.finish_load(lj)
            self.assertIsNotNone(obj)
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            outcome = []

            def read():
                with mock.patch.object(self.sp, "ThreadPoolExecutor", side_effect=RuntimeError("no checksum thread")):
                    try:
                        store.finish_load(store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None))
                    except Exception as exc:                                # noqa: BLE001
                        outcome.append(exc)

            worker = threading.Thread(target=read, daemon=True)
            worker.start()
            worker.join(10)
            self.assertFalse(worker.is_alive())                           # the read ends with an error, never hangs
            self.assertEqual(len(outcome), 1)

    def test_a_failure_to_set_up_the_entry_is_shared_before_any_index_changes(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            new, snap = self.new_state()
            with mock.patch.object(self.sp, "Entry", side_effect=MemoryError("entry")):
                self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(sent[0][2], 1)                               # the failure rode in the one agreement
            self.assertEqual(len(sent), 1)
            self.assertEqual(len(store.index), 1)
            self.assertEqual(store.pending_jobs(), 0)

    def test_an_unreadable_free_space_is_not_an_empty_disk(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            rooms = []
            real = store._make_room
            store._make_room = lambda *a, **k: rooms.append(a) or real(*a, **k)

            def broken():
                raise OSError("stale mount")

            store.disk_free = broken
            self.assertEqual(store.info()["disk_free_bytes"], 0)          # the metrics still answer
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            new, snap = self.new_state()
            self.assertIsNone(store.save(new, snap, skip=("ids",)))       # skipped on every rank
            self.assertEqual(sent[0][2], 1)
            self.assertEqual(rooms, [])
            self.assertEqual(len(store.index), 1)                         # nothing was evicted
            # After the wait, a probe that fails makes every rank skip the save as well.
            store.drain = lambda timeout=None: True

            calls = []

            def once():
                calls.append(1)
                if len(calls) > 1:
                    raise OSError("stale mount")
                return 0

            store.disk_free = once
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(rooms, [])
            self.assertEqual(len(store.index), 1)

    def test_the_start_line_is_printed_by_announce_not_by_the_constructor(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[])
            self.assertNotIn("stored prompts from before this start", out.getvalue())
            with contextlib.redirect_stdout(out):
                store.announce()
            self.assertIn("stored prompts from before this start", out.getvalue())

    def test_the_checksum_thread_starts_before_a_writer_reports_ready(self):
        from unittest import mock

        class Pool:
            def __init__(self, *args, **kwargs):
                pass

            def submit(self, fn, *args):
                raise RuntimeError("can't start new thread")            # what the lazy pool does at its first task

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp, "ThreadPoolExecutor", Pool):
                with self.assertRaises(RuntimeError):
                    self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)

    def test_a_file_that_cannot_be_removed_stops_the_save_that_counted_on_its_bytes(self):
        from unittest import mock

        for removable in (True, False):
            with self.subTest(removable=removable), tempfile.TemporaryDirectory() as root:
                store, ids, classes = self.stored(root)
                path = self.entry_path(store)
                for entry in store.index.values():
                    entry.size = 4 << 20                                      # (the agreement counts whole MiB)
                store.gather, store.world = (lambda row: [list(row), list(row)]), 2
                # The disk is full while the file exists and has room once the file is gone.
                store.disk_free = lambda: 0 if os.path.exists(path) else 1 << 40
                real_remove = os.remove

                def remove(target, *args, **kwargs):
                    if target == path and not removable:
                        raise PermissionError("immutable")
                    return real_remove(target, *args, **kwargs)

                new, snap = self.new_state()
                with mock.patch.object(self.sp.os, "remove", remove):
                    job = store.save(new, snap, skip=("ids",))
                if removable:
                    self.assertIsNotNone(job)
                    self.assertEqual(store.stats.unlink_failed, 0)
                else:
                    self.assertIsNone(job)                                    # the bytes did not come back
                    self.assertEqual(store.stats.unlink_failed, 1)
                    self.assertEqual(store.stats.no_space, 1)
                    self.assertEqual(store.info()["unlink_failed"], 1)

    def test_the_prompt_key_is_made_inside_the_shared_failure_path(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            sent = self.gathered(store, [0, 1, 0, 0, 0])
            new, snap = self.new_state()
            with mock.patch.object(self.sp, "ids_key", side_effect=MemoryError("key")):
                self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(len(sent), 1)                                # it still joined the one agreement
            self.assertEqual(sent[0][2], 1)                               # and shared its failure
            self.assertEqual(len(store.index), 1)

    def test_a_stored_prompt_is_skipped_by_every_rank_through_the_same_agreement(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            sent = self.gathered(store, [0, 1, 0, 1, 0])
            snap = Snap(ids=ids, ring=torch.arange(len(ids) * 4, dtype=torch.uint8).reshape(len(ids), 4))
            skipped = store.stats.skipped
            self.assertIsNone(store.save(ids, snap, skip=("ids",)))
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][3], 1)
            self.assertEqual(store.stats.skipped, skipped + 1)
            # A peer that says "stored" alone also stops this rank, so no rank is left in a later collective.
            sent = self.gathered(store, [0, 1, 0, 1, 0])
            new, snap = self.new_state()
            self.assertIsNone(store.save(new, snap, skip=("ids",)))
            self.assertEqual(len(store.index), 1)

    def test_a_negative_free_space_floor_is_refused(self):
        for value in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                self.sp.SpillConfig(root="/tmp", gib=1, min_free_gib=value)
        self.sp.SpillConfig(root="/tmp", gib=1, min_free_gib=0)

    def test_unfinished_writes_of_a_kept_build_are_removed_at_start(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            partial, done = os.path.join(old, "x.1.partial"), os.path.join(old, "y.safetensors")
            for path in (partial, done):
                with open(path, "wb") as f:
                    f.write(b"x")
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertFalse(os.path.exists(partial))                     # the abandoned write is gone
            self.assertTrue(os.path.exists(done))                         # the kept build keeps its finished files

    def test_a_kept_build_is_trimmed_to_the_current_cap_at_start(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            paths = []
            for number in range(3):
                path = os.path.join(old, f"{number}.safetensors")
                with open(path, "wb") as f:
                    f.write(b"x" * (1 << 20))
                os.utime(path, (1000 + number, 1000 + number))
                paths.append(path)
            cfg = self.sp.SpillConfig(root=root, gib=2.5 / 1024, min_free_gib=0, min_tokens=1, keep_builds=1)
            self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertEqual([os.path.exists(path) for path in paths], [False, True, True])   # the oldest file went

    def test_a_kept_build_trim_goes_on_after_a_file_that_cannot_be_removed(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            paths = []
            for number in range(3):
                path = os.path.join(old, f"{number}.safetensors")
                with open(path, "wb") as f:
                    f.write(b"x" * (1 << 20))
                os.utime(path, (1000 + number, 1000 + number))
                paths.append(path)
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == paths[0]:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1.5 / 1024, min_free_gib=0, min_tokens=1, keep_builds=1)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertEqual([os.path.exists(path) for path in paths], [True, False, False])   # the next victims went
            self.assertEqual(store.stats.unlink_failed, 1)

    def test_an_old_build_that_cannot_be_removed_is_not_counted_as_pruned(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            stuck, other = os.path.join(old, "0.safetensors"), os.path.join(old, "1.safetensors")
            for path in (stuck, other):
                with open(path, "wb") as f:
                    f.write(b"x")
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == stuck:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            out = io.StringIO()
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(out):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[])
                store.announce()
            self.assertTrue(os.path.exists(stuck))
            self.assertFalse(os.path.exists(other))                       # the rest of the build went
            self.assertEqual(store.stats.unlink_failed, 1)
            self.assertNotIn("pruned 1 older builds", out.getvalue())
            self.assertIn("1 files could not be removed", out.getvalue())

    def test_a_read_that_returns_less_than_asked_is_continued(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            ring = torch.arange(100 * 4, dtype=torch.uint8).reshape(100, 4)
            self.assertIsNotNone(store.save(ids, Snap(ids=ids, ring=ring), skip=("ids",)))
            self.assertTrue(store.drain(10))
            real = os.preadv

            def preadv(fd, buffers, offset):
                return real(fd, [buffers[0][:7]], offset)                # at most 7 bytes for each call

            with mock.patch.object(self.sp.os, "preadv", preadv):
                lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
                obj, rows, meta = store.finish_load(lj)
            self.assertTrue(torch.equal(obj.ring, ring))
            self.assertEqual(store.info()["load_failed"], 0)

    def test_a_setup_failure_that_is_not_an_os_error_is_shared_with_the_other_rank(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            calls = []

            def gather(values):
                calls.append(list(values))
                return [list(values), list(values)]                  # (the peer reports what this rank reports)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.SpillStore, "_scan", side_effect=MemoryError("index")):
                with self.assertRaises(RuntimeError) as caught:
                    self.sp.SpillStore(cfg, rank=0, world=2, device="cpu", gather=gather, classes=[])
            self.assertEqual(calls[-1], [1])                          # the failure is in the agreement
            self.assertIn("MemoryError", str(caught.exception))

    def test_a_kept_build_that_cannot_be_listed_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            with open(os.path.join(old, "0.safetensors"), "wb") as f:
                f.write(b"x")
            real_listdir = os.listdir

            def listdir(path="."):
                if path == old:
                    raise PermissionError("denied")
                return real_listdir(path)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            with mock.patch.object(self.sp.os, "listdir", listdir), contextlib.redirect_stdout(io.StringIO()):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertEqual(store.stats.unlink_failed, 1)

    def test_an_old_build_that_cannot_be_walked_is_not_counted_as_pruned(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000", "rank0")
            os.makedirs(old)
            with open(os.path.join(old, "0.safetensors"), "wb") as f:
                f.write(b"x")
            real_scandir = os.scandir

            seen = []

            def scandir(path="."):
                if path == old:
                    seen.append(path)
                    if len(seen) > 1:                                     # the survey can read it, the walk cannot
                        raise PermissionError("denied")
                return real_scandir(path)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            out = io.StringIO()
            with mock.patch.object(self.sp.os, "scandir", scandir), contextlib.redirect_stdout(out):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[])
                store.announce()
            self.assertTrue(os.path.exists(os.path.join(old, "0.safetensors")))
            self.assertEqual(store.stats.unlink_failed, 1)
            self.assertNotIn("pruned 1 older builds", out.getvalue())

    def test_a_current_build_file_that_cannot_be_removed_at_start_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            folder = store.dir
            partial, broken = os.path.join(folder, "x.partial"), os.path.join(folder, "y.safetensors")
            for path in (partial, broken):
                with open(path, "wb") as f:
                    f.write(b"junk")
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target in (partial, broken):
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.stats.unlink_failed, 2)

    def test_a_root_that_cannot_be_listed_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            real_listdir = os.listdir

            def listdir(path="."):
                if path == root:
                    raise PermissionError("execute only")
                return real_listdir(path)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "listdir", listdir), contextlib.redirect_stdout(io.StringIO()):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertEqual(store.stats.unlink_failed, 1)

    def test_a_file_that_cannot_be_read_at_start_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            real_stat = os.lstat

            def stat(target, *args, **kwargs):
                if target == path:
                    raise OSError(5, "I/O error")
                return real_stat(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "lstat", stat), contextlib.redirect_stdout(io.StringIO()):
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.stats.unlink_failed, 1)

    def test_a_dropped_entry_whose_file_cannot_be_removed_after_its_write_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            job = self.sp.Job.__new__(self.sp.Job)
            job.key, job.path, job.entry, job.nbytes = (1, 2), path, None, 4496
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == path:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                store._entry_done(job, True)                          # the key is not in the index: the file goes
            self.assertEqual(store.stats.unlink_failed, 1)
            self.assertIn(path, store.stuck)                              # and its bytes hold capacity

    def test_a_missing_spill_root_is_made_for_the_owner_of_its_nearest_folder(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as base:
            root = os.path.join(base, "a", "b")
            owned = []
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "chown", lambda path, uid, gid: owned.append(path)):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            for path in (os.path.join(base, "a"), root):
                self.assertIn(path, owned)                                # the folders this start made

    def test_the_driver_comes_from_a_second_source_when_the_proc_file_is_missing(self):
        import ctypes
        from unittest import mock

        class Driver:
            def cuDriverGetVersion(self, ref):                           # noqa: N802
                ref._obj.value = 13000
                return 0

        def library(name):
            if name == "libcuda.so.1":
                return Driver()
            raise OSError(name)

        real_open = open

        def no_proc(path, *args, **kwargs):
            if str(path).startswith("/proc/driver"):
                raise FileNotFoundError(path)
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", no_proc), mock.patch.object(ctypes, "CDLL", library):
            self.assertEqual(self.sp._driver_id(), "cuda-driver-api 13000")
        with mock.patch("builtins.open", no_proc), mock.patch.object(ctypes, "CDLL", side_effect=OSError("none")):
            with self.assertRaises(RuntimeError):                        # no source: not an empty key
                self.sp._driver_id()

    def test_an_old_build_folder_that_cannot_be_read_is_counted(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-0000000000000000")
            os.makedirs(os.path.join(old, "rank0"))
            real_stat = os.lstat

            def stat(target, *args, **kwargs):
                if target == old:
                    raise PermissionError("denied")
                return real_stat(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            with mock.patch.object(self.sp.os, "lstat", stat), contextlib.redirect_stdout(io.StringIO()):
                store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertEqual(store.stats.unlink_failed, 1)

    def test_a_memory_failure_while_scanning_keeps_the_file_and_fails_the_start(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp, "_decode_ids", side_effect=MemoryError("header")):
                with self.assertRaises(RuntimeError) as caught:
                    self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertIn("MemoryError", str(caught.exception))
            self.assertTrue(os.path.exists(path))                         # a healthy file is not deleted

    def test_a_linked_rank_folder_of_an_old_build_is_never_followed(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            victim = os.path.join(outside, "keep.safetensors")
            with open(victim, "wb") as f:
                f.write(b"x")
            for name in ("spill-0000000000000001", "spill-0000000000000002"):
                os.makedirs(os.path.join(root, name))
            try:
                os.symlink(outside, os.path.join(root, "spill-0000000000000001", "rank0"))      # an old build to remove
                os.symlink(outside, os.path.join(root, "spill-0000000000000002", "rank0"))      # an old build to keep
            except (OSError, NotImplementedError):
                self.skipTest("no symbolic links here")
            cfg = self.sp.SpillConfig(root=root, gib=1e-9, min_free_gib=0, min_tokens=1, keep_builds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(victim))                       # nothing behind a link was touched

    def test_an_empty_build_of_a_failed_start_takes_no_retention_slot(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            working = os.path.join(root, "spill-0000000000000001", "rank0")
            orphan = os.path.join(root, "spill-0000000000000002", "rank0")
            for folder in (working, orphan):
                os.makedirs(folder)
            kept = os.path.join(working, "0.safetensors")
            with open(kept, "wb") as f:
                f.write(b"x")
            os.utime(os.path.dirname(working), (1000, 1000))              # older than the orphan
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(kept))                         # the build with entries stays
            self.assertFalse(os.path.exists(orphan))                      # the empty one is gone

    def test_a_staging_buffer_that_cannot_be_allocated_fails_the_constructor(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            for failing in (0, 1):                                        # the reader's buffer, then the writer's
                with self.subTest(failing=failing):
                    answers = [torch.empty(8, dtype=torch.uint8), torch.empty(8, dtype=torch.uint8)]
                    answers[failing] = MemoryError("stage")
                    with mock.patch.object(self.sp, "_stage", side_effect=answers):
                        with self.assertRaises((RuntimeError, MemoryError)):
                            self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True,
                                               spanned=["comp.0.0"])

    def test_a_linked_current_namespace_is_refused_and_not_followed(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            first = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            top, rank_dir = first.top, first.dir
            for link, target in ((rank_dir, outside), (top, outside)):
                with self.subTest(link=os.path.basename(link)):
                    victim = os.path.join(outside, "x.partial")
                    with open(victim, "wb") as f:
                        f.write(b"x")
                    if link == top:
                        import shutil

                        shutil.rmtree(top)
                    else:
                        os.rmdir(rank_dir)
                    try:
                        os.symlink(target, link)
                    except (OSError, NotImplementedError):
                        self.skipTest("no symbolic links here")
                    with self.assertRaises(RuntimeError):
                        self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
                    self.assertTrue(os.path.exists(victim))               # nothing behind the link was touched
                    os.remove(link)
                    os.remove(victim)
                    os.makedirs(rank_dir)

    def test_a_failure_while_one_rank_prepares_the_index_agreement_is_shared(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            store.gather, store.share = gather, lambda values: values or []
            store.world = 2

            class Boom(dict):
                def values(self):
                    raise MemoryError("sort")

            store.index = Boom()
            with self.assertRaises(MemoryError):
                store._reconcile()
            self.assertEqual(rounds, [[1]])                                # the failure is in the agreement

    def test_a_fifo_in_the_rank_folder_is_removed_and_never_opened(self):
        import threading

        if not hasattr(os, "mkfifo"):
            self.skipTest("no FIFOs here")
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            first = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            fifo = os.path.join(first.dir, "pipe.safetensors")
            os.mkfifo(fifo)
            outcome = []
            worker = threading.Thread(
                target=lambda: outcome.append(self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)),
                daemon=True)
            worker.start()
            worker.join(15)
            self.assertFalse(worker.is_alive())                           # a blocking open would wait for ever
            self.assertFalse(os.path.exists(fifo))

    def test_a_failure_while_one_rank_expands_the_shared_keys_is_shared(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            class Boom(list):
                def __getitem__(self, index):
                    raise MemoryError("keys")

            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            store.gather, store.share, store.world = gather, lambda values: Boom([1, 2, 3]), 2
            with self.assertRaises(MemoryError):
                store._reconcile()
            self.assertEqual(rounds, [[0], [1]])                           # the failure is in the second agreement

    def test_the_ranks_keep_the_same_builds(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            for name, when in (("spill-00000000000000aa", 1000), ("spill-00000000000000bb", 2000)):
                folder = os.path.join(root, name, "rank0")
                os.makedirs(folder)
                with open(os.path.join(folder, "0.safetensors"), "wb") as f:
                    f.write(b"x")
                os.utime(os.path.join(root, name), (when, when))
            peer = self.sp._encode_builds(["spill-00000000000000aa"])      # the peer lacks the newer build

            flags = iter([0, 0, 1, 0, 0, 0, 0])                           # start, survey, complete, merge, prune...

            def gather(values):
                if len(values) == 8:                                      # the compatibility words
                    return [list(values), list(values)]
                if len(values) == 1:                                      # a failure flag or "my survey is complete"
                    return [list(values), [next(flags)]]
                return [list(values), peer]                               # the builds each rank offers

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    self.sp.SpillStore(cfg, rank=0, world=2, device="cpu", classes=[], quiet=True, gather=gather,
                                       share=lambda values: values or [])
                except Exception:                                         # noqa: BLE001  (the fake peer ends the start)
                    pass
            self.assertTrue(os.path.exists(os.path.join(root, "spill-00000000000000aa", "rank0")))   # common: kept
            self.assertFalse(os.path.exists(os.path.join(root, "spill-00000000000000bb", "rank0")))  # only here: removed

    def test_a_failure_while_one_rank_prunes_builds_is_shared(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "spill-00000000000000aa", "rank0")
            os.makedirs(folder)
            with open(os.path.join(folder, "0.safetensors"), "wb") as f:
                f.write(b"x")
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            store.gather, store.world = gather, 2
            with mock.patch.object(store, "_trim_kept_build", side_effect=MemoryError("sort")):
                with self.assertRaises(MemoryError):
                    store._prune_builds()
            self.assertEqual(rounds[-1], [1])                              # the failure is in the agreement

    def test_the_number_of_kept_builds_is_limited_to_what_the_ranks_exchange(self):
        for value in (-1, self.sp.BUILDS_SHARED + 1):
            with self.assertRaises(ValueError):
                self.sp.SpillConfig(root="/x", gib=1, keep_builds=value)
        self.sp.SpillConfig(root="/x", gib=1, keep_builds=self.sp.BUILDS_SHARED)
        names = [f"spill-{n:016x}" for n in range(self.sp.BUILDS_SHARED)]
        self.assertEqual(self.sp._decode_builds(self.sp._encode_builds(names)), names)

    def test_a_link_in_a_retained_build_takes_no_retention_slot(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            real = os.path.join(root, "spill-00000000000000aa", "rank0")
            linked = os.path.join(root, "spill-00000000000000bb", "rank0")
            for folder in (real, linked):
                os.makedirs(folder)
            with open(os.path.join(real, "0.safetensors"), "wb") as f:
                f.write(b"x")
            target = os.path.join(outside, "target")
            with open(target, "wb") as f:
                f.write(b"x")
            try:
                os.symlink(target, os.path.join(linked, "0.safetensors"))
            except (OSError, NotImplementedError):
                self.skipTest("no symbolic links here")
            os.utime(os.path.join(root, "spill-00000000000000aa"), (1000, 1000))     # the linked build is newer
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(os.path.join(real, "0.safetensors")))      # the healthy rollback stays
            self.assertFalse(os.path.exists(linked))
            self.assertTrue(os.path.exists(target))                                   # the target is not touched

    def test_close_stops_the_threads_and_lets_go_of_the_buffers(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            self.sp.preallocate(cfg)
            self.assertTrue(self.sp._PINNED)
            store, ids, classes = self.stored(root)
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            store.finish_load(lj)                                        # the reader thread runs too
            threads = list(store._threads) + [store._reader]
            store.close()
            self.assertTrue(all(not t.is_alive() for t in threads))
            self.assertFalse(self.sp._PINNED)
            self.assertIsNone(store._read_stage)

    def test_the_joint_choice_of_builds_goes_on_until_enough_common_builds_are_found(self):
        from unittest import mock

        names = [f"spill-{n:016x}" for n in range(6, 0, -1)]            # newest first: ...06 down to ...01
        peer = names[3:]                                                # the peer lacks the three newest
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=2)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            offsets = []

            def gather(values):
                if len(values) == 1:                                      # a failure flag of the agreement
                    return [list(values), list(values)]
                offsets.append(len(offsets) * 3)
                window = offsets[-1]
                return [list(values), self.sp._encode_builds(peer[window:window + 3])]

            with mock.patch.object(self.sp, "BUILDS_SHARED", 3):
                store.gather, store.world = gather, 2
                kept = store._retained_builds(names)
            self.assertEqual(kept, {names[3], names[4]})                # the newest two builds that both have
            self.assertEqual(len(offsets), 2)                           # a second window was needed

    def test_numpy_fields_are_read_into_host_memory(self):
        self.assertEqual(self.sp._numpy_fields({"d": {"host": {"t": "0.host", "np": 1},
                                                      "x": {"l": [{"t": "0.x.0"}, {"t": "0.k", "np": 1}]}}}),
                         {"0.host", "0.k"})
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{KeptLike.__module__}:{KeptLike.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            ids = list(range(100))
            obj = KeptLike(1, 0, 100, 100, np.arange(100, dtype=np.int32),
                           {"32": (torch.arange(8, dtype=torch.uint8), None)}, False, np.arange(100, dtype=np.int64), 1)
            self.assertIsNotNone(store.save(ids, obj, skip=("disk", "hits")))
            self.assertTrue(store.drain(10))
            store.device = "meta"                                         # a device that the NumPy fields must not use
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            lj.done.wait(10)
            kinds = {name: tensor.device.type for name, tensor in lj.dests.items()}
            self.assertEqual(kinds["0.host"], "cpu")
            self.assertEqual(kinds["0.keys"], "cpu")
            self.assertEqual(kinds["0.snaps.32.0"], "meta")

    def test_a_failure_while_one_rank_merges_the_build_windows_is_shared(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store.gather, store.world = gather, 2
            with mock.patch.object(self.sp, "_decode_builds", side_effect=MemoryError("lists")):
                with self.assertRaises(MemoryError):
                    store._retained_builds(["spill-00000000000000aa"])
            self.assertEqual(rounds[-1], [1])                              # the failure is in the agreement

    def test_a_linked_spill_root_is_refused_and_not_followed(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as outside:
            victim = os.path.join(outside, "spill-0000000000000001", "rank0", "0.safetensors")
            os.makedirs(os.path.dirname(victim))
            with open(victim, "wb") as f:
                f.write(b"x")
            root = os.path.join(base, "cache")
            try:
                os.symlink(outside, root)
            except (OSError, NotImplementedError):
                self.skipTest("no symbolic links here")
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            with self.assertRaises(RuntimeError):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(victim))                       # nothing behind the link was touched

    def test_a_capacity_eviction_whose_file_stays_skips_the_save(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=6000 / (1 << 30), min_free_gib=0, min_tokens=1)   # one entry fits
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            first_ids, first = self.new_state(500)
            self.assertIsNotNone(store.save(first_ids, first, skip=("ids",)))
            self.assertTrue(store.drain(10))
            first_path = self.entry_path(store)
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == first_path:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            second_ids, second = self.new_state(900)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(store.save(second_ids, second, skip=("ids",)))     # its room is not free
            self.assertEqual(store.stats.no_space, 1)
            self.assertTrue(os.path.exists(first_path))
            third_ids, third = self.new_state(1300)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(store.save(third_ids, third, skip=("ids",)))       # the stuck file still holds room
            self.assertEqual(len(store.index), 0)
            self.assertIsNotNone(store.save(third_ids, third, skip=("ids",)))        # the file goes at the next try
            self.assertFalse(os.path.exists(first_path))

    def test_a_peer_whose_eviction_file_stays_skips_the_save_on_this_rank(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=((1 << 20) + 1000) / (1 << 30), min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            first_ids, first = self.new_state(500)
            self.assertIsNotNone(store.save(first_ids, first, skip=("ids",)))
            self.assertTrue(store.drain(10))

            def gather(values):
                if len(values) == 5:                                      # the size exchange: the peer agrees
                    return [list(values), list(values)]
                return [list(values), [0, 1]]                             # the peer eviction left its file

            store.gather, store.world = gather, 2
            second_ids, second = self.new_state(900)
            self.assertIsNone(store.save(second_ids, second, skip=("ids",)))
            self.assertEqual(store.stats.no_space, 1)
            self.assertEqual(len(store.index), 0)                         # the index changed alike on both ranks

    def test_a_failure_while_one_rank_makes_room_is_shared(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=((1 << 20) + 1000) / (1 << 30), min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            first_ids, first = self.new_state(500)
            self.assertIsNotNone(store.save(first_ids, first, skip=("ids",)))
            self.assertTrue(store.drain(10))
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store.gather, store.world = gather, 2
            second_ids, second = self.new_state(900)
            with mock.patch.object(store, "_make_room", side_effect=MemoryError("victims")):
                self.assertIsNone(store.save(second_ids, second, skip=("ids",)))
            self.assertEqual(rounds[-1][0], 3)                            # the failure is in the exchange

    def test_an_eviction_that_fails_on_a_rank_is_followed_by_an_index_agreement(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=((1 << 20) + 1000) / (1 << 30), min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            first_ids, first = self.new_state(500)
            self.assertIsNotNone(store.save(first_ids, first, skip=("ids",)))
            self.assertTrue(store.drain(10))
            for mine, theirs in ((3, 0), (0, 3)):                    # the failure is here, or on the peer
                with self.subTest(failing="here" if mine else "peer"):
                    def gather(values, theirs=theirs):
                        if len(values) == 5:                      # the row of the save: the peer is well
                            return [list(values), list(values)]
                        return [list(values), [theirs, 0]]        # the row of the eviction phase

                    store.gather, store.world = gather, 2
                    second_ids, second = self.new_state(900)
                    reconciled = []
                    rooms = mock.patch.object(store, "_make_room", side_effect=MemoryError("victims")) if mine else \
                        mock.patch.object(store, "_make_room", return_value=True)
                    with rooms, mock.patch.object(store, "_reconcile", side_effect=lambda: reconciled.append(1) or 0):
                        store.save(second_ids, second, skip=("ids",))
                    self.assertEqual(reconciled, [1])             # every rank agrees the index before it goes on

    def test_an_entry_swapped_for_a_fifo_or_a_link_after_the_scan_is_never_opened(self):
        import threading

        if not hasattr(os, "mkfifo"):
            self.skipTest("no FIFOs here")
        for how in ("fifo", "link to a fifo"):
            with self.subTest(how=how), tempfile.TemporaryDirectory() as root:
                store, ids, classes = self.stored(root)
                path = self.entry_path(store)
                os.remove(path)
                if how == "fifo":
                    os.mkfifo(path)
                else:
                    target = os.path.join(root, "pipe")
                    os.mkfifo(target)
                    os.symlink(target, path)
                outcome = []

                def attempt():
                    try:
                        store.load_async(self.sp.ids_key(ids), lambda *a: None)
                    except Exception as exc:                      # noqa: BLE001
                        outcome.append(exc)

                worker = threading.Thread(target=attempt, daemon=True)
                worker.start()
                worker.join(15)
                self.assertFalse(worker.is_alive())               # a blocking open would wait for ever
                self.assertEqual(len(outcome), 1)

    def test_a_build_without_a_rank_folder_here_takes_no_retention_slot(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-00000000000000aa", "rank0")
            other = os.path.join(root, "spill-00000000000000bb", "rank1")     # the newer build has only another rank
            for folder in (old, other):
                os.makedirs(folder)
            for folder in (old, other):
                with open(os.path.join(folder, "0.safetensors"), "wb") as f:
                    f.write(b"x")
            os.utime(os.path.join(root, "spill-00000000000000aa"), (1000, 1000))
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(os.path.join(old, "0.safetensors")))      # the build with entries stays
            self.assertTrue(os.path.exists(os.path.join(other, "0.safetensors")))    # another rank's folder is not touched

    def test_no_rank_prunes_builds_when_one_rank_cannot_survey_them(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-00000000000000aa", "rank0")
            os.makedirs(old)
            with open(os.path.join(old, "0.safetensors"), "wb") as f:
                f.write(b"x")
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertFalse(os.path.exists(old))                         # alone, keep_builds=0 removes the build
            os.makedirs(old)
            with open(os.path.join(old, "0.safetensors"), "wb") as f:
                f.write(b"x")
            answers = iter([[[0], [0]], [[1], [0]]])                      # nobody failed; the peer's survey is not complete
            store.gather, store.world = lambda values: next(answers), 2
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(store._prune_builds(), 0)
            self.assertTrue(os.path.exists(old))                          # nothing was pruned on a guess

    def test_a_root_spelled_through_a_link_with_a_trailing_mark_is_refused(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as outside:
            link = os.path.join(base, "cache")
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("no symbolic links here")
            for spelling in (link + "/", link + "/.", link + "//"):
                with self.subTest(spelling=spelling):
                    cfg = self.sp.SpillConfig(root=spelling, gib=1, min_free_gib=0, min_tokens=1)
                    with self.assertRaises(RuntimeError):
                        self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
                    self.assertEqual(os.listdir(outside), [])                 # nothing was made behind the link

    def test_a_memory_failure_while_the_state_is_made_is_shared_at_start(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            with mock.patch.object(self.sp.queue, "Queue", side_effect=MemoryError("queue")):
                with self.assertRaises(RuntimeError):
                    self.sp.SpillStore(cfg, rank=0, world=2, device="cpu", classes=[], quiet=True, gather=gather)
            self.assertEqual(rounds[-1], [1])                              # the failure is in the start agreement

    def test_a_failure_while_one_rank_retries_stuck_files_is_shared(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store.gather, store.world = gather, 2
            ids, snap = self.new_state()
            with mock.patch.object(store, "_retry_stuck", side_effect=MemoryError("stuck")):
                self.assertIsNone(store.save(ids, snap, skip=("ids",)))
            self.assertEqual(rounds[-1][2], 1)                             # the failure flag of the size exchange

    def test_a_failure_to_prepare_the_arguments_of_a_save_is_shared(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            rounds = []

            def gather(values):
                rounds.append(list(values))
                return [list(values), list(values)]

            store.gather, store.world = gather, 2
            ids, snap = self.new_state()
            self.assertIsNone(store.save(ids, snap, skip=("ids",), prepare_failure=MemoryError("rows")))
            self.assertEqual(rounds[-1][2], 1)                             # the failure of the caller joins the exchange

    def test_a_replaced_stuck_file_is_not_stuck_any_more(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            other_ids, other = self.new_state(900)
            job = store.save(other_ids, other, skip=("ids",))
            store.stuck[job.path] = 1                                      # an old record for the path of this write
            self.assertTrue(store.drain(10))
            self.assertNotIn(job.path, store.stuck)

    def test_a_rejected_start_file_that_stays_holds_capacity(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            broken = os.path.join(store.dir, "bad.safetensors")
            with open(broken, "wb") as f:
                f.write(b"junk" * 10)
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == broken:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.stuck, {broken: 40})

    def test_an_unreadable_rank_folder_makes_the_survey_incomplete(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            old = os.path.join(root, "spill-00000000000000aa", "rank0")
            os.makedirs(old)
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=1)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            os.makedirs(old, exist_ok=True)
            with mock.patch.object(self.sp.os, "scandir", side_effect=PermissionError("denied")), \
                    contextlib.redirect_stdout(io.StringIO()):
                builds, complete = store._survey_builds()
            self.assertFalse(complete)
            self.assertEqual([name for name, _ in builds], ["spill-00000000000000aa"])

    def test_folders_that_are_not_builds_of_this_tier_are_never_pruned(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as root:
            keep = os.path.join(root, "spill-backup", "rank0")
            os.makedirs(keep)
            with open(os.path.join(keep, "mine.txt"), "wb") as f:
                f.write(b"x")
            empty = os.path.join(root, "spill-new-scheme", "rank0")
            os.makedirs(empty)
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, keep_builds=0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=[], quiet=True)
            self.assertTrue(os.path.exists(os.path.join(keep, "mine.txt")))
            self.assertTrue(os.path.exists(empty))

    def test_a_partial_file_that_stays_at_start_holds_capacity(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            partial = os.path.join(store.dir, "x.partial")
            with open(partial, "wb") as f:
                f.write(b"p" * 30)
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == partial:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(io.StringIO()):
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.stuck, {partial: 30})

    def test_a_startup_eviction_whose_unlink_fails_is_counted_and_shown(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            real_remove = os.remove

            def remove(target, *args, **kwargs):
                if target == path:
                    raise PermissionError("immutable")
                return real_remove(target, *args, **kwargs)

            cfg = self.sp.SpillConfig(root=root, gib=1e-9, min_free_gib=0, min_tokens=1)
            out = io.StringIO()
            with mock.patch.object(self.sp.os, "remove", remove), contextlib.redirect_stdout(out):
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
                again.announce()
            self.assertEqual(len(again.index), 0)                         # the index is what the ranks agree on
            self.assertEqual(again.stats.unlink_failed, 1)
            self.assertIn("1 files could not be removed", out.getvalue())
            self.assertTrue(os.path.exists(path))                         # the next start retries (reconcile drops it)

    def test_a_single_rank_still_raises_a_preparation_error(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            new = list(range(500, 600))
            snap = Snap(ids=new, ring=torch.arange(len(new) * 4, dtype=torch.uint8).reshape(len(new), 4))
            bad = torch.zeros(4, 4, dtype=torch.uint8).t()
            with self.assertRaises(ValueError):
                store.save(new, snap, skip=("ids",), rows=[("r", bad)])

    def test_header_with_a_shape_larger_than_its_span_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            with open(path, "rb") as f:
                raw = f.read()
            self.assertEqual(raw.count(b'"shape":[100,4]'), 1)
            with open(path, "wb") as f:                  # same length, so the file size and the CRC bytes still fit
                f.write(raw.replace(b'"shape":[100,4]', b'"shape":[100,5]'))
            with self.assertRaises(ValueError):
                store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            again = self.sp.SpillStore(store.cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.info()["entries"], 0)        # the start scan does not trust the header either

    def test_descriptor_check_refuses_gaps_overlaps_negative_numbers_and_unknown_dtypes(self):
        good = {"__metadata__": {}, "a": {"dtype": next(iter(self.sp._TD)), "shape": [2], "data_offsets": [0, 0]}}
        code = good["a"]["dtype"]
        size = torch.empty((), dtype=self.sp._TD[code]).element_size()
        good["a"]["data_offsets"] = [0, 2 * size]
        self.assertEqual(self.sp._check_table(good), 2 * size)
        second = {"dtype": code, "shape": [1], "data_offsets": [2 * size, 3 * size]}
        self.assertEqual(self.sp._check_table({**good, "b": second}), 3 * size)
        bad = [
            {**good, "b": {**second, "data_offsets": [3 * size, 4 * size]}},          # a gap
            {**good, "b": {**second, "data_offsets": [size, 2 * size]}},               # an overlap
            {"a": {**good["a"], "shape": [-2], "data_offsets": [0, 2 * size]}},         # a negative shape
            {"a": {**good["a"], "dtype": "nope"}},                                      # an unknown dtype
            {"a": {**good["a"], "shape": [10 ** 12], "data_offsets": [0, 2 * size]}},   # a huge shape
        ]
        for header in bad:
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    self.sp._check_table(header)

    def test_good_restore_counts_no_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            store.finish_load(store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None))
            self.assertEqual(store.info()["load_failed"], 0)
            self.assertEqual(store.info()["loaded"], 1)

    def test_shutdown_budget_follows_the_flush_setting_below_the_docker_stop_limit(self):
        with tempfile.TemporaryDirectory() as root:
            for flush, budget in ((10.0, 10.0), (60.0, 55.0), (100.0, 55.0), (0.0, 0.0)):
                with self.subTest(flush=flush):
                    cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, flush_s=flush)
                    store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", quiet=True)
                    self.assertEqual(store.shutdown_budget(), budget)
            self.assertLess(self.sp.SHUTDOWN_CAP_S, 60.0)

    def test_shutdown_drain_uses_the_budget_and_reports_unfinished_writes(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, flush_s=7.0)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", quiet=True)
            seen = []
            store.drain = lambda timeout=None: seen.append(timeout) or True
            self.assertTrue(store.shutdown_drain())
            self.assertEqual(seen, [7.0])
            store.drain = lambda timeout=None: False
            self.assertFalse(store.shutdown_drain())

    def test_token_ids_are_stored_as_base64_and_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as root:
            ids = np.arange(5000, dtype=np.int32) * 7 + 3
            store, _, classes = self.stored(root, ids=ids)
            with open(self.entry_path(store), "rb") as f:
                header, _ = self.sp._read_header_file(f)
            tokens = header["__metadata__"]["tokens"]
            self.assertFalse(tokens.startswith("["))
            self.assertTrue(np.array_equal(self.sp._decode_ids(tokens), ids.astype(np.int64)))
            again = self.sp.SpillStore(store.cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertTrue(again.has(ids))
            self.assertEqual(again.info()["entries"], 1)

    def test_image_keyed_prompt_keeps_both_id_lists(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            keys = np.array([5, -(1 << 40), 7, 1 << 50], dtype=np.int64)
            tokens = np.array([1, 2, 3, 4], dtype=np.int32)
            snap = Snap(ids=[], ring=torch.zeros((4, 4), dtype=torch.uint8))
            self.assertIsNotNone(store.save(keys, snap, skip=("ids",), tokens=tokens))
            self.assertTrue(store.drain(10))
            again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            [entry] = again.index.values()
            self.assertTrue(np.array_equal(entry.ids, keys))

    def test_one_million_token_ids_are_cheap_to_encode(self):
        ids = np.arange(1_000_000, dtype=np.int32)
        start = self.sp.time.perf_counter()
        text = self.sp._encode_ids(ids)
        elapsed = self.sp.time.perf_counter() - start
        self.assertTrue(np.array_equal(self.sp._decode_ids(text), ids.astype(np.int64)))
        self.assertLess(elapsed, 0.25, "encoding 1M ids must stay far below a JSON list's cost")


if __name__ == "__main__":
    unittest.main()
