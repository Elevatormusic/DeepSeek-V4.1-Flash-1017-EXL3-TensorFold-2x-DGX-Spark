"""Check extraction and view isolation with small local tensors."""

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "prepare_uncensored", Path(__file__).resolve().parents[1] / "scripts/prepare_uncensored.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
linux = unittest.skipUnless(sys.platform == "linux", "atomic view publication requires Linux")


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.base = self.root / "base"
        for folder in ("model", "engram"):
            (self.base / folder).mkdir(parents=True)
        for index in range(1, 44):
            (self.base / "model" / f"model-{index:05d}-of-00043.safetensors").write_bytes(b"base")
        for name in ("config.json", "chat_template.jinja"):
            (self.base / "model" / name).write_text("base", encoding="utf-8")
        (self.base / "model/model.safetensors.index.json").write_bytes(b"base index")
        for name in ("config.json", "model.safetensors.index.json",
                     "model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
            (self.base / "engram" / name).write_bytes(b"base")
        self.payload = b"\x00\x00\x80\x3f\x00\x40\x40\x40"
        self.name = "layers.10.attn.wo_b.weight"
        self.manifest = {"source": {"revision": "test-source"}, "base": {"revision": "test-base"},
                         "tensors": {self.name: {"file": "source.safetensors", "shape": [2, 2],
                                                "sha256": hashlib.sha256(self.payload).hexdigest()}}}
        self.write_source()

    def write_source(self, dtype="BF16", shape=None, offsets=None):
        entry = {"dtype": dtype, "shape": [2, 2] if shape is None else shape,
                 "data_offsets": [0, 8] if offsets is None else offsets}
        header = json.dumps({self.name: entry}).encode()
        (self.source / "source.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + self.payload)

    @linux
    def test_extract_and_new_view_leave_base_unchanged(self):
        view = self.root / "view"
        digest = PREPARE.build_view(self.source, self.base, view, self.manifest)
        self.assertEqual(digest, PREPARE.check_view(view, self.manifest))
        source = self.base / "model/model-00001-of-00043.safetensors"
        target = view / "model/model-00001-of-00043.safetensors"
        self.assertTrue(os.path.samefile(source, target))
        self.assertFalse((self.base / "model" / PREPARE.OVERRIDE).exists())
        self.assertEqual((self.base / "model/model.safetensors.index.json").read_bytes(), b"base index")
        self.assertFalse((view / "model/model.safetensors.index.json").exists())
        self.assertTrue(os.path.samefile(self.base / "engram/model.safetensors.index.json",
                                        view / "engram/model.safetensors.index.json"))
        with (view / "model" / PREPARE.OVERRIDE).open("rb") as stream:
            header, _ = PREPARE.read_header(stream)
            self.assertEqual(header[self.name]["dtype"], "BF16")
            self.assertEqual(stream.read(), self.payload)

    @linux
    def test_receipt_marks_base_bytes_as_unverified(self):
        view = self.root / "view"
        PREPARE.build_view(self.source, self.base, view, self.manifest)
        receipt = json.loads((view / "override-receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["base"], self.manifest["base"])
        self.assertEqual(receipt["base_verification"], "not_verified_by_preparation")
        self.assertEqual(receipt["omitted_base_files"], ["model/model.safetensors.index.json"])

    @linux
    def test_check_view_accepts_old_receipts_without_base_status(self):
        view = self.root / "view"
        digest = PREPARE.build_view(self.source, self.base, view, self.manifest)
        path = view / "override-receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt.pop("base_verification", None)
        receipt.pop("omitted_base_files", None)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assertEqual(PREPARE.check_view(view, self.manifest), digest)

    @linux
    def test_destination_created_during_extraction_is_not_replaced(self):
        view = self.root / "view"
        extract = PREPARE.extract
        foreign = []

        def extract_and_create(*args):
            extract(*args)
            view.mkdir()
            foreign.append(view.stat())

        with mock.patch.object(PREPARE, "extract", side_effect=extract_and_create):
            with self.assertRaises(FileExistsError):
                PREPARE.build_view(self.source, self.base, view, self.manifest)
        self.assertTrue(os.path.samestat(foreign[0], view.stat()))
        self.assertEqual(list(view.iterdir()), [])
        self.assertEqual(list(self.root.glob(".view-*")), [])

    def test_publication_fails_closed_on_unsupported_platform(self):
        temporary = self.root / "temporary"
        temporary.mkdir()
        view = self.root / "view"
        with mock.patch.object(PREPARE.sys, "platform", "unsupported"):
            with self.assertRaisesRegex(OSError, "requires Linux"):
                PREPARE.publish_view(temporary, view)
        self.assertTrue(temporary.is_dir())
        self.assertFalse(view.exists())

    @linux
    def test_publication_fails_closed_on_unsupported_kernel(self):
        temporary = self.root / "temporary"
        temporary.mkdir()
        view = self.root / "view"
        libc = mock.Mock()
        libc.renameat2.return_value = -1
        with (mock.patch.object(PREPARE.ctypes, "CDLL", return_value=libc),
              mock.patch.object(PREPARE.ctypes, "get_errno", return_value=errno.ENOSYS)):
            with self.assertRaises(OSError) as raised:
                PREPARE.publish_view(temporary, view)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.assertTrue(temporary.is_dir())
        self.assertFalse(view.exists())

    def test_bad_payload_does_not_publish_or_leave_partial_view(self):
        self.manifest["tensors"][self.name]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            PREPARE.build_view(self.source, self.base, self.root / "view", self.manifest)
        self.assertFalse((self.root / "view").exists())
        self.assertEqual(list(self.root.glob(".view-*")), [])

    def test_bad_dtype_shape_and_ranges_fail(self):
        for kwargs in ({"dtype": "F16"}, {"shape": [4, 1]}, {"offsets": [-1, 7]},
                       {"offsets": [0, 7]}, {"offsets": [1, 9]}, {"offsets": [False, 8]}):
            with self.subTest(kwargs=kwargs):
                self.write_source(**kwargs)
                with self.assertRaises(ValueError):
                    PREPARE.build_view(self.source, self.base, self.root / "view", self.manifest)

    def test_refuse_existing_view_and_base_subfolder(self):
        for view in (self.base, self.base / "new-view", self.source / "new-view"):
            with self.subTest(view=view), self.assertRaises(ValueError):
                PREPARE.build_view(self.source, self.base, view, self.manifest)

    def test_header_size_and_repeated_keys_fail(self):
        path = self.source / "bad.safetensors"
        for data in (b"bad", struct.pack("<Q", PREPARE.MAX_HEADER + 1),
                     struct.pack("<Q", 13) + b'{"a":1,"a":2}'):
            with self.subTest(data=data):
                path.write_bytes(data)
                with path.open("rb") as stream, self.assertRaises(ValueError):
                    PREPARE.read_header(stream)

    @linux
    def test_damaged_override_is_rejected(self):
        view = self.root / "view"
        PREPARE.build_view(self.source, self.base, view, self.manifest)
        with (view / "model" / PREPARE.OVERRIDE).open("ab") as stream:
            stream.write(b"damage")
        with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
            PREPARE.check_view(view, self.manifest)

    def test_manifest_has_only_the_pinned_54_tensors(self):
        manifest = json.loads(PREPARE.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["revision"], "89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c")
        self.assertEqual(manifest["base"]["revision"], "afc7aa0aab98ee735c0f54e0a122019af1ac08f1")
        expected = set()
        for layer in range(10, 37):
            for suffix, shape in (("attn.wo_b.weight", [5120, 8192]),
                                  ("ffn.shared_experts.w2.weight", [5120, 2304])):
                name = f"layers.{layer}.{suffix}"
                expected.add(name)
                item = manifest["tensors"][name]
                self.assertEqual(item["shape"], shape)
                self.assertEqual(item["file"], f"model-{layer + 2:05d}-of-00043.safetensors")
                self.assertRegex(item["sha256"], r"^[a-f0-9]{64}$")
        self.assertEqual(set(manifest["tensors"]), expected)


if __name__ == "__main__":
    unittest.main()
