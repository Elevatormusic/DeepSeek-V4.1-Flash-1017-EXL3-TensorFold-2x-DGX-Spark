"""Composition checks. Real source tests need TF_SPILL_SOURCE explicitly."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT.parent / "display-kv" / "apply.py"
SPILL = ROOT / "apply.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose_module():
    return load(ROOT / "compose.py", "display_spill_compose")


class SyntheticCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.display = load(DISPLAY, "display_apply_test")
        self.spill = load(SPILL, "spill_apply_test")
        self.compose = compose_module()
        self.inputs = {}
        for feature in (self.display, self.spill):
            for relative in feature.FILES:
                self.inputs[Path(relative)] = "# fixture " + str(relative) + "\n"
        for relative, text in self.inputs.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        self.hashes = {path: hashlib.sha256(text.encode()).hexdigest()
                       for path, text in self.inputs.items()}
        self.patches = [
            mock.patch.object(self.compose, "load",
                              side_effect=lambda path, name: self.display if "display-kv" in str(path) else self.spill),
            mock.patch.object(self.display, "FILES", self.hashes),
            mock.patch.object(self.spill, "FILES", self.hashes),
            mock.patch.object(self.compose.subprocess, "check_output",
                              return_value=self.display.REVISION + "\n"),
            mock.patch.object(self.display, "transform", side_effect=lambda path, text: text + "# display\n"),
            mock.patch.object(self.spill, "transform", side_effect=lambda path, text: text + "# spill\n"),
            mock.patch.object(self.display, "overlays", return_value={Path("src/display.py"): "display = 1\n"}),
            mock.patch.object(self.spill, "overlays", return_value={Path("src/spill.py"): "spill = 1\n"}),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_composes_both_features_and_validates_before_writes(self):
        self.compose.compose(self.root, DISPLAY, SPILL)
        for relative in self.inputs:
            text = (self.root / relative).read_text()
            self.assertTrue(text.endswith("# display\n# spill\n"), text)
            ast.parse(text)
        self.assertEqual((self.root / "src/display.py").read_text(), "display = 1\n")
        self.assertEqual((self.root / "src/spill.py").read_text(), "spill = 1\n")

    def test_rejects_revision_hash_overlay_and_syntax_without_writes(self):
        before = self.snapshot()
        with mock.patch.object(self.compose.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaises(ValueError):
                self.compose.compose(self.root, DISPLAY, SPILL)
        self.assertEqual(before, self.snapshot())
        target = self.root / next(iter(self.inputs))
        target.write_text("drift\n")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.compose.compose(self.root, DISPLAY, SPILL)
        self.assertEqual(before, self.snapshot())
        target.write_text(self.inputs[target.relative_to(self.root)])
        (self.root / "src/spill.py").write_text("already\n")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.compose.compose(self.root, DISPLAY, SPILL)
        self.assertEqual(before, self.snapshot())
        (self.root / "src/spill.py").unlink()
        with mock.patch.object(self.spill, "transform", return_value="not python!"):
            before = self.snapshot()
            with self.assertRaises(SyntaxError):
                self.compose.compose(self.root, DISPLAY, SPILL)
            self.assertEqual(before, self.snapshot())


@unittest.skipUnless(os.environ.get("TF_SPILL_SOURCE"), "set TF_SPILL_SOURCE to a pristine pinned checkout")
class RealSourceCompositionTests(unittest.TestCase):
    def test_display_then_spill_compiles(self):
        source = Path(os.environ["TF_SPILL_SOURCE"])
        display = load(DISPLAY, "display_real")
        spill = load(SPILL, "spill_real")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for feature in (display, spill):
                for relative in feature.FILES:
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / relative, target)
            with mock.patch.object(subprocess, "check_output", return_value=display.REVISION + "\n"):
                compose_module().compose(root, DISPLAY, SPILL)
            for feature in (display, spill):
                for relative in feature.FILES:
                    ast.parse((root / relative).read_text())
            self.assertTrue((root / "src/tensorfold/cuda/spill.py").is_file())
            self.assertTrue((root / display.FAMILY / "display_kv.py").is_file())


if __name__ == "__main__":
    unittest.main()
