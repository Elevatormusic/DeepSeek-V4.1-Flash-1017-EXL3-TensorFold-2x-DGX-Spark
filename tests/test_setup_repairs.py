"""Offline regressions for the Zig setup and test paths."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_memory_checks import ENV_BASE, Tree, needs_bash, write_exec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pack_check", ROOT / "scripts/check_zig_pack.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class PackLayout(unittest.TestCase):
    def test_exact_weights_and_pinned_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            for name in pack.WEIGHTS:
                (model / name).touch()
            fixture = b'{}'
            saved = pack.MANIFESTS
            self.addCleanup(setattr, pack, "MANIFESTS", saved)
            pack.MANIFESTS = {name: hashlib.sha256(fixture).hexdigest() for name in saved}
            for name in saved:
                (model / name).write_bytes(fixture)
            pack.check(model)
            (model / "model-00001-of-00043.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "exact 42"):
                pack.check(model)
            (model / "model-00001-of-00043.safetensors").unlink()
            (model / "config.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                pack.check(model)

    def test_pins_match_the_recorded_pack(self):
        manifest = (ROOT / "runs/2026-10-22/zig-28/weights-2.8bpw.sha256").read_text()
        for name, digest in pack.MANIFESTS.items():
            self.assertIn(f"{digest}  model/{name}", manifest)


@needs_bash
class SetupPaths(unittest.TestCase):
    def test_view_accepts_each_packs_own_filenames(self):
        for engine, manifest in (("zig", "runs/2026-10-22/zig-28/weights-2.8bpw.sha256"),
                                 ("python", "runs/2026-10-17/weights.sha256")):
            with self.subTest(engine=engine):
                t = Tree(self, "", ("common.sh", "download.sh"))
                (t.dir / ".env").write_text(ENV_BASE.format(d=t.dir) + f"ENGINE={engine}\n")
                source = t.dir / "model-view.src"
                for line in (ROOT / manifest).read_text().splitlines():
                    name = line.split()[1]
                    path = source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fixture")
                result = t.run(["bash", "scripts/download.sh", "view"])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((t.dir / "model-view/model/config.json").samefile(source / "model/config.json"))

    def test_zig_run_all_tags_native_image_and_skips_python_prompt_checks(self):
        t = Tree(self, "", ("common.sh", "zig_common.sh"))
        (t.dir / ".env").write_text(ENV_BASE.format(d=t.dir) +
                                    "ENGINE=zig\nZIG_IMAGE=tensorfold-dsv41-zig:d58d25cc\n")
        tests = t.dir / "tests"
        tests.mkdir()
        shutil.copy(ROOT / "tests/run_all.sh", tests / "run_all.sh")
        write_exec(t.dir / "verify.sh", "#!/bin/sh\necho '5/5 passed'\n")
        (tests / "v41_suite.py").write_text("print('30/30 passed in 1s')\n")
        (tests / "v41_grammar_check.py").write_text("print('14/14 passed')\n")
        t.stub("curl", 'echo \'{"data":[{"id":"DeepSeek-V4.1-Flash-1017"}]}\'\n')
        result = t.run(["bash", "tests/run_all.sh"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("image tag d58d25cc", result.stdout)
        self.assertIn("[SKIP] conformance: this test checks the Python prompt path", result.stdout)
        self.assertIn("[SKIP] differential: this test checks the Python prompt path", result.stdout)
        self.assertTrue(list((t.dir / "runs").glob("*/runall-d58d25cc-*-summary.txt")))

    def test_direct_entry_points_have_executable_git_modes(self):
        names = [f"scripts/{name}.sh" for name in
                 ("zig_start", "zig_inputs", "zig_build_image", "zig_supervise", "zig_gid", "verify_kit")]
        result = subprocess.run(["git", "ls-files", "-s", "--", *names], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(names))
        self.assertTrue(all(line.startswith("100755 ") for line in lines), result.stdout)
