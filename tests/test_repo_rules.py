"""Offline checks of the repo rules in CONTRIBUTING.md (stdlib only; no server, no GPU).

Run: python3 -m unittest discover -s tests -p 'test_*.py'   (or tests/run_offline.sh)
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a git checkout")
class RunsFolder(unittest.TestCase):
    """runs/ holds the evidence for every number in the docs. Git must track it, in one layout."""

    def test_runs_not_ignored(self):
        for probe in ("runs/2099-01-01/INDEX.md", "runs/2099-01-01/bench.log", "runs/2099-01-01/x/results.json"):
            with self.subTest(path=probe):
                r = git("check-ignore", "--no-index", "-q", probe)
                self.assertEqual(r.returncode, 1, f"{probe} is ignored by git; runs/ must stay tracked")

    def test_day_folders(self):
        tracked = git("ls-files", "runs").stdout.split()
        self.assertTrue(tracked, "runs/ has no tracked files")
        for top in sorted({p.split("/")[1] for p in tracked}):
            with self.subTest(entry=top):
                self.assertRegex(top, RUN_DAY, "each entry in runs/ must be a runs/YYYY-MM-DD/ folder")

    def test_day_folders_have_index(self):
        tracked = set(git("ls-files", "runs").stdout.split())
        for day in sorted({p.split("/")[1] for p in tracked if RUN_DAY.match(p.split("/")[1])}):
            with self.subTest(day=day):
                self.assertIn(f"runs/{day}/INDEX.md", tracked, "each run day needs an INDEX.md")


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a git checkout")
class ExecutableScripts(unittest.TestCase):
    """start.sh runs scripts by their path. Git must store each script with the executable bit, or a new clone stops
    with "Permission denied" (issue #15). runs/ is a record and stays as it ran."""

    def test_scripts_are_executable(self):
        bad = []
        for line in git("ls-files", "-s", "--", ":!:runs/**").stdout.splitlines():
            meta, name = line.split("\t", 1)
            mode = meta.split()[0]
            if mode == "120000":
                continue                                      # a symlink: it has no mode of its own
            with open(ROOT / name, "rb") as f:
                shebang = f.read(2) == b"#!"
            if (shebang or name.endswith(".sh")) and mode != "100755":
                bad.append(f"{name} ({mode})")
        self.assertEqual(bad, [], "scripts without the executable bit; run git update-index --chmod=+x <file>:\n"
                         + "\n".join(bad))


class EnvExample(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash") and os.name == "posix", "needs bash on Linux or macOS")
    def test_env_example_sources(self):
        """The scripts source .env with set -a. .env.example must source cleanly."""
        r = subprocess.run(
            ["bash", "-c", 'set -euo pipefail; set -a; . "$1"; set +a', "_", str(ROOT / ".env.example")],
            capture_output=True, text=True, check=False, env={"HOME": "/home/ci", "PATH": os.environ["PATH"]},
        )
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_no_secrets_in_env_example(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for pattern in (r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9_-]{20,}", r"ghp_[A-Za-z0-9]{30,}"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, text), "a token-like value is in .env.example")


@unittest.skipUnless(importlib.util.find_spec("tensorfold"), "needs the tensorfold package (engine image)")
class Conformance(unittest.TestCase):
    """The prompt-path check imports tensorfold. Without the package (GitHub runners), this test skips."""

    def test_conformance_module_imports(self):
        sys.path.insert(0, str(ROOT / "tests"))
        try:
            import v41_conformance  # noqa: F401
        finally:
            sys.path.remove(str(ROOT / "tests"))


if __name__ == "__main__":
    unittest.main()
