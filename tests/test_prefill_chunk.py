"""Run the real launcher with stdlib fixtures and mocked external commands only."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class PrefillChunkTests(unittest.TestCase):
    def launch(self, value=None, preflight_rc=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "start.sh", root / "start.sh")
            common = (ROOT / "scripts/common.sh").read_text()
            common += r'''
# All service and network operations are mocks in this temporary fixture.
export ROOT
docker() { printf 'docker %s\n' "$*" >> "$ROOT/commands"; }
wssh() { printf 'wssh %s\n' "$*" >> "$ROOT/commands"; }
ss() { echo 'LISTEN 0 128 *:29551 '; }
healthy() { return 0; }
sleep() { echo 'unexpected sleep' >&2; exit 99; }
'''
            (root / "scripts/common.sh").write_text(common)
            for script, rc in (("preflight.sh", preflight_rc), ("build_extensions.sh", 0)):
                path = root / "scripts" / script
                path.write_text(f'#!/bin/bash\necho {script} >> "$ROOT/commands"\nexit {rc}\n')
                path.chmod(0o755)
            env = (ROOT / ".env.example").read_text()
            env = "\n".join(line for line in env.splitlines()
                            if not line.startswith("PREFILL_CHUNK="))
            env += f"\nCACHE_DIR={shlex.quote(str(root / 'cache'))}\n"
            env += "ENGINE=python\n"      # these tests check the Python start script
            if value is not None:
                env += f"PREFILL_CHUNK={shlex.quote(value)}\n"
            (root / ".env").write_text(env)
            environment = {key: val for key, val in os.environ.items()
                           if key not in {"PREFILL_CHUNK", "BASH_ENV", "SHELLOPTS"}
                           and not key.startswith("BASH_FUNC_")}
            environment["PREFLIGHT"] = "1"
            result = subprocess.run(["bash", str(root / "start.sh")], env=environment,
                                    capture_output=True, text=True, timeout=10)
            commands = root / "commands"
            return result, commands.read_text() if commands.exists() else ""

    def test_default_and_explicit_values_reach_both_ranks(self):
        for value, expected in ((None, "2048"), ("", "2048"), ("2048", "2048"),
                                ("512", "512"), ("1", "1")):
            with self.subTest(value=value):
                result, commands = self.launch(value)
                self.assertEqual(result.returncode, 0, result.stderr)
                runs = [line for line in commands.splitlines() if "docker run " in line]
                self.assertEqual(len(runs), 2, commands)
                for rank, command in enumerate(runs):
                    args = shlex.split(command)
                    self.assertIn(f"TF_DS_PREFILL_CHUNK={expected}", args)
                    self.assertIn(f"tfv41-r{rank}", args)
                    self.assertIn("--no-thinking", args)
                    self.assertNotIn("--thinking", args)
                self.assertTrue(commands.startswith("preflight.sh\n"), commands)
                self.assertIn("build_extensions.sh\n", commands)

    def test_invalid_values_fail_before_any_commands(self):
        for value in ("0", "000", "-1", "+512", "1.5", "abc", "1e3", "0x200",
                      " 512", "512 ", "512\n1", "512;false", "$(false)"):
            with self.subTest(value=value):
                result, commands = self.launch(value)
                self.assertEqual(result.returncode, 1)
                self.assertIn("PREFILL_CHUNK must be a positive integer", result.stderr)
                self.assertEqual(commands, "")

    def test_preflight_failure_still_blocks_start(self):
        result, commands = self.launch("512", preflight_rc=1)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(commands, "preflight.sh\n")


if __name__ == "__main__":
    unittest.main()
