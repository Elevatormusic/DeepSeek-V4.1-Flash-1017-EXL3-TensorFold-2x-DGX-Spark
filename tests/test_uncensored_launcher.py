"""Check launcher refusal paths with local shell stubs. Start no server."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        for name in ("uncensored.sh", "common.sh", "prepare_uncensored.py", "uncensored_tensors.json"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.trace = self.root / "trace"
        self.env = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                    "TRACE": str(self.trace), "GPU_ACTIVE": "0", "WORKER_GPU_ACTIVE": "0"}
        self.stub("docker", '''#!/usr/bin/env bash
set -eu
case "$1 $2" in
  'ps -q') if [ "$GPU_ACTIVE" = 1 ]; then echo test-gpu; fi ;;
  'inspect --format')
    case "$3" in *DeviceRequests*) echo 1 ;; *) echo runc ;; esac ;;
  *) echo "unexpected docker command: $*" >&2; exit 9 ;;
esac
''')
        self.stub("ssh", '''#!/usr/bin/env bash
set -eu
command=${!#}
case "$command" in
  sha256sum*) printf '%s  model/override-dealign-bf16.safetensors\n' "$WORKER_HASH" ;;
  *) GPU_ACTIVE=$WORKER_GPU_ACTIVE bash -c "$command" ;;
esac
''')
        for action in ("start", "stop", "status", "verify"):
            (self.root / f"{action}.sh").write_text(
                '#!/usr/bin/env bash\nset -eu\n. "$(dirname "$0")/scripts/common.sh"\n'
                f'printf "{action}:%s\\n" "$CACHE_DIR" >> "$TRACE"\n', encoding="utf-8")

    def stub(self, name, text):
        path = self.bin / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def prepare_settings(self):
        view = self.root / "view"
        (view / "model").mkdir(parents=True)
        payload = b"launcher fixture: no model weights"
        digest = hashlib.sha256(payload).hexdigest()
        manifest = json.loads((ROOT / "scripts/uncensored_tensors.json").read_text(encoding="utf-8"))
        receipt = {**manifest, "override_sha256": digest}
        (view / "override-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        (view / "model/override-dealign-bf16.safetensors").write_bytes(payload)
        (view / "override.sha256").write_text(
            f"{digest}  model/override-dealign-bf16.safetensors\n", encoding="ascii")
        (self.root / ".env").write_text(
            'BIND_HOST=127.0.0.1\nPORT=8888\nWORKER_SSH=test-worker\n', encoding="utf-8")
        (self.root / ".env.uncensored").write_text(
            '. "$ROOT/.env"\nTF_COMMIT=test-commit\nIMAGE=tensorfold:test\n'
            'MODEL_VIEW=$ROOT/view\nCACHE_DIR=$ROOT/cache/${UNCENSORED_CACHE_ID:-unverified}\n',
            encoding="utf-8")
        self.env["WORKER_HASH"] = digest
        return digest

    def run_wrapper(self, *arguments):
        return subprocess.run(["bash", str(self.root / "scripts/uncensored.sh"), *arguments],
                              env=self.env, text=True, capture_output=True, timeout=30, check=False)

    def test_help_needs_no_env_and_invalid_commands_fail(self):
        result = self.run_wrapper("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        for arguments in (("not-a-command",), ("start", "extra")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_wrapper(*arguments).returncode, 2)

    def test_worker_hash_mismatch_refuses_start(self):
        self.prepare_settings()
        self.env["WORKER_HASH"] = "0" * 64
        result = self.run_wrapper("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker override differs", result.stdout)
        self.assertFalse(self.trace.exists())

    def test_active_gpu_on_either_node_refuses_start(self):
        self.prepare_settings()
        for variable in ("GPU_ACTIVE", "WORKER_GPU_ACTIVE"):
            with self.subTest(node=variable):
                self.env[variable] = "1"
                result = self.run_wrapper("start")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("a GPU container is running", result.stderr)
                self.assertFalse(self.trace.exists())
                self.env[variable] = "0"

    def test_start_and_status_use_the_full_weight_hash_cache(self):
        digest = self.prepare_settings()
        for action in ("start", "status", "stop", "verify"):
            with self.subTest(action=action):
                result = self.run_wrapper(action)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected_cache = self.root / "cache" / digest
        self.assertEqual(self.trace.read_text(encoding="utf-8").splitlines(),
                         [f"{action}:{expected_cache}" for action in ("start", "status", "stop", "verify")])


if __name__ == "__main__":
    unittest.main()
