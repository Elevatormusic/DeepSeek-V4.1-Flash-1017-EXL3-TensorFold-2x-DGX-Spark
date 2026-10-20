"""Check launcher refusal paths with local shell stubs. Start no server."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"
TEST_IMAGE = f"tensorfold:{TEST_COMMIT[:12]}"


class ExamplePinTests(unittest.TestCase):
    def test_example_image_tag_matches_commit(self):
        text = (ROOT / ".env.uncensored.example").read_text(encoding="utf-8")
        commit = re.search(r"^TF_COMMIT=([0-9a-f]{40})$", text, re.MULTILINE)
        image = re.search(r"^IMAGE=[^\s:]+:([0-9a-f]{12,40})$", text, re.MULTILINE)
        self.assertIsNotNone(commit, "the example needs a full commit SHA")
        self.assertIsNotNone(image, "the example needs a commit prefix image tag")
        self.assertTrue(commit.group(1).startswith(image.group(1)))


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
        self.operations = self.root / "operations"
        self.env = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                    "TRACE": str(self.trace), "OPERATIONS": str(self.operations),
                    "GPU_ACTIVE": "0", "WORKER_GPU_ACTIVE": "0",
                    "DOCKER_PS_FAIL": "0", "WORKER_DOCKER_PS_FAIL": "0",
                    "DOCKER_INSPECT_FAIL": "", "WORKER_DOCKER_INSPECT_FAIL": "",
                    "WORKER_INDEX_QUERY_FAIL": "0"}
        self.stub("docker", '''#!/usr/bin/env bash
set -eu
printf 'docker:%s\\n' "$*" >> "$OPERATIONS"
case "$1 $2" in
  'ps -q')
    [ "$DOCKER_PS_FAIL" != 1 ] || { echo "docker ps failed" >&2; exit 7; }
    if [ "$GPU_ACTIVE" = 1 ]; then echo test-gpu; fi ;;
  'inspect --format')
    case "$3" in
      *DeviceRequests*)
        [ "$DOCKER_INSPECT_FAIL" != gpu ] || exit 8
        echo 1 ;;
      *)
        [ "$DOCKER_INSPECT_FAIL" != runtime ] || exit 8
        echo runc ;;
    esac ;;
  *) echo "unexpected docker command: $*" >&2; exit 9 ;;
esac
''')
        self.stub("ssh", '''#!/usr/bin/env bash
set -eu
command=${!#}
printf 'ssh:%s\\n' "$command" >> "$OPERATIONS"
case "$command" in
  'test ! -e '*)
    [ "$WORKER_INDEX_QUERY_FAIL" != 1 ] || { echo "worker index query failed" >&2; exit 7; }
    bash -c "${command//"$MODEL_VIEW"/"$WORKER_MODEL_VIEW"}" ;;
  sha256sum*) printf '%s  model/override-dealign-bf16.safetensors\n' "$WORKER_HASH" ;;
  *) GPU_ACTIVE=$WORKER_GPU_ACTIVE DOCKER_PS_FAIL=$WORKER_DOCKER_PS_FAIL
     export GPU_ACTIVE DOCKER_PS_FAIL
     DOCKER_INSPECT_FAIL=$WORKER_DOCKER_INSPECT_FAIL bash -c "$command" ;;
esac
''')
        for action in ("start", "stop", "status", "verify"):
            (self.root / f"{action}.sh").write_text(
                '#!/usr/bin/env bash\nset -eu\n. "$(dirname "$0")/scripts/common.sh"\n'
                f'printf "{action}:%s\\n" "$CACHE_DIR" >> "$TRACE"\n', encoding="utf-8")

        (self.root / "docker").mkdir()
        (self.root / "docker/build.sh").write_text(
            '#!/usr/bin/env bash\nprintf "build\\n" >> "$TRACE"\n', encoding="utf-8")

    def stub(self, name, text):
        path = self.bin / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def prepare_settings(self, commit=TEST_COMMIT, image=TEST_IMAGE):
        view = self.root / "view"
        (view / "model").mkdir(parents=True, exist_ok=True)
        payload = b"launcher fixture: no model weights"
        digest = hashlib.sha256(payload).hexdigest()
        manifest = json.loads((ROOT / "scripts/uncensored_tensors.json").read_text(encoding="utf-8"))
        receipt = {**manifest, "override_sha256": digest}
        (view / "override-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        (view / "model/override-dealign-bf16.safetensors").write_bytes(payload)
        (view / "override.sha256").write_text(
            f"{digest}  model/override-dealign-bf16.safetensors\n", encoding="ascii")
        worker_view = self.root / "worker-view"
        (worker_view / "model").mkdir(parents=True, exist_ok=True)
        self.env["WORKER_MODEL_VIEW"] = str(worker_view)
        (self.root / ".env").write_text(
            'BIND_HOST=127.0.0.1\nPORT=8888\nWORKER_SSH=test-worker\n', encoding="utf-8")
        (self.root / ".env.uncensored").write_text(
            f'. "$ROOT/.env"\nTF_COMMIT={commit}\nIMAGE={image}\n'
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

    def test_model_index_on_head_refuses_start_before_worker_or_docker_checks(self):
        self.prepare_settings()
        view = self.root / "view"
        index = view / "model/model.safetensors.index.json"
        receipt = (view / "override-receipt.json").read_bytes()
        for dangling in (False, True):
            with self.subTest(dangling=dangling):
                if dangling:
                    index.symlink_to("missing-index.json")
                else:
                    index.write_bytes(b"stale index")
                result = self.run_wrapper("start")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not contain model/model.safetensors.index.json", result.stderr)
                self.assertFalse(self.trace.exists())
                self.assertFalse(self.operations.exists())
                self.assertEqual((view / "override-receipt.json").read_bytes(), receipt)
                if dangling:
                    self.assertTrue(index.is_symlink())
                    self.assertEqual(index.readlink(), Path("missing-index.json"))
                else:
                    self.assertEqual(index.read_bytes(), b"stale index")
                index.unlink()

    def test_model_index_on_worker_refuses_start_before_hash_or_docker_checks(self):
        self.prepare_settings()
        index = Path(self.env["WORKER_MODEL_VIEW"]) / "model/model.safetensors.index.json"
        receipt_path = self.root / "view/override-receipt.json"
        receipt = receipt_path.read_bytes()
        for dangling in (False, True):
            with self.subTest(dangling=dangling):
                if dangling:
                    index.symlink_to("missing-index.json")
                else:
                    index.write_bytes(b"stale index")
                result = self.run_wrapper("start")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("worker model index check failed", result.stdout)
                self.assertFalse(self.trace.exists())
                operations = self.operations.read_text(encoding="utf-8")
                self.assertNotIn("sha256sum", operations)
                self.assertNotIn("docker:", operations)
                self.assertEqual(receipt_path.read_bytes(), receipt)
                if dangling:
                    self.assertTrue(index.is_symlink())
                    self.assertEqual(index.readlink(), Path("missing-index.json"))
                else:
                    self.assertEqual(index.read_bytes(), b"stale index")
                index.unlink()

    def test_worker_index_query_failure_refuses_start_before_hash_or_docker_checks(self):
        self.prepare_settings()
        self.env["WORKER_INDEX_QUERY_FAIL"] = "1"
        result = self.run_wrapper("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker model index check failed", result.stdout)
        self.assertFalse(self.trace.exists())
        operations = self.operations.read_text(encoding="utf-8")
        self.assertNotIn("sha256sum", operations)
        self.assertNotIn("docker:", operations)

    def test_model_indexes_allow_recovery_commands_without_worker_or_docker_checks(self):
        digest = self.prepare_settings()
        (self.root / "view/model/model.safetensors.index.json").write_bytes(b"head index")
        worker_index = Path(self.env["WORKER_MODEL_VIEW"]) / "model/model.safetensors.index.json"
        worker_index.symlink_to("missing-index.json")
        for action in ("stop", "status", "verify"):
            with self.subTest(action=action):
                result = self.run_wrapper(action)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.operations.exists())
        expected_cache = self.root / "cache" / digest
        self.assertEqual(self.trace.read_text(encoding="utf-8").splitlines(),
                         [f"{action}:{expected_cache}" for action in ("stop", "status", "verify")])
        self.assertEqual((self.root / "view/model/model.safetensors.index.json").read_bytes(), b"head index")
        self.assertTrue(worker_index.is_symlink())

    def test_engram_indexes_allow_start(self):
        self.prepare_settings()
        indexes = []
        for view in (self.root / "view", Path(self.env["WORKER_MODEL_VIEW"])):
            (view / "engram").mkdir()
            index = view / "engram/model.safetensors.index.json"
            index.write_bytes(b"engram index")
            indexes.append(index)
        result = self.run_wrapper("start")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for index in indexes:
            self.assertEqual(index.read_bytes(), b"engram index")

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

    def test_docker_ps_failure_on_either_node_refuses_start(self):
        self.prepare_settings()
        for variable in ("DOCKER_PS_FAIL", "WORKER_DOCKER_PS_FAIL"):
            with self.subTest(node=variable):
                self.env[variable] = "1"
                result = self.run_wrapper("start")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot list running Docker containers", result.stderr)
                self.assertFalse(self.trace.exists())
                self.env[variable] = "0"

    def test_docker_inspect_failure_on_either_node_refuses_start(self):
        self.prepare_settings()
        for prefix in ("", "WORKER_"):
            for field in ("gpu", "runtime"):
                with self.subTest(node=prefix, field=field):
                    self.env[f"{prefix}GPU_ACTIVE"] = "1"
                    self.env[f"{prefix}DOCKER_INSPECT_FAIL"] = field
                    result = self.run_wrapper("start")
                    self.assertNotEqual(result.returncode, 0)
                    message = "GPU requests" if field == "gpu" else "runtime"
                    self.assertIn(f"cannot inspect {message}", result.stderr)
                    self.assertFalse(self.trace.exists())
                    self.env[f"{prefix}GPU_ACTIVE"] = "0"
                    self.env[f"{prefix}DOCKER_INSPECT_FAIL"] = ""

    def test_invalid_commit_refuses_start_and_build(self):
        for commit in ("main", TEST_COMMIT[:12], TEST_COMMIT.upper(), TEST_COMMIT + "0"):
            for action in ("start", "build"):
                with self.subTest(commit=commit, action=action):
                    self.prepare_settings(commit=commit)
                    result = self.run_wrapper(action)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("TF_COMMIT must be a full", result.stdout)
                    self.assertFalse(self.trace.exists())

    def test_invalid_image_tag_refuses_start_and_build(self):
        for image in ("tensorfold:latest", "tensorfold:" + "f" * 12,
                      f"tensorfold:{TEST_COMMIT[:11]}", f"tensorfold:{TEST_COMMIT}0", "tensorfold"):
            for action in ("start", "build"):
                with self.subTest(image=image, action=action):
                    self.prepare_settings(image=image)
                    result = self.run_wrapper(action)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("IMAGE tag must be", result.stdout)
                    self.assertFalse(self.trace.exists())

    def test_build_accepts_commit_prefix_tags(self):
        for length in (12, 24, 40):
            with self.subTest(length=length):
                self.prepare_settings(image=f"tensorfold:{TEST_COMMIT[:length]}")
                result = self.run_wrapper("build")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.trace.read_text(encoding="utf-8").splitlines(), ["build"] * 3)

    def test_invalid_pins_allow_recovery_commands_without_docker_queries(self):
        self.stub("docker", '#!/usr/bin/env bash\nprintf "docker\\n" >> "$TRACE"\nexit 9\n')
        expected = []
        for commit, image in (("edited", TEST_IMAGE), (TEST_COMMIT, "tensorfold:" + "f" * 12)):
            digest = self.prepare_settings(commit=commit, image=image)
            for action in ("stop", "status", "verify"):
                with self.subTest(commit=commit, image=image, action=action):
                    result = self.run_wrapper(action)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    expected.append(f"{action}:{self.root / 'cache' / digest}")
        self.assertEqual(self.trace.read_text(encoding="utf-8").splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
