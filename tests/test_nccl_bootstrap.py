"""Exercise only the preflight bootstrap fragment with fixture commands."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NcclBootstrapTests(unittest.TestCase):
    def probe(self, status=0, timed_out=False, gate=None):
        source = (ROOT / "scripts/preflight.sh").read_text()
        start = source.index("if bootstrap=")
        end = source.index("\nfi\n", start) + len("\nfi\n")
        if gate is not None:                     # the whole check, with the opt-in line around it
            start = source.index('if [ "${PREFLIGHT_NCCL_BOOTSTRAP')
            end = source.index("\nfi\nfi\n", start) + len("\nfi\nfi\n")
        fragment = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "calls"
            docker = root / "docker"
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALL_LOG"\n'
                              'printf "fixture bootstrap\\nsecond line\\n"\nexit "$PROBE_STATUS"\n')
            docker.chmod(0o755)
            timeout = root / "timeout"
            timeout.write_text('#!/bin/sh\n[ "$1" = --kill-after=5s ] && [ "$2" = 20s ] || exit 99\n'
                               '[ "$TIMED_OUT" != 1 ] || exit 124\nshift 2\nexec "$@"\n')
            timeout.chmod(0o755)
            env = os.environ | {
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "CALL_LOG": str(log), "PROBE_STATUS": str(status),
                "TIMED_OUT": "1" if timed_out else "0", "IMAGE": "fixture:image", "PF_IMAGE": "fixture:image",
                "NCCL_SOCKET_IFNAME": "fixture-net",
            }
            env.pop("PREFLIGHT_NCCL_BOOTSTRAP", None)
            if gate:
                env["PREFLIGHT_NCCL_BOOTSTRAP"] = gate
            # Use the same unquoted heredoc expansion as check_node. Only the
            # extracted fragment runs. It can see no real Docker executable.
            result = subprocess.run(["bash", "-s"], input="bash -s <<EOF\n" + fragment + "EOF\n",
                                    env=env, text=True, capture_output=True, check=False)
            return result, log.read_text().splitlines() if log.exists() else []

    def test_success_uses_socket_interface_without_gpu_access(self):
        result, args = self.probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK NCCL socket bootstrap (fixture-net)", result.stdout)
        self.assertIn("NCCL_SOCKET_IFNAME=fixture-net", args)
        self.assertIn("fixture:image", args)
        self.assertIn("--network", args)
        self.assertNotIn("--gpus", args)
        self.assertIn("--init", args)             # signals reach python, which is not PID 1 in the container
        self.assertNotIn("exec", args)
        self.assertIn("ncclGetUniqueId", args[-1])

    def test_the_check_is_off_unless_the_switch_is_1(self):
        for gate in ("", "0", "yes"):            # "" means the variable is not set
            with self.subTest(gate=gate):
                result, args = self.probe(gate=gate)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(args, [])           # no container is started

    def test_the_check_runs_when_the_switch_is_1(self):
        result, args = self.probe(gate="1")
        self.assertIn("OK NCCL socket bootstrap (fixture-net)", result.stdout)
        self.assertIn("fixture:image", args)

    def test_the_switch_is_listed_in_the_example_and_off(self):
        example = (ROOT / ".env.example").read_text()
        self.assertIn("#PREFLIGHT_NCCL_BOOTSTRAP=1", example)

    def test_failure_is_one_diagnostic_line(self):
        result, _ = self.probe(status=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAIL NCCL socket bootstrap (fixture-net)", result.stdout)
        self.assertIn("fixture bootstrap second line", result.stdout)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn("check NCCL_SOCKET_IFNAME in .env", result.stdout)

    def test_timeout_is_a_failure(self):
        result, args = self.probe(timed_out=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAIL NCCL socket bootstrap", result.stdout)
        self.assertEqual(args, [])

    def test_both_nodes_and_existing_checks_remain(self):
        source = (ROOT / "scripts/preflight.sh").read_text()
        self.assertIn('check_node head ""', source)
        self.assertIn("check_node worker wrun", source)
        self.assertIn("elif [ -z", source)
        self.assertIn('RoCE MTU', source)
        self.assertIn('older than the tested', source)


if __name__ == "__main__":
    unittest.main()
