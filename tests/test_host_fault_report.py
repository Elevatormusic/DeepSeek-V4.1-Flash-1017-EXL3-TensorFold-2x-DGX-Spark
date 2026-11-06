"""Check the read-only fault report with command stubs and temporary files."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from test_memory_checks import ENV_BASE, Tree, needs_bash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fault_report", ROOT / "scripts/host_fault_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class HostFaultReport(unittest.TestCase):
    def test_unavailable_is_not_no_events(self):
        for failure in (FileNotFoundError(), PermissionError(), subprocess.TimeoutExpired("journalctl", 30)):
            with self.subTest(failure=failure), mock.patch.object(report.subprocess, "run", side_effect=failure):
                self.assertEqual(report.kernel_events(-1)["status"], "unavailable")
        for code, stderr in ((1, "denied"), (0, "not seeing messages from other users")):
            with mock.patch.object(report.subprocess, "run", return_value=subprocess.CompletedProcess([], code, "", stderr)):
                self.assertEqual(report.kernel_events(-1)["status"], "unavailable")

    def test_successful_empty_journal_has_a_bounded_scope(self):
        with mock.patch.object(report.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ordinary event", "")) as run:
            result = report.kernel_events(-1)
        self.assertEqual(result["status"], "no matching events")
        self.assertIn("last 2000", result["scope"])
        self.assertIn("--no-hostname", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][3], "-1")

    def test_prior_boot_fault_classes_and_privacy(self):
        raw = "\n".join(["AER CmpltTO 10.100.80.1", "CMD_SYNC SMMU", "mlx5 fw error",
                         "NVRM /home/" + "private/file token=private", "other"])
        with mock.patch.object(report.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, raw, "")):
            result = report.kernel_events(-1)
        self.assertEqual(result["status"], "events")
        self.assertEqual(len(result["lines"]), 4)
        self.assertNotIn("private", "\n".join(result["lines"]).replace("private value removed", ""))
        self.assertNotIn("10.100.80.1", "\n".join(result["lines"]))

    def test_driver_firmware_and_thermal_facts_without_process_dumps(self):
        with tempfile.TemporaryDirectory() as directory:
            sysfs = Path(directory)
            firmware = sysfs / "class/infiniband/rocep1s0f1/fw_ver"
            firmware.parent.mkdir(parents=True)
            firmware.write_text("example-firmware")
            zone = sysfs / "class/thermal/thermal_zone0"
            zone.mkdir(parents=True)
            (zone / "type").write_text("cpu")
            (zone / "temp").write_text("42000")
            with mock.patch.object(report.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                result = report.collect(sysfs)
            self.assertEqual(result["rdma_firmware"]["rocep1s0f1"]["value"], "example-firmware")
            self.assertEqual(result["thermal_zones"]["thermal_zone0"]["millidegrees_c"]["value"], "42000")
            self.assertEqual({c.args[0][0] for c in run.call_args_list}, {"uname", "nvidia-smi", "journalctl"})
            self.assertTrue(all("shell" not in c.kwargs for c in run.call_args_list))


@needs_bash
class ReportNodes(unittest.TestCase):
    def tree(self):
        tree = Tree(self, "", ("common.sh", "host_fault_report.sh", "host_fault_report.py"))
        (tree.dir / ".env").write_text(ENV_BASE.format(d=tree.dir))
        tree.stub("python3", 'echo "collector stub"\n')
        return tree

    def test_both_nodes_use_the_fixed_collector_and_prompt_for_privacy_review(self):
        tree = self.tree()
        result = tree.run(["bash", "scripts/host_fault_report.sh"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("collector stub"), 2)
        self.assertIn("Review the complete output for private data", result.stdout)

    def test_worker_failure_is_explicit(self):
        tree = self.tree()
        tree.stub("ssh", "exit 255\n")
        result = tree.run(["bash", "scripts/host_fault_report.sh"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("worker report unavailable", result.stderr)
