"""Offline tests of the memory checks (stdlib only; no server, no GPU, no ssh, no Docker).

They run the real scripts in a temporary folder, with stub commands first on PATH:
- scripts/preflight.sh: the MemAvailable need of the head (111 GiB, 110 with VISION=0) and of the worker (109 GiB);
- margin_check in scripts/common.sh, and its call in status.sh and start.sh: the warning under 2.5 GiB;
- scripts/oom_report.sh: the sections of the report, the 50-line limit, and no secret in the file.

Run: python3 -m unittest discover -s tests -p 'test_*.py'   (or tests/run_offline.sh)
"""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIB_KB = 1048576

# Fake secrets for the report test. They are built here, so that no secret-like text is in the source, and they
# have low entropy, so that a secret scanner does not flag them.
FAKE_HUB = "hf" + "_" + "Qx" * 18
FAKE_SK = "sk" + "-" + "Zm" * 14
FAKE_AUTH = "Tk" * 8
FAKE_QS = "Pw" * 6
FAKE_OTHER = "Lm" * 8
FAKE_ARG_A = "Ak" * 8
FAKE_ARG_B = "Gt" * 8
FAKE_ARG_C = "Ps" * 6
FAKE_WORD_D = "Wd" * 5
FAKE_WORD_E = "Xe" * 5
FAKE_WORD_F = "Yf" * 5
FAKE_GITLAB = "gl" + "pat-" + "Gq" * 10                     # a GitLab token form
FAKE_AWS_ID = "AK" + "IA" + "QZ7" * 5 + "Q"               # an AWS key ID form (4 + 16 characters)
FAKE_LONG = "Ab3" * 12                                    # a key with no known prefix (36 characters, mixed)

needs_bash = unittest.skipUnless(shutil.which("bash") and os.name == "posix", "needs bash on Linux or macOS")

# ssh stub: drops the ssh options and the host, then runs the command here, as the remote shell does. "bash -s" reads
# the script on stdin.
SSH_STUB = r"""#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    -n) shift ;;
    -o) shift 2 ;;
    *) break ;;
  esac
done
shift
exec sh -c "$*"
"""


def fill(text: str) -> str:
    """Puts the fake secrets into a fixture text."""
    for name, value in (("FAKE_HUB", FAKE_HUB), ("FAKE_AUTH", FAKE_AUTH), ("FAKE_QS", FAKE_QS),
                        ("FAKE_OTHER", FAKE_OTHER), ("FAKE_ARG_A", FAKE_ARG_A),
                        ("FAKE_ARG_B", FAKE_ARG_B), ("FAKE_ARG_C", FAKE_ARG_C), ("FAKE_WORD_D", FAKE_WORD_D),
                        ("FAKE_WORD_E", FAKE_WORD_E), ("FAKE_WORD_F", FAKE_WORD_F),
                        ("FAKE_GITLAB", FAKE_GITLAB), ("FAKE_AWS_ID", FAKE_AWS_ID), ("FAKE_LONG", FAKE_LONG)):
        text = text.replace(f"@{name}@", value)
    return text


def write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Tree:
    """A copy of the scripts in a temporary folder, a fixture .env, and a folder of stub commands."""

    def __init__(self, testcase: unittest.TestCase, env_text: str, scripts: tuple[str, ...], top: tuple[str, ...] = ()):
        self.dir = Path(tempfile.mkdtemp(prefix="tfv41-test-"))
        testcase.addCleanup(shutil.rmtree, self.dir, True)
        for name in scripts:
            (self.dir / "scripts").mkdir(exist_ok=True)
            shutil.copy(ROOT / "scripts" / name, self.dir / "scripts" / name)
        for name in top:
            shutil.copy(ROOT / name, self.dir / name)
        (self.dir / ".env").write_text(env_text, encoding="utf-8")
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        write_exec(self.bin / "ssh", SSH_STUB)

    def stub(self, name: str, text: str) -> None:
        write_exec(self.bin / name, "#!/bin/sh\n" + text)

    def meminfo(self, avail_kb: int) -> Path:
        path = self.dir / "meminfo"
        path.write_text(f"MemTotal:       127598832 kB\nMemAvailable:   {avail_kb} kB\n", encoding="utf-8")
        return path

    def run(self, argv, extra_env=None) -> subprocess.CompletedProcess:
        env = {"PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}", "HOME": str(self.dir / "home"), "LC_ALL": "C"}
        env.update(extra_env or {})
        return subprocess.run(argv, cwd=self.dir, env=env, capture_output=True, text=True, timeout=60, check=False)


ENV_BASE = """ENGINE=python
HEAD_IP=10.100.80.1
WORKER_IP=10.100.80.2
WORKER_SSH=10.100.80.2
RDMA_DEVICES=
NCCL_IB_HCA=
NCCL_SOCKET_IFNAME=eth0
GID_INDEX=-1
MODEL_VIEW={d}/model-view
CACHE_DIR={d}/cache
IMAGE=tensorfold-dsv41:test
BIND_HOST=127.0.0.1
PORT=8888
SERVED_NAME=DeepSeek-V4.1-Flash-1017
CGROUP_PARENT=
"""


@needs_bash
class PreflightMemory(unittest.TestCase):
    """preflight.sh runs the node checks with the real shell on this host. Only /proc/meminfo is swapped for a file."""

    def preflight(self, avail_kb: int, vision: str | None = None) -> list[str]:
        t = Tree(self, "", ("common.sh", "preflight.sh"))
        (t.dir / ".env").write_text(ENV_BASE.format(d=t.dir) + (f"VISION={vision}\n" if vision is not None else ""),
                                    encoding="utf-8")
        meminfo = t.meminfo(avail_kb)
        text = (t.dir / "scripts" / "preflight.sh").read_text(encoding="utf-8").replace("/proc/meminfo", str(meminfo))
        (t.dir / "scripts" / "preflight.sh").write_text(text, encoding="utf-8")
        for name in ("ping", "ss", "docker", "systemctl", "ibv_devinfo"):
            t.stub(name, "exit 1\n" if name != "ping" else "exit 0\n")
        r = t.run(["bash", str(t.dir / "scripts" / "preflight.sh")])
        return [line for line in r.stdout.splitlines() if "memory available" in line]

    def test_head_needs_111_and_worker_109(self):
        lines = self.preflight(110 * GIB_KB)                     # the same fixture file serves both nodes
        self.assertEqual(len(lines), 2, lines)
        head, worker = lines
        self.assertTrue(head.startswith("[FAIL] head:"), head)
        self.assertIn("only 110 GiB memory available, need 111", head)
        self.assertIn("VISION=0", head)
        self.assertIn("scripts/host_tuning.sh", head)
        self.assertIn("docs/MEMORY.md", head)
        self.assertEqual(worker, "[ ok ] worker: 110 GiB memory available (need 109)")

    def test_worker_fails_below_109(self):
        head, worker = self.preflight(108 * GIB_KB)
        self.assertIn("need 111", head)
        self.assertTrue(worker.startswith("[FAIL] worker:"), worker)
        self.assertIn("only 108 GiB memory available, need 109", worker)
        self.assertNotIn("VISION=0", worker)                     # the worker has no vision tower

    def test_head_passes_at_111(self):
        head, _ = self.preflight(111 * GIB_KB)
        self.assertEqual(head, "[ ok ] head: 111 GiB memory available (need 111)")

    def test_part_of_a_gib_does_not_round_up(self):
        head, _ = self.preflight(111 * GIB_KB - 1)
        self.assertTrue(head.startswith("[FAIL] head:"), head)

    def test_vision_off_lowers_the_head_need_by_1_gib(self):
        head, _ = self.preflight(110 * GIB_KB, vision="0")
        self.assertEqual(head, "[ ok ] head: 110 GiB memory available (need 110)")

    def test_vision_off_message_does_not_repeat_the_vision_hint(self):
        head, _ = self.preflight(109 * GIB_KB, vision="0")
        self.assertTrue(head.startswith("[FAIL] head:"), head)
        self.assertNotIn("VISION=0", head)


@needs_bash
class MarginCheck(unittest.TestCase):
    """margin_check in common.sh warns when the head has under 2.5 GiB MemAvailable after the start."""

    def margin(self, avail_kb: int | None, env_extra: str = "") -> str:
        t = Tree(self, ENV_BASE.format(d="/x") + env_extra, ("common.sh",))
        meminfo = t.meminfo(avail_kb) if avail_kb is not None else t.dir / "missing"
        r = t.run(["bash", "-c", '. scripts/common.sh; margin_check'], {"MEMINFO_FILE": str(meminfo)})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_ok_at_the_gate_idle_value(self):
        self.assertEqual(self.margin(2941 * 1024), "[ ok ] head: MemAvailable is 2.87 GiB (earlyoom stops the engine at 1.0 GiB)")

    def test_ok_at_exactly_2_5_gib(self):
        self.assertTrue(self.margin(2621440).startswith("[ ok ]"))

    def test_warns_under_2_5_gib(self):
        out = self.margin(2621439)
        self.assertTrue(out.startswith("[warn] head: MemAvailable is 2.50 GiB, under 2.5 GiB"), out)
        for advice in ("VISION=0", "PREFILL_CHUNK=1024", "PARALLEL=2", "scripts/host_tuning.sh", "docs/MEMORY.md"):
            self.assertIn(advice, out)

    def test_no_vision_advice_when_vision_is_already_off(self):
        out = self.margin(1500 * 1024, "VISION=0\n")
        self.assertTrue(out.startswith("[warn]"), out)
        self.assertNotIn("VISION=0", out)
        self.assertIn("PREFILL_CHUNK=1024", out)

    def test_unreadable_meminfo_warns_and_succeeds(self):
        self.assertEqual(self.margin(None), "[warn] head: cannot read MemAvailable")

    def test_status_sh_prints_the_margin_when_the_api_answers(self):
        t = Tree(self, ENV_BASE.format(d="/x"), ("common.sh",), ("status.sh",))
        t.stub("docker", "exit 0\n")
        t.stub("curl", 'echo \'{"data":[{"id":"DeepSeek-V4.1-Flash-1017"}]}\'\n')
        r = t.run(["bash", "status.sh"], {"MEMINFO_FILE": str(t.meminfo(2000 * 1024))})
        self.assertIn("API: serving DeepSeek-V4.1-Flash-1017", r.stdout)
        self.assertIn("[warn] head: MemAvailable is 1.95 GiB", r.stdout)

    def test_status_sh_stays_quiet_when_the_api_is_down(self):
        t = Tree(self, ENV_BASE.format(d="/x"), ("common.sh",), ("status.sh",))
        t.stub("docker", "exit 0\n")
        t.stub("curl", "exit 22\n")
        r = t.run(["bash", "status.sh"], {"MEMINFO_FILE": str(t.meminfo(2000 * 1024))})
        self.assertIn("API: not serving", r.stdout)
        self.assertNotIn("MemAvailable", r.stdout)

    def test_start_sh_checks_the_margin_after_the_api_answers(self):
        lines = (ROOT / "start.sh").read_text(encoding="utf-8").splitlines()
        serving = [i for i, line in enumerate(lines) if "healthy &&" in line and "margin_check" in line]
        self.assertEqual(len(serving), 1, "start.sh must call margin_check where the API first answers")
        self.assertIn("exit 0", lines[serving[0]])
        self.assertLess(lines[serving[0]].index("margin_check"), lines[serving[0]].index("exit 0"))


DOCKER_STUB = r"""case "$1 $2" in
  "logs tfv41-r0")
    i=0
    while [ $i -lt 80 ]; do echo "[tensorfold] line $i host memory is low"; i=$((i + 1)); done
    echo "terminate called without an active exception"
    echo "memory Authorization: Bearer @FAKE_AUTH@"
    echo "memory api_key=@FAKE_QS@"
    echo "memory loaded /home/ubuntu/models/x with @FAKE_HUB@"
    echo "an unrelated line about decode speed"
    ;;
  "logs tfv41-r1")
    echo "NCCL error 4: invalid argument in ncclAllGather"
    echo "rank 1 memory line"
    echo "memory args: --api-key @FAKE_ARG_A@ --hf-token '@FAKE_ARG_B@' --password  @FAKE_ARG_C@"
    echo "memory args: --max-num-batched-tokens 8192 --port 8888 tokenizer ok"
    echo "memory quoted: --password '@FAKE_WORD_D@ @FAKE_WORD_E@ @FAKE_WORD_F@' --port 8888"
    echo 'memory quoted: --secret "@FAKE_WORD_E@ @FAKE_WORD_D@" token="@FAKE_WORD_F@ @FAKE_WORD_E@" end'
    echo "memory quoted: api_key: '@FAKE_WORD_D@ @FAKE_WORD_F@"
    echo 'memory json: {"password": "@FAKE_WORD_D@\" @FAKE_WORD_E@", "port": 8888}'
    echo 'memory odd: --token "@FAKE_WORD_F@"@FAKE_WORD_D@ @FAKE_WORD_E@'
    echo 'memory payload={\"token\":\"@FAKE_WORD_E@\"}'
    echo "memory header Authorization: ApiKey @FAKE_WORD_F@"
    echo "memory header X-Custom: Token @FAKE_WORD_D@"
    echo "memory proxy: Basic @FAKE_WORD_E@== sent"
    echo "memory basic mode on"
    echo "memory args2: --key @FAKE_WORD_D@ --access-key @FAKE_WORD_E@"
    echo "memory store: secret_key=@FAKE_WORD_F@"
    echo "memory keys loaded 5, monkey test ok"
    echo 'memory config={"privateKey":"@FAKE_WORD_D@"}'
    echo "memory aws accessKeyId=@FAKE_WORD_E@ apiToken: @FAKE_WORD_F@"
    echo "memory KeyError in cacheKeys ok"
    echo 'memory config2={"auth":"@FAKE_WORD_D@"}'
    echo "memory args3: --auth @FAKE_WORD_E@"
    echo "memory author and authority ok"
    echo 'memory config3={"authentication":"@FAKE_WORD_D@"}'
    echo "memory args4: --authentication @FAKE_WORD_E@"
    echo "memory args5: --authn @FAKE_WORD_F@"
    echo "memory store: userAuthenticator=@FAKE_WORD_D@"
    echo "memory request failed for @FAKE_GITLAB@ at push"
    echo "memory s3 client id @FAKE_AWS_ID@ region us-east-1"
    echo "memory upstream sent @FAKE_LONG@ then closed"
    echo "memory digest 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 tensorfold_deepseek_v41_graph_capture_window_4 ok"
    ;;
  "image inspect") echo "id sha256:0123456789abcdef, created 2026-10-21T00:00:00Z" ;;
  "ps -a") echo "tfv41-r0 | tensorfold-dsv41:test | Up 3 hours" ;;
esac
"""

JOURNAL_STUB = r"""echo "2026-10-21T10:00:00+0000 spark-test1 earlyoom[10]: mem avail: 2941 of 124546 MiB (2%), swap free: 0 of 0 MiB (0%)"
echo "2026-10-21T10:05:00+0000 spark-test1 earlyoom[10]: sending SIGTERM to process 42 uid 0 \"tensorfold\": oom_score 900"
echo "2026-10-21T10:05:01+0000 spark-test1 other[11]: not an earlyoom line"
"""


@needs_bash
class OomReport(unittest.TestCase):
    ENV = ENV_BASE.format(d="/x") + """PARALLEL=4
CONTEXT=1048576
SPILL_GIB=64
PREFILL_CHUNK=1024
VISION=0
MEM_FLOOR_GIB=1.5
TF_COMMIT=730d41abd8f4
HF_TOKEN=@FAKE_HUB@
SOME_SECRET=@FAKE_OTHER@
API_KEY_FILE={d}/keys
"""

    def prepared(self) -> "Tree":
        t = Tree(self, "", ("common.sh", "oom_report.sh"))
        (t.dir / ".env").write_text(fill(self.ENV.format(d=t.dir)), encoding="utf-8")
        (t.dir / "keys").write_text(f"main: {FAKE_SK}" + chr(10), encoding="utf-8")
        t.stub("docker", fill(DOCKER_STUB))
        t.stub("journalctl", JOURNAL_STUB)
        t.stub("nvidia-smi", "echo 580.178.04\n")
        t.stub("hostname", "echo spark-test1\n")
        t.stub("dmesg", "echo '[  10.0] NVRM: Failed memdescAlloc'\n")
        t.stub("swapon", "exit 0\n")
        t.stub("free", "echo 'Mem: 124546 100 100'\n")
        t.stub("systemctl", "echo active\n")
        return t

    def report(self, output_arg: bool = False) -> str:
        t = self.prepared()
        args = ["bash", "scripts/oom_report.sh"] + ([str(t.dir / "out" / "r.txt")] if output_arg else [])
        r = t.run(args, {"HOME": "/home/ubuntu"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        path = t.dir / "out" / "r.txt" if output_arg else next((t.dir / "logs").glob("oom-report-*.txt"))
        self.assertIn(str(path), r.stdout)
        return path.read_text(encoding="utf-8")

    def test_default_path_is_in_logs(self):
        self.assertIn("OOM report", self.report())

    def test_output_path_argument(self):
        self.assertIn("OOM report", self.report(output_arg=True))

    def test_unwritable_output_fails(self):
        t = self.prepared()
        for target, message in ((t.dir / "out" / "folder", "is a folder"),
                                (t.dir / "ro" / "r.txt", "cannot write the report")):
            with self.subTest(target=target.name):
                if target.name == "folder":
                    target.mkdir(parents=True)
                else:
                    target.parent.mkdir()
                    target.parent.chmod(0o500)
                    self.addCleanup(target.parent.chmod, 0o700)
                    if os.access(target.parent, os.W_OK):      # (root can write anyway)
                        continue
                r = t.run(["bash", "scripts/oom_report.sh", str(target)], {"HOME": "/home/ubuntu"})
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn(message, r.stderr)
                self.assertNotIn("report written", r.stdout)

    def test_report_needs_no_temporary_file(self):
        t = self.prepared()
        t.stub("mktemp", "exit 1\n")                 # no temporary file can be made
        r = t.run(["bash", "scripts/oom_report.sh", str(t.dir / "out" / "r.txt")], {"HOME": "/home/ubuntu"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        text = (t.dir / "out" / "r.txt").read_text(encoding="utf-8")
        self.assertIn("terminate called without an active exception", text)
        self.assertIn("NCCL error 4", text)

    def test_report_holds_the_facts_of_both_nodes(self):
        text = self.report()
        self.assertEqual(text.count("=== head node"), 1)
        self.assertEqual(text.count("=== worker node"), 1)
        for fact in (
            "MemAvailable:", "AnonPages:", "Cached:", "Shmem:",
            "vm.min_free_kbytes =", "vm.watermark_scale_factor =", "sum of the zone low watermarks:",
            "earlyoom journal, last 24 h", "mem avail: 2941 of 124546 MiB", "sending SIGTERM",
            "driver 580.178.04", "NVRM: Failed memdescAlloc", "free -m",
            "IMAGE=tensorfold-dsv41:test", "TF_COMMIT=730d41abd8f4", "id sha256:0123456789abcdef",
            "PARALLEL=4", "CONTEXT=1048576", "SPILL_GIB=64", "PREFILL_CHUNK=1024", "VISION=0",
            "API access file: set (the file is not read)", "NCCL error 4",
        ):
            with self.subTest(fact=fact):
                self.assertIn(fact, text)
        self.assertEqual(text.count("driver 580.178.04"), 2)            # once for each node
        self.assertNotIn("not an earlyoom line", text)

    def test_engine_log_is_limited_to_50_lines_that_mention_the_topic(self):
        text = self.report()
        rank0 = text.split("--- rank 0 engine log")[1].split("=== worker node")[0].splitlines()[1:]
        rank0 = [line for line in rank0 if line.strip()]
        self.assertEqual(len(rank0), 50)
        self.assertTrue(any("terminate called without an active exception" in line for line in rank0))
        self.assertFalse(any("decode speed" in line for line in rank0))
        self.assertIn("[tensorfold] line 79 host memory is low", rank0)

    def test_no_secret_in_the_report(self):
        text = self.report()
        for fake in (FAKE_HUB, FAKE_SK, FAKE_AUTH, FAKE_QS, FAKE_OTHER, FAKE_ARG_A, FAKE_ARG_B,
                     FAKE_ARG_C, FAKE_WORD_D, FAKE_WORD_E, FAKE_WORD_F, FAKE_GITLAB, FAKE_AWS_ID, FAKE_LONG,
                     "SOME_SECRET", "HF_TOKEN",
                     "/home/ubuntu", "spark-test1"):
            with self.subTest(fake=fake):
                self.assertNotIn(fake, text)
        self.assertIn("[removed]", text)
        # each line keeps its text up to the first word that names a credential
        self.assertIn("memory args: --[removed]" + chr(10), text)
        self.assertIn("memory payload={\\\"[removed]" + chr(10), text)
        self.assertIn("memory header [removed]" + chr(10), text)
        self.assertIn("memory header X-Custom: [removed]" + chr(10), text)
        self.assertIn("memory proxy: [removed]" + chr(10), text)
        self.assertIn("memory basic mode on", text)
        self.assertIn("memory args2: --[removed]" + chr(10), text)
        self.assertIn("memory keys loaded 5, monkey test ok", text)
        self.assertIn('memory config={"private[removed]' + chr(10), text)
        self.assertIn("memory aws access[removed]" + chr(10), text)
        self.assertIn("memory KeyError in cacheKeys ok", text)
        self.assertIn('memory config2={"[removed]' + chr(10), text)
        self.assertIn("memory args3: --[removed]" + chr(10), text)
        self.assertIn("memory author and authority ok", text)
        # raw keys with no name before them: known prefixes and long mixed words
        self.assertIn("memory request failed for [removed] at push" + chr(10), text)
        self.assertIn("memory s3 client id [removed] region us-east-1" + chr(10), text)
        self.assertIn("memory upstream sent [removed] then closed" + chr(10), text)
        # a hex hash and a long lower-case name stay
        self.assertIn("memory digest 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 "
                      "tensorfold_deepseek_v41_graph_capture_window_4 ok", text)
        self.assertIn('memory config3={"[removed]' + chr(10), text)
        self.assertIn("memory args4: --[removed]" + chr(10), text)
        self.assertIn("memory args5: --[removed]" + chr(10), text)
        self.assertIn("memory store: user[removed]" + chr(10), text)
        # a flag that only contains "token" in a longer word keeps its value
        self.assertIn("--max-num-batched-tokens 8192 --port 8888 tokenizer ok", text)
        # a quoted value with spaces goes as a whole, and the text after it stays
        # an escaped quote does not end a JSON value, and a quote that a letter follows does not end a value
        self.assertIn("memory odd: --[removed]" + chr(10), text)

    def test_missing_container_falls_back_to_the_saved_log(self):
        t = Tree(self, "", ("common.sh", "oom_report.sh"))
        (t.dir / ".env").write_text(fill(self.ENV.format(d=t.dir)), encoding="utf-8")
        t.stub("docker", 'case "$1" in logs) exit 1 ;; esac\n')
        for name in ("journalctl", "nvidia-smi", "dmesg", "swapon", "free", "systemctl"):
            t.stub(name, "exit 0\n")
        (t.dir / "logs").mkdir()
        (t.dir / "logs" / "r0-20261021T000000Z.log").write_text("old\nearlyoom stopped rank 0 memory\n", encoding="utf-8")
        (t.dir / "logs" / "r1-20261021T000000Z.log").write_text("NCCL error 4\n", encoding="utf-8")
        r = t.run(["bash", "scripts/oom_report.sh"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        text = next((t.dir / "logs").glob("oom-report-*.txt")).read_text(encoding="utf-8")
        self.assertIn("saved log r0-20261021T000000Z.log", text)
        self.assertIn("earlyoom stopped rank 0 memory", text)
        self.assertNotIn("\nold\n", text)


if __name__ == "__main__":
    unittest.main()
