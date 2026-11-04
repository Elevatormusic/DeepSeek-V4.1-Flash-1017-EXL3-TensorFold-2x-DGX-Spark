#!/usr/bin/env python3
"""Hermetic integration tests for start-display-kv.sh.

The copied launcher only sees fixture commands.  In particular, no test can
contact the configured worker or invoke Docker, sudo, or the DRM helper.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import hashlib


RECIPE = Path(__file__).resolve().parents[1]
LAUNCHER = RECIPE / "start-display-kv.sh"
VARIANT = "tensorfold-dsv41:display-kv-0123456789ab"
VARIANT_ID = "sha256:" + "a" * 64
OTHER_ID = "sha256:" + "b" * 64


DISPATCHER = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
name = Path(sys.argv[0]).name
# A symlink to a shebang script is passed as the first script argument.
if name == "stub.py" and args and Path(args[0]).parent.name == "bin":
    name = Path(args.pop(0)).name
elif name == "stub.py":
    name = "python3"
log = Path(os.environ["COMMAND_LOG"])
state = Path(os.environ["STUB_STATE"])

def record(message):
    with log.open("a") as output:
        output.write(message + "\n")

def setting(name, default=""):
    return os.environ.get(name, default)

def worker_command():
    values = list(args)
    while values and values[0] == "-n":
        values.pop(0)
    while values and values[0].startswith("-o"):
        values.pop(0)
        if values and not values[0].startswith("-"):
            values.pop(0)
    if values:
        values.pop(0)  # worker host
    return " ".join(values)

def worker_id():
    if setting("WORKER_FORCE_ABSENT") == "1":
        return ""
    return setting("WORKER_IMAGE_ID", "sha256:" + "a" * 64) if state.exists() else setting("WORKER_INITIAL_ID", "")

record(name + " " + " ".join(args))
if name == "flock":
    sys.exit(int(setting("FLOCK_FAIL", "0")))
if name == "id":
    print(setting("TEST_UID", "0"))
    sys.exit(0)
if name == "sudo":
    if args[:2] == ["-v"]:
        sys.exit(int(setting("SUDO_V_FAIL", "0")))
    sys.exit(int(setting("SUDO_FAIL", "0")))
if name == "docker":
    if args[:1] == ["ps"]:
        print(setting("HEAD_NAMES"))
    elif args[:2] == ["image", "inspect"]:
        image = args[2]
        if image == "tensorfold-dsv41:07aa02585f64":
            if setting("BASE_IMAGE_MISSING") == "1" and not state.with_suffix(".base").exists():
                sys.exit(1)
            print("base-id")
        elif any("org.tensorfold.display-kv.source" in arg for arg in args):
            print(setting("EXISTING_LABEL"))
        else:
            print(setting("VARIANT_ID", "sha256:" + "a" * 64))
    elif args[:1] == ["build"]:
        if "TF_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46" in args:
            state.with_suffix(".base").touch()
        sys.exit(int(setting("DOCKER_BUILD_FAIL", "0")))
    elif args[:1] == ["save"]:
        sys.stdout.write("fixture image stream")
    elif args[:1] == ["run"]:
        sys.exit(int(setting("HEAD_PROBE_FAIL", "0")))
    elif args[:1] == ["inspect"]:
        print(setting("HEAD_CONTAINER_IMAGE", ""))
    sys.exit(0)
if name == "ssh":
    command = worker_command()
    record("worker " + command)
    if setting("SSH_FAIL") == "all" or (setting("SSH_FAIL") and setting("SSH_FAIL") in command):
        sys.exit(1)
    if command == "docker load":
        sys.stdin.read()
        state.write_text("loaded")
    elif "docker ps" in command:
        print(setting("WORKER_NAMES"))
    elif "sudo -n true" in command:
        sys.exit(int(setting("WORKER_NOPASS_FAIL", "0")))
    elif "docker image inspect" in command:
        print(worker_id())
    elif "docker run" in command:
        sys.exit(int(setting("WORKER_PROBE_FAIL", "0")))
    elif "sudo -n bash -s --" in command:
        mode = command.rsplit(" ", 1)[1]
        record("drm:worker:" + mode)
        sys.exit(int(setting("WORKER_DRM_" + mode.upper() + "_FAIL", "0")))
    elif "docker inspect tfv41-r1" in command:
        # Model only the ownership comparison in the fixed cleanup command.
        if setting("START_WORKER_IMAGE", "sha256:" + "a" * 64) == setting("VARIANT_ID", "sha256:" + "a" * 64):
            record("worker docker stop -t 60 tfv41-r1")
    sys.exit(0)
if name == "python3":
    if setting("PREPARE_FAIL") == "1":
        sys.exit(1)
    bundle = Path(args[1])
    image = args[args.index("--image") + 1]
    bundle.mkdir()
    start = bundle / "start.sh"
    start.write_text("#!/usr/bin/env bash\n"
                     f"printf 'bundle-start IMAGE={image} PREFLIGHT=%s\\n' \"${{PREFLIGHT:-}}\" >> \"$COMMAND_LOG\"\n"
                     "[ \"${PREPARED_ACTIVE:-0}\" != 1 ] || exit 1\n"
                     "exit \"${START_FAIL:-0}\"\n")
    start.chmod(0o755)
    record("prepared image=" + image)
    sys.exit(0)
record("unexpected " + name)
sys.exit(0)
'''


class StartDisplayKvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "recipe"
        self.bin = self.root / "bin"
        self.log = self.root / "commands.log"
        self.state = self.root / "worker-loaded"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "experiments" / "display-kv" / "overlay").mkdir(parents=True)
        self.bin.mkdir()
        shutil.copy2(LAUNCHER, self.root / "start-display-kv.sh")
        shutil.copy2(RECIPE / "scripts/display_base.sh", self.root / "scripts/display_base.sh")
        (self.root / ".env").write_text("IMAGE=base:latest\nWORKER_SSH=fixture-worker\nTF_REPO=https://example.invalid/TensorFold.git\n")
        (self.root / "scripts" / "common.sh").write_text(textwrap.dedent("""\
            ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
            set -a
            . "$ROOT/.env"
            set +a
            log() { printf '%s\\n' "$*" >> "$COMMAND_LOG"; }
            wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
        """))
        (self.root / "scripts" / "display_drm.sh").write_text(
            "#!/usr/bin/env bash\n"
            "printf 'drm:head:%s\\n' \"$1\" >> \"$COMMAND_LOG\"\n"
            "failure=HEAD_DRM_${1^^}_FAIL\n"
            "[ \"${!failure:-0}\" = 0 ]\n")
        for name in ("Dockerfile", ".dockerignore", "apply.py", "probe.py", "overlay/stage.py"):
            (self.root / "experiments" / "display-kv" / name).write_text("fixture\n")
        self.write_saved_state()
        self.state.write_text("loaded")
        dispatcher = self.bin / "stub.py"
        dispatcher.write_text(DISPATCHER)
        dispatcher.chmod(0o755)
        for name in ("docker", "ssh", "sudo", "flock", "id", "nvidia-smi", "rmmod", "modprobe"):
            link = self.bin / name
            link.symlink_to(dispatcher)
        (self.bin / "python3").write_text(
            f"#!/usr/bin/env bash\nexec {sys.executable!s} {dispatcher!s} \"$@\"\n")
        (self.bin / "python3").chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, *args, **settings):
        env = os.environ | {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "COMMAND_LOG": str(self.log),
            "STUB_STATE": str(self.state),
            "VARIANT_ID": VARIANT_ID,
            "WORKER_IMAGE_ID": VARIANT_ID,
        } | {key: str(value) for key, value in settings.items()}
        return subprocess.run(["bash", str(self.root / "start-display-kv.sh"), *args],
                              cwd=self.root, text=True, capture_output=True, env=env, check=False)

    def commands(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def write_saved_state(self, variant=VARIANT, image_id=VARIANT_ID):
        (self.root / ".display-kv-image").write_text(f"{variant} {image_id}\n")

    def source_fingerprint(self):
        stage = self.root / "experiments" / "display-kv"
        paths = [stage / name for name in ("Dockerfile", ".dockerignore", "apply.py", "probe.py", "overlay/stage.py")]
        source = "base-id\n".encode()
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source += f"{digest}  {path}\n".encode()
        return hashlib.sha256(source).hexdigest()

    def assert_before(self, first, second, commands=None):
        commands = self.commands() if commands is None else commands
        self.assertLess(next(i for i, line in enumerate(commands) if first in line),
                        next(i for i, line in enumerate(commands) if second in line), commands)

    def test_default_uses_saved_image_prepares_probes_then_starts(self):
        result = self.invoke()
        commands = self.commands()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"using saved image {VARIANT}", "\n".join(commands))
        self.assertNotIn("docker build", "\n".join(commands))
        self.assertNotIn("docker save", "\n".join(commands))
        self.assertNotIn("worker docker load", "\n".join(commands))
        self.assertNotIn("docker image inspect base:latest", "\n".join(commands))
        self.assertIn("drm:worker:on", commands)
        self.assertIn("drm:head:on", commands)
        self.assertIn(f"prepared image={VARIANT}", "\n".join(commands))
        self.assertIn(f"bundle-start IMAGE={VARIANT}", "\n".join(commands))
        self.assert_before("drm:head:on", "docker run", commands)
        self.assert_before("worker docker run", "prepared image=", commands)
        self.assert_before("prepared image=", "bundle-start", commands)

    def test_build_only_never_changes_drm_probes_or_starts(self):
        self.state.unlink()
        result = self.invoke("--build-only")
        commands = "\n".join(self.commands())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker build", commands)
        self.assertIn("docker save", commands)
        self.assertIn("worker docker load", commands)
        self.assertNotIn("drm:", commands)
        self.assertNotIn("docker run", commands)
        self.assertNotIn("bundle-start", commands)

    def test_build_uses_the_exact_display_base_with_a_different_ordinary_image(self):
        result = self.invoke("--build-only", BASE_IMAGE_MISSING=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.commands())
        self.assertIn("TF_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46", commands)
        self.assertIn("BASE_IMAGE=tensorfold-dsv41:07aa02585f64", commands)
        self.assertNotIn("BASE_IMAGE=base:latest", commands)
        self.assertNotIn("drm:", commands)

    def test_explicit_build_copies_then_launches(self):
        self.state.unlink()
        result = self.invoke("--build")
        commands = self.commands()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker build", "\n".join(commands))
        self.assertIn("docker save", "\n".join(commands))
        self.assertIn("worker docker load", "\n".join(commands))
        self.assert_before("worker docker load", "drm:worker:on", commands)
        self.assertIn("bundle-start", "\n".join(commands))

    def test_default_reuses_saved_state_despite_changed_stage_or_missing_base(self):
        (self.root / "experiments" / "display-kv" / "apply.py").write_text("changed fixture\n")
        result = self.invoke(BASE_IMAGE_MISSING=1)
        commands = "\n".join(self.commands())
        self.assertEqual(result.returncode, 0, result.stderr)
        for forbidden in ("docker build", "docker save", "worker docker load", "docker image inspect base:latest"):
            self.assertNotIn(forbidden, commands)
        self.assertIn(f"using saved image {VARIANT}", commands)

    def test_default_refuses_missing_or_mismatched_worker_image_without_copying(self):
        for settings in ({"WORKER_FORCE_ABSENT": 1}, {"WORKER_IMAGE_ID": OTHER_ID}):
            with self.subTest(settings=settings):
                self.log.unlink(missing_ok=True)
                result = self.invoke(**settings)
                commands = "\n".join(self.commands())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("run ./start-display-kv.sh --build-only", commands)
                self.assertNotIn("docker build", commands)
                self.assertNotIn("docker save", commands)
                self.assertNotIn("worker docker load", commands)
                self.assertNotIn("drm:", commands)

    def test_missing_saved_state_refuses_before_build_or_drm(self):
        (self.root / ".display-kv-image").unlink()
        result = self.invoke()
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no saved display-KV image", commands)
        self.assertIn("--build", commands)
        self.assertNotIn("docker build", commands)
        self.assertNotIn("drm:", commands)

    def test_malformed_saved_state_refuses_before_build_or_drm(self):
        for variant, image_id in (("bad-tag", VARIANT_ID), (VARIANT, "not-a-digest")):
            with self.subTest(variant=variant, image_id=image_id):
                self.log.unlink(missing_ok=True)
                self.write_saved_state(variant, image_id)
                result = self.invoke()
                commands = "\n".join(self.commands())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid saved display-KV image", commands)
                self.assertIn("--build-only", commands)
                self.assertNotIn("docker build", commands)
                self.assertNotIn("docker save", commands)
                self.assertNotIn("drm:", commands)

    def test_build_only_saves_state_only_after_worker_id_matches(self):
        original = "previous state\n"
        state_file = self.root / ".display-kv-image"
        state_file.write_text(original)
        result = self.invoke("--build-only", WORKER_IMAGE_ID=OTHER_ID)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state_file.read_text(), original)
        self.assertIn("variant image IDs differ", "\n".join(self.commands()))

        self.log.unlink(missing_ok=True)
        result = self.invoke("--build-only")
        expected_variant = f"tensorfold-dsv41:display-kv-{self.source_fingerprint()[:12]}"
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state_file.read_text(), f"{expected_variant} {VARIANT_ID}\n")

    def test_repeated_build_only_reuses_cached_image_without_copying(self):
        self.state.unlink()
        first = self.invoke("--build-only")
        self.assertEqual(first.returncode, 0, first.stderr)
        first_build = next(line for line in self.commands() if line.startswith("docker build "))
        fingerprint = first_build.split("org.tensorfold.display-kv.source=", 1)[1].split()[0]
        self.log.unlink(missing_ok=True)
        result = self.invoke("--build-only", EXISTING_LABEL=fingerprint)
        commands = "\n".join(self.commands())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("docker build", commands)
        self.assertNotIn("docker save", commands)
        self.assertNotIn("worker docker load", commands)

    def test_check_uses_only_drm_checks(self):
        result = self.invoke("--check")
        commands = "\n".join(self.commands())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("drm:head:check", commands)
        self.assertIn("bash -s -- check", commands)
        for forbidden in ("docker ", "sudo ", "build", "docker run"):
            self.assertNotIn(forbidden, commands)

    def test_active_rank_refuses_before_build_or_sudo_even_with_preflight_disabled(self):
        for setting in ("HEAD_NAMES", "WORKER_NAMES"):
            with self.subTest(setting=setting):
                self.log.unlink(missing_ok=True)
                result = self.invoke(PREFLIGHT=0, **{setting: "tfv41-r0"})
                commands = "\n".join(self.commands())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("REFUSING", commands)
                self.assertNotIn("docker build", commands)
                self.assertNotIn("sudo ", commands)
                self.assertNotIn("drm:", commands)

    def test_docker_and_ssh_failures_fail_closed(self):
        build = self.invoke("--build", DOCKER_BUILD_FAIL=1)
        self.assertNotEqual(build.returncode, 0)
        self.assertNotIn("sudo ", "\n".join(self.commands()))
        self.log.unlink(missing_ok=True)
        ssh = self.invoke(SSH_FAIL="docker ps")
        self.assertNotEqual(ssh.returncode, 0)
        self.assertNotIn("docker build", "\n".join(self.commands()))

    def test_worker_without_passwordless_sudo_aborts_before_drm_changes(self):
        result = self.invoke(WORKER_NOPASS_FAIL=1)
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("passwordless sudo", commands)
        self.assertNotIn("drm:", commands)
        self.assertNotIn("sudo -v", commands)
        self.assertNotIn("bundle-start", commands)

    def test_head_with_passwordless_sudo_never_asks_for_a_password(self):
        result = self.invoke(TEST_UID=1000)
        commands = self.commands()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sudo -n true", commands)
        self.assertNotIn("sudo -v", commands)
        self.assertTrue(any(line.startswith("sudo -n bash ") and line.endswith(" on") for line in commands), commands)

    def test_head_without_passwordless_sudo_falls_back_to_one_password_prompt(self):
        result = self.invoke(TEST_UID=1000, SUDO_FAIL=1)
        commands = self.commands()
        self.assertIn("sudo -n true", commands)
        self.assertIn("sudo -v", commands)
        self.assert_before("sudo -n true", "sudo -v", commands)
        self.assertNotEqual(result.returncode, 0)

    def test_head_sudo_failure_changes_no_drm_setting(self):
        result = self.invoke(TEST_UID=1000, SUDO_FAIL=1, SUDO_V_FAIL=1)
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("head sudo authorization failed", commands)
        self.assertNotIn("drm:worker:on", commands)
        self.assertNotIn("bundle-start", commands)

    def test_prepare_and_probe_failures_restore_drm_without_starting(self):
        for settings in ({"PREPARE_FAIL": 1}, {"HEAD_PROBE_FAIL": 1}, {"WORKER_PROBE_FAIL": 1}):
            with self.subTest(settings=settings):
                self.log.unlink(missing_ok=True)
                result = self.invoke(**settings)
                commands = "\n".join(self.commands())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("drm:worker:off", commands)
                self.assertIn("drm:head:off", commands)
                self.assertNotIn("bundle-start", commands)

    def test_mismatched_image_ids_refuse_before_drm(self):
        result = self.invoke(WORKER_IMAGE_ID=OTHER_ID)
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("saved image missing or changed", commands)
        self.assertIn("--build-only", commands)
        self.assertNotIn("docker save", commands)
        self.assertNotIn("worker docker load", commands)
        self.assertNotIn("drm:", commands)
        self.assertNotIn("bundle-start", commands)

    def test_startup_failure_stops_only_owned_ranks_and_restores_drm(self):
        result = self.invoke(START_FAIL=1, HEAD_CONTAINER_IMAGE=VARIANT_ID, START_WORKER_IMAGE=VARIANT_ID)
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("docker stop -t 60 tfv41-r0", commands)
        self.assertIn("worker docker stop -t 60 tfv41-r1", commands)
        self.assertIn("drm:worker:off", commands)
        self.assertIn("drm:head:off", commands)
        self.assertIn("no ordinary-memory fallback", commands)
        self.log.unlink(missing_ok=True)
        result = self.invoke(START_FAIL=1, HEAD_CONTAINER_IMAGE=OTHER_ID, START_WORKER_IMAGE=OTHER_ID)
        self.assertNotEqual(result.returncode, 0)
        commands = "\n".join(self.commands())
        self.assertNotIn("docker stop -t 60 tfv41-r0", commands)
        self.assertNotIn("worker docker stop -t 60 tfv41-r1", commands)

    def test_head_drm_on_failure_restores_both_nodes(self):
        result = self.invoke(HEAD_DRM_ON_FAIL=1)
        commands = self.commands()
        self.assertNotEqual(result.returncode, 0)
        self.assert_before("drm:worker:on", "drm:head:on", commands)
        self.assertIn("drm:worker:off", commands)
        self.assert_before("drm:head:on", "drm:worker:off", commands)
        self.assertIn("drm:head:off", commands)
        self.assertNotIn("docker run", "\n".join(commands))

    def test_failed_automatic_restore_reports_remediation_after_both_attempts(self):
        result = self.invoke(HEAD_PROBE_FAIL=1, WORKER_DRM_OFF_FAIL=1)
        commands = self.commands()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("drm:worker:off", commands)
        self.assertIn("drm:head:off", commands)
        self.assertIn("automatic DRM restore was refused or failed", "\n".join(commands))
        self.assertIn("./start-display-kv.sh --restore-drm", "\n".join(commands))

    def test_restore_drm_attempts_both_nodes_without_cutover(self):
        result = self.invoke("--restore-drm", WORKER_DRM_OFF_FAIL=1)
        commands = "\n".join(self.commands())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("drm:worker:off", commands)
        self.assertIn("drm:head:off", commands)
        self.assertNotIn("docker build", commands)
        self.assertNotIn("docker run", commands)
        self.assertNotIn("bundle-start", commands)


if __name__ == "__main__":
    unittest.main()
