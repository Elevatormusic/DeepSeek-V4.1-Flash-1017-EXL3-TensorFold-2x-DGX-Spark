"""Offline tests of the Zig engine scripts (stdlib only; no server, no GPU, no ssh, no Docker).

They run the real scripts in a temporary folder, with stub commands first on PATH (docker, ssh, hf, rsync). The ssh stub
runs the command here and marks it as the worker. A small HTTP server on 127.0.0.1 plays the engine API.
- scripts/zig_gid.sh: the RoCE v2 GID with an IPv4 address;
- scripts/zig_start.sh: the docker run of each rank, the order, the first completion, the exit codes, the settings;
- start.sh: the ENGINE switch (zig is the default, python is the rollback);
- scripts/zig_supervise.sh: the exit-code rules (0, 75, 137, other) and the limit on restarts;
- scripts/zig_inputs.sh and scripts/zig_build_image.sh: the guards and the commands;
- scripts/preflight.sh with ENGINE=zig;
- the files: the hash list of the 2.8 bpw pack, the Dockerfile, .env.example.

Run: python3 -m unittest discover -s tests -p 'test_*.py'   (or tests/run_offline.sh)
"""
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from test_memory_checks import Tree, write_exec

ROOT = Path(__file__).resolve().parent.parent
needs_bash = unittest.skipUnless(shutil.which("bash") and os.name == "posix", "needs bash on Linux or macOS")
needs_curl = unittest.skipUnless(shutil.which("curl"), "needs curl")
GIB_KB = 1048576

# The ssh stub drops the options and the host, marks the command as the worker's, and runs it here.
SSH_WORKER = r"""#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    -n) shift ;;
    -o) shift 2 ;;
    *) break ;;
  esac
done
shift
SIDE=worker exec bash -c "$*"
"""

# The docker stub writes one line for each call to $CALLS (SIDE tells the node). "inspect" reads the state of a container
# from $STATE/<side>-<name> ("true 0" when there is no file). "ps" prints $STATE/<side>-ps. "image inspect" fails when
# $STATE/no-image exists, prints the label tf.ref, or prints one image ID (two IDs when $STATE/diff-ids exists). While
# $STATE/base-missing exists, the base image is missing until a "docker build ... -t <base>" call.
DOCKER = r"""
SIDE=${SIDE:-head}
echo "$SIDE docker $*" >> "$CALLS"
case "$1" in
  inspect)
    for last in "$@"; do :; done
    if [ -f "$STATE/$SIDE-$last" ]; then cat "$STATE/$SIDE-$last"; else echo "true 0"; fi
    ;;
  ps) [ -f "$STATE/$SIDE-ps" ] && cat "$STATE/$SIDE-ps"; exit 0 ;;
  image)
    [ -f "$STATE/no-image" ] && exit 1
    case "$*" in *Labels*) echo "$TF_REF_LABEL"; exit 0 ;; esac
    if [ -f "$STATE/base-missing" ] && [ ! -f "$STATE/base-built" ]; then
      case "$*" in *tensorfold-dsv41:3c45bdaeab7d*) exit 1 ;; esac
    fi
    if [ -f "$STATE/diff-ids" ]; then echo "sha256:$SIDE"; else echo sha256:0123456789abcdef; fi
    ;;
  build) case "$*" in *"-t tensorfold-dsv41:3c45bdaeab7d"*) touch "$STATE/base-built" ;; esac ;;
  save) echo imagebytes ;;
  load) cat > /dev/null ;;
  *) exit 0 ;;
esac
"""

ENV_ZIG = """ENGINE=zig
HEAD_IP=10.100.80.1
WORKER_IP=10.100.80.2
WORKER_SSH=10.100.80.2
RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
MODEL_VIEW={d}/view
CACHE_DIR={d}/cache
ZIG_TF_COMMIT=d58d25ccd6dbd624814cc2ccc848f8e5ba84580c
ZIG_IMAGE=tensorfold-dsv41-zig:d58d25cc
ZIG_BASE_COMMIT=3c45bdaeab7d
ZIG_RANK_PORT=29620
ZIG_RDMA_GID_INDEX={gid}
TF_REPO=https://example.invalid/TensorFold.git
BIND_HOST=127.0.0.1
PORT={port}
SERVED_NAME=DeepSeek-V4.1-Flash-1017
CONTEXT=1048576
PARALLEL=4
TEMPERATURE=1.0
TOP_P=1.0
THINKING=0
CGROUP_PARENT=tensorfold.slice
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Api:
    """A fake engine API: /v1/models and /v1/chat/completions."""

    def __init__(self, content="391", finish="stop", status=200):
        self.content, self.finish, self.status = content, finish, status
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": "DeepSeek-V4.1-Flash-1017"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                msg = {"role": "assistant", "content": outer.content}
                self._send(outer.status, {"choices": [{"message": msg, "finish_reason": outer.finish}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ZigCase(unittest.TestCase):
    """A temporary tree with the Zig scripts, a fixture .env and stub commands."""

    SCRIPTS = ("common.sh", "zig_common.sh", "zig_gid.sh", "zig_start.sh")
    TOP = ()

    def tree(self, port=None, gid="auto", extra_env="", scripts=None, top=None) -> Tree:
        t = Tree(self, "", scripts or self.SCRIPTS, self.TOP if top is None else top)
        (t.dir / ".env").write_text(ENV_ZIG.format(d=t.dir, gid=gid, port=port or free_port()) + extra_env,
                                    encoding="utf-8")
        t.state = t.dir / "state"
        t.state.mkdir()
        t.calls = t.dir / "calls"
        t.calls.touch()
        t.stub("docker", DOCKER)
        write_exec(t.bin / "ssh", SSH_WORKER)
        t.stub("ping", "exit 0\n")
        return t

    def run_in(self, t: Tree, argv, extra=None):
        env = {"CALLS": str(t.calls), "STATE": str(t.state), "TF_REF_LABEL": "d58d25ccd6dbd624814cc2ccc848f8e5ba84580c",
               "ZIG_POLL_S": "0", "PREFLIGHT": "0"}
        env.update(extra or {})
        return t.run(argv, env)

    def calls(self, t: Tree) -> list[str]:
        return t.calls.read_text(encoding="utf-8").splitlines()


def fake_sysfs(base: Path, dev: str, gids: dict) -> Path:
    """gids = {index: (type, value)}"""
    port = base / "class" / "infiniband" / dev / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "types").mkdir(parents=True)
    for n, (kind, value) in gids.items():
        (port / "gids" / str(n)).write_text(value + "\n", encoding="utf-8")
        (port / "gid_attrs" / "types" / str(n)).write_text(kind + "\n", encoding="utf-8")
    return base


MAPPED = "0000:0000:0000:0000:0000:ffff:0a64:5001"
LINKLOCAL = "fe80:0000:0000:0000:0000:0000:0000:0001"


@needs_bash
class GidIndex(ZigCase):
    def gid(self, gids):
        t = self.tree(scripts=("zig_gid.sh",))
        sysfs = fake_sysfs(t.dir / "sys", "rocep1s0f1", gids)
        return self.run_in(t, ["bash", "scripts/zig_gid.sh", "rocep1s0f1"], {"SYSFS": str(sysfs)})

    def test_first_roce_v2_entry_with_an_ipv4_address(self):
        r = self.gid({0: ("IB/RoCE v1", MAPPED), 1: ("RoCE v2", LINKLOCAL), 3: ("RoCE v2", MAPPED), 5: ("RoCE v2", MAPPED)})
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "3"))

    def test_a_moved_index_is_found(self):
        r = self.gid({0: ("RoCE v2", LINKLOCAL), 2: ("RoCE v2", MAPPED)})
        self.assertEqual(r.stdout.strip(), "2")

    def test_no_match_gives_exit_1_and_no_output(self):
        r = self.gid({0: ("IB/RoCE v1", MAPPED), 1: ("RoCE v2", LINKLOCAL)})
        self.assertEqual((r.returncode, r.stdout.strip()), (1, ""))


@needs_bash
@needs_curl
class ZigStart(ZigCase):
    def start(self, api=None, gid="3", extra_env="", extra=None, port=None):
        api = api or Api()
        self.addCleanup(api.close)
        t = self.tree(port=port or api.port, gid=gid, extra_env=extra_env)
        return t, self.run_in(t, ["bash", "scripts/zig_start.sh"], extra)

    def runs(self, t):
        """The docker run lines, as (side, text)."""
        return [(c.split(" ", 1)[0], c) for c in self.calls(t) if " docker run " in c]

    def test_rank_1_starts_before_rank_0_and_the_pair_is_checked_with_a_real_completion(self):
        api = Api()
        t, r = self.start(api)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        runs = self.runs(t)
        self.assertEqual([side for side, _ in runs], ["worker", "head"])
        self.assertIn("first completion: HTTP 200, content '391', finish_reason 'stop': PASS", r.stdout)
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.requests[0]["temperature"], 0)
        self.assertEqual(api.requests[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_rank_1_arguments(self):
        t, r = self.start()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        text = self.runs(t)[0][1]
        for part in ("--name tfv41-r1", "--gpus all", "--ipc=host", "--network host", "--oom-score-adj 1000",
                     "--cgroup-parent tensorfold.slice", "--device /dev/infiniband", "--cap-add IPC_LOCK",
                     "--ulimit memlock=-1", "--restart no",
                     "-e TF_DS_EXIT_ON_FAULT=1", "-e TF_DS_REPLAY_FLOOR=1", "-e TF_DS_STREAMS=4", "-e HF_HUB_OFFLINE=1",
                     "-e TF_RDMA_GID_INDEX=3", "-e TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1",
                     "-e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1", "-e NCCL_SOCKET_IFNAME=enp1s0f1np1",
                     f"-v {t.dir}/view/model:/model:ro", f"-v {t.dir}/view/engram:/engram:ro",
                     f"-v {t.dir}/cache/zig-rank-cache-28:/rank-cache:ro", f"-v {t.dir}/cache/zig-kit:/kit:ro",
                     f"-v {t.dir}/cache/zig-data:/data:ro", "tensorfold-dsv41-zig:d58d25cc",
                     "tf-dsv41-lanes /model /rank-cache 1 2 10.100.80.1 29620 /kit",
                     "--engram /engram --token-map /data/dsv41_token_map.json",
                     "--pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1",
                     "--rdma rocep1s0f1,roceP2p1s0f1"):
            self.assertIn(part, text)
        self.assertNotIn("tensorfold-native", text)
        self.assertNotIn("NCCL_IB_GID_INDEX", text)               # the NCCL GID stays automatic

    def test_rank_0_arguments(self):
        t, r = self.start()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        text = self.runs(t)[1][1]
        port = re.search(r"--port (\d+)", text).group(1)
        for part in ("--name tfv41-r0", "-e TF_DS_EXIT_ON_FAULT=1", "-e TF_TP_WORLD=2", "-e TF_TP_MASTER=10.100.80.1",
                     "-e TF_TP_PORT=29620", "-e TF_DS_RANK_CACHE=/rank-cache", "-e TF_DS_KIT=/kit",
                     "-e TENSORFOLD_CUDA_KERNELS=/kit/aot", "-e TF_DS_ENGRAM=/engram",
                     "-e TF_DS_TOKEN_MAP=/data/dsv41_token_map.json", "-e TF_DS_GRAPHS=1", "-e TF_DS_HC_SIDE=1",
                     "-e TF_DS_L2_PREFETCH=1", "-e TF_DS_ENGRAM_AIO=1", "-e TF_RDMA_GID_INDEX=3",
                     "tensorfold-native serve /model --backend cuda --host 127.0.0.1", f"--port {port}",
                     "--name DeepSeek-V4.1-Flash-1017", "--context 1048576 --parallel 4 --max-tokens 32768",
                     "--temperature 1.0 --top-p 1.0 --top-k 20 --no-thinking"):
            self.assertIn(part, text)
        self.assertNotIn("--thinking ", text.replace("--no-thinking", ""))
        self.assertNotIn("--api-key-file", text)
        self.assertNotIn("--alias", text)
        self.assertNotIn("tf-dsv41-lanes", text)

    def test_old_containers_are_removed_first(self):
        t, r = self.start()
        calls = self.calls(t)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(calls[0], "head docker rm -f tfv41-r0")
        self.assertEqual(calls[1], "worker docker rm -f tfv41-r1")

    def test_thinking_effort_alias_and_key_file(self):
        keys = Path(tempfile.mkdtemp()) / "keys"
        keys.write_text("main: k1\n", encoding="utf-8")
        self.addCleanup(shutil.rmtree, keys.parent, True)
        t, r = self.start(extra_env=f"THINKING=1\nREASONING_EFFORT=high\nSERVED_ALIAS=dgx-model\nAPI_KEY_FILE={keys}\n")
        # The fake API ignores the key. The check sends it as a Bearer header; the run must still pass.
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        head, worker = self.runs(t)[1][1], self.runs(t)[0][1]
        self.assertIn("--thinking", head.replace("--no-thinking", ""))
        self.assertIn("--reasoning-effort high", head)
        self.assertIn("--alias dgx-model", head)
        self.assertIn("--api-key-file /run/keys", head)
        self.assertIn(f"-v {keys}:/run/keys:ro", head)
        self.assertNotIn("/run/keys", worker)                     # only rank 0 serves HTTP

    def test_empty_cgroup_parent_leaves_the_docker_default(self):
        t, r = self.start(extra_env="CGROUP_PARENT=\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("--cgroup-parent", self.runs(t)[0][1])

    def test_a_number_in_zig_rdma_gid_index_wins(self):
        t, r = self.start(gid="7")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for _, text in self.runs(t):
            self.assertIn("-e TF_RDMA_GID_INDEX=7", text)

    def test_auto_reads_the_gid_of_each_node(self):
        api = Api()
        self.addCleanup(api.close)
        t = self.tree(port=api.port, gid="auto")
        sysfs = fake_sysfs(t.dir / "sys", "rocep1s0f1", {0: ("RoCE v2", LINKLOCAL), 4: ("RoCE v2", MAPPED)})
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"], {"SYSFS": str(sysfs)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for _, text in self.runs(t):
            self.assertIn("-e TF_RDMA_GID_INDEX=4", text)

    def test_no_gid_stops_the_start_before_any_container_starts(self):
        api = Api()
        self.addCleanup(api.close)
        t = self.tree(port=api.port, gid="auto")
        sysfs = fake_sysfs(t.dir / "sys", "rocep1s0f1", {0: ("RoCE v2", LINKLOCAL)})
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"], {"SYSFS": str(sysfs)})
        self.assertEqual(r.returncode, 1)
        self.assertIn("no RoCE v2 GID with an IPv4 address", r.stdout)
        self.assertEqual(self.runs(t), [])

    def test_a_wrong_first_completion_fails_the_start(self):
        for content, finish in (("392", "stop"), ("391", "length"), ("", "stop")):
            with self.subTest(content=content, finish=finish):
                t, r = self.start(Api(content=content, finish=finish))
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn("FAIL", r.stdout)
                self.assertIn("the pair is not good", r.stdout)

    def test_a_rank_that_exits_75_is_named(self):
        t = self.tree(port=free_port(), gid="3")                   # no API answers on this port
        (t.state / "head-tfv41-r0").write_text("false 75\n", encoding="utf-8")
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("head rank exited with 75: a peer fault", r.stdout)

    def test_a_rank_that_is_killed_is_named(self):
        t = self.tree(port=free_port(), gid="3")
        (t.state / "worker-tfv41-r1").write_text("false 137\n", encoding="utf-8")
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("worker rank exited with 137: it was killed", r.stdout)

    def test_another_exit_code_says_do_not_loop(self):
        t = self.tree(port=free_port(), gid="3")
        (t.state / "head-tfv41-r0").write_text("false 1\n", encoding="utf-8")
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"])
        self.assertIn("Do not restart in a loop", r.stdout)

    def test_wrong_settings_exit_2(self):
        for extra in ("PARALLEL=8\n", "PARALLEL=0\n", "THINKING=2\n", "REASONING_EFFORT=max\n"):
            with self.subTest(extra=extra.strip()):
                t = self.tree(gid="3", extra_env=extra)
                r = self.run_in(t, ["bash", "scripts/zig_start.sh"])
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertEqual(self.runs(t), [])
        t = self.tree(gid="eth0")
        r = self.run_in(t, ["bash", "scripts/zig_start.sh"])
        self.assertEqual(r.returncode, 2)
        self.assertIn("ZIG_RDMA_GID_INDEX", r.stderr)

    def test_the_script_forwards_the_protected_values(self):
        """Every option of the server that takes a setting of .env takes the variable, not a literal value."""
        text = (ROOT / "scripts" / "zig_start.sh").read_text(encoding="utf-8")
        for option, var in (("--host", "BIND_HOST"), ("--port", "PORT"), ("--name", "SERVED_NAME"),
                            ("--context", "CONTEXT"), ("--parallel", "PARALLEL"), ("--temperature", "TEMPERATURE"),
                            ("--top-p", "TOP_P"), ("--pool", "CONTEXT"), ("--streams", "PARALLEL")):
            with self.subTest(option=option):
                self.assertIn(f"{option} ${var}", text, f"zig_start.sh must pass {option} ${var}")
        # Thinking is off unless THINKING=1: the only --thinking in the script is the THINKING=1 branch.
        switch = '[ "${THINKING:-0}" = 1 ] && echo --thinking || echo --no-thinking'
        self.assertIn(switch, text)
        self.assertEqual(re.findall(r"(?<!no-)--thinking\b", text.replace(switch, "")), [])


@needs_bash
class StartSwitch(ZigCase):
    """start.sh hands over to zig_start.sh unless ENGINE=python."""

    def switch(self, engine_line: str, extra_env: str = ""):
        t = self.tree(gid="3", extra_env=extra_env, scripts=("common.sh",), top=("start.sh",))
        env = (t.dir / ".env").read_text(encoding="utf-8").replace("ENGINE=zig\n", engine_line)
        (t.dir / ".env").write_text(env, encoding="utf-8")
        write_exec(t.dir / "scripts" / "zig_start.sh", "#!/bin/sh\necho ZIG-START-RAN \"$@\"\n")
        return self.run_in(t, ["bash", "start.sh", "extra"])

    def test_zig_is_the_default(self):
        r = self.switch("")                                       # no ENGINE line in .env
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ZIG-START-RAN extra"))

    def test_engine_zig(self):
        self.assertIn("ZIG-START-RAN", self.switch("ENGINE=zig\n").stdout)

    def test_engine_python_runs_the_python_start(self):
        r = self.switch("ENGINE=python\n", "MEM_CAP=7\n")          # the Python start checks MEM_CAP first
        self.assertNotIn("ZIG-START-RAN", r.stdout)
        self.assertEqual(r.returncode, 1)
        self.assertIn("MEM_CAP must be 0 or 1", r.stderr)

    def test_another_value_is_refused(self):
        r = self.switch("ENGINE=vllm\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("ENGINE must be zig or python", r.stderr)


class SuperviseCase(ZigCase):
    SCRIPTS = ("common.sh", "zig_common.sh", "zig_supervise.sh")

    def supervise(self, states, restart_script="", extra=None, extra_env=""):
        """states = (head state file text, worker state file text). restart_script runs in the start.sh stub."""
        t = self.tree(gid="3", extra_env=extra_env)
        for name, state in (("head-tfv41-r0", states[0]), ("worker-tfv41-r1", states[1])):
            (t.state / name).write_text(state.replace("exited ", "false ") + "\n", encoding="utf-8")
        write_exec(t.dir / "stop.sh", f"#!/bin/sh\necho stop >> {t.dir}/actions\n")
        write_exec(t.dir / "start.sh", f"#!/bin/sh\necho start >> {t.dir}/actions\n{restart_script}\n")
        env = {"ZIG_GRACE_S": "0"}
        env.update(extra or {})
        r = self.run_in(t, ["bash", "scripts/zig_supervise.sh"], env)
        actions = (t.dir / "actions").read_text(encoding="utf-8").split() if (t.dir / "actions").exists() else []
        return r, actions


@needs_bash
class ZigSupervise(SuperviseCase):
    def test_a_clean_stop_ends_with_0_and_no_restart(self):
        for worker in ("exited 0", "exited 75"):
            with self.subTest(worker=worker):
                r, actions = self.supervise(("exited 0", worker))
                self.assertEqual(r.returncode, 0, r.stdout)
                self.assertEqual(actions, [])
                self.assertIn("rank 0 stopped clean", r.stdout)

    def test_exit_75_stops_both_and_starts_again(self):
        restart = "printf 'false 0\\n' > \"$STATE/head-tfv41-r0\""   # the new pair runs, then stops clean
        r, actions = self.supervise(("exited 75", "exited 75"), restart)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(actions, ["stop", "start"])
        self.assertIn("restart 1 of 3", r.stdout)

    def test_one_rank_with_75_and_the_other_still_running_restarts_after_the_grace(self):
        restart = "printf 'false 0\\n' > \"$STATE/head-tfv41-r0\""
        r, actions = self.supervise(("exited 75", "true 0"), restart, {"ZIG_GRACE_S": "1", "ZIG_POLL_S": "1"})
        self.assertEqual(actions, ["stop", "start"], r.stdout)
        self.assertIn("rank 0 75, rank 1 running", r.stdout)

    def test_a_killed_rank_is_not_restarted(self):
        for states in (("exited 137", "exited 75"), ("exited 75", "exited 137")):
            with self.subTest(states=states):
                r, actions = self.supervise(states)
                self.assertEqual(r.returncode, 1)
                self.assertEqual(actions, [])
                self.assertIn("killed (137): no restart", r.stdout)

    def test_another_error_code_is_not_restarted(self):
        for states in (("exited 1", "exited 75"), ("exited 75", "exited 2")):
            with self.subTest(states=states):
                r, actions = self.supervise(states)
                self.assertEqual(r.returncode, 1)
                self.assertEqual(actions, [])
                self.assertIn("do not restart in a loop", r.stdout)

    def test_the_restarts_in_a_row_have_a_limit(self):
        r, actions = self.supervise(("exited 75", "exited 75"), "", {"ZIG_MAX_RESTARTS": "2"})   # the start stub changes nothing
        self.assertEqual(r.returncode, 1)
        self.assertEqual(actions, ["stop", "start", "stop", "start"])
        self.assertIn("2 restarts in a row: give up", r.stdout)

    def test_a_start_that_fails_ends_the_supervisor(self):
        r, actions = self.supervise(("exited 75", "exited 75"), "exit 1")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(actions, ["stop", "start"])
        self.assertIn("did not end with a good first completion", r.stdout)


@needs_bash
class ZigInputs(ZigCase):
    SCRIPTS = ("common.sh", "zig_common.sh", "zig_inputs.sh", "verify_kit.sh")

    def inputs(self, args, setup=None, extra=None, script_edit=None):
        t = self.tree(gid="3", scripts=self.SCRIPTS)
        (t.dir / "view" / "model").mkdir(parents=True)
        (t.dir / "view" / "model" / "config.json").write_text("{}", encoding="utf-8")
        (t.dir / "view" / "model" / "chat_template.jinja").write_text("t", encoding="utf-8")
        for name in ("rsync", "hf"):
            t.stub(name, f'echo "$SIDE {name} $*" >> "$CALLS"\nexit 0\n')
        if setup:
            setup(t)
        if script_edit:
            path = t.dir / "scripts" / "zig_inputs.sh"
            path.write_text(script_edit(path.read_text(encoding="utf-8")), encoding="utf-8")
        return t, self.run_in(t, ["bash", "scripts/zig_inputs.sh", *args], extra)

    def test_an_unknown_phase_is_refused(self):
        _, r = self.inputs(["weights"])
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown phase: weights", r.stderr)

    def test_help(self):
        _, r = self.inputs(["--help"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("rank-cache", r.stdout)

    def test_rank_cache_refuses_while_a_rank_container_exists(self):
        def setup(t):
            (t.state / "worker-ps").write_text("tfv41-r1\n", encoding="utf-8")
        t, r = self.inputs(["rank-cache"], setup)
        self.assertEqual(r.returncode, 1)
        self.assertIn("run ./stop.sh first", r.stdout)
        self.assertFalse([c for c in self.calls(t) if " docker run " in c])

    def test_rank_cache_makes_one_weight_file_on_each_node(self):
        t, r = self.inputs(["rank-cache"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        runs = [c for c in self.calls(t) if " docker run " in c]
        self.assertEqual(sorted(c.split(" ", 1)[0] for c in runs), ["head", "worker"])
        for side, rank in (("head", "0"), ("worker", "1")):
            text = next(c for c in runs if c.startswith(side))
            self.assertRegex(text, r"--user \d+:\d+ ")
            for part in ("--rm", "--network none", "--memory 24g", "--cpus 16",
                         f"-v {t.dir}/view/model:/model:ro", f"-v {t.dir}/cache/zig-rank-cache-28:/out",
                         "--entrypoint tf-dsv41-rank-cache tensorfold-dsv41-zig:d58d25cc", f"/model /out {rank} 2"):
                self.assertIn(part, text)

    def test_rank_cache_does_nothing_when_both_files_are_there(self):
        def setup(t):
            d = t.dir / "cache" / "zig-rank-cache-28"
            d.mkdir(parents=True)
            (d / "rank0of2-abc.bin").write_text("x", encoding="utf-8")
            (d / "rank1of2-def.bin").write_text("x", encoding="utf-8")        # the ssh stub shares the folder
        t, r = self.inputs(["rank-cache"], setup)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("both weight files are here", r.stdout)
        self.assertFalse([c for c in self.calls(t) if " docker run " in c])

    def test_rank_cache_rebuild_removes_the_old_files_first(self):
        def setup(t):
            d = t.dir / "cache" / "zig-rank-cache-28"
            d.mkdir(parents=True)
            (d / "rank0of2-abc.bin").write_text("x", encoding="utf-8")
            (d / "rank1of2-def.bin").write_text("x", encoding="utf-8")
        t, r = self.inputs(["rank-cache"], setup, {"REBUILD": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(list((t.dir / "cache" / "zig-rank-cache-28").glob("*.bin")), [])
        self.assertEqual(len([c for c in self.calls(t) if " docker run " in c]), 2)

    def test_the_image_must_be_the_same_on_both_nodes(self):
        def setup(t):
            (t.state / "diff-ids").touch()
        _, r = self.inputs(["token-map"], setup)
        self.assertEqual(r.returncode, 1)
        self.assertIn("the worker holds another", r.stdout)

    def test_token_map_is_made_in_the_image_and_copied(self):
        def setup(t):
            (t.dir / "cache" / "zig-data").mkdir(parents=True)
            # the docker stub makes no file; the script then copies the one it finds, so give it one for the check
            (t.dir / "cache" / "zig-data" / "dsv41_token_map.json").write_text("[0]", encoding="utf-8")
        t, r = self.inputs(["token-map"], setup, {"REBUILD": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        run = next(c for c in self.calls(t) if " docker run " in c)
        for part in ("--network none", "--entrypoint python3 tensorfold-dsv41-zig:d58d25cc",
                     "/opt/tensorfold/tools/dsv41-zig/inputs/make_token_map.py /model /data/dsv41_token_map.json"):
            self.assertIn(part, run)
        self.assertIn("the same file is on both nodes", r.stdout)
        self.assertIn("WARNING: sha256", r.stdout)                  # the file is not the one of the tests

    def test_token_map_that_equals_the_tested_file_gets_no_warning(self):
        text = "[0]"
        pin = hashlib.sha256(text.encode()).hexdigest()

        def setup(t):
            (t.dir / "cache" / "zig-data").mkdir(parents=True)
            (t.dir / "cache" / "zig-data" / "dsv41_token_map.json").write_text(text, encoding="utf-8")
        _, r = self.inputs(["token-map"], setup, script_edit=lambda s: re.sub(r"TOKEN_MAP_SHA256=\w+", f"TOKEN_MAP_SHA256={pin}", s))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("the file is the one of the tests", r.stdout)
        self.assertNotIn("WARNING", r.stdout)

    def test_the_token_map_pin_is_the_sha256_of_the_tested_file(self):
        text = (ROOT / "scripts" / "zig_inputs.sh").read_text(encoding="utf-8")
        self.assertIn("TOKEN_MAP_SHA256=de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28", text)

    # --- the kernel kit: a fixed commit, a pinned MANIFEST, the checker of this repo, nothing from the kit runs -------
    @staticmethod
    def mark(kit: Path) -> str:
        """A line for a bad verify_kit.sh: it makes the file 'marker' next to the fixture folder when it runs."""
        return f'touch "{kit.parent.parent / "marker"}"\n'

    def make_kit(self, base: Path, files=None, tamper_verifier=False) -> tuple[Path, str]:
        """A small kit in base/kit. Returns the folder and the sha256 of its MANIFEST."""
        kit = base / "kit"
        files = files or {"aot/k.bin": "kernel", "engram.json": "{}", "README.md": "readme"}
        verifier = "#!/bin/sh\n" + (self.mark(kit) if tamper_verifier else "") + "echo kit OK\n"
        files = {**files, "tools/verify_kit.sh": verifier}
        lines = []
        for name, text in sorted(files.items()):
            path = kit / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            lines.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}")
        manifest = "\n".join(lines) + "\n"
        (kit / "MANIFEST").write_text(manifest, encoding="utf-8", newline="\n")
        return kit, hashlib.sha256(manifest.encode()).hexdigest()

    def kit_run(self, mutate=None, tamper_verifier=False, repin=True, preload=False, damage_after_rsync=False):
        """Run the kit phase with an hf stub that downloads the fixture kit. mutate(kit) changes the kit after MANIFEST."""
        holder = {}

        def setup(t):
            kit, pin = self.make_kit(t.dir / "fixture", tamper_verifier=tamper_verifier)
            if mutate:
                mutate(kit)
            holder["pin"] = pin
            holder["marker"] = t.dir / "marker"
            body = ('echo "$SIDE hf $*" >> "$CALLS"\n'
                    'while [ $# -gt 0 ]; do [ "$1" = --local-dir ] && dir=$2; shift; done\n'
                    f'mkdir -p "$dir" && cp -a "{kit}" "$dir/kit"\n')
            t.stub("hf", body)
            if preload:
                shutil.copytree(kit, t.dir / "cache" / "zig-kit")
            if damage_after_rsync:    # the stub ssh shares the folder, so this plays a copy that arrives damaged
                t.stub("rsync", 'echo "$SIDE rsync $*" >> "$CALLS"\n'
                                f'echo evil > "{t.dir}/cache/zig-kit/aot/k.bin"\n')

        def edit(s):
            pin = holder["pin"] if repin else "0" * 64
            return re.sub(r"KIT_MANIFEST_SHA256=\w+", f"KIT_MANIFEST_SHA256={pin}", s)

        t, r = self.inputs(["kit"], setup, extra=None, script_edit=edit)
        return t, r, holder["marker"]

    def test_the_kit_download_is_fixed_to_one_commit(self):
        t, r, _ = self.kit_run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        call = next(c for c in self.calls(t) if " hf download " in c)
        self.assertIn("--revision 4996441a82d95b9a7853b01a1255e1dafc926387 ", call)
        self.assertRegex((ROOT / "scripts" / "zig_inputs.sh").read_text(encoding="utf-8"), r"KIT_REVISION=[0-9a-f]{40}\n")

    def test_a_good_kit_is_copied_and_checked_on_both_nodes_with_the_checker_of_this_repo(self):
        t, r, marker = self.kit_run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((t.dir / "cache" / "zig-kit" / "MANIFEST").is_file())
        self.assertIn("the head holds a kit that matches its MANIFEST", r.stdout)
        self.assertIn("the worker holds a kit that matches its MANIFEST", r.stdout)
        rsync = next(c for c in self.calls(t) if " rsync " in c)
        self.assertIn("--delete", rsync)
        self.assertFalse(marker.exists())

    def test_a_tampered_file_stops_before_the_kit_is_kept(self):
        t, r, marker = self.kit_run(mutate=lambda kit: (kit / "aot" / "k.bin").write_text("evil", encoding="utf-8"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("a file differs from MANIFEST", r.stdout + r.stderr)
        self.assertFalse((t.dir / "cache" / "zig-kit" / "MANIFEST").exists())
        self.assertFalse([c for c in self.calls(t) if " rsync " in c])
        self.assertEqual(list((t.dir / "cache").glob(".zig-kit-dl.*")), [])

    def test_an_extra_file_stops_before_the_kit_is_kept(self):
        t, r, _ = self.kit_run(mutate=lambda kit: (kit / "aot" / "extra.so").write_text("evil", encoding="utf-8"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("a file is not in MANIFEST: aot/extra.so", r.stdout + r.stderr)
        self.assertFalse((t.dir / "cache" / "zig-kit" / "MANIFEST").exists())
        self.assertFalse([c for c in self.calls(t) if " rsync " in c])

    def test_a_tampered_verify_kit_sh_does_not_run_and_stops_the_kit(self):
        # The pin is the real MANIFEST, but the kit holds a verify_kit.sh with another content.
        def mutate(kit):
            write_exec(kit / "tools" / "verify_kit.sh", "#!/bin/sh\n" + self.mark(kit) + "echo kit OK\n")
        t, r, marker = self.kit_run(mutate=mutate)
        self.assertEqual(r.returncode, 1)
        self.assertIn("tools/verify_kit.sh", r.stdout + r.stderr)
        self.assertFalse(marker.exists())

    def test_a_pinned_kit_never_runs_its_own_verify_kit_sh_on_either_node(self):
        # Pin and MANIFEST agree with the bad file, so the kit passes. The script of the kit must still not run.
        t, r, marker = self.kit_run(tamper_verifier=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(marker.exists())

    def test_a_wrong_manifest_pin_stops_before_the_kit_is_kept(self):
        t, r, _ = self.kit_run(repin=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("MANIFEST has sha256", r.stdout)
        self.assertFalse((t.dir / "cache" / "zig-kit" / "MANIFEST").exists())

    def test_a_kit_that_is_already_there_is_checked_again(self):
        t, r, _ = self.kit_run(preload=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([c for c in self.calls(t) if " hf download " in c])
        (t.dir / "cache" / "zig-kit" / "engram.json").write_text("changed", encoding="utf-8")
        r = self.run_in(t, ["bash", "scripts/zig_inputs.sh", "kit"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not match its MANIFEST", r.stdout + r.stderr)

    def test_a_copy_that_arrives_damaged_is_found_on_the_worker_by_the_checker_from_stdin(self):
        t, r, marker = self.kit_run(damage_after_rsync=True, tamper_verifier=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("the head holds a kit that matches its MANIFEST", r.stdout)
        self.assertIn("the kit on the worker does not match its MANIFEST", r.stdout)
        self.assertFalse(marker.exists())

    def test_the_script_does_not_call_the_kit(self):
        text = (ROOT / "scripts" / "zig_inputs.sh").read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("$ZIG_KIT/tools", code)
        self.assertNotIn("kit/tools", code)
        self.assertIn('wrun bash -s -- "$ZIG_KIT" < "$ROOT/scripts/verify_kit.sh"', code)

    def test_the_kit_pin_is_the_sha256_of_the_tested_manifest(self):
        text = (ROOT / "scripts" / "zig_inputs.sh").read_text(encoding="utf-8")
        self.assertIn("KIT_MANIFEST_SHA256=8c6dcb712a7557a8aeb794d68daf85b07befae7bb1bdfa470223e957718f2845", text)


@needs_bash
class ZigBuildImage(ZigCase):
    SCRIPTS = ("common.sh", "zig_common.sh", "zig_build_image.sh")

    def build(self, extra_env="", setup=None, extra=None):
        t = self.tree(gid="3", scripts=self.SCRIPTS, extra_env=extra_env)
        (t.dir / "docker").mkdir()
        (t.dir / "docker" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (t.dir / "docker" / "Dockerfile.zig").write_text("FROM scratch\n", encoding="utf-8")
        if setup:
            setup(t)
        env = {"MEMINFO_FILE": str(t.meminfo(120 * GIB_KB))}
        env.update(extra or {})
        return t, self.run_in(t, ["bash", "scripts/zig_build_image.sh"], env)

    def test_the_image_tag_must_match_the_commit(self):
        _, r = self.build("ZIG_IMAGE=tensorfold-dsv41-zig:deadbeef\n")
        self.assertEqual(r.returncode, 2)
        self.assertIn("must be tensorfold-dsv41-zig:d58d25cc", r.stderr)

    def test_the_commit_must_be_a_full_hash(self):
        _, r = self.build("ZIG_TF_COMMIT=d58d25cc\n")
        self.assertEqual(r.returncode, 2)
        self.assertIn("full 40-character commit", r.stderr)

    def test_it_refuses_while_a_rank_container_exists(self):
        def setup(t):
            (t.state / "head-ps").write_text("tfv41-r0\n", encoding="utf-8")
        _, r = self.build(setup=setup)
        self.assertEqual(r.returncode, 1)
        self.assertIn("run ./stop.sh first", r.stderr)

    def test_it_needs_40_gib_of_memory(self):
        _, r = self.build(extra={"MEMINFO_FILE": "/nonexistent"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("need 40 for the compile", r.stderr)

    def test_the_base_image_is_built_when_it_is_missing_and_then_the_zig_image(self):
        def setup(t):
            (t.state / "base-missing").touch()                    # "image inspect" of the base fails until the script builds it
        t, r = self.build(setup=setup)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        builds = [c for c in self.calls(t) if " docker build " in c]
        self.assertEqual(len(builds), 2, builds)
        self.assertIn("--build-arg TF_COMMIT=3c45bdaeab7d -t tensorfold-dsv41:3c45bdaeab7d", builds[0])
        for part in ("-f ", "docker/Dockerfile.zig", "--build-arg BASE_IMAGE=tensorfold-dsv41:3c45bdaeab7d",
                     "--build-arg TF_REF=d58d25ccd6dbd624814cc2ccc848f8e5ba84580c", "-t tensorfold-dsv41-zig:d58d25cc"):
            self.assertIn(part, builds[1])
        self.assertTrue([c for c in self.calls(t) if c.startswith("head docker save tensorfold-dsv41-zig:d58d25cc")])
        self.assertTrue([c for c in self.calls(t) if c.startswith("worker docker load")])
        self.assertIn("both nodes hold tensorfold-dsv41-zig:d58d25cc", r.stdout)

    def test_a_wrong_label_fails(self):
        _, r = self.build(extra={"TF_REF_LABEL": "0" * 40})
        self.assertEqual(r.returncode, 1)
        self.assertIn("label tf.ref is", r.stdout)


@needs_bash
class ZigPreflight(ZigCase):
    SCRIPTS = ("common.sh", "zig_common.sh", "preflight.sh")

    def preflight(self, avail_kb=120 * GIB_KB, prepare=False, extra_env="", no_image=True):
        t = self.tree(gid="auto", scripts=self.SCRIPTS, extra_env=extra_env)
        if no_image:
            (t.state / "no-image").touch()
        text = (t.dir / "scripts" / "preflight.sh").read_text(encoding="utf-8").replace("/proc/meminfo", str(t.meminfo(avail_kb)))
        (t.dir / "scripts" / "preflight.sh").write_text(text, encoding="utf-8")
        for name in ("ss", "systemctl", "ibv_devinfo"):
            t.stub(name, "exit 1\n")
        if prepare:
            cache = t.dir / "cache"
            (cache / "zig-kit" / "aot").mkdir(parents=True)
            (cache / "zig-kit" / "MANIFEST").write_text("m", encoding="utf-8")
            (cache / "zig-data").mkdir()
            (cache / "zig-data" / "dsv41_token_map.json").write_text("[]", encoding="utf-8")
            (cache / "zig-rank-cache-28").mkdir()
            (cache / "zig-rank-cache-28" / "rank0of2-a.bin").write_text("x", encoding="utf-8")
            (cache / "zig-rank-cache-28" / "rank1of2-b.bin").write_text("x", encoding="utf-8")
        r = self.run_in(t, ["bash", "scripts/preflight.sh"])
        return r.stdout.splitlines()

    def test_both_nodes_need_110_gib(self):
        lines = [x for x in self.preflight(110 * GIB_KB) if "memory available" in x]
        self.assertEqual(lines, ["[ ok ] head: 110 GiB memory available (need 110)",
                                 "[ ok ] worker: 110 GiB memory available (need 110)"])
        lines = [x for x in self.preflight(109 * GIB_KB) if "memory available" in x]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertTrue(line.startswith("[FAIL]"), line)
            self.assertIn("need 110", line)
            self.assertNotIn("VISION", line)                       # the Zig engine has no vision setting

    def test_no_vision_hint_for_zig_and_vision_does_not_lower_the_need(self):
        lines = [x for x in self.preflight(109 * GIB_KB, extra_env="VISION=0\n") if "memory available" in x]
        self.assertIn("need 110", lines[0])

    def test_missing_inputs_are_named(self):
        out = "\n".join(self.preflight())
        self.assertIn("no kernel kit in", out)
        self.assertIn("zig_inputs.sh kit", out)
        self.assertIn("dsv41_token_map.json: run scripts/zig_inputs.sh token-map", out)
        self.assertIn("no rank cache file for rank 0", out)
        self.assertIn("no rank cache file for rank 1", out)
        self.assertIn("of 42 weight files", out)
        self.assertIn("the nodes hold different images (or none) for tensorfold-dsv41-zig:d58d25cc: run scripts/zig_build_image.sh", out)

    def test_present_inputs_pass(self):
        out = "\n".join(self.preflight(prepare=True))
        self.assertNotIn("no kernel kit", out)
        self.assertNotIn("dsv41_token_map.json: run", out)
        self.assertIn("[ ok ] head: rank cache of rank 0", out)
        self.assertIn("[ ok ] worker: rank cache of rank 1", out)

    def test_the_python_engine_keeps_its_own_checks(self):
        out = "\n".join(self.preflight(extra_env="ENGINE=python\nIMAGE=tensorfold-dsv41:test\nGID_INDEX=-1\n"))
        self.assertIn("of 43 weight shards", out)
        self.assertIn("the nodes hold different images (or none) for tensorfold-dsv41:test: run docker/build.sh", out)
        self.assertNotIn("kernel kit", out)
        self.assertIn("need 111", out)

    def test_an_unknown_engine_is_refused(self):
        t = self.tree(gid="auto", scripts=self.SCRIPTS, extra_env="ENGINE=vllm\n")
        r = self.run_in(t, ["bash", "scripts/preflight.sh"])
        self.assertEqual(r.returncode, 1)


class Files(unittest.TestCase):
    """The checked-in files agree with each other."""

    def env(self):
        values = {}
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            m = re.match(r"([A-Z_][A-Z0-9_]*)=(\S*)", line)
            if m:
                values[m.group(1)] = m.group(2)
        return values

    def test_hash_list_of_the_2_8_pack(self):
        lines = (ROOT / "runs/2026-10-22/zig-28/weights-2.8bpw.sha256").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 51)
        names = [line.split("  ", 1)[1] for line in lines]
        self.assertEqual(len(set(names)), 51)
        for line in lines:
            self.assertRegex(line, r"^[0-9a-f]{64}  (model|engram)/\S+$")
        model = [n for n in names if n.startswith("model/")]
        self.assertEqual(len(model), 47)
        self.assertEqual(len([n for n in model if n.endswith(".safetensors")]), 42)       # download.sh EXL3_SHARDS
        for must in ("model/config.json", "model/tokenizer.json", "model/tokenizer_config.json",
                     "model/model.safetensors.index.json", "model/native.safetensors", "model/head-mtp.safetensors",
                     "model/LICENSE"):
            self.assertIn(must, names)
        self.assertEqual([f"model/layer-{i:02d}.safetensors" for i in range(40)],
                         sorted(n for n in model if "layer-" in n))
        self.assertNotIn("model/chat_template.jinja", names)       # the template comes from scripts/patch_template.py
        self.assertEqual(sorted(n for n in names if n.startswith("engram/")),
                         ["engram/config.json", "engram/model-00047-of-00048.safetensors",
                          "engram/model-00048-of-00048.safetensors", "engram/model.safetensors.index.json"])

    def test_engram_lines_equal_the_lines_of_the_2_9_list(self):
        old = (ROOT / "runs/2026-10-17/weights.sha256").read_text(encoding="utf-8").splitlines()
        new = (ROOT / "runs/2026-10-22/zig-28/weights-2.8bpw.sha256").read_text(encoding="utf-8").splitlines()
        self.assertEqual([x for x in old if "  engram/" in x], [x for x in new if "  engram/" in x])

    def test_download_script_pins_the_2_8_pack(self):
        text = (ROOT / "scripts" / "download.sh").read_text(encoding="utf-8")
        self.assertIn("EXL3_REPO=jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw", text)
        self.assertIn("EXL3_REV=859396591c340195526aaf61f94b4ad5323a5e18", text)
        self.assertIn("EXL3_SHARDS=42", text)
        self.assertIn("EXL3_BYTES=203793355048", text)
        # the 2.9 bpw pack of the rollback keeps its values
        self.assertIn("EXL3_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw", text)
        self.assertIn("EXL3_REV=afc7aa0aab98ee735c0f54e0a122019af1ac08f1", text)
        self.assertIn("EXL3_BYTES=210818691539", text)

    def test_fallbacks_in_the_zig_scripts_equal_env_example(self):
        env = self.env()
        for name in ("scripts/zig_common.sh", "scripts/zig_start.sh", "scripts/zig_supervise.sh", "scripts/zig_inputs.sh",
                     "scripts/zig_build_image.sh", "scripts/preflight.sh", "start.sh"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for key, fallback in set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):?-([^}]*)\}", text)):
                if env.get(key):
                    with self.subTest(file=name, key=key):
                        self.assertEqual(fallback, env[key], f"{name} ${{{key}:-{fallback}}} differs from .env.example")

    def test_dockerfile_defaults_match_env_example(self):
        env = self.env()
        text = (ROOT / "docker" / "Dockerfile.zig").read_text(encoding="utf-8")
        self.assertIn(f"ARG BASE_IMAGE=tensorfold-dsv41:{env['ZIG_BASE_COMMIT']}", text)
        self.assertRegex(text, r"ARG ZIG_VERSION=0\.17\.0\n")
        self.assertRegex(text, r"ARG ZIG_SHA256=[0-9a-f]{64}\n")
        self.assertIn("-Dsm=${SM}", text)
        self.assertIn("-Doptimize=ReleaseFast", text)
        self.assertIn("LABEL tf.engine=zig tf.ref=${TF_REF}", text)
        for binary in ("tensorfold-native", "tf-dsv41-lanes"):
            self.assertIn(binary, text)

    def test_zig_scripts_are_listed_with_the_rollback_note(self):
        for name in ("scripts/zig_start.sh", "scripts/zig_supervise.sh", "scripts/zig_inputs.sh",
                     "scripts/zig_build_image.sh", "scripts/zig_gid.sh"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_the_exit_code_contract_is_in_the_script_and_the_doc(self):
        for name in ("scripts/zig_supervise.sh", "docs/ZIG.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for code in ("75", "137"):
                self.assertIn(code, text, name)


if __name__ == "__main__":
    unittest.main()


@needs_bash
class VerifyKit(unittest.TestCase):
    """scripts/verify_kit.sh on its own."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.kit = self.tmp / "kit"
        for name, text in {"a.bin": "one", "d/b.txt": "two", "tools/verify_kit.sh": "#!/bin/sh\nexit 0\n"}.items():
            (self.kit / name).parent.mkdir(parents=True, exist_ok=True)
            (self.kit / name).write_text(text, encoding="utf-8", newline="\n")
        self.write_manifest()

    def write_manifest(self, extra_lines=()):
        lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(self.kit).as_posix()}"
                 for p in sorted(self.kit.rglob("*")) if p.is_file() and p.name != "MANIFEST"]
        (self.kit / "MANIFEST").write_text("\n".join([*lines, *extra_lines]) + "\n", encoding="utf-8", newline="\n")

    def run_check(self, via_stdin=False):
        script = ROOT / "scripts" / "verify_kit.sh"
        if via_stdin:
            with script.open("rb") as f:
                return subprocess.run(["bash", "-s", "--", str(self.kit)], stdin=f, capture_output=True, text=True)
        return subprocess.run(["bash", str(script), str(self.kit)], capture_output=True, text=True)

    def test_a_correct_kit_passes_also_from_stdin(self):
        for stdin in (False, True):
            r = self.run_check(stdin)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("kit OK: 3 files match MANIFEST", r.stdout)

    def test_a_tampered_file_fails(self):
        (self.kit / "a.bin").write_text("evil", encoding="utf-8")
        r = self.run_check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("a.bin", r.stdout)

    def test_a_tampered_verify_kit_sh_fails_and_does_not_run(self):
        (self.kit / "tools" / "verify_kit.sh").write_text("#!/bin/sh\ntouch pwned\n", encoding="utf-8")
        r = self.run_check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("tools/verify_kit.sh", r.stdout)
        self.assertFalse((self.kit / "pwned").exists())

    def test_an_extra_file_fails_also_in_a_new_folder(self):
        (self.kit / "d" / "extra").mkdir()
        (self.kit / "d" / "extra" / "x.so").write_text("x", encoding="utf-8")
        r = self.run_check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("a file is not in MANIFEST: d/extra/x.so", r.stdout)

    def test_a_missing_file_fails(self):
        (self.kit / "d" / "b.txt").unlink()
        r = self.run_check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("a file of MANIFEST is missing: d/b.txt", r.stdout)

    def test_a_link_fails(self):
        try:
            (self.kit / "link").symlink_to("/etc/passwd")
        except OSError:
            self.skipTest("no symbolic links here")
        r = self.run_check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("link", r.stdout)

    def test_bad_manifest_lines_fail(self):
        for bad in ("short  a.bin", "0" * 64 + "  ../x", "0" * 64 + "  /etc/passwd", "0" * 64 + " one-space"):
            self.write_manifest([bad])
            r = self.run_check()
            self.assertEqual(r.returncode, 1, bad)
            self.assertIn("MANIFEST", r.stdout)

    def test_no_manifest_and_wrong_use(self):
        (self.kit / "MANIFEST").unlink()
        self.assertEqual(self.run_check().returncode, 1)
        r = subprocess.run(["bash", str(ROOT / "scripts" / "verify_kit.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
