"""Offline guard for the default behaviour of ./start.sh (stdlib only; no server, no GPU).

A change to a default changes the server of every user after `git pull` ("Default behaviour" in CONTRIBUTING.md).
This test fails when start.sh or .env.example changes a protected default. To change a default on purpose, follow
"Default behaviour" in CONTRIBUTING.md, then change PROTECTED below in the same pull request.

Run: python3 -m unittest discover -s tests -p 'test_*.py'   (or tests/run_offline.sh)
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOW = "Changing a default? Follow 'Default behaviour' in CONTRIBUTING.md and update PROTECTED in tests/test_defaults.py."

# The defaults that the "Default behaviour" table of CONTRIBUTING.md names. Values are as .env.example writes them.
PROTECTED = {
    "BIND_HOST": "127.0.0.1",
    "CONTEXT": "1048576",
    "PARALLEL": "4",
    "MTP_DRAFTS": "5",
    "SPILL_GIB": "64",
    "THINKING": "0",
    # On since engine 1.1.0: a call to a tool that the request did not offer is dropped.
    "DROP_UNOFFERED_TOOL_CALLS": "1",
    # The engine. Since 2026-10-22 the default is the Zig engine with the 2.8 bpw pack; the Python engine 1.1.1 with the
    # 2.9 bpw pack is the rollback (ENGINE=python). The user decided this on 2026-10-22 (CHANGELOG.md).
    "ENGINE": "zig",
    # The engine builds. A change here must come with new gate results (see CONTRIBUTING.md).
    # Zig engine: fork branch dsv41-zig-dev. Rollback (Python 1.1.1):
    "ZIG_TF_COMMIT": "d58d25ccd6dbd624814cc2ccc848f8e5ba84580c",
    "ZIG_IMAGE": "tensorfold-dsv41-zig:d58d25cc",
    "TF_COMMIT": "cb4ceaff2b8c",
    "IMAGE": "tensorfold-dsv41:cb4ceaff2b8c",
}
PREFILL_CHUNK = "2048"
# Where start.sh gives each protected value to the server. Every use of the option must take the variable, so a
# literal value or a missing option fails.
FORWARDED = {
    "BIND_HOST": "--host",
    "CONTEXT": "--context",
    "PARALLEL": "--parallel",
    "MTP_DRAFTS": "--mtp-drafts",
    "IMAGE": "--entrypoint tensorfold",
    "SPILL_GIB": "TF_DS_SPILL_GIB=",
    "DROP_UNOFFERED_TOOL_CALLS": "TF_DROP_UNOFFERED_TOOL_CALLS=",
    "REASONING_EFFORT": "--reasoning-effort",
}


def env_example() -> dict:
    """KEY=value lines of .env.example, without comments."""
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Z_][A-Z0-9_]*)=(\S*)", line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def start_sh() -> str:
    return (ROOT / "start.sh").read_text(encoding="utf-8")


class ProtectedDefaults(unittest.TestCase):
    def test_env_example_values(self):
        env = env_example()
        for key, want in PROTECTED.items():
            with self.subTest(key=key):
                self.assertEqual(env.get(key), want, f".env.example {key} changed. {HOW}")

    def test_start_sh_forwards_values(self):
        text = start_sh()
        for key, option in FORWARDED.items():
            with self.subTest(key=key):
                sep = "" if option.endswith("=") else r"\s+"
                opt = r"\s+".join(re.escape(word) for word in option.split())
                uses = re.findall(r"(?<![\w-])" + opt + sep + r"(\S+)", text)
                var = r"\$(?:" + key + r"\b|\{" + key + r"(?::-[^}]*)?\})"
                self.assertTrue(uses, f"start.sh does not pass {key} with {option}. {HOW}")
                self.assertEqual([u for u in uses if not re.fullmatch(var, u)], [],
                                 f"start.sh passes {option} without ${key}. {HOW}")

    def test_thinking_off(self):
        """Thinking is off unless THINKING=1: the only --thinking in start.sh is the THINKING=1 branch."""
        text = start_sh()
        switch = '$( [ "${THINKING:-0}" = 1 ] && echo --thinking || echo --no-thinking )'
        self.assertIn(switch, text, f"start.sh must start the server with thinking off unless THINKING=1. {HOW}")
        self.assertIsNone(re.search(r"(?<!no-)--thinking\b", text.replace(switch, "")),
                          f"start.sh turns thinking on. {HOW}")
        self.assertEqual(env_example().get("THINKING"), "0", f".env.example turns thinking on. {HOW}")

    def test_prefill_chunk(self):
        m = re.search(r"TF_DS_PREFILL_CHUNK=(?:\$\{PREFILL_CHUNK:-)?(\d+)", start_sh())
        self.assertIsNotNone(m, "start.sh does not set TF_DS_PREFILL_CHUNK")
        self.assertEqual(m.group(1), PREFILL_CHUNK, f"start.sh prefill chunk changed. {HOW}")
        env = env_example()
        if "PREFILL_CHUNK" in env:
            self.assertEqual(env["PREFILL_CHUNK"], PREFILL_CHUNK, f".env.example PREFILL_CHUNK changed. {HOW}")

    def test_image_matches_commit(self):
        env = env_example()
        self.assertEqual(env["IMAGE"].rsplit(":", 1)[-1], env["TF_COMMIT"],
                         "IMAGE and TF_COMMIT in .env.example name different engine builds")
        self.assertIn(env["TF_COMMIT"], (ROOT / "README.md").read_text(encoding="utf-8"),
                      "README.md does not name the engine build of .env.example")

    def test_zig_image_matches_commit(self):
        """The Zig image tag is the first 8 characters of the full fork commit, and the README names it."""
        env = env_example()
        self.assertEqual(env["ZIG_IMAGE"].rsplit(":", 1)[-1], env["ZIG_TF_COMMIT"][:8],
                         "ZIG_IMAGE and ZIG_TF_COMMIT in .env.example name different engine builds")
        self.assertRegex(env["ZIG_TF_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertIn(env["ZIG_TF_COMMIT"][:8], (ROOT / "README.md").read_text(encoding="utf-8"),
                      "README.md does not name the Zig engine build of .env.example")

    def test_start_sh_dispatches_on_engine(self):
        """start.sh hands over to the Zig start for ENGINE=zig, and keeps the Python start for ENGINE=python."""
        text = start_sh()
        self.assertIn('case "${ENGINE:-zig}" in', text, f"start.sh must read ENGINE with the fallback zig. {HOW}")
        self.assertIn('zig) exec "$ROOT/scripts/zig_start.sh"', text)


class FallbacksMatchEnvExample(unittest.TestCase):
    """A ${VAR:-x} fallback in start.sh must equal the .env.example value of VAR. Else a user whose .env has no
    VAR line gets a different server from a user who copied .env.example."""

    def test_fallbacks(self):
        env = env_example()
        found = re.findall(r"\$\{([A-Z_][A-Z0-9_]*):?-([^}]*)\}", start_sh())
        self.assertTrue(found, "no ${VAR:-default} fallback found in start.sh")
        for key, fallback in sorted(set(found)):
            if key not in env or env[key] == "":
                continue
            with self.subTest(key=key):
                self.assertEqual(fallback, env[key], f"start.sh ${{{key}:-{fallback}}} differs from .env.example "
                                                     f"{key}={env[key]}. {HOW}")


if __name__ == "__main__":
    unittest.main()
