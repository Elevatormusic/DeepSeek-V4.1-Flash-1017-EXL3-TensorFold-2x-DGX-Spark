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
    # The engine build. A change here must come with new gate results (see CONTRIBUTING.md).
    "TF_COMMIT": "07aa02585f64",
    "IMAGE": "tensorfold-dsv41:07aa02585f64",
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
        text = start_sh()
        self.assertIn("--no-thinking", text, f"start.sh must start the server with thinking off. {HOW}")
        self.assertIsNone(re.search(r"(?<!no-)--thinking\b", text), f"start.sh turns thinking on. {HOW}")
        env = env_example()
        if "THINKING" in env:
            self.assertEqual(env["THINKING"], "0", f".env.example turns thinking on. {HOW}")

    def test_prefill_chunk(self):
        m = re.search(r"TF_DS_PREFILL_CHUNK=(?:\$\{TF_DS_PREFILL_CHUNK:-)?(\d+)", start_sh())
        self.assertIsNotNone(m, "start.sh does not set TF_DS_PREFILL_CHUNK")
        self.assertEqual(m.group(1), PREFILL_CHUNK, f"start.sh prefill chunk changed. {HOW}")
        env = env_example()
        if "TF_DS_PREFILL_CHUNK" in env:
            self.assertEqual(env["TF_DS_PREFILL_CHUNK"], PREFILL_CHUNK,
                             f".env.example TF_DS_PREFILL_CHUNK changed. {HOW}")

    def test_image_matches_commit(self):
        env = env_example()
        self.assertEqual(env["IMAGE"].rsplit(":", 1)[-1], env["TF_COMMIT"],
                         "IMAGE and TF_COMMIT in .env.example name different engine builds")
        self.assertIn(env["TF_COMMIT"], (ROOT / "README.md").read_text(encoding="utf-8"),
                      "README.md does not name the engine build of .env.example")


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
