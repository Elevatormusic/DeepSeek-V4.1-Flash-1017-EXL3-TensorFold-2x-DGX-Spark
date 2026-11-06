"""Keep the trusted historical checksum exception narrow."""
from pathlib import Path
import re
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SecretScanPolicy(unittest.TestCase):
    def test_policy_is_inline_and_keeps_default_rules(self):
        workflow = (ROOT / ".github/workflows/secret-scan.yml").read_text()
        block = re.search(r"(?m)^          cat > gitleaks.toml <<'TOML'\n(.*?)^          TOML$",
                          workflow, re.S)
        self.assertIsNotNone(block)
        config = tomllib.loads("\n".join(line.removeprefix("          ") for line in block.group(1).splitlines()))
        self.assertEqual(config["extend"], {"useDefault": True})
        self.assertEqual(set(config), {"extend", "allowlists"})
        self.assertEqual(len(config["allowlists"]), 1)
        exception = config["allowlists"][0]
        self.assertEqual(exception["targetRules"], ["generic-api-key"])
        self.assertEqual(exception["condition"], "AND")
        self.assertEqual(exception["commits"], ["9910ec9b3b03eae999639a08098290e990c0db3e"])
        self.assertEqual(exception["paths"], [r"^(scripts/zig_inputs\.sh|tests/test_zig_engine\.py)$"])
        self.assertEqual(exception["regexTarget"], "secret")
        self.assertEqual(len(exception["regexes"]), 1)
        self.assertRegex(exception["regexes"][0], r"^\^[0-9a-f]{64}\$$")
        self.assertEqual(set(exception), {"description", "targetRules", "condition", "commits", "paths",
                                          "regexTarget", "regexes"})
        self.assertIn("pull_request_target:", workflow)
        self.assertIn('--config "$RUNNER_TEMP/gitleaks.toml"', workflow)
        self.assertIn('--log-opts="--diff-merges=first-parent HEAD"', workflow)
        self.assertIn("--ignore-gitleaks-allow", workflow)
        self.assertIn("if [ -e scan/.gitleaksignore ]; then", workflow)
        self.assertNotIn("--config scan/", workflow)
        self.assertNotIn("source scan/", workflow)

    def test_exact_checksum_is_fixed(self):
        workflow = (ROOT / ".github/workflows/secret-scan.yml").read_text()
        expected = "de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28"
        self.assertIn("^" + expected + "$", workflow)
