"""Offline tests for the speed prompts of v41_suite.py: no server, no GPU.
They check that every speed prompt is unique, has no shared prefix and no repeated sentence, and that a
prefix-cache hit fails a prefill row."""
import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v41_suite  # noqa: E402


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class PromptTests(unittest.TestCase):
    def test_salted_prompts_have_no_shared_prefix(self):
        a, b = v41_suite.salted("same text"), v41_suite.salted("same text")
        self.assertNotEqual(a, b)
        # "Run " and the line end are the only shared characters; the unique part is 32 hex characters.
        self.assertLess(common_prefix(a, b), len("Run ") + 8)
        self.assertRegex(a, r"^Run [0-9a-f]{32}\.\nsame text$")

    def test_varied_prompt_has_no_repeated_sentence(self):
        text = v41_suite.varied_prompt(8000, random.Random(1))
        body = text.split("\n", 1)[1]
        sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", body)
        self.assertEqual(len(sentences), len(set(sentences)))
        self.assertGreaterEqual(len(body), 8000 * 4.7)

    def test_varied_prompt_32k_fits_in_the_pool(self):
        text = v41_suite.varied_prompt(32000, random.Random(2))
        self.assertGreaterEqual(len(text), 32000 * 4.7)

    def test_8k_and_32k_prompts_share_no_prefix(self):
        small = v41_suite.varied_prompt(8000)
        large = v41_suite.varied_prompt(32000)
        # A whole chunk of 2,048 tokens is about 9,600 characters. The shared part must stay far under it.
        self.assertLess(common_prefix(small, large), 200)

    def test_pool_too_small_raises(self):
        with self.assertRaises(RuntimeError):
            v41_suite.varied_prompt(10_000_000)


class PrefillRowTests(unittest.TestCase):
    def test_cold_row_passes(self):
        usage = {"prompt_tokens": 32000, "prompt_tokens_details": {"cached_tokens": 0}}
        ok, info = v41_suite.prefill_row(usage, 16.0)
        self.assertTrue(ok)
        self.assertEqual(info["cached_tokens"], 0)
        self.assertEqual(info["prefill_tok_s"], 2000)

    def test_cache_hit_fails_the_row_and_counts_only_new_tokens(self):
        usage = {"prompt_tokens": 32001, "prompt_tokens_details": {"cached_tokens": 6144}}
        ok, info = v41_suite.prefill_row(usage, 11.66)
        self.assertFalse(ok)
        self.assertEqual(info["cached_tokens"], 6144)
        self.assertEqual(info["prefill_tok_s"], round((32001 - 6144) / 11.66))
        self.assertIn("prefix-cache hit", info["detail"])

    def test_unreported_cached_tokens_is_logged_not_failed(self):
        for usage in ({"prompt_tokens": 100}, {"prompt_tokens": 100, "prompt_tokens_details": None},
                      {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": None}}):
            ok, info = v41_suite.prefill_row(usage, 1.0)
            self.assertTrue(ok)
            self.assertIsNone(info["cached_tokens"])
            self.assertIn("note", info)

    def test_no_first_token_fails_the_row(self):
        ok, info = v41_suite.prefill_row({"prompt_tokens": 100}, None)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
