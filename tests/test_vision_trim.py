"""The vision trim keeps the newest images and leaves everything else alone.

Run: python3 -m unittest discover -s tests -p 'test_vision_trim.py'
"""

import importlib.util
import json
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vision_trim.py"
_SPEC = importlib.util.spec_from_file_location("vision_trim", _PATH)
assert _SPEC and _SPEC.loader
trim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trim)


def _img(n: int) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{n}"}}


def _user(parts: list) -> dict:
    return {"role": "user", "content": parts}


class TrimMessages(unittest.TestCase):
    def test_under_the_cap_is_unchanged(self):
        messages = [_user([{"type": "text", "text": "hi"}, _img(1)]), _user([_img(2)])]
        before = json.dumps(messages)
        self.assertEqual(trim.trim_messages(messages, 12), 0)
        self.assertEqual(json.dumps(messages), before)

    def test_drops_the_oldest_and_notes_that_message(self):
        messages = [
            _user([_img(1), {"type": "text", "text": "first"}]),
            _user([_img(2), _img(3)]),
        ]
        self.assertEqual(trim.trim_messages(messages, 2), 1)
        self.assertEqual(messages[0]["content"], [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "[1 earlier image omitted]"},
        ])
        self.assertEqual(messages[1]["content"], [_img(2), _img(3)])

    def test_one_note_per_message_when_several_images_go(self):
        messages = [_user([_img(1), _img(2), _img(3)]), _user([_img(4)])]
        self.assertEqual(trim.trim_messages(messages, 1), 3)
        self.assertEqual(messages[0]["content"], [
            {"type": "text", "text": "[3 earlier images omitted]"},
        ])
        self.assertEqual(messages[1]["content"], [_img(4)])

    def test_string_content_is_not_an_image(self):
        messages = [{"role": "user", "content": "just text"}, _user([_img(1)])]
        self.assertEqual(trim.trim_messages(messages, 1), 0)
        self.assertEqual(messages[0]["content"], "just text")

    def test_anthropic_image_parts_count(self):
        messages = [_user([{"type": "image", "source": {"type": "base64"}}]), _user([_img(2)])]
        self.assertEqual(trim.trim_messages(messages, 1), 1)
        self.assertEqual(messages[1]["content"], [_img(2)])

    def test_body_that_is_not_json_passes_through(self):
        raw, n = trim.trim_body(b"not-json", 1)
        self.assertEqual(raw, b"not-json")
        self.assertEqual(n, 0)

    def test_body_without_messages_passes_through(self):
        raw = b'{"model":"x","prompt":"hi"}'
        out, n = trim.trim_body(raw, 1)
        self.assertEqual(out, raw)
        self.assertEqual(n, 0)

    def test_body_round_trip_drops_one(self):
        body = {"model": "waverider-local", "messages": [_user([_img(1)]), _user([_img(2)])]}
        out, n = trim.trim_body(json.dumps(body).encode(), 1)
        self.assertEqual(n, 1)
        parsed = json.loads(out)
        self.assertEqual(parsed["messages"][1]["content"], [_img(2)])
        self.assertEqual(parsed["model"], "waverider-local")


if __name__ == "__main__":
    unittest.main()
