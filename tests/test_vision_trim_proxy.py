"""Check proxy security, framing, and streaming with in-memory HTTP stubs only."""

import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vision_trim.py"
_SPEC = importlib.util.spec_from_file_location("vision_trim_proxy", _PATH)
trim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trim)


class SocketStub:
    def __init__(self, data=b""):
        self.input = io.BytesIO(data)
        self.output = io.BytesIO()

    def makefile(self, mode, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)

    def settimeout(self, timeout):
        pass


class Proxy(unittest.TestCase):
    def setUp(self):
        self.old = (trim.Handler.keep, trim.Handler.key_file, trim.Handler.upstream_port)
        trim.Handler.keep = 1
        trim.Handler.key_file = ""
        trim.Handler.upstream_port = 8888

    def tearDown(self):
        trim.Handler.keep, trim.Handler.key_file, trim.Handler.upstream_port = self.old

    def invoke(self, raw, upstream=b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", check_stream=False):
        sock = SocketStub(raw)
        response = http.client.HTTPResponse(SocketStub(upstream))
        response.begin()
        conn = mock.Mock()
        conn.getresponse.return_value = response
        if check_stream:
            read1 = response.read1
            calls = []

            def next_chunk(size):
                if calls:
                    self.assertIn(b"data: first\n\n", sock.output.getvalue())
                calls.append(size)
                return read1(size)

            response.read1 = next_chunk
        with mock.patch.object(trim.http.client, "HTTPConnection", return_value=conn) as factory:
            with mock.patch.object(trim, "DisconnectWatch", return_value=contextlib.nullcontext()):
                trim.Handler(sock, ("127.0.0.1", 1), None)
        return sock.output.getvalue(), conn, factory

    def test_sse_reaches_client_before_the_next_read(self):
        first, second = b"data: first\n\n", b"data: second\n\n"
        upstream = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                    + f"{len(first):x}\r\n".encode() + first + b"\r\n"
                    + f"{len(second):x}\r\n".encode() + second + b"\r\n0\r\n\r\n")
        output, conn, _ = self.invoke(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
                                      upstream, check_stream=True)
        self.assertTrue(output.endswith(first + second))
        self.assertNotIn(b"Transfer-Encoding:", output)
        self.assertIn(b"Connection: close", output)
        conn.close.assert_called_once()

    def test_trims_query_route_and_rebuilds_headers(self):
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "old"}},
            {"type": "image_url", "image_url": {"url": "new"}},
        ]}]}).encode()
        raw = (f"POST /v1/chat/completions?test=1 HTTP/1.1\r\nHost: attacker\r\nContent-Length: {len(body)}\r\n"
               "Connection: X-Private\r\nX-Private: remove\r\nAuthorization: Bearer pass\r\n\r\n").encode() + body
        output, conn, factory = self.invoke(raw)
        factory.assert_called_once_with("127.0.0.1", 8888, timeout=3600)
        args, kwargs = conn.request.call_args
        self.assertEqual(args, ("POST", "/v1/chat/completions?test=1"))
        self.assertEqual(kwargs["headers"]["Content-Length"], str(len(kwargs["body"])))
        self.assertNotIn("X-Private", kwargs["headers"])
        self.assertNotIn("Host", kwargs["headers"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer pass")
        self.assertIn(b"X-Vision-Trimmed: 1", output)

    def test_bad_framing_never_reaches_upstream(self):
        cases = [
            (b"Content-Length: -1", 400),
            (b"Content-Length: nope", 400),
            (b"Content-Length: 0\r\nContent-Length: 0", 400),
            (b"Content-Length: 0\r\nTransfer-Encoding: chunked", 400),
            (b"Transfer-Encoding: chunked", 400),
            (b"Content-Length: 33554433", 413),
            (b"Content-Encoding: gzip\r\nContent-Length: 0", 415),
            (b"Content-Length: 10", 400),
        ]
        for headers, status in cases:
            with self.subTest(headers=headers):
                output, _, factory = self.invoke(b"POST /v1/chat/completions HTTP/1.1\r\n" + headers + b"\r\n\r\n")
                self.assertIn(f"HTTP/1.1 {status}".encode(), output)
                factory.assert_not_called()

    def test_head_and_no_content_have_no_body_or_transfer_coding(self):
        cases = [(b"HEAD", b"200 OK"), (b"GET", b"204 No Content"), (b"GET", b"304 Not Modified")]
        for method, status in cases:
            with self.subTest(status=status):
                output, conn, _ = self.invoke(method + b" /health HTTP/1.1\r\n\r\n",
                                              b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\n\r\n")
                self.assertEqual(output.split(b"\r\n\r\n", 1)[1], b"")
                self.assertNotIn(b"Transfer-Encoding", output)
                conn.close.assert_called_once()

    def test_upstream_errors_and_redirects_are_forwarded_without_following(self):
        for status in (b"307 Temporary Redirect", b"400 Bad Request"):
            with self.subTest(status=status):
                output, conn, _ = self.invoke(b"GET /health HTTP/1.1\r\n\r\n",
                                              b"HTTP/1.1 " + status + b"\r\nLocation: http://example.invalid/\r\n"
                                              b"Connection: X-Secret\r\nX-Secret: remove\r\nContent-Length: 2\r\n\r\n{}")
                self.assertIn(status, output)
                self.assertNotIn(b"X-Secret:", output)
                self.assertEqual(conn.request.call_count, 1)

    def test_auth_is_required_and_file_failure_closes_access(self):
        trim.Handler.key_file = "stub"
        digest = hashlib.sha256(b"test-value").digest()
        for headers, valid in [(b"", False), (b"Authorization: Bearer wrong\r\n", False),
                               (b"Authorization: Bearer test-value\r\n", True),
                               (b"x-api-key: test-value\r\n", True),
                               (b"x-api-key: test-value\r\nx-api-key: test-value\r\n", False)]:
            with self.subTest(headers=headers), mock.patch.object(trim, "_keys", return_value=(digest,)):
                output, _, factory = self.invoke(b"GET /health HTTP/1.1\r\n" + headers + b"\r\n")
                self.assertIn(b"200 OK" if valid else b"401 Unauthorized", output)
                self.assertEqual(factory.called, valid)
        with mock.patch.object(trim, "_keys", side_effect=OSError):
            output, _, factory = self.invoke(b"GET /health HTTP/1.1\r\nx-api-key: test-value\r\n\r\n")
            self.assertIn(b"401 Unauthorized", output)
            factory.assert_not_called()

    def test_upstream_connect_failure_is_closed(self):
        sock = SocketStub(b"GET /health HTTP/1.1\r\n\r\n")
        conn = mock.Mock()
        conn.connect.side_effect = OSError
        with mock.patch.object(trim.http.client, "HTTPConnection", return_value=conn):
            trim.Handler(sock, ("127.0.0.1", 1), None)
        self.assertIn(b"502 Bad Gateway", sock.output.getvalue())
        conn.close.assert_called_once()

    def test_client_write_failure_closes_response_and_connection(self):
        sock = SocketStub(b"GET /health HTTP/1.1\r\n\r\n")
        response = mock.Mock(status=200)
        response.headers = http.client.parse_headers(io.BytesIO(b"Content-Type: text/event-stream\r\n\r\n"))
        response.read1.return_value = b"event"
        conn = mock.Mock()
        conn.getresponse.return_value = response
        original = sock.sendall

        def fail_body(data):
            if data == b"event":
                raise BrokenPipeError
            original(data)

        sock.sendall = fail_body
        with mock.patch.object(trim.http.client, "HTTPConnection", return_value=conn):
            with mock.patch.object(trim, "DisconnectWatch", return_value=contextlib.nullcontext()):
                trim.Handler(sock, ("127.0.0.1", 1), None)
        response.close.assert_called_once()
        conn.close.assert_called_once()


class Configuration(unittest.TestCase):
    def test_launcher_refuses_bad_caps_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "start.sh").write_bytes((_PATH.parent.parent / "start.sh").read_bytes())
            (root / "scripts/common.sh").write_text('ROOT="$(dirname "$0")"\n', encoding="utf-8")
            preflight = root / "scripts/preflight.sh"
            preflight.write_text('#!/bin/sh\necho reached-preflight\nexit 1\n', encoding="utf-8")
            preflight.chmod(0o700)
            for value in ("0", "-1", "1 --host 0.0.0.0", "1; false", "two"):
                with self.subTest(value=value):
                    env = dict(os.environ, VISION_MAX_IMAGES=value)
                    result = subprocess.run(["bash", str(root / "start.sh")], env=env,
                                            capture_output=True, text=True, timeout=5)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("VISION_MAX_IMAGES must be a positive integer", result.stderr)
                    self.assertNotIn("reached-preflight", result.stdout)
            for value in ("", "1", "12"):
                with self.subTest(value=value):
                    result = subprocess.run(["bash", str(root / "start.sh")],
                                            env=dict(os.environ, VISION_MAX_IMAGES=value),
                                            capture_output=True, text=True, timeout=5)
                    self.assertIn("reached-preflight", result.stdout)

    def test_loopback_default_and_lan_rejection_happen_before_bind(self):
        with mock.patch.dict(os.environ, {"VISION_MAX_IMAGES": "2"}, clear=True):
            with mock.patch.object(trim, "TrimServer") as server:
                server.return_value.__enter__.return_value.serve_forever.return_value = None
                trim.main()
                self.assertEqual(server.call_args.args[0], ("127.0.0.1", 8001))
        with mock.patch.dict(os.environ, {"VISION_MAX_IMAGES": "2", "BIND_HOST": "0.0.0.0"}, clear=True):
            with mock.patch.object(trim, "TrimServer") as server:
                with self.assertRaisesRegex(ValueError, "API_KEY_FILE"):
                    trim.main()
                server.assert_not_called()

    def test_invalid_upstream_and_cap_do_not_bind(self):
        for values in ({"VISION_MAX_IMAGES": "-1"}, {"VISION_MAX_IMAGES": "1 --host 0.0.0.0"},
                       {"VISION_MAX_IMAGES": "1", "VISION_TRIM_UPSTREAM": "http://example.invalid:8888"},
                       {"VISION_MAX_IMAGES": "1", "VISION_TRIM_UPSTREAM": "http://127.0.0.1:8001"}):
            with self.subTest(values=values), mock.patch.dict(os.environ, values, clear=True):
                with mock.patch.object(trim, "TrimServer") as server:
                    with self.assertRaises(ValueError):
                        trim.main()
                    server.assert_not_called()

    def test_key_file_rejects_empty_unsafe_and_symlink_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys"
            path.write_text("# comment\n", encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                trim._keys(str(path))
            path.write_text("label: test-value\n", encoding="utf-8")
            self.assertEqual(trim._keys(str(path)), (hashlib.sha256(b"test-value").digest(),))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                trim._keys(str(path))
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                trim._keys(str(link))

    def test_disconnect_stops_a_stalled_upstream(self):
        client, upstream = mock.Mock(), mock.Mock()
        client.recv.return_value = b""
        watcher = trim.DisconnectWatch(client, upstream)
        with mock.patch.object(trim.select, "select", return_value=([client], [], [])):
            watcher._watch()
        upstream.shutdown.assert_called_once_with(trim.socket.SHUT_RDWR)


class BodyFormats(unittest.TestCase):
    def test_nested_and_direct_images_follow_content_order(self):
        old = {"type": "image", "source": {"data": "old"}}
        new = {"type": "image", "source": {"data": "new"}}
        nested = {"type": "tool_result", "tool_use_id": "call", "content": [old]}
        messages = [{"role": "user", "content": [nested, new]}]
        self.assertEqual(trim.trim_messages(messages, 1), 1)
        self.assertEqual(messages[0]["content"][1], new)
        self.assertEqual(nested["content"], [{"type": "text", "text": "[1 earlier image omitted]"}])

    def test_invalid_part_type_passes_through(self):
        messages = [{"role": "user", "content": [{"type": []}]}]
        self.assertEqual(trim.trim_messages(messages, 1), 0)

    def test_responses_input_and_tool_output_use_input_text_notes(self):
        old = {"type": "input_image", "image_url": "old"}
        new = {"type": "input_image", "image_url": "new"}
        body = {"input": [{"type": "function_call_output", "call_id": "call", "output": [old]},
                          {"role": "user", "content": [new]}]}
        out, count = trim.trim_body(json.dumps(body).encode(), 1)
        self.assertEqual(count, 1)
        result = json.loads(out)
        self.assertEqual(result["input"][0]["output"], [{"type": "input_text", "text": "[1 earlier image omitted]"}])
        self.assertEqual(result["input"][1]["content"], [new])

    def test_nested_anthropic_tool_result_images_are_trimmed(self):
        old = {"type": "image", "source": {"type": "base64", "data": "old"}}
        new = {"type": "image", "source": {"type": "base64", "data": "new"}}
        messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call", "content": [old]}]},
                    {"role": "user", "content": [new]}]
        self.assertEqual(trim.trim_messages(messages, 1), 1)
        self.assertEqual(messages[0]["content"][0]["content"], [{"type": "text", "text": "[1 earlier image omitted]"}])
        self.assertEqual(messages[1]["content"], [new])

    def test_invalid_utf8_passes_to_upstream_unchanged(self):
        self.assertEqual(trim.trim_body(b"\xff", 1), (b"\xff", 0))


if __name__ == "__main__":
    unittest.main()
