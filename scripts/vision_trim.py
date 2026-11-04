#!/usr/bin/env python3
"""Drop image parts older than the newest N, then forward the request to TensorFold.

T3 resends every screenshot in the thread. DeepSeek V4.1 rejects the request once
the count passes --vision-max-images, and the head runs out of memory well before
a large count is useful. This proxy keeps the newest N images (N is VISION_MAX_IMAGES)
and replaces the dropped ones in each message with a single text note. The engine
still enforces the same N, so a direct caller cannot exceed it.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

IMAGE_TYPES = {"image_url", "image"}
TRIM_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/responses"}
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}


def _note(n: int) -> str:
    return f"[{n} earlier image omitted]" if n == 1 else f"[{n} earlier images omitted]"


def trim_messages(messages: list, keep: int) -> int:
    """Drop the oldest image parts until ``keep`` remain. Returns how many were dropped."""

    if keep < 0:
        raise ValueError("keep must be >= 0")
    slots: list[tuple[int, int]] = []
    for mi, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in IMAGE_TYPES:
                slots.append((mi, pi))
    if len(slots) <= keep:
        return 0
    drop = set(slots[: len(slots) - keep])
    dropped = 0
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        n = 0
        for pi, part in enumerate(content):
            if (mi, pi) in drop:
                n += 1
                continue
            kept.append(part)
        if n:
            kept.append({"type": "text", "text": _note(n)})
            message["content"] = kept
            dropped += n
    return dropped


def trim_body(raw: bytes, keep: int) -> tuple[bytes, int]:
    """Trim a JSON request body. A body that is not a chat with messages is unchanged."""

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0
    messages = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return raw, 0
    dropped = trim_messages(messages, keep)
    if not dropped:
        return raw, 0
    return json.dumps(obj, separators=(",", ":")).encode(), dropped


def _keep() -> int:
    raw = os.environ.get("VISION_MAX_IMAGES", "12")
    try:
        n = int(raw)
    except ValueError as exc:
        raise SystemExit(f"VISION_MAX_IMAGES must be an integer, not {raw!r}") from exc
    if n < 1:
        raise SystemExit(f"VISION_MAX_IMAGES must be >= 1, not {n}")
    return n


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = os.environ.get("VISION_TRIM_UPSTREAM", "http://127.0.0.1:8000")
    keep = _keep()

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def log_message(self, fmt: str, *args) -> None:
        # Request line only. Bodies carry screenshots and the API key's neighbor headers.
        super().log_message(fmt, *args)

    def _proxy(self) -> None:
        length = self.headers.get("Content-Length")
        body = self.rfile.read(int(length)) if length is not None else b""
        dropped = 0
        path = self.path.split("?", 1)[0]
        if self.command == "POST" and path in TRIM_PATHS and body:
            body, dropped = trim_body(body, self.keep)
        req = request.Request(self.upstream + self.path, data=body if self.command != "GET" else None,
                              method=self.command)
        for key, value in self.headers.items():
            if key.lower() not in HOP:
                req.add_header(key, value)
        try:
            resp = request.urlopen(req, timeout=3600)
        except error.HTTPError as exc:
            resp = exc
        except (error.URLError, TimeoutError, OSError) as exc:
            message = json.dumps({"error": {"message": f"upstream unavailable: {exc}",
                                            "type": "upstream_error"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        try:
            self.send_response(resp.status)
            for key, value in resp.headers.items():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            if dropped:
                self.send_header("X-Vision-Trimmed", str(dropped))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            resp.close()


def main() -> None:
    host = os.environ.get("BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"vision-trim listening on {host}:{port} -> {Handler.upstream} keep={Handler.keep}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
