#!/usr/bin/env python3
"""Keep the newest images, then forward requests to the local TensorFold API."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import re
import select
import socket
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

IMAGE_TYPES = {"image_url", "image", "input_image"}
TRIM_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/responses"}
MAX_BODY = 32 * 1024 * 1024
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}


def _note(n: int) -> str:
    return f"[{n} earlier image omitted]" if n == 1 else f"[{n} earlier images omitted]"


def _image_slots(content: list):
    """Find image locations, including tool results, in request order."""
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if isinstance(kind, str) and kind in IMAGE_TYPES:
            yield content, index
        elif kind == "tool_result" and isinstance(part.get("content"), list):
            yield from _image_slots(part["content"])


def trim_messages(messages: list, keep: int) -> int:
    """Replace the oldest images with one text note in each content list."""
    if keep < 0:
        raise ValueError("keep must be >= 0")
    slots = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", message.get("output"))
            if isinstance(content, list):
                slots.extend(_image_slots(content))
    dropped = max(0, len(slots) - keep)
    groups = {}
    for content, index in slots[:dropped]:
        groups.setdefault(id(content), (content, set()))[1].add(index)
    for content, indices in groups.values():
        responses = any(isinstance(p, dict) and p.get("type") == "input_image" for p in content)
        kept = [p for index, p in enumerate(content) if index not in indices]
        kept.append({"type": "input_text" if responses else "text", "text": _note(len(indices))})
        content[:] = kept
    return dropped


def trim_body(raw: bytes, keep: int) -> tuple[bytes, int]:
    """Trim chat, Messages, or Responses images. Leave other bodies unchanged."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError):
        return raw, 0
    messages = obj.get("messages", obj.get("input")) if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return raw, 0
    dropped = trim_messages(messages, keep)
    if not dropped:
        return raw, 0
    return json.dumps(obj, separators=(",", ":")).encode(), dropped


def _keep() -> int:
    raw = os.environ.get("VISION_MAX_IMAGES", "")
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) < 1:
        raise ValueError("set VISION_MAX_IMAGES to a positive integer")
    return int(raw)


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _keys(path: str) -> tuple[bytes, ...]:
    """Read key digests. Refuse links, unsafe file modes, and empty files."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 1 << 20:
            raise ValueError("API_KEY_FILE must be a regular file with mode 600 and at most 1 MiB")
        with os.fdopen(fd, encoding="utf-8", closefd=False) as stream:
            text = stream.read((1 << 20) + 1)
        keys = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label, separator, key = line.partition(":")
            if separator and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", label.strip()):
                raise ValueError("invalid API key label")
            key = key.strip() if separator else line
            if not key or any(c.isspace() or ord(c) < 32 for c in key):
                raise ValueError("API keys must be nonempty text without spaces")
            keys.append(hashlib.sha256(key.encode()).digest())
        if not keys:
            raise ValueError("API_KEY_FILE contains no keys")
        return tuple(keys)
    finally:
        os.close(fd)


def _hop(headers) -> set[str]:
    """Remove fixed and Connection-named hop headers."""
    return HOP | {v.strip().lower() for line in headers.get_all("Connection", []) for v in line.split(",")}


class DisconnectWatch:
    """Stop a local upstream request when its client closes the connection."""

    def __init__(self, client, upstream):
        self.client = client
        self.upstream = upstream
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        try:
            while not self.done.is_set():
                readable, _, _ = select.select([self.client], [], [], 0.25)
                if readable:
                    if not self.client.recv(1, socket.MSG_PEEK):
                        break
                    self.done.wait(0.25)
            else:
                return
        except OSError:
            pass
        try:
            self.upstream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # main checks these values before it opens the socket.
    upstream_port = 8888
    keep = 1
    key_file = ""
    timeout = 30

    def do_GET(self):
        self._proxy()

    def do_HEAD(self):
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
        # Do not record paths, queries, headers, or bodies.
        return

    def _error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.key_file:
            return True
        try:
            keys = _keys(self.key_file)
        except (OSError, UnicodeError, ValueError):
            return False
        candidates = []
        for name in ("Authorization", "x-api-key"):
            values = self.headers.get_all(name, [])
            value = values[0] if len(values) == 1 else ""
            if name == "Authorization":
                scheme, _, value = value.partition(" ")
                value = value.strip() if scheme.lower() == "bearer" else ""
            candidates.append(hashlib.sha256(value.encode()).digest())
        return any(hmac.compare_digest(key, value) for key in keys for value in candidates)

    def _body(self) -> bytes | None:
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding") or len(lengths) > 1:
            self._error(400, "use one Content-Length header; chunked request bodies are not supported")
            return None
        if lengths and not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            self._error(400, "invalid Content-Length")
            return None
        length = int(lengths[0]) if lengths else 0
        if length > MAX_BODY:
            self._error(413, "request body exceeds 32 MiB")
            return None
        if self.headers.get("Content-Encoding", "identity").lower() != "identity":
            self._error(415, "send an uncompressed request body")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._error(400, "incomplete request body")
            return None
        return body

    def _proxy(self) -> None:
        # Accept one request per connection. Close after the response.
        self.close_connection = True
        if not self.path.startswith("/") or self.path.startswith("//") or "#" in self.path:
            self._error(400, "use a relative API path")
            return
        if not self._authorized():
            self._error(401, "valid API key required")
            return
        try:
            body = self._body()
        except (OSError, ValueError):
            self._error(400, "cannot read request body")
            return
        if body is None:
            return
        dropped = 0
        if self.command == "POST" and self.path.split("?", 1)[0].rstrip("/") in TRIM_PATHS and body:
            body, dropped = trim_body(body, self.keep)
        hop = _hop(self.headers)
        headers = {key: value for key, value in self.headers.items() if key.lower() not in hop}
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        conn = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=3600)
        resp = None
        sent = False
        try:
            conn.connect()
            with DisconnectWatch(self.connection, conn.sock):
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                hop = _hop(resp.headers)
                for key, value in resp.headers.items():
                    if key.lower() not in hop:
                        self.send_header(key, value)
                if dropped:
                    self.send_header("X-Vision-Trimmed", str(dropped))
                no_body = self.command == "HEAD" or resp.status in (204, 304) or resp.status < 200
                if not no_body:
                    self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                sent = True
                if not no_body:
                    while chunk := resp.read1(65536):
                        self.wfile.write(f"{len(chunk):X}\r\n".encode())
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    if resp.length not in (None, 0):
                        raise http.client.IncompleteRead(b"")
                    # Only a complete upstream response gets a terminal chunk.
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
        except (OSError, http.client.HTTPException, ValueError):
            if not sent:
                self._error(502, "upstream unavailable")
        finally:
            if resp is not None:
                resp.close()
            conn.close()


class TrimServer(ThreadingHTTPServer):
    """Limit active request threads and body memory."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(4)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        self.slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main() -> None:
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    upstream = urlsplit(os.environ.get("VISION_TRIM_UPSTREAM", "http://127.0.0.1:8888"))
    if (upstream.scheme != "http" or upstream.hostname != "127.0.0.1" or upstream.username
            or upstream.password or upstream.path or upstream.query or upstream.fragment):
        raise ValueError("VISION_TRIM_UPSTREAM must be http://127.0.0.1:PORT")
    Handler.upstream_port = upstream.port or 80
    if not 1 <= port <= 65535 or port == Handler.upstream_port:
        raise ValueError("proxy and upstream ports must be valid and different")
    Handler.keep = _keep()
    Handler.key_file = os.environ.get("API_KEY_FILE", "")
    if Handler.key_file:
        _keys(Handler.key_file)
    elif not _loopback(host):
        raise ValueError("a LAN bind requires API_KEY_FILE")
    with TrimServer((host, port), Handler) as httpd:
        print(f"vision-trim listening on {host}:{port}; keep={Handler.keep}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
