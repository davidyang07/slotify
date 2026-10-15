"""Shared fixtures for the ``ad_inserter`` tests.

Run from the repository root: ``python -m pytest backend/tests/python -q``.
"""

from __future__ import annotations

import shutil
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest

# backend/tests/python/conftest.py -> backend/, so `import ad_inserter` resolves.
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "ffmpeg: needs ffmpeg/ffprobe on PATH (MP3 encode/decode)"
    )


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    if HAS_FFMPEG:
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if item.get_closest_marker("ffmpeg") is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's credentials and voice settings out of every test."""
    for name in (
        "SLOTIFY_INTERNAL_TOKEN",
        "SLOTIFY_API_TOKEN",
        "SLOTIFY_CALLBACK_TIMEOUT_S",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_DEFAULT_VOICE_ID",
        "ELEVENLABS_VOICE_ID_A",
        "ELEVENLABS_VOICE_ID_B",
        "SLOTIFY_OPENAI_MODELS",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # The fake API below is plain HTTP on loopback; never route it via a proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes


Responder = Callable[[RecordedRequest], Tuple[int, bytes]]


@dataclass
class FakeApi:
    url: str
    requests: List[RecordedRequest] = field(default_factory=list)
    responder: Optional[Responder] = None

    def respond_with(self, responder: Responder) -> None:
        self.responder = responder


@pytest.fixture
def fake_api() -> FakeApi:
    """A loopback HTTP server that records requests; 200 + empty body by default."""

    api = FakeApi(url="")

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            recorded = RecordedRequest(
                method=self.command,
                path=self.path,
                headers={key.lower(): value for key, value in self.headers.items()},
                body=body,
            )
            api.requests.append(recorded)
            status, payload = (200, b"")
            if api.responder is not None:
                status, payload = api.responder(recorded)
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = do_DELETE = _handle

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.block_on_close = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()
