"""http_client.py and its callers: auth header, timeout, multipart fields."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import cli, http_client, tts

AD_INSERTER_DIR = Path(http_client.__file__).resolve().parent


def test_auth_header_prefers_the_internal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    assert http_client.auth_headers() == {}

    monkeypatch.setenv("SLOTIFY_API_TOKEN", "deploy-token")
    assert http_client.auth_headers() == {"Authorization": "Bearer deploy-token"}

    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    assert http_client.auth_headers() == {"Authorization": "Bearer internal-token"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 300.0), ("12.5", 12.5), ("abc", 300.0), ("0", 300.0), ("-4", 300.0), ("inf", 300.0)],
)
def test_callback_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is not None:
        monkeypatch.setenv("SLOTIFY_CALLBACK_TIMEOUT_S", raw)
    assert http_client.callback_timeout_s() == expected


def test_send_attaches_the_bearer_token(fake_api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    fake_api.respond_with(lambda _req: (200, b"ok"))

    body = http_client.post_json(f"{fake_api.url}/api/tts", {"text": "hi"}, label="TTS")

    assert body == b"ok"
    (request,) = fake_api.requests
    assert request.headers["authorization"] == "Bearer internal-token"
    assert json.loads(request.body) == {"text": "hi"}


def test_send_without_a_token_sends_no_authorization(fake_api) -> None:
    http_client.post_json(f"{fake_api.url}/api/tts", {}, label="TTS")

    assert "authorization" not in fake_api.requests[0].headers


def test_third_party_requests_never_carry_the_internal_token(
    fake_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")

    http_client.send(
        f"{fake_api.url}/v1/voices/abc",
        label="Voice delete",
        headers={"xi-api-key": "k"},
        method="DELETE",
        internal_auth=False,
    )

    (request,) = fake_api.requests
    assert request.method == "DELETE"
    assert "authorization" not in request.headers
    assert request.headers["xi-api-key"] == "k"


def test_error_status_becomes_runtime_error(fake_api) -> None:
    fake_api.respond_with(lambda _req: (401, b"unauthorized"))

    with pytest.raises(RuntimeError, match=r"Merge request failed with status 401: unauthorized"):
        http_client.post_multipart(f"{fake_api.url}/api/merge", {}, [], label="Merge")


def test_slow_callback_times_out(fake_api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOTIFY_CALLBACK_TIMEOUT_S", "0.3")

    def slow(_req):
        time.sleep(1.5)
        return 200, b"late"

    fake_api.respond_with(slow)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"TTS request (timed out|failed)"):
        http_client.post_json(f"{fake_api.url}/api/tts", {}, label="TTS")
    assert time.monotonic() - started < 1.4


def test_urlopen_is_only_called_through_the_shared_helper() -> None:
    offenders = [
        path.name
        for path in AD_INSERTER_DIR.glob("*.py")
        if path.name != "http_client.py" and re.search(r"\burlopen\b", path.read_text())
    ]
    assert offenders == []
    source = (AD_INSERTER_DIR / "http_client.py").read_text()
    assert all("timeout=" in call for call in re.findall(r"urlopen\([^)]*\)", source))


@pytest.mark.ffmpeg
def test_tts_posts_json_with_auth(fake_api, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    mp3_path = tmp_path / "speech.mp3"
    Sine(300).to_audio_segment(duration=1_000).export(mp3_path, format="mp3")
    fake_api.respond_with(lambda _req: (200, mp3_path.read_bytes()))

    audio = tts.synthesize_audio(
        tts.TTSRequest(voice_id="voice-1", text="Hello", url=f"{fake_api.url}/api/tts")
    )

    assert 900 <= len(audio) <= 1_200
    (request,) = fake_api.requests
    assert request.headers["authorization"] == "Bearer internal-token"
    assert json.loads(request.body) == {"voiceId": "voice-1", "text": "Hello"}


@pytest.mark.ffmpeg
def test_merge_sends_crossfade_and_pause(fake_api, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    main_path = tmp_path / "main.mp3"
    main_path.write_bytes(b"main-bytes")
    out_path = tmp_path / "out.mp3"
    fake_api.respond_with(lambda _req: (200, b"merged-bytes"))

    cli._merge_with_api(
        merge_url=f"{fake_api.url}/api/merge",
        main_path=main_path,
        insert_audio=AudioSegment.silent(duration=500),
        insert_at_ms=31_250,
        crossfade=0.5,
        pause=0.75,
        out_path=out_path,
    )

    assert out_path.read_bytes() == b"merged-bytes"
    (request,) = fake_api.requests
    assert request.headers["authorization"] == "Bearer internal-token"
    body = request.body.decode("latin-1")
    for name, value in (("insertAt", "31.250"), ("crossfade", "0.5"), ("pause", "0.75")):
        assert f'name="{name}"\r\n\r\n{value}\r\n' in body
