"""insert_ad.py: per-mode voice resolution and cleanup of cloned voices."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import analysis, insert_ad, tts

pytestmark = pytest.mark.ffmpeg  # the run exports its result as MP3


@pytest.fixture
def conversation(tmp_path: Path) -> Path:
    tone = Sine(220).to_audio_segment(duration=3_000, volume=-14)
    audio = tone
    for _ in range(4):
        audio += AudioSegment.silent(duration=900) + tone
    path = tmp_path / "conversation.wav"
    audio.export(path, format="wav")
    return path


@pytest.fixture
def tts_calls(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Replace TTS with a local tone; record the voice ID of each request."""
    calls: List[str] = []

    def fake_synthesize(request: tts.TTSRequest) -> AudioSegment:
        calls.append(request.voice_id)
        return Sine(440).to_audio_segment(duration=1_200, volume=-10)

    monkeypatch.setattr(tts, "synthesize_audio", fake_synthesize)
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    return calls


def _run(monkeypatch: pytest.MonkeyPatch, conversation: Path, *extra: str) -> Path:
    out = conversation.with_name("out.mp3")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "insert_ad",
            "--input",
            str(conversation),
            "--product-name",
            "Sparrow Notes",
            "--product-blurb",
            "A calmer note-taking app",
            "--ad-style",
            "casual",
            "--out",
            str(out),
            "--llm-provider",
            "none",
            *extra,
        ],
    )
    insert_ad.run()
    return out


def test_a_only_needs_only_voice_a(
    monkeypatch: pytest.MonkeyPatch, conversation: Path, tts_calls: List[str]
) -> None:
    monkeypatch.setattr(analysis, "diarize_speakers", lambda _path: None)

    out = _run(monkeypatch, conversation, "--ad-mode", "A_ONLY", "--voice-id-a", "voice-a")

    assert out.exists()
    assert tts_calls == ["voice-a"]


def test_b_only_needs_only_voice_b(
    monkeypatch: pytest.MonkeyPatch, conversation: Path, tts_calls: List[str]
) -> None:
    monkeypatch.setattr(analysis, "diarize_speakers", lambda _path: None)
    monkeypatch.setenv("ELEVENLABS_VOICE_ID_B", "env-voice-b")

    _run(monkeypatch, conversation, "--ad-mode", "B_ONLY")

    assert tts_calls == ["env-voice-b"]


def test_missing_voice_for_a_needed_speaker_still_fails(
    monkeypatch: pytest.MonkeyPatch, conversation: Path, tts_calls: List[str]
) -> None:
    monkeypatch.setattr(analysis, "diarize_speakers", lambda _path: None)

    with pytest.raises(RuntimeError, match="speaker A"):
        _run(monkeypatch, conversation, "--ad-mode", "A_ONLY", "--voice-id-b", "voice-b")


def _diarized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis,
        "diarize_speakers",
        lambda _path: [
            analysis.SpeakerSegment(speaker="A", start_ms=0, end_ms=3_000),
            analysis.SpeakerSegment(speaker="B", start_ms=3_900, end_ms=6_900),
        ],
    )


def test_cloned_voices_are_deleted_after_rendering(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
) -> None:
    _diarized(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setattr(insert_ad, "ELEVENLABS_VOICES_URL", f"{fake_api.url}/v1/voices")
    cloned: List[str] = []

    def fake_clone(_url: str, name: str, _samples: List[Path]) -> str:
        cloned.append(name)
        return f"clone-{len(cloned)}"

    monkeypatch.setattr(insert_ad, "_clone_voice", fake_clone)

    _run(monkeypatch, conversation, "--ad-mode", "DUO", "--clone-voices")

    assert cloned == ["Speaker A Clone", "Speaker B Clone"]
    assert set(tts_calls) == {"clone-1", "clone-2"}
    deletes = [(req.method, req.path) for req in fake_api.requests]
    assert deletes == [("DELETE", "/v1/voices/clone-1"), ("DELETE", "/v1/voices/clone-2")]
    for request in fake_api.requests:
        assert request.headers["xi-api-key"] == "eleven-key"
        assert "authorization" not in request.headers


def test_only_the_needed_speaker_is_cloned(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
) -> None:
    _diarized(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    monkeypatch.setattr(insert_ad, "ELEVENLABS_VOICES_URL", f"{fake_api.url}/v1/voices")
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")

    _run(monkeypatch, conversation, "--ad-mode", "A_ONLY", "--clone-voices")

    assert tts_calls == ["clone-A"]
    assert [req.path for req in fake_api.requests] == ["/v1/voices/clone-A"]


def test_cloned_voices_are_deleted_when_rendering_fails(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    fake_api,
) -> None:
    _diarized(monkeypatch)
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    monkeypatch.setattr(insert_ad, "ELEVENLABS_VOICES_URL", f"{fake_api.url}/v1/voices")
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")

    def failing_tts(_request: tts.TTSRequest) -> AudioSegment:
        raise RuntimeError("TTS request failed with status 500: boom")

    monkeypatch.setattr(tts, "synthesize_audio", failing_tts)

    with pytest.raises(RuntimeError, match="boom"):
        _run(monkeypatch, conversation, "--ad-mode", "DUO", "--clone-voices")

    assert [req.path for req in fake_api.requests] == [
        "/v1/voices/clone-A",
        "/v1/voices/clone-B",
    ]


def test_failed_voice_delete_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _diarized(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    monkeypatch.setattr(insert_ad, "ELEVENLABS_VOICES_URL", f"{fake_api.url}/v1/voices")
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")
    fake_api.respond_with(lambda _req: (500, b"nope"))

    out = _run(monkeypatch, conversation, "--ad-mode", "A_ONLY", "--clone-voices")

    assert out.exists()
    assert "failed to delete cloned voice clone-A" in capsys.readouterr().err


def test_clone_callback_carries_the_internal_token(
    fake_api, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    fake_api.respond_with(lambda _req: (200, b'{"voiceId": "new-voice"}'))
    sample = tmp_path / "speaker_a.wav"
    AudioSegment.silent(duration=500).export(sample, format="wav")

    voice_id = insert_ad._clone_voice(f"{fake_api.url}/api/clone", "Speaker A Clone", [sample])

    assert voice_id == "new-voice"
    (request,) = fake_api.requests
    assert request.headers["authorization"] == "Bearer internal-token"
    assert b'name="name"\r\n\r\nSpeaker A Clone\r\n' in request.body
