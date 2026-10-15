"""insert_ad.py: per-mode voice resolution, cleanup of cloned voices, caps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import analysis, insert_ad, llm, tts

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


def _clone_args(fake_api, *extra: str) -> List[str]:
    return [*extra, "--clone-voices", "--clone-url", f"{fake_api.url}/api/clone"]


def _cloning(monkeypatch: pytest.MonkeyPatch, fake_api) -> None:
    _diarized(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setattr(insert_ad, "ELEVENLABS_VOICES_URL", f"{fake_api.url}/v1/voices")


def test_cloned_voices_are_released_through_the_api_after_rendering(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
) -> None:
    _cloning(monkeypatch, fake_api)
    fake_api.respond_with(lambda req: (200, b'{"released": true}'))
    cloned: List[str] = []

    def fake_clone(_url: str, name: str, _samples: List[Path]) -> str:
        cloned.append(name)
        return f"clone-{len(cloned)}"

    monkeypatch.setattr(insert_ad, "_clone_voice", fake_clone)

    _run(monkeypatch, conversation, *_clone_args(fake_api, "--ad-mode", "DUO"))

    assert cloned == ["Speaker A Clone", "Speaker B Clone"]
    assert set(tts_calls) == {"clone-1", "clone-2"}
    assert [(req.method, req.path) for req in fake_api.requests] == [
        ("POST", "/api/clone/clone-1/release"),
        ("POST", "/api/clone/clone-2/release"),
    ]
    for request in fake_api.requests:
        assert request.headers["authorization"] == "Bearer internal-token"
        assert "xi-api-key" not in request.headers


def test_only_the_needed_speaker_is_cloned(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
) -> None:
    _cloning(monkeypatch, fake_api)
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")

    _run(monkeypatch, conversation, *_clone_args(fake_api, "--ad-mode", "A_ONLY"))

    assert tts_calls == ["clone-A"]
    assert [req.path for req in fake_api.requests] == ["/api/clone/clone-A/release"]


def test_cloned_voices_are_released_when_rendering_fails(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    fake_api,
) -> None:
    _cloning(monkeypatch, fake_api)
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")

    def failing_tts(_request: tts.TTSRequest) -> AudioSegment:
        raise RuntimeError("TTS request failed with status 500: boom")

    monkeypatch.setattr(tts, "synthesize_audio", failing_tts)

    with pytest.raises(RuntimeError, match="boom"):
        _run(monkeypatch, conversation, *_clone_args(fake_api, "--ad-mode", "DUO"))

    assert [req.path for req in fake_api.requests] == [
        "/api/clone/clone-A/release",
        "/api/clone/clone-B/release",
    ]


@pytest.mark.parametrize("release_status", [404, 500])
def test_a_voice_the_api_cannot_release_is_deleted_directly(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
    capsys: pytest.CaptureFixture[str],
    release_status: int,
) -> None:
    _cloning(monkeypatch, fake_api)
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")
    fake_api.respond_with(
        lambda req: (release_status, b"no") if req.path.endswith("/release") else (200, b"")
    )

    _run(monkeypatch, conversation, *_clone_args(fake_api, "--ad-mode", "A_ONLY"))

    release, delete = fake_api.requests
    assert (release.method, release.path) == ("POST", "/api/clone/clone-A/release")
    assert (delete.method, delete.path) == ("DELETE", "/v1/voices/clone-A")
    assert delete.headers["xi-api-key"] == "eleven-key"
    assert "authorization" not in delete.headers
    assert "could not release cloned voice clone-A" in capsys.readouterr().err


def test_failed_voice_cleanup_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    tts_calls: List[str],
    fake_api,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _cloning(monkeypatch, fake_api)
    monkeypatch.setattr(insert_ad, "_clone_voice", lambda _u, name, _s: f"clone-{name[8]}")
    fake_api.respond_with(lambda _req: (500, b"nope"))

    out = _run(monkeypatch, conversation, *_clone_args(fake_api, "--ad-mode", "A_ONLY"))

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


def test_a_model_off_the_allow_list_is_refused_up_front(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(_path: Path) -> AudioSegment:
        raise AssertionError("the run started")

    monkeypatch.setattr(analysis, "load_audio", refuse)
    monkeypatch.setenv("SLOTIFY_OPENAI_MODELS", "gpt-4o-mini, gpt-4.1-mini")

    with pytest.raises(SystemExit) as refused:
        _run(
            monkeypatch,
            conversation,
            "--ad-mode",
            "A_ONLY",
            "--llm-provider",
            "openai",
            "--llm-model",
            "o1-pro",
        )

    assert refused.value.code == 2
    assert "permits: gpt-4o-mini, gpt-4.1-mini" in capsys.readouterr().err


def test_cloning_without_the_elevenlabs_key_is_refused_up_front(
    monkeypatch: pytest.MonkeyPatch,
    conversation: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _diarized(monkeypatch)

    def refuse(*_args: object) -> str:
        raise AssertionError("a voice was cloned that could not be deleted")

    monkeypatch.setattr(insert_ad, "_clone_voice", refuse)

    with pytest.raises(SystemExit) as refused:
        _run(monkeypatch, conversation, "--ad-mode", "A_ONLY", "--clone-voices")

    assert refused.value.code == 2
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("ad_mode", ["A_ONLY", "DUO"])
def test_fallback_script_with_a_huge_blurb_is_capped(
    monkeypatch: pytest.MonkeyPatch, conversation: Path, ad_mode: str
) -> None:
    _diarized(monkeypatch)
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    texts: List[str] = []

    def fake_tts(request: tts.TTSRequest) -> AudioSegment:
        texts.append(request.text)
        return Sine(440).to_audio_segment(duration=1_200, volume=-10)

    monkeypatch.setattr(tts, "synthesize_audio", fake_tts)
    # The LLM returns nothing usable, so the script falls back to the blurb.
    monkeypatch.setattr(
        llm,
        "generate_ad_script",
        lambda **_kwargs: llm.AdScriptResult(segments=[], prompt="", raw_text=""),
    )
    argv_blurb = "a calmer note-taking app " * 400

    _run(
        monkeypatch,
        conversation,
        "--ad-mode",
        ad_mode,
        "--voice-id-a",
        "voice-a",
        "--voice-id-b",
        "voice-b",
        "--product-blurb",
        argv_blurb,
    )

    assert texts
    assert sum(len(text) for text in texts) <= llm.MAX_SCRIPT_CHARS
    assert len(texts) <= llm.MAX_AD_SEGMENTS


def test_voice_cleanup_uses_short_timeouts_once_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ad_inserter import http_client, runtime

    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-key")
    calls: List[Tuple[str, float]] = []

    def failing_send(_url: str, label: str, *_a: object, timeout_s: float, **_k: object) -> bytes:
        calls.append((label, timeout_s))
        raise RuntimeError(f"{label} request failed with status 404: unknown")

    monkeypatch.setattr(http_client, "send", failing_send)

    insert_ad._release_cloned_voice("http://api/api/clone", "voice-1")
    monkeypatch.setattr(runtime, "termination_requested", lambda: True)
    insert_ad._release_cloned_voice("http://api/api/clone", "voice-2")

    shutdown = insert_ad.VOICE_CLEANUP_SHUTDOWN_TIMEOUT_S
    assert calls == [
        ("Voice release", insert_ad.VOICE_RELEASE_TIMEOUT_S),
        ("Voice delete", insert_ad.VOICE_DELETE_TIMEOUT_S),
        ("Voice release", shutdown),
        ("Voice delete", shutdown),
    ]
    # Release + fallback delete for two voices fit in Node's 5 s grace period.
    assert 4 * shutdown < 5.0
