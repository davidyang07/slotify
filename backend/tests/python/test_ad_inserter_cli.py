"""cli.py: Whisper fallback for the chosen point, merge flags, removed --duck-db."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import analysis, cli, tts


@pytest.fixture
def episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tone = Sine(220).to_audio_segment(duration=4_000, volume=-14)
    audio = tone
    for _ in range(5):
        audio += AudioSegment.silent(duration=900) + tone
    path = tmp_path / "episode.wav"
    audio.export(path, format="wav")

    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(
        tts,
        "synthesize_audio",
        lambda _request: Sine(440).to_audio_segment(duration=1_500, volume=-10),
    )
    return path


def _argv(episode: Path, *extra: str) -> List[str]:
    return [
        "cli",
        "--main",
        str(episode),
        "--voice-id",
        "voice-1",
        "--product-name",
        "Sparrow Notes",
        "--product-desc",
        "A calmer note-taking app",
        "--out",
        str(episode.with_name("out.mp3")),
        "--llm-provider",
        "none",
        *extra,
    ]


def test_chosen_point_is_transcribed_when_it_has_no_snippet(
    episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcribed: List[Tuple[int, int]] = []

    def fake_transcribe(_audio: AudioSegment, start_ms: int, end_ms: int) -> str:
        transcribed.append((start_ms, end_ms))
        return "and that's the first half."

    monkeypatch.setattr(analysis, "whisper_available", lambda: True)
    monkeypatch.setattr(analysis, "build_snippets", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(analysis, "transcribe_snippet", fake_transcribe)
    debug_dir = tmp_path / "debug"
    monkeypatch.setattr(sys, "argv", _argv(episode, "--dry-run", "--debug-dir", str(debug_dir)))

    cli.run()

    debug = json.loads((debug_dir / "debug.json").read_text())
    chosen = debug["chosen_insertion_ms"]
    assert transcribed == [(chosen - 15_000, chosen)]
    assert debug["transcript_snippet_before"] == "and that's the first half."


def test_transcript_is_marked_unavailable_without_whisper(
    episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "whisper_available", lambda: False)
    debug_dir = tmp_path / "debug"
    monkeypatch.setattr(sys, "argv", _argv(episode, "--dry-run", "--debug-dir", str(debug_dir)))

    cli.run()

    debug = json.loads((debug_dir / "debug.json").read_text())
    assert debug["transcript_snippet_before"] == "TRANSCRIPT_UNAVAILABLE"


def test_duck_db_is_no_longer_accepted(episode: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", _argv(episode, "--dry-run", "--duck-db", "6"))

    with pytest.raises(SystemExit):
        cli.run()


@pytest.mark.ffmpeg
def test_merge_flags_reach_the_merge_endpoint(
    episode: Path, fake_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "whisper_available", lambda: False)
    monkeypatch.setenv("SLOTIFY_INTERNAL_TOKEN", "internal-token")
    fake_api.respond_with(lambda _req: (200, b"merged-bytes"))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            episode,
            "--merge-url",
            f"{fake_api.url}/api/merge",
            "--merge-crossfade",
            "0.5",
            "--merge-pause",
            "0.7",
        ),
    )

    cli.run()

    assert episode.with_name("out.mp3").read_bytes() == b"merged-bytes"
    (request,) = fake_api.requests
    assert request.path == "/api/merge"
    assert request.headers["authorization"] == "Bearer internal-token"
    body = request.body.decode("latin-1")
    assert 'name="crossfade"\r\n\r\n0.5\r\n' in body
    assert 'name="pause"\r\n\r\n0.7\r\n' in body
