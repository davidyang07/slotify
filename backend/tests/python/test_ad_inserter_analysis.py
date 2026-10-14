"""analysis.py: Whisper model reuse and song-mode valley picking."""

from __future__ import annotations

import sys
import types
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import analysis, analyze_cli

SR = 22050


def _install_fake_whisper(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    loads: List[str] = []

    class FakeModel:
        def transcribe(self, path: str, fp16: bool = False) -> dict:
            return {"text": " and that's the news. "}

    def load_model(name: str) -> FakeModel:
        loads.append(name)
        return FakeModel()

    fake = types.ModuleType("whisper")
    fake.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.setattr(analysis, "_WHISPER_MODELS", {})
    return loads


def test_whisper_model_is_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _install_fake_whisper(monkeypatch)
    audio = Sine(300).to_audio_segment(duration=60_000, volume=-12)
    candidates = [
        {"start_ms": ms - 500, "end_ms": ms + 500, "mid_ms": ms, "silence_ms": 1000}
        for ms in (16_000, 30_000, 45_000, 58_000)
    ]

    snippets = analyze_cli._build_snippets(audio, candidates, 4)

    assert loads == ["base"]
    assert set(snippets.values()) == {"and that's the news."}
    analysis.transcribe_snippet(audio, 0, 2_000)
    assert loads == ["base"]
    analysis.transcribe_snippet(audio, 0, 2_000, model_name="tiny")
    assert loads == ["base", "tiny"]


def _song_like(duration_s: float, dips_s: List[float]) -> np.ndarray:
    """Loud bed, a very quiet intro and outro, and short quieter dips."""
    rng = np.random.default_rng(7)
    n = int(duration_s * SR)
    y = 0.3 * rng.standard_normal(n).astype(np.float32)
    t = np.arange(n) / SR
    envelope = np.ones(n, dtype=np.float32)
    # The quietest audio of all is at the edges: 3 s fade-in and fade-out.
    envelope[t < 3.0] = 0.01
    envelope[t > duration_s - 3.0] = 0.01
    for dip in dips_s:
        envelope[(t >= dip - 0.5) & (t < dip + 0.5)] = 0.05
    y *= envelope
    # A click every 0.5 s (120 BPM) so beat tracking has something to lock to.
    for beat in np.arange(0.25, duration_s, 0.5):
        start = int(beat * SR)
        y[start : start + 200] += 0.9
    return np.clip(y, -1.0, 1.0)


DIPS_S = [15.0, 30.0, 45.0]


def test_valleys_are_separated_and_kept_off_the_edges() -> None:
    duration_s = 60.0
    y = _song_like(duration_s, DIPS_S)

    times = analysis._rms_minima_times(y, SR)

    guard = analysis.SONG_VALLEY_EDGE_GUARD_MS / 1000.0
    separation = analysis.SONG_VALLEY_MIN_SEPARATION_MS / 1000.0
    assert times == sorted(times)
    assert len(times) >= len(DIPS_S)
    assert all(guard <= t <= duration_s - guard for t in times)
    assert all(b - a >= separation for a, b in combinations(times, 2))
    for dip in DIPS_S:
        assert any(abs(t - dip) <= 0.5 for t in times), (dip, times)


def test_valleys_fall_back_when_the_clip_is_shorter_than_the_guards() -> None:
    y = _song_like(8.0, [])

    times = analysis._rms_minima_times(y, SR, top_k=12)

    assert times  # still proposes something
    assert all(b - a >= 6.0 for a, b in combinations(times, 2))


def test_analyze_song_spreads_candidates_and_snaps_them_to_beats(tmp_path: Path) -> None:
    duration_s = 60.0
    path = tmp_path / "song.wav"
    sf.write(path, _song_like(duration_s, DIPS_S), SR)

    song = analysis.analyze_song(path)

    assert song.tempo is not None
    assert song.beat_times_ms
    assert song.candidates_ms == sorted(song.candidates_ms)
    assert set(song.candidates_ms) <= set(song.beat_times_ms)
    # Spread through the song, not collapsed into the quiet intro/outro.
    assert all(4_000 <= ms <= 56_000 for ms in song.candidates_ms)
    assert all(b - a >= 5_000 for a, b in combinations(song.candidates_ms, 2))
    for dip in DIPS_S:
        assert any(abs(ms - dip * 1000) <= 800 for ms in song.candidates_ms), (
            dip,
            song.candidates_ms,
        )


def test_transcribe_snippet_without_whisper_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "whisper", None)  # import raises ImportError

    assert analysis.whisper_available() is False
    assert (
        analysis.transcribe_snippet(AudioSegment.silent(2_000), 0, 1_000)
        == "TRANSCRIPT_UNAVAILABLE"
    )
