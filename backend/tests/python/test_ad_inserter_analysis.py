"""analysis.py: Whisper model reuse, song-mode valley picking, analysis decoding."""

from __future__ import annotations

import sys
import tempfile
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


@pytest.mark.ffmpeg  # song analysis decodes through ffmpeg
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


def _stereo_upload(path: Path, gaps: List[tuple[int, int]], seconds: int = 30) -> Path:
    """A 44.1 kHz stereo tone with silent gaps, the format of a typical upload."""
    tone = Sine(500, sample_rate=44100).to_audio_segment(duration=seconds * 1000, volume=-12)
    audio = tone.set_channels(2)
    for start_s, length_ms in gaps:
        start = start_s * 1000
        gap = AudioSegment.silent(length_ms, 44100).set_channels(2)
        audio = audio[:start] + gap + audio[start + length_ms :]
    audio.export(path, format="wav")
    return path


UPLOAD_GAPS = [(5, 800), (12, 1500), (21, 2400)]


@pytest.mark.ffmpeg
def test_analysis_audio_is_small_and_finds_the_same_silences(tmp_path: Path) -> None:
    path = _stereo_upload(tmp_path / "upload.wav", UPLOAD_GAPS)
    full = analysis.standardize_audio(analysis.load_audio(path))

    reduced = analysis.load_analysis_audio(path)

    assert (reduced.channels, reduced.frame_rate, reduced.sample_width) == (1, 16000, 2)
    assert abs(len(reduced) - len(full)) <= 1
    assert len(reduced.raw_data) * 5 < len(full.raw_data)
    # Thresholds are relative to the clip's dBFS, which the downmix keeps.
    assert reduced.dBFS == pytest.approx(full.dBFS, abs=0.1)
    full_silences = analyze_cli._detect_podcast_silences(full)
    reduced_silences = analyze_cli._detect_podcast_silences(reduced)
    assert len(reduced_silences) == len(full_silences) == len(UPLOAD_GAPS)
    for ours, theirs in zip(reduced_silences, full_silences):
        for key in ("start_ms", "end_ms", "mid_ms"):
            assert ours[key] == pytest.approx(theirs[key], abs=2)
    assert analysis.detect_podcast_candidates(reduced) == pytest.approx(
        analysis.detect_podcast_candidates(full), abs=2
    )


@pytest.mark.ffmpeg
def test_analysis_decode_failure_is_reported_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_audio = tmp_path / "upload.mp3"
    not_audio.write_bytes(b"this is not audio")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    with pytest.raises(RuntimeError, match="Could not decode upload.mp3"):
        analysis.load_analysis_audio(not_audio)

    assert list(scratch.iterdir()) == []


@pytest.mark.ffmpeg
@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_audio_window_matches_the_slice_of_a_full_decode(tmp_path: Path, fmt: str) -> None:
    source = _stereo_upload(tmp_path / "upload.wav", UPLOAD_GAPS, seconds=12)
    path = source if fmt == "wav" else tmp_path / "upload.mp3"
    if fmt == "mp3":
        AudioSegment.from_file(source).export(path, format="mp3")

    window = analysis.load_audio_window(path, 3_000, 7_000)

    full = analysis.standardize_audio(analysis.load_audio(path))
    assert (window.frame_rate, window.channels) == (44100, 2)
    assert window.raw_data == full[3_000:7_000].raw_data


@pytest.mark.ffmpeg
def test_song_samples_are_the_ones_librosa_load_gave(tmp_path: Path) -> None:
    import librosa

    left = _song_like(8.0, [4.0])
    path = tmp_path / "song.wav"
    sf.write(path, np.stack([left, 0.5 * left], axis=1), 44100, subtype="PCM_16")

    ours = analysis.load_song_samples(path)

    theirs, _ = librosa.load(path, sr=analysis.SONG_FRAME_RATE, mono=True)
    assert ours.dtype == np.float32
    np.testing.assert_array_equal(ours, theirs)
