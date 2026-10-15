"""analyze_cli.py: which silences become candidates and which get transcribed."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, List

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import analysis, analyze_cli, heuristic_config

RATE = 8000

# (position of the gap in seconds, gap length in ms). Many short pauses early
# on, the longest ones mid-episode -- the shape that starved the mid-episode
# reward when candidates were the first N in time.
EARLY_GAPS = [(3, 750), (6, 800), (9, 760), (12, 820), (15, 780), (18, 790)]
LONG_GAPS = [(50, 1600), (80, 2400), (120, 2000)]
CLIP_SECONDS = 150


def _speechlike_clip(gaps: List[tuple[int, int]]) -> AudioSegment:
    tone = Sine(500, sample_rate=RATE).to_audio_segment(
        duration=CLIP_SECONDS * 1000, volume=-12
    )
    audio = tone
    for start_s, length_ms in gaps:
        start = start_s * 1000
        audio = audio[:start] + AudioSegment.silent(length_ms, RATE) + audio[start + length_ms :]
    return audio


def _entry(mid_ms: int, silence_ms: int) -> Dict[str, int]:
    return {
        "start_ms": mid_ms - silence_ms // 2,
        "end_ms": mid_ms + silence_ms // 2,
        "mid_ms": mid_ms,
        "silence_ms": silence_ms,
    }


def test_longest_silences_ranks_by_length_and_returns_time_order() -> None:
    silences = [
        _entry(1_000, 800),
        _entry(2_000, 900),
        _entry(30_000, 2_000),
        _entry(40_000, 1_200),
        _entry(70_000, 1_200),
        _entry(90_000, 3_000),
    ]

    def mids(count: int) -> List[int]:
        return [entry["mid_ms"] for entry in analyze_cli._longest_silences(silences, count)]

    # The four longest, output in time order rather than by length.
    assert mids(4) == [30_000, 40_000, 70_000, 90_000]
    # 3000 and 2000, then the 1200 ms tie goes to the earlier one (40 s).
    assert mids(3) == [30_000, 40_000, 90_000]
    assert mids(0) == []
    assert mids(99) == [entry["mid_ms"] for entry in silences]


def test_candidates_spread_through_a_long_clip() -> None:
    audio = _speechlike_clip(EARLY_GAPS + LONG_GAPS)
    silences = analyze_cli._detect_podcast_silences(audio)
    assert len(silences) == len(EARLY_GAPS) + len(LONG_GAPS)

    chosen = analyze_cli._longest_silences(silences, len(LONG_GAPS))

    expected_mids = [start * 1000 + length // 2 for start, length in LONG_GAPS]
    assert [entry["mid_ms"] for entry in chosen] == pytest.approx(expected_mids, abs=50)


def test_snippets_cover_the_longest_silences(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _speechlike_clip(EARLY_GAPS + LONG_GAPS)
    candidates = analyze_cli._longest_silences(
        analyze_cli._detect_podcast_silences(audio), 30
    )
    transcribed: List[int] = []

    def fake_transcribe(_audio: AudioSegment, start_ms: int, end_ms: int) -> str:
        transcribed.append(end_ms)
        return "text."

    monkeypatch.setattr(analysis, "whisper_available", lambda: True)
    monkeypatch.setattr(analysis, "transcribe_snippet", fake_transcribe)

    snippets = analyze_cli._build_snippets(audio, candidates, 2)

    by_length = sorted(candidates, key=lambda entry: -entry["silence_ms"])
    assert sorted(snippets) == sorted(entry["mid_ms"] for entry in by_length[:2])
    assert sorted(transcribed) == sorted(snippets)


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    audio_path: Path,
    *extra: str,
) -> dict:
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(analysis, "whisper_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_cli", "--audio", str(audio_path), "--mode", "podcast", *extra],
    )
    analyze_cli.run()
    return json.loads(capsys.readouterr().out)


def test_cli_defaults_come_from_the_heuristic_config() -> None:
    config = json.loads(heuristic_config.CONFIG_PATH.read_text(encoding="utf-8"))
    detection = config["profiles"][heuristic_config.PRODUCT_PROFILE]["silence_detection"]

    assert heuristic_config.candidate_limits(heuristic_config.PRODUCT_PROFILE) == (
        detection["max_candidates"],
        detection["snippet_count"],
    )


@pytest.mark.ffmpeg  # analysis decodes through ffmpeg
def test_cli_keeps_the_longest_and_its_output_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "episode.wav"
    _speechlike_clip(EARLY_GAPS + LONG_GAPS).export(audio_path, format="wav")
    # No --max-candidates / --snippet-count: the defaults come from the config.
    monkeypatch.setattr(
        heuristic_config, "candidate_limits", lambda _profile: (len(LONG_GAPS), 1)
    )

    payload = _run_cli(monkeypatch, capsys, audio_path)

    assert set(payload) == {
        "mode",
        "duration_ms",
        "candidates",
        "snippets",
        "tempo",
        "beat_times_ms",
    }
    mids = [entry["mid_ms"] for entry in payload["candidates"]]
    expected_mids = [start * 1000 + length // 2 for start, length in LONG_GAPS]
    assert mids == pytest.approx(expected_mids, abs=50)
    for entry in payload["candidates"]:
        assert set(entry) == {"start_ms", "end_ms", "mid_ms", "silence_ms"}


@pytest.mark.ffmpeg
@pytest.mark.parametrize("mode", ["podcast", "song"])
def test_cli_never_decodes_the_upload_at_full_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    audio_path = tmp_path / "episode.wav"
    _speechlike_clip(LONG_GAPS).export(audio_path, format="wav")

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("analysis decoded the upload at full quality")

    monkeypatch.setattr(analysis, "load_audio", refuse)
    monkeypatch.setattr(analysis, "standardize_audio", refuse)
    if mode == "song":
        monkeypatch.setattr(
            analysis,
            "analyze_song",
            lambda _path: analysis.SongAnalysis(candidates_ms=[], tempo=None, beat_times_ms=[]),
        )

    payload = _run_cli(monkeypatch, capsys, audio_path, f"--mode={mode}")

    assert payload["duration_ms"] == pytest.approx(CLIP_SECONDS * 1000, abs=1)
    if mode == "podcast":
        assert len(payload["candidates"]) == len(LONG_GAPS)


@pytest.mark.ffmpeg
def test_cli_work_dir_holds_the_snippet_files_and_is_left_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "episode.wav"
    _speechlike_clip(LONG_GAPS).export(audio_path, format="wav")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    snippet_dirs: List[Path] = []

    class FakeModel:
        def transcribe(self, path: str, fp16: bool = False) -> dict:
            snippet_dirs.append(Path(path).parent.parent)
            return {"text": "and that's the news."}

    fake = types.ModuleType("whisper")
    fake.load_model = lambda _name: FakeModel()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.setattr(analysis, "_WHISPER_MODELS", {})
    monkeypatch.setattr(analysis, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_cli", f"--audio={audio_path}", "--snippet-count=2", f"--work-dir={work_dir}"],
    )

    analyze_cli.run()

    assert len(json.loads(capsys.readouterr().out)["snippets"]) == 2
    assert snippet_dirs == [work_dir, work_dir]
    assert list(work_dir.iterdir()) == []
