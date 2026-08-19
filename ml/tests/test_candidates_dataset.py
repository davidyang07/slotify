"""Candidate generation, merging and the synthetic-exclusion contract.

Silence detection is tested against a synthesised WAV with exact, known
boundaries, and the whole generator is then run over the real repository smoke
fixtures (marked ``audio``) to prove it works on actual encoded audio.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.candidates.audio_candidates import (
    RawCandidate,
    apply_edge_guard,
    detect_silence_ranges,
    fixed_interval_candidates,
    pause_candidates,
    rms_minimum_candidates,
    silence_candidates,
)
from slotify_rank.candidates.config import (
    FixedIntervalConfig,
    GenerationConfig,
    PauseConfig,
    RmsMinimumConfig,
    TranscriptConfig,
    load_generation_config,
)
from slotify_rank.candidates.generate import generate_for_episode
from slotify_rank.candidates.merge import (
    assert_agrees_with_product_merge,
    merge_raw_candidates,
)
from slotify_rank.candidates.transcript_candidates import (
    load_transcript,
    transcript_candidates,
)
from slotify_rank.config.settings import find_repo_root, load_heuristic_config
from slotify_rank.data.audio_io import load_normalized_wav
from slotify_rank.data.paths import DataPaths

from tests.dataset_fixtures import make_episode, write_speech_like_wav, write_wav


# --------------------------------------------------------------------------
# Audio envelope
# --------------------------------------------------------------------------


def test_envelope_reports_millisecond_resolution(tmp_path: Path):
    path = write_speech_like_wav(tmp_path / "a.wav", [(1000, True), (1000, False)])
    envelope = load_normalized_wav(path)
    assert envelope.n_ms == 2000
    assert envelope.sample_rate_hz == 16_000
    assert envelope.region_dbfs(1200, 1800) < envelope.region_dbfs(200, 800)


def test_stereo_audio_is_rejected(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    write_wav(path, np.zeros(3200), channels=2)
    with pytest.raises(ValueError, match="must be mono"):
        load_normalized_wav(path)


def test_empty_audio_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="zero-length"):
        load_normalized_wav(path)


# --------------------------------------------------------------------------
# Silence and pause detection
# --------------------------------------------------------------------------


def test_silence_ranges_match_the_synthesised_boundaries(tmp_path: Path):
    path = write_speech_like_wav(
        tmp_path / "a.wav",
        [(2000, True), (1000, False), (2000, True), (1000, False), (2000, True)],
    )
    envelope = load_normalized_wav(path)
    ranges = detect_silence_ranges(envelope, 700, envelope.overall_dbfs - 16)
    assert len(ranges) == 2
    first_start, first_end = ranges[0]
    assert 1900 <= first_start <= 2100
    assert 2900 <= first_end <= 3100


def test_no_silence_found_in_continuous_tone(tmp_path: Path):
    path = write_speech_like_wav(tmp_path / "a.wav", [(3000, True)])
    envelope = load_normalized_wav(path)
    assert detect_silence_ranges(envelope, 700, envelope.overall_dbfs - 16) == []


def test_silence_candidates_sit_at_the_midpoint(tmp_path: Path):
    path = write_speech_like_wav(
        tmp_path / "a.wav", [(2000, True), (1000, False), (2000, True)]
    )
    envelope = load_normalized_wav(path)
    candidates = silence_candidates(envelope, 700, -16.0, -40.0)
    assert len(candidates) == 1
    assert 2400 <= candidates[0].ms <= 2600
    assert candidates[0].silence_ms >= 700


def test_pause_generator_excludes_full_length_silences(tmp_path: Path):
    """A pause and a silence must never both claim the same gap."""
    path = write_speech_like_wav(
        tmp_path / "a.wav",
        [(1500, True), (300, False), (1500, True), (1200, False), (1500, True)],
    )
    envelope = load_normalized_wav(path)
    pauses = pause_candidates(
        envelope, PauseConfig(min_pause_ms=250, max_pause_ms=700), -40.0
    )
    assert pauses, "the 300 ms gap should be found"
    assert all(candidate.pause_ms < 700 for candidate in pauses)


def test_rms_minimum_respects_spacing(tmp_path: Path):
    segments = []
    for _ in range(6):
        segments += [(1500, True), (400, False)]
    path = write_speech_like_wav(tmp_path / "a.wav", segments)
    envelope = load_normalized_wav(path)
    minima = rms_minimum_candidates(
        envelope, RmsMinimumConfig(min_spacing_ms=3000, percentile=25.0)
    )
    times = sorted(candidate.ms for candidate in minima)
    assert all(later - earlier >= 3000 for earlier, later in zip(times, times[1:]))


def test_fixed_interval_grid_excludes_both_endpoints():
    candidates = fixed_interval_candidates(90_000, FixedIntervalConfig(interval_ms=30_000))
    assert [candidate.ms for candidate in candidates] == [30_000, 60_000]


def test_edge_guard_drops_candidates_near_the_ends():
    raw = [RawCandidate(ms=ms, source="silence") for ms in (1000, 30_000, 59_000)]
    kept, dropped = apply_edge_guard(raw, duration_ms=60_000, edge_guard_ms=5_000)
    assert [candidate.ms for candidate in kept] == [30_000]
    assert dropped == 2


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def test_merge_collapses_near_duplicates_and_unions_sources():
    merged = merge_raw_candidates(
        [
            RawCandidate(ms=10_000, source="silence", silence_ms=900),
            RawCandidate(ms=10_200, source="rms_minimum"),
            RawCandidate(ms=10_300, source="fixed_interval"),
        ],
        tolerance_ms=400,
    )
    assert len(merged) == 1
    assert set(merged[0].sources) == {"silence", "rms_minimum", "fixed_interval"}
    assert merged[0].merged_from_ms == (10_200, 10_300)


def test_merge_keeps_candidates_beyond_the_tolerance():
    merged = merge_raw_candidates(
        [
            RawCandidate(ms=10_000, source="silence"),
            RawCandidate(ms=10_401, source="silence"),
        ],
        tolerance_ms=400,
    )
    assert [candidate.ms for candidate in merged] == [10_000, 10_401]


def test_merge_prefers_the_longer_pause():
    merged = merge_raw_candidates(
        [
            RawCandidate(ms=10_000, source="pause", silence_ms=300),
            RawCandidate(ms=10_200, source="silence", silence_ms=1500),
        ],
        tolerance_ms=400,
    )
    assert merged[0].ms == 10_200
    assert merged[0].silence_ms == 1500


def test_merge_output_is_independent_of_input_order():
    raw = [
        RawCandidate(ms=10_000, source="silence", silence_ms=900),
        RawCandidate(ms=10_200, source="rms_minimum"),
        RawCandidate(ms=40_000, source="fixed_interval"),
    ]
    forward = merge_raw_candidates(raw, 400)
    reverse = merge_raw_candidates(list(reversed(raw)), 400)
    assert [c.ms for c in forward] == [c.ms for c in reverse]
    assert [sorted(c.sources) for c in forward] == [sorted(c.sources) for c in reverse]


def test_dataset_merge_agrees_with_the_product_merge():
    """Anti-drift guard: dataset merging must pick the product's timestamps."""
    raw = [
        RawCandidate(ms=10_000, source="silence", silence_ms=500, snippet="mid thought"),
        RawCandidate(ms=10_200, source="silence", silence_ms=1500, snippet="Done."),
        RawCandidate(ms=20_000, source="pause", silence_ms=300),
        RawCandidate(ms=20_350, source="silence", silence_ms=900, snippet="Right."),
        RawCandidate(ms=60_000, source="fixed_interval"),
    ]
    assert_agrees_with_product_merge(raw, 400)


def test_merge_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        merge_raw_candidates([], -1)


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------


def test_transcript_candidates_carry_context(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text(
        '{"segments": ['
        '{"id": 0, "start": 0.0, "end": 5.5, "text": "That is settled."},'
        '{"id": 1, "start": 6.0, "end": 9.0, "text": "Now for something else."}]}',
        encoding="utf-8",
    )
    candidates = transcript_candidates(load_transcript(path), TranscriptConfig())
    assert len(candidates) == 1, "the final segment has no 'after' context"
    assert candidates[0].ms == 5500
    assert candidates[0].sentence_end is True
    assert candidates[0].transcript_after == "Now for something else."


def test_transcript_accepts_millisecond_bounds(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text(
        '[{"start_ms": 0, "end_ms": 4200, "text": "One."},'
        ' {"start_ms": 4500, "end_ms": 8000, "text": "Two."}]',
        encoding="utf-8",
    )
    assert load_transcript(path)[0].end_ms == 4200


def test_transcript_rejects_reversed_bounds(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text('[{"start_ms": 5000, "end_ms": 1000, "text": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="precedes start"):
        load_transcript(path)


def test_transcript_rejects_missing_timestamps(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text('[{"text": "x"}]', encoding="utf-8")
    with pytest.raises(KeyError, match="start_ms"):
        load_transcript(path)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_silence_settings_default_to_the_canonical_profile():
    profile = load_heuristic_config().canonical
    assert GenerationConfig().resolved_silence(profile) == (700, -16.0, -40.0)


def test_repository_generation_config_loads():
    config = load_generation_config(find_repo_root() / "ml" / "configs" / "dataset_v1.yaml")
    assert config.dataset_version == "dataset_v1"
    assert config.merge_tolerance_ms == 400
    assert config.silence.min_silence_len_ms is None, (
        "the committed config must inherit the canonical silence thresholds"
    )


def test_unknown_generation_setting_is_rejected(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("candidate_generation:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown setting"):
        load_generation_config(path)


def test_invalid_percentile_is_rejected():
    with pytest.raises(ValueError, match="percentile"):
        GenerationConfig(rms_minimum=RmsMinimumConfig(percentile=0.0))


# --------------------------------------------------------------------------
# End-to-end generation
# --------------------------------------------------------------------------


def _episode_with_audio(tmp_path: Path, segments) -> tuple:
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(title="Synthetic", duration_ms=sum(d for d, _ in segments))
    write_speech_like_wav(paths.absolute(episode.normalized_path), segments)
    return episode, paths


def test_generation_produces_scored_in_bounds_candidates(tmp_path: Path):
    segments = []
    for _ in range(10):
        segments += [(4000, True), (1200, False)]
    episode, paths = _episode_with_audio(tmp_path, segments)

    records, report = generate_for_episode(episode, paths)

    assert records, "a 52 s clip with ten pauses should yield candidates"
    assert report.count_before_merge >= report.count_after_merge
    duration_ms = sum(d for d, _ in segments)
    for record in records:
        assert 5000 <= record.timestamp_ms <= duration_ms - 5000
        assert record.heuristic_score is not None
        assert 0.0 <= record.heuristic_score <= 1.0
        assert record.baseline_version == "heuristic_offline_v1"
        assert record.raw_component_scores["raw_total"] is not None
        assert not record.is_synthetic
        assert record.eligible_for_labelling and record.eligible_for_evaluation
    assert len({record.candidate_id for record in records}) == len(records)


def test_generation_is_deterministic(tmp_path: Path):
    segments = [(3000, True), (1000, False)] * 8
    episode, paths = _episode_with_audio(tmp_path, segments)
    first, _ = generate_for_episode(episode, paths)
    second, _ = generate_for_episode(episode, paths)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


def test_source_flags_survive_generation(tmp_path: Path):
    segments = [(3000, True), (1000, False)] * 12
    episode, paths = _episode_with_audio(tmp_path, segments)
    records, _ = generate_for_episode(episode, paths)
    flags = {source for record in records for source in record.candidate_sources}
    assert flags & {"silence", "fixed_interval", "rms_minimum", "pause"}
    assert all(record.candidate_sources for record in records)


def test_generation_refuses_unnormalized_episodes(tmp_path: Path):
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episode = make_episode(status="probed", normalized_path=None)
    with pytest.raises(ValueError, match="require normalized audio"):
        generate_for_episode(episode, paths)


def test_product_padding_is_recorded_but_never_eligible(tmp_path: Path):
    """A near-silent clip yields too few candidates, forcing the product to pad."""
    segments = [(6000, True), (1500, False), (6000, True)]
    episode, paths = _episode_with_audio(tmp_path, segments)
    records, report = generate_for_episode(
        episode, paths, include_product_padding=True
    )
    synthetic = [record for record in records if record.is_synthetic]
    assert synthetic, "this clip is expected to require padding"
    assert report.synthetic_recorded == len(synthetic)
    for record in synthetic:
        assert record.candidate_sources == ("product_padding",)
        assert not record.eligible_for_labelling
        assert not record.eligible_for_evaluation


def test_density_cap_is_reported_not_silent(tmp_path: Path):
    segments = [(700, True), (800, False)] * 40
    episode, paths = _episode_with_audio(tmp_path, segments)
    config = GenerationConfig(max_candidates_per_minute=2.0)
    records, report = generate_for_episode(episode, paths, config=config)
    assert report.dropped_density_cap > 0
    assert any("density cap" in note for note in report.notes)
    assert len(records) == report.count_after_density_cap


# --------------------------------------------------------------------------
# Real smoke audio
# --------------------------------------------------------------------------


@pytest.mark.audio
def test_generation_runs_on_a_real_smoke_fixture(tmp_path: Path):
    """Runs the generator over actual encoded audio from the repository.

    Skipped when the fixture or FFmpeg is unavailable; it decodes a real MP3
    rather than a synthesised tone, which is the only way to know the pipeline
    survives real-world encoding.
    """
    from slotify_rank.data.ffmpeg import FFmpegNotFound
    from slotify_rank.data.normalize import normalize_episode
    from slotify_rank.data.probe import probe_audio

    fixture = find_repo_root() / "backend" / "audio_tests" / "convo.mp3"
    if not fixture.is_file():
        pytest.skip("smoke fixture is not present in this checkout")

    # The scratch directory is its own repository root, and the fixture is
    # copied into its raw directory first -- exactly what `dataset import-local`
    # does. A manifest can only describe paths under its own root, so pointing a
    # scratch data root at a file still sitting in the checkout is not a
    # supported combination.
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    imported = paths.raw_dir / fixture.name
    shutil.copy2(fixture, imported)
    from slotify_rank.data.checksum import sha256_file

    digest = sha256_file(imported)
    try:
        metadata = probe_audio(imported)
    except FFmpegNotFound:
        pytest.skip("ffprobe is not available")

    episode = make_episode(
        title="Smoke convo",
        content_type="conversational",
        status="probed",
        normalized_path=None,
        normalized_sha256=None,
        preprocessing_version=None,
        original_path=paths.relative(imported),
        sha256=digest,
        duration_ms=metadata.duration_ms,
    )
    normalized = normalize_episode(episode, paths).episode
    records, report = generate_for_episode(normalized, paths)

    assert report.count_before_merge > 0
    assert all(0 <= r.timestamp_ms <= normalized.duration_ms for r in records)
    assert all(r.local_energy_summary is not None for r in records)
