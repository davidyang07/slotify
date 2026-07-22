"""Handcrafted acoustic and structural features on synthetic audio.

Synthetic signals rather than real audio, because the point is to assert
*relationships* that must hold by construction -- silence must be quieter than a
tone, a window clipped at an episode edge must say so -- and only a signal we
built ourselves lets us state the expected answer exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotify_rank.config.feature_settings import AcousticConfig, WindowConfig
from slotify_rank.data.schema import DatasetCandidate, EnergySummary
from slotify_rank.features.acoustic import compute_frames, extract_acoustic_features
from slotify_rank.features.schema import FeatureSpec
from slotify_rank.features.structural import extract_structural_features
from slotify_rank.features.windows import build_windows

SAMPLE_RATE = 16_000
CONFIG = AcousticConfig()
SCALES = WindowConfig().as_pairs()


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _tone(seconds: float, frequency: float = 220.0, amplitude: float = 0.3):
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _speech_like(seconds: float, seed: int = 0) -> np.ndarray:
    """Amplitude-modulated noise: broadband, with syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    noise = rng.normal(0, 0.25, n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    return (noise * envelope).astype(np.float32)


def _windows(timestamp_ms: int, duration_ms: int):
    return build_windows(timestamp_ms, duration_ms, SCALES)


def _extract(audio: np.ndarray, timestamp_ms: int):
    frames = compute_frames(audio, SAMPLE_RATE, CONFIG)
    duration_ms = int(audio.size * 1000 / SAMPLE_RATE)
    return extract_acoustic_features(frames, _windows(timestamp_ms, duration_ms))


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_features_extract_from_pure_silence():
    values, missing = _extract(_silence(6.0), 3_000)
    assert values
    assert set(values) == set(missing)


def test_features_extract_from_a_tone():
    values, _ = _extract(_tone(6.0), 3_000)
    assert values["spectral_centroid_mean_before_short"] > 0


def test_features_extract_from_speech_like_audio():
    values, missing = _extract(_speech_like(6.0), 3_000)
    assert not missing["rms_mean_before_short"]


def test_no_feature_is_nan_or_infinite():
    for audio in (_silence(6.0), _tone(6.0), _speech_like(6.0)):
        values, _ = _extract(audio, 3_000)
        bad = {
            name: value
            for name, value in values.items()
            if not np.isfinite(value)
        }
        assert bad == {}, f"non-finite features: {bad}"


# ---------------------------------------------------------------------------
# Expected relationships
# ---------------------------------------------------------------------------


def test_silence_is_quieter_than_a_tone():
    quiet, _ = _extract(_silence(6.0), 3_000)
    loud, _ = _extract(_tone(6.0), 3_000)
    assert quiet["rms_mean_before_short"] < loud["rms_mean_before_short"]


def test_a_tone_has_a_lower_centroid_than_broadband_noise():
    """A 220 Hz sine concentrates energy far below broadband noise."""
    tonal, _ = _extract(_tone(6.0), 3_000)
    broadband, _ = _extract(_speech_like(6.0), 3_000)
    assert (
        tonal["spectral_centroid_mean_before_short"]
        < broadband["spectral_centroid_mean_before_short"]
    )


def test_a_candidate_in_a_gap_reports_local_silence():
    """Speech, a two-second gap, speech. The candidate sits in the gap."""
    audio = np.concatenate([_speech_like(3.0), _silence(2.0), _speech_like(3.0, 1)])
    values, _ = _extract(audio, 4_000)
    assert values["silence_duration_ms"] > 1_000


def test_a_candidate_inside_speech_reports_no_local_silence():
    values, _ = _extract(_speech_like(6.0), 3_000)
    assert values["silence_duration_ms"] == 0.0


def test_energy_drops_across_a_speech_to_silence_boundary():
    audio = np.concatenate([_speech_like(3.0), _silence(3.0)])
    values, _ = _extract(audio, 3_000)
    assert values["rms_mean_after_short"] < values["rms_mean_before_short"]
    assert values["rms_delta_across_short"] < 0
    assert values["energy_discontinuity_short"] > 0


def test_energy_discontinuity_is_near_zero_in_steady_audio():
    values, _ = _extract(_speech_like(8.0), 4_000)
    assert values["energy_discontinuity_short"] < 12.0


def test_time_to_neighbouring_speech_is_measured_in_a_gap():
    audio = np.concatenate([_speech_like(3.0), _silence(2.0), _speech_like(3.0, 1)])
    values, missing = _extract(audio, 4_000)
    assert not missing["ms_since_prior_speech"]
    assert not missing["ms_until_next_speech"]
    assert values["ms_since_prior_speech"] > 0
    assert values["ms_until_next_speech"] > 0


# ---------------------------------------------------------------------------
# Boundary clipping
# ---------------------------------------------------------------------------


def test_windows_clip_at_the_episode_start():
    windows = _windows(500, 20_000)
    assert windows.before["context"].start_ms == 0
    assert windows.before["context"].available_fraction == pytest.approx(0.05)


def test_windows_clip_at_the_episode_end():
    windows = _windows(19_500, 20_000)
    assert windows.after["context"].end_ms == 20_000
    assert windows.after["context"].available_fraction == pytest.approx(0.05)


def test_a_fully_interior_window_is_completely_available():
    windows = _windows(30_000, 60_000)
    assert windows.before["context"].available_fraction == 1.0
    assert windows.edge_clipped is False


def test_availability_is_reported_as_a_feature():
    values, _ = _extract(_speech_like(20.0), 500)
    assert values["window_available_before_context"] < 0.1
    assert values["window_available_after_context"] == pytest.approx(1.0)


def test_a_candidate_outside_the_episode_is_rejected():
    """A manifest integrity failure, not a window to clip."""
    with pytest.raises(ValueError, match="outside the episode"):
        build_windows(30_000, 20_000, SCALES)


def test_boundary_candidates_still_produce_finite_features():
    audio = _speech_like(20.0)
    for timestamp in (0, 100, 19_900, 20_000):
        values, _ = _extract(audio, timestamp)
        assert all(np.isfinite(value) for value in values.values())


def test_a_window_with_too_few_frames_masks_variance_and_slope():
    """Variance and slope are undefined for a single frame."""
    audio = _speech_like(20.0)
    values, missing = _extract(audio, 0)
    # The 1 s "before" window at t=0 is empty.
    assert missing["rms_slope_before_short"] is True
    assert values["rms_slope_before_short"] == 0.0


def test_a_masked_value_is_never_meaningful():
    values, missing = _extract(_speech_like(20.0), 0)
    for name, is_missing in missing.items():
        if is_missing:
            assert values[name] == 0.0


# ---------------------------------------------------------------------------
# Stable ordering and the spec
# ---------------------------------------------------------------------------


def test_feature_names_are_identical_across_episodes():
    first, _ = _extract(_speech_like(6.0, 0), 3_000)
    second, _ = _extract(_tone(9.0), 4_000)
    assert sorted(first) == sorted(second)


def test_extraction_is_deterministic():
    audio = _speech_like(6.0, seed=7)
    first, _ = _extract(audio, 3_000)
    second, _ = _extract(audio, 3_000)
    assert first == second


def test_values_and_mask_always_share_their_keys():
    values, missing = _extract(_speech_like(6.0), 3_000)
    assert set(values) == set(missing)


def test_spec_vectorization_is_positional_and_sorted():
    values, missing = _extract(_speech_like(6.0), 3_000)
    spec = FeatureSpec.from_names(list(values))
    vector, mask = spec.vectorize(values, missing)
    assert len(vector) == len(mask) == spec.dimension
    assert list(spec.names) == sorted(spec.names)
    assert vector[spec.names.index("silence_duration_ms")] == pytest.approx(
        values["silence_duration_ms"]
    )


def test_spec_rejects_an_unknown_feature():
    values, missing = _extract(_speech_like(6.0), 3_000)
    spec = FeatureSpec.from_names(list(values))
    with pytest.raises(ValueError, match="not in the spec"):
        spec.vectorize({**values, "surprise": 1.0}, {**missing, "surprise": False})


def test_spec_rejects_a_short_vector():
    values, missing = _extract(_speech_like(6.0), 3_000)
    spec = FeatureSpec.from_names(list(values))
    trimmed = dict(values)
    dropped = trimmed.pop("silence_duration_ms")
    with pytest.raises(ValueError, match="in the spec but were not extracted"):
        spec.vectorize(trimmed, missing)


def test_spec_requires_sorted_names():
    with pytest.raises(ValueError, match="sorted order"):
        FeatureSpec(names=("zebra", "alpha"))


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


def _candidate(**overrides) -> DatasetCandidate:
    payload = dict(
        episode_id="ep_0123456789ab",
        candidate_id="ep_0123456789ab:000030000",
        timestamp_ms=30_000,
        candidate_sources=("silence", "rms_minimum"),
        pause_duration_ms=800,
        silence_duration_ms=1_200,
        sentence_end=True,
        local_energy_summary=EnergySummary(
            window_ms=1_000, mean_dbfs=-40.0, min_dbfs=-60.0, max_dbfs=-20.0
        ),
        raw_component_scores={
            "silence": 0.8,
            "position": 0.5,
            "sentence": 1.0,
            "spacing": 0.3,
        },
        heuristic_score=0.71,
    )
    payload.update(overrides)
    return DatasetCandidate(**payload)


def test_structural_position_features():
    values, _ = extract_structural_features(_candidate(), 60_000, 5_000)
    assert values["normalized_episode_position"] == pytest.approx(0.5)
    assert values["ms_from_episode_start"] == 30_000
    assert values["ms_to_episode_end"] == 30_000


def test_candidate_source_flags_are_set():
    values, _ = extract_structural_features(_candidate(), 60_000, 5_000)
    assert values["source_silence"] == 1.0
    assert values["source_rms_minimum"] == 1.0
    assert values["source_fixed_interval"] == 0.0
    assert values["merged_source_count"] == 2.0


def test_heuristic_scores_are_carried_through():
    values, missing = extract_structural_features(_candidate(), 60_000, 5_000)
    assert values["heuristic_total_score"] == pytest.approx(0.71)
    assert values["heuristic_component_sentence"] == pytest.approx(1.0)
    assert not missing["heuristic_total_score"]


def test_absent_heuristic_score_is_masked_not_zeroed():
    values, missing = extract_structural_features(
        _candidate(heuristic_score=None, raw_component_scores={}), 60_000, 5_000
    )
    assert missing["heuristic_total_score"] is True
    assert missing["heuristic_component_silence"] is True


def test_unknown_sentence_end_is_masked():
    values, missing = extract_structural_features(
        _candidate(sentence_end=None), 60_000, 5_000
    )
    assert missing["sentence_end"] is True
    assert values["sentence_end"] == 0.0


def test_edge_guard_violation_flag():
    inside, _ = extract_structural_features(
        _candidate(timestamp_ms=1_000, candidate_id="ep_0123456789ab:000001000"),
        60_000,
        5_000,
    )
    assert inside["edge_guard_violation"] == 1.0
    outside, _ = extract_structural_features(_candidate(), 60_000, 5_000)
    assert outside["edge_guard_violation"] == 0.0


def test_structural_values_and_mask_share_their_keys():
    values, missing = extract_structural_features(_candidate(), 60_000, 5_000)
    assert set(values) == set(missing)


def test_no_label_derived_feature_is_produced():
    """Label leakage check: nothing here may come from a rating."""
    values, _ = extract_structural_features(_candidate(), 60_000, 5_000)
    forbidden = ("label", "rating", "acceptable", "rank", "human")
    leaked = [name for name in values if any(word in name for word in forbidden)]
    assert leaked == []
