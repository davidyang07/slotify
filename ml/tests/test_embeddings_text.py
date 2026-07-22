"""Transcript-context selection, MiniLM vector construction and text features.

The MiniLM model itself is mocked: a fake 384-dimensional encoder proves the
construction arithmetic without a download. Real weights are exercised only by
the opt-in model-smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotify_rank.config.feature_settings import TranscriptContextConfig
from slotify_rank.embeddings.minilm_text import (
    TEXT_EMBEDDING_BLOCKS,
    build_constructed_vector,
    cosine_similarity,
)
from slotify_rank.features.transcript import (
    build_transcript_context,
    extract_text_features,
)
from slotify_rank.transcription.segments import build_segments

MINILM_DIMENSION = 384
CONSTRUCTED_DIMENSION = MINILM_DIMENSION * 4
EPISODE = "ep_0123456789ab"
CONFIG = TranscriptContextConfig()


def _segments(spans):
    return build_segments(EPISODE, [(s, e, t, ()) for s, e, t in spans], 120_000)


def _mock_encode(texts, dimension: int = MINILM_DIMENSION, seed: int = 0):
    """Deterministic unit-norm vectors keyed on the text, like a real encoder."""
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32) + seed)
        vector = rng.normal(size=dimension).astype(np.float32)
        vectors.append(vector / np.linalg.norm(vector))
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Context selection
# ---------------------------------------------------------------------------


def test_context_selects_the_nearest_text_either_side():
    segments = _segments(
        [
            (0, 2_000, "First thought here."),
            (2_000, 4_000, "Second thought here."),
            (6_000, 8_000, "Third thought begins."),
        ]
    )
    context = build_transcript_context(segments, 5_000, CONFIG)
    assert "Second thought here." in context.before_text
    assert "Third thought begins." in context.after_text


def test_context_selection_is_deterministic():
    segments = _segments([(0, 2_000, "Alpha."), (4_000, 6_000, "Beta.")])
    first = build_transcript_context(segments, 3_000, CONFIG)
    second = build_transcript_context(segments, 3_000, CONFIG)
    assert first == second


def test_before_context_reads_chronologically():
    segments = _segments(
        [(0, 1_000, "One."), (1_000, 2_000, "Two."), (2_000, 3_000, "Three.")]
    )
    context = build_transcript_context(segments, 3_500, CONFIG)
    assert context.before_text == "One. Two. Three."


def test_character_budget_drops_the_furthest_text():
    segments = _segments(
        [(0, 1_000, "A" * 100), (1_000, 2_000, "B" * 100), (2_000, 3_000, "C" * 100)]
    )
    context = build_transcript_context(
        segments, 3_500, TranscriptContextConfig(max_chars_before=150)
    )
    assert "C" * 100 in context.before_text
    assert "A" * 100 not in context.before_text


def test_segment_budget_is_respected():
    segments = _segments([(i * 1_000, (i + 1) * 1_000, f"S{i}.") for i in range(10)])
    context = build_transcript_context(
        segments, 9_500, TranscriptContextConfig(max_segments_before=2)
    )
    assert len(context.before_segment_ids) <= 2


def test_contributing_segments_are_recorded():
    segments = _segments([(0, 1_000, "Alpha."), (2_000, 3_000, "Beta.")])
    context = build_transcript_context(segments, 1_500, CONFIG)
    assert context.before_segment_ids == (segments[0].segment_id,)
    assert context.after_segment_ids == (segments[1].segment_id,)


def test_temporal_offsets_are_recorded():
    segments = _segments([(0, 1_000, "Alpha."), (3_000, 4_000, "Beta.")])
    context = build_transcript_context(segments, 2_000, CONFIG)
    assert context.gap_before_ms == 1_000
    assert context.gap_after_ms == 1_000


# ---------------------------------------------------------------------------
# Missing context
# ---------------------------------------------------------------------------


def test_no_transcript_yields_both_sides_absent():
    context = build_transcript_context([], 5_000, CONFIG)
    assert context.has_before is False
    assert context.has_after is False
    assert context.before_absent_reason == "no_transcript"


def test_a_candidate_at_the_episode_start_has_no_before_context():
    segments = _segments([(1_000, 2_000, "Only thought.")])
    context = build_transcript_context(segments, 0, CONFIG)
    assert context.has_before is False
    assert context.before_absent_reason == "episode_start"
    assert context.has_after is True


def test_a_candidate_at_the_episode_end_has_no_after_context():
    segments = _segments([(1_000, 2_000, "Only thought.")])
    context = build_transcript_context(segments, 5_000, CONFIG)
    assert context.has_after is False
    assert context.after_absent_reason == "episode_end"


def test_an_excessive_gap_is_treated_as_absent_not_as_adjacent():
    """Text a minute away is not 'the sentence before'."""
    segments = _segments([(0, 1_000, "Long ago."), (90_000, 91_000, "Much later.")])
    context = build_transcript_context(
        segments, 45_000, TranscriptContextConfig(max_gap_ms=10_000)
    )
    assert context.before_absent_reason == "gap_too_large"
    assert context.after_absent_reason == "gap_too_large"


def test_a_candidate_mid_sentence_belongs_to_neither_side():
    """The clearest possible evidence of a bad breakpoint."""
    segments = _segments([(0, 10_000, "One long unbroken sentence here.")])
    context = build_transcript_context(segments, 5_000, CONFIG)
    assert context.has_before is False
    assert context.has_after is False
    assert context.before_absent_reason == "mid_segment"


# ---------------------------------------------------------------------------
# Constructed vector
# ---------------------------------------------------------------------------


def test_the_constructed_vector_is_1536_dimensional():
    """384 x 4 blocks. 1536 is arithmetic, not MiniLM's output width."""
    before, after = _mock_encode(["before text", "after text"])
    vector = build_constructed_vector(before, after)
    assert vector.shape == (CONSTRUCTED_DIMENSION,)
    assert len(TEXT_EMBEDDING_BLOCKS) == 4


def test_minilm_native_dimension_is_384():
    vectors = _mock_encode(["one", "two"])
    assert vectors.shape == (2, 384)


def test_the_four_blocks_are_before_after_absdiff_product():
    before, after = _mock_encode(["a", "b"])
    vector = build_constructed_vector(before, after)
    d = MINILM_DIMENSION
    assert np.allclose(vector[:d], before)
    assert np.allclose(vector[d : 2 * d], after)
    assert np.allclose(vector[2 * d : 3 * d], np.abs(before - after))
    assert np.allclose(vector[3 * d :], before * after)


def test_a_missing_side_yields_no_constructed_vector():
    """Zero-filling would make the product block read as a real signal."""
    before = _mock_encode(["a"])[0]
    assert build_constructed_vector(before, None) is None
    assert build_constructed_vector(None, before) is None


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        build_constructed_vector(np.zeros(384), np.zeros(128))


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def test_identical_text_has_similarity_one():
    vector = _mock_encode(["same text"])[0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_similarity_is_clamped_to_the_valid_range():
    """Floating-point error must never make 1 - cosine negative."""
    vector = _mock_encode(["x"])[0]
    similarity = cosine_similarity(vector, vector)
    assert -1.0 <= similarity <= 1.0
    assert 1.0 - similarity >= 0.0


def test_unrelated_text_is_less_similar_than_identical_text():
    a, b = _mock_encode(["alpha", "beta"])
    assert cosine_similarity(a, b) < cosine_similarity(a, a)


def test_similarity_is_none_when_a_side_is_missing():
    vector = _mock_encode(["x"])[0]
    assert cosine_similarity(vector, None) is None
    assert cosine_similarity(None, None) is None


def test_a_zero_vector_has_no_defined_similarity():
    assert cosine_similarity(np.zeros(384), _mock_encode(["x"])[0]) is None


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batched_encoding_matches_individual_encoding():
    texts = ["alpha", "beta", "gamma", "delta"]
    batched = _mock_encode(texts)
    individually = np.vstack([_mock_encode([text]) for text in texts])
    assert np.allclose(batched, individually)


def test_encoding_no_text_yields_an_empty_matrix():
    assert _mock_encode([]).shape[0] == 0


# ---------------------------------------------------------------------------
# Scalar text features
# ---------------------------------------------------------------------------


def test_word_and_character_counts():
    segments = _segments([(0, 1_000, "One two three."), (2_000, 3_000, "Four five.")])
    context = build_transcript_context(segments, 1_500, CONFIG)
    values, missing = extract_text_features(context, transcript_available=True)
    assert values["words_before"] == 3
    assert values["words_after"] == 2
    assert not missing["words_before"]


def test_missing_context_is_masked_not_zeroed():
    context = build_transcript_context([], 5_000, CONFIG)
    values, missing = extract_text_features(context, transcript_available=False)
    assert missing["words_before"] is True
    assert values["words_before"] == 0.0
    assert values["text_before_available"] == 0.0
    assert missing["text_before_available"] is False


def test_availability_flags_are_never_masked():
    """A masked availability flag would be circular."""
    context = build_transcript_context([], 5_000, CONFIG)
    _, missing = extract_text_features(context, transcript_available=False)
    for name in ("transcript_available", "text_before_available", "text_after_available"):
        assert missing[name] is False


def test_sentence_end_and_punctuation_come_from_the_prior_segment():
    segments = _segments([(0, 1_000, "A complete thought."), (2_000, 3_000, "Next.")])
    context = build_transcript_context(segments, 1_500, CONFIG)
    values, _ = extract_text_features(context, transcript_available=True)
    assert values["transcript_sentence_end"] == 1.0
    assert values["terminal_punctuation_ordinal"] == 5.0


def test_an_unfinished_prior_segment_scores_lower_punctuation():
    segments = _segments([(0, 1_000, "and then we"), (2_000, 3_000, "Next.")])
    context = build_transcript_context(segments, 1_500, CONFIG)
    values, _ = extract_text_features(context, transcript_available=True)
    assert values["transcript_sentence_end"] == 0.0
    assert values["terminal_punctuation_ordinal"] == 0.0


def test_inter_segment_pause_is_measured():
    segments = _segments([(0, 1_000, "Alpha."), (3_000, 4_000, "Beta.")])
    context = build_transcript_context(segments, 2_000, CONFIG)
    values, _ = extract_text_features(context, transcript_available=True)
    assert values["transcript_inter_segment_pause_ms"] == 2_000


def test_text_values_and_mask_share_their_keys():
    segments = _segments([(0, 1_000, "Alpha."), (3_000, 4_000, "Beta.")])
    context = build_transcript_context(segments, 2_000, CONFIG)
    values, missing = extract_text_features(context, transcript_available=True)
    assert set(values) == set(missing)
