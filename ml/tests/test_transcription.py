"""Transcript schema, segment derivation and chunk-boundary reconciliation.

No model is loaded and no network is touched: every test here drives the pure
functions in :mod:`slotify_rank.transcription.segments` with synthetic model
output, which is exactly what makes the chunk-stitching logic testable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slotify_rank.config.versions import TRANSCRIPTION_VERSION
from slotify_rank.pipeline.identity import CacheIdentity
from slotify_rank.transcription.cache import (
    load_transcript,
    read_transcript,
    transcript_identity,
    write_transcript,
)
from slotify_rank.transcription.schema import (
    EpisodeTranscript,
    TranscriptSegment,
    TranscriptWord,
    classify_terminal_punctuation,
    make_segment_id,
)
from slotify_rank.transcription.segments import (
    RawSegment,
    build_segments,
    normalize_text,
    offset_segments,
    plan_chunks,
    reconcile_overlapping_segments,
)

EPISODE = "fixture-episode_0123456789ab"


def _segment(
    index: int, start_ms: int, end_ms: int, text: str = "Hello there."
) -> TranscriptSegment:
    return build_segments(EPISODE, [(start_ms, end_ms, text, ())], end_ms + 1_000)[0]


def _transcript(segments) -> EpisodeTranscript:
    return EpisodeTranscript(
        episode_id=EPISODE,
        language="en",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        audio_sha256="a" * 64,
        audio_duration_ms=60_000,
        segments=tuple(segments),
        identity_digest="d" * 64,
    )


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------


def test_chunks_cover_the_whole_episode():
    chunks = plan_chunks(70_000, 30_000, 5_000)
    assert chunks[0].start_ms == 0
    assert chunks[-1].end_ms == 70_000
    # Every millisecond is inside at least one chunk.
    for boundary in (0, 24_999, 25_000, 49_999, 69_999):
        assert any(c.start_ms <= boundary < c.end_ms for c in chunks)


def test_chunk_stride_accounts_for_overlap():
    chunks = plan_chunks(70_000, 30_000, 5_000)
    assert [c.start_ms for c in chunks] == [0, 25_000, 50_000]


def test_short_audio_yields_one_clipped_chunk():
    assert plan_chunks(4_000, 30_000, 5_000) == plan_chunks(4_000, 30_000, 5_000)
    chunks = plan_chunks(4_000, 30_000, 5_000)
    assert len(chunks) == 1
    assert chunks[0].end_ms == 4_000


def test_overlap_at_least_the_chunk_length_is_rejected():
    """A non-positive stride would loop forever rather than fail."""
    with pytest.raises(ValueError, match="never advance"):
        plan_chunks(70_000, 30_000, 30_000)


# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------


def test_seconds_become_absolute_milliseconds():
    resolved = offset_segments(
        [RawSegment(start_seconds=1.5, end_seconds=2.25, text="hi")], offset_ms=25_000
    )
    assert resolved == [(26_500, 27_250, "hi", ())]


def test_segments_without_a_start_timestamp_are_dropped():
    """An untimed segment cannot be placed on the episode timeline."""
    resolved = offset_segments(
        [RawSegment(start_seconds=None, end_seconds=2.0, text="orphan")], 0
    )
    assert resolved == []


def test_a_missing_end_timestamp_does_not_lose_the_segment():
    resolved = offset_segments(
        [RawSegment(start_seconds=1.0, end_seconds=None, text="tail")], 0
    )
    assert resolved == [(1_000, 1_000, "tail", ())]


def test_empty_text_is_dropped():
    assert offset_segments([RawSegment(0.0, 1.0, "   ")], 0) == []


def test_word_timestamps_are_offset_too():
    resolved = offset_segments(
        [
            RawSegment(
                start_seconds=0.0,
                end_seconds=1.0,
                text="hi there",
                words=(("hi", 0.0, 0.4), ("there", 0.5, 1.0)),
            )
        ],
        offset_ms=10_000,
    )
    words = resolved[0][3]
    assert [(w.word, w.start_ms, w.end_ms) for w in words] == [
        ("hi", 10_000, 10_400),
        ("there", 10_500, 11_000),
    ]


# ---------------------------------------------------------------------------
# Sentence-end detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("That is settled.", "period"),
        ("Is it though?", "question"),
        ("Absolutely!", "exclamation"),
        ("Well, I mean...", "ellipsis"),
        ("He said “stop.”", "period"),
        ('She replied "yes."', "period"),
        ("and then we", "none"),
        ("first, second,", "comma"),
        ("wait --", "dash"),
        ("", "none"),
    ],
)
def test_terminal_punctuation_classification(text, expected):
    assert classify_terminal_punctuation(text) == expected


def test_ellipsis_is_not_classified_as_a_period():
    """A trailing-off is a weaker boundary than a full stop."""
    assert classify_terminal_punctuation("I suppose...") == "ellipsis"
    assert classify_terminal_punctuation("I suppose.") == "period"


def test_sentence_end_is_derived_from_the_raw_text():
    segments = build_segments(
        EPISODE, [(0, 1_000, "Done.", ()), (1_000, 2_000, "and then", ())], 5_000
    )
    assert segments[0].sentence_end is True
    assert segments[1].sentence_end is False


def test_normalized_text_strips_punctuation_and_case():
    assert normalize_text("Hello, THERE!  Really?") == "hello there really"


def test_raw_text_is_preserved_alongside_the_normalized_form():
    segment = _segment(0, 0, 1_000, "Hello, there!")
    assert segment.text == "Hello, there!"
    assert segment.normalized_text == "hello there"


# ---------------------------------------------------------------------------
# Chunk overlap reconciliation
# ---------------------------------------------------------------------------


def test_duplicate_segments_in_an_overlap_collapse_to_one():
    # Same speech seen by chunk 0 (centre 15000) and chunk 1 (centre 40000).
    collected = [
        (26_000, 27_000, "the quick brown fox", (), 0, 15_000.0),
        (26_000, 27_000, "the quick brown fox", (), 1, 40_000.0),
    ]
    reconciled = reconcile_overlapping_segments(collected)
    assert len(reconciled) == 1


def test_the_chunk_that_saw_the_words_whole_wins():
    """A boundary word is better transcribed with context on both sides."""
    # Segment at 26000-27000. Chunk 0 ends at 30000 (centre 15000, distance
    # 11500); chunk 1 spans 25000-55000 (centre 40000, distance 13500).
    # Chunk 0's centre is nearer, so its text is kept.
    collected = [
        (26_000, 27_000, "clipped versionn", (), 0, 15_000.0),
        (26_000, 27_000, "clipped version", (), 1, 40_000.0),
    ]
    reconciled = reconcile_overlapping_segments(collected)
    assert len(reconciled) == 1
    assert reconciled[0][2] == "clipped versionn"


def test_different_speech_in_an_overlap_is_kept_and_de_overlapped():
    """A genuine disagreement must not silently delete speech."""
    collected = [
        (26_000, 27_500, "alpha beta gamma", (), 0, 15_000.0),
        (27_000, 28_000, "completely different words here", (), 1, 40_000.0),
    ]
    reconciled = reconcile_overlapping_segments(collected)
    assert len(reconciled) == 2
    # Ordered and non-overlapping, which the schema requires.
    assert reconciled[0][1] <= reconciled[1][0]


def test_reconciled_segments_are_ordered():
    collected = [
        (5_000, 6_000, "second part", (), 1, 40_000.0),
        (1_000, 2_000, "first part", (), 0, 15_000.0),
    ]
    reconciled = reconcile_overlapping_segments(collected)
    assert [item[0] for item in reconciled] == [1_000, 5_000]


def test_non_overlapping_segments_are_all_kept():
    collected = [
        (0, 1_000, "alpha", (), 0, 15_000.0),
        (2_000, 3_000, "beta", (), 0, 15_000.0),
        (4_000, 5_000, "gamma", (), 0, 15_000.0),
    ]
    assert len(reconcile_overlapping_segments(collected)) == 3


# ---------------------------------------------------------------------------
# Bounds and schema invariants
# ---------------------------------------------------------------------------


def test_segments_are_clipped_to_the_audio_duration():
    """Whisper pads to 30 s and can emit timestamps inside the padding."""
    segments = build_segments(EPISODE, [(9_000, 30_000, "tail end", ())], 10_000)
    assert segments[0].end_ms == 10_000


def test_segments_starting_past_the_audio_are_dropped():
    assert build_segments(EPISODE, [(12_000, 13_000, "padding", ())], 10_000) == []


def test_a_segment_past_the_episode_end_is_rejected_by_the_schema():
    segment = TranscriptSegment(
        segment_id=make_segment_id(EPISODE, 0, 0),
        start_ms=0,
        end_ms=90_000,
        text="too long",
        normalized_text="too long",
        sentence_end=False,
        terminal_punctuation="none",
    )
    with pytest.raises(ValueError, match="offset was applied twice"):
        _transcript([segment])


def test_overlapping_segments_are_rejected_by_the_schema():
    first = _segment(0, 0, 5_000)
    second = TranscriptSegment(
        segment_id=make_segment_id(EPISODE, 1, 3_000),
        start_ms=3_000,
        end_ms=8_000,
        text="overlapping",
        normalized_text="overlapping",
        sentence_end=False,
        terminal_punctuation="none",
    )
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        _transcript([first, second])


def test_an_empty_transcript_is_a_failure_not_a_success():
    with pytest.raises(ValueError, match="refusing to store an empty transcript"):
        _transcript([])


def test_segment_ids_are_deterministic():
    assert make_segment_id(EPISODE, 3, 12_345) == make_segment_id(EPISODE, 3, 12_345)
    assert make_segment_id(EPISODE, 3, 12_345) != make_segment_id(EPISODE, 4, 12_345)


def test_word_end_before_start_is_rejected():
    with pytest.raises(ValueError, match="precedes start_ms"):
        TranscriptWord(word="x", start_ms=500, end_ms=100)


def test_transcribed_duration_excludes_the_silence_between_segments():
    transcript = _transcript(
        build_segments(
            EPISODE, [(0, 1_000, "a.", ()), (10_000, 11_000, "b.", ())], 60_000
        )
    )
    assert transcript.transcribed_duration_ms == 2_000
    assert transcript.audio_duration_ms == 60_000


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def _identity(sha: str = "a" * 64, model: str = "openai/whisper-tiny.en"):
    return transcript_identity(
        episode_id=EPISODE,
        normalized_audio_sha256=sha,
        model_id=model,
        model_revision="main",
        transcription_config_digest="cfg",
        transcription_version=TRANSCRIPTION_VERSION,
        feature_pipeline_version="pipeline-v1",
        library_versions={"transformers": "4.57.6"},
    )


def test_a_matching_transcript_is_a_cache_hit(tmp_path: Path):
    identity = _identity()
    transcript = EpisodeTranscript(
        episode_id=EPISODE,
        language="en",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        audio_sha256="a" * 64,
        audio_duration_ms=60_000,
        segments=tuple(build_segments(EPISODE, [(0, 1_000, "hi.", ())], 60_000)),
        identity_digest=identity.digest,
    )
    path = tmp_path / "t.json"
    write_transcript(path, transcript)
    assert load_transcript(path, identity) is not None


def test_a_different_audio_checksum_invalidates_the_transcript(tmp_path: Path):
    original = _identity()
    transcript = EpisodeTranscript(
        episode_id=EPISODE,
        language="en",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        audio_sha256="a" * 64,
        audio_duration_ms=60_000,
        segments=tuple(build_segments(EPISODE, [(0, 1_000, "hi.", ())], 60_000)),
        identity_digest=original.digest,
    )
    path = tmp_path / "t.json"
    write_transcript(path, transcript)
    assert load_transcript(path, _identity(sha="b" * 64)) is None


def test_a_different_model_invalidates_the_transcript(tmp_path: Path):
    original = _identity()
    transcript = EpisodeTranscript(
        episode_id=EPISODE,
        language="en",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        audio_sha256="a" * 64,
        audio_duration_ms=60_000,
        segments=tuple(build_segments(EPISODE, [(0, 1_000, "hi.", ())], 60_000)),
        identity_digest=original.digest,
    )
    path = tmp_path / "t.json"
    write_transcript(path, transcript)
    assert load_transcript(path, _identity(model="openai/whisper-base.en")) is None


def test_an_absent_transcript_is_a_miss_not_an_error(tmp_path: Path):
    assert load_transcript(tmp_path / "absent.json", _identity()) is None


def test_an_unsupported_transcription_version_is_rejected(tmp_path: Path):
    import json

    path = tmp_path / "t.json"
    transcript = EpisodeTranscript(
        episode_id=EPISODE,
        language="en",
        model_id="openai/whisper-tiny.en",
        model_revision="main",
        audio_sha256="a" * 64,
        audio_duration_ms=60_000,
        segments=tuple(build_segments(EPISODE, [(0, 1_000, "hi.", ())], 60_000)),
        identity_digest="d" * 64,
    )
    write_transcript(path, transcript)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["transcription_version"] = "transcript-v99"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported transcription_version"):
        read_transcript(path)


def test_transcript_round_trips_through_disk(tmp_path: Path):
    transcript = _transcript(
        build_segments(EPISODE, [(0, 1_500, "Hello there.", ())], 60_000)
    )
    path = tmp_path / "t.json"
    write_transcript(path, transcript)
    assert read_transcript(path).to_dict() == transcript.to_dict()


def test_preferring_a_nearer_duplicate_cannot_leave_an_overlap_behind():
    """The regression that failed a whole 22-minute episode.

    Replacing an already-kept segment with a nearer-centre duplicate widens it
    in place, which can push its end past the start of a segment kept after it.
    A trim applied only as each segment arrives never sees that, so the overlap
    survives to `build_segments`, which rejects the entire episode.
    """
    from slotify_rank.transcription.segments import reconcile_overlapping_segments

    # (start, end, text, words, chunk_index, chunk_centre)
    candidates = [
        # Kept first, transcribed near the edge of chunk 0.
        (955_000, 955_100, "he said quietly", (), 0, 940_000),
        # A different sentence, kept after it with no overlap at this point.
        (955_150, 955_400, "and then she left", (), 0, 940_000),
        # The same first sentence from chunk 1, whose centre is much nearer, and
        # whose span is wider -- it ends at 955_360, past the second segment's
        # start of 955_150.
        (955_000, 955_360, "he said quietly", (), 1, 955_100),
    ]
    resolved = reconcile_overlapping_segments(candidates)

    assert resolved, "reconciliation must not drop every segment"
    for previous, following in zip(resolved, resolved[1:]):
        assert following[0] >= previous[1], (
            f"segment starting at {following[0]} overlaps the previous one "
            f"ending at {previous[1]}"
        )
    # The nearer-centre version won, and nothing was silently deleted.
    assert resolved[0][:2] == (955_000, 955_360)
    assert "and then she left" in [segment[2] for segment in resolved]


def test_a_span_wholly_swallowed_by_its_predecessor_is_dropped_not_zero_length():
    from slotify_rank.transcription.segments import reconcile_overlapping_segments

    candidates = [
        (1_000, 5_000, "a long stretch of speech", (), 0, 3_000),
        (2_000, 3_000, "something else entirely", (), 0, 3_000),
    ]
    resolved = reconcile_overlapping_segments(candidates)
    assert all(end > start for start, end, _text, _words in resolved)


def test_reconciliation_always_returns_ordered_non_overlapping_spans():
    """A property, not an example.

    `build_segments` rejects an entire episode when any segment starts before
    the previous one ended, and two real episodes were lost that way. Asserting
    the invariant over randomised overlapping input is what makes the guarantee
    structural rather than a patch for the two cases that happened to be seen.
    """
    import random

    from slotify_rank.transcription.segments import reconcile_overlapping_segments

    phrases = [
        "he said quietly",
        "and then she left",
        "the door closed",
        "he said quietly",  # deliberate duplicates, which is what triggers it
        "nothing happened after that",
    ]
    rng = random.Random(20260829)
    for _ in range(200):
        candidates = []
        for _ in range(rng.randint(2, 12)):
            start = rng.randrange(0, 60_000)
            start -= start % 20
            end = start + rng.randrange(20, 4_000)
            chunk_index = rng.randrange(0, 4)
            candidates.append(
                (
                    start,
                    end,
                    rng.choice(phrases),
                    (),
                    chunk_index,
                    chunk_index * 25_000 + 15_000,
                )
            )

        resolved = reconcile_overlapping_segments(candidates)
        for previous, following in zip(resolved, resolved[1:]):
            assert following[0] >= previous[1], (
                f"{following[0]} starts before {previous[1]} in {resolved}"
            )
        for start, end, _text, _words in resolved:
            assert end > start, "a zero-length span must be dropped, not emitted"
