"""Turning raw model output into canonical, reconciled transcript segments.

This module is deliberately model-agnostic and free of heavy imports: it takes
plain ``(start_seconds, end_seconds, text)`` triples and produces validated
:class:`~slotify_rank.transcription.schema.TranscriptSegment` objects. That is
what lets the whole chunk-reconciliation story be unit-tested without
downloading a model or touching the network.

**Chunking.** Whisper sees 30-second windows. Long audio is therefore cut into
chunks that are transcribed independently and stitched back together. Two things
go wrong when stitching, and both are handled here:

*Timestamps are chunk-relative.* Every model timestamp is offset by the chunk's
start before anything else happens (:func:`offset_segments`). Forgetting this
produces a transcript where every segment after the first chunk is wrong by a
multiple of the chunk length -- and the text still reads perfectly, so it is
easy to miss.

*Chunks overlap on purpose.* A word cut in half at a chunk boundary is
transcribed badly, so chunks overlap by a few seconds and the model gets to see
each boundary word whole. The cost is that the overlap region is transcribed
twice, producing near-duplicate segments.
:func:`reconcile_overlapping_segments` resolves this by preferring the segment
from the chunk whose *centre* is closer to it -- a boundary word is better
transcribed by the chunk that saw it with context on both sides than by the one
that saw it clipped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from slotify_rank.candidates.heuristic import ends_with_sentence_boundary
from slotify_rank.transcription.schema import (
    TranscriptSegment,
    TranscriptWord,
    classify_terminal_punctuation,
    make_segment_id,
)

__all__ = [
    "RawSegment",
    "ChunkPlan",
    "plan_chunks",
    "normalize_text",
    "offset_segments",
    "reconcile_overlapping_segments",
    "build_segments",
]

_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RawSegment:
    """Model output before validation: seconds, and possibly nonsense.

    Timestamps are ``float`` seconds and may be ``None``: Whisper occasionally
    omits the end timestamp of the final segment in a window. Repairing that is
    :func:`build_segments`' job, not the caller's.
    """

    start_seconds: float | None
    end_seconds: float | None
    text: str
    words: tuple[tuple[str, float | None, float | None], ...] = ()


@dataclass(frozen=True)
class ChunkPlan:
    """One window of audio to transcribe."""

    index: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def centre_ms(self) -> float:
        return (self.start_ms + self.end_ms) / 2.0


def plan_chunks(
    duration_ms: int, chunk_ms: int, overlap_ms: int
) -> list[ChunkPlan]:
    """Cut ``duration_ms`` into overlapping windows.

    The stride is ``chunk_ms - overlap_ms``. Overlap must be strictly less than
    the chunk length, otherwise the stride is non-positive and chunking never
    terminates -- a hang rather than an error, so it is rejected up front.

    The final chunk is clipped to the episode end rather than padded, and a
    trailing chunk that would be entirely inside its predecessor is dropped: it
    would contribute nothing but duplicate segments to reconcile.
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")
    if chunk_ms <= 0:
        raise ValueError(f"chunk_ms must be positive, got {chunk_ms}")
    if overlap_ms < 0:
        raise ValueError(f"overlap_ms must be non-negative, got {overlap_ms}")
    if overlap_ms >= chunk_ms:
        raise ValueError(
            f"overlap_ms ({overlap_ms}) must be strictly less than chunk_ms "
            f"({chunk_ms}); otherwise the chunk stride is non-positive and "
            "chunking would never advance"
        )

    stride = chunk_ms - overlap_ms
    chunks: list[ChunkPlan] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        chunks.append(ChunkPlan(index=index, start_ms=start, end_ms=end))
        if end >= duration_ms:
            break
        start += stride
        index += 1
    return chunks


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used for word counts and matching only. The raw text keeps its punctuation
    because that is precisely the sentence-boundary signal the ranker needs.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).lower()
    without_punctuation = _PUNCTUATION_RE.sub(" ", folded)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


def offset_segments(
    segments: Iterable[RawSegment], offset_ms: int
) -> list[tuple[int, int, str, tuple[TranscriptWord, ...]]]:
    """Convert one chunk's output to absolute integer milliseconds.

    Returns ``(start_ms, end_ms, text, words)`` tuples. Segments with no usable
    start timestamp are dropped -- an untimed segment cannot be placed on the
    episode timeline, and guessing its position would corrupt every candidate
    that joins against it.
    """
    resolved: list[tuple[int, int, str, tuple[TranscriptWord, ...]]] = []
    for segment in segments:
        if segment.start_seconds is None:
            continue
        text = str(segment.text or "").strip()
        if not text:
            continue
        start_ms = offset_ms + int(round(float(segment.start_seconds) * 1000.0))
        if segment.end_seconds is None:
            # Whisper omits the closing timestamp of a trailing segment often
            # enough that dropping those segments would lose real speech. Give it
            # a nominal span instead and let the caller clip it to the audio.
            end_ms = start_ms
        else:
            end_ms = offset_ms + int(round(float(segment.end_seconds) * 1000.0))
        start_ms = max(0, start_ms)
        end_ms = max(start_ms, end_ms)

        words: list[TranscriptWord] = []
        for word_text, word_start, word_end in segment.words:
            if word_start is None:
                continue
            word_start_ms = max(0, offset_ms + int(round(float(word_start) * 1000.0)))
            word_end_ms = (
                word_start_ms
                if word_end is None
                else offset_ms + int(round(float(word_end) * 1000.0))
            )
            words.append(
                TranscriptWord(
                    word=str(word_text).strip(),
                    start_ms=word_start_ms,
                    end_ms=max(word_start_ms, word_end_ms),
                )
            )
        resolved.append((start_ms, end_ms, text, tuple(words)))
    return resolved


#: Character-similarity above which two overlap transcriptions are the same
#: speech. Character-level rather than token-level because the two decodings of
#: an overlap region typically differ by a contraction, a plural or a single
#: mis-heard letter -- and on a two-word segment, one differing token drops
#: token-set overlap to 0.5 while the strings are visibly identical.
_DUPLICATE_SIMILARITY = 0.75


def _similar(first: str, second: str) -> bool:
    """True when two normalized texts are near-duplicates."""
    from difflib import SequenceMatcher

    if not first or not second:
        return False
    if first == second:
        return True
    return SequenceMatcher(None, first, second).ratio() >= _DUPLICATE_SIMILARITY


def reconcile_overlapping_segments(
    candidates: Sequence[tuple[int, int, str, tuple[TranscriptWord, ...], int, float]],
) -> list[tuple[int, int, str, tuple[TranscriptWord, ...]]]:
    """Drop duplicate segments produced by overlapping chunks.

    Each input carries its originating chunk index and that chunk's centre. When
    two segments overlap in time *and* say roughly the same thing, the one from
    the chunk whose centre is nearer is kept: a segment near the edge of its
    window was transcribed with context missing on one side, and the other chunk
    saw it whole.

    Segments that overlap in time but say *different* things are both kept and
    the later one is trimmed to start where the earlier one ended. That case is
    a genuine model disagreement, not a duplicate, and silently discarding one
    side would delete speech.

    Ordering is re-established once at the end rather than assumed during the
    walk. Preferring a nearer-centre duplicate replaces an already-kept
    segment's span in place, which can push its end past the start of a segment
    kept after it -- so a trim applied only as each segment arrives leaves an
    overlap behind, and ``build_segments`` then rejects the whole episode. This
    is not hypothetical: it is what made a 22-minute dramatic reading fail with
    a 360 ms overlap after the rest of the corpus transcribed cleanly.
    """
    ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
    kept: list[list] = []
    for start_ms, end_ms, text, words, _chunk_index, chunk_centre in ordered:
        normalized = normalize_text(text)
        duplicate_of = None
        for existing in kept:
            if existing[1] <= start_ms:  # no temporal overlap
                continue
            if _similar(normalize_text(existing[2]), normalized):
                duplicate_of = existing
                break

        if duplicate_of is not None:
            incumbent_distance = abs(
                (duplicate_of[0] + duplicate_of[1]) / 2.0 - duplicate_of[5]
            )
            challenger_distance = abs((start_ms + end_ms) / 2.0 - chunk_centre)
            if challenger_distance < incumbent_distance:
                duplicate_of[:] = [
                    start_ms,
                    end_ms,
                    text,
                    words,
                    _chunk_index,
                    chunk_centre,
                ]
            continue

        kept.append([start_ms, end_ms, text, words, _chunk_index, chunk_centre])

    return _enforce_ordering(kept)


def _enforce_ordering(
    kept: Sequence[Sequence],
) -> list[tuple[int, int, str, tuple[TranscriptWord, ...]]]:
    """Sort by start and trim each span to begin where the previous one ended.

    A span trimmed to nothing is dropped: it was wholly inside its predecessor,
    so its text is already covered, and emitting a zero-length segment would
    only move the failure downstream. Text is never edited -- only the span --
    because a timestamp that is 200 ms generous is a smaller error than a
    sentence that disappears.
    """
    resolved: list[tuple[int, int, str, tuple[TranscriptWord, ...]]] = []
    previous_end = 0
    for start_ms, end_ms, text, words, *_ in sorted(
        kept, key=lambda item: (item[0], item[1])
    ):
        start = max(int(start_ms), previous_end)
        end = max(int(end_ms), start)
        if end <= start:
            continue
        resolved.append((start, end, text, words))
        previous_end = end
    return resolved


def build_segments(
    episode_id: str,
    resolved: Sequence[tuple[int, int, str, tuple[TranscriptWord, ...]]],
    audio_duration_ms: int,
) -> list[TranscriptSegment]:
    """Validate, clip and label reconciled spans as canonical segments.

    Clipping to ``audio_duration_ms`` is not defensive padding: Whisper pads its
    final 30-second window with silence and will occasionally emit a timestamp
    inside that padding, past the real end of the audio. Those are clipped;
    anything starting beyond the end entirely is dropped, since it describes
    padding rather than speech.
    """
    if audio_duration_ms <= 0:
        raise ValueError(
            f"{episode_id}: audio_duration_ms must be positive, got "
            f"{audio_duration_ms}"
        )

    segments: list[TranscriptSegment] = []
    index = 0
    for start_ms, end_ms, text, words in resolved:
        if start_ms >= audio_duration_ms:
            continue
        clipped_end = min(end_ms, audio_duration_ms)
        if clipped_end < start_ms:
            continue
        stripped = str(text).strip()
        if not stripped:
            continue
        kept_words = tuple(
            word
            for word in words
            if word.start_ms < audio_duration_ms
        )
        segments.append(
            TranscriptSegment(
                segment_id=make_segment_id(episode_id, index, start_ms),
                start_ms=start_ms,
                end_ms=clipped_end,
                text=stripped,
                normalized_text=normalize_text(stripped),
                sentence_end=ends_with_sentence_boundary(stripped),
                terminal_punctuation=classify_terminal_punctuation(stripped),
                words=tuple(
                    TranscriptWord(
                        word=word.word,
                        start_ms=word.start_ms,
                        end_ms=min(word.end_ms, audio_duration_ms),
                    )
                    for word in kept_words
                ),
            )
        )
        index += 1
    return segments
