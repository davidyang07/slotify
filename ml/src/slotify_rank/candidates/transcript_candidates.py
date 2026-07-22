"""Candidates from a supplied timestamped transcript.

Phase 2 does **not** generate transcripts -- no Whisper, no downloads. If the
operator has a timestamped transcript, this reads it; if not, the generator
contributes nothing and the run summary says so explicitly, rather than the
absence being invisible.

Accepted shapes (all are things real ASR tools emit):

.. code-block:: json

    {"segments": [{"id": 0, "start": 12.5, "end": 17.25, "text": "..."}]}
    [{"start_ms": 12500, "end_ms": 17250, "text": "..."}]

``start``/``end`` in seconds and ``start_ms``/``end_ms`` in milliseconds are both
understood; milliseconds win where both are present, since they need no rounding.
A candidate is placed at each segment *end*, carrying the text before and after
so the labelling UI can show context and the scorer can see whether the segment
finished a sentence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from slotify_rank.candidates.audio_candidates import RawCandidate
from slotify_rank.candidates.config import TranscriptConfig
from slotify_rank.candidates.heuristic import ends_with_sentence_boundary

__all__ = ["TranscriptSegment", "load_transcript", "transcript_candidates"]


class TranscriptSegment:
    """One timestamped segment. Immutable, with millisecond bounds."""

    __slots__ = ("segment_id", "start_ms", "end_ms", "text")

    def __init__(self, segment_id: str, start_ms: int, end_ms: int, text: str):
        if start_ms < 0 or end_ms < 0:
            raise ValueError(f"segment {segment_id}: timestamps must be non-negative")
        if end_ms < start_ms:
            raise ValueError(
                f"segment {segment_id}: end ({end_ms} ms) precedes start ({start_ms} ms)"
            )
        self.segment_id = segment_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TranscriptSegment({self.segment_id!r}, {self.start_ms}, "
            f"{self.end_ms}, {self.text!r})"
        )


def _to_ms(raw: Mapping[str, Any], ms_key: str, seconds_key: str, where: str) -> int:
    if raw.get(ms_key) is not None:
        value = raw[ms_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: {ms_key!r} must be numeric, got {value!r}")
        return int(round(float(value)))
    if raw.get(seconds_key) is not None:
        value = raw[seconds_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: {seconds_key!r} must be numeric, got {value!r}")
        return int(round(float(value) * 1000))
    raise KeyError(f"{where}: needs either {ms_key!r} or {seconds_key!r}")


def load_transcript(path: Path | str) -> list[TranscriptSegment]:
    """Parse a transcript file into ordered segments."""
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Transcript not found: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_path} is not valid JSON: {error}") from error

    if isinstance(raw, Mapping):
        raw_segments: Any = raw.get("segments")
        if raw_segments is None:
            raise ValueError(
                f"{file_path}: object form must have a 'segments' key holding a list"
            )
    else:
        raw_segments = raw
    if not isinstance(raw_segments, Iterable) or isinstance(raw_segments, (str, bytes)):
        raise ValueError(f"{file_path}: segments must be a list")

    segments: list[TranscriptSegment] = []
    for index, entry in enumerate(raw_segments):
        where = f"{file_path} segment[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be an object")
        start_ms = _to_ms(entry, "start_ms", "start", where)
        end_ms = _to_ms(entry, "end_ms", "end", where)
        segments.append(
            TranscriptSegment(
                segment_id=str(entry.get("id", index)),
                start_ms=start_ms,
                end_ms=end_ms,
                text=str(entry.get("text", "") or "").strip(),
            )
        )
    segments.sort(key=lambda segment: (segment.start_ms, segment.end_ms))
    return segments


def transcript_candidates(
    segments: Sequence[TranscriptSegment], config: TranscriptConfig
) -> list[RawCandidate]:
    """One candidate at the end of each segment, with surrounding text.

    The final segment is skipped: its end is the end of the episode, which the
    edge guard would discard anyway, and it has no "after" context to show an
    annotator.
    """
    candidates: list[RawCandidate] = []
    for index, segment in enumerate(segments[:-1]):
        before = segment.text[-config.context_chars :] if segment.text else ""
        after_segment = segments[index + 1]
        after = (
            after_segment.text[: config.context_chars] if after_segment.text else ""
        )
        candidates.append(
            RawCandidate(
                ms=segment.end_ms,
                source="transcript_segment_end",
                snippet=before,
                sentence_end=ends_with_sentence_boundary(before) if before else None,
                transcript_segment_id=segment.segment_id,
                transcript_before=before or None,
                transcript_after=after or None,
            )
        )
    return candidates
