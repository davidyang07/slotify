"""Deterministic transcript context either side of a candidate.

The question this answers is "what was being said just before this moment, and
what is said just after?" -- because the strongest evidence that a break is
natural is that one thought finished and another began.

**Determinism.** Context selection walks the segment list by index with fixed
bounds. It never depends on iteration order, floating-point comparison of
timestamps, or how many candidates were processed first. The same candidate and
the same transcript always yield the same two strings.

**Bounded gaps.** A candidate sitting in a 90-second musical interlude has no
"sentence before" in any meaningful sense; the nearest segment is a minute away
and about something else. Context beyond ``max_gap_ms`` is treated as absent and
masked, rather than pulled in and labelled as adjacent. Without that bound the
model would learn from text that has nothing to do with the breakpoint.

**No label leakage.** Only the transcript and the candidate timestamp are read.
Nothing in this module can see a rating, an acceptability flag, or a split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from slotify_rank.config.feature_settings import TranscriptContextConfig
from slotify_rank.transcription.schema import TranscriptSegment

__all__ = ["TranscriptContext", "build_transcript_context", "extract_text_features"]


@dataclass(frozen=True)
class TranscriptContext:
    """Bounded text either side of a candidate, plus its provenance."""

    before_text: str
    after_text: str
    before_segment_ids: tuple[str, ...] = ()
    after_segment_ids: tuple[str, ...] = ()
    #: Milliseconds from the candidate back to the end of the last segment
    #: before it (and forward to the start of the first after). ``None`` when
    #: that side has no usable context.
    gap_before_ms: int | None = None
    gap_after_ms: int | None = None
    #: The segment immediately preceding the candidate, which is what carries
    #: the sentence-boundary signal.
    prior_segment: TranscriptSegment | None = None
    following_segment: TranscriptSegment | None = None
    #: Why a side is empty, when it is. Distinguishes "the episode has no
    #: transcript" from "the candidate is at the very start" from "the nearest
    #: speech is too far away" -- three different situations that a single
    #: empty string would flatten into one.
    before_absent_reason: str | None = None
    after_absent_reason: str | None = None

    @property
    def has_before(self) -> bool:
        return bool(self.before_text.strip())

    @property
    def has_after(self) -> bool:
        return bool(self.after_text.strip())


def _accumulate(
    segments: Sequence[TranscriptSegment],
    indices: Sequence[int],
    max_chars: int,
    max_segments: int,
) -> tuple[str, tuple[str, ...]]:
    """Join segment texts in ``indices`` order until a bound is hit.

    ``indices`` is given in *proximity* order (nearest the candidate first) so
    the truncation drops the furthest text, but the returned string is
    reassembled in chronological order so it reads correctly.
    """
    chosen: list[int] = []
    used_chars = 0
    for index in indices[:max_segments]:
        text = segments[index].text.strip()
        if not text:
            continue
        # Always take the nearest segment even if it alone exceeds the budget;
        # an empty "before" for a long sentence would be worse than a long one.
        if chosen and used_chars + len(text) + 1 > max_chars:
            break
        chosen.append(index)
        used_chars += len(text) + 1

    if not chosen:
        return ("", ())
    chosen.sort()
    joined = " ".join(segments[index].text.strip() for index in chosen)
    if len(joined) > max_chars:
        # Trim from the left: the words nearest the candidate matter most.
        joined = joined[-max_chars:].lstrip()
    return (joined, tuple(segments[index].segment_id for index in chosen))


def build_transcript_context(
    segments: Sequence[TranscriptSegment],
    timestamp_ms: int,
    config: TranscriptContextConfig,
) -> TranscriptContext:
    """Select bounded text before and after ``timestamp_ms``.

    A segment counts as "before" when it *ends* at or before the candidate and
    as "after" when it *starts* at or after it. A segment straddling the
    candidate -- the candidate lands mid-sentence -- belongs to neither: it is
    the clearest possible evidence of a bad breakpoint, and splitting its text
    across both sides would disguise that as a clean boundary. It is reported
    through ``before_absent_reason``/``after_absent_reason`` instead.
    """
    if not segments:
        return TranscriptContext(
            before_text="",
            after_text="",
            before_absent_reason="no_transcript",
            after_absent_reason="no_transcript",
        )

    ordered = list(segments)
    before_indices = [
        index
        for index, segment in enumerate(ordered)
        if segment.end_ms <= timestamp_ms
    ]
    after_indices = [
        index
        for index, segment in enumerate(ordered)
        if segment.start_ms >= timestamp_ms
    ]
    straddling = any(
        segment.start_ms < timestamp_ms < segment.end_ms for segment in ordered
    )

    before_reason: str | None = None
    after_reason: str | None = None
    prior = ordered[before_indices[-1]] if before_indices else None
    following = ordered[after_indices[0]] if after_indices else None

    gap_before = None if prior is None else timestamp_ms - prior.end_ms
    gap_after = None if following is None else following.start_ms - timestamp_ms

    before_text, before_ids = ("", ())
    if prior is None:
        before_reason = "mid_segment" if straddling else "episode_start"
    elif gap_before is not None and gap_before > config.max_gap_ms:
        before_reason = "gap_too_large"
        gap_before = None
        prior = None
    else:
        before_text, before_ids = _accumulate(
            ordered,
            list(reversed(before_indices)),
            config.max_chars_before,
            config.max_segments_before,
        )
        if not before_text:
            before_reason = "empty_segments"

    after_text, after_ids = ("", ())
    if following is None:
        after_reason = "mid_segment" if straddling else "episode_end"
    elif gap_after is not None and gap_after > config.max_gap_ms:
        after_reason = "gap_too_large"
        gap_after = None
        following = None
    else:
        after_text, after_ids = _accumulate(
            ordered,
            after_indices,
            config.max_chars_after,
            config.max_segments_after,
        )
        if not after_text:
            after_reason = "empty_segments"

    return TranscriptContext(
        before_text=before_text,
        after_text=after_text,
        before_segment_ids=before_ids,
        after_segment_ids=after_ids,
        gap_before_ms=gap_before,
        gap_after_ms=gap_after,
        prior_segment=prior,
        following_segment=following,
        before_absent_reason=before_reason,
        after_absent_reason=after_reason,
    )


#: Punctuation categories encoded as an ordinal feature. Ordered from strongest
#: boundary to weakest so the numeric ordering is meaningful to a linear model.
_PUNCTUATION_ORDINAL: dict[str, float] = {
    "period": 5.0,
    "question": 4.0,
    "exclamation": 4.0,
    "ellipsis": 3.0,
    "dash": 2.0,
    "comma": 1.0,
    "none": 0.0,
}


def extract_text_features(
    context: TranscriptContext, transcript_available: bool
) -> tuple[dict[str, float], dict[str, bool]]:
    """Scalar text features derived from the selected context.

    The cosine-similarity and semantic-change features are *not* computed here:
    they need the MiniLM vectors and are added during assembly. This function
    covers everything derivable from the text alone.
    """
    values: dict[str, float] = {}
    missing: dict[str, bool] = {}

    def put(name: str, value: float | None) -> None:
        values[name] = 0.0 if value is None else float(value)
        missing[name] = value is None

    put(
        "words_before",
        len(context.before_text.split()) if context.has_before else None,
    )
    put("words_after", len(context.after_text.split()) if context.has_after else None)
    put("chars_before", len(context.before_text) if context.has_before else None)
    put("chars_after", len(context.after_text) if context.has_after else None)

    prior = context.prior_segment
    following = context.following_segment
    put("prior_segment_duration_ms", None if prior is None else prior.duration_ms)
    put(
        "following_segment_duration_ms",
        None if following is None else following.duration_ms,
    )
    put("transcript_gap_before_ms", context.gap_before_ms)
    put("transcript_gap_after_ms", context.gap_after_ms)

    if prior is not None and following is not None:
        put("transcript_inter_segment_pause_ms", following.start_ms - prior.end_ms)
    else:
        put("transcript_inter_segment_pause_ms", None)

    put(
        "transcript_sentence_end",
        None if prior is None else (1.0 if prior.sentence_end else 0.0),
    )
    put(
        "terminal_punctuation_ordinal",
        None
        if prior is None
        else _PUNCTUATION_ORDINAL.get(prior.terminal_punctuation, 0.0),
    )

    # Availability indicators. Never masked -- their whole job is to state
    # whether the other features are present, so a masked availability flag
    # would be circular.
    values["transcript_available"] = 1.0 if transcript_available else 0.0
    missing["transcript_available"] = False
    values["text_before_available"] = 1.0 if context.has_before else 0.0
    missing["text_before_available"] = False
    values["text_after_available"] = 1.0 if context.has_after else 0.0
    missing["text_after_available"] = False

    if set(values) != set(missing):
        raise AssertionError(
            "text values and mask disagree on keys: "
            f"{sorted(set(values) ^ set(missing))}"
        )
    return values, missing
