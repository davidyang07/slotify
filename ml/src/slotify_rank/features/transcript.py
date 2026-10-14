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

**Mid-segment candidates.** A candidate strictly inside a segment interrupts
that segment's speech, which is the clearest evidence there is of a bad
breakpoint, and it is reported as exactly that: ``transcript_sentence_end`` 0,
no pause, gaps of 0. The straddled segment's text is *split at the candidate* --
by word timings when the segment has them, otherwise in proportion to the time
elapsed, snapped to a word boundary -- and its first part closes the "before"
context while the rest opens the "after" context, so an annotator (the labelling
UI uses this same function) and the text encoder both see the sentence being
cut. Reading the boundary signal off the segment *before* the straddled one, as
featurepipeline-v1.0.0 did, made a mid-sentence candidate look like a clean
sentence end and dropped the interrupted sentence from both sides.

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
    #: the sentence-boundary signal. ``None`` when the candidate is inside a
    #: segment: then nothing ends at the candidate (see ``straddling_segment``).
    prior_segment: TranscriptSegment | None = None
    following_segment: TranscriptSegment | None = None
    #: Why a side is empty, when it is. Distinguishes "the episode has no
    #: transcript" from "the candidate is at the very start" from "the nearest
    #: speech is too far away" from "the candidate is inside a segment and that
    #: side of the cut holds no words" -- situations that a single empty string
    #: would flatten into one.
    before_absent_reason: str | None = None
    after_absent_reason: str | None = None
    #: The segment the candidate falls strictly inside, when it does. Its text
    #: is split across ``before_text``/``after_text`` and its id appears in both
    #: id lists (for whichever side received words).
    straddling_segment: TranscriptSegment | None = None
    #: The candidate timestamp this context was selected for.
    timestamp_ms: int | None = None

    @property
    def has_before(self) -> bool:
        return bool(self.before_text.strip())

    @property
    def has_after(self) -> bool:
        return bool(self.after_text.strip())

    @property
    def mid_segment(self) -> bool:
        """True when the candidate interrupts a segment -- mid-sentence."""
        return self.straddling_segment is not None


def _split_at(segment: TranscriptSegment, timestamp_ms: int) -> tuple[str, str]:
    """Split a straddled segment's text at ``timestamp_ms``: ``(head, tail)``.

    With word timings that line up with the text's words, every word that has
    started before the candidate goes to the head (a word the candidate cuts in
    half was begun before it). Otherwise the cut is placed in proportion to the
    time elapsed through the segment, measured in characters and snapped to the
    nearest word boundary -- speech rate is far closer to constant per
    character than per word. Whitespace inside the text is normalized to single
    spaces, as the context strings are joined with them anyway.
    """
    tokens = segment.text.split()
    if not tokens:
        return ("", "")
    if segment.words and len(segment.words) == len(tokens):
        cut = sum(1 for word in segment.words if word.start_ms < timestamp_ms)
    else:
        span = max(1, segment.end_ms - segment.start_ms)
        fraction = min(1.0, max(0.0, (timestamp_ms - segment.start_ms) / span))
        total_chars = sum(len(token) for token in tokens) + len(tokens) - 1
        target = fraction * total_chars
        cut = 0
        offset = 0
        for token in tokens:
            if offset + len(token) / 2.0 >= target:
                break
            cut += 1
            offset += len(token) + 1
    return (" ".join(tokens[:cut]), " ".join(tokens[cut:]))


def _accumulate(
    pieces: Sequence[tuple[str, str]],
    max_chars: int,
    max_segments: int,
    before: bool,
) -> tuple[str, tuple[str, ...]]:
    """Join ``(segment_id, text)`` pieces until a bound is hit.

    ``pieces`` is given in *proximity* order (nearest the candidate first) so
    the truncation drops the furthest text, but the returned string is
    reassembled in chronological order so it reads correctly -- reversed for
    the ``before`` side, as given for the after side.
    """
    chosen: list[tuple[str, str]] = []
    used_chars = 0
    for segment_id, raw_text in pieces[:max_segments]:
        text = raw_text.strip()
        if not text:
            continue
        # Always take the nearest piece even if it alone exceeds the budget;
        # an empty "before" for a long sentence would be worse than a long one.
        if chosen and used_chars + len(text) + 1 > max_chars:
            break
        chosen.append((segment_id, text))
        used_chars += len(text) + 1

    if not chosen:
        return ("", ())
    if before:
        chosen.reverse()
    joined = " ".join(text for _, text in chosen)
    if len(joined) > max_chars:
        # Trim the end furthest from the candidate: the words nearest it matter
        # most. That is the start of the "before" text and the end of the
        # "after" text.
        joined = (
            joined[-max_chars:].lstrip() if before else joined[:max_chars].rstrip()
        )
    return (joined, tuple(segment_id for segment_id, _ in chosen))


def build_transcript_context(
    segments: Sequence[TranscriptSegment],
    timestamp_ms: int,
    config: TranscriptContextConfig,
) -> TranscriptContext:
    """Select bounded text before and after ``timestamp_ms``.

    A segment counts as "before" when it *ends* at or before the candidate and
    as "after" when it *starts* at or after it. A segment straddling the
    candidate -- the candidate lands mid-sentence -- is split at the candidate
    (:func:`_split_at`): its first part is the nearest "before" text and its
    remainder the nearest "after" text, both at a gap of 0 ms, and the context
    is marked :attr:`TranscriptContext.mid_segment` so the sentence-boundary
    features say "interrupted" rather than borrowing the boundary of the
    segment before it. A side the split leaves with no words at all reports
    ``mid_segment`` as its absent reason.
    """
    if not segments:
        return TranscriptContext(
            before_text="",
            after_text="",
            before_absent_reason="no_transcript",
            after_absent_reason="no_transcript",
            timestamp_ms=timestamp_ms,
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
    straddlers = [
        segment
        for segment in ordered
        if segment.start_ms < timestamp_ms < segment.end_ms
    ]
    if straddlers:
        # Reconciled segments never overlap, so there is at most one; should
        # unreconciled input overlap, the most recently started is nearest.
        straddler = max(straddlers, key=lambda s: (s.start_ms, s.end_ms, s.segment_id))
        return _mid_segment_context(
            ordered, before_indices, after_indices, straddler, timestamp_ms, config
        )

    before_reason: str | None = None
    after_reason: str | None = None
    prior = ordered[before_indices[-1]] if before_indices else None
    following = ordered[after_indices[0]] if after_indices else None

    gap_before = None if prior is None else timestamp_ms - prior.end_ms
    gap_after = None if following is None else following.start_ms - timestamp_ms

    before_text, before_ids = ("", ())
    if prior is None:
        before_reason = "episode_start"
    elif gap_before is not None and gap_before > config.max_gap_ms:
        before_reason = "gap_too_large"
        gap_before = None
        prior = None
    else:
        before_text, before_ids = _accumulate(
            [
                (ordered[index].segment_id, ordered[index].text)
                for index in reversed(before_indices)
            ],
            config.max_chars_before,
            config.max_segments_before,
            before=True,
        )
        if not before_text:
            before_reason = "empty_segments"

    after_text, after_ids = ("", ())
    if following is None:
        after_reason = "episode_end"
    elif gap_after is not None and gap_after > config.max_gap_ms:
        after_reason = "gap_too_large"
        gap_after = None
        following = None
    else:
        after_text, after_ids = _accumulate(
            [
                (ordered[index].segment_id, ordered[index].text)
                for index in after_indices
            ],
            config.max_chars_after,
            config.max_segments_after,
            before=False,
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
        timestamp_ms=timestamp_ms,
    )


def _mid_segment_context(
    ordered: Sequence[TranscriptSegment],
    before_indices: Sequence[int],
    after_indices: Sequence[int],
    straddler: TranscriptSegment,
    timestamp_ms: int,
    config: TranscriptContextConfig,
) -> TranscriptContext:
    """Context for a candidate inside ``straddler``: its text split at the cut.

    Speech runs through the candidate, so both gaps are 0 and the bounded-gap
    rule has nothing to measure; further segments are chained on by the usual
    character and segment budgets, nearest first.
    """
    head, tail = _split_at(straddler, timestamp_ms)
    before_text, before_ids = _accumulate(
        [(straddler.segment_id, head)]
        + [
            (ordered[index].segment_id, ordered[index].text)
            for index in reversed(before_indices)
        ],
        config.max_chars_before,
        config.max_segments_before,
        before=True,
    )
    after_text, after_ids = _accumulate(
        [(straddler.segment_id, tail)]
        + [(ordered[index].segment_id, ordered[index].text) for index in after_indices],
        config.max_chars_after,
        config.max_segments_after,
        before=False,
    )
    return TranscriptContext(
        before_text=before_text,
        after_text=after_text,
        before_segment_ids=before_ids,
        after_segment_ids=after_ids,
        gap_before_ms=0,
        gap_after_ms=0,
        prior_segment=None,
        following_segment=None,
        before_absent_reason=None if before_text else "mid_segment",
        after_absent_reason=None if after_text else "mid_segment",
        straddling_segment=straddler,
        timestamp_ms=timestamp_ms,
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

    put("transcript_gap_before_ms", context.gap_before_ms)
    put("transcript_gap_after_ms", context.gap_after_ms)

    straddler = context.straddling_segment
    if straddler is not None:
        # Mid-sentence. The speech spans on either side of the cut are the two
        # parts of the interrupted segment; there is no pause between them,
        # nothing ends at the candidate, and no punctuation marks the cut.
        cut = (
            context.timestamp_ms
            if context.timestamp_ms is not None
            else (straddler.start_ms + straddler.end_ms) // 2
        )
        put("prior_segment_duration_ms", cut - straddler.start_ms)
        put("following_segment_duration_ms", straddler.end_ms - cut)
        put("transcript_inter_segment_pause_ms", 0)
        put("transcript_sentence_end", 0.0)
        put("terminal_punctuation_ordinal", _PUNCTUATION_ORDINAL["none"])
    else:
        prior = context.prior_segment
        following = context.following_segment
        put("prior_segment_duration_ms", None if prior is None else prior.duration_ms)
        put(
            "following_segment_duration_ms",
            None if following is None else following.duration_ms,
        )
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
