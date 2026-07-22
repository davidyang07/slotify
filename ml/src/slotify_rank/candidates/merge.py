"""Deterministic merging and de-duplication of raw candidates.

Five generators looking at the same moment will each propose it, so merging is
not an optimisation -- without it the pool is dominated by duplicates and any
per-source statistic is meaningless.

The survivor rule is the **same rule the product uses**
(:func:`slotify_rank.candidates.heuristic.merge_candidates`): prefer the longer
pause; failing that, prefer the one that ends on a sentence boundary. Keeping
the rules aligned means a dataset candidate sits at the timestamp the product
would have chosen, so a label about it is a label about a real product decision.
:func:`assert_agrees_with_product_merge` pins that alignment, and a test calls it.

What differs from the product rule -- deliberately -- is that merging here is
*lossless in provenance*: the survivor absorbs every source flag, the longest
pause and silence seen in the group, any transcript context, and the timestamps
it replaced. Nothing about where a candidate came from is lost.
"""

from __future__ import annotations

from typing import Sequence

from slotify_rank.candidates.audio_candidates import RawCandidate
from slotify_rank.candidates.heuristic import (
    ends_with_sentence_boundary,
    merge_candidates as product_merge_candidates,
)
from slotify_rank.candidates.schema import Candidate

__all__ = ["MergedCandidate", "merge_raw_candidates", "assert_agrees_with_product_merge"]

#: Priority used only to break exact ties, so the result never depends on which
#: generator happened to run first. Ordered by how much evidence the source
#: carries about the moment itself.
_SOURCE_PRIORITY = {
    "silence": 0,
    "transcript_segment_end": 1,
    "pause": 2,
    "rms_minimum": 3,
    "fixed_interval": 4,
}


class MergedCandidate:
    """A survivor plus everything absorbed from the candidates it replaced."""

    __slots__ = (
        "ms",
        "sources",
        "silence_ms",
        "pause_ms",
        "snippet",
        "sentence_end",
        "transcript_segment_id",
        "transcript_before",
        "transcript_after",
        "merged_from_ms",
    )

    def __init__(
        self,
        ms: int,
        sources: tuple[str, ...],
        silence_ms: int,
        pause_ms: int,
        snippet: str,
        sentence_end: bool | None,
        transcript_segment_id: str | None,
        transcript_before: str | None,
        transcript_after: str | None,
        merged_from_ms: tuple[int, ...],
    ):
        self.ms = ms
        self.sources = sources
        self.silence_ms = silence_ms
        self.pause_ms = pause_ms
        self.snippet = snippet
        self.sentence_end = sentence_end
        self.transcript_segment_id = transcript_segment_id
        self.transcript_before = transcript_before
        self.transcript_after = transcript_after
        self.merged_from_ms = merged_from_ms

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MergedCandidate({self.ms}, {self.sources})"


def _sort_key(candidate: RawCandidate) -> tuple[int, int, str]:
    return (
        candidate.ms,
        _SOURCE_PRIORITY.get(candidate.source, len(_SOURCE_PRIORITY)),
        candidate.source,
    )


def _prefer(challenger: RawCandidate, incumbent: RawCandidate) -> bool:
    """The product's survivor rule verbatim (``candidates.ts::mergeCandidates``).

    Note the ``or``: a shorter pause still wins if it is the only one of the two
    that ends on a sentence boundary.
    """
    return challenger.silence_ms > incumbent.silence_ms or (
        ends_with_sentence_boundary(challenger.snippet)
        and not ends_with_sentence_boundary(incumbent.snippet)
    )


def merge_raw_candidates(
    candidates: Sequence[RawCandidate], tolerance_ms: int
) -> list[MergedCandidate]:
    """Collapse candidates within ``tolerance_ms`` of each other.

    Grouping follows the product exactly: candidates are visited in ascending
    timestamp order and each is compared against the *current survivor*, not
    against the previous input, so a run of near-duplicates chains through
    whichever candidate is currently winning. Ties in timestamp are ordered by
    source priority, so the output is a pure function of the input set and
    generator execution order cannot change it.
    """
    if tolerance_ms < 0:
        raise ValueError(f"tolerance_ms must be non-negative, got {tolerance_ms}")
    ordered = sorted(candidates, key=_sort_key)

    groups: list[list[RawCandidate]] = []
    survivors: list[RawCandidate] = []
    for candidate in ordered:
        if survivors and abs(candidate.ms - survivors[-1].ms) <= tolerance_ms:
            groups[-1].append(candidate)
            if _prefer(candidate, survivors[-1]):
                survivors[-1] = candidate
        else:
            groups.append([candidate])
            survivors.append(candidate)

    merged: list[MergedCandidate] = []
    for group, survivor in zip(groups, survivors):
        sources: list[str] = []
        for member in group:
            for source in member.sources:
                if source not in sources:
                    sources.append(source)

        transcript_member = next(
            (member for member in group if member.transcript_segment_id is not None),
            None,
        )
        snippet = survivor.snippet or next(
            (member.snippet for member in group if member.snippet), ""
        )
        sentence_end = survivor.sentence_end
        if sentence_end is None:
            sentence_end = next(
                (
                    member.sentence_end
                    for member in group
                    if member.sentence_end is not None
                ),
                None,
            )

        merged.append(
            MergedCandidate(
                ms=survivor.ms,
                sources=tuple(sources),
                silence_ms=max(member.silence_ms for member in group),
                pause_ms=max(member.pause_ms for member in group),
                snippet=snippet,
                sentence_end=sentence_end,
                transcript_segment_id=(
                    None
                    if transcript_member is None
                    else transcript_member.transcript_segment_id
                ),
                transcript_before=(
                    None if transcript_member is None else transcript_member.transcript_before
                ),
                transcript_after=(
                    None if transcript_member is None else transcript_member.transcript_after
                ),
                merged_from_ms=tuple(
                    sorted({member.ms for member in group if member.ms != survivor.ms})
                ),
            )
        )

    merged.sort(key=lambda candidate: candidate.ms)
    return merged


def assert_agrees_with_product_merge(
    candidates: Sequence[RawCandidate], tolerance_ms: int
) -> None:
    """Fail if dataset merging picks different timestamps than the product's.

    The two implementations exist for different reasons -- one preserves
    provenance, one matches a TypeScript function byte-for-byte -- so this is
    the guard that stops them drifting apart. It is exercised by the test suite
    on generated smoke candidates, not called during normal generation.

    The check is skipped for inputs containing exact timestamp ties, where the
    two are legitimately allowed to differ: the product's chained pairwise merge
    compares each candidate only against the running survivor, while grouping
    here considers the whole cluster.
    """
    timestamps = [candidate.ms for candidate in candidates]
    if len(timestamps) != len(set(timestamps)):
        return
    product = product_merge_candidates(
        [
            Candidate(
                ms=candidate.ms,
                silence_ms=candidate.silence_ms,
                snippet=candidate.snippet,
                sources=(),
            )
            for candidate in sorted(candidates, key=_sort_key)
        ],
        [],
        tolerance_ms,
    )
    ours = merge_raw_candidates(candidates, tolerance_ms)
    product_ms = [candidate.ms for candidate in product]
    our_ms = [candidate.ms for candidate in ours]
    if product_ms != our_ms:
        raise AssertionError(
            "Dataset merging has drifted from the product's mergeCandidates.\n"
            f"  product: {product_ms}\n"
            f"  dataset: {our_ms}"
        )
