"""Canonical Python port of the ``heuristic_offline_v1`` baseline.

Source of truth (the running product code):

===========================================  ===================================
TypeScript                                   Python
===========================================  ===================================
``lib/candidates.ts::mergeCandidates``        :func:`merge_candidates`
``lib/candidates.ts::scoreCandidate``         :func:`score_candidate`
``lib/candidates.ts::selectTopSlots``         :func:`select_top_slots`
``lib/text.ts::endsWithSentenceBoundary``     :func:`ends_with_sentence_boundary`
``routes/insert-sections.ts:136-260``         :func:`rank_episode`
===========================================  ===================================

The divergent LLM-driven path in ``backend/ad_inserter/cli.py`` is **not** part
of this baseline; its constants are recorded under the ``legacy_cli_v1`` profile
purely so the difference stays visible.

Two properties are load-bearing and are enforced by tests:

1. **Bit-exact parity with TypeScript.** Arithmetic is written in the same order
   as the original, and JavaScript rounding semantics are used via
   :mod:`slotify_rank.jsnum`.
2. **Deterministic tie-breaking**: higher total score, then earlier timestamp,
   then candidate ID ascending. The third key cannot be reached after merging
   (merge guarantees candidates are more than ``min_gap_ms`` apart, so two
   candidates cannot share a timestamp); it exists so the ordering is total
   regardless of input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from slotify_rank.config.settings import HeuristicConfig, HeuristicProfile, load_heuristic_config
from slotify_rank.jsnum import js_round, js_to_fixed
from slotify_rank.candidates.schema import (
    Candidate,
    CandidateRecord,
    ComponentScores,
    EpisodeInput,
    EpisodeRanking,
    InsertionMode,
    SelectionOrigin,
    SlotRecord,
    make_candidate_id,
)

__all__ = [
    "ends_with_sentence_boundary",
    "merge_candidates",
    "score_candidate",
    "select_top_slots",
    "rank_episode",
    "ScoredCandidate",
    "SelectionEntry",
]

# Port of backend/src/lib/text.ts:35. Applied to the *trimmed* string, exactly as
# the TypeScript does (`String(text ?? "").trim()`).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]?\s*$")


def ends_with_sentence_boundary(text: str | None) -> bool:
    """True when ``text`` ends with terminal punctuation, optionally quoted."""
    return bool(_SENTENCE_BOUNDARY_RE.search(str(text if text is not None else "").strip()))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Port of ``lib/text.ts::clamp`` -- ``Math.min(max, Math.max(min, value))``."""
    return min(maximum, max(minimum, value))


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with its score and the component breakdown behind it."""

    candidate: Candidate
    score: float
    components: ComponentScores
    sentence_end: bool | None
    normalized_position: float | None


@dataclass(frozen=True)
class SelectionEntry:
    """One entry chosen by :func:`select_top_slots`.

    ``origin`` distinguishes a real candidate from the two kinds of padding the
    product invents when it cannot find enough candidates. Padding must never be
    treated as a prediction during evaluation.
    """

    ms: int
    silence_ms: int
    snippet: str
    score: float
    origin: SelectionOrigin


def merge_candidates(
    base: Sequence[Candidate],
    extra: Sequence[Candidate],
    min_gap_ms: int,
) -> list[Candidate]:
    """Port of ``mergeCandidates``.

    Candidates within ``min_gap_ms`` collapse to one. The incumbent is replaced
    when the newcomer has a longer pause, or ends on a sentence boundary while
    the incumbent does not. ``sorted`` is stable in Python and ``Array.prototype
    .sort`` is stable in V8, so equal timestamps preserve ``base``-then-``extra``
    order in both languages.
    """
    combined = sorted([*base, *extra], key=lambda candidate: candidate.ms)
    merged: list[Candidate] = []
    for candidate in combined:
        if not merged:
            merged.append(candidate)
            continue
        last = merged[-1]
        if abs(candidate.ms - last.ms) > min_gap_ms:
            merged.append(candidate)
            continue
        prefer = candidate.silence_ms > last.silence_ms or (
            ends_with_sentence_boundary(candidate.snippet)
            and not ends_with_sentence_boundary(last.snippet)
        )
        # The dropped candidate's provenance is preserved on the survivor so a
        # merged record can still explain where its timestamp came from.
        survivor, dropped = (candidate, last) if prefer else (last, candidate)
        merged[-1] = Candidate(
            ms=survivor.ms,
            silence_ms=survivor.silence_ms,
            snippet=survivor.snippet,
            sources=tuple(dict.fromkeys([*survivor.sources, *dropped.sources])),
        )
    return merged


def score_candidate(
    candidate: Candidate,
    duration_seconds: float | None,
    mode: InsertionMode,
    profile: HeuristicProfile,
) -> ScoredCandidate:
    """Port of ``scoreCandidate``, retaining the component breakdown.

    Additions happen in the same order as the TypeScript so the accumulated
    double is bit-identical. Consumers should use ``components.raw_total``
    rather than re-summing the components, since a different summation order can
    produce a different result.
    """
    scoring = profile.scoring
    time_seconds = candidate.ms / 1000

    base = float(scoring["base_score"])
    score = base

    pause_term = 0.0
    if candidate.silence_ms:
        pause_term = min(
            float(scoring["pause_reward_max"]),
            (candidate.silence_ms / float(scoring["pause_reward_saturation_ms"]))
            * float(scoring["pause_reward_max"]),
        )
        score += pause_term

    mode_term = 0.0
    if mode == "song":
        mode_term = float(scoring["song_mode_bonus"])
        score += mode_term

    sentence_term = 0.0
    sentence_end: bool | None = None
    if candidate.snippet and candidate.snippet != scoring["unavailable_snippet_sentinel"]:
        sentence_end = ends_with_sentence_boundary(candidate.snippet)
        if sentence_end:
            sentence_term = float(scoring["sentence_end_reward"])
        else:
            sentence_term = -float(scoring["no_sentence_end_penalty"])
        score += sentence_term

    position_term = 0.0
    edge_term = 0.0
    normalized_position: float | None = None
    if duration_seconds:  # falsy for None and 0, matching `if (durationSeconds)`
        ratio = time_seconds / duration_seconds
        normalized_position = ratio
        if (
            ratio >= float(scoring["mid_episode_min_ratio"])
            and ratio <= float(scoring["mid_episode_max_ratio"])
        ):
            position_term = float(scoring["mid_episode_reward"])
            score += position_term
        if (
            time_seconds < float(scoring["edge_guard_seconds"])
            or time_seconds > duration_seconds - float(scoring["edge_guard_seconds"])
        ):
            edge_term = -float(scoring["edge_penalty"])
            score += edge_term

    raw_total = score
    total = _clamp(raw_total, float(scoring["score_min"]), float(scoring["score_max"]))

    return ScoredCandidate(
        candidate=candidate,
        score=total,
        components=ComponentScores(
            base=base,
            pause=pause_term,
            mode=mode_term,
            sentence=sentence_term,
            position=position_term,
            edge=edge_term,
            raw_total=raw_total,
            clamped=total != raw_total,
        ),
        sentence_end=sentence_end,
        normalized_position=normalized_position,
    )


def select_top_slots(
    scored: Sequence[ScoredCandidate],
    duration_seconds: float | None,
    min_separation_seconds: float,
    count: int,
    profile: HeuristicProfile,
    episode_id: str = "",
) -> list[SelectionEntry]:
    """Port of ``selectTopSlots``, including both padding behaviours.

    Ordering is (score desc, timestamp asc, candidate id asc). The product pads
    an under-filled selection first with fixed ratio positions and then with
    multiples of the minimum separation; both are marked so evaluation can
    exclude them.
    """
    selection = profile.selection
    min_separation_ms = min_separation_seconds * 1000

    ordered = sorted(
        scored,
        key=lambda item: (
            -item.score,
            item.candidate.ms,
            make_candidate_id(episode_id, item.candidate.ms),
        ),
    )

    selected: list[SelectionEntry] = []
    for item in ordered:
        too_close = any(
            abs(entry.ms - item.candidate.ms) < min_separation_ms for entry in selected
        )
        if not too_close:
            selected.append(
                SelectionEntry(
                    ms=item.candidate.ms,
                    silence_ms=item.candidate.silence_ms,
                    snippet=item.candidate.snippet,
                    score=item.score,
                    origin="candidate",
                )
            )
        if len(selected) >= count:
            break

    if duration_seconds:
        fallback_times = [
            ratio * duration_seconds * 1000
            for ratio in selection["ratio_fallback_positions"]
        ]
        for fallback in fallback_times:
            if len(selected) >= count:
                break
            too_close = any(
                abs(entry.ms - fallback) < min_separation_ms for entry in selected
            )
            if not too_close and 0 <= fallback <= duration_seconds * 1000:
                selected.append(
                    SelectionEntry(
                        ms=js_round(fallback),
                        silence_ms=0,
                        snippet="",
                        score=float(selection["ratio_fallback_score"]),
                        origin="ratio_fallback",
                    )
                )

    while len(selected) < count:
        base = min_separation_ms
        if selected:
            latest = sorted(selected, key=lambda entry: entry.ms)[-1]
            base = latest.ms + min_separation_ms
        too_close = any(abs(entry.ms - base) < min_separation_ms for entry in selected)
        candidate_ms = base + min_separation_ms if too_close else base
        selected.append(
            SelectionEntry(
                ms=js_round(candidate_ms),
                silence_ms=0,
                snippet="",
                score=float(selection["spacing_fallback_score"]),
                origin="spacing_fallback",
            )
        )

    return selected[:count]


def _build_route_fallback_candidates(
    duration_seconds: float | None, profile: HeuristicProfile
) -> list[Candidate]:
    """Port of ``insert-sections.ts:146-161``."""
    fallback = profile.route_fallback_candidates
    if duration_seconds and duration_seconds > 0:
        return [
            Candidate(
                ms=js_round(duration_seconds * ratio * 1000),
                silence_ms=0,
                snippet="",
                sources=("route_fallback",),
            )
            for ratio in fallback["ratio_positions"]
        ]
    return [
        Candidate(ms=int(ms), silence_ms=0, snippet="", sources=("route_fallback",))
        for ms in fallback["fixed_positions_ms"]
    ]


def rank_episode(
    episode: EpisodeInput,
    config: HeuristicConfig | None = None,
    profile_name: str | None = None,
) -> EpisodeRanking:
    """Run the full canonical baseline for one episode.

    Reproduces the deterministic portion of ``POST /api/insert-sections``:
    duration bounding, merging, route-level fallback candidates, scoring,
    spacing selection, end-guard clamping, de-duplication, confidence mapping
    and the final confidence sort. The OpenAI pros/cons enrichment
    (``insert-sections.ts:226-258``) is excluded: it is non-deterministic,
    requires a paid API, and does not affect ranking.
    """
    config = config or load_heuristic_config()
    profile = config.profile(profile_name)

    duration_seconds = episode.duration_seconds
    mode = episode.mode

    # insert-sections.ts:136-140
    max_ms = duration_seconds * 1000 if duration_seconds else None
    bounded = (
        [c for c in episode.silence_candidates if c.ms <= max_ms]
        if max_ms is not None
        else list(episode.silence_candidates)
    )

    # insert-sections.ts:141-144
    combined = merge_candidates(
        bounded, list(episode.transcript_candidates), int(profile.merge["min_gap_ms"])
    )

    # insert-sections.ts:146-161
    used_route_fallback = not combined
    usable = (
        _build_route_fallback_candidates(duration_seconds, profile)
        if used_route_fallback
        else combined
    )

    # insert-sections.ts:162-167
    scored = [score_candidate(c, duration_seconds, mode, profile) for c in usable]

    # insert-sections.ts:174-179
    requested = max(int(profile.selection["requested_count"]), episode.count)
    selected = select_top_slots(
        scored,
        duration_seconds,
        float(profile.selection["min_separation_seconds"]),
        requested,
        profile,
        episode.episode_id,
    )[: int(profile.selection["max_returned"])]

    # insert-sections.ts:181-198 -- end-guard clamp then de-duplicate by timestamp
    finalisation = profile.slot_finalisation
    max_slot_ms = (
        max(0, js_round(duration_seconds * 1000) - int(finalisation["end_guard_ms"]))
        if duration_seconds is not None
        else None
    )
    normalized = [
        entry if max_slot_ms is None else _with_ms(entry, min(entry.ms, max_slot_ms))
        for entry in selected
    ]
    deduped: list[SelectionEntry] = []
    seen_ms: set[int] = set()
    for entry in normalized:
        if entry.ms in seen_ms:
            continue
        seen_ms.add(entry.ms)
        deduped.append(entry)

    # insert-sections.ts:200-224
    slots: list[SlotRecord] = []
    for entry in deduped:
        time_seconds = entry.ms / 1000
        clamped_time_seconds = (
            min(max(0, time_seconds), duration_seconds)
            if duration_seconds is not None
            else max(0, time_seconds)
        )
        confidence = js_round(
            _clamp(
                float(finalisation["confidence_base"])
                + entry.score * float(finalisation["confidence_scale"]),
                float(finalisation["confidence_min"]),
                float(finalisation["confidence_max"]),
            )
        )
        slots.append(
            SlotRecord(
                insertion_ms=js_round(clamped_time_seconds * 1000),
                insertion_time_seconds=js_to_fixed(
                    clamped_time_seconds, int(finalisation["time_seconds_decimals"])
                ),
                confidence_percent=confidence,
                silence_ms=entry.silence_ms,
                snippet=entry.snippet,
                score=entry.score,
                candidate_id=_entry_candidate_id(episode.episode_id, entry),
                is_synthetic=entry.origin != "candidate",
            )
        )

    # insert-sections.ts:260 -- stable sort by descending confidence
    slots.sort(key=lambda slot: -slot.confidence_percent)

    records = _build_candidate_records(
        episode_id=episode.episode_id,
        scored=scored,
        selected=deduped,
        baseline_version=config.canonical_profile_name,
        config_version=config.config_version,
    )

    return EpisodeRanking(
        episode_id=episode.episode_id,
        baseline_version=config.canonical_profile_name,
        config_version=config.config_version,
        mode=mode,
        duration_seconds=duration_seconds,
        used_route_fallback_candidates=used_route_fallback,
        candidates=tuple(records),
        slots=tuple(slots),
        points=tuple(slot.insertion_time_seconds for slot in slots),
        confidences=tuple(slot.confidence_percent for slot in slots),
    )


def _with_ms(entry: SelectionEntry, ms: int) -> SelectionEntry:
    return SelectionEntry(
        ms=ms,
        silence_ms=entry.silence_ms,
        snippet=entry.snippet,
        score=entry.score,
        origin=entry.origin,
    )


def _entry_candidate_id(episode_id: str, entry: SelectionEntry) -> str:
    """Synthetic padding gets its own ID namespace so it can never collide with,
    or be mistaken for, a real candidate."""
    if entry.origin == "candidate":
        return make_candidate_id(episode_id, entry.ms)
    return f"{episode_id}:fallback:{entry.ms:09d}"


def _build_candidate_records(
    episode_id: str,
    scored: Sequence[ScoredCandidate],
    selected: Sequence[SelectionEntry],
    baseline_version: str,
    config_version: str,
) -> list[CandidateRecord]:
    """Assign ranks and selection state, then emit one record per candidate.

    ``rank`` is the pure score ordering over real candidates -- the ordering
    evaluation consumes. ``selection_rank`` is the position in the product's
    spacing-constrained top-3, which is a different thing and is recorded
    separately.
    """
    ordered = sorted(
        scored,
        key=lambda item: (
            -item.score,
            item.candidate.ms,
            make_candidate_id(episode_id, item.candidate.ms),
        ),
    )
    rank_by_ms = {item.candidate.ms: index + 1 for index, item in enumerate(ordered)}
    selection_rank_by_ms = {
        entry.ms: index + 1
        for index, entry in enumerate(selected)
        if entry.origin == "candidate"
    }

    records: list[CandidateRecord] = []
    for item in scored:
        ms = item.candidate.ms
        records.append(
            CandidateRecord(
                episode_id=episode_id,
                candidate_id=make_candidate_id(episode_id, ms),
                timestamp_ms=ms,
                candidate_sources=tuple(item.candidate.sources),
                pause_duration_ms=item.candidate.silence_ms,
                sentence_end=item.sentence_end,
                snippet=item.candidate.snippet,
                normalized_episode_position=item.normalized_position,
                raw_component_scores=item.components,
                total_score=item.score,
                rank=rank_by_ms[ms],
                selected=ms in selection_rank_by_ms,
                selection_rank=selection_rank_by_ms.get(ms),
                selection_origin="candidate",
                is_synthetic=False,
                merged_from_ms=(),
                baseline_version=baseline_version,
                config_version=config_version,
            )
        )

    for index, entry in enumerate(selected):
        if entry.origin == "candidate":
            continue
        records.append(
            CandidateRecord(
                episode_id=episode_id,
                candidate_id=_entry_candidate_id(episode_id, entry),
                timestamp_ms=entry.ms,
                candidate_sources=(entry.origin,),
                pause_duration_ms=entry.silence_ms,
                sentence_end=None,
                snippet=entry.snippet,
                normalized_episode_position=None,
                raw_component_scores=ComponentScores(
                    base=entry.score,
                    pause=0.0,
                    mode=0.0,
                    sentence=0.0,
                    position=0.0,
                    edge=0.0,
                    raw_total=entry.score,
                    clamped=False,
                ),
                total_score=entry.score,
                # Synthetic padding is not a ranked prediction and must never
                # enter an evaluation ranking.
                rank=None,
                selected=True,
                selection_rank=index + 1,
                selection_origin=entry.origin,
                is_synthetic=True,
                merged_from_ms=(),
                baseline_version=baseline_version,
                config_version=config_version,
            )
        )

    return records


def rank_episodes(
    episodes: Iterable[EpisodeInput],
    config: HeuristicConfig | None = None,
    profile_name: str | None = None,
) -> list[EpisodeRanking]:
    """Rank several episodes, rejecting duplicate episode IDs."""
    config = config or load_heuristic_config()
    seen: set[str] = set()
    rankings: list[EpisodeRanking] = []
    for episode in episodes:
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate episode_id in input: {episode.episode_id!r}")
        seen.add(episode.episode_id)
        rankings.append(rank_episode(episode, config=config, profile_name=profile_name))
    return rankings
