"""Canonical candidate schema for the Slotify ranking project.

Design rules:

* **Timestamps are canonical in integer milliseconds.** The product works in
  milliseconds and the identity of a candidate must not depend on float
  formatting. Seconds-valued fields are derived and are for humans and for the
  product contract.
* **Candidate IDs are deterministic**: ``f"{episode_id}:{timestamp_ms:09d}"``.
  The same episode processed twice yields the same IDs, which is what makes
  labels, predictions and splits joinable across runs.
* **Every score is explained.** ``raw_component_scores`` records the individual
  contributions in the exact order the TypeScript implementation accumulates
  them, so a total can be reproduced by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Sequence

from slotify_rank.config.versions import CANDIDATE_SCHEMA_VERSION

__all__ = [
    "CandidateSource",
    "SelectionOrigin",
    "InsertionMode",
    "Candidate",
    "ComponentScores",
    "CandidateRecord",
    "SlotRecord",
    "EpisodeInput",
    "EpisodeRanking",
    "make_candidate_id",
]

InsertionMode = Literal["podcast", "song"]

# Phase 1 generators only. Phase 2 adds whisper_word_gap, rms_minimum,
# spectral_change, speaker_turn and fixed_interval.
CandidateSource = Literal[
    "silence",
    "transcript_segment_end",
    "route_fallback",
]

SelectionOrigin = Literal["candidate", "ratio_fallback", "spacing_fallback"]


def make_candidate_id(episode_id: str, timestamp_ms: int) -> str:
    """Deterministic candidate identity. Stable across runs and machines."""
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms must be non-negative, got {timestamp_ms}")
    return f"{episode_id}:{timestamp_ms:09d}"


@dataclass(frozen=True)
class Candidate:
    """A raw candidate insertion point, before scoring.

    Mirrors the TypeScript ``Candidate`` interface (``backend/src/types.ts:6``)
    plus provenance that the TypeScript type does not carry.
    """

    ms: int
    silence_ms: int
    snippet: str
    sources: tuple[CandidateSource, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ms, int):
            raise TypeError(f"Candidate.ms must be an int, got {type(self.ms).__name__}")
        if self.ms < 0:
            raise ValueError(f"Candidate.ms must be non-negative, got {self.ms}")
        if self.silence_ms < 0:
            raise ValueError(
                f"Candidate.silence_ms must be non-negative, got {self.silence_ms}"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], default_source: CandidateSource) -> "Candidate":
        if "ms" not in raw:
            raise KeyError(f"Candidate is missing required key 'ms': {raw!r}")
        ms = raw["ms"]
        if isinstance(ms, bool) or not isinstance(ms, (int, float)):
            raise TypeError(f"Candidate 'ms' must be numeric, got {ms!r}")
        if isinstance(ms, float) and not ms.is_integer():
            raise ValueError(
                f"Candidate 'ms' must be a whole number of milliseconds, got {ms!r}"
            )
        silence = raw.get("silence_ms", raw.get("silenceMs", 0))
        if isinstance(silence, bool) or not isinstance(silence, (int, float)):
            raise TypeError(f"Candidate 'silence_ms' must be numeric, got {silence!r}")
        sources = tuple(raw.get("sources", ()) or ()) or (default_source,)
        return cls(
            ms=int(ms),
            silence_ms=int(silence),
            snippet=str(raw.get("snippet", "") or ""),
            sources=sources,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ComponentScores:
    """Additive contributions to the heuristic score, in accumulation order.

    ``raw_total`` is the sequential sum before clamping; ``clamped`` records
    whether the clamp changed the value. Summing the components in a different
    order can produce a different double, so consumers should prefer
    ``raw_total``.
    """

    base: float
    pause: float
    mode: float
    sentence: float
    position: float
    edge: float
    raw_total: float
    clamped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    """A scored, ranked candidate: the canonical serialized unit of Phase 1."""

    episode_id: str
    candidate_id: str
    timestamp_ms: int
    candidate_sources: tuple[str, ...]
    pause_duration_ms: int
    sentence_end: bool | None
    snippet: str
    normalized_episode_position: float | None
    raw_component_scores: ComponentScores
    total_score: float
    rank: int | None
    selected: bool
    selection_rank: int | None
    selection_origin: SelectionOrigin
    is_synthetic: bool
    merged_from_ms: tuple[int, ...]
    baseline_version: str
    config_version: str
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_sources"] = list(self.candidate_sources)
        payload["merged_from_ms"] = list(self.merged_from_ms)
        return payload


@dataclass(frozen=True)
class SlotRecord:
    """A finalised slot, matching the product response shape.

    Mirrors ``Slot`` (``backend/src/types.ts:24``) minus the non-deterministic
    ``pros``/``cons``/``rationale``, which come from an OpenAI call and are
    therefore not part of the offline baseline.
    """

    insertion_ms: int
    insertion_time_seconds: float
    confidence_percent: int
    silence_ms: int
    snippet: str
    score: float
    candidate_id: str
    is_synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeInput:
    """Audio-free input to the canonical baseline.

    Phase 1 consumes candidate *metadata* so the baseline can be tested with no
    audio, no Whisper and no paid APIs. Phase 3 adds an audio-driven generator
    that produces these same structures.
    """

    episode_id: str
    duration_seconds: float | None
    mode: InsertionMode = "podcast"
    silence_candidates: tuple[Candidate, ...] = ()
    transcript_candidates: tuple[Candidate, ...] = ()
    count: int = 3

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError(
                f"duration_seconds must be non-negative or null, got {self.duration_seconds}"
            )
        if self.mode not in ("podcast", "song"):
            raise ValueError(f"mode must be 'podcast' or 'song', got {self.mode!r}")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "EpisodeInput":
        if not isinstance(raw, dict):
            raise TypeError(f"Episode input must be a JSON object, got {type(raw).__name__}")
        if "episode_id" not in raw:
            raise KeyError("Episode input is missing required key 'episode_id'")
        duration = raw.get("duration_seconds")
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
        ):
            raise TypeError(f"'duration_seconds' must be numeric or null, got {duration!r}")
        return cls(
            episode_id=str(raw["episode_id"]),
            duration_seconds=None if duration is None else float(duration),
            mode=str(raw.get("mode", "podcast")),  # type: ignore[arg-type]
            silence_candidates=tuple(
                Candidate.from_mapping(entry, "silence")
                for entry in raw.get("silence_candidates", ())
            ),
            transcript_candidates=tuple(
                Candidate.from_mapping(entry, "transcript_segment_end")
                for entry in raw.get("transcript_candidates", ())
            ),
            count=int(raw.get("count", 3)),
        )


@dataclass(frozen=True)
class EpisodeRanking:
    """The full, explainable result of ranking one episode."""

    episode_id: str
    baseline_version: str
    config_version: str
    mode: InsertionMode
    duration_seconds: float | None
    used_route_fallback_candidates: bool
    candidates: tuple[CandidateRecord, ...]
    slots: tuple[SlotRecord, ...]
    points: tuple[float, ...]
    confidences: tuple[int, ...]
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    @property
    def ranked_candidate_ids(self) -> list[str]:
        """Real (non-synthetic) candidate IDs in ranked order.

        This is the ordering evaluation consumes. It is the pure score ordering,
        *not* the spacing-constrained product selection: NDCG measures how well
        the ranker orders candidates, while spacing is a product constraint
        applied afterwards.
        """
        ranked = [c for c in self.candidates if c.rank is not None]
        ranked.sort(key=lambda c: c.rank)  # type: ignore[arg-type,return-value]
        return [c.candidate_id for c in ranked]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "baseline_version": self.baseline_version,
            "config_version": self.config_version,
            "schema_version": self.schema_version,
            "mode": self.mode,
            "duration_seconds": self.duration_seconds,
            "used_route_fallback_candidates": self.used_route_fallback_candidates,
            "ranked_candidate_ids": self.ranked_candidate_ids,
            "candidates": [c.to_dict() for c in self.candidates],
            "slots": [s.to_dict() for s in self.slots],
            "points": list(self.points),
            "confidences": list(self.confidences),
        }
