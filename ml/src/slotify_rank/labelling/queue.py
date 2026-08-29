"""Deterministic, stratified labelling-queue construction.

The failure this module exists to prevent is a biased label set. If you label
whichever candidates the pipeline emits first, or only the ones the heuristic
already likes, the resulting model is graded on a distribution the heuristic
chose -- which inflates the baseline and starves the model of the negative and
ambiguous comparisons it needs to learn an ordering. So the queue is built, not
taken as it comes.

What "stratified" means here, stated precisely so the report can be checked:

* **Hard strata: dataset split x heuristic-score tertile.** The target count is
  allocated across splits by :attr:`QueueConfig.allocation`, and each split's
  quota is spread evenly across the low / medium / high score tertiles. The
  tertile boundaries are computed from the eligible pool and recorded, so
  "strong", "ambiguous" and "likely-negative" candidates are all present by
  construction rather than by luck.
* **Episode and series spread: round-robin.** Within a (split, tertile) cell,
  candidates are drawn round-robin across episodes, so one long episode cannot
  dominate. A hard per-episode cap enforces the same thing globally.
* **Signal disagreement is front-loaded.** Where a transcript is available, a
  candidate whose acoustic signal (a long pause) and transcript signal (a
  sentence boundary) disagree is drawn first inside its cell, because those are
  the candidates the model most needs a human opinion on.
* **Everything else is coverage-reported.** Candidate source, normalized
  position, silence duration, sentence-boundary status, transcript availability
  and content type are measured across the selected set and written into the
  artifact, so a skew is visible even when it is not a hard constraint.
* **Optionally, only featurisable candidates are queued.**
  ``require_complete_features`` excludes candidates whose multimodal feature
  record is missing or failed. They cannot enter training, and the readiness
  gate refuses a labelled set containing them -- so queuing one spends human
  attention on a judgement no model will ever see.

Determinism: every ordering and every sample is keyed on a SHA-256 of the queue
seed and the candidate id, never on manifest order or wall-clock time, so the
same corpus and config produce a byte-identical queue on any machine.

Stages, all disjoint in the unique set except where noted:

* **pilot** (20-30): a diverse subset labelled first to validate the rubric,
  the context window and the candidate generator before the bulk run.
* **primary** (~200-250): the rest of the unique set.
* **overlap** (30-50): a subset of the unique set flagged for a *second*
  annotator, so inter-annotator agreement can be measured later. These are the
  same candidate ids, not new ones.
* **consistency** (10-15): a small subset re-presented later under a distinct
  ``presentation_id`` to measure intra-annotator consistency. These repeats are
  never counted as unique candidates and never enter the training manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    LABELLING_QUEUE_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes, sha256_text
from slotify_rank.data.schema import (
    TARGET_DOMAIN_CONTENT_TYPES,
    DatasetCandidate,
    EpisodeRecord,
)

__all__ = [
    "ScoreStrata",
    "QueueConfig",
    "CandidateView",
    "QueuePresentation",
    "LabellingQueue",
    "build_candidate_views",
    "build_queue",
    "read_queue",
    "write_queue",
    "QueueError",
]


class QueueError(ValueError):
    """A queue cannot be built from the given pool and configuration."""


# ---------------------------------------------------------------------------
# Score strata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreStrata:
    """Tertile boundaries over the eligible pool's heuristic scores.

    Boundaries are recorded rather than hard-coded so the report can state
    exactly where "low", "medium" and "high" were cut for this corpus.
    """

    low_max: float
    medium_max: float
    minimum: float
    maximum: float

    def classify(self, score: float) -> str:
        if score <= self.low_max:
            return "low"
        if score <= self.medium_max:
            return "medium"
        return "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_max": self.low_max,
            "medium_max": self.medium_max,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "labels": ["low", "medium", "high"],
        }

    @classmethod
    def from_scores(cls, scores: Sequence[float]) -> "ScoreStrata":
        if not scores:
            raise QueueError("Cannot compute score strata from an empty pool")
        ordered = sorted(float(value) for value in scores)

        def quantile(fraction: float) -> float:
            # Nearest-rank on a sorted list: deterministic, dependency-free, and
            # good enough to define three roughly equal buckets.
            index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
            return ordered[index]

        return cls(
            low_max=quantile(1 / 3),
            medium_max=quantile(2 / 3),
            minimum=ordered[0],
            maximum=ordered[-1],
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScoreStrata":
        return cls(
            low_max=float(raw["low_max"]),
            medium_max=float(raw["medium_max"]),
            minimum=float(raw["minimum"]),
            maximum=float(raw["maximum"]),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueConfig:
    """Everything that determines which candidates are queued and how."""

    version: str = "v1"
    seed: int = 20260723
    target_unique: int = 280
    pilot_size: int = 24
    overlap_size: int = 40
    consistency_size: int = 12
    #: Global cap so one long episode cannot dominate the labelled set.
    max_per_episode: int = 90
    allocation: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.66, "validation": 0.17, "test": 0.17}
    )
    include_fixtures: bool = False
    #: Queue only candidates with a ``complete`` multimodal feature record.
    #: Off by default so a queue can be built before the feature pipeline has
    #: run; on for any round whose labels are meant to train a model.
    require_complete_features: bool = False
    eligible_content_types: tuple[str, ...] = tuple(sorted(TARGET_DOMAIN_CONTENT_TYPES))
    position_buckets: int = 5
    #: Ascending millisecond boundaries defining the silence-duration buckets
    #: used for coverage reporting.
    silence_bucket_edges_ms: tuple[int, ...] = (500, 1000, 2000)
    #: A pause at or beyond this counts as an "acoustic break" for the
    #: signal-disagreement flag.
    long_silence_ms: int = 1500

    def __post_init__(self) -> None:
        if self.target_unique < 1:
            raise QueueError("target_unique must be positive")
        if self.pilot_size < 0 or self.pilot_size >= self.target_unique:
            raise QueueError("pilot_size must be in [0, target_unique)")
        if self.overlap_size < 0 or self.overlap_size > self.target_unique:
            raise QueueError("overlap_size must be in [0, target_unique]")
        if self.consistency_size < 0 or self.consistency_size > self.target_unique:
            raise QueueError("consistency_size must be in [0, target_unique]")
        if self.max_per_episode < 1:
            raise QueueError("max_per_episode must be at least 1")
        total = sum(self.allocation.values())
        if abs(total - 1.0) > 1e-6:
            raise QueueError(
                f"allocation ratios must sum to 1.0, got {total} ({dict(self.allocation)})"
            )
        if any(value < 0 for value in self.allocation.values()):
            raise QueueError("allocation ratios must be non-negative")
        if self.position_buckets < 1:
            raise QueueError("position_buckets must be at least 1")
        if list(self.silence_bucket_edges_ms) != sorted(self.silence_bucket_edges_ms):
            raise QueueError("silence_bucket_edges_ms must be ascending")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "target_unique": self.target_unique,
            "pilot_size": self.pilot_size,
            "overlap_size": self.overlap_size,
            "consistency_size": self.consistency_size,
            "max_per_episode": self.max_per_episode,
            "allocation": dict(self.allocation),
            "include_fixtures": self.include_fixtures,
            "require_complete_features": self.require_complete_features,
            "eligible_content_types": list(self.eligible_content_types),
            "position_buckets": self.position_buckets,
            "silence_bucket_edges_ms": list(self.silence_bucket_edges_ms),
            "long_silence_ms": self.long_silence_ms,
        }

    def digest(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))[:16]


def load_queue_config(path: str | Path | None) -> QueueConfig:
    """Load ``ml/configs/labelling_queue_v1.yaml``; defaults with no path."""
    if path is None:
        return QueueConfig()
    import yaml

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise QueueError(f"{config_path} must contain a YAML mapping")
    section = loaded.get("queue") or {}
    if not isinstance(section, Mapping):
        raise QueueError(f"{config_path}: 'queue' must be a mapping")
    payload = dict(section)
    if "allocation" in payload and not isinstance(payload["allocation"], Mapping):
        raise QueueError(f"{config_path}: 'queue.allocation' must be a mapping")
    if "eligible_content_types" in payload:
        payload["eligible_content_types"] = tuple(payload["eligible_content_types"])
    if "silence_bucket_edges_ms" in payload:
        payload["silence_bucket_edges_ms"] = tuple(
            int(edge) for edge in payload["silence_bucket_edges_ms"]
        )
    if "allocation" in payload:
        payload["allocation"] = {
            str(k): float(v) for k, v in payload["allocation"].items()
        }
    known = set(QueueConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise QueueError(f"{config_path}: unknown queue setting(s) {unknown}")
    return QueueConfig(**payload)


# ---------------------------------------------------------------------------
# Candidate view: everything the queue stratifies or reports on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    episode_id: str
    series_id: str
    content_type: str
    dataset_split: str
    heuristic_score: float
    primary_source: str
    n_sources: int
    normalized_position: float
    silence_ms: int
    sentence_end: bool | None
    transcript_available: bool
    audio_available: bool
    text_available: bool
    feature_status: str = ""

    #: Acoustic and transcript signals point opposite ways. Only ever True when a
    #: transcript exists: a long pause with no sentence boundary (acoustic says
    #: "break", transcript says "mid-sentence"), or a sentence boundary with
    #: almost no pause. Computed in :func:`build_candidate_views`.
    is_disagreement: bool = False

    def position_bucket(self, buckets: int) -> str:
        index = min(buckets - 1, int(self.normalized_position * buckets))
        return f"p{index}"

    def silence_bucket(self, edges: Sequence[int]) -> str:
        for i, edge in enumerate(edges):
            if self.silence_ms < edge:
                return f"s{i}"
        return f"s{len(edges)}"


def build_candidate_views(
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    config: QueueConfig,
    feature_records: Mapping[str, Any] | None = None,
) -> list[CandidateView]:
    """Join candidates onto their episode and (optional) feature record.

    Filters out synthetic padding, non-eligible candidates, out-of-domain
    content types and -- unless ``include_fixtures`` -- repository fixtures. The
    result is the pool the queue is drawn from.
    """
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    features = feature_records or {}
    views: list[CandidateView] = []
    for candidate in candidates:
        if candidate.is_synthetic or not candidate.eligible_for_labelling:
            continue
        episode = episode_by_id.get(candidate.episode_id)
        if episode is None:
            # A candidate whose episode is missing cannot be presented (no audio
            # to play), so it is not a queue-eligibility question but a corpus
            # inconsistency; leave it to the validators and skip it here.
            continue
        if episode.content_type not in config.eligible_content_types:
            continue
        if (
            not config.include_fixtures
            and episode.source_type == "existing_repository_fixture"
        ):
            continue
        if candidate.heuristic_score is None:
            continue

        record = features.get(candidate.candidate_id)
        feature_status = str(getattr(record, "feature_status", "") or "")
        if config.require_complete_features and feature_status != "complete":
            continue
        transcript_available = bool(
            getattr(record, "transcript_available", False)
        )
        audio_available = bool(getattr(record, "audio_embedding_available", False))
        text_available = bool(getattr(record, "text_embedding_available", False))

        silence_ms = int(candidate.silence_duration_ms)
        sentence_end = candidate.sentence_end
        disagreement = False
        if sentence_end is not None:
            acoustic_break = silence_ms >= config.long_silence_ms
            disagreement = acoustic_break != bool(sentence_end)

        views.append(
            CandidateView(
                candidate_id=candidate.candidate_id,
                episode_id=candidate.episode_id,
                series_id=episode.series_id,
                content_type=episode.content_type,
                dataset_split=candidate.dataset_split,
                heuristic_score=float(candidate.heuristic_score),
                primary_source=(
                    candidate.candidate_sources[0]
                    if candidate.candidate_sources
                    else "unknown"
                ),
                n_sources=len(candidate.candidate_sources),
                normalized_position=(
                    float(candidate.normalized_episode_position)
                    if candidate.normalized_episode_position is not None
                    else 0.0
                ),
                silence_ms=silence_ms,
                sentence_end=sentence_end,
                transcript_available=transcript_available,
                audio_available=audio_available,
                text_available=text_available,
                is_disagreement=disagreement,
                feature_status=feature_status,
            )
        )
    return views


# ---------------------------------------------------------------------------
# Presentation and queue artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueuePresentation:
    """One item as the labelling UI would serve it, in queue order.

    A unique candidate produces one presentation whose ``presentation_id`` is
    the candidate id. A consistency re-check produces a second presentation with
    a distinct ``presentation_id`` pointing back at the same candidate.
    """

    presentation_id: str
    candidate_id: str
    episode_id: str
    series_id: str
    dataset_split: str
    stage: str
    score_stratum: str
    is_repeat: bool
    repeat_of: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "presentation_id": self.presentation_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "series_id": self.series_id,
            "dataset_split": self.dataset_split,
            "stage": self.stage,
            "score_stratum": self.score_stratum,
            "is_repeat": self.is_repeat,
            "repeat_of": self.repeat_of,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "QueuePresentation":
        return cls(
            presentation_id=str(raw["presentation_id"]),
            candidate_id=str(raw["candidate_id"]),
            episode_id=str(raw["episode_id"]),
            series_id=str(raw["series_id"]),
            dataset_split=str(raw["dataset_split"]),
            stage=str(raw["stage"]),
            score_stratum=str(raw["score_stratum"]),
            is_repeat=bool(raw["is_repeat"]),
            repeat_of=raw.get("repeat_of"),
        )


@dataclass(frozen=True)
class LabellingQueue:
    """A frozen, versioned labelling assignment for one round."""

    queue_version: str
    schema_version: str
    seed: int
    config: Mapping[str, Any]
    config_hash: str
    candidate_manifest_hash: str
    split_manifest_hash: str | None
    score_strata: ScoreStrata
    presentations: tuple[QueuePresentation, ...]
    unique_candidate_ids: tuple[str, ...]
    pilot_candidate_ids: tuple[str, ...]
    primary_candidate_ids: tuple[str, ...]
    overlap_candidate_ids: tuple[str, ...]
    consistency_candidate_ids: tuple[str, ...]
    coverage: Mapping[str, Any]

    @property
    def unique_count(self) -> int:
        return len(self.unique_candidate_ids)

    def stage_of(self, candidate_id: str) -> str:
        if candidate_id in set(self.pilot_candidate_ids):
            return "pilot"
        if candidate_id in set(self.primary_candidate_ids):
            return "primary"
        raise KeyError(f"{candidate_id!r} is not in this queue")

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_version": self.queue_version,
            "schema_version": self.schema_version,
            "package_version": PACKAGE_VERSION,
            "seed": self.seed,
            "config": dict(self.config),
            "config_hash": self.config_hash,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "score_strata": self.score_strata.to_dict(),
            "unique_candidate_count": len(self.unique_candidate_ids),
            "unique_candidate_ids": list(self.unique_candidate_ids),
            "pilot_candidate_ids": list(self.pilot_candidate_ids),
            "primary_candidate_ids": list(self.primary_candidate_ids),
            "overlap_candidate_ids": list(self.overlap_candidate_ids),
            "consistency_candidate_ids": list(self.consistency_candidate_ids),
            "presentation_count": len(self.presentations),
            "presentations": [p.to_dict() for p in self.presentations],
            "coverage": dict(self.coverage),
        }

    def content_hash(self) -> str:
        """SHA-256 of the queue's identity-defining content.

        Excludes nothing volatile: the whole serialized form is hashed, so a
        freeze can pin exactly this queue.
        """
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LabellingQueue":
        return cls(
            queue_version=str(raw["queue_version"]),
            schema_version=str(raw["schema_version"]),
            seed=int(raw["seed"]),
            config=dict(raw["config"]),
            config_hash=str(raw["config_hash"]),
            candidate_manifest_hash=str(raw["candidate_manifest_hash"]),
            split_manifest_hash=raw.get("split_manifest_hash"),
            score_strata=ScoreStrata.from_mapping(raw["score_strata"]),
            presentations=tuple(
                QueuePresentation.from_mapping(p) for p in raw.get("presentations", ())
            ),
            unique_candidate_ids=tuple(raw["unique_candidate_ids"]),
            pilot_candidate_ids=tuple(raw["pilot_candidate_ids"]),
            primary_candidate_ids=tuple(raw["primary_candidate_ids"]),
            overlap_candidate_ids=tuple(raw["overlap_candidate_ids"]),
            consistency_candidate_ids=tuple(raw["consistency_candidate_ids"]),
            coverage=dict(raw.get("coverage", {})),
        )


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _key(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "\x1f" + "\x1f".join(parts)).encode()).hexdigest()


def _round_robin(
    seed: int,
    cell_key: str,
    by_episode: Mapping[str, list[CandidateView]],
    quota: int,
    per_episode_used: dict[str, int],
    max_per_episode: int,
) -> list[CandidateView]:
    """Draw up to ``quota`` candidates round-robin across episodes.

    Episodes are visited in a seeded order; inside each episode candidates are
    ordered disagreement-first, then by a seeded hash. A per-episode budget that
    persists across cells enforces the global cap.
    """
    episodes = sorted(by_episode, key=lambda e: _key(seed, cell_key, e))
    ordered_within: dict[str, list[CandidateView]] = {}
    for episode_id in episodes:
        ordered_within[episode_id] = sorted(
            by_episode[episode_id],
            key=lambda v: (not v.is_disagreement, _key(seed, v.candidate_id)),
        )

    picked: list[CandidateView] = []
    cursors = {episode_id: 0 for episode_id in episodes}
    while len(picked) < quota:
        progressed = False
        for episode_id in episodes:
            if len(picked) >= quota:
                break
            if per_episode_used[episode_id] >= max_per_episode:
                continue
            cursor = cursors[episode_id]
            members = ordered_within[episode_id]
            if cursor >= len(members):
                continue
            picked.append(members[cursor])
            cursors[episode_id] = cursor + 1
            per_episode_used[episode_id] += 1
            progressed = True
        if not progressed:
            break
    return picked


def _allocate(total: int, weights: Mapping[str, float], available: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` across keys by ``weights``, clamped to availability.

    Any shortfall from clamping is redistributed to the keys that still have
    room, largest-weight first, so the queue reaches ``total`` whenever the pool
    is large enough overall.
    """
    keys = [k for k in weights if available.get(k, 0) > 0]
    raw = {k: weights[k] * total for k in keys}
    alloc = {k: min(available[k], int(raw[k])) for k in keys}
    # Distribute the remainder deterministically by descending fractional part.
    assigned = sum(alloc.values())
    remainder = min(total, sum(available[k] for k in keys)) - assigned
    order = sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), -weights[k], k))
    idx = 0
    while remainder > 0 and order:
        k = order[idx % len(order)]
        if alloc[k] < available[k]:
            alloc[k] += 1
            remainder -= 1
        elif all(alloc[j] >= available[j] for j in order):
            break
        idx += 1
    return alloc


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def build_queue(
    candidates: Sequence[DatasetCandidate],
    episodes: Sequence[EpisodeRecord],
    config: QueueConfig | None = None,
    feature_records: Mapping[str, Any] | None = None,
    candidate_manifest_hash: str = "",
    split_manifest_hash: str | None = None,
) -> LabellingQueue:
    config = config or QueueConfig()
    pool = build_candidate_views(candidates, episodes, config, feature_records)
    if not pool:
        raise QueueError(
            "No eligible candidates to queue. Generate candidates for real "
            "target-domain episodes first (see docs/human-labelling-workflow.md)."
        )

    strata = ScoreStrata.from_scores([v.heuristic_score for v in pool])
    target = min(config.target_unique, len(pool))

    # Group the pool by (split, score tertile).
    cells: dict[tuple[str, str], list[CandidateView]] = {}
    for view in pool:
        key = (view.dataset_split, strata.classify(view.heuristic_score))
        cells.setdefault(key, []).append(view)

    splits_present = sorted({v.dataset_split for v in pool})
    per_split_available = {
        split: sum(1 for v in pool if v.dataset_split == split)
        for split in splits_present
    }
    split_weights = {
        split: config.allocation.get(split, 0.0) for split in splits_present
    }
    if sum(split_weights.values()) <= 0:
        # None of the present splits are named in the allocation (e.g. a degraded
        # single 'development' partition). Fall back to equal weight so the queue
        # still stratifies on score.
        split_weights = {split: 1.0 / len(splits_present) for split in splits_present}
    else:
        norm = sum(split_weights.values())
        split_weights = {k: v / norm for k, v in split_weights.items()}

    split_targets = _allocate(target, split_weights, per_split_available)

    per_episode_used: dict[str, int] = {v.episode_id: 0 for v in pool}
    selected: list[CandidateView] = []
    for split in splits_present:
        split_quota = split_targets.get(split, 0)
        if split_quota <= 0:
            continue
        tiers = ["low", "medium", "high"]
        available_by_tier = {
            tier: len(cells.get((split, tier), [])) for tier in tiers
        }
        tier_weights = {tier: 1.0 / len(tiers) for tier in tiers}
        tier_targets = _allocate(split_quota, tier_weights, available_by_tier)
        for tier in tiers:
            quota = tier_targets.get(tier, 0)
            if quota <= 0:
                continue
            members = cells.get((split, tier), [])
            by_episode: dict[str, list[CandidateView]] = {}
            for view in members:
                by_episode.setdefault(view.episode_id, []).append(view)
            picked = _round_robin(
                config.seed,
                f"{split}:{tier}",
                by_episode,
                quota,
                per_episode_used,
                config.max_per_episode,
            )
            selected.extend(picked)

    if not selected:
        raise QueueError(
            "Stratified selection produced no candidates; the per-episode cap or "
            "allocation is too tight for this pool."
        )

    # Deterministic overall queue order.
    selected.sort(key=lambda v: _key(config.seed, "order", v.candidate_id))
    selected_ids = [v.candidate_id for v in selected]
    view_by_id = {v.candidate_id: v for v in selected}

    # Stages. Pilot draws round-robin across (split, tier) so it is itself
    # diverse; primary is the remainder.
    pilot_ids = _draw_diverse(
        config.seed, "pilot", selected, strata, min(config.pilot_size, len(selected))
    )
    pilot_set = set(pilot_ids)
    primary_ids = [cid for cid in selected_ids if cid not in pilot_set]

    overlap_ids = _draw_diverse(
        config.seed, "overlap", selected, strata, min(config.overlap_size, len(selected))
    )
    # Consistency prefers primary-stage candidates so the pilot stays a clean
    # first pass; falls back to the full set when primary is too small.
    consistency_pool = [view_by_id[cid] for cid in primary_ids] or selected
    consistency_ids = _draw_diverse(
        config.seed,
        "consistency",
        consistency_pool,
        strata,
        min(config.consistency_size, len(consistency_pool)),
    )

    presentations = _build_presentations(
        config.seed, selected, view_by_id, strata, pilot_set, consistency_ids
    )
    coverage = _coverage(selected, strata, config, split_targets)

    return LabellingQueue(
        queue_version=config.version,
        schema_version=LABELLING_QUEUE_SCHEMA_VERSION,
        seed=config.seed,
        config=config.to_dict(),
        config_hash=config.digest(),
        candidate_manifest_hash=candidate_manifest_hash,
        split_manifest_hash=split_manifest_hash,
        score_strata=strata,
        presentations=tuple(presentations),
        unique_candidate_ids=tuple(selected_ids),
        pilot_candidate_ids=tuple(pilot_ids),
        primary_candidate_ids=tuple(primary_ids),
        overlap_candidate_ids=tuple(overlap_ids),
        consistency_candidate_ids=tuple(consistency_ids),
        coverage=coverage,
    )


def _draw_diverse(
    seed: int,
    tag: str,
    pool: Sequence[CandidateView],
    strata: ScoreStrata,
    count: int,
) -> list[str]:
    """Pick ``count`` ids spread across (split, tier) cells, round-robin."""
    if count <= 0:
        return []
    by_cell: dict[tuple[str, str], list[CandidateView]] = {}
    for view in pool:
        cell = (view.dataset_split, strata.classify(view.heuristic_score))
        by_cell.setdefault(cell, []).append(view)
    for cell in by_cell:
        by_cell[cell].sort(key=lambda v: _key(seed, tag, v.candidate_id))
    cells = sorted(by_cell, key=lambda c: _key(seed, tag, c[0], c[1]))
    chosen: list[str] = []
    cursors = {cell: 0 for cell in cells}
    while len(chosen) < count:
        progressed = False
        for cell in cells:
            if len(chosen) >= count:
                break
            cursor = cursors[cell]
            if cursor < len(by_cell[cell]):
                chosen.append(by_cell[cell][cursor].candidate_id)
                cursors[cell] = cursor + 1
                progressed = True
        if not progressed:
            break
    return chosen


def _build_presentations(
    seed: int,
    selected: Sequence[CandidateView],
    view_by_id: Mapping[str, CandidateView],
    strata: ScoreStrata,
    pilot_set: set[str],
    consistency_ids: Sequence[str],
) -> list[QueuePresentation]:
    presentations: list[QueuePresentation] = []
    for view in selected:
        presentations.append(
            QueuePresentation(
                presentation_id=view.candidate_id,
                candidate_id=view.candidate_id,
                episode_id=view.episode_id,
                series_id=view.series_id,
                dataset_split=view.dataset_split,
                stage="pilot" if view.candidate_id in pilot_set else "primary",
                score_stratum=strata.classify(view.heuristic_score),
                is_repeat=False,
                repeat_of=None,
            )
        )
    # Consistency re-checks are appended so they appear "later" in the queue and
    # never collide with the first presentation of the same candidate.
    for candidate_id in consistency_ids:
        view = view_by_id[candidate_id]
        presentations.append(
            QueuePresentation(
                presentation_id=f"{candidate_id}__recheck",
                candidate_id=candidate_id,
                episode_id=view.episode_id,
                series_id=view.series_id,
                dataset_split=view.dataset_split,
                stage="consistency",
                score_stratum=strata.classify(view.heuristic_score),
                is_repeat=True,
                repeat_of=candidate_id,
            )
        )
    return presentations


def _coverage(
    selected: Sequence[CandidateView],
    strata: ScoreStrata,
    config: QueueConfig,
    split_targets: Mapping[str, int],
) -> dict[str, Any]:
    def counts(fn) -> dict[str, int]:
        out: dict[str, int] = {}
        for view in selected:
            out[fn(view)] = out.get(fn(view), 0) + 1
        return dict(sorted(out.items()))

    return {
        "selected_count": len(selected),
        "by_split": counts(lambda v: v.dataset_split),
        "by_split_target": dict(split_targets),
        "by_score_stratum": counts(lambda v: strata.classify(v.heuristic_score)),
        "by_series": counts(lambda v: v.series_id),
        "by_episode": counts(lambda v: v.episode_id),
        "by_primary_source": counts(lambda v: v.primary_source),
        "by_content_type": counts(lambda v: v.content_type),
        "by_position_bucket": counts(
            lambda v: v.position_bucket(config.position_buckets)
        ),
        "by_silence_bucket": counts(
            lambda v: v.silence_bucket(config.silence_bucket_edges_ms)
        ),
        "by_sentence_boundary": counts(
            lambda v: (
                "unknown" if v.sentence_end is None else str(bool(v.sentence_end))
            )
        ),
        "by_feature_status": counts(lambda v: v.feature_status or "unknown"),
        "transcript_available": sum(1 for v in selected if v.transcript_available),
        "audio_available": sum(1 for v in selected if v.audio_available),
        "text_available": sum(1 for v in selected if v.text_available),
        "signal_disagreement": sum(1 for v in selected if v.is_disagreement),
        "max_per_episode_selected": max(
            (list(counts(lambda v: v.episode_id).values()) or [0])
        ),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_queue(path: Path, queue: LabellingQueue, force: bool = False) -> None:
    """Write a queue artifact, refusing to silently overwrite a different one.

    A queue is a commitment for a labelling round: annotators work through it and
    the readiness gate reads it. Overwriting one with different content would
    orphan the labels already collected against it, so an existing file with
    different bytes is a hard error unless ``force`` is set.
    """
    destination = Path(path)
    payload = json.dumps(queue.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if destination.exists() and not force:
        existing = destination.read_text(encoding="utf-8")
        if json.loads(existing) == queue.to_dict():
            return
        raise FileExistsError(
            f"{destination} already exists with different content. A labelling "
            "queue is immutable once labels are collected against it: bump the "
            "queue version, or pass force=True if no labels depend on it yet."
        )
    atomic_write_bytes(destination, payload.encode("utf-8"))


def read_queue(path: Path) -> LabellingQueue:
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Queue artifact not found: {file_path}") from error
    if str(raw.get("schema_version")) != LABELLING_QUEUE_SCHEMA_VERSION:
        raise QueueError(
            f"{file_path} is queue schema {raw.get('schema_version')!r}, but this "
            f"build reads {LABELLING_QUEUE_SCHEMA_VERSION!r}. Regenerate the queue."
        )
    return LabellingQueue.from_mapping(raw)
