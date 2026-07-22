"""Build an episode's dataset candidate pool.

Pipeline, in order:

1. Load the normalized 16 kHz mono render as an energy envelope.
2. Run every enabled generator (silence, pause, RMS minimum, fixed interval,
   transcript segment ends).
3. Apply the edge guard.
4. Merge near-duplicates, preserving every source flag.
5. Apply the density cap, if configured.
6. Score each survivor with the **Phase 1 baseline scorer**, unchanged, and
   attach the component breakdown.
7. Emit :class:`~slotify_rank.data.schema.DatasetCandidate` records.

Step 6 is a straight call into ``heuristic_offline_v1``; nothing is
reimplemented here. That is what makes ``heuristic_score`` on a dataset row and
the score the product would compute the same number.

The product's *padding* fallbacks -- the ratio and spacing slots it invents when
it cannot find three candidates -- are not produced by this module at all.
:func:`product_padding_candidates` exists to record them for auditing, and every
record it emits is ``is_synthetic=True`` with both eligibility flags off, which
the schema enforces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.candidates.audio_candidates import (
    RawCandidate,
    apply_edge_guard,
    energy_summary_values,
    fixed_interval_candidates,
    pause_candidates,
    rms_minimum_candidates,
    silence_candidates,
)
from slotify_rank.candidates.config import GenerationConfig
from slotify_rank.candidates.heuristic import rank_episode, score_candidate
from slotify_rank.candidates.merge import MergedCandidate, merge_raw_candidates
from slotify_rank.candidates.schema import (
    Candidate,
    EpisodeInput,
    make_candidate_id,
)
from slotify_rank.config.settings import HeuristicConfig, load_heuristic_config
from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    PREPROCESSING_VERSION,
)
from slotify_rank.data.audio_io import AudioEnvelope, load_normalized_wav
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EnergySummary, EpisodeRecord
from slotify_rank.candidates.transcript_candidates import (
    load_transcript,
    transcript_candidates,
)

__all__ = [
    "GenerationReport",
    "generate_for_episode",
    "product_padding_candidates",
]


@dataclass
class GenerationReport:
    """What happened, in enough detail to explain any change in the totals."""

    episode_id: str
    duration_ms: int
    count_by_source_before_merge: dict[str, int] = field(default_factory=dict)
    count_before_merge: int = 0
    count_after_merge: int = 0
    count_after_density_cap: int = 0
    dropped_edge_guard: int = 0
    dropped_density_cap: int = 0
    synthetic_recorded: int = 0
    transcript_available: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        return self.duration_ms / 60_000.0

    @property
    def candidates_per_minute(self) -> float:
        minutes = self.duration_minutes
        return 0.0 if minutes <= 0 else self.count_after_density_cap / minutes

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "duration_ms": self.duration_ms,
            "count_by_source_before_merge": dict(
                sorted(self.count_by_source_before_merge.items())
            ),
            "count_before_merge": self.count_before_merge,
            "count_after_merge": self.count_after_merge,
            "count_after_density_cap": self.count_after_density_cap,
            "dropped_edge_guard": self.dropped_edge_guard,
            "dropped_density_cap": self.dropped_density_cap,
            "synthetic_recorded": self.synthetic_recorded,
            "transcript_available": self.transcript_available,
            "candidates_per_minute": round(self.candidates_per_minute, 4),
            "notes": list(self.notes),
        }


def _collect_raw(
    envelope: AudioEnvelope,
    episode: EpisodeRecord,
    config: GenerationConfig,
    heuristic: HeuristicConfig,
    paths: DataPaths,
    report: GenerationReport,
) -> list[RawCandidate]:
    profile = heuristic.canonical
    min_silence_ms, offset_db, fallback_dbfs = config.resolved_silence(profile)
    duration_ms = envelope.n_ms
    raw: list[RawCandidate] = []

    if config.silence.enabled:
        raw.extend(
            silence_candidates(envelope, min_silence_ms, offset_db, fallback_dbfs)
        )
    if config.pause.enabled:
        raw.extend(pause_candidates(envelope, config.pause, fallback_dbfs))
    if config.rms_minimum.enabled:
        raw.extend(rms_minimum_candidates(envelope, config.rms_minimum))
    if config.fixed_interval.enabled:
        raw.extend(fixed_interval_candidates(duration_ms, config.fixed_interval))

    if config.transcript.enabled:
        if episode.transcript_path:
            segments = load_transcript(paths.absolute(episode.transcript_path))
            raw.extend(transcript_candidates(segments, config.transcript))
            report.transcript_available = True
        else:
            report.notes.append(
                "no transcript supplied; transcript_segment_end contributed 0 "
                "candidates (Phase 2 does not generate transcripts)"
            )

    for candidate in raw:
        report.count_by_source_before_merge[candidate.source] = (
            report.count_by_source_before_merge.get(candidate.source, 0) + 1
        )
    report.count_before_merge = len(raw)
    return raw


def _apply_density_cap(
    merged: Sequence[MergedCandidate],
    scored: Sequence[float],
    duration_ms: int,
    config: GenerationConfig,
    report: GenerationReport,
) -> list[int]:
    """Return the indices to keep, dropping the lowest-scoring over the cap."""
    limit = config.max_candidates_per_minute
    minutes = duration_ms / 60_000.0
    if limit is None or minutes <= 0:
        return list(range(len(merged)))
    maximum = max(1, int(minutes * limit))
    if len(merged) <= maximum:
        return list(range(len(merged)))
    # Keep the highest-scoring candidates; ties break by earlier timestamp so the
    # cap is deterministic.
    order = sorted(
        range(len(merged)), key=lambda index: (-scored[index], merged[index].ms)
    )
    kept = sorted(order[:maximum])
    report.dropped_density_cap = len(merged) - len(kept)
    report.notes.append(
        f"density cap of {limit}/min kept {len(kept)} of {len(merged)} merged "
        f"candidates; raise candidate_generation.max_candidates_per_minute to keep more"
    )
    return kept


def generate_for_episode(
    episode: EpisodeRecord,
    paths: DataPaths,
    config: GenerationConfig | None = None,
    heuristic: HeuristicConfig | None = None,
    include_product_padding: bool = False,
) -> tuple[list[DatasetCandidate], GenerationReport]:
    """Generate the candidate pool for one normalized episode."""
    config = config or GenerationConfig()
    heuristic = heuristic or load_heuristic_config()
    profile = heuristic.canonical

    if episode.status != "normalized" or not episode.normalized_path:
        raise ValueError(
            f"{episode.episode_id}: candidates require normalized audio. Run "
            "`dataset normalize` first."
        )
    envelope = load_normalized_wav(paths.absolute(episode.normalized_path))
    duration_ms = envelope.n_ms
    duration_seconds = duration_ms / 1000.0
    report = GenerationReport(episode_id=episode.episode_id, duration_ms=duration_ms)

    raw = _collect_raw(envelope, episode, config, heuristic, paths, report)
    guarded, dropped = apply_edge_guard(raw, duration_ms, config.edge_guard_ms)
    report.dropped_edge_guard = dropped

    merged = merge_raw_candidates(guarded, config.merge_tolerance_ms)
    report.count_after_merge = len(merged)

    scores = [
        score_candidate(
            Candidate(
                ms=candidate.ms,
                silence_ms=candidate.silence_ms,
                snippet=candidate.snippet,
                sources=(),
            ),
            duration_seconds,
            "podcast",
            profile,
        )
        for candidate in merged
    ]
    kept_indices = _apply_density_cap(
        merged, [scored.score for scored in scores], duration_ms, config, report
    )
    report.count_after_density_cap = len(kept_indices)

    records: list[DatasetCandidate] = []
    for index in kept_indices:
        candidate = merged[index]
        scored = scores[index]
        mean_dbfs, min_dbfs, max_dbfs = energy_summary_values(
            envelope, candidate.ms, config.energy_summary_window_ms
        )
        records.append(
            DatasetCandidate(
                episode_id=episode.episode_id,
                candidate_id=make_candidate_id(episode.episode_id, candidate.ms),
                timestamp_ms=candidate.ms,
                candidate_sources=candidate.sources,
                is_synthetic=False,
                eligible_for_labelling=True,
                eligible_for_evaluation=True,
                pause_duration_ms=candidate.pause_ms,
                silence_duration_ms=candidate.silence_ms,
                local_energy_summary=EnergySummary(
                    window_ms=config.energy_summary_window_ms,
                    mean_dbfs=round(mean_dbfs, 3),
                    min_dbfs=round(min_dbfs, 3),
                    max_dbfs=round(max_dbfs, 3),
                ),
                sentence_end=candidate.sentence_end,
                transcript_segment_id=candidate.transcript_segment_id,
                transcript_before=candidate.transcript_before,
                transcript_after=candidate.transcript_after,
                normalized_episode_position=(
                    None if duration_ms <= 0 else candidate.ms / duration_ms
                ),
                raw_component_scores=scored.components.to_dict(),
                heuristic_score=scored.score,
                baseline_version=heuristic.canonical_profile_name,
                config_version=heuristic.config_version,
                candidate_generation_version=CANDIDATE_GENERATION_VERSION,
                preprocessing_version=(
                    episode.preprocessing_version or PREPROCESSING_VERSION
                ),
                merged_from_ms=candidate.merged_from_ms,
            )
        )

    if include_product_padding:
        padding = product_padding_candidates(
            episode, records, duration_seconds, heuristic
        )
        report.synthetic_recorded = len(padding)
        records.extend(padding)

    return records, report


def product_padding_candidates(
    episode: EpisodeRecord,
    real_candidates: Sequence[DatasetCandidate],
    duration_seconds: float,
    heuristic: HeuristicConfig,
) -> list[DatasetCandidate]:
    """Record (never generate) the product's invented padding slots.

    When fewer than three real candidates exist, the product fabricates slots at
    fixed ratios or fixed spacings so the UI always has three. Those timestamps
    are not evidence about the audio and must never be labelled, trained on, or
    counted. They are captured here purely so an audit can show *which*
    timestamps the product would have invented, and every one is pinned to
    ``is_synthetic=True`` / both eligibility flags false -- an invariant
    :class:`~slotify_rank.data.schema.DatasetCandidate` enforces in its
    constructor.
    """
    ranking = rank_episode(
        EpisodeInput(
            episode_id=episode.episode_id,
            duration_seconds=duration_seconds,
            mode="podcast",
            silence_candidates=tuple(
                Candidate(
                    ms=candidate.timestamp_ms,
                    silence_ms=candidate.silence_duration_ms,
                    snippet=candidate.transcript_before or "",
                    sources=(),
                )
                for candidate in real_candidates
            ),
        ),
        config=heuristic,
    )
    return [
        DatasetCandidate(
            episode_id=episode.episode_id,
            candidate_id=record.candidate_id,
            timestamp_ms=record.timestamp_ms,
            candidate_sources=("product_padding",),
            is_synthetic=True,
            eligible_for_labelling=False,
            eligible_for_evaluation=False,
            heuristic_score=record.total_score,
            raw_component_scores=record.raw_component_scores.to_dict(),
            baseline_version=heuristic.canonical_profile_name,
            config_version=heuristic.config_version,
            preprocessing_version=(
                episode.preprocessing_version or PREPROCESSING_VERSION
            ),
            normalized_episode_position=None,
        )
        for record in ranking.candidates
        if record.is_synthetic
    ]


def generate_for_episodes(
    episodes: Sequence[EpisodeRecord],
    paths: DataPaths,
    config: GenerationConfig | None = None,
    heuristic: HeuristicConfig | None = None,
    include_product_padding: bool = False,
) -> tuple[list[DatasetCandidate], list[GenerationReport]]:
    config = config or GenerationConfig()
    heuristic = heuristic or load_heuristic_config()
    all_records: list[DatasetCandidate] = []
    reports: list[GenerationReport] = []
    for episode in episodes:
        records, report = generate_for_episode(
            episode,
            paths,
            config=config,
            heuristic=heuristic,
            include_product_padding=include_product_padding,
        )
        all_records.extend(records)
        reports.append(report)
    return all_records, reports
