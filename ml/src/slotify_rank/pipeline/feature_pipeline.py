"""End-to-end orchestration: manifests in, validated feature records out.

The stage order is not arbitrary. Transcription must precede handcrafted
features (which read transcript context), handcrafted features must precede text
embeddings (which embed the context strings those features describe), and
assembly must come last. Running them in one process lets each model be loaded,
used across the whole selection, and released before the next one loads --
keeping peak memory at one model rather than two.

Every stage is individually resumable, so an interrupted run continues from
where it stopped rather than from the beginning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from slotify_rank.config.versions import FEATURE_PIPELINE_VERSION
from slotify_rank.data import manifests
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.features.assemble import (
    AssemblyResult,
    assemble_episode,
    build_spec,
    write_feature_manifest,
)
from slotify_rank.features.schema import CandidateFeatureRecord, FeatureSpec
from slotify_rank.pipeline.stages import (
    StageContext,
    StageOutcome,
    audio_embedding_path,
    eligible_candidates,
    read_handcrafted,
    run_acoustic,
    run_audio_embeddings,
    run_text_embeddings,
    run_transcribe,
    text_embedding_path,
)

__all__ = ["PipelineResult", "run_feature_pipeline", "run_assemble"]

#: Stages a caller may request individually via ``--stage``.
SELECTABLE_STAGES = ("transcribe", "acoustic", "audio_embedding", "text_embedding", "assemble")


@dataclass
class PipelineResult:
    outcomes: list[StageOutcome] = field(default_factory=list)
    records: list[CandidateFeatureRecord] = field(default_factory=list)
    spec: FeatureSpec | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)
    excluded_synthetic: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "stages": [outcome.to_dict() for outcome in self.outcomes],
            "record_count": len(self.records),
            "failures": [
                {"candidate_id": cid, "reason": reason} for cid, reason in self.failures
            ],
            "excluded_synthetic": self.excluded_synthetic,
            "seconds": round(self.seconds, 3),
        }


def run_assemble(
    context: StageContext,
    episodes: Sequence[EpisodeRecord],
    candidates_by_episode: Mapping[str, Sequence[DatasetCandidate]],
    split_lookup: Mapping[str, str],
) -> tuple[StageOutcome, AssemblyResult, FeatureSpec]:
    """Join handcrafted features and embeddings into candidate records."""
    started = time.perf_counter()
    outcome = StageOutcome(stage="assemble")

    payloads = []
    for episode in episodes:
        handcrafted = read_handcrafted(context.paths, episode.episode_id)
        if handcrafted is not None:
            payloads.append(handcrafted)

    spec = build_spec(payloads)
    combined = AssemblyResult()
    for episode in episodes:
        handcrafted = read_handcrafted(context.paths, episode.episode_id)
        result = assemble_episode(
            paths=context.paths,
            episode_id=episode.episode_id,
            candidates=candidates_by_episode.get(episode.episode_id, ()),
            handcrafted=handcrafted,
            split_lookup=split_lookup,
            audio_array_path=audio_embedding_path(context.paths, episode.episode_id),
            text_array_path=text_embedding_path(context.paths, episode.episode_id),
            spec=spec,
        )
        combined.extend(result)
        outcome.processed += 1

    outcome.failed = len(combined.failures)
    outcome.details["records"] = len(combined.records)
    outcome.details["excluded_synthetic"] = combined.excluded_synthetic
    outcome.seconds = time.perf_counter() - started
    return outcome, combined, spec


def run_feature_pipeline(
    context: StageContext,
    episodes: Sequence[EpisodeRecord],
    candidates: Sequence[DatasetCandidate],
    split_lookup: Mapping[str, str] | None = None,
    stages: Sequence[str] = SELECTABLE_STAGES,
    edge_guard_ms: int = 5_000,
) -> PipelineResult:
    """Run the requested stages in order over the selected episodes."""
    started = time.perf_counter()
    result = PipelineResult()
    split_lookup = dict(split_lookup or {})

    by_episode: dict[str, list[DatasetCandidate]] = {}
    for candidate in candidates:
        by_episode.setdefault(candidate.episode_id, []).append(candidate)

    unknown = sorted(set(stages) - set(SELECTABLE_STAGES))
    if unknown:
        raise ValueError(
            f"Unknown stage(s) {unknown}; known stages are {list(SELECTABLE_STAGES)}"
        )

    if "transcribe" in stages:
        context.log("transcribe")
        outcome, _ = run_transcribe(context, episodes)
        result.outcomes.append(outcome)

    if "acoustic" in stages:
        context.log("acoustic + structural features")
        outcome, _ = run_acoustic(context, episodes, by_episode, edge_guard_ms)
        result.outcomes.append(outcome)

    if "audio_embedding" in stages:
        context.log("whisper audio representations")
        outcome, _ = run_audio_embeddings(context, episodes, by_episode)
        result.outcomes.append(outcome)

    if "text_embedding" in stages:
        context.log("minilm transcript embeddings")
        outcome, _ = run_text_embeddings(context, episodes)
        result.outcomes.append(outcome)

    if "assemble" in stages:
        context.log("assemble")
        outcome, assembly, spec = run_assemble(
            context, episodes, by_episode, split_lookup
        )
        result.outcomes.append(outcome)
        result.records = assembly.records
        result.failures = assembly.failures
        result.excluded_synthetic = assembly.excluded_synthetic
        result.spec = spec

        write_feature_manifest(
            context.paths.features_manifest,
            spec,
            assembly.records,
            metadata={
                "audio_embedding_model": context.embeddings.audio.model_id,
                "text_embedding_model": context.embeddings.text.model_id,
                "transcription_model": context.transcription.model_id,
            },
        )
        context.log(
            f"  wrote {len(assembly.records)} record(s) to "
            f"{context.paths.features_manifest}"
        )

    result.seconds = time.perf_counter() - started
    return result
