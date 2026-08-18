"""Building multimodal features for one ad-hoc audio file.

The corpus pipeline is manifest-driven: it walks ``data/manifests/episodes.jsonl``
and caches every stage under ``data/``. A product request has none of that -- it
has one uploaded file and needs an answer.

Rather than reimplement the feature path (which would drift from the layout the
checkpoint was trained on, silently, and produce a model scoring mis-columned
inputs), this module builds a **throwaway single-episode corpus** in a scratch
directory and runs the real Phase 3 stages over it. Same normalization, same
candidate generator, same Whisper encoder, same MiniLM encoder, same assembler,
same loader. The only thing that differs is where the bytes live.

Cost of that choice: a scratch directory per request, and the models are loaded
per process. Benefit: there is exactly one implementation of "what a candidate's
feature vector is", so a training/serving skew bug is not expressible.

The episode is assigned the ``development`` split. It is not train, validation
or test -- an uploaded file has no place in any of them, and ``development`` is
the assigned split that carries no evaluation meaning.
"""

from __future__ import annotations

import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from slotify_rank.candidates.config import GenerationConfig, load_generation_config
from slotify_rank.candidates.generate import generate_for_episode
from slotify_rank.config.feature_settings import (
    EmbeddingConfig,
    FeatureConfig,
    PipelineConfig,
    TranscriptionConfig,
    load_embedding_config,
    load_feature_config,
    load_pipeline_config,
    load_transcription_config,
)
from slotify_rank.data.checksum import sha256_file
from slotify_rank.data.normalize import normalize_episode
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.probe import probe_audio
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.datasets.loader import EligibilityConfig, LoadedDataset, build_examples
from slotify_rank.features.assemble import assemble_episode
from slotify_rank.pipeline.stages import (
    StageContext,
    audio_embedding_path,
    read_handcrafted,
    run_acoustic,
    run_audio_embeddings,
    run_text_embeddings,
    run_transcribe,
    text_embedding_path,
)

__all__ = [
    "InferenceWorkspace",
    "PreparedEpisode",
    "workspace",
    "prepare_episode",
]

#: The split an uploaded file belongs to. Never train/validation/test: a product
#: request must not be able to look like an evaluation row.
INFERENCE_SPLIT = "development"

#: Episode id for an ad-hoc upload. Deterministic, so two runs over the same
#: audio produce identical candidate ids.
DEFAULT_EPISODE_ID = "upload"


@dataclass
class InferenceWorkspace:
    """A scratch single-episode corpus.

    ``repo_root`` is the scratch directory itself, not the real repository, so
    every manifest-relative path the pipeline writes stays inside it and nothing
    can touch the committed corpus.
    """

    root: Path
    paths: DataPaths

    @property
    def raw_dir(self) -> Path:
        return self.paths.raw_dir


@contextmanager
def workspace(root: Path | str, cleanup: bool = True) -> Iterator[InferenceWorkspace]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    paths = DataPaths(repo_root=directory, data_root=directory / "data")
    paths.mkdirs()
    try:
        yield InferenceWorkspace(root=directory, paths=paths)
    finally:
        if cleanup:
            shutil.rmtree(directory, ignore_errors=True)


@dataclass
class PreparedEpisode:
    """Everything scoring needs for one uploaded file."""

    episode: EpisodeRecord
    candidates: tuple[DatasetCandidate, ...]
    dataset: LoadedDataset
    duration_seconds: float | None
    timestamps_ms: dict[str, int]
    feature_status: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    timings_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfigs:
    """The four configs the Phase 3 stages need, loaded once."""

    transcription: TranscriptionConfig
    features: FeatureConfig
    embeddings: EmbeddingConfig
    pipeline: PipelineConfig
    generation: GenerationConfig

    @classmethod
    def load(cls, config_dir: Path | str | None = None) -> "PipelineConfigs":
        base = Path(config_dir) if config_dir else None
        return cls(
            transcription=load_transcription_config(
                base / "transcription_v1.yaml" if base else None
            ),
            features=load_feature_config(base / "features_v1.yaml" if base else None),
            embeddings=load_embedding_config(
                base / "embeddings_v1.yaml" if base else None
            ),
            pipeline=load_pipeline_config(
                base / "feature_pipeline_v1.yaml" if base else None
            ),
            generation=load_generation_config(base / "dataset_v1.yaml" if base else None),
        )


def _register(
    space: InferenceWorkspace, audio_path: Path, episode_id: str
) -> EpisodeRecord:
    """Copy the upload into the scratch corpus and probe it."""
    destination = space.paths.raw_dir / f"{episode_id}{audio_path.suffix or '.audio'}"
    shutil.copyfile(audio_path, destination)
    metadata = probe_audio(destination)
    return EpisodeRecord(
        episode_id=episode_id,
        series_id=episode_id,
        title=audio_path.name,
        source_type="local_file",
        source_uri=str(audio_path),
        source_name="product_upload",
        license_name=None,
        license_url=None,
        attribution=None,
        language="en",
        content_type="podcast",
        original_path=space.paths.relative(destination),
        sha256=sha256_file(destination),
        status="probed",
        duration_ms=metadata.duration_ms,
        sample_rate_hz=metadata.sample_rate_hz,
        channels=metadata.channels,
        file_format=metadata.file_format,
        # An upload is not corpus material: it is never split, never labelled,
        # and never counted in a dataset statistic.
        is_target_domain=True,
        notes="Ad-hoc product upload scored by the learned ranker; not corpus data.",
    )


def prepare_episode(
    space: InferenceWorkspace,
    audio_path: Path | str,
    configs: PipelineConfigs,
    episode_id: str = DEFAULT_EPISODE_ID,
    transcribe: bool = True,
    log: Any = None,
) -> PreparedEpisode:
    """Normalize, generate candidates, extract features, and build examples.

    ``transcribe=False`` skips Whisper transcription. Every candidate then has
    no transcript context, so its text block is masked and its ``feature_status``
    is ``audio_only`` -- which the model handles by construction and which the
    response reports. It is a latency trade, never a silent one.
    """
    logger = log or (lambda _message: None)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    started = time.perf_counter()
    episode = _register(space, Path(audio_path), episode_id)
    result = normalize_episode(episode, space.paths)
    episode = result.episode
    timings["normalize"] = round(time.perf_counter() - started, 3)
    duration_seconds = (
        (episode.normalized_duration_ms or episode.duration_ms or 0) / 1000.0 or None
    )

    started = time.perf_counter()
    candidates, report = generate_for_episode(
        episode,
        space.paths,
        config=configs.generation,
        include_product_padding=False,
    )
    timings["candidates"] = round(time.perf_counter() - started, 3)
    logger(f"  candidates: {len(candidates)}")
    if not candidates:
        return PreparedEpisode(
            episode=episode,
            candidates=(),
            dataset=LoadedDataset(),
            duration_seconds=duration_seconds,
            timestamps_ms={},
            feature_status={},
            warnings=["Candidate generation found no eligible insertion point."],
            timings_seconds=timings,
        )

    by_episode = {episode.episode_id: candidates}
    context = StageContext(
        paths=space.paths,
        transcription=configs.transcription,
        features=configs.features,
        embeddings=configs.embeddings,
        pipeline=configs.pipeline,
        log=logger,
    )

    if transcribe:
        started = time.perf_counter()
        outcome, _ = run_transcribe(context, [episode])
        timings["transcribe"] = round(time.perf_counter() - started, 3)
        if outcome.failed:
            warnings.append(
                "Transcription failed; candidates are scored with their text "
                "modality masked."
            )
    else:
        warnings.append(
            "Transcription was skipped; candidates are scored with their text "
            "modality masked."
        )

    started = time.perf_counter()
    acoustic_outcome, _ = run_acoustic(context, [episode], by_episode)
    timings["acoustic"] = round(time.perf_counter() - started, 3)
    if acoustic_outcome.failed:
        raise RuntimeError(
            "Handcrafted feature extraction failed for this audio; no candidate "
            "can be scored."
        )

    started = time.perf_counter()
    run_audio_embeddings(context, [episode], by_episode)
    timings["audio_embeddings"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    run_text_embeddings(context, [episode])
    timings["text_embeddings"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    split_lookup = {episode.episode_id: INFERENCE_SPLIT}
    assembly = assemble_episode(
        space.paths,
        episode.episode_id,
        candidates,
        read_handcrafted(space.paths, episode.episode_id),
        split_lookup,
        audio_embedding_path(space.paths, episode.episode_id),
        text_embedding_path(space.paths, episode.episode_id),
        _spec_for(space, episode.episode_id),
    )
    for candidate_id, reason in assembly.failures:
        warnings.append(f"{candidate_id}: {reason}")

    header = {
        "handcrafted_feature_names": list(
            _spec_for(space, episode.episode_id).names
        ),
        "feature_spec_version": (
            assembly.records[0].feature_spec_version if assembly.records else ""
        ),
        "feature_pipeline_version": (
            assembly.records[0].feature_pipeline_version if assembly.records else ""
        ),
    }
    dataset = build_examples(
        space.paths,
        header,
        assembly.records,
        labels=None,
        # Inference scores unlabelled candidates by definition, and a candidate
        # with no transcript is legitimately audio-only rather than unusable.
        config=EligibilityConfig(
            require_labels=False,
            require_acceptability_label=False,
            require_complete_multimodal=False,
            splits=(INFERENCE_SPLIT,),
        ),
        candidates=candidates,
        split_lookup=split_lookup,
    )
    timings["assemble"] = round(time.perf_counter() - started, 3)

    return PreparedEpisode(
        episode=episode,
        candidates=tuple(candidates),
        dataset=dataset,
        duration_seconds=duration_seconds,
        timestamps_ms={c.candidate_id: int(c.timestamp_ms) for c in candidates},
        feature_status={r.candidate_id: r.feature_status for r in assembly.records},
        warnings=warnings,
        timings_seconds=timings,
    )


def _spec_for(space: InferenceWorkspace, episode_id: str):
    """The feature spec this episode's extracted features imply.

    Built from the extracted values rather than hard-coded, exactly as the
    corpus pipeline does, so the column order here is the column order there.
    """
    from slotify_rank.features.assemble import build_spec

    handcrafted = read_handcrafted(space.paths, episode_id)
    if handcrafted is None:
        raise RuntimeError(
            f"{episode_id}: no handcrafted features were extracted, so the feature "
            "column layout is unknown."
        )
    return build_spec([handcrafted])


def episode_examples(prepared: PreparedEpisode) -> Sequence[Any]:
    """The eligible examples, in a deterministic order."""
    return sorted(
        prepared.dataset.examples,
        key=lambda example: (
            prepared.timestamps_ms.get(example.candidate_id, 0),
            example.candidate_id,
        ),
    )


def exclusion_dicts(prepared: PreparedEpisode) -> list[Mapping[str, str]]:
    return [
        {
            "candidate_id": exclusion.candidate_id,
            "reason": exclusion.reason,
            "detail": exclusion.detail,
        }
        for exclusion in prepared.dataset.exclusions
    ]
