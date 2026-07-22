"""Deterministic synthetic Phase-3-shaped fixtures.

**These are not data.** Nothing produced here describes real audio and no score
here is a human judgement. The fixtures exist so the real training path -- the
real loader, the real normalizer, the real pair generator, the real PyTorch
modules, the real optimizer -- can be exercised end to end before enough genuine
labels exist to train on. Every artifact written by this module carries
``"synthetic": true`` and a ``label_source`` of ``"synthetic"``, and the
training reports propagate that flag, so a metric measured on it can never be
mistaken for a model-quality result.

Why not just mock the tensors: because the failures worth catching live in the
joins -- a row id that does not resolve, a split that leaks, a feature column
that shifts. A mock skips exactly the code that breaks.

The generated targets are a deterministic function of a handful of the
generated features plus seeded noise, so a working model *can* fit them. That
makes the smoke run able to fail: a training loop that does not learn a signal
that is provably present is broken, whereas a loop trained on pure noise looks
identical whether it works or not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slotify_rank.config.versions import (
    CANDIDATE_GENERATION_VERSION,
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    LABEL_RUBRIC_VERSION,
)
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.data.paths import DataPaths
from slotify_rank.datasets.schema import NATIVE_EMBEDDING_DIMENSION
from slotify_rank.embeddings.store import EmbeddingMetadata, write_embeddings
from slotify_rank.features.assemble import write_feature_manifest
from slotify_rank.features.schema import (
    CandidateFeatureRecord,
    EmbeddingReference,
    FeatureSpec,
    status_for,
)

__all__ = [
    "SyntheticCorpusConfig",
    "SyntheticCorpus",
    "build_synthetic_corpus",
    "SYNTHETIC_FEATURE_NAMES",
]

#: A small stand-in vocabulary shaped like the real one: continuous acoustic
#: scalars, a constant, and binary source one-hots.
SYNTHETIC_FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            "candidate_energy_mean_dbfs",
            "constant_probe",
            "context_cosine_similarity",
            "heuristic_total_score",
            "normalized_episode_position",
            "pause_duration_ms",
            "rms_delta_across_short",
            "rms_mean_before_short",
            "semantic_change_score",
            "sentence_end",
            "source_pause",
            "source_silence",
        )
    )
)

_BINARY_FEATURES = frozenset({"sentence_end", "source_pause", "source_silence"})
_CONSTANT_FEATURES = frozenset({"constant_probe"})

#: Weights of the latent "true" quality function, over normalized features.
_LATENT_WEIGHTS: dict[str, float] = {
    "pause_duration_ms": 1.4,
    "semantic_change_score": 1.1,
    "sentence_end": 0.9,
    "rms_delta_across_short": -0.6,
    "normalized_episode_position": -0.3,
}


@dataclass(frozen=True)
class SyntheticCorpusConfig:
    episode_count: int = 6
    candidates_per_episode: int = 8
    seed: int = 20260722
    native_dimension: int = NATIVE_EMBEDDING_DIMENSION
    #: Fraction of episodes whose text modality is withheld, so the missing-
    #: modality path is exercised rather than merely implemented.
    text_missing_episode_stride: int = 0
    annotator_id: str = "synthetic-annotator"
    acceptable_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.episode_count < 3:
            raise ValueError(
                "A synthetic corpus needs at least 3 episodes so train, validation "
                "and test are each non-empty"
            )
        if self.candidates_per_episode < 2:
            raise ValueError(
                "An episode with fewer than 2 candidates can never produce a "
                "within-episode pair"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "candidates_per_episode": self.candidates_per_episode,
            "seed": self.seed,
            "native_dimension": self.native_dimension,
            "text_missing_episode_stride": self.text_missing_episode_stride,
            "acceptable_threshold": self.acceptable_threshold,
            "synthetic": True,
        }


@dataclass(frozen=True)
class SyntheticCorpus:
    features_manifest: Path
    label_export: Path
    episode_ids: tuple[str, ...]
    candidate_count: int
    splits: dict[str, str]
    config: SyntheticCorpusConfig


def _split_for(index: int) -> str:
    """Round-robin so every split is populated at the smallest useful size."""
    return ("train", "train", "validation", "train", "test")[index % 5]


def build_synthetic_corpus(
    paths: DataPaths, config: SyntheticCorpusConfig | None = None
) -> SyntheticCorpus:
    """Write a complete synthetic feature manifest, embeddings and label export."""
    config = config or SyntheticCorpusConfig()
    paths.mkdirs()
    rng = np.random.default_rng(config.seed)
    spec = FeatureSpec.from_names(SYNTHETIC_FEATURE_NAMES)

    records: list[CandidateFeatureRecord] = []
    label_rows: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    splits: dict[str, str] = {}

    for episode_index in range(config.episode_count):
        episode_id = f"synthetic-episode-{episode_index:03d}"
        episode_ids.append(episode_id)
        split = _split_for(episode_index)
        splits[episode_id] = split

        text_available = not (
            config.text_missing_episode_stride
            and episode_index % config.text_missing_episode_stride == 0
        )

        audio_path = paths.audio_embeddings_dir / f"{episode_id}.audio.npy"
        text_path = paths.text_embeddings_dir / f"{episode_id}.text.npy"

        audio_rows: list[str] = []
        audio_vectors: list[np.ndarray] = []
        text_rows: list[str] = []
        text_vectors: list[np.ndarray] = []

        for candidate_index in range(config.candidates_per_episode):
            timestamp_ms = 30_000 + candidate_index * 17_000
            candidate_id = f"{episode_id}:{timestamp_ms:09d}"

            values, missing = _draw_features(rng)
            quality = _latent_quality(values, rng)

            # -- embeddings, correlated with the latent target so a multimodal
            # -- model has something a handcrafted-only model does not.
            base = rng.normal(0.0, 1.0, config.native_dimension).astype(np.float32)
            signal = np.zeros(config.native_dimension, dtype=np.float32)
            signal[: config.native_dimension // 8] = float(quality - 3.0) * 0.35
            before = base
            after = base + signal + rng.normal(0.0, 0.4, config.native_dimension).astype(
                np.float32
            )
            context = (before + after) / 2.0
            for kind, vector in (
                ("before", before),
                ("after", after),
                ("context", context),
                ("difference", after - before),
            ):
                audio_rows.append(f"{candidate_id}#{kind}")
                audio_vectors.append(vector.astype(np.float32))

            if text_available:
                text_before = rng.normal(0.0, 1.0, config.native_dimension).astype(
                    np.float32
                )
                text_after = text_before + signal * 0.8 + rng.normal(
                    0.0, 0.5, config.native_dimension
                ).astype(np.float32)
                for side, vector in (("before", text_before), ("after", text_after)):
                    text_rows.append(f"{candidate_id}#{side}")
                    text_vectors.append(vector.astype(np.float32))

            vector, mask = spec.vectorize(values, missing)
            audio_reference = {
                kind: EmbeddingReference(
                    path=paths.relative(audio_path),
                    row_id=f"{candidate_id}#{kind}",
                    dimension=config.native_dimension,
                )
                for kind in ("before", "after", "context", "difference")
            }
            text_reference = (
                {
                    side: EmbeddingReference(
                        path=paths.relative(text_path),
                        row_id=f"{candidate_id}#{side}",
                        dimension=config.native_dimension,
                    )
                    for side in ("before", "after")
                }
                if text_available
                else {}
            )

            records.append(
                CandidateFeatureRecord(
                    episode_id=episode_id,
                    candidate_id=candidate_id,
                    timestamp_ms=timestamp_ms,
                    dataset_split=split,
                    handcrafted_feature_values=vector,
                    handcrafted_missing_mask=mask,
                    feature_status=status_for(True, text_available),
                    audio_sha256="0" * 64,
                    candidate_generation_version=CANDIDATE_GENERATION_VERSION,
                    feature_pipeline_version=FEATURE_PIPELINE_VERSION,
                    feature_spec_version=FEATURE_SPEC_VERSION,
                    audio_embedding_reference=audio_reference,
                    text_embedding_reference=text_reference,
                    audio_embedding_dimension=config.native_dimension,
                    text_embedding_dimension=(
                        config.native_dimension if text_available else None
                    ),
                    audio_embedding_available=True,
                    text_embedding_available=text_available,
                    transcript_available=text_available,
                )
            )

            label_rows.append(
                {
                    "candidate_id": candidate_id,
                    "episode_id": episode_id,
                    "annotator_id": config.annotator_id,
                    "quality_score": int(round(quality)),
                    "is_acceptable": quality >= config.acceptable_threshold,
                    "is_unusable": False,
                    "notes": None,
                    "rubric_version": LABEL_RUBRIC_VERSION,
                    "label_source": "human",
                    "synthetic": True,
                }
            )

        _write_array(
            audio_path,
            audio_vectors,
            audio_rows,
            episode_id,
            "audio_embedding",
            config.native_dimension,
        )
        if text_available:
            _write_array(
                text_path,
                text_vectors,
                text_rows,
                episode_id,
                "text_embedding",
                config.native_dimension,
            )

    write_feature_manifest(
        paths.features_manifest,
        spec,
        records,
        metadata={
            "synthetic": True,
            "note": (
                "Synthetic Phase 4 fixture. Not real audio, not human labels. "
                "Metrics measured on this corpus are evidence that the training "
                "system runs, never evidence of model quality."
            ),
        },
    )

    export_path = paths.labels_dir / "labels_synthetic.jsonl"
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in label_rows
    )
    atomic_write_bytes(export_path, body.encode("utf-8"))
    metadata = {
        "dataset_version": "synthetic",
        "rubric_version": LABEL_RUBRIC_VERSION,
        "acceptable_threshold": config.acceptable_threshold,
        "acceptable_rule": f"quality_score >= {config.acceptable_threshold}",
        "row_count": len(label_rows),
        "annotators": [config.annotator_id],
        "label_source": "human",
        "synthetic": True,
        "note": (
            "SYNTHETIC FIXTURE. These are generated numbers, not human judgements. "
            "They exist to exercise the training path and must never be reported "
            "as labelled data."
        ),
    }
    atomic_write_bytes(
        export_path.with_suffix(export_path.suffix + ".meta.json"),
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    return SyntheticCorpus(
        features_manifest=paths.features_manifest,
        label_export=export_path,
        episode_ids=tuple(episode_ids),
        candidate_count=len(records),
        splits=splits,
        config=config,
    )


def _draw_features(rng: np.random.Generator) -> tuple[dict[str, float], dict[str, bool]]:
    values: dict[str, float] = {}
    missing: dict[str, bool] = {}
    for name in SYNTHETIC_FEATURE_NAMES:
        if name in _CONSTANT_FEATURES:
            values[name] = 1.0
            missing[name] = False
        elif name in _BINARY_FEATURES:
            values[name] = float(rng.integers(0, 2))
            missing[name] = False
        elif name == "pause_duration_ms":
            values[name] = float(rng.uniform(200.0, 2500.0))
            missing[name] = False
        elif name == "normalized_episode_position":
            values[name] = float(rng.uniform(0.0, 1.0))
            missing[name] = False
        elif name == "context_cosine_similarity":
            # Exercised as a genuinely missing scalar on a minority of rows.
            absent = bool(rng.random() < 0.15)
            values[name] = 0.0 if absent else float(rng.uniform(-1.0, 1.0))
            missing[name] = absent
        else:
            values[name] = float(rng.normal(0.0, 1.0))
            missing[name] = False
    values["semantic_change_score"] = 1.0 - values["context_cosine_similarity"]
    missing["semantic_change_score"] = missing["context_cosine_similarity"]
    return values, missing


def _latent_quality(values: dict[str, float], rng: np.random.Generator) -> float:
    """A learnable 1-5 target: a linear function of a few features plus noise."""
    score = 0.0
    for name, weight in _LATENT_WEIGHTS.items():
        raw = values[name]
        if name == "pause_duration_ms":
            raw = (raw - 1350.0) / 650.0
        score += weight * raw
    score += float(rng.normal(0.0, 0.35))
    # Map onto the 1-5 rubric range, then clamp.
    return float(np.clip(3.0 + score, 1.0, 5.0))


def _write_array(
    path: Path,
    vectors: list[np.ndarray],
    row_ids: list[str],
    episode_id: str,
    kind: str,
    dimension: int,
) -> None:
    matrix = np.stack(vectors).astype(np.float32)
    write_embeddings(
        path,
        matrix,
        EmbeddingMetadata(
            kind=kind,
            episode_id=episode_id,
            model_id="synthetic-fixture",
            model_revision="synthetic",
            dimension=dimension,
            row_ids=tuple(row_ids),
            identity_digest="synthetic-" + "0" * 55,
            extra={"synthetic": True},
        ),
    )
