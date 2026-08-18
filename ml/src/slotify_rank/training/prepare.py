"""Assembling a training-ready dataset from the Phase 3 artifacts.

Order matters here and is fixed:

1. read the feature manifest and the human labels;
2. decide eligibility, accounting for every excluded candidate;
3. apply any episode/candidate caps (smoke runs only, always recorded);
4. fit the scalar normalizer **on the training split alone**;
5. apply it to every split;
6. generate within-episode pairs, per split.

Step 4 before step 5 and never the reverse: fitting after merging is the single
most common way a small-dataset result turns out to be optimistic.

Nothing here loads a model. The embeddings are read from the ``.npy`` arrays the
Phase 3 pipeline already wrote, so preparing a dataset never re-runs Whisper or
MiniLM -- which is both a correctness property (the cached vectors are the ones
the checkpoint's schema describes) and the reason this takes seconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate
from slotify_rank.datasets.labels import (
    DEFAULT_ALLOWED_LABEL_SOURCES,
    LabelSet,
    read_label_export,
)
from slotify_rank.datasets.loader import LoadedDataset, build_examples
from slotify_rank.datasets.normalizer import FeatureNormalizer, fit_normalizer
from slotify_rank.datasets.ranking_dataset import (
    EpisodeGroups,
    MaterializedFeatures,
    materialize,
)
from slotify_rank.datasets.schema import DatasetSchema, TrainingExample
from slotify_rank.features.assemble import read_feature_manifest
from slotify_rank.ranking.pairs import PairSet, generate_pairs
from slotify_rank.training.config import TrainingConfig

__all__ = ["PreparedDataset", "prepare_dataset", "dataset_fingerprint"]


@dataclass
class PreparedDataset:
    """Everything the trainer needs, plus the accounting behind it."""

    schema: DatasetSchema
    normalizer: FeatureNormalizer
    features: MaterializedFeatures
    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]
    train_pairs: PairSet
    validation_pairs: PairSet
    loaded: LoadedDataset
    fingerprint: str
    label_set: LabelSet | None = None
    limits_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def validation_groups(self) -> EpisodeGroups:
        return EpisodeGroups(self.features, list(self.validation_rows))

    @property
    def train_groups(self) -> EpisodeGroups:
        return EpisodeGroups(self.features, list(self.train_rows))

    def summary(self) -> dict[str, Any]:
        train_episodes = {self.features.episode_ids[row] for row in self.train_rows}
        validation_episodes = {
            self.features.episode_ids[row] for row in self.validation_rows
        }
        # A candidate in both splits would be a leak that no metric can see.
        overlap = train_episodes & validation_episodes
        return {
            "dataset_fingerprint": self.fingerprint,
            "training_episode_count": len(train_episodes),
            "training_candidate_count": len(self.train_rows),
            "validation_episode_count": len(validation_episodes),
            "validation_candidate_count": len(self.validation_rows),
            "training_pair_count": len(self.train_pairs),
            "validation_pair_count": len(self.validation_pairs),
            "episode_overlap_between_splits": sorted(overlap),
            "limits_applied": dict(self.limits_applied),
            "eligibility": self.loaded.summary(),
            "train_pairs": self.train_pairs.summary(),
            "validation_pairs": self.validation_pairs.summary(),
            "normalizer": {
                key: value
                for key, value in self.normalizer.to_dict().items()
                # The full statistics live in normalizer.json; repeating 110
                # means and deviations here would bury the summary.
                if key
                not in ("means", "standard_deviations", "constant_feature_mask",
                        "binary_feature_mask", "feature_names")
            },
        }


def dataset_fingerprint(
    features_manifest: Path,
    label_export: Path | None,
    split_manifest: Path | None = None,
) -> str:
    """Content hash of the artifacts a run trained against.

    Hashing the file bytes rather than a version string: two corpora can share
    a pipeline version and contain entirely different episodes, and a run id
    that cannot tell them apart is worse than no run id.
    """
    digest = hashlib.sha256()
    for path in (features_manifest, label_export, split_manifest):
        if path is None or not Path(path).is_file():
            digest.update(b"\x00absent\x00")
            continue
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _apply_limits(
    examples: Sequence[TrainingExample],
    max_episodes: int | None,
    max_candidates: int | None,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Cap the dataset for a smoke run, keeping whole episodes together.

    Episodes are truncated as units, never candidates across them: half an
    episode would silently change what a within-episode ranking metric means.
    """
    applied: dict[str, Any] = {}
    kept = list(examples)

    if max_episodes is not None:
        # Deterministic: the first N episodes in canonical order, and enough of
        # each split that validation does not vanish.
        by_split: dict[str, list[str]] = {}
        for example in kept:
            bucket = by_split.setdefault(example.split, [])
            if example.episode_id not in bucket:
                bucket.append(example.episode_id)
        allowed: set[str] = set()
        for split in sorted(by_split):
            allowed.update(by_split[split][:max_episodes])
        before = len({e.episode_id for e in kept})
        kept = [example for example in kept if example.episode_id in allowed]
        applied["max_episodes"] = {
            "limit": max_episodes,
            "episodes_before": before,
            "episodes_after": len({e.episode_id for e in kept}),
        }

    if max_candidates is not None:
        before = len(kept)
        per_split: dict[str, int] = {}
        capped: list[TrainingExample] = []
        for example in kept:
            count = per_split.get(example.split, 0)
            if count >= max_candidates:
                continue
            per_split[example.split] = count + 1
            capped.append(example)
        kept = capped
        applied["max_candidates"] = {
            "limit": max_candidates,
            "candidates_before": before,
            "candidates_after": len(kept),
            "note": "applied per split",
        }

    return kept, applied


def prepare_dataset(
    paths: DataPaths,
    config: TrainingConfig,
    label_export: Path,
    candidates: Sequence[DatasetCandidate] | None = None,
    split_lookup: Mapping[str, str] | None = None,
    features_manifest: Path | None = None,
    native_embedding_dimension: int | None = None,
    allowed_label_sources: Sequence[str] = DEFAULT_ALLOWED_LABEL_SOURCES,
) -> PreparedDataset:
    """Build a training-ready dataset. Raises rather than degrading quietly.

    ``allowed_label_sources`` defaults to human labels only. A caller that names
    another source (the ``weak_heuristic`` bootstrap run) gets it recorded on the
    returned :class:`LabelSet`, and from there into the run report and the
    checkpoint -- so a run trained on weak labels can never present itself as
    human-supervised.
    """
    manifest_path = features_manifest or paths.features_manifest
    header, records = read_feature_manifest(manifest_path)
    labels = read_label_export(label_export, allowed_label_sources=allowed_label_sources)

    declared = native_embedding_dimension
    if declared is None:
        declared = _infer_native_dimension(records)

    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        config=config.eligibility_config(),
        candidates=candidates,
        split_lookup=split_lookup,
        native_embedding_dimension=declared,
    )
    if not loaded.examples:
        raise RuntimeError(
            "No candidate survived eligibility checking, so there is nothing to "
            f"train on. Exclusion counts: {dict(loaded.counts)}"
        )
    assert loaded.schema is not None

    examples, limits = _apply_limits(
        loaded.examples, config.max_episodes, config.max_candidates
    )

    train = [e for e in examples if e.split == config.train_split]
    validation = [e for e in examples if e.split == config.validation_split]
    if not train:
        raise RuntimeError(
            f"The {config.train_split!r} split is empty after eligibility and "
            "limits. Check the split manifest and the label coverage."
        )
    if not validation:
        raise RuntimeError(
            f"The {config.validation_split!r} split is empty, so no checkpoint "
            "could be selected. Training without validation is refused rather "
            "than run to a fixed epoch count."
        )

    train_episodes = {e.episode_id for e in train}
    validation_episodes = {e.episode_id for e in validation}
    overlap = train_episodes & validation_episodes
    if overlap:
        raise RuntimeError(
            f"Episode(s) {sorted(overlap)[:5]} appear in both the training and "
            "validation splits. That is leakage; refusing to train."
        )

    normalizer = fit_normalizer(
        train,
        loaded.schema.handcrafted_feature_names,
        training_split_hash=_split_hash(train),
        feature_pipeline_version=loaded.schema.feature_pipeline_version,
        feature_spec_version=loaded.schema.feature_spec_version,
        expected_split=config.train_split,
    )

    ordered = train + validation
    features = materialize(ordered, normalizer)
    train_rows = tuple(range(len(train)))
    validation_rows = tuple(range(len(train), len(ordered)))

    pair_config = config.pair_config()
    train_pairs = generate_pairs(train, pair_config, split=config.train_split)
    validation_pairs = generate_pairs(
        validation, pair_config, split=config.validation_split
    )

    if config.max_pairs is not None and len(train_pairs) > config.max_pairs:
        before = len(train_pairs)
        train_pairs.pairs = train_pairs.pairs[: config.max_pairs]
        train_pairs.pairs_per_episode = {}
        for pair in train_pairs.pairs:
            train_pairs.pairs_per_episode[pair.episode_id] = (
                train_pairs.pairs_per_episode.get(pair.episode_id, 0) + 1
            )
        limits["max_pairs"] = {
            "limit": config.max_pairs,
            "pairs_before": before,
            "pairs_after": len(train_pairs),
        }

    if not train_pairs.pairs:
        raise RuntimeError(
            "No within-episode training pair could be formed. Every training "
            "episode's candidates are tied at or within "
            f"{config.minimum_score_difference} of each other, so there is no "
            "preference to learn."
        )

    fingerprint = dataset_fingerprint(manifest_path, label_export)

    return PreparedDataset(
        schema=loaded.schema,
        normalizer=normalizer,
        features=features,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        loaded=loaded,
        fingerprint=fingerprint,
        label_set=labels,
        limits_applied=limits,
    )


def _infer_native_dimension(records: Sequence[Any]) -> int:
    """Read the embedding width from the artifacts rather than assuming 384."""
    from slotify_rank.datasets.schema import NATIVE_EMBEDDING_DIMENSION

    observed = {
        record.audio_embedding_dimension
        for record in records
        if record.audio_embedding_dimension
    } | {
        record.text_embedding_dimension
        for record in records
        if record.text_embedding_dimension
    }
    if not observed:
        return NATIVE_EMBEDDING_DIMENSION
    if len(observed) > 1:
        raise ValueError(
            f"The feature manifest mixes embedding widths {sorted(observed)}. A "
            "single model input layout cannot describe both."
        )
    return int(next(iter(observed)))


def _split_hash(examples: Sequence[TrainingExample]) -> str:
    """Hash of the exact candidate set a normalizer was fitted on."""
    digest = hashlib.sha256()
    for candidate_id in sorted(example.candidate_id for example in examples):
        digest.update(candidate_id.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:32]
