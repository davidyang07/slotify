"""Shared helpers for the Phase 4 training tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slotify_rank.data.paths import DataPaths
from slotify_rank.datasets.labels import read_label_export
from slotify_rank.datasets.loader import EligibilityConfig, LoadedDataset, build_examples
from slotify_rank.datasets.synthetic import SyntheticCorpusConfig, build_synthetic_corpus
from slotify_rank.features.assemble import read_feature_manifest

__all__ = ["make_corpus", "load_corpus", "paths_for"]


def paths_for(tmp_path: Path) -> DataPaths:
    """A DataPaths rooted inside a test's tmp dir, never the real corpus."""
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    return paths


def make_corpus(tmp_path: Path, **overrides: Any):
    paths = paths_for(tmp_path)
    corpus = build_synthetic_corpus(paths, SyntheticCorpusConfig(**overrides))
    return paths, corpus


def load_corpus(
    tmp_path: Path,
    eligibility: EligibilityConfig | None = None,
    **overrides: Any,
) -> tuple[DataPaths, Any, LoadedDataset]:
    paths, corpus = make_corpus(tmp_path, **overrides)
    header, records = read_feature_manifest(paths.features_manifest)
    labels = read_label_export(corpus.label_export)
    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        config=eligibility or EligibilityConfig(),
    )
    return paths, corpus, loaded
