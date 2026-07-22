"""Shared helpers for the Phase 4 training tests.

Corpora are cached per configuration and shared across tests. Building one
writes ~20 embedding arrays and their sidecars, each fsync'd and atomically
renamed, which on this OneDrive-backed checkout costs about 20 seconds -- so
rebuilding it for every test dominated the suite's runtime. A generated corpus
is read-only once written, and every test still gets its own run directory, so
sharing changes nothing a test can observe.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from slotify_rank.data.paths import DataPaths
from slotify_rank.datasets.labels import read_label_export
from slotify_rank.datasets.loader import EligibilityConfig, LoadedDataset, build_examples
from slotify_rank.datasets.synthetic import SyntheticCorpusConfig, build_synthetic_corpus
from slotify_rank.features.assemble import read_feature_manifest

__all__ = ["make_corpus", "load_corpus", "paths_for", "shared_corpus"]

#: config-overrides -> (DataPaths, SyntheticCorpus), built at most once.
_CORPUS_CACHE: dict[tuple, tuple[DataPaths, Any]] = {}


def shared_corpus(**overrides: Any) -> tuple[DataPaths, Any]:
    """A read-only synthetic corpus, built once per distinct configuration."""
    key = tuple(sorted(overrides.items()))
    cached = _CORPUS_CACHE.get(key)
    if cached is None:
        root = Path(tempfile.mkdtemp(prefix="slotify-corpus-"))
        cached = make_corpus(root, **overrides)
        _CORPUS_CACHE[key] = cached
    return cached


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
