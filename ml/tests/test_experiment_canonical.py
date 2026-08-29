"""The canonical experiment definition and its resolved manifest.

The definition exists so the protocol is fixed before the test split is read.
These tests are about the ways that promise can be broken: a config that
contradicts the split on disk, a degraded split, a weak label source smuggled
into the allowed list, a label count below the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.data.paths import DataPaths
from slotify_rank.data.splits import SplitConfig, compute_splits, write_split_manifest
from slotify_rank.experiment.canonical import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    ExperimentConfigError,
    load_experiment_config,
    resolve_manifest,
    write_manifest,
)
from tests.dataset_fixtures import make_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "ml" / "configs" / "experiment_resume_v1.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


_VALID = f"""
experiment_version: test-experiment-v1
schema_version: {EXPERIMENT_CONFIG_SCHEMA_VERSION}
corpus:
  plan: ml/configs/corpus_resume_v1.yaml
  source_registries: [ml/configs/sources_real_v1.yaml]
  candidate_generation_config: ml/configs/dataset_v1.yaml
split:
  config: ml/configs/splits_v3.yaml
  version: v3
  group_by: series
  seed: 42
  ratios: {{train: 0.70, validation: 0.15, test: 0.15}}
labels:
  queue_config: ml/configs/labelling_queue_resume_v1.yaml
  queue_artifact: data/labels/queue_resume_v1.json
  snapshot_version: resume-v1
  rubric: docs/labelling-guide.md
  minimum_human_labels: 2400
  allowed_label_sources: [human]
model:
  headline_variant: gated
  ablations: [handcrafted, text_only, audio_only, concat, gated]
  config_dir: ml/configs/models
  training_config: ml/configs/training_v1.yaml
  seeds: [42, 43, 44]
  seed_selection: median_validation_ndcg_at_3
  checkpoint_selection_metric: validation_ndcg_at_3
baselines:
  canonical: heuristic_offline_v1
  classical: classical_handcrafted_v1
evaluation:
  primary_metric: ndcg_at_3
  cutoffs: [1, 3, 5]
  relevance_threshold: 4.0
  gain_offset: 1.0
  bootstrap_resamples: 2000
  bootstrap_confidence: 0.95
  bootstrap_seed: 1
  require_independent_metric_crosscheck: true
claim:
  statement: "test"
  minimum_human_labels: 2400
  minimum_relative_improvement_percent: 18.0
"""


# --------------------------------------------------------------------------
# The committed definition
# --------------------------------------------------------------------------


def test_the_committed_experiment_definition_loads():
    config = load_experiment_config(COMMITTED)
    assert config.experiment_version == "experiment-resume-v1"
    assert config.canonical_baseline == "heuristic_offline_v1"
    assert config.minimum_human_labels == 2400
    assert config.minimum_relative_improvement_percent == 18.0
    assert config.model["headline_variant"] in config.model["ablations"]


def test_the_committed_definition_allows_only_human_labels():
    """A weak-label run is a different experiment and must not borrow this one."""
    config = load_experiment_config(COMMITTED)
    assert list(config.labels["allowed_label_sources"]) == ["human"]


def test_the_committed_definition_has_a_stable_digest():
    """Two loads of the same bytes hash the same, so a manifest pins a config."""
    assert load_experiment_config(COMMITTED).digest() == load_experiment_config(
        COMMITTED
    ).digest()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_a_weak_label_source_is_refused(tmp_path: Path):
    body = _VALID.replace(
        "allowed_label_sources: [human]",
        "allowed_label_sources: [human, weak_heuristic]",
    )
    with pytest.raises(ExperimentConfigError, match="only be \\['human'\\]"):
        load_experiment_config(_write(tmp_path, body))


def test_a_headline_variant_outside_the_ablations_is_refused(tmp_path: Path):
    body = _VALID.replace("headline_variant: gated", "headline_variant: mystery")
    with pytest.raises(ExperimentConfigError, match="not among the declared ablations"):
        load_experiment_config(_write(tmp_path, body))


def test_ratios_that_do_not_sum_to_one_are_refused(tmp_path: Path):
    body = _VALID.replace(
        "{train: 0.70, validation: 0.15, test: 0.15}",
        "{train: 0.80, validation: 0.15, test: 0.15}",
    )
    with pytest.raises(ExperimentConfigError, match="must sum to 1.0"):
        load_experiment_config(_write(tmp_path, body))


def test_an_unknown_schema_version_is_refused(tmp_path: Path):
    body = _VALID.replace(EXPERIMENT_CONFIG_SCHEMA_VERSION, "experiment-config-v9")
    with pytest.raises(ExperimentConfigError, match="schema_version"):
        load_experiment_config(_write(tmp_path, body))


# --------------------------------------------------------------------------
# Manifest resolution
# --------------------------------------------------------------------------


def _paths_with_split(tmp_path: Path, group_by: str = "series", seed: int = 42):
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episodes = [
        make_episode(
            title=f"Episode {index}",
            series_id=f"series-{index}",
            sha_seed=chr(ord("a") + index),
            duration_ms=600_000,
        )
        for index in range(8)
    ]
    manifest = compute_splits(
        episodes, SplitConfig(version="v3", group_by=group_by, seed=seed)
    )
    write_split_manifest(paths.split_manifest("v3"), manifest)
    # A feature manifest has to exist for the experiment to resolve at all. A
    # bare header with no records is enough: nothing here reads the features.
    paths.features_manifest.write_text(
        json.dumps(
            {
                "record_type": "header",
                "schema_version": "feature-record-schema-v1.0.0",
                "feature_spec_version": "featurespec-v1.1.0",
                "feature_pipeline_version": "featurepipeline-v1.0.0",
                "handcrafted_feature_count": 110,
                "handcrafted_feature_names": [],
                "record_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def test_a_resolved_manifest_pins_hashes_and_records_the_split(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = _paths_with_split(tmp_path)
    manifest = resolve_manifest(config, tmp_path, paths, human_label_count=2400)

    assert manifest.split_summary["group_by"] == "series"
    assert manifest.split_summary["degraded"] is False
    assert manifest.artifact_hashes["split_manifest"]
    assert manifest.artifact_hashes["experiment_config"] is None or isinstance(
        manifest.artifact_hashes["experiment_config"], str
    )
    assert manifest.label_summary["human_label_count"] == 2400
    groups = manifest.split_summary["groups_by_split"]
    assert set(groups) == {"train", "validation", "test"}
    # The property the whole design exists for.
    assert not set(groups["test"]) & set(groups["train"])
    assert not set(groups["test"]) & set(groups["validation"])


def test_a_split_that_contradicts_the_config_blocks_the_experiment(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = _paths_with_split(tmp_path, group_by="episode")
    manifest = resolve_manifest(config, tmp_path, paths, human_label_count=2400)
    assert not manifest.ready
    assert any("group_by" in reason for reason in manifest.blocking_reasons)


def test_a_different_split_seed_blocks_the_experiment(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = _paths_with_split(tmp_path, seed=7)
    manifest = resolve_manifest(config, tmp_path, paths, human_label_count=2400)
    assert not manifest.ready
    assert any("seed" in reason for reason in manifest.blocking_reasons)


def test_too_few_labels_blocks_the_experiment(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = _paths_with_split(tmp_path)
    manifest = resolve_manifest(config, tmp_path, paths, human_label_count=17)
    assert not manifest.ready
    assert any("17 human-labelled" in reason for reason in manifest.blocking_reasons)


def test_a_degraded_split_blocks_the_experiment(tmp_path: Path):
    """Everything in 'development' cannot support a held-out claim."""
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    paths.mkdirs()
    episodes = [
        make_episode(title=f"E{i}", series_id=f"s-{i}", sha_seed=chr(ord("a") + i))
        for i in range(3)
    ]
    manifest_v3 = compute_splits(episodes, SplitConfig(version="v3"))
    assert manifest_v3.degraded
    write_split_manifest(paths.split_manifest("v3"), manifest_v3)
    resolved = resolve_manifest(config, tmp_path, paths, human_label_count=2400)
    assert not resolved.ready
    assert any("degraded" in reason for reason in resolved.blocking_reasons)


def test_a_written_manifest_round_trips(tmp_path: Path):
    config = load_experiment_config(_write(tmp_path, _VALID))
    paths = _paths_with_split(tmp_path)
    manifest = resolve_manifest(config, tmp_path, paths, human_label_count=2400)
    destination = tmp_path / "manifest.json"
    write_manifest(destination, manifest)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["experiment_version"] == "test-experiment-v1"
    assert payload["config_digest"] == manifest.config_digest
    assert payload["ready"] is manifest.ready
