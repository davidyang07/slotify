"""The Phase 5B readiness gate and the frozen label snapshot."""

from __future__ import annotations

import pytest

from slotify_rank.datasets.labels import AggregatedLabel
from slotify_rank.experiment.freeze import (
    FrozenLabelSnapshot,
    build_snapshot,
    read_snapshot,
    write_snapshot,
)
from slotify_rank.experiment.readiness import (
    ReadinessGate,
    compute_readiness,
    count_usable_pairs,
)
from slotify_rank.labelling.quality import check_label_quality
from tests.phase5_fixtures import build_corpus, ready_corpus


def _readiness(corpus, feature_status=None):
    quality = check_label_quality(corpus.raw_labels, corpus.candidates, corpus.episodes)
    return compute_readiness(
        corpus.aggregated,
        corpus.raw_labels,
        corpus.candidates,
        corpus.episodes,
        quality,
        feature_status_by_id=(
            corpus.feature_status if feature_status is None else feature_status
        ),
    )


def test_empty_store_blocks_with_all_reasons():
    corpus = build_corpus({"s-a": "train"}, candidates_per_episode=3, labelled=False)
    report = _readiness(corpus)
    assert report.ready is False
    reasons = " ".join(report.blocking_reasons)
    assert "human-labelled unique candidates 0 < 200" in reasons
    assert "test series" in reasons


def test_ready_corpus_passes_the_gate():
    corpus = ready_corpus()
    report = _readiness(corpus)
    assert report.ready is True, report.blocking_reasons
    m = report.metrics
    assert m["human_labelled_unique_candidates"] >= 200
    assert m["human_labelled_episodes"] >= 8
    assert m["human_labelled_series"] >= 6
    assert len(m["train_series"]) >= 4
    assert m["usable_pairs_train"] > 0
    assert m["usable_pairs_validation"] > 0
    assert m["test_relevant_candidate_count"] > 0


def test_incomplete_features_block_the_gate():
    corpus = ready_corpus()
    # Knock one labelled candidate's features to "failed".
    broken = dict(corpus.feature_status)
    first = next(iter(corpus.aggregated))
    broken[first] = "failed"
    report = _readiness(corpus, feature_status=broken)
    assert report.ready is False
    assert any("complete multimodal feature record" in r for r in report.blocking_reasons)


def test_integrity_error_blocks_the_gate():
    corpus = ready_corpus()
    # An orphan label creates an integrity error.
    from slotify_rank.labelling.database import LabelRecord

    corpus.raw_labels.append(
        LabelRecord(
            label_id=999999,
            candidate_id="ghost:000000001",
            episode_id="ghost",
            annotator_id="a",
            quality_score=4,
            is_acceptable=True,
            is_unusable=False,
            notes=None,
            rubric_version="rubric-v1.0.0",
            created_at="2026-07-23T00:00:00+00:00",
            updated_at="2026-07-23T00:00:00+00:00",
        )
    )
    report = _readiness(corpus)
    assert report.ready is False
    assert any("integrity" in r for r in report.blocking_reasons)


def test_count_usable_pairs():
    labels = {
        "ep1": [
            AggregatedLabel("c1", "ep1", 5.0, True, 1, ("a",), "r"),
            AggregatedLabel("c2", "ep1", 2.0, False, 1, ("a",), "r"),
            AggregatedLabel("c3", "ep1", 5.0, True, 1, ("a",), "r"),
        ],
        "ep2": [  # all equal -> no usable pair
            AggregatedLabel("c4", "ep2", 3.0, True, 1, ("a",), "r"),
            AggregatedLabel("c5", "ep2", 3.0, True, 1, ("a",), "r"),
        ],
    }
    total, without = count_usable_pairs(labels, min_difference=1.0)
    # ep1: (c1,c2) and (c3,c2) clear the gap; (c1,c3) do not.
    assert total == 2
    assert without == ["ep2"]


def test_gate_is_not_silently_weakened():
    # The shipped gate must match the specified thresholds.
    gate = ReadinessGate()
    assert gate.min_unique_candidates == 200
    assert gate.min_episodes == 8
    assert gate.min_series == 6
    assert gate.min_train_series == 4
    assert gate.min_validation_series == 1
    assert gate.min_test_series == 1


# -- freeze -----------------------------------------------------------------


def test_snapshot_build_and_immutability(tmp_path):
    corpus = build_corpus({"s-a": "train", "s-b": "test"}, candidates_per_episode=4)
    candidate_split = {c.candidate_id: c.dataset_split for c in corpus.candidates}
    snapshot = build_snapshot(
        corpus.aggregated,
        candidate_split,
        label_export_path=tmp_path / "labels_v1.jsonl",
        candidate_manifest_path=tmp_path / "candidates.jsonl",
        feature_manifest_path=None,
        split_manifest_path=None,
        excluded_labels=[],
        snapshot_version="v1",
    )
    assert snapshot.unique_candidate_count == len(corpus.aggregated)
    assert snapshot.rubric_version == "rubric-v1.0.0"

    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)
    loaded = read_snapshot(path)
    assert loaded.unique_candidate_count == snapshot.unique_candidate_count

    # A snapshot of the same version with different content is refused.
    corpus2 = build_corpus({"s-a": "train"}, candidates_per_episode=2)
    other = build_snapshot(
        corpus2.aggregated,
        {c.candidate_id: c.dataset_split for c in corpus2.candidates},
        label_export_path=tmp_path / "labels_v1.jsonl",
        candidate_manifest_path=tmp_path / "candidates.jsonl",
        feature_manifest_path=None,
        split_manifest_path=None,
        excluded_labels=[],
        snapshot_version="v1",
    )
    with pytest.raises(FileExistsError):
        write_snapshot(path, other)


def test_snapshot_read_rejects_wrong_schema(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema_version": "label-snapshot-v0.0.0"}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_snapshot(path)


def test_the_gate_can_be_raised_by_a_named_experiment(tmp_path):
    """The generic gate is a floor, not the benchmark experiment's bar.

    Passing at 200 labels and only revealing the real requirement two commands
    later would send someone away thinking they were done.
    """
    import dataclasses
    from pathlib import Path

    from slotify_rank.experiment.canonical import load_experiment_config
    from slotify_rank.experiment.readiness import ReadinessGate

    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "configs" / "experiment_v1.yaml"
    )
    raised = dataclasses.replace(
        ReadinessGate(), min_unique_candidates=config.minimum_human_labels
    )
    assert ReadinessGate().min_unique_candidates < raised.min_unique_candidates
    assert raised.min_unique_candidates == 2400
    # Everything else the generic gate requires is unchanged.
    assert raised.min_series == ReadinessGate().min_series
    assert raised.relevance_threshold == ReadinessGate().relevance_threshold
