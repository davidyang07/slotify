"""Weak, heuristic-derived labels and the guard rails around them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.datasets.labels import (
    DEFAULT_ALLOWED_LABEL_SOURCES,
    aggregate_labels,
    read_label_export,
)
from slotify_rank.labelling.weak import (
    WEAK_ANNOTATOR_ID,
    WEAK_LABEL_SOURCE,
    build_weak_label_rows,
    grade_within_episode,
    write_weak_label_export,
)
from tests.dataset_fixtures import make_candidate


def candidates_with_scores(scores, episode_id: str = "ep-1"):
    return [
        make_candidate(
            episode_id=episode_id,
            timestamp_ms=(index + 1) * 10_000,
            heuristic_score=score,
        )
        for index, score in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_grades_span_the_rubric_within_an_episode() -> None:
    grades = grade_within_episode([0.1, 0.2, 0.3, 0.4, 0.5])
    assert grades == [1, 2, 3, 4, 5]


def test_ties_receive_the_same_grade() -> None:
    """Two candidates the baseline could not separate must not be separated
    here either -- otherwise the weak target invents a preference."""
    grades = grade_within_episode([0.5, 0.5, 0.9])
    assert grades[0] == grades[1]
    assert grades[2] > grades[0]


def test_a_single_candidate_expresses_no_preference() -> None:
    assert grade_within_episode([0.9]) == [3]


def test_no_candidates_produce_no_grades() -> None:
    assert grade_within_episode([]) == []


def test_a_flat_episode_produces_a_flat_grade() -> None:
    assert len(set(grade_within_episode([0.6] * 4))) == 1


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def test_every_row_is_stamped_weak() -> None:
    rows, summary = build_weak_label_rows(candidates_with_scores([0.1, 0.5, 0.9]))
    assert len(rows) == 3
    assert {row["label_source"] for row in rows} == {WEAK_LABEL_SOURCE}
    assert {row["annotator_id"] for row in rows} == {WEAK_ANNOTATOR_ID}
    assert summary.candidate_count == 3


def test_synthetic_candidates_never_receive_a_weak_label() -> None:
    real = candidates_with_scores([0.4, 0.8])
    synthetic = [
        make_candidate(
            episode_id="ep-1",
            timestamp_ms=99_000,
            heuristic_score=0.5,
            is_synthetic=True,
            eligible_for_labelling=False,
            eligible_for_evaluation=False,
        )
    ]
    rows, _ = build_weak_label_rows([*real, *synthetic])
    assert len(rows) == 2


def test_the_sidecar_records_zero_human_labels(tmp_path: Path) -> None:
    rows, summary = build_weak_label_rows(candidates_with_scores([0.2, 0.7]))
    destination = tmp_path / "weak_labels_v1.jsonl"
    write_weak_label_export(destination, rows, summary)

    metadata = json.loads(
        destination.with_suffix(destination.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["label_source"] == WEAK_LABEL_SOURCE
    assert metadata["human_labelled_candidate_count"] == 0
    assert metadata["human_annotator_count"] == 0
    assert "not from human judgement" in metadata["warning"]


# ---------------------------------------------------------------------------
# The loader refuses weak labels unless asked by name
# ---------------------------------------------------------------------------


def test_weak_labels_are_refused_by_default(tmp_path: Path) -> None:
    rows, summary = build_weak_label_rows(candidates_with_scores([0.2, 0.7]))
    destination = tmp_path / "weak_labels_v1.jsonl"
    write_weak_label_export(destination, rows, summary)

    with pytest.raises(ValueError, match="label_source"):
        read_label_export(destination)


def test_weak_labels_load_when_named_and_the_set_says_so(tmp_path: Path) -> None:
    rows, summary = build_weak_label_rows(candidates_with_scores([0.2, 0.7]))
    destination = tmp_path / "weak_labels_v1.jsonl"
    write_weak_label_export(destination, rows, summary)

    label_set = read_label_export(
        destination, allowed_label_sources=("human", WEAK_LABEL_SOURCE)
    )
    assert len(label_set) == 2
    assert label_set.label_source == WEAK_LABEL_SOURCE
    assert label_set.to_dict()["label_source"] == WEAK_LABEL_SOURCE


def test_the_default_allowlist_is_human_only() -> None:
    assert DEFAULT_ALLOWED_LABEL_SOURCES == ("human",)


def test_a_mixed_export_reports_both_sources() -> None:
    label_set = aggregate_labels(
        [
            {
                "candidate_id": "ep-1:000010000",
                "episode_id": "ep-1",
                "annotator_id": "a1",
                "quality_score": 4,
                "is_acceptable": True,
                "rubric_version": "r1",
                "label_source": "human",
            },
            {
                "candidate_id": "ep-1:000020000",
                "episode_id": "ep-1",
                "annotator_id": WEAK_ANNOTATOR_ID,
                "quality_score": 2,
                "is_acceptable": False,
                "rubric_version": "r1",
                "label_source": WEAK_LABEL_SOURCE,
            },
        ],
        acceptable_threshold=4.0,
        allowed_label_sources=("human", WEAK_LABEL_SOURCE),
    )
    # Joined, never rounded to "human": a summary reader must be able to see
    # that some of this supervision was not a person.
    assert label_set.label_source == f"human+{WEAK_LABEL_SOURCE}"


def test_an_empty_allowlist_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        aggregate_labels([], acceptable_threshold=4.0, allowed_label_sources=())
