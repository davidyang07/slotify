"""The training matrix and the rule for choosing which run gets quoted.

The rule is the point: the reported run is the **median** seed by validation
NDCG@3. Training three seeds and reporting the best is seed cherry-picking with
extra steps -- it reports the upper tail of a distribution as its centre, while
staying entirely within the letter of "we used a held-out test set". These tests
pin that behaviour so it cannot drift back to `max`.
"""

from __future__ import annotations

import json
from pathlib import Path

from slotify_rank.experiment.matrix import (
    RunOutcome,
    plan_runs,
    read_outcome,
    select_reported_seed,
    summarise_matrix,
    write_matrix_summary,
)


def _outcome(variant: str, seed: int, ndcg: float | None, **overrides) -> RunOutcome:
    payload = {
        "variant": variant,
        "seed": seed,
        "run_dir": f"artifacts/training/{variant}-seed{seed}",
        "exit_code": 0,
        "label_source": "human",
        "validation_ndcg_at_3": ndcg,
        "parameter_count": 489477,
        "run_id": f"{variant}-{seed}",
    }
    payload.update(overrides)
    return RunOutcome(**payload)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_the_plan_is_deterministic_and_covers_every_cell(tmp_path: Path):
    runs = plan_runs(["gated", "handcrafted"], [44, 42, 43], tmp_path)
    assert [run.name for run in runs] == [
        "gated-seed42",
        "gated-seed43",
        "gated-seed44",
        "handcrafted-seed42",
        "handcrafted-seed43",
        "handcrafted-seed44",
    ]
    assert plan_runs(["gated", "handcrafted"], [44, 42, 43], tmp_path) == runs


# --------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------


def test_the_median_seed_is_reported_not_the_best():
    outcomes = [
        _outcome("gated", 42, 0.61),
        _outcome("gated", 43, 0.88),  # the flattering one
        _outcome("gated", 44, 0.70),
    ]
    reported = select_reported_seed(outcomes)
    assert reported.seed == 44
    assert reported.validation_ndcg_at_3 == 0.70


def test_an_even_number_of_seeds_takes_the_lower_central_run():
    outcomes = [
        _outcome("gated", 42, 0.60),
        _outcome("gated", 43, 0.70),
        _outcome("gated", 44, 0.80),
        _outcome("gated", 45, 0.90),
    ]
    assert select_reported_seed(outcomes).validation_ndcg_at_3 == 0.70


def test_ties_break_on_the_lower_seed():
    outcomes = [
        _outcome("gated", 44, 0.70),
        _outcome("gated", 42, 0.70),
        _outcome("gated", 43, 0.70),
    ]
    assert select_reported_seed(outcomes).seed == 43


def test_a_run_with_no_validation_score_is_not_selectable():
    outcomes = [_outcome("gated", 42, None), _outcome("gated", 43, 0.5)]
    assert select_reported_seed(outcomes).seed == 43
    assert select_reported_seed([_outcome("gated", 42, None)]) is None


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def test_the_summary_reports_the_spread_per_variant():
    outcomes = [
        _outcome("gated", 42, 0.60),
        _outcome("gated", 43, 0.80),
        _outcome("gated", 44, 0.70),
        _outcome("handcrafted", 42, 0.50),
        _outcome("handcrafted", 43, 0.52),
        _outcome("handcrafted", 44, 0.51),
    ]
    summary = summarise_matrix("exp-v1", "gated", [42, 43, 44], outcomes)
    assert summary.ok is True
    by_variant = summary.by_variant()
    assert by_variant["gated"]["validation_ndcg_at_3_median"] == 0.70
    assert by_variant["gated"]["validation_ndcg_at_3_min"] == 0.60
    assert by_variant["gated"]["validation_ndcg_at_3_max"] == 0.80
    assert by_variant["handcrafted"]["seed_count"] == 3
    # The ablation comparison a "multimodal" claim needs.
    assert (
        by_variant["gated"]["validation_ndcg_at_3_median"]
        > by_variant["handcrafted"]["validation_ndcg_at_3_median"]
    )


def test_a_weakly_trained_run_blocks_the_matrix():
    outcomes = [
        _outcome("gated", 42, 0.7, label_source="weak_heuristic"),
        _outcome("gated", 43, 0.7),
    ]
    summary = summarise_matrix("exp-v1", "gated", [42, 43], outcomes)
    assert summary.ok is False
    assert any("weak_heuristic" in reason for reason in summary.blocking_reasons)


def test_a_failed_run_blocks_the_matrix():
    outcomes = [_outcome("gated", 42, None, exit_code=1), _outcome("gated", 43, 0.7)]
    summary = summarise_matrix("exp-v1", "gated", [42, 43], outcomes)
    assert summary.ok is False
    assert any("failed" in reason for reason in summary.blocking_reasons)


def test_a_missing_headline_variant_blocks_the_matrix():
    outcomes = [_outcome("handcrafted", 42, 0.7)]
    summary = summarise_matrix("exp-v1", "gated", [42], outcomes)
    assert summary.ok is False
    assert any("headline variant" in reason for reason in summary.blocking_reasons)


def test_the_summary_serialises_with_its_selection_rule_stated(tmp_path: Path):
    outcomes = [_outcome("gated", 42, 0.6), _outcome("gated", 43, 0.7)]
    summary = summarise_matrix("exp-v1", "gated", [42, 43], outcomes)
    destination = tmp_path / "matrix.json"
    write_matrix_summary(destination, summary)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["seed_selection"] == "median_validation_ndcg_at_3"
    assert "not the best" in payload["seed_selection_note"]
    assert payload["reported_run"]["seed"] == 42
    assert len(payload["runs"]) == 2


# --------------------------------------------------------------------------
# Reading a run back
# --------------------------------------------------------------------------


def test_an_outcome_is_read_from_the_runs_own_summary(tmp_path: Path):
    from slotify_rank.experiment.matrix import PlannedRun

    run_dir = tmp_path / "gated-seed42"
    run_dir.mkdir()
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "gated-abc",
                "label_source": "human",
                "model_parameter_count": 489477,
                "best_validation_ndcg_at_3": 0.6421,
                "best_validation_pairwise_accuracy": 0.81,
            }
        ),
        encoding="utf-8",
    )
    outcome = read_outcome(PlannedRun("gated", 42, run_dir), 0)
    assert outcome.ok is True
    assert outcome.validation_ndcg_at_3 == 0.6421
    assert outcome.label_source == "human"


def test_a_run_that_wrote_nothing_is_recorded_as_such(tmp_path: Path):
    from slotify_rank.experiment.matrix import PlannedRun

    outcome = read_outcome(PlannedRun("gated", 42, tmp_path / "absent"), 0)
    assert outcome.ok is False
    assert "no training_summary.json" in outcome.error
