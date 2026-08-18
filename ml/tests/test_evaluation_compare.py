"""The held-out comparison, its arithmetic, and every reason it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.datasets.schema import TrainingExample
from slotify_rank.evaluation.compare import (
    CANONICAL_BASELINE,
    ComparisonInputs,
    compare,
    relative_improvement_percent,
    render_summary,
    write_comparison,
)


def example(
    candidate_id: str, episode_id: str, quality: float, width: int = 4
) -> TrainingExample:
    return TrainingExample(
        candidate_id=candidate_id,
        episode_id=episode_id,
        split="test",
        handcrafted=np.ones(width, dtype=np.float32),
        handcrafted_missing_mask=np.zeros(width, dtype=bool),
        audio=np.ones(width, dtype=np.float32),
        audio_available=True,
        text=np.ones(width, dtype=np.float32),
        text_available=True,
        quality_score=quality,
        is_acceptable=quality >= 4,
    )


def inputs(**overrides) -> ComparisonInputs:
    base = {
        "split": "test",
        "label_source": "human",
        "label_export": "data/labels/labels_v1.jsonl",
        "features_manifest": "data/manifests/features.jsonl",
        "split_manifest": "data/manifests/splits_v2.json",
        "split_version": "v2",
        "model_run_id": "gated-abc",
        "model_variant": "gated",
        "model_checkpoint": "artifacts/training/gated-abc/best_checkpoint.pt",
        "model_training_label_source": "human",
        "baseline_version": CANONICAL_BASELINE,
        "baseline_config_version": "heuristic-config-v1.0.0",
        "git_sha": "deadbeef",
        "dataset_version": "training-dataset-v1.0.0",
    }
    base.update(overrides)
    return ComparisonInputs(**base)


def three_episodes():
    """Three episodes where the model orders perfectly and the baseline does not."""
    examples = []
    baseline: dict[str, float] = {}
    model: dict[str, float] = {}
    for episode in range(3):
        for index, quality in enumerate([5.0, 4.0, 2.0, 1.0]):
            candidate_id = f"ep-{episode}:{index:09d}"
            examples.append(example(candidate_id, f"ep-{episode}", quality))
            model[candidate_id] = 10.0 - index          # perfect order
            baseline[candidate_id] = float(index)       # exactly reversed
    return examples, baseline, model


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def test_the_improvement_formula() -> None:
    assert relative_improvement_percent(0.72, 0.61) == pytest.approx(18.0327, abs=1e-4)
    assert relative_improvement_percent(0.61, 0.61) == 0.0
    assert relative_improvement_percent(0.5, 0.61) == pytest.approx(-18.0327, abs=1e-4)


def test_an_undefined_side_gives_an_undefined_improvement() -> None:
    assert relative_improvement_percent(None, 0.61) is None
    assert relative_improvement_percent(0.72, None) is None


def test_a_zero_baseline_does_not_produce_an_enormous_percentage() -> None:
    # Reporting "infinite improvement" over a baseline that scored nothing would
    # be the single most misleading number this module could emit.
    assert relative_improvement_percent(0.5, 0.0) is None


# ---------------------------------------------------------------------------
# Scoring both systems on the same set
# ---------------------------------------------------------------------------


def test_a_better_ordering_produces_a_positive_improvement() -> None:
    examples, baseline, model = three_episodes()
    result = compare(examples, baseline, model, inputs(), "eval-test")
    headline = result.headline()
    assert headline["model"]["ndcg_at_3"] > headline["baseline"]["ndcg_at_3"]
    assert headline["relative_improvement_percent"] > 0


def test_an_identical_ordering_produces_zero_improvement() -> None:
    examples, baseline, _ = three_episodes()
    result = compare(examples, baseline, baseline, inputs(), "eval-test")
    assert result.headline()["relative_improvement_percent"] == 0.0


def test_both_systems_must_score_every_candidate() -> None:
    examples, baseline, model = three_episodes()
    model.pop(examples[0].candidate_id)
    with pytest.raises(ValueError, match="same set"):
        compare(examples, baseline, model, inputs(), "eval-test")


def test_an_empty_evaluation_set_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="nothing"):
        compare([], {}, {}, inputs(), "eval-test")


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_a_clean_comparison_is_publishable() -> None:
    examples, baseline, model = three_episodes()
    result = compare(examples, baseline, model, inputs(), "eval-test")
    assert result.publishable, result.blocking_reasons


def test_weak_ground_truth_blocks_the_headline() -> None:
    examples, baseline, model = three_episodes()
    result = compare(
        examples, baseline, model, inputs(label_source="weak_heuristic"), "eval-test"
    )
    assert not result.publishable
    assert any("circular" in reason for reason in result.blocking_reasons)


def test_a_weakly_trained_model_blocks_the_headline() -> None:
    examples, baseline, model = three_episodes()
    result = compare(
        examples,
        baseline,
        model,
        inputs(model_training_label_source="weak_heuristic"),
        "eval-test",
    )
    assert not result.publishable


def test_evaluating_off_the_test_split_blocks_the_headline() -> None:
    examples, baseline, model = three_episodes()
    result = compare(examples, baseline, model, inputs(split="validation"), "eval-test")
    assert not result.publishable
    assert any("held-out test split" in reason for reason in result.blocking_reasons)


def test_a_non_canonical_baseline_blocks_the_headline() -> None:
    examples, baseline, model = three_episodes()
    result = compare(
        examples, baseline, model, inputs(baseline_version="something_else"), "eval-test"
    )
    assert not result.publishable


def test_training_on_an_evaluation_episode_is_leakage_and_blocks() -> None:
    examples, baseline, model = three_episodes()
    result = compare(
        examples,
        baseline,
        model,
        inputs(),
        "eval-test",
        training_episode_ids=["ep-1"],
    )
    assert not result.publishable
    assert any("leakage" in reason for reason in result.blocking_reasons)


def test_no_relevant_candidate_blocks_rather_than_scoring_zero() -> None:
    examples = [example(f"ep-0:{i:09d}", "ep-0", 1.0) for i in range(4)]
    scores = {e.candidate_id: float(i) for i, e in enumerate(examples)}
    result = compare(examples, scores, scores, inputs(), "eval-test")
    assert not result.publishable
    assert any("undefined" in reason for reason in result.blocking_reasons)


def test_a_thin_evaluation_set_warns() -> None:
    examples = [example(f"ep-0:{i:09d}", "ep-0", 5.0 - i) for i in range(4)]
    scores = {e.candidate_id: 4.0 - i for i, e in enumerate(examples)}
    result = compare(examples, scores, scores, inputs(), "eval-test")
    assert any("only 1 evaluation episode" in warning for warning in result.warnings)
    assert any("no seed-to-seed variance" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# The artifacts
# ---------------------------------------------------------------------------


def test_every_artifact_is_written(tmp_path: Path) -> None:
    examples, baseline, model = three_episodes()
    result = compare(examples, baseline, model, inputs(), "eval-test")
    written = write_comparison(tmp_path, result)
    assert set(written) == {
        "comparison.json",
        "metrics.json",
        "per_episode_metrics.json",
        "ranked_candidates.jsonl",
        "summary.md",
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "ranked_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == len(examples)


def test_the_summary_is_generated_from_the_same_payload() -> None:
    """No number in summary.md may be typed independently of the JSON."""
    examples, baseline, model = three_episodes()
    result = compare(examples, baseline, model, inputs(), "eval-test")
    payload = result.to_dict()
    summary = render_summary(payload)
    improvement = payload["headline"]["relative_improvement_percent"]
    assert f"{improvement:.2f} %" in summary
    assert f"{payload['headline']['baseline']['ndcg_at_3']:.4f}" in summary


def test_a_blocked_summary_says_so_before_showing_the_number(tmp_path: Path) -> None:
    examples, baseline, model = three_episodes()
    result = compare(
        examples, baseline, model, inputs(label_source="weak_heuristic"), "eval-test"
    )
    summary = render_summary(result.to_dict())
    assert "NOT PUBLISHABLE" in summary
    assert summary.index("NOT PUBLISHABLE") < summary.index("Relative improvement")
