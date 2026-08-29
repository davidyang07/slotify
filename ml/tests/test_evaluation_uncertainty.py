"""Uncertainty, cohort accounting and the two extra opinions on the result.

A point estimate of NDCG@3 over a few dozen episodes is not a result on its own,
and a metric verified only against itself is not verified. These cover the parts
of the comparison that exist to stop a plausible number being mistaken for a
solid one:

* the bootstrap interval, and that it resamples *episodes* rather than
  candidates;
* the cohort counts, so a reader can see how much data the number rests on;
* the independent metric cross-check, whose failure blocks publication;
* the classical comparison point, reported alongside and never as the
  denominator.
"""

from __future__ import annotations

import pytest

from slotify_rank.evaluation.compare import (
    BootstrapConfig,
    bootstrap_interval,
    compare,
    render_summary,
)
from slotify_rank.evaluation.crosscheck import sklearn_available
from tests.test_evaluation_compare import example, inputs, three_episodes

requires_sklearn = pytest.mark.skipif(
    not sklearn_available(), reason="scikit-learn is not installed"
)


def _many_episodes(count: int = 12):
    """Enough episodes that a bootstrap has something to resample."""
    examples = []
    baseline: dict[str, float] = {}
    model: dict[str, float] = {}
    for episode in range(count):
        for index, quality in enumerate([5.0, 4.0, 2.0, 1.0]):
            candidate_id = f"ep-{episode}:{index:09d}"
            examples.append(example(candidate_id, f"ep-{episode}", quality))
            model[candidate_id] = 10.0 - index
            # Half the episodes the baseline gets right, half it reverses, so the
            # per-episode scores actually vary and the interval is not degenerate.
            baseline[candidate_id] = (
                10.0 - index if episode % 2 == 0 else float(index)
            )
    return examples, baseline, model


# --------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------


def test_the_bootstrap_resamples_episodes_not_candidates():
    per_episode_baseline = {f"ep-{i}": 0.5 for i in range(10)}
    per_episode_model = {f"ep-{i}": 0.7 for i in range(10)}
    result = bootstrap_interval(
        per_episode_baseline, per_episode_model, BootstrapConfig(resamples=200)
    )
    assert result["measured"] is True
    assert result["episode_count"] == 10
    assert result["config"]["unit"] == "episode"


def test_a_single_episode_cannot_be_bootstrapped_and_says_so():
    result = bootstrap_interval({"ep-0": 0.5}, {"ep-0": 0.7}, BootstrapConfig())
    assert result["measured"] is False
    assert "at least two" in result["reason"]


def test_the_bootstrap_is_deterministic_for_a_seed():
    baseline = {f"ep-{i}": 0.3 + 0.05 * i for i in range(8)}
    model = {f"ep-{i}": 0.5 + 0.04 * i for i in range(8)}
    config = BootstrapConfig(resamples=300, seed=11)
    first = bootstrap_interval(baseline, model, config)
    second = bootstrap_interval(baseline, model, config)
    assert first == second


def test_the_two_systems_are_resampled_on_the_same_episode_draw():
    """The scores are paired observations of one episode.

    Identical per-episode scores must give an improvement interval of exactly
    zero. Resampling the two independently would produce a spread instead, which
    is the bug this pairing exists to avoid.
    """
    scores = {f"ep-{i}": 0.2 + 0.07 * i for i in range(10)}
    result = bootstrap_interval(scores, dict(scores), BootstrapConfig(resamples=400))
    interval = result["relative_improvement_percent"]
    assert interval["low"] == pytest.approx(0.0)
    assert interval["high"] == pytest.approx(0.0)


def test_an_interval_spanning_zero_is_reported_as_a_caveat():
    examples, baseline, model = _many_episodes()
    # Make the model no better than the baseline overall.
    result = compare(
        examples=examples,
        baseline_scores=model,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-flat",
        bootstrap=BootstrapConfig(resamples=300),
    )
    interval = result.bootstrap["relative_improvement_percent"]
    assert interval["low"] <= 0.0 <= interval["high"]
    assert any("spans zero" in warning for warning in result.warnings)


def test_the_comparison_reports_an_interval_alongside_the_point_estimate():
    examples, baseline, model = _many_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-1",
        bootstrap=BootstrapConfig(resamples=300),
    )
    payload = result.to_dict()
    assert payload["bootstrap"]["measured"] is True
    markdown = render_summary(payload)
    assert "## Uncertainty" in markdown
    assert "Percentile bootstrap" in markdown


# --------------------------------------------------------------------------
# Cohort accounting
# --------------------------------------------------------------------------


def test_the_comparison_reports_how_much_data_the_number_rests_on():
    examples, baseline, model = three_episodes()
    train = [example(f"tr:{i:09d}", f"tr-ep-{i % 4}", 3.0) for i in range(40)]
    validation = [example(f"va:{i:09d}", f"va-ep-{i % 2}", 3.0) for i in range(10)]
    series = {f"ep-{i}": f"series-{i}" for i in range(3)}
    series.update({f"tr-ep-{i}": f"train-series-{i % 2}" for i in range(4)})
    series.update({f"va-ep-{i}": "validation-series" for i in range(2)})

    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-cohort",
        series_by_episode=series,
        cohort_examples={
            "train": train,
            "validation": validation,
            "test": examples,
        },
    )
    cohort = result.cohort
    assert cohort.candidates_by_split == {"test": 12, "train": 40, "validation": 10}
    assert cohort.episodes_by_split == {"test": 3, "train": 4, "validation": 2}
    assert cohort.series_by_split == {"test": 3, "train": 2, "validation": 1}
    assert cohort.labelled_candidate_count == 62
    assert cohort.evaluated_series == ("series-0", "series-1", "series-2")

    markdown = render_summary(result.to_dict())
    assert "## What the number rests on" in markdown
    assert "| test | 12 | 3 | 3 |" in markdown


# --------------------------------------------------------------------------
# The independent metric cross-check
# --------------------------------------------------------------------------


@requires_sklearn
def test_a_clean_comparison_records_an_agreeing_cross_check():
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-cc",
    )
    assert result.metric_crosscheck["available"] is True
    assert result.metric_crosscheck["agrees"] is True
    assert result.publishable is True
    assert "Independent metric verification" in render_summary(result.to_dict())


def test_a_missing_cross_check_blocks_by_default(monkeypatch):
    """An unverified metric is not a publishable one."""
    import slotify_rank.evaluation.crosscheck as module

    monkeypatch.setattr(module, "sklearn_available", lambda: False)
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-nocc",
    )
    assert result.publishable is False
    assert any("not independently verified" in r for r in result.blocking_reasons)


def test_a_missing_cross_check_can_be_downgraded_to_a_warning(monkeypatch):
    import slotify_rank.evaluation.crosscheck as module

    monkeypatch.setattr(module, "sklearn_available", lambda: False)
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-nocc-2",
        require_metric_crosscheck=False,
    )
    assert result.publishable is True
    assert any("not independently verified" in w for w in result.warnings)


@requires_sklearn
def test_a_disagreeing_cross_check_blocks_publication(monkeypatch):
    import slotify_rank.evaluation.crosscheck as module

    monkeypatch.setattr(
        module, "ndcg_at_k", lambda ranked, relevance, k=3, gain_offset=1.0: 0.123
    )
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-bad",
    )
    assert result.publishable is False
    assert any("disagrees with scikit-learn" in r for r in result.blocking_reasons)


# --------------------------------------------------------------------------
# The classical comparison point
# --------------------------------------------------------------------------


def test_the_classical_block_is_recorded_and_rendered():
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-classical",
        classical_baseline={
            "available": True,
            "ndcg_at_3": 0.71,
            "best_params": {"learning_rate": 0.1},
            "train_example_count": 1680,
            "train_episode_count": 40,
            "feature_dimension": 220,
            "sklearn_version": "1.9.0",
        },
    )
    markdown = render_summary(result.to_dict())
    assert "Classical comparison point" in markdown
    assert "0.7100" in markdown
    # It is never the denominator: the headline still names the canonical one.
    assert result.headline()["baseline"]["name"] == inputs().baseline_version


def test_an_absent_classical_baseline_is_a_warning_not_a_silence():
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-noclassical",
        classical_baseline={"available": False, "reason": "scikit-learn absent"},
    )
    assert any("classical scikit-learn baseline did not run" in w for w in result.warnings)
    assert "Did not run" in render_summary(result.to_dict())


# --------------------------------------------------------------------------
# Provenance carried on the comparison
# --------------------------------------------------------------------------


def test_artifact_hashes_and_the_experiment_are_recorded():
    examples, baseline, model = three_episodes()
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(
            experiment_version="experiment-resume-v1",
            experiment_config_digest="c" * 64,
            artifact_hashes={"model_checkpoint": "d" * 64, "label_export": None},
        ),
        evaluation_id="eval-prov",
    )
    payload = result.to_dict()["inputs"]
    assert payload["experiment_version"] == "experiment-resume-v1"
    assert payload["artifact_hashes"]["model_checkpoint"] == "d" * 64
    # A missing artifact is recorded as missing, not omitted.
    assert payload["artifact_hashes"]["label_export"] is None
