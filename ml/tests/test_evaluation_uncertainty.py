"""Uncertainty, cohort accounting and the two extra opinions on the result.

A point estimate of NDCG@3 over a few dozen episodes is not a result on its own,
and a metric verified only against itself is not verified. These cover the parts
of the comparison that exist to stop a plausible number being mistaken for a
solid one:

* the bootstrap interval, and that it resamples *series* -- the unit episodes
  are correlated within -- rather than episodes or candidates;
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


def test_without_a_series_mapping_the_unit_is_the_episode_and_says_so():
    per_episode_baseline = {f"ep-{i}": 0.5 for i in range(10)}
    per_episode_model = {f"ep-{i}": 0.7 for i in range(10)}
    result = bootstrap_interval(
        per_episode_baseline, per_episode_model, BootstrapConfig(resamples=200)
    )
    assert result["measured"] is True
    assert result["episode_count"] == 10
    assert result["config"]["unit"] == "episode"


def test_the_bootstrap_resamples_series_not_episodes():
    """Two shows, five episodes each, the model better on one and worse on the
    other. Ten "independent" episodes give a tight interval around no change;
    two series -- which is what the data actually holds -- cannot rule out a
    large effect either way, and the cluster interval must say so."""
    series = {f"a-{i}": "show-a" for i in range(5)} | {f"b-{i}": "show-b" for i in range(5)}
    baseline = {f"a-{i}": 0.5 for i in range(5)} | {f"b-{i}": 0.9 for i in range(5)}
    model = {f"a-{i}": 0.9 for i in range(5)} | {f"b-{i}": 0.5 for i in range(5)}
    config = BootstrapConfig(resamples=1000, seed=3)

    clustered = bootstrap_interval(baseline, model, config, series_by_episode=series)
    naive = bootstrap_interval(baseline, model, config)

    assert clustered["config"]["unit"] == "series"
    assert clustered["series_count"] == 2
    assert clustered["episode_count"] == 10
    assert clustered["distinct_resamples_possible"] == 3

    def width(result):
        block = result["relative_improvement_percent"]
        return block["high"] - block["low"]

    assert width(clustered) > width(naive)


def test_a_drawn_series_contributes_every_one_of_its_episodes():
    """One series of one episode scoring 0, one of three scoring 1. The only
    resamples are {A,A}, {A,B} and {B,B}, whose episode macro-averages are 0,
    3/4 and 1 -- an episode bootstrap could also produce 1/4 or 1/2."""
    series = {"a-0": "a", "b-0": "b", "b-1": "b", "b-2": "b"}
    model = {"a-0": 0.0, "b-0": 1.0, "b-1": 1.0, "b-2": 1.0}
    baseline = dict.fromkeys(model, 0.5)
    result = bootstrap_interval(
        baseline, model, BootstrapConfig(resamples=500, seed=1), series_by_episode=series
    )
    interval = result["model_ndcg_at_3"]
    assert interval["low"] in (0.0, 0.75, 1.0)
    assert interval["high"] in (0.0, 0.75, 1.0)
    assert interval["low"] < interval["high"]


def test_a_single_series_cannot_be_bootstrapped_and_says_so():
    one_show = {f"ep-{i}": "only-show" for i in range(4)}
    scores = {f"ep-{i}": 0.5 for i in range(4)}
    result = bootstrap_interval(
        scores, dict(scores), BootstrapConfig(), series_by_episode=one_show
    )
    assert result["measured"] is False
    assert "at least two" in result["reason"]
    assert "series" in result["reason"]


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
    is the bug this pairing exists to avoid. Holds for the series draw too.
    """
    scores = {f"ep-{i}": 0.2 + 0.07 * i for i in range(10)}
    series = {f"ep-{i}": f"show-{i % 3}" for i in range(10)}
    for mapping in (None, series):
        result = bootstrap_interval(
            scores, dict(scores), BootstrapConfig(resamples=400), series_by_episode=mapping
        )
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
            experiment_version="experiment-v1",
            experiment_config_digest="c" * 64,
            artifact_hashes={"model_checkpoint": "d" * 64, "label_export": None},
        ),
        evaluation_id="eval-prov",
    )
    payload = result.to_dict()["inputs"]
    assert payload["experiment_version"] == "experiment-v1"
    assert payload["artifact_hashes"]["model_checkpoint"] == "d" * 64
    # A missing artifact is recorded as missing, not omitted.
    assert payload["artifact_hashes"]["label_export"] is None


def test_a_single_series_test_partition_is_called_out():
    """Six episodes of one show is not six independent observations.

    They share hosts, room, mic chain and editing rhythm, so a macro-average
    over them measures that show as much as the model. The point estimate looks
    exactly as confident either way, so the caveat has to be explicit.
    """
    examples, baseline, model = three_episodes()
    one_show = {f"ep-{i}": "the-only-show" for i in range(3)}
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-one-series",
        series_by_episode=one_show,
    )
    assert any("1 series" in warning for warning in result.warnings)
    assert any("one show's quirks" in warning for warning in result.warnings)


def test_three_series_is_still_called_out():
    """The committed test split is 7 episodes from exactly 3 series. A caveat
    that fired only below 3 would stay silent on the one split that exists."""
    examples, baseline, model = three_episodes()
    three = {f"ep-{i}": f"series-{i}" for i in range(3)}
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-three-series",
        series_by_episode=three,
    )
    assert any("3 series" in warning for warning in result.warnings)


def test_a_broad_test_partition_raises_no_series_caveat():
    examples, baseline, model = _many_episodes(count=8)
    many = {f"ep-{i}": f"series-{i % 4}" for i in range(8)}
    result = compare(
        examples=examples,
        baseline_scores=baseline,
        model_scores=model,
        inputs=inputs(),
        evaluation_id="eval-many-series",
        series_by_episode=many,
    )
    assert not any("series (" in warning for warning in result.warnings)
    assert result.bootstrap["config"]["unit"] == "series"
    assert result.bootstrap["series_count"] == 4
