"""scikit-learn's two load-bearing jobs, and the properties that justify them.

Both are optional at runtime, so every test skips cleanly when scikit-learn is
absent -- and both call sites are asserted to degrade to a recorded
"not available" rather than raising, because a missing optional dependency must
not take an evaluation down.

What is under test:

* **The independent NDCG cross-check.** The headline is a ratio of two NDCG
  values from one implementation, so a bug in it moves numerator and denominator
  together. The point of the cross-check is that it would catch such a bug, so
  there is a test that deliberately breaks the ranking and asserts the two
  implementations then disagree -- a checker that agrees with everything checks
  nothing.
* **The classical comparison point.** Grouped cross-validation must never put an
  episode on both sides of a fold, selection must never see the evaluation
  split, and two runs of the same configuration must select the same model.
"""

from __future__ import annotations

import numpy as np
import pytest

from slotify_rank.baselines.classical import (
    ClassicalBaselineConfig,
    build_matrix,
    fit_classical_baseline,
    sklearn_available,
)
from slotify_rank.datasets.schema import TrainingExample
from slotify_rank.evaluation.crosscheck import (
    CrossCheckResult,
    cross_check_ndcg,
    sklearn_ndcg_at_k,
)
from slotify_rank.evaluation.metrics import (
    EpisodeJudgements,
    EpisodePrediction,
    ndcg_at_k,
)

requires_sklearn = pytest.mark.skipif(
    not sklearn_available(), reason="scikit-learn is not installed"
)

#: The real config fits 300 boosting iterations per grid point per fold. These
#: tests are about the protocol -- grouping, determinism, leakage -- not about
#: fit quality, so they use a fraction of that and stay under a second.
def _config(**overrides) -> ClassicalBaselineConfig:
    settings = {
        "max_iterations": 20,
        "learning_rates": (0.1,),
        "max_leaf_nodes": (15, 31),
        "min_samples_leaf": (5,),
    }
    settings.update(overrides)
    return ClassicalBaselineConfig(**settings)


# --------------------------------------------------------------------------
# Independent metric verification
# --------------------------------------------------------------------------


@requires_sklearn
@pytest.mark.parametrize(
    "relevance_order",
    [
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [5.0, 1.0, 5.0, 1.0, 3.0],
        [1.0, 1.0, 1.0, 1.0, 5.0],
    ],
)
def test_sklearn_reproduces_this_projects_ndcg(relevance_order):
    ids = [f"c{i}" for i in range(len(relevance_order))]
    relevance = dict(zip(ids, relevance_order))
    ours = ndcg_at_k(ids, relevance, k=3)
    theirs = sklearn_ndcg_at_k(ids, relevance, k=3)
    assert ours is not None and theirs is not None
    assert ours == pytest.approx(theirs, abs=1e-9)


@requires_sklearn
def test_the_cross_check_reports_agreement_across_episodes():
    predictions = [
        EpisodePrediction("ep-a", ["a1", "a2", "a3", "a4"]),
        EpisodePrediction("ep-b", ["b3", "b1", "b2"]),
    ]
    judgements = [
        EpisodeJudgements("ep-a", {"a1": 5.0, "a2": 2.0, "a3": 4.0, "a4": 1.0}),
        EpisodeJudgements("ep-b", {"b1": 4.0, "b2": 1.0, "b3": 3.0}),
    ]
    result = cross_check_ndcg(predictions, judgements, k=3)
    assert result.available is True
    assert result.agrees is True
    assert result.compared_episode_count == 2
    assert result.max_absolute_difference == pytest.approx(0.0, abs=1e-9)


@requires_sklearn
def test_the_cross_check_would_catch_a_broken_ndcg(monkeypatch):
    """A checker that agrees with everything is not a check.

    The project's NDCG is monkeypatched to a plausible-but-wrong implementation
    -- no rank discount at all -- and the cross-check must notice.
    """
    import slotify_rank.evaluation.crosscheck as module

    def broken(ranked_ids, relevance, k=3, gain_offset=1.0):
        gains = [max(0.0, relevance[cid] - gain_offset) for cid in ranked_ids[:k]]
        best = sorted(
            (max(0.0, v - gain_offset) for v in relevance.values()), reverse=True
        )[:k]
        total = sum(best)
        return None if total == 0 else sum(gains) / total

    monkeypatch.setattr(module, "ndcg_at_k", broken)
    predictions = [EpisodePrediction("ep-a", ["a2", "a1", "a3"])]
    judgements = [EpisodeJudgements("ep-a", {"a1": 5.0, "a2": 1.0, "a3": 3.0})]
    result = cross_check_ndcg(predictions, judgements, k=3)
    assert result.agrees is False
    assert result.disagreements[0]["episode_id"] == "ep-a"


def test_the_cross_check_degrades_to_unavailable_rather_than_raising(monkeypatch):
    import slotify_rank.evaluation.crosscheck as module

    monkeypatch.setattr(module, "sklearn_available", lambda: False)
    result = cross_check_ndcg([], [], k=3)
    assert isinstance(result, CrossCheckResult)
    assert result.available is False
    assert result.agrees is False  # unavailable is never "agrees"
    assert "scikit-learn" in result.reason


@requires_sklearn
def test_an_episode_with_no_gain_is_skipped_by_both_implementations():
    ids = ["a", "b", "c"]
    relevance = {"a": 1.0, "b": 1.0, "c": 1.0}  # every gain is zero
    assert ndcg_at_k(ids, relevance, k=3) is None
    assert sklearn_ndcg_at_k(ids, relevance, k=3) is None


# --------------------------------------------------------------------------
# The classical comparison point
# --------------------------------------------------------------------------


def _example(
    candidate_id: str,
    episode_id: str,
    quality: float,
    seed: int,
    dimension: int = 12,
) -> TrainingExample:
    rng = np.random.default_rng(seed)
    handcrafted = rng.normal(size=dimension).astype(np.float32)
    # One informative column, so a fitted model can beat noise and the test is
    # not asserting that a regressor learns nothing.
    handcrafted[0] = np.float32(quality + rng.normal(scale=0.1))
    return TrainingExample(
        candidate_id=candidate_id,
        episode_id=episode_id,
        split="train",
        handcrafted=handcrafted,
        handcrafted_missing_mask=np.zeros(dimension, dtype=np.float32),
        audio=np.zeros(4, dtype=np.float32),
        audio_available=False,
        text=np.zeros(4, dtype=np.float32),
        text_available=False,
        quality_score=quality,
        is_acceptable=quality >= 3.0,
    )


def _corpus(episodes: int = 6, per_episode: int = 12, offset: int = 0):
    return [
        _example(
            f"ep{e}:{c:03d}",
            f"ep{e}",
            quality=1.0 + ((c + e) % 5),
            seed=offset + e * 100 + c,
        )
        for e in range(episodes)
        for c in range(per_episode)
    ]


def test_the_matrix_carries_the_missing_mask_alongside_the_values():
    """"Absent" and "measured zero" must not be the same input to the model."""
    examples = _corpus(episodes=2, per_episode=2)
    features, targets, groups = build_matrix(examples)
    assert features.shape == (4, 24)  # 12 values + 12 mask columns
    assert targets.shape == (4,)
    assert sorted(set(groups.tolist())) == ["ep0", "ep1"]


@requires_sklearn
def test_the_classical_baseline_fits_and_scores_the_evaluation_split():
    train = _corpus(episodes=6, per_episode=12)
    evaluation = _corpus(episodes=2, per_episode=8, offset=9000)
    result = fit_classical_baseline(train, evaluation, _config())
    assert result.available is True
    assert result.train_episode_count == 6
    assert result.feature_dimension == 24
    assert set(result.scores) == {e.candidate_id for e in evaluation}
    assert result.best_params in [entry["params"] for entry in result.cv_results]


@requires_sklearn
def test_selection_is_deterministic():
    train = _corpus()
    evaluation = _corpus(episodes=2, per_episode=6, offset=500)
    first = fit_classical_baseline(train, evaluation, _config(seed=7))
    second = fit_classical_baseline(train, evaluation, _config(seed=7))
    assert first.best_params == second.best_params
    assert first.scores == second.scores


@requires_sklearn
def test_cross_validation_never_puts_an_episode_on_both_sides_of_a_fold():
    """Grouped by episode; a random k-fold would leak within-episode structure."""
    from sklearn.model_selection import GroupKFold

    _, targets, groups = build_matrix(_corpus())
    splitter = GroupKFold(n_splits=3)
    for train_index, held_index in splitter.split(
        np.zeros((len(targets), 1)), targets, groups
    ):
        assert not set(groups[train_index]) & set(groups[held_index])


@requires_sklearn
def test_the_evaluation_split_is_never_used_for_selection():
    """Selection depends only on the training split.

    Two runs whose evaluation sets differ entirely must select the same
    hyperparameters -- otherwise the "held-out" set is participating in model
    development.
    """
    train = _corpus()
    first = fit_classical_baseline(
        train, _corpus(episodes=2, per_episode=6, offset=1), _config()
    )
    second = fit_classical_baseline(
        train,
        _corpus(episodes=3, per_episode=10, offset=77_000),
        _config(),
    )
    assert first.best_params == second.best_params
    assert first.cv_results == second.cv_results


@requires_sklearn
def test_a_single_episode_cannot_be_grouped_and_says_so():
    train = _corpus(episodes=1, per_episode=10)
    result = fit_classical_baseline(train, [], _config())
    assert result.available is False
    assert "at least two episodes" in result.reason


def test_the_classical_baseline_degrades_to_unavailable_rather_than_raising(
    monkeypatch,
):
    import slotify_rank.baselines.classical as module

    monkeypatch.setattr(module, "sklearn_available", lambda: False)
    result = fit_classical_baseline(_corpus(), [], _config())
    assert result.available is False
    assert "scikit-learn" in result.reason
    assert result.scores == {}
