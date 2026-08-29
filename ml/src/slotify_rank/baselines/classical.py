"""A classical, non-neural ranking baseline over the handcrafted features.

The question this exists to answer is the one a sceptical reader should ask
about any multimodal model: *would a good tabular model on the cheap features
have done just as well?* If a gradient-boosted tree over 110 handcrafted scalars
matches the neural ranker, then Whisper and MiniLM are not earning their
inference cost and the "multimodal" claim, while true, is not doing any work.

So this trains scikit-learn's :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
on exactly the same human labels, the same split and the same episodes as the
neural ranker, and is reported next to it. It is deliberately *not* the published
denominator -- that stays ``heuristic_offline_v1``, which needs no labels, no
training and no dependencies, so anyone can reproduce it.

Three things make this a fair comparison rather than a straw man:

* **The same supervision.** It regresses the pooled human ``quality_score``,
  which is the same target the neural model's auxiliary head sees, and it is
  scored with the same episode-level NDCG@3.
* **Real model selection, grouped by episode.** Hyperparameters are chosen by
  cross-validation *inside the training split* using
  :class:`~sklearn.model_selection.GroupKFold` with the episode as the group, so
  a fold never scores a model on an episode it was fitted on. Tuning it on a
  random k-fold would leak within-episode structure and flatter this baseline;
  tuning it on the validation split would spend the model-selection budget the
  neural ranker also needs.
* **The features it is actually given.** The handcrafted block *and* its missing
  mask, so "this value is absent" is available to it as a signal rather than
  being confused with a real zero -- the same information the neural model's
  handcrafted branch receives.

scikit-learn is an optional extra. Every entry point degrades to a recorded
"not available" rather than failing when it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from slotify_rank.datasets.schema import TrainingExample

__all__ = [
    "CLASSICAL_BASELINE_VERSION",
    "ClassicalBaselineConfig",
    "ClassicalBaselineResult",
    "sklearn_available",
    "build_matrix",
    "fit_classical_baseline",
]

#: Bumped when the feature construction or the search space changes, so a
#: reported classical number always names the recipe that produced it.
CLASSICAL_BASELINE_VERSION = "classical_handcrafted_v1"


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ClassicalBaselineConfig:
    """The search space and the protocol, recorded with every result."""

    seed: int = 42
    #: Folds for the grouped cross-validation inside the training split.
    cv_folds: int = 4
    #: Small on purpose. This is a comparison point, not a competitor being
    #: tuned to death, and a large grid over a few thousand rows would mostly be
    #: fitting the cross-validation noise.
    learning_rates: tuple[float, ...] = (0.05, 0.1)
    max_leaf_nodes: tuple[int, ...] = (15, 31)
    min_samples_leaf: tuple[int, ...] = (10, 20)
    max_iterations: int = 300
    l2_regularization: float = 1.0

    def grid(self) -> list[dict[str, Any]]:
        return [
            {
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": leaf,
            }
            for rate in self.learning_rates
            for leaves in self.max_leaf_nodes
            for leaf in self.min_samples_leaf
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_version": CLASSICAL_BASELINE_VERSION,
            "seed": self.seed,
            "cv_folds": self.cv_folds,
            "learning_rates": list(self.learning_rates),
            "max_leaf_nodes": list(self.max_leaf_nodes),
            "min_samples_leaf": list(self.min_samples_leaf),
            "max_iterations": self.max_iterations,
            "l2_regularization": self.l2_regularization,
        }


@dataclass
class ClassicalBaselineResult:
    """A fitted classical baseline, its selection trace and its scores."""

    available: bool
    reason: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    best_params: Mapping[str, Any] = field(default_factory=dict)
    cv_results: list[dict[str, Any]] = field(default_factory=list)
    train_example_count: int = 0
    train_episode_count: int = 0
    feature_dimension: int = 0
    sklearn_version: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "config": dict(self.config),
            "best_params": dict(self.best_params),
            "cv_results": list(self.cv_results),
            "train_example_count": self.train_example_count,
            "train_episode_count": self.train_episode_count,
            "feature_dimension": self.feature_dimension,
            "sklearn_version": self.sklearn_version,
            "scored_candidate_count": len(self.scores),
        }


def build_matrix(examples: Sequence[TrainingExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, groups)`` for a set of examples.

    ``X`` is the handcrafted block concatenated with its missing mask, so an
    absent measurement is distinguishable from a measured zero. ``groups`` is the
    episode id, which is what the grouped cross-validation splits on.
    """
    if not examples:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=object),
        )
    features = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(example.handcrafted, dtype=np.float64),
                    np.asarray(example.handcrafted_missing_mask, dtype=np.float64),
                ]
            )
            for example in examples
        ]
    )
    targets = np.asarray(
        [float(example.quality_score) for example in examples], dtype=np.float64
    )
    groups = np.asarray([example.episode_id for example in examples], dtype=object)
    return features, targets, groups


def fit_classical_baseline(
    train_examples: Sequence[TrainingExample],
    score_examples: Sequence[TrainingExample],
    config: ClassicalBaselineConfig | None = None,
) -> ClassicalBaselineResult:
    """Select hyperparameters on the train split, fit, and score ``score_examples``.

    ``score_examples`` is normally the held-out test split. It is never used for
    selection -- the grid is chosen entirely by grouped cross-validation inside
    ``train_examples``, so this baseline sees the test split exactly once, at the
    same moment the neural ranker does.
    """
    config = config or ClassicalBaselineConfig()
    if not sklearn_available():
        return ClassicalBaselineResult(
            available=False,
            reason=(
                "scikit-learn is not installed, so the classical baseline did not "
                'run. Install it with pip install -e ".[sklearn]".'
            ),
            config=config.to_dict(),
        )

    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    features, targets, groups = build_matrix(train_examples)
    if features.size == 0:
        return ClassicalBaselineResult(
            available=False,
            reason="no training example was supplied",
            config=config.to_dict(),
        )

    distinct_groups = len(set(groups.tolist()))
    if distinct_groups < 2:
        return ClassicalBaselineResult(
            available=False,
            reason=(
                f"grouped cross-validation needs at least two episodes in the "
                f"training split, found {distinct_groups}"
            ),
            config=config.to_dict(),
        )

    folds = max(2, min(config.cv_folds, distinct_groups))
    splitter = GroupKFold(n_splits=folds)

    def make_model(params: Mapping[str, Any]) -> Any:
        return HistGradientBoostingRegressor(
            max_iter=config.max_iterations,
            l2_regularization=config.l2_regularization,
            early_stopping=False,
            random_state=config.seed,
            **params,
        )

    cv_results: list[dict[str, Any]] = []
    for params in config.grid():
        fold_scores: list[float] = []
        for train_index, held_index in splitter.split(features, targets, groups):
            model = make_model(params)
            model.fit(features[train_index], targets[train_index])
            predicted = model.predict(features[held_index])
            # Mean squared error, negated so "larger is better" throughout.
            fold_scores.append(
                -float(np.mean((predicted - targets[held_index]) ** 2))
            )
        cv_results.append(
            {
                "params": dict(params),
                "mean_negative_mse": float(np.mean(fold_scores)),
                "std_negative_mse": float(np.std(fold_scores)),
                "fold_scores": [float(value) for value in fold_scores],
            }
        )

    # Deterministic tie-break on the parameter dict's text so two runs with equal
    # cross-validation scores select the same model.
    cv_results.sort(
        key=lambda entry: (-entry["mean_negative_mse"], str(entry["params"]))
    )
    best = cv_results[0]

    final = make_model(best["params"])
    final.fit(features, targets)

    scores: dict[str, float] = {}
    if score_examples:
        score_features, _, _ = build_matrix(score_examples)
        predictions = final.predict(score_features)
        scores = {
            example.candidate_id: float(value)
            for example, value in zip(score_examples, predictions)
        }

    return ClassicalBaselineResult(
        available=True,
        reason=None,
        config=config.to_dict(),
        best_params=dict(best["params"]),
        cv_results=cv_results,
        train_example_count=len(train_examples),
        train_episode_count=distinct_groups,
        feature_dimension=int(features.shape[1]),
        sklearn_version=sklearn.__version__,
        scores=scores,
    )
