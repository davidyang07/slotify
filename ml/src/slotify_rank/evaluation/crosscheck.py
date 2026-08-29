"""Independent verification of this project's NDCG against scikit-learn's.

The headline resume number is a ratio of two NDCG@3 values. Both come out of
:mod:`slotify_rank.evaluation.metrics`, so a bug in that one function would move
the numerator and the denominator together and could easily produce a plausible,
wrong improvement that every test in the repository agrees with -- because every
test would be checking the same implementation against itself.

So the metric is checked against a second implementation nobody here wrote:
:func:`sklearn.metrics.ndcg_score`. The two agree only if the discount, the
truncation at k and the ideal ordering all match, which is exactly the part that
is easy to get subtly wrong.

**The gain function is ours, deliberately.** scikit-learn's ``ndcg_score`` treats
the values it is given as gains directly, with no exponential transform. This
project's gain is ``2 ** max(0, relevance - 1) - 1`` (see ``metrics._gain``), so
a rubric score of 1 -- "disruptive, inside speech" -- contributes nothing rather
than counting as a third of a perfect break. Feeding scikit-learn the transformed
gains rather than the raw scores keeps the transform under test where it belongs
(``test_metrics.py``) and puts the *ranking* arithmetic under an independent
implementation here. Cross-checking the transform as well would require
reimplementing it in the checker, which would just be a third copy of the same
possible mistake.

Ties: the predicted order is passed as strictly decreasing pseudo-scores rather
than the model's real scores, so scikit-learn ranks the list in exactly the order
this project ranked it. Handing it the raw scores would let its tie-breaking
differ from ours and produce a disagreement that is about tie order, not about
either implementation being wrong.

This module is optional: scikit-learn is in the ``sklearn`` extra, and every
caller degrades to ``available=False`` with a reason rather than failing when it
is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from slotify_rank.evaluation.metrics import (
    DEFAULT_GAIN_OFFSET,
    EpisodeJudgements,
    EpisodePrediction,
    _gain,
    ndcg_at_k,
)

__all__ = [
    "CrossCheckResult",
    "sklearn_available",
    "sklearn_ndcg_at_k",
    "cross_check_ndcg",
    "DEFAULT_TOLERANCE",
]

#: Both implementations do the same float arithmetic in a different order, so
#: they agree to well within this. It is a floating-point tolerance, not a
#: "close enough" allowance for a real disagreement.
DEFAULT_TOLERANCE = 1e-9


def sklearn_available() -> bool:
    try:
        import sklearn.metrics  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class CrossCheckResult:
    """Whether the two implementations agree, episode by episode."""

    available: bool
    reason: str | None
    k: int
    tolerance: float
    compared_episode_count: int = 0
    skipped_episode_count: int = 0
    max_absolute_difference: float | None = None
    disagreements: tuple[Mapping[str, Any], ...] = ()
    sklearn_version: str | None = None

    @property
    def agrees(self) -> bool:
        """True only when a comparison actually ran and found no disagreement."""
        return self.available and not self.disagreements

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "k": self.k,
            "tolerance": self.tolerance,
            "sklearn_version": self.sklearn_version,
            "compared_episode_count": self.compared_episode_count,
            "skipped_episode_count": self.skipped_episode_count,
            "max_absolute_difference": self.max_absolute_difference,
            "agrees": self.agrees,
            "disagreements": [dict(entry) for entry in self.disagreements],
        }


def sklearn_ndcg_at_k(
    ranked_candidate_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
    gain_offset: float = DEFAULT_GAIN_OFFSET,
) -> float | None:
    """NDCG@k for one episode, computed by scikit-learn.

    ``None`` where this project's metric is also undefined (no gain available),
    or where the list is too short for ``ndcg_score``, which needs at least two
    documents to have anything to rank.
    """
    import numpy as np
    from sklearn.metrics import ndcg_score

    gains = [_gain(float(relevance[cid]), gain_offset) for cid in ranked_candidate_ids]
    if not gains or max(gains) <= 0.0:
        return None
    if len(gains) < 2:
        # ndcg_score refuses a single-document list. Our metric is defined there
        # (it is trivially 1.0), so this is a limit of the checker, not a
        # disagreement, and the episode is reported as skipped.
        return None
    # Strictly decreasing pseudo-scores reproduce the given order exactly.
    pseudo_scores = [float(len(gains) - index) for index in range(len(gains))]
    return float(
        ndcg_score(np.asarray([gains]), np.asarray([pseudo_scores]), k=k)
    )


def cross_check_ndcg(
    predictions: Sequence[EpisodePrediction],
    judgements: Sequence[EpisodeJudgements],
    k: int,
    gain_offset: float = DEFAULT_GAIN_OFFSET,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CrossCheckResult:
    """Compare this project's per-episode NDCG@k against scikit-learn's."""
    if not sklearn_available():
        return CrossCheckResult(
            available=False,
            reason=(
                "scikit-learn is not installed, so the independent NDCG "
                'cross-check did not run. Install it with pip install -e ".[sklearn]".'
            ),
            k=k,
            tolerance=tolerance,
        )

    import sklearn

    by_episode = {j.episode_id: j for j in judgements}
    compared = 0
    skipped = 0
    worst: float | None = None
    disagreements: list[dict[str, Any]] = []

    for prediction in predictions:
        judgement = by_episode.get(prediction.episode_id)
        if judgement is None:
            skipped += 1
            continue
        ours = ndcg_at_k(
            prediction.ranked_candidate_ids, judgement.relevance, k, gain_offset
        )
        theirs = sklearn_ndcg_at_k(
            prediction.ranked_candidate_ids, judgement.relevance, k, gain_offset
        )
        if ours is None or theirs is None:
            skipped += 1
            continue
        difference = abs(ours - theirs)
        compared += 1
        worst = difference if worst is None else max(worst, difference)
        if difference > tolerance:
            disagreements.append(
                {
                    "episode_id": prediction.episode_id,
                    "slotify_ndcg": ours,
                    "sklearn_ndcg": theirs,
                    "absolute_difference": difference,
                }
            )

    return CrossCheckResult(
        available=True,
        reason=None,
        k=k,
        tolerance=tolerance,
        compared_episode_count=compared,
        skipped_episode_count=skipped,
        max_absolute_difference=worst,
        disagreements=tuple(disagreements),
        sklearn_version=sklearn.__version__,
    )
