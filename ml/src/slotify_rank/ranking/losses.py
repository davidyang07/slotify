"""The pairwise ranking objective and the auxiliary acceptability objective.

Sign convention, stated once and tested directly
------------------------------------------------

``target = +1`` means **the left candidate should score higher than the right
one**. The loss is

    ``max(0, -target * (left_score - right_score) + margin)``

which is exactly :class:`torch.nn.MarginRankingLoss`. So with ``target = +1`` the
loss falls to zero once ``left_score >= right_score + margin``, and rises as the
ordering reverses. Getting this backwards trains a model that is confidently
wrong while every loss curve looks healthy, which is why
``tests/test_ranking_losses.py`` asserts the direction rather than trusting the
formula.

The margin
----------

A margin of 0 is satisfied by an infinitesimal score difference, so the model
never learns a *confident* ordering and its scores collapse toward each other.
The default of 0.2 asks for a visible separation without demanding a
scale the scores have no reason to adopt.

Not double-counting the auxiliary loss
--------------------------------------

A pair batch names candidates twice over, and a strong candidate appears in many
pairs. Computing the acceptability loss over the batch as-drawn would weight
each candidate by how many pairs it happens to appear in -- which correlates
with its label, so the auxiliary head would be trained on a distorted class
distribution. :func:`combined_loss` therefore takes the *unique* candidates of
the batch, which the collate function has already isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from slotify_rank.datasets.schema import TrainingExample

__all__ = [
    "LossConfig",
    "LossComponents",
    "margin_ranking_loss",
    "pairwise_accuracy",
    "acceptability_loss",
    "combined_loss",
    "class_weight_from_training",
]

DEFAULT_MARGIN = 0.2
DEFAULT_AUXILIARY_WEIGHT = 0.25


@dataclass(frozen=True)
class LossConfig:
    margin: float = DEFAULT_MARGIN
    auxiliary_weight: float = DEFAULT_AUXILIARY_WEIGHT
    auxiliary_enabled: bool = True
    #: Balance the acceptability head with a positive-class weight derived from
    #: the training split. Off by default because on a balanced split it only
    #: adds variance.
    use_class_weights: bool = False

    def __post_init__(self) -> None:
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.auxiliary_weight < 0:
            raise ValueError(
                f"auxiliary_weight must be non-negative, got {self.auxiliary_weight}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin": self.margin,
            "auxiliary_weight": self.auxiliary_weight,
            "auxiliary_enabled": self.auxiliary_enabled,
            "use_class_weights": self.use_class_weights,
        }


@dataclass
class LossComponents:
    """The pieces of the objective, reported separately on purpose.

    A single blended number cannot tell you whether the ranking head stopped
    improving or the auxiliary head started dominating.
    """

    total: torch.Tensor
    ranking: torch.Tensor
    auxiliary: torch.Tensor | None
    pair_count: int
    unique_candidate_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_loss": float(self.total.detach()),
            "ranking_loss": float(self.ranking.detach()),
            "auxiliary_loss": (
                float(self.auxiliary.detach()) if self.auxiliary is not None else None
            ),
            "pair_count": self.pair_count,
            "unique_candidate_count": self.unique_candidate_count,
        }


def margin_ranking_loss(
    left_scores: torch.Tensor,
    right_scores: torch.Tensor,
    target: torch.Tensor,
    margin: float = DEFAULT_MARGIN,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean margin ranking loss. ``target`` is +1 or -1 per pair."""
    if left_scores.shape != right_scores.shape or left_scores.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: left {tuple(left_scores.shape)}, right "
            f"{tuple(right_scores.shape)}, target {tuple(target.shape)}"
        )
    per_pair = functional.margin_ranking_loss(
        left_scores, right_scores, target, margin=margin, reduction="none"
    )
    if weights is None:
        return per_pair.mean()
    if weights.shape != per_pair.shape:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match the "
            f"{tuple(per_pair.shape)} pairs"
        )
    total_weight = weights.sum()
    if float(total_weight) <= 0:
        raise ValueError(
            "Total pair weight is zero; a batch that contributes nothing should "
            "not have been assembled"
        )
    # Weighted *mean*, not sum: otherwise the loss scale moves with the batch
    # size and the configured learning rate stops meaning anything.
    return (per_pair * weights).sum() / total_weight


def pairwise_accuracy(
    left_scores: torch.Tensor, right_scores: torch.Tensor, target: torch.Tensor
) -> float:
    """Share of pairs ordered correctly. An exact tie scores half credit."""
    if left_scores.numel() == 0:
        return float("nan")
    difference = (left_scores - right_scores) * target
    correct = (difference > 0).float().sum() + 0.5 * (difference == 0).float().sum()
    return float(correct / left_scores.numel())


def acceptability_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with logits over unique candidates."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} disagree"
        )
    return functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )


def combined_loss(
    left_scores: torch.Tensor,
    right_scores: torch.Tensor,
    target: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    unique_logits: torch.Tensor | None = None,
    unique_targets: torch.Tensor | None = None,
    config: LossConfig | None = None,
    positive_weight: torch.Tensor | None = None,
) -> LossComponents:
    """``ranking_loss + auxiliary_weight * acceptability_loss``.

    ``unique_logits``/``unique_targets`` must be the batch's *unique*
    candidates, not its pair slots -- see the module docstring.
    """
    config = config or LossConfig()
    ranking = margin_ranking_loss(
        left_scores, right_scores, target, margin=config.margin, weights=pair_weights
    )

    auxiliary: torch.Tensor | None = None
    unique_count = 0
    if config.auxiliary_enabled and unique_logits is not None and unique_targets is not None:
        auxiliary = acceptability_loss(unique_logits, unique_targets, positive_weight)
        unique_count = int(unique_logits.numel())

    total = ranking if auxiliary is None else ranking + config.auxiliary_weight * auxiliary
    return LossComponents(
        total=total,
        ranking=ranking,
        auxiliary=auxiliary,
        pair_count=int(left_scores.numel()),
        unique_candidate_count=unique_count,
    )


def class_weight_from_training(
    examples: Sequence[TrainingExample], expected_split: str = "train"
) -> dict[str, Any]:
    """Positive-class weight for the auxiliary head, from the train split only.

    Deriving it from validation or test rows would leak the held-out label
    distribution into training, so the split is checked rather than assumed.
    """
    if not examples:
        raise ValueError("Cannot derive class weights from zero examples")
    offenders = sorted({e.split for e in examples if e.split != expected_split})
    if offenders:
        raise ValueError(
            f"Refusing to derive class weights from split(s) {offenders}; only "
            f"{expected_split!r} may inform them."
        )
    positives = sum(1 for example in examples if example.is_acceptable)
    negatives = len(examples) - positives
    if positives == 0 or negatives == 0:
        # A single-class split cannot define a ratio. Weighting is disabled
        # rather than substituted with an arbitrary constant.
        weight = 1.0
        degenerate = True
    else:
        weight = negatives / positives
        degenerate = False
    return {
        "positive_weight": float(weight),
        "positive_count": positives,
        "negative_count": negatives,
        "degenerate": degenerate,
        "fitted_on": expected_split,
    }
