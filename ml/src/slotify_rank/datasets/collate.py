"""Batch assembly for the pair and candidate views.

The pair collate is where the auxiliary head's double-counting bug is prevented.
A pair batch names candidates by row, and a candidate that is preferred in six
pairs appears six times in that batch. Running the acceptability loss over the
batch as-drawn would weight that candidate six times as heavily as one that
appears once -- silently, and in proportion to how often the annotators gave it
a distinctive score. :func:`collate_pairs` therefore also returns the *unique*
rows of the batch, and the trainer computes the auxiliary loss over those.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from slotify_rank.datasets.ranking_dataset import MaterializedFeatures

__all__ = ["PairBatch", "collate_pairs", "collate_candidates", "make_pair_collate"]


class PairBatch(dict):
    """A pair batch. A plain dict subclass so it moves to a device generically."""

    def to(self, device: torch.device) -> "PairBatch":
        moved = PairBatch()
        for key, value in self.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved


def collate_pairs(
    items: Sequence[Mapping[str, Any]], features: MaterializedFeatures
) -> PairBatch:
    """Gather both sides of each pair plus the batch's unique candidates."""
    preferred_rows = torch.tensor(
        [int(item["preferred_row"]) for item in items], dtype=torch.long
    )
    nonpreferred_rows = torch.tensor(
        [int(item["nonpreferred_row"]) for item in items], dtype=torch.long
    )
    weights = torch.tensor(
        [float(item["pair_weight"]) for item in items], dtype=torch.float32
    )

    # sorted=True keeps the unique set deterministic, so the auxiliary loss is
    # reproducible for a fixed batch rather than depending on kernel ordering.
    unique_rows = torch.unique(
        torch.cat([preferred_rows, nonpreferred_rows]), sorted=True
    )

    batch = PairBatch()
    for side, rows in (("preferred", preferred_rows), ("nonpreferred", nonpreferred_rows)):
        for key, tensor in features.features_at(rows).items():
            batch[f"{side}_{key}"] = tensor
    for key, tensor in features.features_at(unique_rows).items():
        batch[f"unique_{key}"] = tensor

    batch["pair_weight"] = weights
    batch["preferred_row"] = preferred_rows
    batch["nonpreferred_row"] = nonpreferred_rows
    batch["unique_row"] = unique_rows
    batch["unique_is_acceptable"] = features.is_acceptable[unique_rows]
    batch["unique_quality_score"] = features.quality_score[unique_rows]
    batch["preferred_quality_score"] = features.quality_score[preferred_rows]
    batch["nonpreferred_quality_score"] = features.quality_score[nonpreferred_rows]
    return batch


def make_pair_collate(features: MaterializedFeatures):
    """Bind a collate function to one materialized block, for a ``DataLoader``."""

    def _collate(items: Sequence[Mapping[str, Any]]) -> PairBatch:
        return collate_pairs(items, features)

    return _collate


def collate_candidates(
    rows: Sequence[int], features: MaterializedFeatures
) -> dict[str, torch.Tensor]:
    """Model inputs plus targets for a set of candidate rows."""
    selector = torch.as_tensor(list(rows), dtype=torch.long)
    batch = dict(features.features_at(selector))
    batch["is_acceptable"] = features.is_acceptable[selector]
    batch["quality_score"] = features.quality_score[selector]
    batch["row"] = selector
    return batch
