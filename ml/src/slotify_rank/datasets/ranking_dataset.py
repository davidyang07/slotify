"""PyTorch views over the eligible examples: candidates, episodes and pairs.

Three views, because training and evaluation want different shapes:

``CandidateDataset``
    One example per item. Used for the auxiliary acceptability head (where each
    candidate must count exactly once) and for scoring at validation time.

``EpisodeGroups``
    All eligible candidates of one episode, kept together. Ranking metrics are
    only meaningful within an episode, so this is what validation iterates.

``PairDataset``
    One within-episode preference pair per item. This is what the ranking loss
    consumes.

Tensors are materialized once, up front, rather than per ``__getitem__``. The
corpus this trains on is thousands of candidates of a few thousand floats each
-- tens of megabytes -- so a lazy loader would buy nothing and would reintroduce
per-item file I/O on Windows, where it is slowest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from slotify_rank.datasets.normalizer import FeatureNormalizer
from slotify_rank.datasets.schema import TrainingExample

__all__ = [
    "MaterializedFeatures",
    "CandidateDataset",
    "EpisodeGroups",
    "PairDataset",
    "materialize",
]


@dataclass(frozen=True)
class MaterializedFeatures:
    """Every eligible example as one contiguous set of tensors.

    Row order is the order of ``examples`` and is stable, so a row index is a
    usable candidate handle throughout training.
    """

    candidate_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    splits: tuple[str, ...]
    handcrafted: torch.Tensor
    handcrafted_missing_mask: torch.Tensor
    audio: torch.Tensor
    audio_available: torch.Tensor
    text: torch.Tensor
    text_available: torch.Tensor
    quality_score: torch.Tensor
    is_acceptable: torch.Tensor

    def __len__(self) -> int:
        return len(self.candidate_ids)

    @property
    def index_of(self) -> dict[str, int]:
        return {candidate_id: index for index, candidate_id in enumerate(self.candidate_ids)}

    def features_at(self, indices: torch.Tensor | Sequence[int]) -> dict[str, torch.Tensor]:
        """Gather the model-input tensors for a set of rows."""
        selector = torch.as_tensor(indices, dtype=torch.long)
        return {
            "handcrafted": self.handcrafted[selector],
            "handcrafted_missing_mask": self.handcrafted_missing_mask[selector],
            "audio": self.audio[selector],
            "audio_available": self.audio_available[selector],
            "text": self.text[selector],
            "text_available": self.text_available[selector],
        }

    def subset(self, indices: Sequence[int]) -> "MaterializedFeatures":
        selector = torch.as_tensor(list(indices), dtype=torch.long)
        return MaterializedFeatures(
            candidate_ids=tuple(self.candidate_ids[i] for i in indices),
            episode_ids=tuple(self.episode_ids[i] for i in indices),
            splits=tuple(self.splits[i] for i in indices),
            handcrafted=self.handcrafted[selector],
            handcrafted_missing_mask=self.handcrafted_missing_mask[selector],
            audio=self.audio[selector],
            audio_available=self.audio_available[selector],
            text=self.text[selector],
            text_available=self.text_available[selector],
            quality_score=self.quality_score[selector],
            is_acceptable=self.is_acceptable[selector],
        )


def materialize(
    examples: Sequence[TrainingExample], normalizer: FeatureNormalizer
) -> MaterializedFeatures:
    """Apply the fitted normalizer and stack everything into tensors."""
    if not examples:
        raise ValueError("Cannot materialize zero examples")

    raw = np.stack([example.handcrafted for example in examples])
    masks = np.stack([example.handcrafted_missing_mask for example in examples])
    normalized = normalizer.transform(raw, masks)
    if not np.isfinite(normalized).all():
        raise ValueError(
            "Normalization produced NaN or infinity. The fitted statistics do not "
            "describe these features -- refit rather than training through it."
        )

    return MaterializedFeatures(
        candidate_ids=tuple(example.candidate_id for example in examples),
        episode_ids=tuple(example.episode_id for example in examples),
        splits=tuple(example.split for example in examples),
        handcrafted=torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)),
        handcrafted_missing_mask=torch.from_numpy(
            np.ascontiguousarray(masks, dtype=np.float32)
        ),
        audio=torch.from_numpy(
            np.ascontiguousarray(
                np.stack([example.audio for example in examples]), dtype=np.float32
            )
        ),
        audio_available=torch.tensor(
            [float(example.audio_available) for example in examples], dtype=torch.float32
        ),
        text=torch.from_numpy(
            np.ascontiguousarray(
                np.stack([example.text for example in examples]), dtype=np.float32
            )
        ),
        text_available=torch.tensor(
            [float(example.text_available) for example in examples], dtype=torch.float32
        ),
        quality_score=torch.tensor(
            [float(example.quality_score) for example in examples], dtype=torch.float32
        ),
        is_acceptable=torch.tensor(
            [float(example.is_acceptable) for example in examples], dtype=torch.float32
        ),
    )


class CandidateDataset(Dataset):
    """One candidate per item, over a chosen subset of rows."""

    def __init__(self, features: MaterializedFeatures, indices: Sequence[int] | None = None):
        self.features = features
        self.indices = list(range(len(features))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.indices[item]
        payload: dict[str, Any] = {"row": row}
        for key, tensor in self.features.features_at([row]).items():
            payload[key] = tensor[0]
        payload["quality_score"] = self.features.quality_score[row]
        payload["is_acceptable"] = self.features.is_acceptable[row]
        return payload


@dataclass(frozen=True)
class EpisodeGroup:
    episode_id: str
    rows: tuple[int, ...]


class EpisodeGroups:
    """Row indices grouped by episode, in deterministic order.

    Episode order and within-episode row order both follow the materialized row
    order, which follows the manifest's canonical sort. Nothing here depends on
    dict iteration order or on a set.
    """

    def __init__(self, features: MaterializedFeatures, indices: Sequence[int] | None = None):
        self.features = features
        rows = list(range(len(features))) if indices is None else list(indices)
        grouped: dict[str, list[int]] = {}
        for row in rows:
            grouped.setdefault(features.episode_ids[row], []).append(row)
        self.groups: tuple[EpisodeGroup, ...] = tuple(
            EpisodeGroup(episode_id=episode_id, rows=tuple(grouped[episode_id]))
            for episode_id in sorted(grouped)
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[EpisodeGroup]:
        return iter(self.groups)

    @property
    def candidate_count(self) -> int:
        return sum(len(group.rows) for group in self.groups)

    def relevance(self, group: EpisodeGroup) -> dict[str, float]:
        return {
            self.features.candidate_ids[row]: float(self.features.quality_score[row])
            for row in group.rows
        }


class PairDataset(Dataset):
    """One within-episode preference pair per item.

    Items carry *row indices*, not tensors: the collate function gathers from
    the shared materialized block, so a candidate appearing in twenty pairs is
    stored once rather than twenty times.

    Slots are ``left``/``right`` with a ``target`` of +1 or -1, matching what the
    margin ranking loss consumes, because pair generation may have shuffled
    which slot holds the preferred candidate.
    """

    def __init__(self, pairs: Sequence[Mapping[str, Any]], index_of: Mapping[str, int]):
        self.rows_left: list[int] = []
        self.rows_right: list[int] = []
        self.targets: list[int] = []
        self.weights: list[float] = []
        for pair in pairs:
            left = index_of.get(str(pair["left_candidate_id"]))
            right = index_of.get(str(pair["right_candidate_id"]))
            if left is None or right is None:
                raise KeyError(
                    f"Pair references a candidate that is not in this split: "
                    f"{pair['left_candidate_id']!r} / {pair['right_candidate_id']!r}. "
                    "Pairs and examples must be built from the same eligible set."
                )
            target = int(pair["target"])
            if target not in (1, -1):
                raise ValueError(f"Pair target must be +1 or -1, got {target}")
            self.rows_left.append(left)
            self.rows_right.append(right)
            self.targets.append(target)
            self.weights.append(float(pair.get("pair_weight", 1.0)))

    def __len__(self) -> int:
        return len(self.rows_left)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return {
            "left_row": self.rows_left[item],
            "right_row": self.rows_right[item],
            "target": self.targets[item],
            "pair_weight": self.weights[item],
        }
