"""The shared ranker interface and the building blocks every variant reuses.

One interface, five variants. The trainer, the checkpoint writer and the
validation loop all speak :class:`RankerOutput` and never ask which model they
hold -- so adding a sixth variant is a registry entry, not a change to the
training loop.

Every model emits a **ranking score** (a scalar, higher is a better breakpoint;
only its ordering within an episode is meaningful) and an **acceptability
logit** (pre-sigmoid, for the auxiliary binary head). The auxiliary output can
be switched off in configuration, in which case it is ``None`` -- the *field*
still exists, so downstream code stays uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from slotify_rank.datasets.schema import DatasetSchema
from slotify_rank.models.schema import ModelConfig

__all__ = [
    "RankerOutput",
    "BaseRanker",
    "ModalityProjection",
    "count_parameters",
]


@dataclass
class RankerOutput:
    """What every variant returns.

    ``gates`` is populated only by the gated variant and is kept for inspection:
    a fusion model whose gate values nobody can look at is a model whose
    "multimodal" claim cannot be checked.
    """

    ranking_score: torch.Tensor
    acceptability_logit: torch.Tensor | None = None
    gates: torch.Tensor | None = None
    modality_names: tuple[str, ...] = ()


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


class ModalityProjection(nn.Module):
    """Project one modality to a common width, zeroing it when unavailable.

    LayerNorm rather than BatchNorm throughout: batches here are small (pair
    batches of a few dozen) and, more importantly, a masked modality would
    contribute its zeros to a batch statistic, so BatchNorm would let one
    candidate's *availability* shift another candidate's activations.

    The multiplication by the availability flag happens after the non-linearity,
    so an unavailable modality contributes exactly zero rather than the bias
    term's image -- which is a small constant vector, and is precisely the thing
    a model would otherwise learn to read as "text was missing".
    """

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be positive, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, features: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got "
                f"{features.shape[-1]}. The dataset's column layout does not match "
                "the one this model was built for."
            )
        projected = self.block(features)
        if available is not None:
            projected = projected * available.reshape(-1, 1)
        return projected


class BaseRanker(nn.Module):
    """Common construction, validation and heads for every variant."""

    #: Set by each subclass; recorded in the checkpoint and checked on load.
    variant: str = ""

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__()
        if config.variant != self.variant:
            raise ValueError(
                f"{type(self).__name__} is variant {self.variant!r} but was given a "
                f"config for {config.variant!r}"
            )
        self.config = config
        self.schema = schema
        self.handcrafted_input_dimension = schema.handcrafted_dimension * (
            2 if config.include_missing_mask else 1
        )

    def build_trunk_and_heads(self, fused_dimension: int) -> None:
        """The shared post-fusion trunk plus the two output heads."""
        config = self.config
        self.trunk = nn.Sequential(
            nn.Linear(fused_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.ranking_head = nn.Linear(config.hidden_dimension, 1)
        self.acceptability_head = (
            nn.Linear(config.hidden_dimension, 1) if config.auxiliary_head else None
        )

    def heads(self, hidden: torch.Tensor, gates: torch.Tensor | None = None) -> RankerOutput:
        representation = self.trunk(hidden)
        ranking_score = self.ranking_head(representation).squeeze(-1)
        acceptability = (
            self.acceptability_head(representation).squeeze(-1)
            if self.acceptability_head is not None
            else None
        )
        return RankerOutput(
            ranking_score=ranking_score,
            acceptability_logit=acceptability,
            gates=gates,
            modality_names=self.modality_names,
        )

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ()

    def handcrafted_input(
        self, handcrafted: torch.Tensor, missing_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.include_missing_mask:
            return handcrafted
        return torch.cat([handcrafted, missing_mask], dim=-1)

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        raise NotImplementedError

    def parameter_count(self) -> int:
        return count_parameters(self)

    def describe(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "experiment_name": self.config.experiment_name,
            "trainable_parameters": self.parameter_count(),
            "modalities": list(self.modality_names),
            "auxiliary_head": self.config.auxiliary_head,
            "handcrafted_input_dimension": self.handcrafted_input_dimension,
            "config": self.config.to_dict(),
            "input_schema": self.schema.to_dict(),
        }
