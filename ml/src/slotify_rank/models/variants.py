"""The five ranker variants.

They form a deliberate ladder, so Phase 5's ablation can attribute any gain to a
specific modality rather than to "the neural one is better":

1. ``handcrafted`` -- the learned non-embedding baseline. Same information the
   Phase 1 heuristic has, fitted rather than hand-tuned.
2. ``text_only`` -- transcript representation alone.
3. ``audio_only`` -- speech representation alone (optionally plus the acoustic
   scalars; the two are named as separate experiments).
4. ``concat`` -- all three, projected separately and concatenated.
5. ``gated`` -- all three, combined by learned per-candidate gates.

4 and 5 differ in exactly one thing: how the modalities are combined. That is
what makes "gating helps" a checkable claim rather than a stylistic preference.
"""

from __future__ import annotations

import torch
from torch import nn

from slotify_rank.datasets.schema import DatasetSchema
from slotify_rank.models.base import BaseRanker, ModalityProjection, RankerOutput
from slotify_rank.models.schema import ModelConfig

__all__ = [
    "HandcraftedRanker",
    "TextOnlyRanker",
    "AudioOnlyRanker",
    "ConcatRanker",
    "GatedRanker",
]


class HandcraftedRanker(BaseRanker):
    """Small MLP over the normalized scalar features and their missing masks."""

    variant = "handcrafted"

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__(config, schema)
        self.projection = ModalityProjection(
            self.handcrafted_input_dimension,
            config.handcrafted_projection,
            config.dropout,
        )
        self.build_trunk_and_heads(config.handcrafted_projection)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ("handcrafted",)

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        hidden = self.projection(
            self.handcrafted_input(handcrafted, handcrafted_missing_mask)
        )
        return self.heads(hidden)


class TextOnlyRanker(BaseRanker):
    """MLP over the constructed transcript representation alone.

    The availability flag is appended to the projected state rather than only
    used to zero it: with text as the *only* modality, a candidate whose
    transcript is missing produces an all-zero projection, and without the flag
    the model could not distinguish that from a genuinely neutral context.
    """

    variant = "text_only"

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__(config, schema)
        self.projection = ModalityProjection(
            schema.text_dimension, config.text_projection, config.dropout
        )
        self.build_trunk_and_heads(config.text_projection + 1)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ("text",)

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        projected = self.projection(text, text_available)
        hidden = torch.cat([projected, text_available.reshape(-1, 1)], dim=-1)
        return self.heads(hidden)


class AudioOnlyRanker(BaseRanker):
    """The cached Whisper representation, optionally with the acoustic scalars.

    ``include_handcrafted_features`` selects between two named experiments --
    ``audio_embedding_only`` and ``audio_plus_acoustic`` -- because "audio-only"
    is ambiguous about whether hand-computed RMS and spectral descriptors count
    as audio. Both are reproducible; neither is the default silently.
    """

    variant = "audio_only"

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__(config, schema)
        self.audio_projection = ModalityProjection(
            schema.audio_dimension, config.audio_projection, config.dropout
        )
        fused = config.audio_projection + 1
        self.scalar_projection: ModalityProjection | None = None
        if config.include_handcrafted_features:
            self.scalar_projection = ModalityProjection(
                self.handcrafted_input_dimension,
                config.handcrafted_projection,
                config.dropout,
            )
            fused += config.handcrafted_projection
        self.build_trunk_and_heads(fused)

    @property
    def modality_names(self) -> tuple[str, ...]:
        if self.config.include_handcrafted_features:
            return ("audio", "handcrafted")
        return ("audio",)

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        parts = [
            self.audio_projection(audio, audio_available),
            audio_available.reshape(-1, 1),
        ]
        if self.scalar_projection is not None:
            parts.append(
                self.scalar_projection(
                    self.handcrafted_input(handcrafted, handcrafted_missing_mask)
                )
            )
        return self.heads(torch.cat(parts, dim=-1))


class ConcatRanker(BaseRanker):
    """Project each modality separately, then concatenate.

    Concatenation keeps every modality's contribution addressable by the fusion
    layer, which a sum would not. The availability flags are concatenated too:
    a zeroed block and a genuinely zero-valued block are otherwise identical to
    the first linear layer.
    """

    variant = "concat"

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__(config, schema)
        self.handcrafted_projection = ModalityProjection(
            self.handcrafted_input_dimension,
            config.handcrafted_projection,
            config.dropout,
        )
        self.audio_projection = ModalityProjection(
            schema.audio_dimension, config.audio_projection, config.dropout
        )
        self.text_projection = ModalityProjection(
            schema.text_dimension, config.text_projection, config.dropout
        )
        self.fused_dimension = (
            config.handcrafted_projection
            + config.audio_projection
            + config.text_projection
        )
        self.build_trunk_and_heads(self.fused_dimension + 2)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ("handcrafted", "audio", "text")

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        hidden = torch.cat(
            [
                self.handcrafted_projection(
                    self.handcrafted_input(handcrafted, handcrafted_missing_mask)
                ),
                self.audio_projection(audio, audio_available),
                self.text_projection(text, text_available),
                audio_available.reshape(-1, 1),
                text_available.reshape(-1, 1),
            ],
            dim=-1,
        )
        return self.heads(hidden)


class GatedRanker(BaseRanker):
    """Learned per-candidate modality gates over a common projection width.

    Each modality is projected to the same width and the three are combined as a
    weighted sum whose weights the network computes from the projected states
    and the availability flags. An unavailable modality's gate logit is set to
    ``-inf`` before the softmax, so its weight is exactly zero -- not merely
    small -- and the remaining weights still sum to one. That last part matters:
    without renormalization a candidate missing its transcript would receive a
    systematically smaller fused representation than a complete one, and the
    model would learn to rank on transcript availability rather than on content.

    This is deliberately not attention. With three modalities and a few hundred
    labelled candidates, a query/key/value block would add parameters and
    training instability to express what three scalars already express.
    """

    variant = "gated"

    def __init__(self, config: ModelConfig, schema: DatasetSchema):
        super().__init__(config, schema)
        widths = {
            config.handcrafted_projection,
            config.audio_projection,
            config.text_projection,
        }
        if len(widths) != 1:
            raise ValueError(
                "The gated ranker forms a weighted sum of the modality "
                "projections, so handcrafted_projection, audio_projection and "
                f"text_projection must be equal; got {sorted(widths)}"
            )
        self.width = config.handcrafted_projection

        self.handcrafted_projection = ModalityProjection(
            self.handcrafted_input_dimension, self.width, config.dropout
        )
        self.audio_projection = ModalityProjection(
            schema.audio_dimension, self.width, config.dropout
        )
        self.text_projection = ModalityProjection(
            schema.text_dimension, self.width, config.dropout
        )

        # Gates see all three projected states plus the two availability flags,
        # so the weighting is conditioned on content, not only on presence.
        self.gate = nn.Sequential(
            nn.Linear(self.width * 3 + 2, self.width),
            nn.GELU(),
            nn.Linear(self.width, 3),
        )
        self.build_trunk_and_heads(self.width)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ("handcrafted", "audio", "text")

    def forward(
        self,
        handcrafted: torch.Tensor,
        handcrafted_missing_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_available: torch.Tensor,
        text: torch.Tensor,
        text_available: torch.Tensor,
    ) -> RankerOutput:
        handcrafted_state = self.handcrafted_projection(
            self.handcrafted_input(handcrafted, handcrafted_missing_mask)
        )
        audio_state = self.audio_projection(audio, audio_available)
        text_state = self.text_projection(text, text_available)

        availability = torch.stack(
            [
                torch.ones_like(audio_available),
                audio_available,
                text_available,
            ],
            dim=-1,
        )
        gate_input = torch.cat(
            [
                handcrafted_state,
                audio_state,
                text_state,
                audio_available.reshape(-1, 1),
                text_available.reshape(-1, 1),
            ],
            dim=-1,
        )
        logits = self.gate(gate_input)
        # Unavailable modalities are removed from the softmax entirely rather
        # than given a small weight, so their (zero) projection cannot leak in.
        masked = logits.masked_fill(availability <= 0, float("-inf"))
        gates = torch.softmax(masked, dim=-1)

        stacked = torch.stack([handcrafted_state, audio_state, text_state], dim=1)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)
        return self.heads(fused, gates=gates)
