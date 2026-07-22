"""Variant name -> constructor, plus the descriptions the CLI prints.

The registry is the only place that maps a name to a class. That keeps
``models list`` honest (it enumerates what actually exists rather than a
hand-maintained list) and means a checkpoint's recorded ``model_variant`` can be
resolved without importing anything variant-specific.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from slotify_rank.datasets.schema import DatasetSchema
from slotify_rank.models.base import BaseRanker
from slotify_rank.models.schema import ModelConfig
from slotify_rank.models.variants import (
    AudioOnlyRanker,
    ConcatRanker,
    GatedRanker,
    HandcraftedRanker,
    TextOnlyRanker,
)

__all__ = [
    "MODEL_REGISTRY",
    "MODEL_DESCRIPTIONS",
    "build_model",
    "list_models",
    "describe_variant",
]

MODEL_REGISTRY: Mapping[str, Callable[[ModelConfig, DatasetSchema], BaseRanker]] = {
    HandcraftedRanker.variant: HandcraftedRanker,
    TextOnlyRanker.variant: TextOnlyRanker,
    AudioOnlyRanker.variant: AudioOnlyRanker,
    ConcatRanker.variant: ConcatRanker,
    GatedRanker.variant: GatedRanker,
}

MODEL_DESCRIPTIONS: Mapping[str, str] = {
    "handcrafted": (
        "MLP over the normalized handcrafted scalars and their missing masks. "
        "The learned non-embedding baseline."
    ),
    "text_only": "MLP over the constructed transcript representation alone.",
    "audio_only": (
        "MLP over the cached Whisper representation. Set "
        "include_handcrafted_features to switch between the "
        "audio_embedding_only and audio_plus_acoustic experiments."
    ),
    "concat": (
        "Projects handcrafted, audio and text separately, then concatenates "
        "them into a shared fusion trunk."
    ),
    "gated": (
        "Projects each modality to a common width and combines them with "
        "learned gates; an unavailable modality receives exactly zero weight."
    ),
}


def build_model(config: ModelConfig, schema: DatasetSchema) -> BaseRanker:
    """Construct the variant named by ``config``."""
    factory = MODEL_REGISTRY.get(config.variant)
    if factory is None:
        raise KeyError(
            f"Unknown model variant {config.variant!r}. Known variants: "
            f"{sorted(MODEL_REGISTRY)}"
        )
    return factory(config, schema)


def list_models() -> list[dict[str, str]]:
    return [
        {"variant": name, "description": MODEL_DESCRIPTIONS.get(name, "")}
        for name in sorted(MODEL_REGISTRY)
    ]


def describe_variant(
    variant: str, config: ModelConfig, schema: DatasetSchema
) -> dict[str, Any]:
    """Build the model once purely to report its real shape and size."""
    if variant not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model variant {variant!r}. Known variants: "
            f"{sorted(MODEL_REGISTRY)}"
        )
    model = build_model(config, schema)
    payload = model.describe()
    payload["description"] = MODEL_DESCRIPTIONS.get(variant, "")
    return payload
