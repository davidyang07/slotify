"""Product-facing inference: load a trained ranker and score real candidates.

Separate from ``training`` on purpose. Nothing here can fit anything, so a
"prediction" can never quietly be a fresh model; and nothing in ``training``
imports this, so the product path cannot drag a training dependency into a
request.
"""

from slotify_rank.inference.schema import (
    INFERENCE_SCHEMA_VERSION,
    ExcludedCandidate,
    InferenceResult,
    ModelIdentity,
    ScoredCandidate,
    normalize_scores,
)

__all__ = [
    "INFERENCE_SCHEMA_VERSION",
    "ExcludedCandidate",
    "InferenceResult",
    "ModelIdentity",
    "ScoredCandidate",
    "normalize_scores",
]
