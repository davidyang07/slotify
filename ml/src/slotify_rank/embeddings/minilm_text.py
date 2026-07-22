"""Frozen MiniLM embeddings of transcript context around each candidate.

``sentence-transformers/all-MiniLM-L6-v2``. **Native output dimension: 384.**

The transcript representation handed to the future ranker is a *constructed*
1,536-dimensional vector::

    [ before (384) | after (384) | |before - after| (384) | before * after (384) ]

1,536 is arithmetic, not a model property, and this distinction is worth being
pedantic about: a claim that "the text encoder produces 1,536 dimensions" would
be false. MiniLM produces 384; the concatenation of four 384-blocks produces
1,536. Both numbers are recorded separately in the manifest
(``text_embedding_dimension`` is the native 384; the constructed width is
derived from the number of blocks stored).

The absolute difference and the elementwise product are the standard
sentence-pair interaction features (as in InferSent and SBERT's classification
head): the difference captures *how much* the topic moved, the product captures
*where* the two contexts agree. A model given only the concatenation has to
learn those interactions from scratch through a linear layer, which it cannot
do -- they are not linear functions of the inputs.

No fine-tuning. The model runs in inference mode only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from slotify_rank.config.feature_settings import TextEmbeddingConfig
from slotify_rank.embeddings.device import resolve_device

__all__ = [
    "MiniLmTextEncoder",
    "TEXT_EMBEDDING_BLOCKS",
    "build_constructed_vector",
    "cosine_similarity",
]

#: Blocks of the constructed transcript vector, in concatenation order.
TEXT_EMBEDDING_BLOCKS: tuple[str, ...] = (
    "before",
    "after",
    "absolute_difference",
    "elementwise_product",
)


@dataclass
class MiniLmTextEncoder:
    """A loaded MiniLM encoder with batching."""

    config: TextEmbeddingConfig
    _model: Any = None
    _device: str = "cpu"

    def load(self) -> "MiniLmTextEncoder":
        from sentence_transformers import SentenceTransformer

        self._device = resolve_device(self.config.device)
        model = SentenceTransformer(
            self.config.model_id,
            revision=self.config.model_revision,
            device=self._device,
        )
        model.max_seq_length = self.config.max_seq_length
        model.eval()
        self._model = model
        return self

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int:
        """Native embedding width, read from the model (384 for MiniLM-L6-v2)."""
        if self._model is None:
            raise RuntimeError("MiniLmTextEncoder.load() must be called first")
        return int(self._model.get_sentence_embedding_dimension())

    def release(self) -> None:
        from slotify_rank.embeddings.device import free_model

        model, self._model = self._model, None
        if model is not None:
            free_model(model)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings into ``(n, 384)`` float32.

        Empty input returns an empty ``(0, dim)`` array rather than raising:
        an episode where no candidate had usable context is a legitimate
        outcome, and the caller has already recorded the masks.
        """
        if self._model is None:
            raise RuntimeError("MiniLmTextEncoder.load() must be called first")
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"Text encoder returned a {array.ndim}-D array; expected (n, dim)"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                "Text encoder produced NaN or infinity. Refusing to cache a "
                "corrupt embedding."
            )
        return array


def build_constructed_vector(
    before: np.ndarray | None, after: np.ndarray | None
) -> np.ndarray | None:
    """Concatenate the four blocks into the 1,536-dimensional vector.

    ``None`` when either side is missing. The alternative -- substituting zeros
    for the absent side -- would make the product block all-zero and the
    difference block equal to the present side, which a model would read as a
    strong, meaningful signal rather than as absence. Missing context is carried
    by the mask instead.
    """
    if before is None or after is None:
        return None
    left = np.asarray(before, dtype=np.float32)
    right = np.asarray(after, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(
            f"before {left.shape} and after {right.shape} embeddings must have the "
            "same shape"
        )
    return np.concatenate(
        [left, right, np.abs(left - right), left * right]
    ).astype(np.float32)


def cosine_similarity(
    before: np.ndarray | None, after: np.ndarray | None
) -> float | None:
    """Cosine similarity between the two context embeddings.

    ``None`` when either side is missing -- there is no defensible similarity
    against absent text, and 0.0 would read as "completely unrelated", which is
    a specific and wrong claim.
    """
    if before is None or after is None:
        return None
    left = np.asarray(before, dtype=np.float64)
    right = np.asarray(after, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    # Clamp: floating-point error can push a unit-vector dot product a hair
    # outside [-1, 1], which would make `1 - cosine` negative.
    return max(-1.0, min(1.0, similarity))
