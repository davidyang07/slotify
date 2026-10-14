"""The training example, the modality construction, and the exclusion taxonomy.

Model input dimensions
----------------------

Phase 3 stores 384-dimensional vectors (Whisper ``tiny.en``'s encoder width and
MiniLM's embedding width are both 384). A model does not consume one of those;
it consumes a *constructed* representation, and the construction is pinned here
so a checkpoint's weights can never be applied to a differently-ordered input.

``audio`` -- 4 x 384 = 1536. All four parts are read straight from the stored
per-episode array: ``before``, ``after``, ``context`` and ``difference``
(``after - before``, written by the Phase 3 pooling stage).

``text`` -- 4 x 384 = 1536. Phase 3 stores only ``before`` and ``after``; the
other two parts are cheap algebra over those stored vectors -- ``difference``
(``after - before``) and ``product`` (elementwise ``before * after``). This is
arithmetic on cached arrays, **not** a re-run of MiniLM. It exists because a
ranker's question is "did the topic change here", which is a relation between
the two sides, and asking an MLP to discover an elementwise product from a bare
concatenation is a poor use of a small network on a small dataset.

``handcrafted`` -- whatever the feature manifest header declares (110 under
``featurespec-v1.1.0``). Read from the header, never hard-coded; a manifest with
a different column count is a hard failure rather than a truncation.

Missing modalities
------------------

A candidate with no transcript legitimately has no text vector. Its ``text``
block is zero-filled and its ``text_available`` flag is ``False`` -- the two
always travel together, so a model can distinguish "the topic did not change"
from "we do not know whether the topic changed". A record Phase 3 marked
``failed`` is a different thing entirely: it is dropped, not masked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import (
    MODEL_INPUT_SCHEMA_VERSION,
    TRAINING_DATASET_VERSION,
)

__all__ = [
    "AUDIO_EMBEDDING_PARTS",
    "TEXT_EMBEDDING_PARTS",
    "TEXT_STORED_SIDES",
    "NATIVE_EMBEDDING_DIMENSION",
    "ExclusionReason",
    "EXCLUSION_REASONS",
    "TrainingExample",
    "DatasetSchema",
    "TEACHER_FEATURE_PREFIX",
    "audio_dimension",
    "text_dimension",
    "teacher_feature_names",
    "project_handcrafted",
]

#: Prefix of every handcrafted column that is ``heuristic_offline_v1``'s own
#: opinion of a candidate: its total score, its additive components, its raw
#: total and its clamp flag (see ``features/structural.py``). Weak labels are
#: within-episode quantiles of that same score, so a model trained on weak labels
#: with these columns as inputs can reproduce its teacher by copying one column.
TEACHER_FEATURE_PREFIX = "heuristic_"


def teacher_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    """The columns that carry the heuristic baseline's own score, in order."""
    return tuple(name for name in names if str(name).startswith(TEACHER_FEATURE_PREFIX))

#: Native width of one stored vector. Both Whisper tiny.en's encoder and MiniLM
#: happen to be 384; this constant is a documented *expectation*, and every
#: loader verifies the artifact's own declared dimension against it rather than
#: assuming it.
NATIVE_EMBEDDING_DIMENSION = 384

#: Order of the concatenated audio block. Matches the Phase 3 pooling kinds.
AUDIO_EMBEDDING_PARTS: tuple[str, ...] = ("before", "after", "context", "difference")

#: Sides Phase 3 actually stores for text.
TEXT_STORED_SIDES: tuple[str, ...] = ("before", "after")

#: Order of the concatenated text block. ``difference`` and ``product`` are
#: derived from the stored sides at load time.
TEXT_EMBEDDING_PARTS: tuple[str, ...] = ("before", "after", "difference", "product")


def audio_dimension(native: int = NATIVE_EMBEDDING_DIMENSION) -> int:
    return native * len(AUDIO_EMBEDDING_PARTS)


def text_dimension(native: int = NATIVE_EMBEDDING_DIMENSION) -> int:
    return native * len(TEXT_EMBEDDING_PARTS)


#: Why a candidate did not become a training example. Every excluded candidate
#: is counted under exactly one of these; a run that silently drops rows is a
#: run whose dataset size cannot be reconciled with the corpus.
#: ``ineligible_candidate`` is additive to the required taxonomy: a candidate can
#: carry ``eligible_for_labelling=False`` without being synthetic padding, and
#: folding that into ``synthetic_excluded`` would misattribute the exclusion.
ExclusionReason = Literal[
    "eligible",
    "missing_label",
    "missing_features",
    "stale_features",
    "invalid_features",
    "synthetic_excluded",
    "ineligible_candidate",
    "split_mismatch",
    "unsupported_version",
]

EXCLUSION_REASONS: tuple[str, ...] = (
    "eligible",
    "missing_label",
    "missing_features",
    "stale_features",
    "invalid_features",
    "synthetic_excluded",
    "ineligible_candidate",
    "split_mismatch",
    "unsupported_version",
)


@dataclass(frozen=True)
class TrainingExample:
    """One candidate, projected onto the fixed model input layout.

    Arrays are ``float32`` because that is what the model consumes; carrying
    ``float64`` here would double the dataset's memory for no numerical benefit
    on a CPU-trained MLP.
    """

    candidate_id: str
    episode_id: str
    split: str
    handcrafted: np.ndarray
    handcrafted_missing_mask: np.ndarray
    audio: np.ndarray
    audio_available: bool
    text: np.ndarray
    text_available: bool
    quality_score: float
    is_acceptable: bool
    feature_status: str = ""

    def __post_init__(self) -> None:
        if self.handcrafted.shape != self.handcrafted_missing_mask.shape:
            raise ValueError(
                f"{self.candidate_id}: handcrafted values {self.handcrafted.shape} and "
                f"mask {self.handcrafted_missing_mask.shape} disagree"
            )
        for name, array in (
            ("handcrafted", self.handcrafted),
            ("audio", self.audio),
            ("text", self.text),
        ):
            if array.ndim != 1:
                raise ValueError(
                    f"{self.candidate_id}: {name} must be a 1-D vector, got shape "
                    f"{array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"{self.candidate_id}: {name} contains NaN or infinity. An "
                    "example that cannot be trained on must be excluded upstream, "
                    "not repaired here."
                )
        # A zero-filled block is only meaningful next to a False flag. An
        # available modality whose vector is entirely zero is indistinguishable
        # from a masked one, which defeats the whole point of the mask.
        if self.audio_available and not self.audio.any():
            raise ValueError(
                f"{self.candidate_id}: marked audio-available but the vector is all "
                "zeros, which is exactly what an unavailable modality looks like"
            )
        if self.text_available and not self.text.any():
            raise ValueError(
                f"{self.candidate_id}: marked text-available but the vector is all zeros"
            )


@dataclass(frozen=True)
class DatasetSchema:
    """The column contract a model is built against and a checkpoint records."""

    handcrafted_feature_names: tuple[str, ...]
    handcrafted_dimension: int
    audio_dimension: int
    text_dimension: int
    native_embedding_dimension: int
    audio_parts: tuple[str, ...] = AUDIO_EMBEDDING_PARTS
    text_parts: tuple[str, ...] = TEXT_EMBEDDING_PARTS
    feature_spec_version: str = ""
    feature_pipeline_version: str = ""
    input_schema_version: str = MODEL_INPUT_SCHEMA_VERSION
    dataset_version: str = TRAINING_DATASET_VERSION

    def __post_init__(self) -> None:
        if len(self.handcrafted_feature_names) != self.handcrafted_dimension:
            raise ValueError(
                f"{len(self.handcrafted_feature_names)} feature names but "
                f"handcrafted_dimension={self.handcrafted_dimension}"
            )
        expected_audio = self.native_embedding_dimension * len(self.audio_parts)
        if self.audio_dimension != expected_audio:
            raise ValueError(
                f"audio_dimension {self.audio_dimension} != "
                f"{len(self.audio_parts)} parts x {self.native_embedding_dimension}"
            )
        expected_text = self.native_embedding_dimension * len(self.text_parts)
        if self.text_dimension != expected_text:
            raise ValueError(
                f"text_dimension {self.text_dimension} != "
                f"{len(self.text_parts)} parts x {self.native_embedding_dimension}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "handcrafted_feature_names": list(self.handcrafted_feature_names),
            "handcrafted_dimension": self.handcrafted_dimension,
            "audio_dimension": self.audio_dimension,
            "text_dimension": self.text_dimension,
            "native_embedding_dimension": self.native_embedding_dimension,
            "audio_parts": list(self.audio_parts),
            "text_parts": list(self.text_parts),
            "feature_spec_version": self.feature_spec_version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "input_schema_version": self.input_schema_version,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSchema":
        return cls(
            handcrafted_feature_names=tuple(
                str(name) for name in raw["handcrafted_feature_names"]
            ),
            handcrafted_dimension=int(raw["handcrafted_dimension"]),
            audio_dimension=int(raw["audio_dimension"]),
            text_dimension=int(raw["text_dimension"]),
            native_embedding_dimension=int(raw["native_embedding_dimension"]),
            audio_parts=tuple(str(part) for part in raw.get("audio_parts", AUDIO_EMBEDDING_PARTS)),
            text_parts=tuple(str(part) for part in raw.get("text_parts", TEXT_EMBEDDING_PARTS)),
            feature_spec_version=str(raw.get("feature_spec_version", "")),
            feature_pipeline_version=str(raw.get("feature_pipeline_version", "")),
            input_schema_version=str(raw.get("input_schema_version", "")),
            dataset_version=str(raw.get("dataset_version", "")),
        )

    def with_handcrafted(self, names: Sequence[str]) -> "DatasetSchema":
        """This schema with its handcrafted block narrowed to ``names``.

        Everything else -- the learned blocks, their construction and every
        version -- is kept, because a column subset is still the same feature
        spec read through the same pipeline; only the model's view of it shrinks.
        """
        kept = tuple(str(name) for name in names)
        unknown = [name for name in kept if name not in self.handcrafted_feature_names]
        if unknown:
            raise ValueError(
                f"Cannot narrow the schema to columns it does not have: {unknown[:5]}"
            )
        return replace(
            self, handcrafted_feature_names=kept, handcrafted_dimension=len(kept)
        )

    def incompatibilities(self, other: "DatasetSchema") -> list[str]:
        """Every reason ``other`` cannot be substituted for this schema.

        Returned as a list rather than a bool so a rejected checkpoint can say
        which of five things changed instead of "incompatible".
        """
        reasons: list[str] = []
        if self.handcrafted_feature_names != other.handcrafted_feature_names:
            if self.handcrafted_dimension != other.handcrafted_dimension:
                reasons.append(
                    f"handcrafted dimension {other.handcrafted_dimension} != "
                    f"{self.handcrafted_dimension}"
                )
            else:
                changed = [
                    f"{a}->{b}"
                    for a, b in zip(
                        self.handcrafted_feature_names, other.handcrafted_feature_names
                    )
                    if a != b
                ]
                reasons.append(f"handcrafted feature ordering differs: {changed[:3]}")
        for field_name in ("audio_dimension", "text_dimension", "native_embedding_dimension"):
            if getattr(self, field_name) != getattr(other, field_name):
                reasons.append(
                    f"{field_name} {getattr(other, field_name)} != {getattr(self, field_name)}"
                )
        for field_name in ("audio_parts", "text_parts"):
            if getattr(self, field_name) != getattr(other, field_name):
                reasons.append(
                    f"{field_name} {list(getattr(other, field_name))} != "
                    f"{list(getattr(self, field_name))}"
                )
        for field_name in (
            "feature_spec_version",
            "feature_pipeline_version",
            "input_schema_version",
        ):
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            if mine and theirs and mine != theirs:
                reasons.append(f"{field_name} {theirs!r} != {mine!r}")
        return reasons


def project_handcrafted(
    examples: Sequence[TrainingExample],
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> list[TrainingExample]:
    """Re-column every example's handcrafted block onto ``target_names``, by name.

    The values and their missing mask move together, so a masked feature stays
    masked in its new position. Every target column must exist in the source,
    exactly once: a missing column is a refusal rather than a zero-fill, because
    a zero standing in for an absent measurement is indistinguishable from a
    real zero to the model. Extra source columns are dropped, which is the point
    -- a model trained without a column must never be shown it.
    """
    source = tuple(str(name) for name in source_names)
    target = tuple(str(name) for name in target_names)
    duplicates = sorted({name for name in source if source.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"The source layout names {duplicates[:5]} more than once, so a "
            "projection by name would be ambiguous."
        )
    position = {name: index for index, name in enumerate(source)}
    missing = [name for name in target if name not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} required column(s) are absent from the source layout: "
            f"{missing[:5]}"
        )
    if source == target:
        return list(examples)
    indices = np.asarray([position[name] for name in target], dtype=np.int64)
    projected: list[TrainingExample] = []
    for example in examples:
        if example.handcrafted.shape[0] != len(source):
            raise ValueError(
                f"{example.candidate_id}: {example.handcrafted.shape[0]} handcrafted "
                f"values but the source layout declares {len(source)} columns"
            )
        projected.append(
            replace(
                example,
                handcrafted=np.ascontiguousarray(example.handcrafted[indices]),
                handcrafted_missing_mask=np.ascontiguousarray(
                    example.handcrafted_missing_mask[indices]
                ),
            )
        )
    return projected
