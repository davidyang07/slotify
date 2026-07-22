"""Train-only standardization of the handcrafted scalar features.

The rule this module exists to enforce: **statistics are fitted on the training
split and on nothing else.** Fitting over the whole corpus is the most common
way a small-dataset result turns out to be optimistic -- the validation
episodes' means and variances leak into every training input, and the leak is
invisible because the numbers all look reasonable. :func:`fit` therefore takes
only the examples it is given, and :func:`fit_from_examples` refuses anything
that is not the training split.

Three details that are easy to get wrong and are handled explicitly:

**Missing values are excluded from the fit.** A masked feature carries a
sentinel, not a measurement. Averaging the sentinel in would shift the mean
toward it in proportion to how often the feature is missing.

**Binary flags are not standardized.** ``sentence_end`` and the ``source_*``
one-hots are indicator variables; rescaling them by a sample standard deviation
turns 0/1 into two arbitrary reals that vary between runs and destroys the
"absence is exactly zero" property the masks rely on. They are detected from the
fitted data (every observed value in {0, 1}) and passed through, and which
columns those were is recorded in the artifact.

**Constant features are centered, never scaled.** Dividing by a near-zero
standard deviation converts a feature that carries no information into one with
enormous magnitude, which then dominates the first layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slotify_rank.config.versions import NORMALIZER_VERSION
from slotify_rank.data.checksum import atomic_write_bytes
from slotify_rank.datasets.schema import TrainingExample

__all__ = [
    "FeatureNormalizer",
    "DEFAULT_EPSILON",
    "fit_normalizer",
    "write_normalizer",
    "read_normalizer",
]

#: Floor for the standard deviation. A feature whose train-split spread is below
#: this is treated as constant rather than amplified by 1/eps.
DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureNormalizer:
    """Fitted statistics plus the provenance that makes them re-checkable."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    standard_deviations: np.ndarray
    constant_feature_mask: np.ndarray
    binary_feature_mask: np.ndarray
    fit_episode_count: int
    fit_candidate_count: int
    training_split_hash: str = ""
    feature_pipeline_version: str = ""
    feature_spec_version: str = ""
    epsilon: float = DEFAULT_EPSILON
    normalizer_version: str = NORMALIZER_VERSION

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        for name, array in (
            ("means", self.means),
            ("standard_deviations", self.standard_deviations),
            ("constant_feature_mask", self.constant_feature_mask),
            ("binary_feature_mask", self.binary_feature_mask),
        ):
            if array.shape != (width,):
                raise ValueError(
                    f"{name} has shape {array.shape}, expected ({width},) to match "
                    "feature_names"
                )
        if not np.isfinite(self.means).all():
            raise ValueError("Fitted means contain NaN or infinity")
        if not np.isfinite(self.standard_deviations).all():
            raise ValueError("Fitted standard deviations contain NaN or infinity")
        if (self.standard_deviations <= 0).any():
            raise ValueError(
                "A standard deviation is non-positive; constant features must be "
                "recorded with a divisor of 1.0, never 0.0"
            )

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def transform(
        self, values: np.ndarray, missing_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Standardize a ``(n, d)`` or ``(d,)`` block of raw feature values.

        Masked entries are set to exactly zero *after* standardization: the
        sentinel they carried is not a measurement, and zero is the value the
        availability mask tells the model to ignore.
        """
        matrix = np.atleast_2d(np.asarray(values, dtype=np.float32))
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Cannot normalize a {matrix.shape[1]}-column block with a "
                f"{self.dimension}-column normalizer. The feature layout changed; "
                "refit rather than reinterpreting the columns."
            )
        standardized = (matrix - self.means) / self.standard_deviations
        # Binary indicators pass through unchanged.
        binary = self.binary_feature_mask
        standardized[:, binary] = matrix[:, binary]
        if missing_mask is not None:
            mask = np.atleast_2d(np.asarray(missing_mask, dtype=bool))
            if mask.shape != matrix.shape:
                raise ValueError(
                    f"missing_mask shape {mask.shape} does not match values "
                    f"{matrix.shape}"
                )
            standardized[mask] = 0.0
        result = standardized.astype(np.float32, copy=False)
        return result[0] if np.asarray(values).ndim == 1 else result

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer_version": self.normalizer_version,
            "feature_names": list(self.feature_names),
            "means": [float(value) for value in self.means],
            "standard_deviations": [float(value) for value in self.standard_deviations],
            "constant_feature_mask": [bool(value) for value in self.constant_feature_mask],
            "binary_feature_mask": [bool(value) for value in self.binary_feature_mask],
            "fit_episode_count": self.fit_episode_count,
            "fit_candidate_count": self.fit_candidate_count,
            "training_split_hash": self.training_split_hash,
            "feature_pipeline_version": self.feature_pipeline_version,
            "feature_spec_version": self.feature_spec_version,
            "epsilon": self.epsilon,
            "fitted_on": "train split only",
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FeatureNormalizer":
        version = str(raw.get("normalizer_version", ""))
        if version != NORMALIZER_VERSION:
            raise ValueError(
                f"Unsupported normalizer_version {version!r}; this build reads "
                f"{NORMALIZER_VERSION!r}"
            )
        return cls(
            feature_names=tuple(str(name) for name in raw["feature_names"]),
            means=np.asarray(raw["means"], dtype=np.float32),
            standard_deviations=np.asarray(raw["standard_deviations"], dtype=np.float32),
            constant_feature_mask=np.asarray(raw["constant_feature_mask"], dtype=bool),
            binary_feature_mask=np.asarray(raw["binary_feature_mask"], dtype=bool),
            fit_episode_count=int(raw["fit_episode_count"]),
            fit_candidate_count=int(raw["fit_candidate_count"]),
            training_split_hash=str(raw.get("training_split_hash", "")),
            feature_pipeline_version=str(raw.get("feature_pipeline_version", "")),
            feature_spec_version=str(raw.get("feature_spec_version", "")),
            epsilon=float(raw.get("epsilon", DEFAULT_EPSILON)),
            normalizer_version=version,
        )


def fit_normalizer(
    examples: Sequence[TrainingExample],
    feature_names: Sequence[str],
    epsilon: float = DEFAULT_EPSILON,
    training_split_hash: str = "",
    feature_pipeline_version: str = "",
    feature_spec_version: str = "",
    expected_split: str | None = "train",
) -> FeatureNormalizer:
    """Fit standardization statistics over ``examples``.

    ``expected_split`` is a guard, not a filter: passing examples from another
    split raises rather than being quietly dropped, because a caller that
    reached here with validation rows believed it was doing something valid.
    """
    if not examples:
        raise ValueError(
            "Cannot fit a normalizer on zero examples. The training split is "
            "empty, which means either the labels or the split assignment is "
            "missing -- both are failures, not something to default around."
        )
    if expected_split is not None:
        offenders = sorted({e.split for e in examples if e.split != expected_split})
        if offenders:
            raise ValueError(
                f"Refusing to fit normalization statistics on split(s) {offenders}. "
                f"Only {expected_split!r} examples may be fitted on; anything else "
                "leaks held-out statistics into every training input."
            )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    names = tuple(str(name) for name in feature_names)
    width = len(names)
    matrix = np.stack([example.handcrafted for example in examples]).astype(np.float32)
    masks = np.stack(
        [example.handcrafted_missing_mask for example in examples]
    ).astype(bool)
    if matrix.shape[1] != width:
        raise ValueError(
            f"Examples carry {matrix.shape[1]} features but {width} names were given"
        )

    observed = ~masks
    counts = observed.sum(axis=0)

    means = np.zeros(width, dtype=np.float32)
    deviations = np.ones(width, dtype=np.float32)
    constant = np.zeros(width, dtype=bool)
    binary = np.zeros(width, dtype=bool)

    for column in range(width):
        values = matrix[observed[:, column], column]
        if values.size == 0:
            # Never observed on the train split: centering it on an arbitrary
            # value would be a guess, so it is left as an untouched constant.
            constant[column] = True
            continue
        unique = np.unique(values)
        if unique.size == 1:
            # Constant beats binary: a feature that took one value on the train
            # split carries no information, and centering it to zero says so.
            # Classifying it as a binary indicator instead would pass its
            # constant magnitude straight into the first layer as a second bias.
            constant[column] = True
            means[column] = float(unique[0])
            deviations[column] = 1.0
            continue
        if unique.size == 2 and np.isin(unique, (0.0, 1.0)).all():
            binary[column] = True
            # Recorded for completeness; transform() bypasses them anyway.
            means[column] = 0.0
            deviations[column] = 1.0
            continue
        mean = float(values.mean())
        deviation = float(values.std())
        means[column] = mean
        if deviation < epsilon:
            constant[column] = True
            deviations[column] = 1.0
        else:
            deviations[column] = deviation

    return FeatureNormalizer(
        feature_names=names,
        means=means,
        standard_deviations=deviations,
        constant_feature_mask=constant,
        binary_feature_mask=binary,
        fit_episode_count=len({example.episode_id for example in examples}),
        fit_candidate_count=len(examples),
        training_split_hash=training_split_hash,
        feature_pipeline_version=feature_pipeline_version,
        feature_spec_version=feature_spec_version,
        epsilon=epsilon,
    )


def write_normalizer(path: Path, normalizer: FeatureNormalizer) -> None:
    """Atomically write the fitted artifact (OneDrive-safe, see data.checksum)."""
    payload = json.dumps(normalizer.to_dict(), indent=2, ensure_ascii=False) + "\n"
    atomic_write_bytes(Path(path), payload.encode("utf-8"))


def read_normalizer(path: Path) -> FeatureNormalizer:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Normalizer artifact not found: {file_path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_path} is not valid JSON: {error}") from error
    return FeatureNormalizer.from_mapping(raw)
