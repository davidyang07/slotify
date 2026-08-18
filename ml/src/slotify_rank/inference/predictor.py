"""Loading a trained ranker and scoring candidates with it.

The predictor is deliberately narrow: it takes examples that have already been
built by the Phase 3 feature path and returns scores. It does not build
features, does not read manifests, and cannot train -- so there is no code path
in which a "prediction" is really a fresh fit.

Four safety properties, each with a test:

**The checkpoint is verified before it is used.** ``assert_compatible`` refuses
a checkpoint whose variant, feature ordering, dimensions or pipeline versions
differ from the data being scored. A 110-column model handed 96 differently
ordered columns does not crash; it produces confident nonsense, which is worse.

**The normalizer travels with the checkpoint.** Scoring with statistics fitted
on different data silently shifts every input. The checkpoint records the
normalizer's identity and the load checks it.

**Weights only, eval mode, no grad.** ``torch.load(..., weights_only=True)``,
``model.eval()`` so dropout is off, ``torch.inference_mode()`` so nothing
accumulates a graph, and ``map_location="cpu"`` so a CUDA-trained checkpoint
loads on a laptop.

**The model is loaded once.** Constructing a predictor reads the checkpoint;
scoring reuses it. The Express API spawns one process per request today, so
this matters mostly for batch evaluation -- but it also means a persistent
service can be added without touching this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from slotify_rank.datasets.normalizer import FeatureNormalizer, read_normalizer
from slotify_rank.datasets.schema import DatasetSchema, TrainingExample
from slotify_rank.models.base import BaseRanker, count_parameters
from slotify_rank.models.registry import build_model
from slotify_rank.models.schema import ModelConfig
from slotify_rank.training.checkpoint import (
    CheckpointError,
    IncompatibleCheckpoint,
    assert_compatible,
    load_checkpoint,
)
from slotify_rank.inference.schema import (
    ModelIdentity,
    ScoredCandidate,
    normalize_scores,
)

__all__ = ["RankerPredictor", "PredictorError", "score_examples"]


class PredictorError(RuntimeError):
    """The predictor could not be built or could not score."""


def _training_summary_beside(checkpoint_path: Path) -> Mapping[str, Any]:
    """Read the run's summary, if it sits next to the checkpoint.

    Used only to surface `label_source` / `data_provenance` in the response.
    A missing summary is reported as `unknown` rather than assumed to be real
    supervision -- an unlabelled provenance must never read as a human-labelled
    one.
    """
    summary_path = checkpoint_path.parent / "training_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


@dataclass
class RankerPredictor:
    """A loaded, verified, eval-mode ranker."""

    model: BaseRanker
    normalizer: FeatureNormalizer
    schema: DatasetSchema
    identity: ModelIdentity

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str,
        normalizer_path: Path | str | None = None,
        device: str = "cpu",
    ) -> "RankerPredictor":
        """Build a predictor from a checkpoint on disk.

        ``normalizer_path`` defaults to ``normalizer.json`` beside the
        checkpoint, which is where ``training run`` writes it.
        """
        checkpoint = Path(checkpoint_path)
        payload = load_checkpoint(checkpoint)

        normalizer_file = (
            Path(normalizer_path)
            if normalizer_path is not None
            else checkpoint.parent / "normalizer.json"
        )
        if not normalizer_file.is_file():
            raise PredictorError(
                f"No normalizer at {normalizer_file}. A checkpoint cannot be used "
                "without the statistics its inputs were standardized by; scoring "
                "raw features through it would shift every input silently."
            )
        normalizer = read_normalizer(normalizer_file)

        raw_schema = payload.get("input_schema")
        if not isinstance(raw_schema, Mapping):
            raise IncompatibleCheckpoint(
                f"{checkpoint} records no input schema, so its column layout is unknown."
            )
        schema = DatasetSchema.from_mapping(raw_schema)

        raw_model_config = payload.get("model_config")
        if not isinstance(raw_model_config, Mapping):
            raise IncompatibleCheckpoint(
                f"{checkpoint} records no model config, so the architecture it was "
                "trained with cannot be rebuilt."
            )
        model_config = ModelConfig.from_mapping(
            raw_model_config, where=f"{checkpoint}::model_config"
        )
        variant = str(payload.get("model_variant", ""))

        # Refuse a mismatch before any weight is loaded. Split hashes are allowed
        # to differ: scoring a new upload with a trained model is the whole point.
        assert_compatible(
            payload,
            model_variant=variant,
            schema=schema,
            normalizer=normalizer,
            allow_split_mismatch=True,
        )

        model = build_model(model_config, schema)
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise CheckpointError(f"{checkpoint} has no usable model_state_dict")
        missing, unexpected = model.load_state_dict(state, strict=True)  # type: ignore[misc]
        if missing or unexpected:  # pragma: no cover - strict=True already raises
            raise IncompatibleCheckpoint(
                f"{checkpoint}: state dict does not match the rebuilt model "
                f"(missing={list(missing)}, unexpected={list(unexpected)})"
            )
        model.to(torch.device(device))
        model.eval()

        summary = _training_summary_beside(checkpoint)
        identity = ModelIdentity(
            model_variant=variant,
            model_run_id=str(payload.get("run_id", "")),
            checkpoint_path=str(checkpoint),
            checkpoint_git_commit=str(payload.get("git_commit", "")),
            parameter_count=count_parameters(model, trainable_only=False),
            handcrafted_dimension=schema.handcrafted_dimension,
            audio_dimension=schema.audio_dimension,
            text_dimension=schema.text_dimension,
            feature_spec_version=schema.feature_spec_version,
            feature_pipeline_version=schema.feature_pipeline_version,
            input_schema_version=schema.input_schema_version,
            normalizer_version=normalizer.normalizer_version,
            training_label_source=str(summary.get("label_source", "unknown")),
            training_data_provenance=str(summary.get("data_provenance", "unknown")),
        )

        return cls(model=model, normalizer=normalizer, schema=schema, identity=identity)

    def assert_examples_compatible(self, examples: Sequence[TrainingExample]) -> None:
        """Refuse examples whose vector widths differ from the trained layout."""
        for example in examples:
            if example.handcrafted.shape[0] != self.schema.handcrafted_dimension:
                raise IncompatibleCheckpoint(
                    f"{example.candidate_id}: {example.handcrafted.shape[0]} handcrafted "
                    f"features but the checkpoint was trained on "
                    f"{self.schema.handcrafted_dimension}."
                )
            if example.audio.shape[0] != self.schema.audio_dimension:
                raise IncompatibleCheckpoint(
                    f"{example.candidate_id}: audio block is "
                    f"{example.audio.shape[0]}-d, expected {self.schema.audio_dimension}."
                )
            if example.text.shape[0] != self.schema.text_dimension:
                raise IncompatibleCheckpoint(
                    f"{example.candidate_id}: text block is "
                    f"{example.text.shape[0]}-d, expected {self.schema.text_dimension}."
                )

    def score(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[float], list[float | None]]:
        """Score examples in the order given. Returns (ranking, acceptability).

        Deterministic: no dropout (eval mode), no sampling, no gradient, and the
        whole batch goes through in one forward pass so a batch boundary cannot
        change a result.
        """
        if not examples:
            return [], []
        self.assert_examples_compatible(examples)

        raw = np.stack([example.handcrafted for example in examples])
        masks = np.stack([example.handcrafted_missing_mask for example in examples])
        normalized = self.normalizer.transform(raw, masks)
        if not np.isfinite(normalized).all():
            raise PredictorError(
                "Normalization produced NaN or infinity on these candidates. The "
                "fitted statistics do not describe this audio's features; scoring "
                "through it would produce numbers with no meaning."
            )

        inputs = {
            "handcrafted": torch.from_numpy(
                np.ascontiguousarray(normalized, dtype=np.float32)
            ),
            "handcrafted_missing_mask": torch.from_numpy(
                np.ascontiguousarray(masks.astype(np.float32))
            ),
            "audio": torch.from_numpy(
                np.stack([example.audio for example in examples]).astype(np.float32)
            ),
            "audio_available": torch.tensor(
                [float(example.audio_available) for example in examples],
                dtype=torch.float32,
            ),
            "text": torch.from_numpy(
                np.stack([example.text for example in examples]).astype(np.float32)
            ),
            "text_available": torch.tensor(
                [float(example.text_available) for example in examples],
                dtype=torch.float32,
            ),
        }

        with torch.inference_mode():
            output = self.model(**inputs)

        ranking = [float(value) for value in output.ranking_score.detach().cpu()]
        if output.acceptability_logit is None:
            acceptability: list[float | None] = [None] * len(examples)
        else:
            acceptability = [
                float(value) for value in output.acceptability_logit.detach().cpu()
            ]
        return ranking, acceptability


def score_examples(
    predictor: RankerPredictor,
    examples: Sequence[TrainingExample],
    timestamps_ms: Mapping[str, int],
    feature_status: Mapping[str, str],
) -> list[ScoredCandidate]:
    """Score and rank, best first.

    Ties break on the earlier timestamp then the candidate id, so two runs over
    the same audio produce byte-identical output.
    """
    ranking, acceptability = predictor.score(examples)
    normalized = normalize_scores(ranking)

    ordered = sorted(
        range(len(examples)),
        key=lambda index: (
            -ranking[index],
            timestamps_ms.get(examples[index].candidate_id, 0),
            examples[index].candidate_id,
        ),
    )

    scored: list[ScoredCandidate] = []
    for rank, index in enumerate(ordered, start=1):
        example = examples[index]
        timestamp_ms = int(timestamps_ms.get(example.candidate_id, 0))
        scored.append(
            ScoredCandidate(
                candidate_id=example.candidate_id,
                insertion_time_seconds=round(timestamp_ms / 1000.0, 3),
                insertion_ms=timestamp_ms,
                raw_score=round(ranking[index], 6),
                normalized_score=normalized[index],
                rank=rank,
                acceptability_logit=(
                    None
                    if acceptability[index] is None
                    else round(float(acceptability[index]), 6)
                ),
                audio_available=bool(example.audio_available),
                text_available=bool(example.text_available),
                feature_status=str(
                    feature_status.get(example.candidate_id, example.feature_status)
                ),
            )
        )
    return scored
