"""Loading a trained ranker and scoring candidates with it.

The predictor is deliberately narrow: it takes examples that have already been
built by the Phase 3 feature path and returns scores. It does not build
features, does not read manifests, and cannot train -- so there is no code path
in which a "prediction" is really a fresh fit.

Five safety properties, each with a test:

**The checkpoint is consistent with itself before it is used.** At load,
``assert_compatible`` checks the checkpoint's declared input schema against its
own stored feature ordering and against the normalizer, and the weights are
loaded ``strict=True`` into an architecture rebuilt from that schema, so a
checkpoint whose metadata and weights disagree is refused rather than run.

**The data is checked against the checkpoint, by name.** Load-time checks
compare the checkpoint with itself; they say nothing about the data a request
actually produced. :func:`align_examples` compares the *served* schema -- the
column names and order the feature path built for this audio -- with the
checkpoint's input schema before anything is scored. A learned block of a
different width or construction is refused. A handcrafted column the
checkpoint needs and the served features lack is refused with
``IncompatibleCheckpoint``: a zero standing in for a missing measurement looks
like a real zero to the model. A served layout that is a reordering or a
superset of the checkpoint's is projected onto the checkpoint's columns by
name, which is how a weak-label checkpoint (trained without the ``heuristic_*``
columns) scores features that still carry them. A feature spec or pipeline
version that differs from the checkpoint's is disclosed in the response, not
refused: the names and widths are the contract the weights depend on, and a
version bump elsewhere in the repository must not take the served model down.
A 110-column model handed 110 differently named columns does not crash; it
produces confident nonsense, which is worse.

**The normalizer is the one the checkpoint was trained behind.** Scoring with
statistics fitted on different data silently shifts every input. The checkpoint
records the normalizer's identity -- version, pipeline and spec versions, and the
hash of the exact training candidates it was fitted on -- and the load refuses a
``normalizer.json`` (beside the checkpoint, or named with ``--normalizer`` /
``SLOTIFY_RANKER_NORMALIZER``) whose identity or column names differ.

**Weights only, eval mode, no grad.** ``torch.load(..., weights_only=True)``,
``model.eval()`` so dropout is off, ``torch.inference_mode()`` so nothing
accumulates a graph, and ``map_location="cpu"`` so a CUDA-trained checkpoint
loads on a laptop.

**The model is loaded once.** Constructing a predictor reads the checkpoint;
scoring reuses it. The Express API spawns one process per request today, so
this matters mostly for batch evaluation -- but it also means a persistent
service can be added without touching this file.

What a model was trained on is read from the checkpoint itself
(``label_source`` / ``data_provenance``). Checkpoints written before those keys
existed -- the committed bootstrap among them -- fall back to the
``training_summary.json`` of the same run beside them, and only when that
summary's ``run_id`` is the checkpoint's; otherwise the provenance is reported
as ``unknown``, never assumed to be human.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from slotify_rank.datasets.normalizer import FeatureNormalizer, read_normalizer
from slotify_rank.datasets.schema import (
    DatasetSchema,
    TrainingExample,
    project_handcrafted,
)
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
    quantize_scores,
)

__all__ = [
    "AlignedExamples",
    "EpisodeRanking",
    "RankerPredictor",
    "PredictorError",
    "align_examples",
    "rank_examples",
    "score_examples",
]


class PredictorError(RuntimeError):
    """The predictor could not be built or could not score."""


def _training_summary_beside(checkpoint_path: Path) -> Mapping[str, Any]:
    """Read the run's summary, if it sits next to the checkpoint.

    Used only for checkpoints that predate stored provenance. A missing or
    unreadable summary yields ``{}``, which the caller reports as ``unknown``
    rather than assuming real supervision -- an unlabelled provenance must never
    read as a human-labelled one.
    """
    summary_path = checkpoint_path.parent / "training_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _training_provenance(
    payload: Mapping[str, Any], checkpoint_path: Path
) -> tuple[str, str]:
    """``(label_source, data_provenance)`` for a checkpoint.

    The checkpoint's own record wins. Only a legacy checkpoint (no stored
    ``label_source``) consults the sidecar summary, and only a summary written by
    the same run: a ``training_summary.json`` that happens to sit beside a
    checkpoint copied from elsewhere describes some other model.
    """
    stored = payload.get("label_source")
    if isinstance(stored, str) and stored:
        provenance = payload.get("data_provenance")
        return stored, provenance if isinstance(provenance, str) and provenance else "unknown"

    summary = _training_summary_beside(checkpoint_path)
    if summary and str(summary.get("run_id", "")) == str(payload.get("run_id", "")):
        return (
            str(summary.get("label_source") or "unknown"),
            str(summary.get("data_provenance") or "unknown"),
        )
    return "unknown", "unknown"


@dataclass(frozen=True)
class AlignedExamples:
    """Examples re-columned onto a checkpoint's layout, and what that took."""

    examples: list[TrainingExample]
    #: True when the served handcrafted layout differed and was projected by name.
    projected: bool
    #: Served columns the checkpoint does not consume (withheld from the model).
    withheld_columns: tuple[str, ...]
    #: Version differences that were disclosed rather than refused.
    notes: tuple[str, ...]


def align_examples(
    examples: Sequence[TrainingExample],
    served: DatasetSchema,
    trained: DatasetSchema,
) -> AlignedExamples:
    """Check served data against a checkpoint's input schema; project by name.

    Refuses (``IncompatibleCheckpoint``) when the learned blocks differ in width,
    parts or input-schema version, when the served layout names a column twice,
    or when a column the checkpoint was trained on is absent. Projects a
    reordered or superset handcrafted layout onto the checkpoint's columns.
    Reports differing feature spec / pipeline versions as notes.
    """
    reasons: list[str] = []
    for field_name in ("native_embedding_dimension", "audio_dimension", "text_dimension"):
        mine, theirs = getattr(trained, field_name), getattr(served, field_name)
        if mine != theirs:
            reasons.append(f"{field_name}: served {theirs}, checkpoint {mine}")
    for field_name in ("audio_parts", "text_parts"):
        mine, theirs = getattr(trained, field_name), getattr(served, field_name)
        if tuple(mine) != tuple(theirs):
            reasons.append(f"{field_name}: served {list(theirs)}, checkpoint {list(mine)}")
    if (
        trained.input_schema_version
        and served.input_schema_version
        and trained.input_schema_version != served.input_schema_version
    ):
        reasons.append(
            f"input_schema_version: served {served.input_schema_version!r}, "
            f"checkpoint {trained.input_schema_version!r} (the learned blocks are "
            "constructed differently)"
        )

    served_names = tuple(served.handcrafted_feature_names)
    trained_names = tuple(trained.handcrafted_feature_names)
    duplicates = sorted({name for name in served_names if served_names.count(name) > 1})
    if duplicates:
        reasons.append(f"served handcrafted layout names {duplicates[:5]} more than once")
    available = set(served_names)
    missing = [name for name in trained_names if name not in available]
    if missing:
        reasons.append(
            f"{len(missing)} handcrafted column(s) the checkpoint was trained on are "
            f"absent from the served features: {missing[:5]}"
        )
    if reasons:
        raise IncompatibleCheckpoint(
            "The served features do not match the checkpoint's input schema:\n  - "
            + "\n  - ".join(reasons)
        )

    try:
        projected = project_handcrafted(examples, served_names, trained_names)
    except ValueError as error:
        raise IncompatibleCheckpoint(str(error)) from error

    trained_set = set(trained_names)
    withheld = tuple(name for name in served_names if name not in trained_set)

    notes: list[str] = []
    for field_name, label in (
        ("feature_spec_version", "feature spec"),
        ("feature_pipeline_version", "feature pipeline"),
    ):
        mine, theirs = getattr(trained, field_name), getattr(served, field_name)
        if mine and theirs and mine != theirs:
            notes.append(
                f"The features were built under {label} {theirs!r} but the model "
                f"was trained under {mine!r}. Columns were matched by name; a "
                "column whose computation changed between the two would be scored "
                "as if it had not."
            )
    return AlignedExamples(
        examples=projected,
        projected=served_names != trained_names,
        withheld_columns=withheld,
        notes=tuple(notes),
    )


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

        # Refuse an internally inconsistent checkpoint, or a normalizer that is
        # not the one it was trained behind, before any weight is loaded. This
        # compares the checkpoint with itself and with its normalizer; the data
        # being scored is compared later, in `align_examples`. Split hashes are
        # allowed to differ: scoring a new upload is the whole point.
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
        try:
            model.load_state_dict(state, strict=True)  # type: ignore[arg-type]
        except RuntimeError as error:
            # The checkpoint's declared schema and its actual weights disagree.
            # torch reports this as a bare RuntimeError about tensor shapes;
            # translated here so a caller sees the same failure type as every
            # other kind of incompatible checkpoint.
            raise IncompatibleCheckpoint(
                f"{checkpoint}: the stored weights do not match the architecture "
                f"the checkpoint describes. {error}"
            ) from error
        model.to(torch.device(device))
        model.eval()

        label_source, provenance = _training_provenance(payload, checkpoint)
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
            training_label_source=label_source,
            training_data_provenance=provenance,
        )

        return cls(model=model, normalizer=normalizer, schema=schema, identity=identity)

    def align(
        self, examples: Sequence[TrainingExample], served_schema: DatasetSchema | None
    ) -> AlignedExamples:
        """Check and project served examples onto this checkpoint's columns.

        ``served_schema`` is the layout the examples were built under
        (``LoadedDataset.schema``). ``None`` is refused: without it, the only
        thing left to check is the vector width, and equal widths are exactly
        the case in which mis-columned inputs go unnoticed.
        """
        if served_schema is None:
            raise IncompatibleCheckpoint(
                "The served examples carry no schema, so their column names cannot "
                "be compared with the checkpoint's. Refusing to score by width alone."
            )
        return align_examples(examples, served_schema, self.schema)

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
    """Score and rank examples already in the checkpoint's layout, best first.

    Scores are quantized once (:func:`quantize_scores`, the 6 decimals
    ``raw_score`` reports) *before* sorting and normalizing. A float32 forward
    pass does not give two identical candidates bit-identical outputs -- two
    copies of one vector in one batch have been observed 7e-9 apart -- and
    sorting or min-max scaling the unquantized values would let that noise
    decide the order and stretch it into a 0-vs-100 ``normalized_score``. After
    quantization, ordering, normalization and the reported ``raw_score`` all
    describe the same numbers, and ties break on the earlier timestamp then the
    candidate id, so two runs over the same audio produce byte-identical output.

    Use :func:`rank_examples` for served data: it checks the served schema first.
    """
    ranking, acceptability = predictor.score(examples)
    quantized = quantize_scores(ranking)
    normalized = normalize_scores(quantized)

    ordered = sorted(
        range(len(examples)),
        key=lambda index: (
            -quantized[index],
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
                raw_score=quantized[index],
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


@dataclass(frozen=True)
class EpisodeRanking:
    """A ranked episode plus what the response must disclose about it."""

    ranked: list[ScoredCandidate]
    warnings: list[str]
    withheld_columns: tuple[str, ...] = ()


def rank_examples(
    predictor: RankerPredictor,
    examples: Sequence[TrainingExample],
    served_schema: DatasetSchema | None,
    timestamps_ms: Mapping[str, int],
    feature_status: Mapping[str, str],
) -> EpisodeRanking:
    """The product path: check the served schema, project, then score and rank."""
    if not examples:
        return EpisodeRanking(ranked=[], warnings=[])
    aligned = predictor.align(examples, served_schema)
    ranked = score_examples(predictor, aligned.examples, timestamps_ms, feature_status)
    return EpisodeRanking(
        ranked=ranked,
        warnings=list(aligned.notes),
        withheld_columns=aligned.withheld_columns,
    )
