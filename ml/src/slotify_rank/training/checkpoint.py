"""Versioned, safe, resumable checkpoints.

**No arbitrary Python objects.** Everything saved here is a ``state_dict``, a
tensor, or a JSON-primitive, and every load passes ``weights_only=True``. A
checkpoint format that has to be *trusted* before it can be read is a remote
code execution primitive wearing a ``.pt`` extension, and model files get
emailed around. Pickling the model object would also silently bind every
checkpoint to the exact class layout that wrote it.

**Loads are checked, not hopeful.** A checkpoint carries the model variant, the
input schema, the feature ordering, the pipeline versions, the normalizer
identity and the split hash. :func:`assert_compatible` refuses a mismatch rather
than letting a 110-column model be handed 96 columns of differently-ordered
features -- which does not crash, and produces confident nonsense.

**Writes are atomic, on OneDrive too.** ``torch.save`` streams, so a crash
mid-write leaves a truncated file that often still loads. Saves go to a
temporary file in the same directory and are renamed through the Phase 2
Windows-safe retry helper, which outlasts the transient sharing violations a
sync client causes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from slotify_rank.config.versions import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
)
from slotify_rank.data.checksum import _replace_with_retry
from slotify_rank.datasets.normalizer import FeatureNormalizer
from slotify_rank.datasets.schema import DatasetSchema

__all__ = [
    "CheckpointError",
    "IncompatibleCheckpoint",
    "atomic_torch_save",
    "build_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "assert_compatible",
    "inspect_checkpoint",
    "normalizer_reference",
]


class CheckpointError(ValueError):
    """A checkpoint file is missing, truncated or unreadable."""


class IncompatibleCheckpoint(CheckpointError):
    """A checkpoint is readable but describes a different model or dataset."""


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """``torch.save`` via a temp file in the same directory, then rename.

    Deliberately mirrors :func:`slotify_rank.data.checksum.atomic_write_bytes`
    rather than calling it: ``torch.save`` wants a file object, not bytes. The
    rename goes through the same retry helper, so checkpoints get exactly the
    OneDrive-safe behaviour every other artifact in this repository has.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def normalizer_reference(normalizer: FeatureNormalizer) -> dict[str, Any]:
    """The normalizer's identity, not its statistics.

    The statistics live in ``normalizer.json`` next to the checkpoint; what the
    checkpoint needs is enough to detect that a *different* normalizer is being
    applied at inference time.
    """
    return {
        "normalizer_version": normalizer.normalizer_version,
        "training_split_hash": normalizer.training_split_hash,
        "feature_pipeline_version": normalizer.feature_pipeline_version,
        "feature_spec_version": normalizer.feature_spec_version,
        "fit_candidate_count": normalizer.fit_candidate_count,
        "fit_episode_count": normalizer.fit_episode_count,
        "epsilon": normalizer.epsilon,
    }


@dataclass(frozen=True)
class RngState:
    """Generator states, so a resumed run draws the same batches."""

    torch_state: torch.Tensor
    loader_state: torch.Tensor

    def to_payload(self) -> dict[str, Any]:
        return {"torch": self.torch_state, "loader": self.loader_state}


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_metric: float | None,
    best_epoch: int,
    training_config: Mapping[str, Any],
    training_config_hash: str,
    model_config: Mapping[str, Any],
    model_variant: str,
    schema: DatasetSchema,
    normalizer: FeatureNormalizer,
    dataset_split_hash: str,
    label_manifest_hash: str,
    random_seed: int,
    resolved_device: str,
    resolved_dtype: str,
    dependency_versions: Mapping[str, str],
    git_commit: str,
    run_id: str,
    early_stopping_state: Mapping[str, Any],
    rng_state: RngState | None = None,
) -> dict[str, Any]:
    """Assemble the payload. Every field is a primitive or a state_dict."""
    payload: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_metric": (
            float(best_validation_metric) if best_validation_metric is not None else None
        ),
        "best_epoch": int(best_epoch),
        "training_config": json.dumps(dict(training_config), sort_keys=True),
        "training_config_hash": training_config_hash,
        "model_config": json.dumps(dict(model_config), sort_keys=True),
        "model_variant": model_variant,
        "input_schema": json.dumps(schema.to_dict(), sort_keys=True),
        "feature_names": list(schema.handcrafted_feature_names),
        "feature_pipeline_version": schema.feature_pipeline_version,
        "normalizer_reference": json.dumps(
            normalizer_reference(normalizer), sort_keys=True
        ),
        "dataset_split_hash": dataset_split_hash,
        "label_manifest_hash": label_manifest_hash,
        "random_seed": int(random_seed),
        "resolved_device": resolved_device,
        "resolved_dtype": resolved_dtype,
        "resolved_dependency_versions": json.dumps(
            dict(dependency_versions), sort_keys=True
        ),
        "git_commit": git_commit,
        "run_id": run_id,
        "early_stopping_state": json.dumps(dict(early_stopping_state), sort_keys=True),
    }
    if rng_state is not None:
        payload["rng_state"] = rng_state.to_payload()
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_torch_save(Path(path), payload)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint with ``weights_only=True`` and decode its metadata."""
    file_path = Path(path)
    if not file_path.is_file():
        raise CheckpointError(f"Checkpoint not found: {file_path}")
    try:
        payload = torch.load(file_path, map_location="cpu", weights_only=True)
    except Exception as error:  # torch raises a variety of types on corruption
        raise CheckpointError(
            f"{file_path} could not be read as a checkpoint: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointError(
            f"{file_path} does not contain a checkpoint mapping (got "
            f"{type(payload).__name__})"
        )

    version = str(payload.get("checkpoint_schema_version", ""))
    if version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise IncompatibleCheckpoint(
            f"Unsupported checkpoint_schema_version {version!r}; this build reads "
            f"{sorted(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}"
        )
    for required in ("model_state_dict", "model_variant", "input_schema"):
        if required not in payload:
            raise CheckpointError(f"{file_path} is missing {required!r}")

    # JSON-encoded metadata is decoded here so callers never touch the raw form.
    for key in (
        "training_config",
        "model_config",
        "input_schema",
        "normalizer_reference",
        "resolved_dependency_versions",
        "early_stopping_state",
    ):
        raw = payload.get(key)
        if isinstance(raw, str):
            try:
                payload[key] = json.loads(raw)
            except json.JSONDecodeError as error:
                raise CheckpointError(
                    f"{file_path}: {key} is not valid JSON: {error}"
                ) from error
    return payload


def assert_compatible(
    payload: Mapping[str, Any],
    model_variant: str,
    schema: DatasetSchema,
    normalizer: FeatureNormalizer | None = None,
    dataset_split_hash: str | None = None,
    allow_split_mismatch: bool = False,
) -> None:
    """Refuse a checkpoint that does not describe this model and dataset.

    ``allow_split_mismatch`` exists for inference-only use: scoring a new corpus
    with a trained model is legitimate, and its split hash will never match. It
    does *not* relax the schema checks, because those govern whether the weights
    line up with the columns at all.
    """
    reasons: list[str] = []

    stored_variant = str(payload.get("model_variant", ""))
    if stored_variant != model_variant:
        reasons.append(
            f"checkpoint holds variant {stored_variant!r}, not {model_variant!r}"
        )

    stored_schema_raw = payload.get("input_schema")
    if isinstance(stored_schema_raw, Mapping):
        stored_schema = DatasetSchema.from_mapping(stored_schema_raw)
        reasons.extend(schema.incompatibilities(stored_schema))
    else:
        reasons.append("checkpoint records no input schema")

    stored_names = list(payload.get("feature_names", ()))
    if stored_names and list(schema.handcrafted_feature_names) != stored_names:
        reasons.append(
            "handcrafted feature ordering differs from the checkpoint's "
            f"({len(stored_names)} stored names)"
        )

    if normalizer is not None:
        reference = payload.get("normalizer_reference")
        if isinstance(reference, Mapping):
            stored_version = str(reference.get("normalizer_version", ""))
            if stored_version and stored_version != normalizer.normalizer_version:
                reasons.append(
                    f"normalizer version {stored_version!r} != "
                    f"{normalizer.normalizer_version!r}"
                )
            stored_pipeline = str(reference.get("feature_pipeline_version", ""))
            if (
                stored_pipeline
                and normalizer.feature_pipeline_version
                and stored_pipeline != normalizer.feature_pipeline_version
            ):
                reasons.append(
                    f"normalizer feature_pipeline_version {stored_pipeline!r} != "
                    f"{normalizer.feature_pipeline_version!r}"
                )

    if dataset_split_hash is not None and not allow_split_mismatch:
        stored_split = str(payload.get("dataset_split_hash", ""))
        if stored_split and stored_split != dataset_split_hash:
            reasons.append(
                f"dataset_split_hash {stored_split!r} != {dataset_split_hash!r}. "
                "Pass allow_split_mismatch for inference-only use on a new corpus."
            )

    if reasons:
        raise IncompatibleCheckpoint(
            "Refusing to load an incompatible checkpoint:\n  - "
            + "\n  - ".join(reasons)
        )


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Human-readable summary, without instantiating a model."""
    payload = load_checkpoint(Path(path))
    state = payload.get("model_state_dict", {})
    parameter_count = sum(
        int(tensor.numel()) for tensor in state.values() if hasattr(tensor, "numel")
    )
    schema = payload.get("input_schema") or {}
    return {
        "path": str(path),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
        "model_variant": payload.get("model_variant"),
        "run_id": payload.get("run_id"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "best_epoch": payload.get("best_epoch"),
        "best_validation_metric": payload.get("best_validation_metric"),
        "parameter_count": parameter_count,
        "state_dict_entries": len(state),
        "training_config_hash": payload.get("training_config_hash"),
        "dataset_split_hash": payload.get("dataset_split_hash"),
        "label_manifest_hash": payload.get("label_manifest_hash"),
        "random_seed": payload.get("random_seed"),
        "resolved_device": payload.get("resolved_device"),
        "resolved_dtype": payload.get("resolved_dtype"),
        "git_commit": payload.get("git_commit"),
        "feature_pipeline_version": payload.get("feature_pipeline_version"),
        "handcrafted_dimension": schema.get("handcrafted_dimension"),
        "audio_dimension": schema.get("audio_dimension"),
        "text_dimension": schema.get("text_dimension"),
        "normalizer_reference": payload.get("normalizer_reference"),
        "resolved_dependency_versions": payload.get("resolved_dependency_versions"),
    }
