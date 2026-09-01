"""The canonical experiment: its committed definition and its resolved manifest.

``ml/configs/experiment_v2.yaml`` says what the experiment *is*. This
module reads it, checks that the repository actually matches what it claims, and
resolves it into a manifest that pins the *bytes*: the split manifest's hash, the
candidate manifest's hash, the label snapshot's hash, the feature schema
versions, the model configuration and the baseline configuration.

The distinction matters. A config file says "split v3, seed 42". A manifest says
"split v3, seed 42, and the manifest that produced it hashes to a1b2c3...". Only
the second one makes a published number checkable, because only the second one
changes when someone regenerates the split with different episodes.

What this module refuses to do:

* **Resolve a manifest whose split disagrees with the config.** If the config
  says ``group_by: series`` and the on-disk split manifest says ``episode``, the
  experiment is not the experiment that was declared, and pretending otherwise
  would put a leakage-safe label on a leaky split.
* **Resolve a manifest over a degraded split.** A split that assigned everything
  to ``development`` cannot support a held-out claim.
* **Silently accept fewer labels than the gate.** The label count is recorded and
  compared against ``labels.minimum_human_labels``; a shortfall is a blocking
  reason on the manifest, not a footnote.

Nothing here trains, evaluates or touches the test split's contents. It reads
metadata and hashes files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import (
    FEATURE_PIPELINE_VERSION,
    FEATURE_SPEC_VERSION,
    PACKAGE_VERSION,
    SPLIT_ALGORITHM_VERSION,
    SUPPORTED_SPLIT_ALGORITHM_VERSIONS,
)
from slotify_rank.data.checksum import atomic_write_bytes, sha256_file, sha256_text

__all__ = [
    "EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "ExperimentConfigError",
    "ExperimentConfig",
    "ExperimentManifest",
    "load_experiment_config",
    "resolve_manifest",
    "write_manifest",
    "read_manifest",
]

EXPERIMENT_CONFIG_SCHEMA_VERSION = "experiment-config-v1.0.0"


class ExperimentConfigError(ValueError):
    """The experiment definition is malformed or contradicts the repository."""


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"'{name}' must be a mapping")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """The committed definition, parsed and validated."""

    experiment_version: str
    schema_version: str
    corpus: Mapping[str, Any]
    split: Mapping[str, Any]
    labels: Mapping[str, Any]
    model: Mapping[str, Any]
    baselines: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    claim: Mapping[str, Any]
    source_path: str = ""

    @property
    def minimum_human_labels(self) -> int:
        return int(self.labels["minimum_human_labels"])

    @property
    def minimum_relative_improvement_percent(self) -> float:
        return float(self.claim["minimum_relative_improvement_percent"])

    @property
    def canonical_baseline(self) -> str:
        return str(self.baselines["canonical"])

    @property
    def split_version(self) -> str:
        return str(self.split["version"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.model["seeds"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "corpus": dict(self.corpus),
            "split": dict(self.split),
            "labels": dict(self.labels),
            "model": dict(self.model),
            "baselines": dict(self.baselines),
            "evaluation": dict(self.evaluation),
            "claim": dict(self.claim),
        }

    def digest(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k != "source_path"}
        return sha256_text(json.dumps(payload, sort_keys=True))


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate ``ml/configs/experiment_*.yaml``."""
    import yaml

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExperimentConfigError(
            f"Experiment definition not found: {config_path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ExperimentConfigError(f"{config_path} must contain a YAML mapping")

    schema_version = str(raw.get("schema_version") or "")
    if schema_version != EXPERIMENT_CONFIG_SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"{config_path}: schema_version is {schema_version!r}, but this build "
            f"reads {EXPERIMENT_CONFIG_SCHEMA_VERSION!r}"
        )
    experiment_version = str(raw.get("experiment_version") or "")
    if not experiment_version:
        raise ExperimentConfigError(f"{config_path}: 'experiment_version' is required")

    config = ExperimentConfig(
        experiment_version=experiment_version,
        schema_version=schema_version,
        corpus=_section(raw, "corpus"),
        split=_section(raw, "split"),
        labels=_section(raw, "labels"),
        model=_section(raw, "model"),
        baselines=_section(raw, "baselines"),
        evaluation=_section(raw, "evaluation"),
        claim=_section(raw, "claim"),
        source_path=config_path.as_posix(),
    )

    if "human" not in list(config.labels.get("allowed_label_sources", ())):
        raise ExperimentConfigError(
            f"{config_path}: labels.allowed_label_sources must include 'human'"
        )
    if [
        source
        for source in config.labels.get("allowed_label_sources", ())
        if source != "human"
    ]:
        raise ExperimentConfigError(
            f"{config_path}: labels.allowed_label_sources may only be ['human'] "
            "for the benchmark experiment. A weak-label run is a different "
            "experiment and must carry a different experiment_version."
        )
    headline = str(config.model.get("headline_variant") or "")
    if headline not in list(config.model.get("ablations", ())):
        raise ExperimentConfigError(
            f"{config_path}: model.headline_variant {headline!r} is not among the "
            f"declared ablations {list(config.model.get('ablations', ()))}"
        )
    if not config.seeds:
        raise ExperimentConfigError(f"{config_path}: model.seeds must not be empty")
    ratios = config.split.get("ratios") or {}
    total = sum(float(value) for value in ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ExperimentConfigError(
            f"{config_path}: split.ratios must sum to 1.0, got {total}"
        )
    return config


@dataclass
class ExperimentManifest:
    """The resolved experiment: definition plus the hashes of what it ran on."""

    experiment_version: str
    config_digest: str
    config: Mapping[str, Any]
    resolved_at: str
    artifact_hashes: Mapping[str, str | None]
    split_summary: Mapping[str, Any]
    label_summary: Mapping[str, Any]
    feature_schema: Mapping[str, Any]
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        return not self.blocking_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "experiment_version": self.experiment_version,
            "config_digest": self.config_digest,
            "resolved_at": self.resolved_at,
            "ready": self.ready,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "artifact_hashes": dict(self.artifact_hashes),
            "split": dict(self.split_summary),
            "labels": dict(self.label_summary),
            "feature_schema": dict(self.feature_schema),
            "config": dict(self.config),
        }


def _hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if Path(path).is_file() else None


def resolve_manifest(
    config: ExperimentConfig,
    repo_root: Path,
    paths: Any,
    human_label_count: int | None = None,
) -> ExperimentManifest:
    """Pin the experiment to the bytes currently on disk, and check them."""
    from datetime import datetime, timezone

    from slotify_rank.data.splits import read_split_manifest

    root = Path(repo_root)
    blocking: list[str] = []
    warnings: list[str] = []

    split_path = paths.split_manifest(config.split_version)
    split_summary: dict[str, Any] = {"manifest_path": str(split_path)}
    if not split_path.is_file():
        blocking.append(
            f"split manifest {split_path} does not exist; run `dataset split "
            f"--config {config.split['config']}` first"
        )
    else:
        manifest = read_split_manifest(split_path)
        split_summary.update(
            {
                "version": manifest.version,
                "algorithm_version": manifest.algorithm_version,
                "seed": manifest.seed,
                "group_by": manifest.group_by,
                "ratios": dict(manifest.ratios),
                "degraded": manifest.degraded,
                "group_count": len(manifest.assignments),
                "groups_by_split": {
                    split: sorted(
                        a.group_id for a in manifest.assignments if a.split == split
                    )
                    for split in sorted({a.split for a in manifest.assignments})
                },
            }
        )
        if manifest.degraded:
            blocking.append(
                "the split manifest is degraded (everything in 'development'); a "
                "held-out claim is not possible from it"
            )
        if manifest.group_by != str(config.split.get("group_by")):
            blocking.append(
                f"the experiment declares group_by={config.split.get('group_by')!r} "
                f"but the split manifest was built with {manifest.group_by!r}"
            )
        if int(manifest.seed) != int(config.split.get("seed", manifest.seed)):
            blocking.append(
                f"the experiment declares split seed {config.split.get('seed')} but "
                f"the manifest records {manifest.seed}"
            )
        if manifest.algorithm_version not in SUPPORTED_SPLIT_ALGORITHM_VERSIONS:
            blocking.append(
                f"the split manifest was produced by {manifest.algorithm_version!r}, "
                f"but this build implements "
                f"{list(SUPPORTED_SPLIT_ALGORITHM_VERSIONS)}"
            )
        declared_algorithm = config.split.get("algorithm_version")
        if declared_algorithm and manifest.algorithm_version != declared_algorithm:
            blocking.append(
                f"the experiment declares split algorithm {declared_algorithm!r} "
                f"but the manifest records {manifest.algorithm_version!r}"
            )

    snapshot_path = (
        paths.labels_dir / f"label_snapshot_{config.labels['snapshot_version']}.json"
    )
    label_export = (
        paths.labels_dir / f"labels_{config.labels['snapshot_version']}.jsonl"
    )
    label_summary: dict[str, Any] = {
        "snapshot_version": config.labels["snapshot_version"],
        "snapshot_path": str(snapshot_path),
        "export_path": str(label_export),
        "minimum_human_labels": config.minimum_human_labels,
        "human_label_count": human_label_count,
        "allowed_label_sources": list(config.labels["allowed_label_sources"]),
    }
    if human_label_count is None:
        blocking.append(
            "the human label count was not supplied, so the label gate could not "
            "be evaluated"
        )
    elif human_label_count < config.minimum_human_labels:
        blocking.append(
            f"{human_label_count} human-labelled candidate(s) exist, fewer than the "
            f"{config.minimum_human_labels} this experiment requires"
        )
    if not snapshot_path.is_file():
        warnings.append(
            f"no frozen label snapshot at {snapshot_path}; freeze one with "
            "`experiment freeze` before evaluating"
        )

    queue_path = root / str(config.labels["queue_artifact"])
    if not queue_path.is_file():
        queue_path = Path(str(config.labels["queue_artifact"]))

    artifact_hashes: dict[str, str | None] = {
        "experiment_config": _hash_or_none(root / config.source_path),
        "corpus_plan": _hash_or_none(root / str(config.corpus["plan"])),
        "split_config": _hash_or_none(root / str(config.split["config"])),
        "split_manifest": _hash_or_none(split_path),
        "candidate_manifest": _hash_or_none(paths.candidates_manifest),
        "episode_manifest": _hash_or_none(paths.episodes_manifest),
        "feature_manifest": _hash_or_none(paths.features_manifest),
        "labelling_queue_config": _hash_or_none(
            root / str(config.labels["queue_config"])
        ),
        "labelling_queue": _hash_or_none(queue_path),
        "label_export": _hash_or_none(label_export),
        "label_snapshot": _hash_or_none(snapshot_path),
        "training_config": _hash_or_none(root / str(config.model["training_config"])),
        "baseline_config": _hash_or_none(
            root / "config" / "heuristic_offline_v1.json"
        ),
    }
    for registry in config.corpus["source_registries"]:
        artifact_hashes[f"source_registry:{Path(str(registry)).name}"] = _hash_or_none(
            root / str(registry)
        )
    for variant in config.model["ablations"]:
        artifact_hashes[f"model_config:{variant}"] = _hash_or_none(
            root / str(config.model["config_dir"]) / f"{variant}_v1.yaml"
        )

    feature_schema = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
    }
    if paths.features_manifest.is_file():
        from slotify_rank.features.assemble import read_feature_manifest

        header, records = read_feature_manifest(paths.features_manifest)
        feature_schema.update(
            {
                "manifest_feature_spec_version": header.get("feature_spec_version"),
                "manifest_feature_pipeline_version": header.get(
                    "feature_pipeline_version"
                ),
                "handcrafted_dimension": header.get("handcrafted_dimension"),
                "record_count": len(records),
            }
        )
    else:
        blocking.append(
            f"no feature manifest at {paths.features_manifest}; run the feature "
            "pipeline before resolving the experiment"
        )

    return ExperimentManifest(
        experiment_version=config.experiment_version,
        config_digest=config.digest(),
        config=config.to_dict(),
        resolved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        artifact_hashes=artifact_hashes,
        split_summary=split_summary,
        label_summary=label_summary,
        feature_schema=feature_schema,
        blocking_reasons=tuple(blocking),
        warnings=tuple(warnings),
    )


def write_manifest(path: Path, manifest: ExperimentManifest) -> None:
    atomic_write_bytes(
        Path(path),
        (json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
