"""Training run artifacts.

Everything a run produces lands under ``artifacts/training/<run_id>/``:

``resolved_config.json``  the configuration actually used, including every override
``environment.json``      device, dtype, threads, dependency versions
``dataset_summary.json``  eligibility accounting, splits, pair statistics
``normalizer.json``       the fitted train-only statistics
``epoch_metrics.jsonl``   one line per epoch, appended as training proceeds
``training_summary.json`` the machine-readable result
``training_summary.md``   the same result, readable
``best_checkpoint.pt`` / ``last_checkpoint.pt``

Every JSON write goes through the Phase 2 atomic writer, so a crashed or
sync-interrupted run leaves either the previous file or the new one, never half
of either. ``epoch_metrics.jsonl`` is the one file written incrementally -- it
has to survive an interruption *with* its partial contents, which is exactly
what a whole-file atomic rewrite would defeat -- so it is appended and each line
is flushed.

Smoke provenance is carried explicitly. When a run trains on synthetic fixtures
the summary says so in its first field and in its title, because a number
labelled "NDCG@3: 0.94" with no provenance is exactly the kind of thing that
ends up quoted somewhere it should not be.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION, TRAINING_DATASET_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "RunArtifacts",
    "write_json",
    "append_epoch_metrics",
    "write_training_summary",
    "SYNTHETIC_WARNING",
]

SYNTHETIC_WARNING = (
    "SYNTHETIC SMOKE RUN. This model was trained on generated fixtures, not on "
    "real audio or human labels. The metrics below demonstrate that the training "
    "system works end to end. They are NOT evidence of model quality and must "
    "never be quoted as a result."
)

SMOKE_WARNING = (
    "SMOKE RUN. Data limits and/or a reduced epoch budget were applied (see "
    "resolved_config.json -> overrides). These metrics show that the pipeline "
    "runs; they are not a model-quality measurement."
)


@dataclass(frozen=True)
class RunArtifacts:
    """Paths for one run. Creating it creates the directory."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def resolved_config(self) -> Path:
        return self.directory / "resolved_config.json"

    @property
    def environment(self) -> Path:
        return self.directory / "environment.json"

    @property
    def dataset_summary(self) -> Path:
        return self.directory / "dataset_summary.json"

    @property
    def normalizer(self) -> Path:
        return self.directory / "normalizer.json"

    @property
    def epoch_metrics(self) -> Path:
        return self.directory / "epoch_metrics.jsonl"

    @property
    def pairs(self) -> Path:
        return self.directory / "pairs.jsonl"

    @property
    def best_checkpoint(self) -> Path:
        return self.directory / "best_checkpoint.pt"

    @property
    def last_checkpoint(self) -> Path:
        return self.directory / "last_checkpoint.pt"

    @property
    def summary_json(self) -> Path:
        return self.directory / "training_summary.json"

    @property
    def summary_markdown(self) -> Path:
        return self.directory / "training_summary.md"

    def sizes(self) -> dict[str, int]:
        return {
            path.name: path.stat().st_size
            for path in sorted(self.directory.iterdir())
            if path.is_file()
        }


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write, matching every other artifact in the repository."""
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=_encode) + "\n"
    atomic_write_bytes(Path(path), body.encode("utf-8"))


def _encode(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def append_epoch_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one epoch's line, flushed immediately.

    Appended rather than atomically rewritten on purpose: this file's value is
    that it survives an interruption *with the epochs that completed*, which a
    whole-file replace would lose.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_encode)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def write_pairs(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_bytes(Path(path), body.encode("utf-8"))


def build_summary(
    run_id: str,
    result: Any,
    dataset_summary: Mapping[str, Any],
    model_description: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifacts: RunArtifacts,
    synthetic: bool,
    smoke: bool,
    overrides: Mapping[str, Any],
    label_source: str = "human",
) -> dict[str, Any]:
    """The machine-readable result. No metric is invented here."""
    best = dict(result.best_metrics or {})
    provenance = (
        "synthetic_fixture" if synthetic else ("smoke_limited" if smoke else "real")
    )
    summary: dict[str, Any] = {
        "evidence_class": (
            "synthetic smoke run - not a model-quality result"
            if synthetic
            else "limited smoke run - not a model-quality result"
            if smoke
            else "measured on real labelled data"
        ),
        "data_provenance": provenance,
        "label_source": label_source,
        "synthetic_data": synthetic,
        "smoke": smoke,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": PACKAGE_VERSION,
        "training_dataset_version": TRAINING_DATASET_VERSION,
        "model_variant": model_description.get("variant"),
        "experiment_name": model_description.get("experiment_name"),
        "model_parameter_count": model_description.get("trainable_parameters"),
        "training_episode_count": dataset_summary.get("training_episode_count"),
        "training_candidate_count": dataset_summary.get("training_candidate_count"),
        "training_pair_count": dataset_summary.get("training_pair_count"),
        "validation_episode_count": dataset_summary.get("validation_episode_count"),
        "validation_candidate_count": dataset_summary.get("validation_candidate_count"),
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "best_validation_ndcg_at_3": result.best_validation_metric,
        "best_validation_pairwise_accuracy": best.get("pairwise_accuracy"),
        "best_validation_classification_f1": best.get("classification_f1"),
        "best_validation_metrics": best,
        "runtime_seconds": round(result.seconds, 3),
        "device": result.device,
        "resolved_dtype": result.dtype,
        "stop_reason": result.stop_reason,
        "interrupted": result.interrupted,
        "overrides": dict(overrides),
        "artifacts": {
            "directory": str(artifacts.directory),
            "best_checkpoint": result.best_checkpoint_path,
            "last_checkpoint": result.last_checkpoint_path,
            "epoch_metrics": str(artifacts.epoch_metrics),
            "dataset_summary": str(artifacts.dataset_summary),
            "normalizer": str(artifacts.normalizer),
            "resolved_config": str(artifacts.resolved_config),
        },
        "environment": dict(environment),
    }
    if synthetic:
        summary["warning"] = SYNTHETIC_WARNING
    elif smoke:
        summary["warning"] = SMOKE_WARNING
    return summary


def _format(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_training_summary(artifacts: RunArtifacts, summary: Mapping[str, Any]) -> None:
    """Write both the JSON and the Markdown rendering."""
    write_json(artifacts.summary_json, summary)

    warning = summary.get("warning")
    lines = [f"# Training run `{summary['run_id']}`", ""]
    if warning:
        lines += [f"> **{warning}**", ""]
    lines += [
        f"- **Evidence class**: {summary['evidence_class']}",
        f"- **Data provenance**: {summary['data_provenance']}"
        f" (label source: {summary['label_source']})",
        f"- **Generated**: {summary['generated_at']}",
        "",
        "## Model",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| Variant | `{summary['model_variant']}` |",
        f"| Experiment | `{summary['experiment_name']}` |",
        f"| Trainable parameters | {summary['model_parameter_count']:,} |",
        "",
        "## Dataset",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Training episodes | {summary['training_episode_count']} |",
        f"| Training candidates | {summary['training_candidate_count']} |",
        f"| Training pairs | {summary['training_pair_count']} |",
        f"| Validation episodes | {summary['validation_episode_count']} |",
        f"| Validation candidates | {summary['validation_candidate_count']} |",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Best epoch | {summary['best_epoch']} |",
        f"| Epochs run | {summary['epochs_run']} |",
        f"| Validation NDCG@3 | {_format(summary['best_validation_ndcg_at_3'])} |",
        f"| Validation pairwise accuracy | "
        f"{_format(summary['best_validation_pairwise_accuracy'])} |",
        f"| Validation classification F1 | "
        f"{_format(summary['best_validation_classification_f1'])} |",
        f"| Runtime (s) | {_format(summary['runtime_seconds'])} |",
        f"| Device | `{summary['device']}` ({summary['resolved_dtype']}) |",
        f"| Stop reason | {summary['stop_reason']} |",
        "",
    ]

    overrides = summary.get("overrides") or {}
    lines += ["## Resolved overrides", ""]
    if overrides:
        lines += ["| Setting | From | To |", "| --- | --- | --- |"]
        for name, change in sorted(overrides.items()):
            if isinstance(change, Mapping):
                lines.append(f"| `{name}` | {change.get('from')} | {change.get('to')} |")
            else:
                lines.append(f"| `{name}` | | {change} |")
    else:
        lines.append("None; the configuration file was used unchanged.")
    lines += [
        "",
        "## Artifacts",
        "",
        *(f"- `{name}`" for name in sorted(summary.get("artifacts", {}))),
        "",
    ]
    atomic_write_bytes(
        artifacts.summary_markdown, ("\n".join(lines)).encode("utf-8")
    )
