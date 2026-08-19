"""The generated model-evidence report.

One job: read what the repository has actually produced, and say for each
capability what the evidence currently establishes. Not one number in the output
is typed. Every field is read from a generated artifact, and a field whose
artifact does not exist reports :data:`NOT_AVAILABLE` -- never ``0``, never
``null``, never a plausible-looking placeholder.

The distinction that matters most:

    0 human labels     is a measurement. It means someone counted, and the
                       answer was zero.
    NOT YET AVAILABLE  means nothing produced that number at all.

Rendering both as ``0`` would let a reader conclude that a metric was measured
and came out badly, when in fact it was never measured. So the two are
different strings and stay different all the way to the markdown.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "NOT_AVAILABLE",
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CapabilityEvidence",
    "ModelEvidence",
    "collect_evidence",
    "render_markdown",
    "write_evidence",
]

EVIDENCE_SCHEMA_VERSION = "model-evidence-v1.0.0"

#: The one string used for "nothing produced this". Compared against by tests so
#: it cannot drift into something that reads like a value.
NOT_AVAILABLE = "NOT YET AVAILABLE"

#: Evidence statuses, in the vocabulary the report uses everywhere.
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY SUPPORTED"
UNSUPPORTED = "NOT YET SUPPORTED"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return NOT_AVAILABLE
    sha = result.stdout.strip()
    return sha or NOT_AVAILABLE


@dataclass
class CapabilityEvidence:
    """What the generated artifacts establish about one capability."""

    capability_id: str
    capability: str
    requirement: str
    status: str
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "capability": self.capability,
            "requirement": self.requirement,
            "status": self.status,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


@dataclass
class ModelEvidence:
    git_sha: str
    generated_at: str
    measurements: dict[str, Any]
    capabilities: list[CapabilityEvidence]
    artifacts_read: dict[str, str]
    artifacts_missing: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "generated_at": self.generated_at,
            "git_sha": self.git_sha,
            "unavailable_marker": NOT_AVAILABLE,
            "measurements": dict(self.measurements),
            "capabilities": [
                capability.to_dict() for capability in self.capabilities
            ],
            "artifacts_read": dict(self.artifacts_read),
            "artifacts_missing": list(self.artifacts_missing),
        }


def _best_training_run(training_dir: Path) -> tuple[str, dict[str, Any]] | None:
    """The most recent run summary, whatever its provenance.

    Chosen by generated_at rather than by metric: picking the best-scoring run
    would be seed cherry-picking, which is precisely what this report exists to
    make impossible.
    """
    candidates: list[tuple[str, dict[str, Any]]] = []
    if not training_dir.is_dir():
        return None
    for summary_path in sorted(training_dir.glob("*/training_summary.json")):
        summary = _read_json(summary_path)
        if summary:
            candidates.append((str(summary_path), summary))
    if not candidates:
        return None
    candidates.sort(key=lambda entry: str(entry[1].get("generated_at", "")))
    return candidates[-1]


def _publishable_comparison(evaluation_dir: Path) -> tuple[str, dict[str, Any]] | None:
    """The most recent comparison whose headline is publishable, if any.

    A comparison that ran but was blocked is deliberately not returned: the
    headline metric fields must stay NOT YET AVAILABLE until one passes.
    """
    if not evaluation_dir.is_dir():
        return None
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(evaluation_dir.glob("*/comparison.json")):
        payload = _read_json(path)
        if payload and payload.get("headline_publishable") is True:
            found.append((str(path), payload))
    if not found:
        return None
    found.sort(key=lambda entry: str(entry[1].get("generated_at", "")))
    return found[-1]


def _blocked_comparisons(evaluation_dir: Path) -> list[dict[str, Any]]:
    if not evaluation_dir.is_dir():
        return []
    blocked = []
    for path in sorted(evaluation_dir.glob("*/comparison.json")):
        payload = _read_json(path)
        if payload and payload.get("headline_publishable") is False:
            blocked.append(
                {
                    "evaluation_id": payload.get("evaluation_id"),
                    "blocking_reasons": payload.get("blocking_reasons", []),
                    "measured_baseline_ndcg_at_3": payload["headline"]["baseline"][
                        "ndcg_at_3"
                    ],
                    "measured_model_ndcg_at_3": payload["headline"]["model"][
                        "ndcg_at_3"
                    ],
                    "measured_relative_improvement_percent": payload["headline"][
                        "relative_improvement_percent"
                    ],
                }
            )
    return blocked


def _display(path: Path, root: Path) -> str:
    """Repository-relative when possible, so the report is portable."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def collect_evidence(repo_root: Path, artifacts_root: Path | None = None) -> ModelEvidence:
    """Read every artifact and decide each capability's evidence status."""
    root = Path(repo_root)
    artifacts = Path(artifacts_root) if artifacts_root else root / "artifacts"

    sources = {
        "label_statistics": artifacts / "dataset" / "label_statistics.json",
        "candidate_statistics": artifacts / "dataset" / "candidate_statistics.json",
        "dataset_statistics": artifacts / "dataset" / "dataset_statistics.json",
        "split_statistics": artifacts / "dataset" / "split_statistics.json",
        "feature_statistics": artifacts / "features" / "feature_statistics.json",
        "embedding_statistics": artifacts / "features" / "embedding_statistics.json",
        "readiness_report": artifacts / "experiments" / "readiness_report.json",
    }
    read: dict[str, str] = {}
    missing: list[str] = []
    loaded: dict[str, dict[str, Any] | None] = {}
    for name, path in sources.items():
        payload = _read_json(path)
        loaded[name] = payload
        if payload is None:
            missing.append(_display(path, root))
        else:
            read[name] = _display(path, root)

    labels = loaded["label_statistics"] or {}
    candidates = loaded["candidate_statistics"] or {}
    dataset = loaded["dataset_statistics"] or {}
    features = loaded["feature_statistics"] or {}
    embeddings = loaded["embedding_statistics"] or {}
    # The counts live one level down in the feature report; flattened here so
    # every measurement below is a single, named read.
    feature_candidates = (features.get("candidates") or {}) if features else {}
    readiness = loaded["readiness_report"] or {}

    def measured(source: Mapping[str, Any], key: str) -> Any:
        return source.get(key, NOT_AVAILABLE) if source else NOT_AVAILABLE

    training = _best_training_run(artifacts / "training")
    if training:
        read["training_summary"] = _display(Path(training[0]), root)
    else:
        missing.append(
            _display(artifacts / "training" / "*" / "training_summary.json", root)
        )
    training_summary = training[1] if training else {}

    publishable = _publishable_comparison(artifacts / "evaluation")
    if publishable:
        read["comparison"] = _display(Path(publishable[0]), root)
    comparison = publishable[1] if publishable else None
    blocked = _blocked_comparisons(artifacts / "evaluation")

    measurements: dict[str, Any] = {
        "human_labelled_candidate_count": measured(
            labels, "human_labelled_candidate_count"
        ),
        "human_labelled_audio_hours": measured(labels, "human_labelled_audio_hours"),
        "human_annotator_count": measured(labels, "annotator_count"),
        "weakly_labelled_candidate_count": measured(
            labels, "weakly_labelled_candidate_count"
        ),
        "held_out_evaluation_candidate_count": measured(
            labels, "held_out_evaluation_candidate_count"
        ),
        "generated_candidate_count": measured(candidates, "generated_candidate_count"),
        "processed_audio_hours": measured(dataset, "processed_audio_hours"),
        "processed_episode_count": measured(dataset, "processed_episode_count"),
        "complete_multimodal_candidate_count": measured(
            feature_candidates, "complete_multimodal"
        ),
        "featurised_candidate_count": measured(feature_candidates, "processed"),
        "audio_only_candidate_count": measured(feature_candidates, "audio_only"),
        "transcribed_audio_hours": measured(
            (features.get("transcription") or {}) if features else {},
            "transcribed_audio_hours",
        ),
        "audio_embedding_model": measured(embeddings, "audio_embedding_model"),
        "text_embedding_model": measured(embeddings, "text_embedding_model"),
        "phase_5b_ready": readiness.get("phase_5b_ready", NOT_AVAILABLE),
        "model_variant": training_summary.get("model_variant", NOT_AVAILABLE),
        "model_parameter_count": training_summary.get(
            "model_parameter_count", NOT_AVAILABLE
        ),
        "model_run_id": training_summary.get("run_id", NOT_AVAILABLE),
        "model_training_label_source": training_summary.get(
            "label_source", NOT_AVAILABLE
        ),
        "model_training_data_provenance": training_summary.get(
            "data_provenance", NOT_AVAILABLE
        ),
        "baseline_ndcg_at_3": (
            comparison["headline"]["baseline"]["ndcg_at_3"]
            if comparison
            else NOT_AVAILABLE
        ),
        "model_ndcg_at_3": (
            comparison["headline"]["model"]["ndcg_at_3"]
            if comparison
            else NOT_AVAILABLE
        ),
        "relative_improvement_percent": (
            comparison["headline"]["relative_improvement_percent"]
            if comparison
            else NOT_AVAILABLE
        ),
        "evaluation_id": (
            comparison["evaluation_id"] if comparison else NOT_AVAILABLE
        ),
        "blocked_comparisons": blocked,
    }

    human_labels = measurements["human_labelled_candidate_count"]
    has_human_labels = isinstance(human_labels, (int, float)) and human_labels > 0

    capabilities: list[CapabilityEvidence] = []

    # --- Multimodal learned ranking --------------------------------------
    reasons_model: list[str] = []
    status_model = UNSUPPORTED
    trained = training_summary.get("model_variant") is not None and bool(training_summary)
    multimodal = (
        measurements["audio_embedding_model"] != NOT_AVAILABLE
        and measurements["text_embedding_model"] != NOT_AVAILABLE
    )
    if trained and multimodal:
        status_model = SUPPORTED
        reasons_model.append(
            f"A {measurements['model_variant']} PyTorch ranker with "
            f"{measurements['model_parameter_count']} parameters consumes "
            f"{measurements['audio_embedding_model']} speech representations and "
            f"{measurements['text_embedding_model']} transcript embeddings, and is "
            "served through the product's inference path."
        )
        if measurements["model_training_label_source"] != "human":
            reasons_model.append(
                "The served checkpoint was trained with label_source="
                f"{measurements['model_training_label_source']}, so it demonstrates "
                "the architecture and the serving path, not ranking quality."
            )
    else:
        reasons_model.append("No trained multimodal checkpoint was found.")

    capabilities.append(
        CapabilityEvidence(
            capability_id="multimodal_ranking",
            capability="Multimodal learned ranking",
            requirement=(
                "A multimodal PyTorch ranker consumes audio and transcript "
                "features to rank candidate podcast ad breaks, and serves the "
                "product's insertion analysis."
            ),
            status=status_model,
            reasons=reasons_model,
            evidence={
                "model_variant": measurements["model_variant"],
                "model_parameter_count": measurements["model_parameter_count"],
                "audio_embedding_model": measurements["audio_embedding_model"],
                "text_embedding_model": measurements["text_embedding_model"],
                "complete_multimodal_candidate_count": measurements[
                    "complete_multimodal_candidate_count"
                ],
            },
        )
    )

    # --- Human-labelled dataset scale ------------------------------------
    reasons_dataset = [
        f"human_labelled_candidate_count = {human_labels} "
        "(measured from the label database, not estimated).",
        f"generated_candidate_count = {measurements['generated_candidate_count']} "
        "-- candidates produced by the generators, which are NOT labels.",
    ]
    if not has_human_labels:
        status_dataset = UNSUPPORTED
        reasons_dataset.append(
            "No human label exists, so no labelled-candidate count can be quoted "
            "and no human-ground-truth evaluation can run."
        )
    elif measurements["phase_5b_ready"] is True:
        status_dataset = SUPPORTED
        reasons_dataset.append(
            "The readiness gate passes on the human labels that exist, so they "
            "are sufficient in number and spread to support a held-out "
            "evaluation."
        )
    else:
        status_dataset = PARTIALLY_SUPPORTED
        reasons_dataset.append(
            f"{human_labels} human labels exist, but the readiness gate does not "
            "pass, so they do not yet support a held-out evaluation. Run "
            "`experiment readiness` for the blocking conditions."
        )
    capabilities.append(
        CapabilityEvidence(
            capability_id="human_labelled_dataset",
            capability="Human-labelled dataset scale",
            requirement=(
                "Enough human-labelled candidates exist, spread across enough "
                "episodes and series, for the readiness gate to pass and a "
                "held-out evaluation to be possible."
            ),
            status=status_dataset,
            reasons=reasons_dataset,
            evidence={
                "human_labelled_candidate_count": human_labels,
                "generated_candidate_count": measurements["generated_candidate_count"],
                "weakly_labelled_candidate_count": measurements[
                    "weakly_labelled_candidate_count"
                ],
                "held_out_evaluation_candidate_count": measurements[
                    "held_out_evaluation_candidate_count"
                ],
                "phase_5b_ready": measurements["phase_5b_ready"],
            },
        )
    )

    # --- Held-out ranking improvement -------------------------------------
    reasons_improvement: list[str] = []
    if comparison is None:
        status_improvement = UNSUPPORTED
        reasons_improvement.append(
            "No publishable held-out comparison exists, so no NDCG@3 improvement "
            "can be quoted."
        )
        if blocked:
            reasons_improvement.append(
                f"{len(blocked)} comparison(s) ran but were blocked; see "
                "measurements.blocked_comparisons for the measured values and the "
                "reasons they may not be published."
            )
    else:
        improvement = measurements["relative_improvement_percent"]
        status_improvement = (
            SUPPORTED if isinstance(improvement, (int, float)) else UNSUPPORTED
        )
        reasons_improvement.append(
            f"Held-out comparison {measurements['evaluation_id']} measured "
            f"{improvement} % relative NDCG@3 improvement over "
            "heuristic_offline_v1."
        )
    capabilities.append(
        CapabilityEvidence(
            capability_id="held_out_ranking_improvement",
            capability="Held-out ranking improvement",
            requirement=(
                "A publishable held-out comparison reports the measured relative "
                "NDCG@3 improvement of the learned ranker over "
                "heuristic_offline_v1, whatever that improvement turns out to be."
            ),
            status=status_improvement,
            reasons=reasons_improvement,
            evidence={
                "baseline_ndcg_at_3": measurements["baseline_ndcg_at_3"],
                "model_ndcg_at_3": measurements["model_ndcg_at_3"],
                "relative_improvement_percent": measurements[
                    "relative_improvement_percent"
                ],
                "evaluation_id": measurements["evaluation_id"],
            },
        )
    )

    return ModelEvidence(
        git_sha=_git_sha(root),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        measurements=measurements,
        capabilities=capabilities,
        artifacts_read=read,
        artifacts_missing=missing,
    )


def _value(measurements: Mapping[str, Any], key: str) -> str:
    value = measurements.get(key, NOT_AVAILABLE)
    if value is None:
        return NOT_AVAILABLE
    return str(value)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Generated from the same dict as the JSON. A test asserts they agree."""
    measurements = payload["measurements"]
    lines = [
        "# Model evidence and evaluation status",
        "",
        "**Generated file - do not edit.** Produced by `slotify-rank report "
        "model-evidence`, which reads the artifacts named at the bottom. Every "
        "number here comes from one of them.",
        "",
        f"- Git SHA: `{payload['git_sha']}`",
        f"- Generated: {payload['generated_at']}",
        f"- Package version: {payload['package_version']}",
        "",
        f"`{NOT_AVAILABLE}` means no artifact produced that value. It is not a "
        "zero: a measured zero (for example, zero human labels) is printed as `0`.",
        "",
        "## Capability status",
        "",
    ]
    for capability in payload["capabilities"]:
        lines += [
            f"### {capability['capability']} - {capability['status']}",
            "",
            f"> {capability['requirement']}",
            "",
        ]
        for reason in capability["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")

    lines += [
        "## Measured quantities",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Human labelled candidates | {_value(measurements, 'human_labelled_candidate_count')} |",
        f"| Human annotators | {_value(measurements, 'human_annotator_count')} |",
        f"| Human labelled audio hours | {_value(measurements, 'human_labelled_audio_hours')} |",
        f"| Generated candidates (NOT labels) | {_value(measurements, 'generated_candidate_count')} |",
        f"| Complete multimodal feature records | {_value(measurements, 'complete_multimodal_candidate_count')} |",
        f"| Processed audio hours | {_value(measurements, 'processed_audio_hours')} |",
        f"| Processed episodes | {_value(measurements, 'processed_episode_count')} |",
        f"| Held-out evaluation candidates | {_value(measurements, 'held_out_evaluation_candidate_count')} |",
        f"| Dataset/evaluation readiness gate | {_value(measurements, 'phase_5b_ready')} |",
        "",
        "## Model",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Variant | {_value(measurements, 'model_variant')} |",
        f"| Parameters | {_value(measurements, 'model_parameter_count')} |",
        f"| Run id | {_value(measurements, 'model_run_id')} |",
        f"| Trained on label source | {_value(measurements, 'model_training_label_source')} |",
        f"| Training data provenance | {_value(measurements, 'model_training_data_provenance')} |",
        f"| Audio encoder | {_value(measurements, 'audio_embedding_model')} |",
        f"| Text encoder | {_value(measurements, 'text_embedding_model')} |",
        "",
        "## Held-out comparison",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation id | {_value(measurements, 'evaluation_id')} |",
        f"| Baseline NDCG@3 | {_value(measurements, 'baseline_ndcg_at_3')} |",
        f"| Model NDCG@3 | {_value(measurements, 'model_ndcg_at_3')} |",
        f"| Relative improvement | {_value(measurements, 'relative_improvement_percent')} |",
        "",
    ]

    blocked = measurements.get("blocked_comparisons") or []
    if blocked:
        lines += [
            "### Comparisons that ran but may not be published",
            "",
            "These measured real numbers on real artifacts. They are shown so the "
            "pipeline is auditable, and they are **not** a result.",
            "",
        ]
        for entry in blocked:
            lines.append(
                f"- `{entry['evaluation_id']}`: baseline "
                f"{entry['measured_baseline_ndcg_at_3']}, model "
                f"{entry['measured_model_ndcg_at_3']}, relative "
                f"{entry['measured_relative_improvement_percent']} %"
            )
            for reason in entry["blocking_reasons"]:
                lines.append(f"  - blocked: {reason}")
        lines.append("")

    lines += ["## Artifacts read", ""]
    for name, path in sorted(payload["artifacts_read"].items()):
        lines.append(f"- `{name}`: `{path}`")
    if payload["artifacts_missing"]:
        lines += ["", "## Artifacts missing", ""]
        for path in payload["artifacts_missing"]:
            lines.append(f"- `{path}`")

    return "\n".join(lines) + "\n"


def write_evidence(directory: Path, evidence: ModelEvidence) -> dict[str, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = evidence.to_dict()
    json_path = target / "model_evidence.json"
    md_path = target / "model_evidence.md"
    atomic_write_bytes(
        json_path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    atomic_write_bytes(md_path, render_markdown(payload).encode("utf-8"))
    return {"model_evidence.json": json_path, "model_evidence.md": md_path}
