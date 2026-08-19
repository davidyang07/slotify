"""The generated model-evidence report.

The property under test throughout: an unmeasured quantity and a measured zero
must never render the same way, and no amount of generated or weakly labelled
data may stand in for human ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from slotify_rank.evaluation.evidence import (
    NOT_AVAILABLE,
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    UNSUPPORTED,
    collect_evidence,
    render_markdown,
    write_evidence,
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_artifacts(
    root: Path,
    human_labels: int = 0,
    label_source: str = "weak_heuristic",
    comparison: dict | None = None,
    readiness: bool = False,
    with_training: bool = True,
    with_features: bool = True,
) -> Path:
    artifacts = root / "artifacts"
    write(
        artifacts / "dataset" / "label_statistics.json",
        {
            "human_labelled_candidate_count": human_labels,
            "human_labelled_audio_hours": 0.0,
            "annotator_count": 0,
            "weakly_labelled_candidate_count": 0,
            "held_out_evaluation_candidate_count": 0,
        },
    )
    write(
        artifacts / "dataset" / "candidate_statistics.json",
        {"generated_candidate_count": 595},
    )
    write(
        artifacts / "dataset" / "dataset_statistics.json",
        {"processed_audio_hours": 0.8387, "processed_episode_count": 13},
    )
    write(artifacts / "dataset" / "split_statistics.json", {"split_version": "v2"})
    write(
        artifacts / "experiments" / "readiness_report.json",
        {"phase_5b_ready": readiness},
    )
    if with_features:
        write(
            artifacts / "features" / "feature_statistics.json",
            {"candidates": {"processed": 595, "complete_multimodal": 583, "audio_only": 12},
             "transcription": {"transcribed_audio_hours": 0.8213}},
        )
        write(
            artifacts / "features" / "embedding_statistics.json",
            {
                "audio_embedding_model": "openai/whisper-tiny.en",
                "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        )
    if with_training:
        write(
            artifacts / "training" / "gated-abc" / "training_summary.json",
            {
                "generated_at": "2026-08-18T00:00:00+00:00",
                "model_variant": "gated",
                "model_parameter_count": 489_477,
                "run_id": "gated-abc",
                "label_source": label_source,
                "data_provenance": "weak_supervision",
            },
        )
    if comparison is not None:
        write(artifacts / "evaluation" / "eval-abc" / "comparison.json", comparison)
    return artifacts


def publishable_comparison() -> dict:
    """A comparison the gates let through.

    The numbers are a fixture for the arithmetic, not a target: 0.72 against
    0.61 is a 18.0327 % relative improvement, and the report must print whatever
    the comparison measured.
    """
    return {
        "evaluation_id": "eval-abc",
        "generated_at": "2026-08-18T00:00:00+00:00",
        "headline_publishable": True,
        "blocking_reasons": [],
        "headline": {
            "baseline": {"ndcg_at_3": 0.61},
            "model": {"ndcg_at_3": 0.72},
            "relative_improvement_percent": 18.0327,
        },
    }


def blocked_comparison() -> dict:
    payload = publishable_comparison()
    payload["headline_publishable"] = False
    payload["blocking_reasons"] = ["ground truth is label_source='weak_heuristic'"]
    return payload


def status(evidence, capability_id: str) -> str:
    return next(
        c.status for c in evidence.capabilities if c.capability_id == capability_id
    )


# ---------------------------------------------------------------------------
# Unavailable is not zero
# ---------------------------------------------------------------------------


def test_an_unmeasured_metric_reports_not_yet_available(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    evidence = collect_evidence(tmp_path, artifacts)
    assert evidence.measurements["relative_improvement_percent"] == NOT_AVAILABLE
    assert evidence.measurements["baseline_ndcg_at_3"] == NOT_AVAILABLE
    assert evidence.measurements["model_ndcg_at_3"] == NOT_AVAILABLE


def test_a_measured_zero_stays_zero(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=0)
    evidence = collect_evidence(tmp_path, artifacts)
    assert evidence.measurements["human_labelled_candidate_count"] == 0
    markdown = render_markdown(evidence.to_dict())
    assert "| Human labelled candidates | 0 |" in markdown


def test_missing_artifacts_are_listed_rather_than_defaulted(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, with_features=False, with_training=False)
    evidence = collect_evidence(tmp_path, artifacts)
    assert evidence.artifacts_missing
    assert evidence.measurements["model_variant"] == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Capability status
# ---------------------------------------------------------------------------


def test_multimodal_ranking_is_supported_by_a_trained_multimodal_model(
    tmp_path: Path,
) -> None:
    artifacts = build_artifacts(tmp_path)
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "multimodal_ranking") == SUPPORTED
    reasons = next(
        c.reasons for c in evidence.capabilities if c.capability_id == "multimodal_ranking"
    )
    # Supported, but the weak supervision must still be stated.
    assert any("weak_heuristic" in reason for reason in reasons)


def test_multimodal_ranking_is_unsupported_without_a_checkpoint(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, with_training=False)
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "multimodal_ranking") == UNSUPPORTED


def test_zero_human_labels_support_no_dataset_scale(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=0)
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "human_labelled_dataset") == UNSUPPORTED


def test_human_labels_below_the_readiness_gate_are_only_partial(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=120, readiness=False)
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "human_labelled_dataset") == PARTIALLY_SUPPORTED


def test_human_labels_that_pass_the_readiness_gate_are_supported(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=240, readiness=True)
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "human_labelled_dataset") == SUPPORTED


def test_generated_candidates_are_never_counted_as_human_labels(tmp_path: Path) -> None:
    """595 generated candidates must not be read as 595 labels."""
    artifacts = build_artifacts(tmp_path, human_labels=0)
    evidence = collect_evidence(tmp_path, artifacts)
    capability = next(
        c for c in evidence.capabilities if c.capability_id == "human_labelled_dataset"
    )
    assert capability.evidence["generated_candidate_count"] == 595
    assert capability.evidence["human_labelled_candidate_count"] == 0
    assert capability.status == UNSUPPORTED


def test_weak_ground_truth_never_yields_a_published_improvement(tmp_path: Path) -> None:
    """A comparison the gates blocked is auditable, but it is not a result."""
    artifacts = build_artifacts(tmp_path, comparison=blocked_comparison())
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "held_out_ranking_improvement") == UNSUPPORTED
    # The measured numbers are still surfaced, clearly marked as unpublishable.
    blocked = evidence.measurements["blocked_comparisons"]
    assert blocked and blocked[0]["measured_relative_improvement_percent"] == 18.0327
    assert evidence.measurements["relative_improvement_percent"] == NOT_AVAILABLE


def test_a_publishable_comparison_reports_the_measured_improvement(
    tmp_path: Path,
) -> None:
    artifacts = build_artifacts(
        tmp_path,
        human_labels=240,
        label_source="human",
        readiness=True,
        comparison=publishable_comparison(),
    )
    evidence = collect_evidence(tmp_path, artifacts)
    assert status(evidence, "held_out_ranking_improvement") == SUPPORTED
    assert evidence.measurements["baseline_ndcg_at_3"] == 0.61
    assert evidence.measurements["model_ndcg_at_3"] == 0.72
    assert evidence.measurements["relative_improvement_percent"] == 18.0327


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_and_json_agree(tmp_path: Path) -> None:
    artifacts = build_artifacts(
        tmp_path,
        human_labels=240,
        label_source="human",
        readiness=True,
        comparison=publishable_comparison(),
    )
    evidence = collect_evidence(tmp_path, artifacts)
    payload = evidence.to_dict()
    markdown = render_markdown(payload)
    assert str(payload["measurements"]["relative_improvement_percent"]) in markdown
    assert str(payload["measurements"]["human_labelled_candidate_count"]) in markdown
    assert payload["git_sha"] in markdown


def test_write_produces_both_files(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    evidence = collect_evidence(tmp_path, artifacts)
    written = write_evidence(tmp_path / "reports", evidence)
    assert set(written) == {"model_evidence.json", "model_evidence.md"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
