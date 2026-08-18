"""The generated resume-evidence report.

The property under test throughout: an unmeasured quantity and a measured zero
must never render the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from slotify_rank.evaluation.evidence import (
    NOT_SUPPORTED,
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
    write(artifacts / "experiments" / "readiness_report.json", {"phase_5b_ready": False})
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


# ---------------------------------------------------------------------------
# Unavailable is not zero
# ---------------------------------------------------------------------------


def test_an_unmeasured_metric_reports_not_yet_supported(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    evidence = collect_evidence(tmp_path, artifacts)
    assert evidence.measurements["relative_improvement_percent"] == NOT_SUPPORTED
    assert evidence.measurements["baseline_ndcg_at_3"] == NOT_SUPPORTED


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
    assert evidence.measurements["model_variant"] == NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Claim verdicts
# ---------------------------------------------------------------------------


def verdict(evidence, claim_id: str) -> str:
    return next(c.verdict for c in evidence.claims if c.claim_id == claim_id)


def test_claim_a_is_supported_by_a_trained_multimodal_model(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "A") == "SUPPORTED"
    reasons = next(c.reasons for c in evidence.claims if c.claim_id == "A")
    # Supported, but the weak supervision must still be stated.
    assert any("weak_heuristic" in reason for reason in reasons)


def test_claim_a_is_unsupported_without_a_checkpoint(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, with_training=False)
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "A") == NOT_SUPPORTED


def test_claim_b_is_unsupported_with_zero_human_labels(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=0)
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "B") == NOT_SUPPORTED


def test_claim_b_is_partial_below_the_stated_count(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=300)
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "B") == "PARTIALLY SUPPORTED"


def test_claim_b_needs_the_stated_count_to_be_supported(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, human_labels=2400)
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "B") == "SUPPORTED"


def test_generated_candidates_never_satisfy_claim_b(tmp_path: Path) -> None:
    """595 generated candidates must not be read as 595 labels."""
    artifacts = build_artifacts(tmp_path, human_labels=0)
    evidence = collect_evidence(tmp_path, artifacts)
    claim = next(c for c in evidence.claims if c.claim_id == "B")
    assert claim.evidence["generated_candidate_count"] == 595
    assert claim.evidence["human_labelled_candidate_count"] == 0
    assert claim.verdict == NOT_SUPPORTED


def test_claim_c_is_unsupported_while_every_comparison_is_blocked(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path, comparison=blocked_comparison())
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "C") == NOT_SUPPORTED
    # The measured numbers are still surfaced, clearly marked as unpublishable.
    blocked = evidence.measurements["blocked_comparisons"]
    assert blocked and blocked[0]["measured_relative_improvement_percent"] == 18.0327
    assert evidence.measurements["relative_improvement_percent"] == NOT_SUPPORTED


def test_claim_c_is_supported_by_a_publishable_comparison(tmp_path: Path) -> None:
    artifacts = build_artifacts(
        tmp_path, human_labels=2400, label_source="human", comparison=publishable_comparison()
    )
    evidence = collect_evidence(tmp_path, artifacts)
    assert verdict(evidence, "C") == "SUPPORTED"
    assert evidence.measurements["relative_improvement_percent"] == 18.0327


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_and_json_agree(tmp_path: Path) -> None:
    artifacts = build_artifacts(
        tmp_path, human_labels=2400, label_source="human", comparison=publishable_comparison()
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
    assert set(written) == {"resume_evidence.json", "resume_evidence.md"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
