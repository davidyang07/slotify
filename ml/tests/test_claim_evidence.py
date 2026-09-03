"""The generated claim-evidence report.

This report is the thing a reader is asked to trust, so the tests are mostly
about the ways it could lie:

* reporting an unmeasured claim as supported, or as failed;
* letting a threshold be typed into the checker instead of read from the
  committed experiment definition;
* counting a `pyproject.toml` line as evidence that a library is used;
* letting the markdown and the JSON disagree;
* passing the award check on wording that overstates the award.

Every fixture below writes artifacts into a temporary tree, so nothing here
reads or depends on the repository's real numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slotify_rank.evaluation.claim_evidence import (
    AWARD_WORDING,
    EMPIRICAL,
    FAIL,
    IMPLEMENTATION,
    IMPLEMENTATION_CHECKS,
    NOT_MEASURED,
    PASS,
    collect_claim_evidence,
    declared_dependencies,
    importing_modules,
    render_markdown,
    write_claim_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = REPO_ROOT / "ml" / "configs" / "experiment_v1.yaml"


# --------------------------------------------------------------------------
# A synthetic repository whose artifacts say whatever a test needs
# --------------------------------------------------------------------------


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _artifacts(
    root: Path,
    *,
    human_labels: int = 0,
    training_label_source: str = "weak_heuristic",
    with_comparison: bool = False,
    relative_improvement: float | None = None,
    publishable: bool = True,
    crosscheck_ok: bool = True,
    classical_ok: bool = True,
    librosa_version: str | None = "0.11.0",
    degraded_split: bool = False,
) -> Path:
    artifacts = root / "artifacts"
    _write(
        artifacts / "dataset" / "label_statistics.json",
        {"human_labelled_candidate_count": human_labels},
    )
    _write(
        artifacts / "dataset" / "candidate_statistics.json",
        {
            "generated_candidate_count": 6909,
            "candidates_by_primary_source": {
                "silence": 3000,
                "pause": 2000,
                "rms_minimum": 1500,
                "fixed_interval": 409,
            },
        },
    )
    _write(
        artifacts / "dataset" / "dataset_statistics.json",
        {
            "processed_audio_hours": 9.64,
            "processed_episode_count": 61,
            "series_count": 31,
        },
    )
    _write(
        artifacts / "features" / "feature_statistics.json",
        {
            "features": {
                "handcrafted_feature_count": 110,
                "handcrafted_feature_names": [
                    "spectral_centroid_before_short",
                    "onset_strength_mean_around_short",
                    "ms_from_episode_start",
                ],
            },
            "candidates": {"complete_multimodal": 6800},
            "library_versions": (
                {"librosa": librosa_version, "transformers": "4.57.6"}
                if librosa_version
                else {"librosa": "absent", "transformers": "4.57.6"}
            ),
        },
    )
    _write(
        artifacts / "features" / "embedding_statistics.json",
        {
            "audio_embedding_model": "openai/whisper-tiny.en",
            "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )
    run = artifacts / "training" / "run-1"
    _write(
        run / "training_summary.json",
        {
            "run_id": "run-1",
            "generated_at": "2026-08-29T00:00:00+00:00",
            "model_variant": "gated",
            "model_parameter_count": 489477,
            "label_source": training_label_source,
        },
    )
    _write(run / "environment.json", {"dependency_versions": {"torch": "2.13.0"}})

    groups = (
        {"development": ["s1", "s2", "s3"]}
        if degraded_split
        else {
            "train": ["s1", "s2", "s3"],
            "validation": ["s4"],
            "test": ["s5"],
        }
    )
    _write(
        artifacts / "experiments" / "experiment-v1.json",
        {
            "experiment_version": "experiment-v1",
            "package_version": "0.2.0",
            "config_digest": "e" * 64,
            "split": {
                "version": "v3",
                "group_by": "series",
                "seed": 42,
                "degraded": degraded_split,
                "groups_by_split": groups,
            },
            "artifact_hashes": {
                "split_manifest": "a" * 64,
                "baseline_config": "b" * 64,
            },
        },
    )

    if with_comparison:
        _write(
            artifacts / "evaluation" / "eval-1" / "comparison.json",
            {
                "evaluation_id": "eval-1",
                "generated_at": "2026-08-29T01:00:00+00:00",
                "headline_publishable": publishable,
                "inputs": {"baseline_version": "heuristic_offline_v1"},
                "headline": {
                    "baseline": {"ndcg_at_3": 0.5},
                    "model": {"ndcg_at_3": 0.6},
                    "absolute_improvement": 0.1,
                    "relative_improvement_percent": relative_improvement,
                },
                "bootstrap": {
                    "measured": True,
                    "episode_count": 6,
                    "config": {"resamples": 2000, "confidence": 0.95, "seed": 1},
                    "relative_improvement_percent": {"low": 4.0, "high": 34.0},
                },
                "cohort": {
                    "labelled_candidate_count": 2400,
                    "candidates_by_split": {"train": 1680, "validation": 360, "test": 360},
                    "episodes_by_split": {"train": 40, "validation": 8, "test": 6},
                    "series_by_split": {"train": 20, "validation": 4, "test": 3},
                },
                "metric_crosscheck": {
                    "available": crosscheck_ok,
                    "agrees": crosscheck_ok,
                    "sklearn_version": "1.9.0",
                },
                "classical_baseline": {
                    "available": classical_ok,
                    "ndcg_at_3": 0.55,
                    "sklearn_version": "1.9.0",
                },
            },
        )
    return artifacts


def _readme(root: Path, text: str) -> None:
    (root / "README.md").write_text(text, encoding="utf-8")


def _collect(root: Path, **kwargs):
    artifacts = _artifacts(root, **kwargs)
    return collect_claim_evidence(
        repo_root=root,
        artifacts_root=artifacts,
        experiment_config_path=EXPERIMENT_CONFIG,
    )


# --------------------------------------------------------------------------
# Measured vs unmeasured vs failed
# --------------------------------------------------------------------------


def test_no_labels_and_no_comparison_is_a_mix_of_fail_and_not_measured(tmp_path: Path):
    _readme(tmp_path, f"Won {AWARD_WORDING}.")
    evidence = _collect(tmp_path)

    # A counted zero is a measurement, and 0 < 2400 is a failure.
    assert evidence.check("human_label_volume").status == FAIL
    # Nobody has run a held-out comparison, which is not the same as a bad one.
    assert evidence.check("measured_relative_improvement").status == NOT_MEASURED
    assert evidence.check("improvement_meets_claim").status == NOT_MEASURED
    assert evidence.all_pass is False


def test_generated_candidates_are_never_counted_as_labels(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)
    check = evidence.check("human_label_volume")
    assert check.evidence["human_labelled_candidate_count"] == 0
    assert check.evidence["generated_candidate_count"] == 6909
    assert "GENERATED" in check.detail


def test_enough_labels_passes_the_volume_check(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path, human_labels=2400)
    assert evidence.check("human_label_volume").status == PASS


def test_one_label_short_fails_the_volume_check(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path, human_labels=2399)
    assert evidence.check("human_label_volume").status == FAIL


def test_a_weakly_trained_checkpoint_fails_the_provenance_check(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path, training_label_source="weak_heuristic")
    check = evidence.check("human_trained_checkpoint")
    assert check.status == FAIL
    assert "weak_heuristic" in check.detail


def test_a_human_trained_checkpoint_passes(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path, training_label_source="human")
    assert evidence.check("human_trained_checkpoint").status == PASS


# --------------------------------------------------------------------------
# The threshold comparison -- the claim itself
# --------------------------------------------------------------------------


def test_an_improvement_above_the_claim_passes(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=21.4,
    )
    check = evidence.check("improvement_meets_claim")
    assert check.status == PASS
    assert check.evidence["measured_relative_improvement_percent"] == 21.4
    assert check.evidence["required_relative_improvement_percent"] == 18.0


def test_an_improvement_below_the_claim_fails_and_reports_the_real_number(
    tmp_path: Path,
):
    """The measured number is the result. The threshold is not a target."""
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=6.2,
    )
    check = evidence.check("improvement_meets_claim")
    assert check.status == FAIL
    assert "6.20 %" in check.detail
    assert evidence.measurements["relative_improvement_percent"] == 6.2
    # The three underlying measurements are still PASS: they were measured.
    assert evidence.check("measured_relative_improvement").status == PASS


def test_a_blocked_comparison_is_not_treated_as_a_result(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=99.0,
        publishable=False,
    )
    assert evidence.check("measured_model_ndcg").status == NOT_MEASURED
    assert evidence.check("improvement_meets_claim").status == NOT_MEASURED
    assert evidence.measurements["relative_improvement_percent"] == NOT_MEASURED


def test_the_threshold_comes_from_the_committed_config_not_the_code(tmp_path: Path):
    """Changing the committed claim changes the verdict; nothing else does."""
    _readme(tmp_path, AWARD_WORDING)
    relaxed = tmp_path / "relaxed.yaml"
    relaxed.write_text(
        EXPERIMENT_CONFIG.read_text(encoding="utf-8").replace(
            "minimum_relative_improvement_percent: 18.0",
            "minimum_relative_improvement_percent: 5.0",
        ),
        encoding="utf-8",
    )
    artifacts = _artifacts(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=6.2,
    )
    strict = collect_claim_evidence(tmp_path, artifacts, EXPERIMENT_CONFIG)
    lenient = collect_claim_evidence(tmp_path, artifacts, relaxed)
    assert strict.check("improvement_meets_claim").status == FAIL
    assert lenient.check("improvement_meets_claim").status == PASS


def test_a_degraded_split_fails_the_held_out_check(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path, degraded_split=True)
    assert evidence.check("frozen_held_out_test_set").status == FAIL


def test_a_disjoint_grouped_split_passes_the_held_out_check(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)
    check = evidence.check("frozen_held_out_test_set")
    assert check.status == PASS
    assert check.evidence["group_by"] == "series"


# --------------------------------------------------------------------------
# Library claims
# --------------------------------------------------------------------------


def test_declared_dependencies_parses_extras_and_normalises_names():
    declared = declared_dependencies(REPO_ROOT / "ml" / "pyproject.toml")
    assert "torch" in declared
    assert "transformers" in declared
    assert "librosa" in declared
    assert "scikit-learn" in declared


def test_importing_modules_finds_real_imports_and_ignores_mentions_in_text():
    source_root = REPO_ROOT / "ml" / "src" / "slotify_rank"
    sklearn_importers = importing_modules(source_root, "sklearn")
    assert any("crosscheck" in name for name in sklearn_importers)
    assert any("classical" in name for name in sklearn_importers)
    # claim_evidence.py names every library in prose but imports none of them.
    assert not any("claim_evidence" in name for name in sklearn_importers)


def test_sklearn_needs_an_artifact_not_just_a_dependency_line(tmp_path: Path):
    """A declared, imported library with nothing to show for it is not evidence."""
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)  # no comparison at all
    assert evidence.check("sklearn_used").status == NOT_MEASURED

    with_artifacts = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=20.0,
    )
    check = with_artifacts.check("sklearn_used")
    assert check.status == PASS
    assert check.evidence["metric_crosscheck_agrees"] is True
    assert check.evidence["classical_baseline_available"] is True


def test_sklearn_fails_the_claim_when_only_half_of_its_job_ran(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=20.0,
        classical_ok=False,
    )
    assert evidence.check("sklearn_used").status == NOT_MEASURED


def test_librosa_needs_a_recorded_version_and_derived_features(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    assert _collect(tmp_path).check("librosa_used").status == PASS
    absent = _collect(tmp_path, librosa_version=None)
    assert absent.check("librosa_used").status == NOT_MEASURED


def test_pytorch_and_transformers_are_evidenced_by_artifacts(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)
    assert evidence.check("pytorch_used").status == PASS
    assert evidence.check("pytorch_used").evidence["training_torch_version"] == "2.13.0"
    assert evidence.check("huggingface_used").status == PASS


# --------------------------------------------------------------------------
# Award wording
# --------------------------------------------------------------------------


def test_the_precise_award_wording_passes(tmp_path: Path):
    _readme(tmp_path, f"# Slotify\n\n{AWARD_WORDING}\n")
    assert _collect(tmp_path).check("award_wording").status == PASS


def test_the_ascii_dash_variant_also_passes(tmp_path: Path):
    _readme(tmp_path, "UofTHacks 13 - MLH Best Use of ElevenLabs\n")
    assert _collect(tmp_path).check("award_wording").status == PASS


def test_missing_wording_fails(tmp_path: Path):
    _readme(tmp_path, "UofTHacks 13 winner (MLH Best Use of ElevenLabs).\n")
    check = _collect(tmp_path).check("award_wording")
    assert check.status == FAIL
    assert "does not contain the precise wording" in check.detail


@pytest.mark.parametrize(
    "claim",
    [
        "Hackathon winner at UofTHacks 13.",
        "We took first place.",
        "Grand prize, UofTHacks 13.",
        "Best overall project.",
    ],
)
def test_overstating_the_award_fails_even_with_the_right_wording(
    tmp_path: Path, claim: str
):
    _readme(tmp_path, f"{AWARD_WORDING}\n\n{claim}\n")
    check = _collect(tmp_path).check("award_wording")
    assert check.status == FAIL
    assert check.evidence["overstated_patterns_matched"]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_the_markdown_and_the_json_cannot_disagree(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(
        tmp_path,
        human_labels=2400,
        training_label_source="human",
        with_comparison=True,
        relative_improvement=21.4,
    )
    written = write_claim_evidence(tmp_path / "reports", evidence)
    payload = json.loads(
        written["claim_evidence.json"].read_text(encoding="utf-8")
    )
    markdown = written["claim_evidence.md"].read_text(encoding="utf-8")
    assert markdown == render_markdown(payload)
    for check in payload["checks"]:
        assert check["status"] in (PASS, FAIL, NOT_MEASURED)
        assert check["claim"] in markdown
    assert "21.4" in markdown


def test_an_unmeasured_value_renders_as_not_measured_and_a_zero_as_zero(
    tmp_path: Path,
):
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)
    markdown = render_markdown(evidence.to_dict())
    assert "| Human-labelled candidates | 0 |" in markdown
    assert f"| Model NDCG@3 | {NOT_MEASURED} |" in markdown


def test_the_report_contains_no_hard_coded_thresholds():
    """The one number a reader must not find typed into this module.

    18.0 is the claim threshold and lives in the committed experiment
    definition; if it also appeared as a literal here, editing the config would
    stop changing the verdict.
    """
    source = (
        REPO_ROOT
        / "ml"
        / "src"
        / "slotify_rank"
        / "evaluation"
        / "claim_evidence.py"
    ).read_text(encoding="utf-8")
    for literal in ("18.0", "2400", "0.18"):
        assert literal not in source, f"{literal!r} must not be hard-coded"


# --------------------------------------------------------------------------
# Implementation evidence versus empirical evidence
# --------------------------------------------------------------------------


def test_every_check_is_classified_exactly_once(tmp_path: Path):
    """The two classes partition the report, and the table drives the class.

    If a check key were missing from IMPLEMENTATION_CHECKS it would silently
    land in the empirical table, where a PASS reads as a measurement.
    """
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)
    keys = [check.key for check in evidence.checks]
    assert len(keys) == len(set(keys))
    for check in evidence.checks:
        assert check.evidence_class in (IMPLEMENTATION, EMPIRICAL)
        assert (check.evidence_class == IMPLEMENTATION) == (
            check.key in IMPLEMENTATION_CHECKS
        )
    # No key in the table that the report never produces: a stale entry there
    # would classify nothing while looking like coverage.
    assert IMPLEMENTATION_CHECKS <= set(keys)


def test_a_complete_implementation_never_supports_an_empirical_claim(
    tmp_path: Path,
):
    """The failure this whole split exists to prevent.

    Every capability can be implemented, tested and artifact-backed while zero
    human labels exist and no held-out number has been measured. The empirical
    verdicts must be unmoved by that.
    """
    _readme(tmp_path, AWARD_WORDING)
    _queue_summary(tmp_path / "artifacts", unique=2400, target=2400)
    evidence = _collect(tmp_path, human_labels=0)

    implemented = [
        c for c in evidence.checks if c.evidence_class == IMPLEMENTATION
    ]
    assert implemented, "the report tracks implementation claims"
    assert all(c.status == PASS for c in implemented), [
        (c.key, c.status, c.detail) for c in implemented if c.status != PASS
    ]

    assert evidence.check("human_label_volume").status == FAIL
    assert evidence.check("measured_relative_improvement").status == NOT_MEASURED
    assert evidence.check("improvement_meets_claim").status == NOT_MEASURED
    assert evidence.check("sklearn_used").status == NOT_MEASURED


def _queue_summary(artifacts: Path, *, unique: int, target: int, repeats: int = 60):
    _write(
        artifacts / "labelling" / "queue_summary.json",
        {
            "queue_version": "full-v2",
            "target_unique": target,
            "unique_candidate_count": unique,
            "presentation_count": unique + repeats,
            "blind_repeat_presentations": repeats,
            "candidate_manifest_hash": "c" * 64,
            "split_manifest_hash": "d" * 64,
            "coverage": {"by_split": {"train": 1680, "validation": 360, "test": 360}},
        },
    )


def test_the_labelling_check_reports_the_queue_not_the_labels(tmp_path: Path):
    """A built queue is infrastructure. It is never a count of collected labels."""
    _readme(tmp_path, AWARD_WORDING)
    _queue_summary(tmp_path / "artifacts", unique=2400, target=2400)
    evidence = _collect(tmp_path, human_labels=0)

    check = evidence.check("labelling_infrastructure")
    assert check.status == PASS
    assert check.evidence_class == IMPLEMENTATION
    assert check.evidence["unique_candidate_count"] == 2400
    assert check.evidence["blind_repeat_presentations"] == 60
    # The same report, in the same object, still says nobody has labelled any.
    assert check.evidence["human_labels_collected_against_it"] == 0
    assert evidence.check("human_label_volume").status == FAIL
    assert "not a count of labels collected against it" in check.detail


def test_a_queue_short_of_its_target_fails(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    _queue_summary(tmp_path / "artifacts", unique=120, target=2400)
    check = _collect(tmp_path).check("labelling_infrastructure")
    assert check.status == FAIL


def test_no_queue_summary_is_unmeasured_not_failed(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    check = _collect(tmp_path).check("labelling_infrastructure")
    assert check.status == NOT_MEASURED


def test_the_matrix_check_reads_the_experiments_own_ablations(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    check = _collect(tmp_path).check("experiment_matrix")
    assert check.status == PASS
    ablations = check.evidence["ablations"]
    seeds = check.evidence["seeds"]
    assert check.evidence["cell_count"] == len(ablations) * len(seeds)
    assert check.evidence["headline_variant"] in ablations
    # Every declared variant is a committed config, not just a name in YAML.
    for variant in ablations:
        assert any(
            stem.startswith(variant)
            for stem in check.evidence["committed_model_configs"]
        ), variant


def test_selection_must_be_a_validation_metric(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    check = _collect(tmp_path).check("validation_checkpoint_selection")
    assert check.status == PASS
    assert str(check.evidence["checkpoint_selection_metric"]).startswith("validation")


def test_the_sklearn_capabilities_are_named_symbols_not_a_dependency_line(
    tmp_path: Path,
):
    """Declaring scikit-learn is not implementing a baseline or a cross-check."""
    _readme(tmp_path, AWARD_WORDING)
    evidence = _collect(tmp_path)

    classical = evidence.check("classical_baseline_implemented")
    assert classical.status == PASS
    assert "HistGradientBoostingRegressor" in classical.evidence["referenced_symbols"]
    assert "GroupKFold" in classical.evidence["referenced_symbols"]
    assert classical.evidence["tests"]

    crosscheck = evidence.check("ndcg_crosscheck_implemented")
    assert crosscheck.status == PASS
    assert "ndcg_score" in crosscheck.evidence["referenced_symbols"]
    assert crosscheck.evidence["tests"]


def test_the_markdown_renders_the_two_classes_in_separate_tables(tmp_path: Path):
    _readme(tmp_path, AWARD_WORDING)
    _queue_summary(tmp_path / "artifacts", unique=2400, target=2400)
    evidence = _collect(tmp_path)
    markdown = render_markdown(evidence.to_dict())

    assert "## Implemented capabilities" in markdown
    assert "## Empirical results" in markdown
    implemented_section = markdown.split("## Implemented capabilities")[1].split(
        "## Empirical results"
    )[0]
    empirical_section = markdown.split("## Empirical results")[1].split("## Detail")[0]
    for check in evidence.checks:
        section = (
            implemented_section
            if check.evidence_class == IMPLEMENTATION
            else empirical_section
        )
        assert check.claim in section, check.key


def test_a_manifest_without_a_digest_does_not_establish_reproducibility(
    tmp_path: Path,
):
    """A manifest that pins nothing is not a reproducibility record."""
    _readme(tmp_path, AWARD_WORDING)
    artifacts = _artifacts(tmp_path)
    manifest = artifacts / "experiments" / "experiment-v1.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("config_digest")
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    evidence = collect_claim_evidence(
        repo_root=tmp_path,
        artifacts_root=artifacts,
        experiment_config_path=EXPERIMENT_CONFIG,
    )
    assert evidence.check("reproducibility_manifest").status == FAIL
