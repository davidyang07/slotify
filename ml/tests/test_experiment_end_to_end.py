"""The whole post-labelling path, driven through the real CLI.

Train → compare → resume evidence, on a synthetic corpus, with nothing mocked
and no network. The point is that the sequence a person runs *after* the labels
exist is proven to work before they spend ten hours producing them: an argument
that does not exist, a manifest that does not join, or a report that cannot read
a comparison would otherwise surface at the worst possible moment.

The corpus is synthetic, and the last test is the one that matters most: it
asserts the pipeline **refuses to publish** a headline measured on it. The
fixture carries ``label_source: human`` so it can exercise this path at all,
which makes it precisely the input that could turn generated numbers into a
plausible-looking result.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from slotify_rank.cli import main
from slotify_rank.config.settings import find_repo_root

REPO_ROOT = Path(find_repo_root())
SCRATCH_ROOT = REPO_ROOT / "ml" / ".pytest-tmp" / "end-to-end"


@pytest.fixture(scope="module")
def trained():
    """A synthetic corpus and one trained checkpoint, shared by the module.

    Training is the expensive part (a real PyTorch loop, ~10 s), and the corpus
    is read-only once written, so it is built once for the module.
    """
    pytest.importorskip("torch")
    directory = SCRATCH_ROOT / f"run-{uuid.uuid4().hex[:12]}"
    (directory / "data").mkdir(parents=True, exist_ok=True)
    try:
        assert (
            main(
                [
                    "training",
                    "synthesize",
                    "--data-root",
                    str(directory / "data"),
                    "--episodes",
                    "10",
                    "--candidates-per-episode",
                    "10",
                ]
            )
            == 0
        )
        labels = directory / "data" / "labels" / "labels_synthetic.jsonl"
        run_dir = directory / "run"
        assert (
            main(
                [
                    "training",
                    "run",
                    "--data-root",
                    str(directory / "data"),
                    "--labels",
                    str(labels),
                    "--model",
                    "handcrafted",
                    "--run-dir",
                    str(run_dir),
                    "--smoke",
                ]
            )
            == 0
        )
        yield directory, labels, run_dir
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _compare(trained, output_name: str, *extra: str) -> tuple[int, Path]:
    directory, labels, run_dir = trained
    out = directory / output_name
    code = main(
        [
            "evaluation",
            "compare",
            "--data-root",
            str(directory / "data"),
            "--labels",
            str(labels),
            "--model",
            str(run_dir / "best_checkpoint.pt"),
            "--split",
            "test",
            "--output-dir",
            str(out),
            "--bootstrap-resamples",
            "200",
            "--skip-classical-baseline",
            *extra,
        ]
    )
    return code, out


def test_the_comparison_runs_end_to_end_and_writes_every_artifact(trained):
    code, out = _compare(trained, "eval")
    assert code == 0
    for name in (
        "comparison.json",
        "metrics.json",
        "per_episode_metrics.json",
        "ranked_candidates.jsonl",
        "summary.md",
    ):
        assert (out / name).is_file(), name

    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    headline = payload["headline"]
    assert isinstance(headline["baseline"]["ndcg_at_3"], float)
    assert isinstance(headline["model"]["ndcg_at_3"], float)
    assert payload["cohort"]["candidates_by_split"]["train"] > 0
    assert payload["cohort"]["candidates_by_split"]["test"] > 0
    assert payload["inputs"]["artifact_hashes"]["model_checkpoint"]


def test_a_synthetic_corpus_can_never_produce_a_publishable_headline(trained):
    """The guard that matters: generated numbers are not a result."""
    _, out = _compare(trained, "eval-synthetic")
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["headline_publishable"] is False
    assert any(
        "synthetic" in reason for reason in payload["blocking_reasons"]
    ), payload["blocking_reasons"]
    assert "NOT PUBLISHABLE" in (out / "summary.md").read_text(encoding="utf-8")


def test_require_publishable_exits_non_zero_on_a_blocked_headline(trained):
    code, _ = _compare(trained, "eval-required", "--require-publishable")
    assert code == 1


def test_the_bootstrap_is_recorded_on_a_real_run(trained):
    _, out = _compare(trained, "eval-bootstrap")
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    bootstrap = payload["bootstrap"]
    if bootstrap["measured"]:
        assert bootstrap["config"]["unit"] == "episode"
        assert bootstrap["episode_count"] >= 2
    else:
        # A synthetic corpus can end up with a single scorable test episode.
        assert "at least two" in bootstrap["reason"]


def test_the_experiment_manifest_resolves_against_a_real_run(trained):
    directory, _, _ = trained
    output = directory / "manifest.json"
    code = main(
        [
            "experiment",
            "manifest",
            "--data-root",
            str(directory / "data"),
            "--config",
            str(REPO_ROOT / "ml" / "configs" / "experiment_resume_v1.yaml"),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    # No split manifest and no labels in this scratch corpus, so it is blocked --
    # and it names both reasons rather than resolving optimistically.
    assert payload["ready"] is False
    assert payload["experiment_version"] == "experiment-resume-v1"
    assert payload["config_digest"]


def test_the_resume_report_reads_a_comparison_it_is_pointed_at(trained):
    """The report joins to real artifacts, not just to fixtures."""
    directory, _, run_dir = trained
    _, out = _compare(trained, "eval-for-report")

    artifacts = directory / "artifacts"
    (artifacts / "evaluation").mkdir(parents=True, exist_ok=True)
    shutil.copytree(out, artifacts / "evaluation" / "eval-1", dirs_exist_ok=True)
    (artifacts / "training" / "run-1").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        run_dir / "training_summary.json",
        artifacts / "training" / "run-1" / "training_summary.json",
    )

    reports = directory / "reports"
    code = main(
        [
            "report",
            "resume-evidence",
            "--data-root",
            str(directory / "data"),
            "--artifacts-root",
            str(artifacts),
            "--output-dir",
            str(reports),
        ]
    )
    assert code == 0
    payload = json.loads((reports / "resume_evidence.json").read_text(encoding="utf-8"))
    statuses = {check["key"]: check["status"] for check in payload["checks"]}

    # The comparison was blocked as synthetic, so the headline stays unmeasured.
    assert statuses["measured_relative_improvement"] == "NOT MEASURED"
    assert statuses["improvement_meets_claim"] == "NOT MEASURED"
    # ...but the report still read the run and reported what it found there.
    assert statuses["multimodal_ranker_exists"] == "PASS"
    assert (reports / "resume_evidence.md").is_file()
