"""The whole post-labelling path, driven through the real CLI.

Train → compare → training matrix, on a synthetic corpus, with nothing mocked
and no network. The point is that the sequence a person runs *after* the labels
exist is proven to work before they spend ten hours producing them: an argument
that does not exist or a manifest that does not join would otherwise surface at
the worst possible moment.

The corpus is synthetic, and the guard that matters most is that the pipeline
**refuses to publish** a headline measured on it. The fixture is stamped
``label_source: synthetic``, so every step has to name that source to run at
all -- and the comparison is still blocked on it, by the ground-truth gate and
by the fixture's own ``synthetic`` flag.
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
                    "--allow-label-source",
                    "synthetic",
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
            # The fixture's labels are synthetic and there is no training
            # matrix: both have to be named for the comparison to run at all,
            # and both are then recorded as reasons it is not publishable.
            "--allow-label-source",
            "synthetic",
            "--allow-unnominated",
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
    # The CLI always supplies the series mapping, so the unit is the series.
    assert bootstrap["config"]["unit"] == "series"
    if bootstrap["measured"]:
        assert bootstrap["series_count"] >= 2
    else:
        # The fixture has one series per episode and can end up with a single
        # scorable test episode.
        assert "at least two" in bootstrap["reason"]


def test_an_unnominated_checkpoint_is_refused_without_the_flag(trained, capsys):
    """The protocol evaluates the matrix's median seed and nothing else."""
    directory, labels, run_dir = trained
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
            "--allow-label-source",
            "synthetic",
            "--output-dir",
            str(directory / "eval-refused"),
            "--skip-classical-baseline",
        ]
    )
    assert code == 1
    assert "not the run the training matrix nominated" in capsys.readouterr().err
    assert not (directory / "eval-refused" / "comparison.json").exists()


def test_an_unnominated_unfrozen_comparison_is_recorded_as_such(trained):
    _, out = _compare(trained, "eval-protocol")
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    assert protocol["nomination"]["nominated"] is False
    assert protocol["nomination"]["allowed_unnominated"] is True
    assert protocol["label_snapshot"]["verified"] is False
    reasons = " ".join(payload["blocking_reasons"])
    assert "not the run the training matrix nominated" in reasons
    assert "frozen label snapshot" in reasons
    # No experiment config, so there is no claim to judge -- said, not omitted.
    assert payload["claim"]["evaluated"] is False
    assert payload["schema_version"] == "evaluation-v1.1.0"


def test_the_comparison_records_repo_relative_inputs(trained):
    _, out = _compare(trained, "eval-paths")
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    for key in ("label_export", "features_manifest", "model_checkpoint"):
        value = payload["inputs"][key]
        assert not Path(value).is_absolute(), (key, value)
        assert "\\" not in value


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
            str(REPO_ROOT / "ml" / "configs" / "experiment_v1.yaml"),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    # No split manifest and no labels in this scratch corpus, so it is blocked --
    # and it names both reasons rather than resolving optimistically.
    assert payload["ready"] is False
    assert payload["experiment_version"] == "experiment-v1"
    assert payload["config_digest"]


def test_the_training_matrix_command_runs_a_cell_and_summarises_it(trained):
    """`experiment train` is the command a person runs once labels exist.

    One cell only: the point is that the command wires up, selects and reports,
    not that a synthetic corpus produces a good model.
    """
    directory, labels, _ = trained
    summary_path = directory / "matrix.json"
    code = main(
        [
            "experiment",
            "train",
            "--data-root",
            str(directory / "data"),
            "--config",
            str(REPO_ROOT / "ml" / "configs" / "experiment_v1.yaml"),
            "--labels",
            str(labels),
            "--output-root",
            str(directory / "matrix-runs"),
            "--summary",
            str(summary_path),
            "--variant",
            "handcrafted",
            "--seed",
            "42",
        ]
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["seed_selection"] == "median_validation_ndcg_at_3"
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["variant"] == "handcrafted"
    assert run["seed"] == 42

    # The experiment trains on human labels only, and the fixture is stamped
    # label_source=synthetic, so the loader refuses it before a single epoch:
    # the cell fails rather than training on generated numbers and being caught
    # later. The matrix names both reasons it is not quotable.
    assert code == 1
    assert run["exit_code"] == 1
    assert run["label_source"] is None
    reasons = payload["blocking_reasons"]
    assert any("run(s) failed" in reason for reason in reasons), reasons
    assert any("headline variant" in reason for reason in reasons), reasons
    assert payload["reported_run"] is None


def _matrix_cell(trained, root_name: str, fingerprint: str | None) -> dict:
    """Run `experiment train` for one cell whose directory already holds a run."""
    directory, labels, _ = trained
    cell = directory / root_name / "handcrafted-seed42"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "handcrafted-cached",
                "label_source": "human",
                "dataset_fingerprint": fingerprint,
                "best_validation_ndcg_at_3": 0.5,
                "model_parameter_count": 1,
            }
        ),
        encoding="utf-8",
    )
    summary_path = directory / f"{root_name}.json"
    main(
        [
            "experiment",
            "train",
            "--data-root",
            str(directory / "data"),
            "--config",
            str(REPO_ROOT / "ml" / "configs" / "experiment_v1.yaml"),
            "--labels",
            str(labels),
            "--output-root",
            str(directory / root_name),
            "--summary",
            str(summary_path),
            "--variant",
            "handcrafted",
            "--seed",
            "42",
        ]
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))["runs"][0]


def test_a_cached_cell_is_reused_only_on_the_same_data(trained):
    from slotify_rank.training.prepare import dataset_fingerprint

    directory, labels, _ = trained
    current = dataset_fingerprint(
        directory / "data" / "manifests" / "features.jsonl", labels
    )

    reused = _matrix_cell(trained, "matrix-same-data", current)
    assert reused["exit_code"] == 0
    assert reused["run_id"] == "handcrafted-cached"

    # Trained on other data: not reused. Retraining is attempted, and on this
    # synthetic corpus the human-only loader refuses it -- so the stale run is
    # neither reported nor silently kept.
    stale = _matrix_cell(trained, "matrix-other-data", "0" * 64)
    assert stale["exit_code"] == 1
    assert stale["run_id"] != "handcrafted-cached"


def _protocol_inputs(trained, name: str, run_id: str | None = None):
    """A nominating training matrix and a frozen snapshot of the export."""
    from slotify_rank.experiment.freeze import build_snapshot, write_snapshot
    from slotify_rank.training.checkpoint import load_checkpoint

    directory, labels, run_dir = trained
    data = directory / "data"
    nominated = run_id or str(load_checkpoint(run_dir / "best_checkpoint.pt")["run_id"])
    matrix = directory / f"{name}-matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_version": "experiment-v1",
                "ok": True,
                "blocking_reasons": [],
                "reported_run": {"run_id": nominated, "variant": "handcrafted", "seed": 42},
            }
        ),
        encoding="utf-8",
    )
    snapshot = directory / f"{name}-snapshot.json"
    write_snapshot(
        snapshot,
        build_snapshot(
            {},
            {},
            label_export_path=labels,
            candidate_manifest_path=data / "manifests" / "candidates.jsonl",
            feature_manifest_path=data / "manifests" / "features.jsonl",
            split_manifest_path=None,
            excluded_labels=[],
            snapshot_version=name,
        ),
    )
    return matrix, snapshot


def _protocol_compare(trained, name: str, matrix: Path, snapshot: Path):
    directory, labels, run_dir = trained
    out = directory / name
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
            "--allow-label-source",
            "synthetic",
            "--experiment-config",
            str(REPO_ROOT / "ml" / "configs" / "experiment_v1.yaml"),
            "--training-matrix",
            str(matrix),
            "--label-snapshot",
            str(snapshot),
            "--output-dir",
            str(out),
            "--bootstrap-resamples",
            "200",
            "--skip-classical-baseline",
        ]
    )
    return code, out


def test_a_nominated_frozen_comparison_is_judged_against_the_claim(trained):
    matrix, snapshot = _protocol_inputs(trained, "followed")
    code, out = _protocol_compare(trained, "eval-followed", matrix, snapshot)
    assert code == 0
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["nomination"]["nominated"] is True
    assert payload["protocol"]["label_snapshot"]["verified"] is True
    assert not any("nominated" in r for r in payload["blocking_reasons"])
    # Judged, and FAIL: the labels are synthetic, so the headline is blocked.
    claim = payload["claim"]
    assert claim["evaluated"] is True
    assert claim["minimum_relative_improvement_percent"] == 18.0
    assert claim["verdict"] == "FAIL"
    assert "## Claim verdict: FAIL" in (out / "summary.md").read_text(encoding="utf-8")


def test_a_checkpoint_the_matrix_did_not_nominate_is_refused(trained, capsys):
    matrix, snapshot = _protocol_inputs(trained, "other-run", run_id="gated-not-this-one")
    code, out = _protocol_compare(trained, "eval-other-run", matrix, snapshot)
    assert code == 1
    assert "nominated 'gated-not-this-one'" in capsys.readouterr().err
    assert not (out / "comparison.json").exists()


def test_an_export_that_moved_after_the_freeze_is_refused(trained, capsys):
    from slotify_rank.experiment.freeze import read_snapshot, write_snapshot

    matrix, snapshot = _protocol_inputs(trained, "moved")
    frozen = read_snapshot(snapshot)
    from dataclasses import replace

    write_snapshot(snapshot, replace(frozen, label_manifest_hash="0" * 64), force=True)
    code, out = _protocol_compare(trained, "eval-moved", matrix, snapshot)
    assert code == 1
    assert "do not match the frozen label snapshot" in capsys.readouterr().err
    assert not (out / "comparison.json").exists()


def test_the_manifest_actually_read_is_the_one_hashed_and_recorded(trained):
    """--features-manifest used to be read but the DEFAULT manifest was hashed."""
    from slotify_rank.data.checksum import sha256_file

    directory, _, _ = trained
    default = directory / "data" / "manifests" / "features.jsonl"
    elsewhere = directory / "features-elsewhere.jsonl"
    # Same records, different bytes (a trailing blank line the reader skips).
    elsewhere.write_text(default.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert sha256_file(elsewhere) != sha256_file(default)

    _, out = _compare(trained, "eval-manifest", "--features-manifest", str(elsewhere))
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["inputs"]["artifact_hashes"]["feature_manifest"] == sha256_file(elsewhere)
    assert payload["inputs"]["features_manifest"].endswith("features-elsewhere.jsonl")
