"""End-to-end CLI: synthesize, prepare, pairs, run, validate, inspect, resume.

These drive the real ``python -m slotify_rank.cli`` entry points against a real
synthetic corpus and a real PyTorch training loop. Nothing is mocked, no network
is touched, no model weights are downloaded, and no GPU is required.

The data root has to live *inside* the checkout: manifest paths are
repository-relative by Phase 2 design, and ``to_repo_relative`` refuses a path
outside the repo. ``ml/.pytest-tmp`` is the project's own git-ignored scratch
area, so that is where these runs go -- rather than ``tmp_path``, which is only
inside the repo on machines where conftest's fallback is active.
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
SCRATCH_ROOT = REPO_ROOT / "ml" / ".pytest-tmp" / "cli-runs"


@pytest.fixture(scope="module")
def corpus():
    """One synthetic corpus for the whole module.

    Generating a corpus writes ~20 fsync'd embedding arrays and costs about 20
    seconds on this OneDrive-backed checkout, so building one per test dominated
    the suite. The corpus is read-only once written and every test still gets
    its own output directory, so sharing it changes nothing a test observes.
    """
    directory = SCRATCH_ROOT / f"corpus-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory, synthesize(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def workspace(corpus):
    """Per-test output directory, sharing the module's corpus data root."""
    root, labels = corpus
    directory = SCRATCH_ROOT / f"out-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield _Workspace(data_owner=root, labels=labels, out=directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class _Workspace:
    """The shared corpus plus this test's own scratch directory."""

    def __init__(self, data_owner: Path, labels: Path, out: Path):
        self.data_owner = data_owner
        self.labels = labels
        self.out = out

    @property
    def data_root(self) -> Path:
        return self.data_owner / "data"

    def __truediv__(self, name: str) -> Path:
        return self.out / name


def synthesize(workspace: Path, episodes: int = 8, candidates: int = 8, **extra) -> Path:
    args = [
        "training",
        "synthesize",
        "--data-root",
        str(workspace / "data"),
        "--episodes",
        str(episodes),
        "--candidates-per-episode",
        str(candidates),
    ]
    for key, value in extra.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    assert main(args) == 0
    return workspace / "data" / "labels" / "labels_synthetic.jsonl"


def base_args(workspace, labels: Path) -> list[str]:
    return ["--data-root", str(workspace.data_root), "--labels", str(labels)]


def train(
    workspace,
    labels: Path,
    variant: str = "handcrafted",
    run_dir: str = "run",
    extra: list[str] | None = None,
) -> tuple[int, Path]:
    directory = workspace / run_dir
    code = main(
        [
            "training",
            "run",
            *base_args(workspace, labels),
            "--model",
            variant,
            "--run-dir",
            str(directory),
            *(extra or []),
        ]
    )
    return code, directory


# ---------------------------------------------------------------------------
# The individual commands
# ---------------------------------------------------------------------------


def test_synthesize_writes_a_self_declaring_fixture(workspace):
    labels = workspace.labels
    assert labels.is_file()
    metadata = json.loads(
        labels.with_suffix(labels.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    assert metadata["synthetic"] is True
    assert "SYNTHETIC" in metadata["note"]
    assert (workspace.data_root / "manifests" / "features.jsonl").is_file()


def test_prepare_reports_eligibility(workspace, capsys):
    labels = workspace.labels
    output = workspace / "prepare.json"
    assert main(
        ["training", "prepare", *base_args(workspace, labels), "--output", str(output)]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_candidate_count"] > 0
    assert payload["validation_candidate_count"] > 0
    assert payload["episode_overlap_between_splits"] == []
    assert payload["synthetic_data"] is True
    assert "SYNTHETIC" in capsys.readouterr().out


def test_pairs_reports_the_distribution(workspace):
    labels = workspace.labels
    output = workspace / "pairs.json"
    rows = workspace / "pairs.jsonl"
    assert main(
        [
            "training",
            "pairs",
            *base_args(workspace, labels),
            "--output",
            str(output),
            "--pairs-output",
            str(rows),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["train"]["pair_count"] > 0
    assert payload["train"]["score_difference_histogram"]
    assert payload["validation"]["pair_count"] > 0
    lines = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == payload["train"]["pair_count"]
    assert {"preferred_candidate_id", "nonpreferred_candidate_id"} <= set(lines[0])


def test_run_produces_every_artifact(workspace):
    labels = workspace.labels
    code, directory = train(workspace, labels, extra=["--epochs", "3"])
    assert code == 0
    for name in (
        "resolved_config.json",
        "environment.json",
        "dataset_summary.json",
        "normalizer.json",
        "epoch_metrics.jsonl",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "training_summary.json",
        "training_summary.md",
    ):
        path = directory / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 0, f"{name} is empty"


def test_epoch_metrics_has_one_line_per_epoch(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, extra=["--epochs", "4"])
    lines = (directory / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    epochs = [json.loads(line)["epoch"] for line in lines]
    assert epochs == [1, 2, 3, 4]


def test_the_summary_labels_synthetic_data_loudly(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, extra=["--epochs", "2"])
    summary = json.loads((directory / "training_summary.json").read_text("utf-8"))
    assert summary["synthetic_data"] is True
    assert summary["data_provenance"] == "synthetic_fixture"
    assert summary["label_source"] == "synthetic"
    assert "NOT evidence of model quality" in summary["warning"]
    markdown = (directory / "training_summary.md").read_text("utf-8")
    assert "SYNTHETIC SMOKE RUN" in markdown


def test_the_summary_records_the_measurable_facts(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, "gated", extra=["--epochs", "3"])
    summary = json.loads((directory / "training_summary.json").read_text("utf-8"))
    assert summary["model_variant"] == "gated"
    assert summary["model_parameter_count"] > 0
    assert summary["training_pair_count"] > 0
    assert summary["training_episode_count"] > 0
    assert summary["validation_episode_count"] > 0
    assert summary["device"] == "cpu"
    assert summary["epochs_run"] == 3
    assert 0.0 <= summary["best_validation_ndcg_at_3"] <= 1.0


def test_smoke_records_every_override(workspace):
    labels = workspace.labels
    code, directory = train(workspace, labels, extra=["--smoke"])
    assert code == 0
    resolved = json.loads((directory / "resolved_config.json").read_text("utf-8"))
    assert resolved["smoke"] is True
    overrides = resolved["overrides"]
    assert overrides, "a smoke run must record what it changed"
    assert overrides["epochs"]["to"] == 3
    assert resolved["training"]["epochs"] == 3
    markdown = (directory / "training_summary.md").read_text("utf-8")
    assert "`epochs`" in markdown


def test_run_id_is_stable_across_reruns(workspace):
    labels = workspace.labels
    _, first = train(workspace, labels, run_dir="a", extra=["--epochs", "2"])
    _, second = train(workspace, labels, run_dir="b", extra=["--epochs", "2"])
    left = json.loads((first / "training_summary.json").read_text("utf-8"))
    right = json.loads((second / "training_summary.json").read_text("utf-8"))
    assert left["run_id"] == right["run_id"]
    assert left["best_validation_ndcg_at_3"] == pytest.approx(
        right["best_validation_ndcg_at_3"]
    )


def test_a_different_model_gets_a_different_run_id(workspace):
    labels = workspace.labels
    _, first = train(workspace, labels, "handcrafted", run_dir="a", extra=["--epochs", "2"])
    _, second = train(workspace, labels, "gated", run_dir="b", extra=["--epochs", "2"])
    left = json.loads((first / "training_summary.json").read_text("utf-8"))
    right = json.loads((second / "training_summary.json").read_text("utf-8"))
    assert left["run_id"] != right["run_id"]


# The trainer-level suite exercises all five variants; here two representatives
# (the simplest and the primary multimodal one) confirm the CLI wiring without
# five full training runs per corpus.
@pytest.mark.parametrize("variant", ["handcrafted", "gated"])
def test_every_variant_trains_through_the_cli(workspace, variant):
    labels = workspace.labels
    code, directory = train(
        workspace, labels, variant, run_dir=variant, extra=["--epochs", "2"]
    )
    assert code == 0
    summary = json.loads((directory / "training_summary.json").read_text("utf-8"))
    assert summary["model_variant"] == variant


# ---------------------------------------------------------------------------
# validate / inspect / resume
# ---------------------------------------------------------------------------


def test_validate_reproduces_the_best_checkpoint_metric(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, "gated", extra=["--epochs", "5"])
    summary = json.loads((directory / "training_summary.json").read_text("utf-8"))

    report_path = workspace / "validation.json"
    assert main(
        [
            "training",
            "validate",
            *base_args(workspace, labels),
            "--checkpoint",
            str(directory / "best_checkpoint.pt"),
            "--output",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ndcg_at_3"] == pytest.approx(
        summary["best_validation_ndcg_at_3"], abs=1e-6
    )
    assert report["model_variant"] == "gated"


def test_inspect_prints_the_checkpoint_metadata(workspace, capsys):
    labels = workspace.labels
    _, directory = train(workspace, labels, extra=["--epochs", "2"])
    capsys.readouterr()  # discard the training run's stdout before inspecting
    assert main(
        [
            "training",
            "inspect",
            "--checkpoint",
            str(directory / "best_checkpoint.pt"),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["model_variant"] == "handcrafted"
    assert printed["parameter_count"] > 0
    assert printed["resolved_device"] == "cpu"


def test_resume_continues_an_interrupted_run(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, run_dir="partial", extra=["--epochs", "2"])
    partial = json.loads((directory / "training_summary.json").read_text("utf-8"))
    assert partial["epochs_run"] == 2

    code = main(
        [
            "training",
            "resume",
            *base_args(workspace, labels),
            "--model",
            "handcrafted",
            "--run-dir",
            str(directory),
            "--checkpoint",
            str(directory / "last_checkpoint.pt"),
            "--epochs",
            "5",
        ]
    )
    assert code == 0
    resumed = json.loads((directory / "training_summary.json").read_text("utf-8"))
    assert resumed["epochs_run"] == 3, "resume should have run only the remaining epochs"
    lines = (directory / "epoch_metrics.jsonl").read_text("utf-8").splitlines()
    epochs = [json.loads(line)["epoch"] for line in lines]
    assert epochs == [1, 2, 3, 4, 5], f"epoch log should be continuous, got {epochs}"


def test_resuming_a_finished_run_is_a_no_op(workspace, capsys):
    labels = workspace.labels
    _, directory = train(workspace, labels, run_dir="done", extra=["--epochs", "2"])
    code = main(
        [
            "training",
            "resume",
            *base_args(workspace, labels),
            "--model",
            "handcrafted",
            "--run-dir",
            str(directory),
            "--checkpoint",
            str(directory / "last_checkpoint.pt"),
            "--epochs",
            "2",
        ]
    )
    assert code == 0
    assert "Nothing to do" in capsys.readouterr().out


def test_an_incompatible_checkpoint_is_refused(workspace, capsys):
    """Resuming with a model variant the checkpoint was not trained for fails.

    ``validate`` reads the variant from the checkpoint itself, so it cannot be
    given a wrong one; ``resume`` builds the model from ``--model`` and then
    checks it against the checkpoint, which is the CLI-reachable mismatch.
    """
    labels = workspace.labels
    _, directory = train(workspace, labels, "handcrafted", extra=["--epochs", "2"])
    code = main(
        [
            "training",
            "resume",
            *base_args(workspace, labels),
            "--model",
            "gated",
            "--run-dir",
            str(workspace / "resume-mismatch"),
            "--checkpoint",
            str(directory / "best_checkpoint.pt"),
        ]
    )
    assert code == 1
    assert "incompatible" in capsys.readouterr().err.lower()


def test_resuming_from_a_corrupt_checkpoint_exits_non_zero(workspace, capsys):
    labels = workspace.labels
    broken = workspace / "broken.pt"
    broken.write_bytes(b"definitely not a checkpoint")
    code = main(
        [
            "training",
            "resume",
            *base_args(workspace, labels),
            "--model",
            "handcrafted",
            "--checkpoint",
            str(broken),
        ]
    )
    assert code == 1
    assert capsys.readouterr().err.strip()


def test_a_missing_checkpoint_exits_non_zero(workspace):
    labels = workspace.labels
    assert main(
        [
            "training",
            "resume",
            *base_args(workspace, labels),
            "--model",
            "handcrafted",
            "--checkpoint",
            str(workspace / "absent.pt"),
        ]
    ) == 1


def test_a_missing_label_export_exits_non_zero(workspace):
    assert main(
        [
            "training",
            "prepare",
            "--data-root",
            str(workspace.data_root),
            "--labels",
            str(workspace / "nope.jsonl"),
        ]
    ) == 1


def test_disagreeing_auxiliary_head_settings_are_refused(workspace, capsys):
    labels = workspace.labels
    config = workspace / "no_aux.yaml"
    config.write_text(
        "model:\n  variant: gated\n  auxiliary_head: false\n", encoding="utf-8"
    )
    code = main(
        [
            "training",
            "run",
            *base_args(workspace, labels),
            "--model-config",
            str(config),
            "--run-dir",
            str(workspace / "run"),
        ]
    )
    assert code == 1
    assert "auxiliary_head" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models_list(capsys):
    assert main(["models", "list"]) == 0
    printed = capsys.readouterr().out
    for variant in ("handcrafted", "text_only", "audio_only", "concat", "gated"):
        assert variant in printed


def test_models_describe_reports_real_shape(capsys):
    assert main(["models", "describe", "--model", "gated"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["variant"] == "gated"
    assert described["trainable_parameters"] > 0
    assert described["trainable_parameters"] < 1_000_000
    assert described["input_schema"]["audio_dimension"] == 1536
    assert described["input_schema"]["text_dimension"] == 1536


def test_models_describe_rejects_an_unknown_variant():
    assert main(["models", "describe", "--model", "nope"]) == 1


# ---------------------------------------------------------------------------
# Artifact safety
# ---------------------------------------------------------------------------


def test_artifact_writes_leave_no_temporary_debris(workspace):
    """The Windows/OneDrive-safe atomic write must clean up after itself."""
    labels = workspace.labels
    _, directory = train(workspace, labels, extra=["--epochs", "2"])
    leftovers = [path.name for path in directory.iterdir() if path.name.endswith(".tmp")]
    assert not leftovers, f"atomic writes left temp files behind: {leftovers}"
    hidden = [path.name for path in directory.iterdir() if path.name.startswith(".")]
    assert not hidden, f"atomic writes left hidden temp files behind: {hidden}"


def test_every_json_artifact_is_valid_json(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, extra=["--epochs", "2"])
    for path in directory.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in directory.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)


def test_rerunning_replaces_artifacts_atomically(workspace):
    labels = workspace.labels
    _, directory = train(workspace, labels, run_dir="same", extra=["--epochs", "2"])
    first = (directory / "training_summary.json").read_text("utf-8")
    _, again = train(workspace, labels, run_dir="same", extra=["--epochs", "3"])
    assert again == directory
    second = (directory / "training_summary.json").read_text("utf-8")
    assert json.loads(second)["epochs_run"] == 3
    assert first != second
    json.loads(second)
