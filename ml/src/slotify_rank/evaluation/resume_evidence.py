"""The generated resume-evidence report.

One job: take each claim a resume might make about this repository and say
**PASS** or **FAIL**, with the artifact the verdict was read from.

Every number here is read out of a generated artifact. Nothing is typed, nothing
is rounded up, and there is no code path that produces a target value. In
particular:

* the improvement threshold comes from the committed experiment definition
  (``claim.minimum_relative_improvement_percent``), and the *measured*
  improvement comes from a published comparison artifact. This module compares
  them; it cannot influence either.
* a check whose evidence does not exist reports :data:`NOT_MEASURED`, which is
  distinct from a measured failure. "Nobody has measured this" and "this was
  measured and came out short" are different facts and stay different strings
  all the way to the markdown.
* library claims are not satisfied by a line in ``pyproject.toml``. Each one
  requires three independent things: the dependency is declared, at least one
  committed source module imports it, and a generated artifact records it
  actually running. A dependency added to a manifest and never used fails here,
  which is the point -- that is precisely the resume-driven dependency stuffing
  this report exists to catch.

The award check is a string check against the committed README, and it fails
both ways: the precise wording must be present, *and* the stronger claim
("hackathon winner", "first place", "grand prize") must be absent unless it is
separately evidenced.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data.checksum import atomic_write_bytes

__all__ = [
    "RESUME_EVIDENCE_SCHEMA_VERSION",
    "PASS",
    "FAIL",
    "NOT_MEASURED",
    "AWARD_WORDING",
    "OVERSTATED_AWARD_PATTERNS",
    "ResumeCheck",
    "ResumeEvidence",
    "collect_resume_evidence",
    "declared_dependencies",
    "importing_modules",
    "package_source_root",
    "package_pyproject",
    "render_markdown",
    "write_resume_evidence",
]

RESUME_EVIDENCE_SCHEMA_VERSION = "resume-evidence-v1.0.0"

PASS = "PASS"
FAIL = "FAIL"
#: No artifact produced this. Deliberately not ``FAIL``: a claim nobody measured
#: is a different state from a claim that was measured and fell short, and the
#: report must not let the two be confused.
NOT_MEASURED = "NOT MEASURED"

#: The exact award wording. Bare-ASCII and em-dash forms are both accepted so a
#: docs file written on a different keyboard does not fail the check.
AWARD_WORDING = "UofTHacks 13 — MLH Best Use of ElevenLabs"
_AWARD_VARIANTS = (
    "UofTHacks 13 — MLH Best Use of ElevenLabs",
    "UofTHacks 13 - MLH Best Use of ElevenLabs",
)

#: Claims stronger than the award actually won. Present in the README without
#: separate evidence, any of these fails the award check.
OVERSTATED_AWARD_PATTERNS: tuple[str, ...] = (
    r"\bhackathon winner\b",
    r"\bwon uofthacks\b",
    r"\bfirst place\b",
    r"\bgrand prize\b",
    r"\boverall winner\b",
    r"\bbest overall\b",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
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
        return NOT_MEASURED
    return result.stdout.strip() or NOT_MEASURED


def _display(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


#: This module's own package directory. The library claims are about the source
#: of *this* package, so when ``repo_root`` does not contain a checkout -- a
#: report generated against an artifacts tree elsewhere -- the scan falls back
#: here rather than reporting "no module imports torch" because it was looking
#: in the wrong place.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_source_root(repo_root: Path) -> Path:
    candidate = Path(repo_root) / "ml" / "src" / "slotify_rank"
    return candidate if candidate.is_dir() else _PACKAGE_ROOT


def package_pyproject(repo_root: Path) -> Path:
    candidate = Path(repo_root) / "ml" / "pyproject.toml"
    if candidate.is_file():
        return candidate
    return _PACKAGE_ROOT.parents[1] / "pyproject.toml"


# ---------------------------------------------------------------------------
# Library evidence
# ---------------------------------------------------------------------------


def declared_dependencies(pyproject: Path) -> set[str]:
    """Distribution names declared in ``ml/pyproject.toml``, extras included.

    Parsed rather than grepped, so ``scikit-learn>=1.4,<2.0`` and
    ``scikit_learn`` normalise to the same name and a commented-out line is not
    counted as a declaration.
    """
    if not Path(pyproject).is_file():
        return set()
    try:
        import tomllib

        data = tomllib.loads(Path(pyproject).read_text(encoding="utf-8"))
    except (ImportError, ValueError, OSError):
        return set()
    project = data.get("project") or {}
    requirements: list[str] = list(project.get("dependencies") or [])
    for extra in (project.get("optional-dependencies") or {}).values():
        requirements.extend(extra)
    names: set[str] = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~\[; ]", str(requirement).strip(), maxsplit=1)[0]
        if name:
            names.add(name.strip().lower().replace("_", "-"))
    return names


def importing_modules(source_root: Path, module: str) -> list[str]:
    """Committed modules that import ``module``, found by parsing, not grepping.

    An AST walk means a name inside a string, a comment or a docstring never
    counts as a use -- which matters here, because this file itself mentions
    every library by name.
    """
    root = Path(source_root)
    found: list[str] = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == module or name.startswith(f"{module}.") for name in names):
                found.append(path.relative_to(root).as_posix())
                break
    return found


@dataclass(frozen=True)
class LibraryClaim:
    """One "the stack includes X" claim and what would establish it."""

    key: str
    label: str
    #: Import name, e.g. ``sklearn``.
    module: str
    #: Distribution name as declared, e.g. ``scikit-learn``.
    distribution: str
    #: Human description of the artifact evidence required.
    artifact_requirement: str


LIBRARY_CLAIMS: tuple[LibraryClaim, ...] = (
    LibraryClaim(
        key="pytorch_used",
        label="PyTorch",
        module="torch",
        distribution="torch",
        artifact_requirement=(
            "a training run recorded a torch version and a parameter count"
        ),
    ),
    LibraryClaim(
        key="huggingface_used",
        label="Hugging Face `transformers`",
        module="transformers",
        distribution="transformers",
        artifact_requirement=(
            "the embedding statistics name Hugging Face model ids for both the "
            "speech and the transcript encoder"
        ),
    ),
    LibraryClaim(
        key="librosa_used",
        label="librosa",
        module="librosa",
        distribution="librosa",
        artifact_requirement=(
            "the feature statistics record a librosa version and the handcrafted "
            "vocabulary contains librosa-derived spectral descriptors"
        ),
    ),
    LibraryClaim(
        key="sklearn_used",
        label="scikit-learn, used meaningfully",
        module="sklearn",
        distribution="scikit-learn",
        artifact_requirement=(
            "a published comparison records BOTH an independent scikit-learn NDCG "
            "cross-check and a fitted scikit-learn classical baseline"
        ),
    ),
)

#: Handcrafted feature names that only exist because librosa computed them.
_LIBROSA_FEATURE_PREFIXES = (
    "spectral_",
    "onset_strength_",
)


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass
class ResumeCheck:
    """One resume claim, its verdict, and where the verdict came from."""

    key: str
    claim: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "claim": self.claim,
            "status": self.status,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "sources": list(self.sources),
        }


@dataclass
class ResumeEvidence:
    git_sha: str
    generated_at: str
    checks: list[ResumeCheck]
    measurements: dict[str, Any]
    artifacts_read: dict[str, str]
    artifacts_missing: list[str]

    @property
    def all_pass(self) -> bool:
        return all(check.status == PASS for check in self.checks)

    def check(self, key: str) -> ResumeCheck:
        for entry in self.checks:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "generated_at": self.generated_at,
            "git_sha": self.git_sha,
            "unavailable_marker": NOT_MEASURED,
            "all_claims_supported": self.all_pass,
            "summary": {
                "pass": sum(1 for c in self.checks if c.status == PASS),
                "fail": sum(1 for c in self.checks if c.status == FAIL),
                "not_measured": sum(1 for c in self.checks if c.status == NOT_MEASURED),
            },
            "measurements": dict(self.measurements),
            "checks": [check.to_dict() for check in self.checks],
            "artifacts_read": dict(self.artifacts_read),
            "artifacts_missing": list(self.artifacts_missing),
        }


# ---------------------------------------------------------------------------
# Artifact selection
# ---------------------------------------------------------------------------


def _published_comparison(
    evaluation_dir: Path,
) -> tuple[str, dict[str, Any]] | None:
    """The most recent comparison whose headline is publishable.

    A comparison that ran but was blocked is deliberately not returned; its
    numbers are surfaced separately as "measured but not publishable" so the
    pipeline stays auditable without the blocked number becoming the result.
    """
    if not Path(evaluation_dir).is_dir():
        return None
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(Path(evaluation_dir).glob("*/comparison.json")):
        payload = _read_json(path)
        if payload and payload.get("headline_publishable") is True:
            found.append((str(path), payload))
    if not found:
        return None
    found.sort(key=lambda entry: str(entry[1].get("generated_at", "")))
    return found[-1]


def _human_trained_runs(training_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    if not Path(training_dir).is_dir():
        return runs
    for path in sorted(Path(training_dir).glob("*/training_summary.json")):
        summary = _read_json(path)
        if summary and summary.get("label_source") == "human":
            runs.append((str(path), summary))
    runs.sort(key=lambda entry: str(entry[1].get("generated_at", "")))
    return runs


def _latest_run(training_dir: Path) -> tuple[str, dict[str, Any]] | None:
    runs: list[tuple[str, dict[str, Any]]] = []
    if not Path(training_dir).is_dir():
        return None
    for path in sorted(Path(training_dir).glob("*/training_summary.json")):
        summary = _read_json(path)
        if summary:
            runs.append((str(path), summary))
    if not runs:
        return None
    runs.sort(key=lambda entry: str(entry[1].get("generated_at", "")))
    return runs[-1]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def collect_resume_evidence(
    repo_root: Path,
    artifacts_root: Path | None = None,
    experiment_config_path: Path | None = None,
) -> ResumeEvidence:
    """Read every artifact and decide each resume claim's verdict."""
    root = Path(repo_root)
    artifacts = Path(artifacts_root) if artifacts_root else root / "artifacts"
    read: dict[str, str] = {}
    missing: list[str] = []

    def load(name: str, path: Path) -> dict[str, Any]:
        payload = _read_json(path)
        if payload is None:
            missing.append(_display(path, root))
            return {}
        read[name] = _display(path, root)
        return payload

    label_statistics = load(
        "label_statistics", artifacts / "dataset" / "label_statistics.json"
    )
    candidate_statistics = load(
        "candidate_statistics", artifacts / "dataset" / "candidate_statistics.json"
    )
    dataset_statistics = load(
        "dataset_statistics", artifacts / "dataset" / "dataset_statistics.json"
    )
    feature_statistics = load(
        "feature_statistics", artifacts / "features" / "feature_statistics.json"
    )
    embedding_statistics = load(
        "embedding_statistics", artifacts / "features" / "embedding_statistics.json"
    )
    readiness = load(
        "readiness_report", artifacts / "experiments" / "readiness_report.json"
    )

    # The committed experiment definition supplies the thresholds. They are read,
    # never defaulted here: a threshold with a fallback in this file would be a
    # resume number typed into the checker.
    config_path = (
        Path(experiment_config_path)
        if experiment_config_path
        else root / "ml" / "configs" / "experiment_resume_v1.yaml"
    )
    experiment_config: dict[str, Any] = {}
    experiment_error: str | None = None
    try:
        from slotify_rank.experiment.canonical import load_experiment_config

        parsed = load_experiment_config(config_path)
        experiment_config = parsed.to_dict()
        read["experiment_config"] = _display(config_path, root)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        experiment_error = f"{type(error).__name__}: {error}"
        missing.append(_display(config_path, root))

    experiment_manifest: dict[str, Any] = {}
    manifest_glob = sorted((artifacts / "experiments").glob("experiment-*.json"))
    if manifest_glob:
        experiment_manifest = _read_json(manifest_glob[-1]) or {}
        if experiment_manifest:
            read["experiment_manifest"] = _display(manifest_glob[-1], root)
    else:
        missing.append(
            _display(artifacts / "experiments" / "experiment-*.json", root)
        )

    published = _published_comparison(artifacts / "evaluation")
    comparison = published[1] if published else None
    if published:
        read["published_comparison"] = _display(Path(published[0]), root)

    human_runs = _human_trained_runs(artifacts / "training")
    latest_run = _latest_run(artifacts / "training")
    if human_runs:
        read["human_trained_run"] = _display(Path(human_runs[-1][0]), root)
    elif latest_run:
        read["latest_training_run"] = _display(Path(latest_run[0]), root)

    environment: dict[str, Any] = {}
    if latest_run:
        environment = _read_json(Path(latest_run[0]).parent / "environment.json") or {}

    # -- measurements ------------------------------------------------------
    def measured(source: Mapping[str, Any], key: str) -> Any:
        value = source.get(key) if source else None
        return NOT_MEASURED if value is None else value

    minimum_labels = (
        (experiment_config.get("claim") or {}).get("minimum_human_labels", NOT_MEASURED)
        if experiment_config
        else NOT_MEASURED
    )
    minimum_improvement = (
        (experiment_config.get("claim") or {}).get(
            "minimum_relative_improvement_percent", NOT_MEASURED
        )
        if experiment_config
        else NOT_MEASURED
    )

    headline = (comparison or {}).get("headline") or {}
    measurements: dict[str, Any] = {
        "human_labelled_candidate_count": measured(
            label_statistics, "human_labelled_candidate_count"
        ),
        "generated_candidate_count": measured(
            candidate_statistics, "generated_candidate_count"
        ),
        "complete_multimodal_candidate_count": measured(
            (feature_statistics.get("candidates") or {}), "complete_multimodal"
        ),
        "processed_audio_hours": measured(dataset_statistics, "processed_audio_hours"),
        "processed_episode_count": measured(
            dataset_statistics, "processed_episode_count"
        ),
        "series_count": measured(dataset_statistics, "series_count"),
        "audio_embedding_model": measured(
            embedding_statistics, "audio_embedding_model"
        ),
        "text_embedding_model": measured(embedding_statistics, "text_embedding_model"),
        "model_variant": measured(
            (human_runs[-1][1] if human_runs else (latest_run[1] if latest_run else {})),
            "model_variant",
        ),
        "model_parameter_count": measured(
            (human_runs[-1][1] if human_runs else (latest_run[1] if latest_run else {})),
            "model_parameter_count",
        ),
        "model_training_label_source": measured(
            (human_runs[-1][1] if human_runs else (latest_run[1] if latest_run else {})),
            "label_source",
        ),
        "baseline_ndcg_at_3": (
            headline.get("baseline", {}).get("ndcg_at_3", NOT_MEASURED)
            if comparison
            else NOT_MEASURED
        ),
        "model_ndcg_at_3": (
            headline.get("model", {}).get("ndcg_at_3", NOT_MEASURED)
            if comparison
            else NOT_MEASURED
        ),
        "absolute_improvement": (
            headline.get("absolute_improvement", NOT_MEASURED)
            if comparison
            else NOT_MEASURED
        ),
        "relative_improvement_percent": (
            headline.get("relative_improvement_percent", NOT_MEASURED)
            if comparison
            else NOT_MEASURED
        ),
        "evaluation_id": (
            comparison.get("evaluation_id", NOT_MEASURED)
            if comparison
            else NOT_MEASURED
        ),
        "required_minimum_human_labels": minimum_labels,
        "required_minimum_relative_improvement_percent": minimum_improvement,
        "phase_5b_ready": readiness.get("phase_5b_ready", NOT_MEASURED),
    }
    if comparison:
        measurements["bootstrap"] = comparison.get("bootstrap")
        measurements["cohort"] = comparison.get("cohort")
        measurements["metric_crosscheck"] = comparison.get("metric_crosscheck")
        measurements["classical_baseline"] = comparison.get("classical_baseline")

    checks: list[ResumeCheck] = []

    def add(
        key: str,
        claim: str,
        status: str,
        detail: str,
        evidence: Mapping[str, Any] | None = None,
        sources: Sequence[str] = (),
    ) -> None:
        checks.append(
            ResumeCheck(
                key=key,
                claim=claim,
                status=status,
                detail=detail,
                evidence=dict(evidence or {}),
                sources=list(sources),
            )
        )

    if experiment_error:
        add(
            "experiment_definition",
            "A committed, versioned experiment definition fixes the protocol in "
            "advance.",
            FAIL,
            f"{config_path} could not be read: {experiment_error}",
            sources=[_display(config_path, root)],
        )
    else:
        add(
            "experiment_definition",
            "A committed, versioned experiment definition fixes the protocol in "
            "advance.",
            PASS,
            f"{experiment_config['experiment_version']} declares the split, the "
            f"seeds, the headline variant, the canonical baseline and the claim "
            "thresholds this report is checked against.",
            {
                "experiment_version": experiment_config["experiment_version"],
                "split_version": (experiment_config.get("split") or {}).get("version"),
                "seeds": (experiment_config.get("model") or {}).get("seeds"),
                "headline_variant": (experiment_config.get("model") or {}).get(
                    "headline_variant"
                ),
            },
            [read.get("experiment_config", "")],
        )

    # -- 1. multimodal PyTorch ranker exists -------------------------------
    run_summary = human_runs[-1][1] if human_runs else (latest_run[1] if latest_run else {})
    has_run = bool(run_summary.get("model_variant"))
    add(
        "multimodal_ranker_exists",
        "A multimodal PyTorch ranker exists and is trained.",
        PASS if has_run else NOT_MEASURED,
        (
            f"{run_summary.get('model_variant')} with "
            f"{run_summary.get('model_parameter_count')} parameters, trained on "
            f"label_source={run_summary.get('label_source')}."
            if has_run
            else "No training run summary was found."
        ),
        {
            "model_variant": measurements["model_variant"],
            "model_parameter_count": measurements["model_parameter_count"],
            "training_label_source": measurements["model_training_label_source"],
        },
        [read.get("human_trained_run") or read.get("latest_training_run", "")],
    )

    # -- 2. both modalities are used ---------------------------------------
    audio_model = measurements["audio_embedding_model"]
    text_model = measurements["text_embedding_model"]
    both = audio_model != NOT_MEASURED and text_model != NOT_MEASURED
    add(
        "audio_and_transcript_modalities",
        "The ranker consumes both audio and transcript features.",
        PASS if both else NOT_MEASURED,
        (
            f"Speech representations from {audio_model}; transcript embeddings "
            f"from {text_model}; plus "
            f"{(feature_statistics.get('features') or {}).get('handcrafted_feature_count')} "
            "handcrafted acoustic and structural scalars."
            if both
            else "The embedding statistics do not name both encoders."
        ),
        {
            "audio_embedding_model": audio_model,
            "text_embedding_model": text_model,
            "handcrafted_feature_count": (
                feature_statistics.get("features") or {}
            ).get("handcrafted_feature_count", NOT_MEASURED),
            "complete_multimodal_candidate_count": measurements[
                "complete_multimodal_candidate_count"
            ],
        },
        [read.get("embedding_statistics", ""), read.get("feature_statistics", "")],
    )

    # -- 3. label volume ----------------------------------------------------
    labels_measured = measurements["human_labelled_candidate_count"]
    if labels_measured == NOT_MEASURED or minimum_labels == NOT_MEASURED:
        label_status = NOT_MEASURED
        label_detail = (
            "Either the label statistics or the experiment's label gate is absent, "
            "so the count could not be compared against the requirement."
        )
    elif int(labels_measured) >= int(minimum_labels):
        label_status = PASS
        label_detail = (
            f"{labels_measured} human-labelled candidates, at or above the "
            f"{minimum_labels} the experiment requires."
        )
    else:
        label_status = FAIL
        label_detail = (
            f"{labels_measured} human-labelled candidates, short of the "
            f"{minimum_labels} the experiment requires. "
            f"{measurements['generated_candidate_count']} candidates have been "
            "GENERATED, which is not the same thing and is never counted as one."
        )
    add(
        "human_label_volume",
        f"At least {minimum_labels} human-labelled candidates exist.",
        label_status,
        label_detail,
        {
            "human_labelled_candidate_count": labels_measured,
            "required": minimum_labels,
            "generated_candidate_count": measurements["generated_candidate_count"],
        },
        [read.get("label_statistics", ""), read.get("candidate_statistics", "")],
    )

    # -- 4. human-trained checkpoint ---------------------------------------
    add(
        "human_trained_checkpoint",
        "The evaluated checkpoint was trained on human labels, not weak ones.",
        PASS if human_runs else (FAIL if latest_run else NOT_MEASURED),
        (
            f"{len(human_runs)} training run(s) record label_source=human; the most "
            f"recent is {human_runs[-1][1].get('run_id')}."
            if human_runs
            else (
                "Every training run records a non-human label source "
                f"({run_summary.get('label_source')}), so no checkpoint reflects "
                "human judgement."
                if latest_run
                else "No training run was found."
            )
        ),
        {
            "human_trained_run_count": len(human_runs),
            "latest_training_label_source": measurements[
                "model_training_label_source"
            ],
        },
        [read.get("human_trained_run") or read.get("latest_training_run", "")],
    )

    # -- 5. frozen held-out test set ---------------------------------------
    split_block = experiment_manifest.get("split") or {}
    groups = split_block.get("groups_by_split") or {}
    frozen = bool(
        split_block
        and not split_block.get("degraded")
        and groups.get("test")
        and groups.get("train")
        and groups.get("validation")
        and not (set(groups.get("test", [])) & set(groups.get("train", [])))
        and not (set(groups.get("test", [])) & set(groups.get("validation", [])))
    )
    add(
        "frozen_held_out_test_set",
        "A grouped, non-degraded train/validation/test split exists and its test "
        "groups appear in no other partition.",
        PASS if frozen else (FAIL if split_block else NOT_MEASURED),
        (
            f"Split {split_block.get('version')} groups by "
            f"{split_block.get('group_by')} with seed {split_block.get('seed')}: "
            f"{len(groups.get('train', []))} train / "
            f"{len(groups.get('validation', []))} validation / "
            f"{len(groups.get('test', []))} test series, disjoint."
            if frozen
            else (
                "The resolved split is degraded, incomplete, or shares groups "
                "between partitions."
                if split_block
                else "No experiment manifest has been resolved."
            )
        ),
        {
            "split_version": split_block.get("version", NOT_MEASURED),
            "group_by": split_block.get("group_by", NOT_MEASURED),
            "degraded": split_block.get("degraded", NOT_MEASURED),
            "series_by_split": {k: len(v) for k, v in groups.items()},
            "split_manifest_hash": (
                (experiment_manifest.get("artifact_hashes") or {}).get("split_manifest")
                or NOT_MEASURED
            ),
        },
        [read.get("experiment_manifest", "")],
    )

    # -- 6. canonical baseline ---------------------------------------------
    declared_baseline = (experiment_config.get("baselines") or {}).get("canonical")
    used_baseline = ((comparison or {}).get("inputs") or {}).get("baseline_version")
    baseline_ok = bool(
        declared_baseline and used_baseline and declared_baseline == used_baseline
    )
    add(
        "canonical_baseline",
        "The published improvement is quoted against the canonical frozen "
        "heuristic baseline.",
        PASS if baseline_ok else (FAIL if used_baseline else NOT_MEASURED),
        (
            f"The comparison used {used_baseline}, which is the baseline the "
            "experiment declares."
            if baseline_ok
            else (
                f"The comparison used {used_baseline!r} but the experiment declares "
                f"{declared_baseline!r}."
                if used_baseline
                else "No publishable comparison exists, so no baseline was used."
            )
        ),
        {
            "declared_baseline": declared_baseline or NOT_MEASURED,
            "comparison_baseline": used_baseline or NOT_MEASURED,
            "baseline_config_hash": (
                (experiment_manifest.get("artifact_hashes") or {}).get(
                    "baseline_config"
                )
                or NOT_MEASURED
            ),
        },
        [read.get("published_comparison", ""), read.get("experiment_manifest", "")],
    )

    # -- 7/8/9. the three measured numbers ---------------------------------
    for key, label, measurement_key in (
        (
            "measured_baseline_ndcg",
            "The baseline's held-out NDCG@3 is measured.",
            "baseline_ndcg_at_3",
        ),
        (
            "measured_model_ndcg",
            "The learned model's held-out NDCG@3 is measured.",
            "model_ndcg_at_3",
        ),
        (
            "measured_relative_improvement",
            "The relative NDCG@3 improvement is measured.",
            "relative_improvement_percent",
        ),
    ):
        value = measurements[measurement_key]
        add(
            key,
            label,
            PASS if isinstance(value, (int, float)) else NOT_MEASURED,
            (
                f"{value}"
                if isinstance(value, (int, float))
                else "No publishable held-out comparison exists."
            ),
            {measurement_key: value, "evaluation_id": measurements["evaluation_id"]},
            [read.get("published_comparison", "")],
        )

    # -- 10. does the improvement clear the claimed threshold? -------------
    improvement = measurements["relative_improvement_percent"]
    if not isinstance(improvement, (int, float)) or minimum_improvement == NOT_MEASURED:
        threshold_status = NOT_MEASURED
        threshold_detail = (
            "No measured improvement, or no declared threshold, so the claim "
            "cannot be evaluated. It is NOT supported by default."
        )
    elif float(improvement) >= float(minimum_improvement):
        threshold_status = PASS
        threshold_detail = (
            f"Measured {improvement:.2f} %, at or above the claimed "
            f"{minimum_improvement} %."
        )
    else:
        threshold_status = FAIL
        threshold_detail = (
            f"Measured {improvement:.2f} %, below the claimed "
            f"{minimum_improvement} %. The resume claim is NOT supported. The "
            "measured number is the result; the threshold is not a target the "
            "pipeline may be steered toward."
        )
    bootstrap = (comparison or {}).get("bootstrap") or {}
    interval = (
        bootstrap.get("relative_improvement_percent")
        if bootstrap.get("measured")
        else None
    )
    add(
        "improvement_meets_claim",
        f"The relative NDCG@3 improvement is at least "
        f"{minimum_improvement} % over the heuristic baseline.",
        threshold_status,
        threshold_detail,
        {
            "measured_relative_improvement_percent": improvement,
            "required_relative_improvement_percent": minimum_improvement,
            "bootstrap_interval": interval or NOT_MEASURED,
        },
        [read.get("published_comparison", ""), read.get("experiment_config", "")],
    )

    # -- 11-14. library claims ---------------------------------------------
    pyproject = package_pyproject(root)
    declared = declared_dependencies(pyproject)
    source_root = package_source_root(root)
    feature_libraries = feature_statistics.get("library_versions") or {}
    handcrafted_names = list(
        (feature_statistics.get("features") or {}).get("handcrafted_feature_names", [])
    )
    crosscheck = (comparison or {}).get("metric_crosscheck") or {}
    classical = (comparison or {}).get("classical_baseline") or {}

    for library in LIBRARY_CLAIMS:
        is_declared = library.distribution.lower() in declared
        importers = importing_modules(source_root, library.module)
        artifact_evidence: dict[str, Any] = {}
        artifact_ok = False

        if library.key == "pytorch_used":
            torch_version = (environment.get("dependency_versions") or {}).get("torch")
            artifact_evidence = {
                "training_torch_version": torch_version or NOT_MEASURED,
                "model_parameter_count": measurements["model_parameter_count"],
            }
            artifact_ok = bool(torch_version) and isinstance(
                measurements["model_parameter_count"], int
            )
        elif library.key == "huggingface_used":
            artifact_evidence = {
                "audio_embedding_model": audio_model,
                "text_embedding_model": text_model,
                "feature_pipeline_transformers_version": feature_libraries.get(
                    "transformers", NOT_MEASURED
                ),
            }
            artifact_ok = both
        elif library.key == "librosa_used":
            version = feature_libraries.get("librosa")
            derived = [
                name
                for name in handcrafted_names
                if name.startswith(_LIBROSA_FEATURE_PREFIXES)
            ]
            artifact_evidence = {
                "feature_pipeline_librosa_version": version or NOT_MEASURED,
                "librosa_derived_feature_count": len(derived),
                "example_features": derived[:4],
            }
            artifact_ok = bool(version) and version != "absent" and bool(derived)
        elif library.key == "sklearn_used":
            artifact_evidence = {
                "metric_crosscheck_available": crosscheck.get(
                    "available", NOT_MEASURED
                ),
                "metric_crosscheck_agrees": crosscheck.get("agrees", NOT_MEASURED),
                "metric_crosscheck_sklearn_version": crosscheck.get(
                    "sklearn_version", NOT_MEASURED
                ),
                "classical_baseline_available": classical.get(
                    "available", NOT_MEASURED
                ),
                "classical_baseline_ndcg_at_3": classical.get(
                    "ndcg_at_3", NOT_MEASURED
                ),
            }
            artifact_ok = bool(
                crosscheck.get("available")
                and crosscheck.get("agrees")
                and classical.get("available")
            )

        if not is_declared:
            status, detail = FAIL, (
                f"{library.distribution} is not declared in ml/pyproject.toml."
            )
        elif not importers:
            status, detail = FAIL, (
                f"{library.distribution} is declared but no committed module under "
                "ml/src/slotify_rank/ imports it. A dependency nobody imports is "
                "not part of the stack."
            )
        elif artifact_ok:
            status, detail = PASS, (
                f"Declared, imported by {len(importers)} module(s) "
                f"(e.g. {', '.join(importers[:3])}), and "
                f"{library.artifact_requirement}."
            )
        else:
            status, detail = NOT_MEASURED, (
                f"Declared and imported by {len(importers)} module(s), but no "
                f"artifact yet shows it running: {library.artifact_requirement}."
            )

        add(
            library.key,
            f"The AI/ML stack includes {library.label}.",
            status,
            detail,
            {
                "declared_in_pyproject": is_declared,
                "importing_modules": importers[:8],
                "importing_module_count": len(importers),
                **artifact_evidence,
            },
            [
                _display(pyproject, root),
                read.get("feature_statistics", ""),
                read.get("published_comparison", ""),
            ],
        )

    # -- 15. award wording --------------------------------------------------
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    exact = any(variant in readme for variant in _AWARD_VARIANTS)
    overstated = [
        pattern
        for pattern in OVERSTATED_AWARD_PATTERNS
        if re.search(pattern, readme, flags=re.IGNORECASE)
    ]
    if not readme:
        award_status, award_detail = NOT_MEASURED, "README.md could not be read."
    elif exact and not overstated:
        award_status, award_detail = PASS, (
            f"README.md states {AWARD_WORDING!r} and makes no stronger claim."
        )
    elif not exact:
        award_status, award_detail = FAIL, (
            f"README.md does not contain the precise wording {AWARD_WORDING!r}."
        )
    else:
        award_status, award_detail = FAIL, (
            "README.md claims more than the award evidences; matched "
            f"{overstated}."
        )
    add(
        "award_wording",
        f"The award is described precisely as {AWARD_WORDING!r}, with no implied "
        "overall win.",
        award_status,
        award_detail,
        {
            "exact_wording_present": exact,
            "overstated_patterns_matched": overstated,
        },
        ["README.md"],
    )

    return ResumeEvidence(
        git_sha=_git_sha(root),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checks=checks,
        measurements=measurements,
        artifacts_read=read,
        artifacts_missing=missing,
    )


def _value(measurements: Mapping[str, Any], key: str) -> str:
    value = measurements.get(key, NOT_MEASURED)
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Generated from the same dict as the JSON. A test asserts they agree."""
    measurements = payload["measurements"]
    summary = payload["summary"]
    lines = [
        "# Resume evidence",
        "",
        "**Generated file - do not edit.** Produced by `slotify-rank report "
        "resume-evidence` (`npm run resume-evidence`), which reads the artifacts "
        "named at the bottom. Every number here comes from one of them; the "
        "thresholds come from the committed experiment definition.",
        "",
        f"- Git SHA: `{payload['git_sha']}`",
        f"- Generated: {payload['generated_at']}",
        f"- Package version: {payload['package_version']}",
        f"- Verdict: **{summary['pass']} PASS / {summary['fail']} FAIL / "
        f"{summary['not_measured']} {NOT_MEASURED}**",
        "",
        f"`{NOT_MEASURED}` means no artifact produced that value. It is not a "
        "failure and it is not a zero: a claim nobody has measured and a claim "
        "measured and found short are different facts.",
        "",
        "## Checklist",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check['claim']} | **{check['status']}** | {detail} |")

    lines += [
        "",
        "## Measured quantities",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Human-labelled candidates | {_value(measurements, 'human_labelled_candidate_count')} |",
        f"| ...required by the experiment | {_value(measurements, 'required_minimum_human_labels')} |",
        f"| Generated candidates (NOT labels) | {_value(measurements, 'generated_candidate_count')} |",
        f"| Complete multimodal feature records | {_value(measurements, 'complete_multimodal_candidate_count')} |",
        f"| Processed audio hours | {_value(measurements, 'processed_audio_hours')} |",
        f"| Episodes | {_value(measurements, 'processed_episode_count')} |",
        f"| Series | {_value(measurements, 'series_count')} |",
        f"| Model variant | {_value(measurements, 'model_variant')} |",
        f"| Model parameters | {_value(measurements, 'model_parameter_count')} |",
        f"| Trained on label source | {_value(measurements, 'model_training_label_source')} |",
        f"| Audio encoder | {_value(measurements, 'audio_embedding_model')} |",
        f"| Text encoder | {_value(measurements, 'text_embedding_model')} |",
        "",
        "## Held-out result",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation id | {_value(measurements, 'evaluation_id')} |",
        f"| Baseline NDCG@3 | {_value(measurements, 'baseline_ndcg_at_3')} |",
        f"| Model NDCG@3 | {_value(measurements, 'model_ndcg_at_3')} |",
        f"| Absolute improvement | {_value(measurements, 'absolute_improvement')} |",
        f"| Relative improvement | {_value(measurements, 'relative_improvement_percent')} |",
        f"| ...required by the claim | {_value(measurements, 'required_minimum_relative_improvement_percent')} |",
        "",
    ]

    bootstrap = measurements.get("bootstrap") or {}
    if bootstrap.get("measured"):
        interval = bootstrap.get("relative_improvement_percent") or {}
        lines += [
            f"Bootstrap ({bootstrap['config']['resamples']} resamples over "
            f"{bootstrap['episode_count']} episodes, "
            f"{int(bootstrap['config']['confidence'] * 100)}% percentile interval): "
            f"relative improvement in "
            f"[{interval.get('low', float('nan')):.2f}, "
            f"{interval.get('high', float('nan')):.2f}] %.",
            "",
        ]

    cohort = measurements.get("cohort") or {}
    if cohort:
        lines += [
            "### Split sizes",
            "",
            "| Split | Labelled candidates | Episodes | Series |",
            "| --- | --- | --- | --- |",
        ]
        for split in sorted(cohort.get("candidates_by_split") or {}):
            lines.append(
                f"| {split} | {cohort['candidates_by_split'][split]} "
                f"| {(cohort.get('episodes_by_split') or {}).get(split, 0)} "
                f"| {(cohort.get('series_by_split') or {}).get(split, 0)} |"
            )
        lines.append("")

    lines += ["## Detail", ""]
    for check in payload["checks"]:
        lines += [
            f"### {check['claim']} - {check['status']}",
            "",
            check["detail"],
            "",
            "```json",
            json.dumps(check["evidence"], indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]

    lines += ["## Artifacts read", ""]
    for name, path in sorted(payload["artifacts_read"].items()):
        if path:
            lines.append(f"- `{name}`: `{path}`")
    if payload["artifacts_missing"]:
        lines += ["", "## Artifacts missing", ""]
        for path in payload["artifacts_missing"]:
            lines.append(f"- `{path}`")

    return "\n".join(lines) + "\n"


def write_resume_evidence(
    directory: Path, evidence: ResumeEvidence
) -> dict[str, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = evidence.to_dict()
    json_path = target / "resume_evidence.json"
    md_path = target / "resume_evidence.md"
    atomic_write_bytes(
        json_path,
        (json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        ),
    )
    atomic_write_bytes(md_path, render_markdown(payload).encode("utf-8"))
    return {"resume_evidence.json": json_path, "resume_evidence.md": md_path}
