"""``slotify-rank evaluation`` -- the held-out comparison and its artifacts.

    slotify-rank evaluation compare \\
        --labels data/labels/labels_v1.jsonl \\
        --model artifacts/training/<run>/best_checkpoint.pt \\
        --baseline heuristic_offline_v1 \\
        --split test

Writes ``artifacts/evaluation/<evaluation_id>/`` containing metrics.json,
per_episode_metrics.json, ranked_candidates.jsonl, comparison.json and
summary.md. The evaluation id is content-addressed over the inputs, so re-running
the same comparison overwrites its own directory rather than accumulating
near-duplicates.

Exit code 1 when ``--require-publishable`` is passed and the headline is not
publishable, so a CI job can assert the claim still holds rather than assuming
it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.data.paths import DataPaths
from slotify_rank.evaluation.compare import CANONICAL_BASELINE

__all__ = ["register"]


def _paths(args: argparse.Namespace) -> DataPaths:
    paths = DataPaths(data_root=Path(args.data_root) if args.data_root else None)
    paths.mkdirs()
    return paths


def _training_episode_ids(checkpoint: Path) -> list[str]:
    """Episodes the model was trained or validated on, from the run's own report.

    Used for the leakage check. A missing summary means the check cannot run,
    which is itself reported -- absence of evidence is not evidence of no leak.
    """
    summary_path = checkpoint.parent / "dataset_summary.json"
    if not summary_path.is_file():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    episodes: set[str] = set()
    for key in ("train_pairs", "validation_pairs"):
        block = summary.get(key) or {}
        episodes.update((block.get("pairs_per_episode") or {}).keys())
        episodes.update(block.get("episodes_without_pairs") or [])
    return sorted(episodes)


def _evaluation_id(parts: Sequence[str]) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"eval-{digest}"


def _cmd_compare(args: argparse.Namespace) -> int:
    from slotify_rank.data import manifests
    from slotify_rank.data.splits import read_split_manifest
    from slotify_rank.datasets.loader import EligibilityConfig, build_examples
    from slotify_rank.datasets.labels import read_label_export
    from slotify_rank.evaluation.compare import (
        ComparisonInputs,
        compare,
        write_comparison,
    )
    from slotify_rank.features.assemble import read_feature_manifest
    from slotify_rank.inference.predictor import RankerPredictor
    from slotify_rank.training.config import git_commit

    paths = _paths(args)
    checkpoint = Path(args.model)
    label_export = Path(args.labels)

    allowed = tuple(dict.fromkeys(["human", *(args.allow_label_source or [])]))
    labels = read_label_export(label_export, allowed_label_sources=allowed)

    header, records = read_feature_manifest(
        Path(args.features_manifest) if args.features_manifest else paths.features_manifest
    )
    split_manifest = paths.split_manifest(args.split_version)
    split_lookup = (
        dict(read_split_manifest(split_manifest).by_episode)
        if split_manifest.is_file()
        else {}
    )
    candidates = (
        manifests.read_candidates(paths.candidates_manifest)
        if paths.candidates_manifest.is_file()
        else None
    )

    loaded = build_examples(
        paths=paths,
        header=header,
        records=records,
        labels=labels,
        config=EligibilityConfig(
            require_labels=True,
            require_acceptability_label=False,
            splits=(args.split,),
        ),
        candidates=candidates,
        split_lookup=split_lookup,
    )
    examples = loaded.examples
    if not examples:
        print(
            f"error: no labelled candidate in the {args.split!r} split. "
            f"Exclusions: {dict(loaded.counts)}",
            file=sys.stderr,
        )
        return 1

    # The baseline's score is heuristic_offline_v1's own, computed by the
    # canonical scorer at candidate-generation time and carried on the record.
    # It is read, never recomputed here, so the denominator cannot drift.
    by_id = {candidate.candidate_id: candidate for candidate in (candidates or ())}
    baseline_scores: dict[str, float] = {}
    baseline_versions: set[str] = set()
    config_versions: set[str] = set()
    for example in examples:
        candidate = by_id.get(example.candidate_id)
        if candidate is None or candidate.heuristic_score is None:
            print(
                f"error: {example.candidate_id} carries no heuristic score, so the "
                "baseline cannot be evaluated on the same candidate set.",
                file=sys.stderr,
            )
            return 1
        baseline_scores[example.candidate_id] = float(candidate.heuristic_score)
        baseline_versions.add(candidate.baseline_version)
        config_versions.add(candidate.config_version)

    if len(baseline_versions) != 1:
        print(
            f"error: candidates carry mixed baseline versions {sorted(baseline_versions)}; "
            "regenerate them before comparing.",
            file=sys.stderr,
        )
        return 1

    predictor = RankerPredictor.load(checkpoint, args.normalizer)
    ranking, _ = predictor.score(examples)
    model_scores = {
        example.candidate_id: score for example, score in zip(examples, ranking)
    }

    commit = git_commit(paths.repo_root)
    inputs = ComparisonInputs(
        split=args.split,
        label_source=labels.label_source,
        label_export=str(label_export),
        features_manifest=str(paths.features_manifest),
        split_manifest=str(split_manifest),
        split_version=args.split_version,
        model_run_id=predictor.identity.model_run_id,
        model_variant=predictor.identity.model_variant,
        model_checkpoint=str(checkpoint),
        model_training_label_source=predictor.identity.training_label_source,
        baseline_version=sorted(baseline_versions)[0],
        baseline_config_version=sorted(config_versions)[0],
        git_sha=commit,
        dataset_version=(loaded.schema.dataset_version if loaded.schema else ""),
        seeds=tuple(args.seed or ()),
    )

    evaluation_id = args.evaluation_id or _evaluation_id(
        [
            inputs.model_run_id,
            inputs.split,
            inputs.split_version,
            inputs.label_source,
            inputs.baseline_version,
        ]
    )

    training_episodes = _training_episode_ids(checkpoint)
    result = compare(
        examples=examples,
        baseline_scores=baseline_scores,
        model_scores=model_scores,
        inputs=inputs,
        evaluation_id=evaluation_id,
        training_episode_ids=training_episodes,
        relevance_threshold=args.relevance_threshold,
    )
    if not training_episodes:
        result.warnings.append(
            "the training run's episode list was unavailable, so the leakage check "
            "could not be performed."
        )

    directory = (
        Path(args.output_dir)
        if args.output_dir
        else paths.data_root.parent / "artifacts" / "evaluation" / evaluation_id
    )
    written = write_comparison(directory, result)

    headline = result.headline()
    print(f"Evaluation {evaluation_id}")
    print(f"  episodes: {result.episode_count}  candidates: {result.candidate_count}")
    print(f"  baseline NDCG@3: {headline['baseline']['ndcg_at_3']}")
    print(f"  model    NDCG@3: {headline['model']['ndcg_at_3']}")
    improvement = headline["relative_improvement_percent"]
    print(
        "  relative improvement: "
        + ("undefined" if improvement is None else f"{improvement:.2f} %")
    )
    if result.publishable:
        print("  HEADLINE PUBLISHABLE")
    else:
        print("  HEADLINE NOT PUBLISHABLE:")
        for reason in result.blocking_reasons:
            print(f"    - {reason}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for name, path in sorted(written.items()):
        print(f"  wrote {path}")

    if args.require_publishable and not result.publishable:
        return 1
    return 0


def _cmd_resume_evidence(args: argparse.Namespace) -> int:
    from slotify_rank.evaluation.evidence import collect_evidence, write_evidence

    paths = _paths(args)
    artifacts_root = (
        Path(args.artifacts_root)
        if args.artifacts_root
        else paths.data_root.parent / "artifacts"
    )
    evidence = collect_evidence(paths.repo_root, artifacts_root)
    directory = (
        Path(args.output_dir) if args.output_dir else artifacts_root / "reports"
    )
    written = write_evidence(directory, evidence)

    print(f"Resume evidence at git {evidence.git_sha}")
    for claim in evidence.claims:
        print(f"  Claim {claim.claim_id}: {claim.verdict}")
    for key in (
        "human_labelled_candidate_count",
        "generated_candidate_count",
        "processed_audio_hours",
        "baseline_ndcg_at_3",
        "model_ndcg_at_3",
        "relative_improvement_percent",
    ):
        print(f"  {key}: {evidence.measurements.get(key)}")
    for path in written.values():
        print(f"  wrote {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluation", help="Held-out comparison against the canonical baseline."
    )
    sub = parser.add_subparsers(dest="evaluation_command", required=True)

    compare_parser = sub.add_parser(
        "compare", help="Compare a trained model against heuristic_offline_v1."
    )
    compare_parser.add_argument("--data-root", default=None)
    compare_parser.add_argument("--labels", required=True, help="Label export JSONL.")
    compare_parser.add_argument("--model", required=True, help="Checkpoint to evaluate.")
    compare_parser.add_argument("--normalizer", default=None)
    compare_parser.add_argument(
        "--baseline",
        default=CANONICAL_BASELINE,
        help=f"Recorded for the report; the canonical denominator is {CANONICAL_BASELINE}.",
    )
    compare_parser.add_argument("--split", default="test")
    compare_parser.add_argument("--split-version", default="v2", dest="split_version")
    compare_parser.add_argument(
        "--features-manifest", default=None, dest="features_manifest"
    )
    compare_parser.add_argument(
        "--allow-label-source",
        dest="allow_label_source",
        action="append",
        metavar="SOURCE",
        help="Evaluate against a non-human ground truth. The comparison still "
        "runs, but the headline is marked NOT PUBLISHABLE and the reason is "
        "written into the artifact.",
    )
    compare_parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        dest="relevance_threshold",
    )
    compare_parser.add_argument(
        "--seed", type=int, action="append", help="Training seed(s) behind this model."
    )
    compare_parser.add_argument("--evaluation-id", default=None, dest="evaluation_id")
    compare_parser.add_argument("--output-dir", default=None, dest="output_dir")
    compare_parser.add_argument(
        "--require-publishable",
        action="store_true",
        dest="require_publishable",
        help="Exit non-zero when the headline is not publishable.",
    )
    compare_parser.set_defaults(func=_cmd_compare)

    report_parser = subparsers.add_parser(
        "report", help="Generated evidence reports."
    )
    report_sub = report_parser.add_subparsers(dest="report_command", required=True)
    evidence_parser = report_sub.add_parser(
        "resume-evidence",
        help="Read every artifact and report which resume claims they support.",
    )
    evidence_parser.add_argument("--data-root", default=None)
    evidence_parser.add_argument(
        "--artifacts-root", default=None, dest="artifacts_root"
    )
    evidence_parser.add_argument("--output-dir", default=None, dest="output_dir")
    evidence_parser.set_defaults(func=_cmd_resume_evidence)
