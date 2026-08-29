"""``experiment`` commands: the Phase 5B readiness gate and the label freeze.

Registered onto the Phase 1 parser by :func:`register`, following the same
conventions as every other command group: a human summary on stdout, optional
machine-readable JSON to ``--output``, exit 0 on success and 1 on a handled
failure. ``experiment readiness --require-ready`` additionally exits non-zero
when the gate is not met, so a training entry point can gate on it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slotify_rank.data import manifests
from slotify_rank.data.checksum import sha256_file
from slotify_rank.data.paths import DataPaths
from slotify_rank.datasets.labels import AggregatedLabel, aggregate_labels
from slotify_rank.labelling.database import DEFAULT_ACCEPTABLE_THRESHOLD, LabelDatabase
from slotify_rank.labelling.export import build_export_rows

__all__ = ["register"]


def _paths(args: argparse.Namespace) -> DataPaths:
    paths = DataPaths(data_root=Path(args.data_root) if args.data_root else None)
    paths.mkdirs()
    return paths


def _feature_status_by_id(paths: DataPaths) -> dict[str, str]:
    if not paths.features_manifest.is_file():
        return {}
    from slotify_rank.features.assemble import read_feature_manifest

    _, records = read_feature_manifest(paths.features_manifest)
    return {record.candidate_id: record.feature_status for record in records}


def _load_labels(paths: DataPaths, threshold: int):
    """Return ``(raw_records, aggregated, candidate_by_id)`` from the DB.

    Aggregation reuses the exact export + pooling path, so a live readiness check
    reads the same numbers a fresh ``label export`` would produce.
    """
    candidates = manifests.read_candidates(paths.candidates_manifest)
    candidate_by_id = {c.candidate_id: c for c in candidates}
    database = LabelDatabase(paths.label_database, threshold)
    raw = database.all_labels()
    rows, _orphans = build_export_rows(raw, candidate_by_id)
    aggregated = aggregate_labels(rows, acceptable_threshold=float(threshold)).labels
    return raw, aggregated, candidates


def _cmd_readiness(args: argparse.Namespace) -> int:
    from slotify_rank.experiment.readiness import ReadinessGate, compute_readiness
    from slotify_rank.labelling.quality import check_label_quality
    from slotify_rank.labelling.queue import read_queue

    paths = _paths(args)
    raw, aggregated, candidates = _load_labels(paths, args.acceptable_threshold)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    queue = read_queue(Path(args.queue)) if args.queue else None
    split_path = paths.split_manifest(args.split_version)

    quality = check_label_quality(
        raw,
        candidates,
        episodes,
        queue=queue,
        acceptable_threshold=args.acceptable_threshold,
        current_candidate_manifest_hash=(
            sha256_file(paths.candidates_manifest)
            if paths.candidates_manifest.is_file()
            else None
        ),
        current_split_manifest_hash=(
            sha256_file(split_path) if split_path.is_file() else None
        ),
    )
    report = compute_readiness(
        aggregated,
        raw,
        candidates,
        episodes,
        quality,
        feature_status_by_id=_feature_status_by_id(paths),
        gate=ReadinessGate(),
    )

    destination = (
        Path(args.output)
        if args.output
        else paths.data_root.parent / "artifacts" / "experiments" / "readiness_report.json"
    )
    report.write(destination)

    m = report.metrics
    print("Phase 5B readiness gate")
    print(f"  human-labelled unique candidates : {m['human_labelled_unique_candidates']}")
    print(f"  human-labelled episodes          : {m['human_labelled_episodes']}")
    print(f"  human-labelled series            : {m['human_labelled_series']}")
    print(f"  labels by split                  : {m['labels_by_split']}")
    print(f"  labels by quality score          : {m['labels_by_quality_score']}")
    print(
        f"  acceptable / unacceptable        : {m['acceptable_count']} / "
        f"{m['unacceptable_count']}"
    )
    print(
        f"  series train/val/test            : {len(m['train_series'])}/"
        f"{len(m['validation_series'])}/{len(m['test_series'])}"
    )
    print(
        f"  usable pairs train/val           : {m['usable_pairs_train']}/"
        f"{m['usable_pairs_validation']}"
    )
    print(f"  test relevant candidates         : {m['test_relevant_candidate_count']}")
    print(
        f"  feature-complete labelled        : {m['feature_complete_labelled']} "
        f"(incomplete: {len(m['feature_incomplete_labelled'])})"
    )
    print(
        f"  labels pass integrity            : {m['labels_pass_integrity']} "
        f"({m['integrity_error_count']} error(s), {m['integrity_warning_count']} warning(s))"
    )
    print()
    print(f"Phase 5B ready: {report.ready}")
    if report.blocking_reasons:
        print("Blocking reasons:")
        for reason in report.blocking_reasons:
            print(f"  - {reason}")
    print(f"Wrote {destination}")

    if args.require_ready and not report.ready:
        print(
            "error: --require-ready set and the gate is not met; refusing to proceed.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Resolve the committed experiment definition into a hashed manifest."""
    from slotify_rank.experiment.canonical import (
        ExperimentConfigError,
        load_experiment_config,
        resolve_manifest,
        write_manifest,
    )

    paths = _paths(args)
    try:
        config = load_experiment_config(Path(args.config))
    except ExperimentConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _raw, aggregated, _candidates = _load_labels(paths, args.acceptable_threshold)
    manifest = resolve_manifest(
        config,
        repo_root=paths.repo_root,
        paths=paths,
        human_label_count=len(aggregated),
    )

    destination = (
        Path(args.output)
        if args.output
        else paths.data_root.parent
        / "artifacts"
        / "experiments"
        / f"{config.experiment_version}.json"
    )
    write_manifest(destination, manifest)

    print(f"Experiment {manifest.experiment_version}")
    print(f"  config digest      : {manifest.config_digest[:16]}")
    print(f"  human labels       : {manifest.label_summary['human_label_count']} "
          f"(gate: {manifest.label_summary['minimum_human_labels']})")
    print(f"  split              : {manifest.split_summary.get('version')} "
          f"group_by={manifest.split_summary.get('group_by')} "
          f"seed={manifest.split_summary.get('seed')} "
          f"degraded={manifest.split_summary.get('degraded')}")
    groups = manifest.split_summary.get("groups_by_split") or {}
    print(f"  series per split   : { {k: len(v) for k, v in groups.items()} }")
    print(f"  seeds              : {list(config.seeds)}")
    print(f"  headline variant   : {config.model['headline_variant']}")
    print(f"  canonical baseline : {config.canonical_baseline}")
    print("  artifact hashes:")
    for name, digest in sorted(manifest.artifact_hashes.items()):
        shown = "MISSING" if digest is None else digest[:16]
        print(f"    {name:<34} {shown}")
    print(f"  ready: {manifest.ready}")
    for reason in manifest.blocking_reasons:
        print(f"    - blocked: {reason}")
    for warning in manifest.warnings:
        print(f"    - warning: {warning}")
    print(f"Wrote {destination}")

    if args.require_ready and not manifest.ready:
        print(
            "error: --require-ready set and the experiment is not resolvable yet.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train every declared ablation at every declared seed, then pick one.

    The picking is the part worth reading: the reported run is the *median* seed
    by validation NDCG@3, not the best. Reporting the best of several seeds
    reports the upper tail of a distribution as if it were its centre, and every
    run's score is written out so a reader can see the spread rather than take it
    on trust.
    """
    from slotify_rank.experiment.canonical import (
        ExperimentConfigError,
        load_experiment_config,
    )
    from slotify_rank.experiment.matrix import (
        plan_runs,
        read_outcome,
        summarise_matrix,
        write_matrix_summary,
    )
    from slotify_rank.training_cli import _cmd_run

    paths = _paths(args)
    try:
        config = load_experiment_config(Path(args.config))
    except ExperimentConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    variants = (
        [args.variant] if args.variant else list(config.model["ablations"])
    )
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    output_root = Path(args.output_root or (paths.data_root.parent / "artifacts" / "training"))
    runs = plan_runs(variants, seeds, output_root)

    print(
        f"Training matrix for {config.experiment_version}: "
        f"{len(variants)} variant(s) x {len(seeds)} seed(s) = {len(runs)} run(s)."
    )
    print(f"  labels : {args.labels}")
    print(f"  output : {output_root}")

    outcomes = []
    for index, run in enumerate(runs, start=1):
        print()
        print(f"[{index}/{len(runs)}] {run.name}")
        print("-" * 72)
        if run.run_dir.joinpath("training_summary.json").is_file() and not args.force:
            print("  already trained; reading its summary (pass --force to retrain)")
            outcomes.append(read_outcome(run, 0))
            continue
        # The same namespace `training run` would build. Label source is left
        # unset on purpose: this experiment trains on human labels only, and the
        # loader's default allowlist is human-only, so a weak export fails here
        # rather than being trained on and caught later.
        run_args = argparse.Namespace(
            data_root=args.data_root,
            labels=args.labels,
            allow_label_source=None,
            training_config=args.training_config,
            split_version=args.split_version,
            output=None,
            epochs=None,
            seed=run.seed,
            device=None,
            batch_size=None,
            learning_rate=None,
            max_episodes=None,
            max_candidates=None,
            max_pairs=None,
            num_threads=None,
            smoke=False,
            model=run.variant,
            model_config=None,
            run_dir=str(run.run_dir),
        )
        try:
            code = int(_cmd_run(run_args))
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            code = 1
            print(f"  run raised: {type(error).__name__}: {error}")
        outcomes.append(read_outcome(run, code))

    summary = summarise_matrix(
        experiment_version=config.experiment_version,
        headline_variant=str(config.model["headline_variant"]),
        seeds=seeds,
        outcomes=outcomes,
        allowed_label_sources=tuple(config.labels["allowed_label_sources"]),
    )
    destination = (
        Path(args.summary)
        if args.summary
        else paths.data_root.parent
        / "artifacts"
        / "experiments"
        / f"{config.experiment_version}-training-matrix.json"
    )
    write_matrix_summary(destination, summary)

    print()
    print("=" * 72)
    print("Validation NDCG@3 by variant (median over seeds)")
    for variant, block in summary.by_variant().items():
        print(
            f"  {variant:<14} {block['validation_ndcg_at_3_median']:.4f}  "
            f"[{block['validation_ndcg_at_3_min']:.4f}, "
            f"{block['validation_ndcg_at_3_max']:.4f}]  "
            f"({block['seed_count']} seed(s), {block['model_parameter_count']} params)"
        )
    print()
    if summary.reported:
        print(
            f"Reported run: {summary.reported.variant} seed {summary.reported.seed} "
            f"(median), validation NDCG@3 "
            f"{summary.reported.validation_ndcg_at_3:.4f}"
        )
        print(f"  checkpoint: {Path(summary.reported.run_dir) / 'best_checkpoint.pt'}")
    for reason in summary.blocking_reasons:
        print(f"  blocked: {reason}")
    print("=" * 72)
    print(f"Wrote {destination}")
    return 0 if summary.ok else 1


def _cmd_freeze(args: argparse.Namespace) -> int:
    from slotify_rank.experiment.freeze import build_snapshot, write_snapshot

    paths = _paths(args)
    raw, aggregated, candidates = _load_labels(paths, args.acceptable_threshold)
    if not aggregated:
        print(
            "error: no human labels to freeze. Run the labelling session first "
            "(see docs/human-labelling-workflow.md).",
            file=sys.stderr,
        )
        return 1

    candidate_by_id = {c.candidate_id: c for c in candidates}
    candidate_split = {
        cid: candidate_by_id[cid].dataset_split
        for cid in aggregated
        if cid in candidate_by_id
    }

    # Exclusions: orphans (labelled candidate gone from the manifest) and any
    # candidate an annotator marked unusable.
    excluded: list[dict[str, Any]] = []
    for cid in aggregated:
        if cid not in candidate_by_id:
            excluded.append({"candidate_id": cid, "reason": "orphan_no_candidate"})
    unusable = {
        label.candidate_id for label in raw if label.is_unusable
    }
    for cid in sorted(unusable & set(aggregated)):
        excluded.append({"candidate_id": cid, "reason": "marked_unusable"})

    label_export = paths.labels_dir / f"labels_{args.dataset_version}.jsonl"
    queue_hash = (
        sha256_file(Path(args.queue)) if args.queue and Path(args.queue).is_file() else None
    )
    snapshot = build_snapshot(
        aggregated,
        candidate_split,
        label_export_path=label_export,
        candidate_manifest_path=paths.candidates_manifest,
        feature_manifest_path=paths.features_manifest,
        split_manifest_path=paths.split_manifest(args.split_version),
        excluded_labels=excluded,
        snapshot_version=args.snapshot_version,
        queue_hash=queue_hash,
    )
    destination = (
        Path(args.output)
        if args.output
        else paths.labels_dir / f"label_snapshot_{args.snapshot_version}.json"
    )
    write_snapshot(destination, snapshot, force=args.force)
    print(
        f"Froze label snapshot {snapshot.snapshot_version}: "
        f"{snapshot.unique_candidate_count} unique candidate(s), "
        f"{len(snapshot.annotator_ids)} annotator(s), "
        f"{len(snapshot.excluded_labels)} excluded."
    )
    print(f"  content hash: {snapshot.content_hash()[:16]}")
    print(f"Wrote {destination}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    experiment = subparsers.add_parser(
        "experiment", help="Phase 5: readiness gate and label freeze."
    )
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)

    readiness = experiment_sub.add_parser(
        "readiness", help="Report whether the genuine-label Phase 5B gate is met."
    )
    readiness.add_argument("--data-root", default=None)
    readiness.add_argument("--queue", default=None, help="Assigned queue artifact.")
    readiness.add_argument("--split-version", default="v2")
    readiness.add_argument("--output", default=None)
    readiness.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    readiness.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the gate is not met (for use in a training script).",
    )
    readiness.set_defaults(func=_cmd_readiness)

    manifest = experiment_sub.add_parser(
        "manifest",
        help=(
            "Resolve the committed experiment definition into a manifest that "
            "pins every artifact's hash."
        ),
    )
    manifest.add_argument("--data-root", default=None)
    manifest.add_argument(
        "--config",
        default="ml/configs/experiment_resume_v1.yaml",
        help="The committed experiment definition.",
    )
    manifest.add_argument("--output", default=None)
    manifest.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    manifest.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the experiment cannot yet be resolved.",
    )
    manifest.set_defaults(func=_cmd_manifest)

    train = experiment_sub.add_parser(
        "train",
        help=(
            "Train every declared ablation at every declared seed and select the "
            "median seed by validation NDCG@3."
        ),
    )
    train.add_argument("--data-root", default=None)
    train.add_argument(
        "--config", default="ml/configs/experiment_resume_v1.yaml"
    )
    train.add_argument(
        "--labels", required=True, help="Human label export to train on."
    )
    train.add_argument(
        "--training-config", default=None, dest="training_config"
    )
    train.add_argument("--split-version", default="v3", dest="split_version")
    train.add_argument(
        "--output-root",
        default=None,
        dest="output_root",
        help="Where run directories go (default artifacts/training).",
    )
    train.add_argument("--summary", default=None)
    train.add_argument(
        "--variant",
        default=None,
        help="Train only this variant instead of every declared ablation.",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Train only this seed instead of every declared seed.",
    )
    train.add_argument(
        "--force",
        action="store_true",
        help="Retrain cells that already have a training_summary.json.",
    )
    train.set_defaults(func=_cmd_train)

    freeze = experiment_sub.add_parser(
        "freeze", help="Freeze an immutable label snapshot for the experiment."
    )
    freeze.add_argument("--data-root", default=None)
    freeze.add_argument("--queue", default=None)
    freeze.add_argument("--split-version", default="v2")
    freeze.add_argument("--dataset-version", default="v1")
    freeze.add_argument("--snapshot-version", default="v1")
    freeze.add_argument("--output", default=None)
    freeze.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    freeze.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a snapshot of the same version (discouraged; prefer a bump).",
    )
    freeze.set_defaults(func=_cmd_freeze)
