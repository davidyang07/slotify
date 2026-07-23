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
