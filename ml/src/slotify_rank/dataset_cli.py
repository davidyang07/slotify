"""``dataset``, ``candidates`` and ``label`` command implementations.

Kept out of :mod:`slotify_rank.cli` so the Phase 1 commands stay readable, and
wired in by :func:`register`. Every command follows the conventions established
in Phase 1: machine-readable JSON to files, a human summary on stdout, exit 0 on
success, 1 on a handled runtime failure, 2 on a usage error.

All of these are resumable. Import, probe and normalize upsert into the episode
manifest by ``episode_id`` and skip work that is already done, so re-running
after an interruption costs only the work that was actually lost.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.candidates.config import load_generation_config
from slotify_rank.candidates.generate import generate_for_episode
from slotify_rank.config.versions import PACKAGE_VERSION
from slotify_rank.data import manifests
from slotify_rank.data.fetch import fetch_source
from slotify_rank.data.import_local import import_source
from slotify_rank.data.normalize import normalize_episode
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.probe import probe_audio
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.data.sources import load_sources
from slotify_rank.data.splits import (
    InsufficientGroups,
    compute_splits,
    load_split_config,
    read_split_manifest,
    write_split_manifest,
)
from slotify_rank.data.stats import compute_statistics, write_statistics
from slotify_rank.data.validate import validate_dataset
from slotify_rank.labelling.database import DEFAULT_ACCEPTABLE_THRESHOLD, LabelDatabase
from slotify_rank.labelling.export import export_labels
from slotify_rank.labelling.queue import (
    build_queue,
    load_queue_config,
    queue_summary,
    read_queue,
    write_queue,
    write_queue_summary,
)

__all__ = ["register"]

_GENERATION_REPORTS = "generation_reports.json"


def _paths(args: argparse.Namespace) -> DataPaths:
    paths = DataPaths(data_root=Path(args.data_root) if args.data_root else None)
    paths.mkdirs()
    return paths


def _selected(episodes: Sequence[EpisodeRecord], ids: Sequence[str] | None):
    if not ids:
        return list(episodes)
    by_id = {episode.episode_id: episode for episode in episodes}
    missing = [episode_id for episode_id in ids if episode_id not in by_id]
    if missing:
        raise KeyError(f"Unknown episode_id(s): {missing}")
    return [by_id[episode_id] for episode_id in ids]


def _load_split(paths: DataPaths, version: str):
    path = paths.split_manifest(version)
    return read_split_manifest(path) if path.is_file() else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def _cmd_discover(args: argparse.Namespace) -> int:
    """Resolve a corpus plan into a committed source registry."""
    from slotify_rank.data.discover import (
        DiscoveryError,
        InternetArchiveClient,
        discover,
        load_corpus_plan,
        render_sources_yaml,
        summarise,
        write_sources_yaml,
    )

    plan_path = Path(args.plan)
    try:
        plan = load_corpus_plan(plan_path)
    except DiscoveryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    client = InternetArchiveClient(timeout=args.timeout)
    try:
        report = discover(plan, client)
    except DiscoveryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for line in summarise(report):
        print(line)

    destination = Path(args.output)
    rendered = render_sources_yaml(report, plan_path.as_posix())
    if args.check:
        # CI mode: prove the committed registry still matches what the plan and
        # the upstream metadata produce, without writing.
        existing = (
            destination.read_text(encoding="utf-8") if destination.is_file() else ""
        )
        if existing == rendered:
            print(f"{destination} is up to date.")
            return 0
        print(
            f"error: {destination} differs from what the plan now discovers. "
            "Re-run without --check to regenerate it.",
            file=sys.stderr,
        )
        return 1

    write_sources_yaml(destination, report, plan_path.as_posix())
    print(f"Wrote {destination}")

    if args.report:
        _write_json(Path(args.report), report.to_dict())
        print(f"Wrote {args.report}")
    return 0


def _cmd_prepare_experiment(args: argparse.Namespace) -> int:
    """Take the committed corpus plan all the way to "ready to label".

    Fetch, probe, normalize, generate, split, transcribe, featurise, embed,
    assemble, validate, count, queue -- then measure whether what came out is
    enough for the labelling target and say so plainly either way.
    """
    from slotify_rank.experiment.orchestrate import (
        build_steps,
        run_preparation,
        summarise_readiness,
    )

    paths = _paths(args)
    # The queue file is named after the queue CONFIG's version, not this
    # command's arguments, so `label run-experiment` looks in the same place
    # `label queue` wrote to. Passing --queue-output overrides both.
    queue_output = args.queue_output or str(
        paths.labels_dir / f"queue_{args.queue_version}.json"
    )
    steps = build_steps(
        data_root=args.data_root,
        source_registries=list(
            args.sources or getattr(args, "sources_default", [])
        ),
        dataset_config=args.dataset_config,
        split_config=args.split_config,
        split_version=args.split_version,
        queue_config=args.queue_config,
        queue_output=queue_output,
        fixtures_registry=args.fixtures,
        fetch_timeout=args.timeout,
        skip_fetch=args.skip_fetch,
        skip_features=args.skip_features,
        force_split=args.force_split,
        force_queue=args.force_queue,
        limit=args.limit,
    )

    print(f"Preparing the benchmark experiment corpus ({len(steps)} steps).")
    report = run_preparation(steps, corpus_version=args.corpus_version)

    readiness = summarise_readiness(
        paths,
        Path(queue_output),
        target_labels=args.target_labels,
        split_version=args.split_version,
    )
    report.readiness = readiness

    destination = (
        Path(args.report)
        if args.report
        else paths.data_root.parent
        / "artifacts"
        / "experiments"
        / "prepare_report.json"
    )
    report.write(destination)

    print()
    print("=" * 72)
    print("Corpus readiness")
    print(f"  generated candidates        : {readiness.generated_candidate_count}")
    print(f"  labellable (target domain)  : {readiness.labellable_candidate_count}")
    print(f"  complete multimodal records : {readiness.complete_multimodal_count}")
    print(f"  queued unique candidates    : {readiness.queued_unique_count}")
    print(f"  labelling target            : {readiness.target_labels}")
    print(f"  episodes / series           : {readiness.episode_count} / {readiness.series_count}")
    print(f"  target-domain audio hours   : {readiness.processed_audio_hours}")
    print(f"  series by split             : "
          f"{ {k: len(v) for k, v in readiness.series_by_split.items()} }")
    print(f"  ready to label              : {readiness.ready_to_label}")
    for reason in readiness.blocking_reasons:
        print(f"    - {reason}")
    print("=" * 72)
    print(f"Wrote {destination}")

    if not report.ok:
        failed = [step.name for step in report.steps if not step.ok]
        print(f"error: step(s) failed: {failed}", file=sys.stderr)
        return 1
    if args.require_ready and not readiness.ready_to_label:
        print(
            "error: --require-ready set and the corpus cannot support the "
            "labelling target.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_import_local(args: argparse.Namespace) -> int:
    paths = _paths(args)
    registry = load_sources(args.sources)
    entries = [
        entry
        for entry in registry.select(args.source_id)
        if entry.source_type != "direct_download"
    ]
    if not entries:
        print("No local or fixture sources selected; nothing to import.")
        return 0

    existing = {e.episode_id: e for e in manifests.read_episodes(paths.episodes_manifest)}
    imported: list[EpisodeRecord] = []
    for entry in entries:
        result = import_source(entry, paths, existing)
        imported.append(result.episode)
        print(
            f"  {result.action:<10} {entry.id} -> {result.episode.episode_id} "
            f"({result.episode.content_type})"
        )
    added, updated, total = manifests.upsert_episodes(paths.episodes_manifest, imported)
    print(
        f"Imported {len(imported)} source(s): {added} new, {updated} updated, "
        f"{total} episode(s) in the manifest."
    )
    print(f"Manifest: {paths.episodes_manifest}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    paths = _paths(args)
    registry = load_sources(args.sources)
    entries = [
        entry
        for entry in registry.select(args.source_id)
        if entry.source_type == "direct_download"
    ]
    if not entries:
        print("No direct_download sources selected; nothing to fetch.")
        return 0

    existing = {e.episode_id: e for e in manifests.read_episodes(paths.episodes_manifest)}
    fetched: list[EpisodeRecord] = []
    for entry in entries:
        result = fetch_source(entry, paths, existing, timeout=args.timeout)
        fetched.append(result.episode)
        print(
            f"  {result.action:<10} {entry.id} -> {result.episode.episode_id} "
            f"({result.bytes_written} bytes)"
        )
    added, updated, total = manifests.upsert_episodes(paths.episodes_manifest, fetched)
    print(
        f"Fetched {len(fetched)} source(s): {added} new, {updated} updated, "
        f"{total} episode(s) in the manifest."
    )
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    """Drop manifest episodes no committed registry declares."""
    from slotify_rank.data.reconcile import reconcile_manifest, write_report

    paths = _paths(args)
    registries = list(args.sources or getattr(args, "sources_default", []))
    if not registries:
        print("error: no source registry given", file=sys.stderr)
        return 1

    report = reconcile_manifest(
        episodes_path=paths.episodes_manifest,
        candidates_path=paths.candidates_manifest,
        features_path=paths.features_manifest,
        registries=registries,
        keep_local=not args.drop_local,
        dry_run=args.check,
    )
    for line in report.lines():
        print(line)
    if args.report:
        write_report(Path(args.report), report)
        print(f"Wrote {args.report}")
    if args.check and report.changed:
        print(
            "error: the episode manifest holds episodes no committed registry "
            "declares. Re-run without --check to remove them.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_orphans(args: argparse.Namespace) -> int:
    """Report generated files no manifest episode references. Never deletes."""
    from slotify_rank.data.orphans import scan_orphans

    paths = _paths(args)
    report = scan_orphans(paths)
    for line in report.lines():
        print(line)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.report}")
    if args.check and not report.clean:
        print(
            "error: generated files exist that no manifest episode references.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    selected = _selected(episodes, args.episode_id)
    if not selected:
        print("No episodes in the manifest. Run `dataset import-local` first.")
        return 0

    updated: list[EpisodeRecord] = []
    for episode in selected:
        if episode.duration_ms is not None and not args.force:
            print(f"  cached     {episode.episode_id} ({episode.duration_ms} ms)")
            continue
        metadata = probe_audio(paths.repo_root / episode.original_path)
        updated.append(
            episode.replace(
                duration_ms=metadata.duration_ms,
                sample_rate_hz=metadata.sample_rate_hz,
                channels=metadata.channels,
                file_format=metadata.file_format,
                status="probed" if episode.status in ("registered", "fetched") else episode.status,
            )
        )
        print(
            f"  probed     {episode.episode_id}: {metadata.duration_ms} ms, "
            f"{metadata.sample_rate_hz} Hz, {metadata.channels}ch, "
            f"{metadata.file_format}/{metadata.codec_name}"
        )
    if updated:
        manifests.upsert_episodes(paths.episodes_manifest, updated)
    total_ms = sum(
        (e.duration_ms or 0)
        for e in manifests.read_episodes(paths.episodes_manifest)
    )
    print(
        f"Probed {len(updated)} episode(s). Manifest total: "
        f"{total_ms / 3_600_000:.4f} h."
    )
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    selected = _selected(episodes, args.episode_id)
    if not selected:
        print("No episodes in the manifest. Run `dataset import-local` first.")
        return 0

    updated: list[EpisodeRecord] = []
    rendered = 0
    for episode in selected:
        result = normalize_episode(episode, paths, force=args.force)
        updated.append(result.episode)
        rendered += int(result.rendered)
        print(f"  {result.action:<10} {episode.episode_id}")
    manifests.upsert_episodes(paths.episodes_manifest, updated)
    print(
        f"Normalized {len(selected)} episode(s) to 16 kHz mono PCM WAV "
        f"({rendered} rendered, {len(selected) - rendered} already cached)."
    )
    print(f"Output: {paths.normalized_dir}")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    processed = [e for e in episodes if e.status == "normalized"]
    if not processed:
        print(
            "error: no normalized episodes to split. Run `dataset normalize` first.",
            file=sys.stderr,
        )
        return 1

    config = load_split_config(args.config)
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    manifest = compute_splits(processed, config)
    destination = paths.split_manifest(manifest.version)
    write_split_manifest(destination, manifest, force=args.force)

    # Stamp the assignment onto every candidate so a candidate row alone is
    # enough to know which partition it belongs to.
    candidates = manifests.read_candidates(paths.candidates_manifest)
    if candidates:
        by_episode = manifest.by_episode
        manifests.write_candidates(
            paths.candidates_manifest,
            [
                candidate.replace(
                    dataset_split=by_episode.get(candidate.episode_id, "unassigned")
                )
                for candidate in candidates
            ],
        )

    if manifest.degraded:
        print(f"WARNING: {manifest.reason}")
    counts: dict[str, int] = {}
    for assignment in manifest.assignments:
        counts[assignment.split] = counts.get(assignment.split, 0) + len(
            assignment.episode_ids
        )
    print(
        f"Split {len(processed)} episode(s) into {len(manifest.assignments)} group(s) "
        f"by {manifest.group_by} (seed {manifest.seed})."
    )
    for split, count in sorted(counts.items()):
        print(f"  {split:<12} {count} episode(s)")
    print(f"Wrote {destination}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    split_manifest = _load_split(paths, args.split_version)
    labels = (
        LabelDatabase(paths.label_database).all_labels()
        if paths.label_database.is_file()
        else []
    )

    report = validate_dataset(
        episodes, candidates, paths, labels, split_manifest, deep=args.deep
    )
    destination = Path(args.output) if args.output else paths.artifacts_dir / "validation_report.json"
    report.write(destination)

    for finding in report.findings:
        prefix = "ERROR  " if finding.severity == "error" else "warning"
        subject = f" [{finding.subject}]" if finding.subject else ""
        print(f"  {prefix} {finding.check}{subject}: {finding.message}")
    print(
        f"{len(set(report.checks_run))} check(s) run: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)."
    )
    print(f"Wrote {destination}")
    if not report.ok:
        print("Dataset validation FAILED. Integrity errors must be fixed.")
        return 1
    print("Dataset validation passed.")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    split_manifest = _load_split(paths, args.split_version)
    database = LabelDatabase(paths.label_database, args.acceptable_threshold)
    labels = database.all_labels()

    reports_path = paths.manifests_dir / _GENERATION_REPORTS
    generation_reports = (
        json.loads(reports_path.read_text(encoding="utf-8"))
        if reports_path.is_file()
        else []
    )

    bundle = compute_statistics(
        episodes,
        candidates,
        labels,
        split_manifest,
        generation_reports,
        acceptable_threshold=args.acceptable_threshold,
    )
    directory = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    written = write_statistics(bundle, directory)

    dataset = bundle.dataset
    label_stats = bundle.labels
    print("Measured quantities (targets are goals, not results):")
    print(
        f"  processed_audio_hours              {dataset['processed_audio_hours']} "
        f"(target 50)"
    )
    print(f"  processed_episode_count            {dataset['processed_episode_count']}")
    print(
        f"  generated_candidate_count          "
        f"{bundle.candidates['generated_candidate_count']} (target 10000)"
    )
    # Read from the targets block, which takes it from the committed experiment,
    # rather than restating it here. This line said "(target 1500)" while the
    # artifact beside it recorded the frozen gate of 2400 -- the console
    # contradicting the file it had just written, on the one number the whole
    # label gate turns on.
    print(
        f"  human_labelled_candidate_count     "
        f"{label_stats['human_labelled_candidate_count']} "
        f"(target {dataset['targets']['human_labelled_candidate_count']})"
    )
    print(
        f"  human_labelled_audio_hours         "
        f"{label_stats['human_labelled_audio_hours']}"
    )
    print(
        f"  weakly_labelled_candidate_count    "
        f"{label_stats['weakly_labelled_candidate_count']}"
    )
    print(
        f"  unlabelled_candidate_count         "
        f"{label_stats['unlabelled_candidate_count']}"
    )
    print(
        f"  held_out_evaluation_candidate_count "
        f"{label_stats['held_out_evaluation_candidate_count']}"
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    selected = [
        episode
        for episode in _selected(episodes, args.episode_id)
        if episode.status == "normalized"
    ]
    if not selected:
        print(
            "error: no normalized episodes. Run `dataset normalize` first.",
            file=sys.stderr,
        )
        return 1

    config = load_generation_config(args.config)
    records: list[DatasetCandidate] = []
    reports: list[dict[str, Any]] = []
    for episode in selected:
        episode_records, report = generate_for_episode(
            episode,
            paths,
            config=config,
            include_product_padding=args.include_product_padding,
        )
        records.extend(episode_records)
        reports.append(report.to_dict())
        print(
            f"  {episode.episode_id}: {report.count_before_merge} raw -> "
            f"{report.count_after_merge} merged -> {report.count_after_density_cap} kept "
            f"({report.candidates_per_minute:.2f}/min)"
        )
        for note in report.notes:
            print(f"      note: {note}")

    removed, total = manifests.replace_episode_candidates(
        paths.candidates_manifest,
        [episode.episode_id for episode in selected],
        records,
    )
    _write_json(paths.manifests_dir / _GENERATION_REPORTS, reports)

    real = sum(1 for record in records if not record.is_synthetic)
    synthetic = len(records) - real
    print(
        f"Generated {real} candidate(s) across {len(selected)} episode(s) "
        f"(replaced {removed}; manifest now holds {total})."
    )
    if synthetic:
        print(
            f"Recorded {synthetic} synthetic product-padding slot(s), all marked "
            "is_synthetic=true and excluded from labelling and evaluation."
        )
    print(f"Manifest: {paths.candidates_manifest}")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def _feature_records_by_id(paths: DataPaths) -> dict[str, Any]:
    """Map candidate_id -> feature record, or empty when features are absent.

    Optional: the queue stratifies on the candidate manifest alone and only uses
    features (transcript / embedding availability, sentence boundaries) as extra
    coverage signal when the Phase 3 pipeline has run.
    """
    if not paths.features_manifest.is_file():
        return {}
    from slotify_rank.features.assemble import read_feature_manifest

    _, records = read_feature_manifest(paths.features_manifest)
    return {record.candidate_id: record for record in records}


def _queue_destination(paths: DataPaths, args: argparse.Namespace, version: str) -> Path:
    if getattr(args, "output", None):
        return Path(args.output)
    return paths.labels_dir / f"queue_{version}.json"


def _cmd_label_queue(args: argparse.Namespace) -> int:
    paths = _paths(args)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    if not candidates:
        print(
            "error: no candidates. Run `candidates generate` on real episodes first.",
            file=sys.stderr,
        )
        return 1

    from slotify_rank.data.checksum import sha256_file

    config = load_queue_config(args.config)
    split_path = paths.split_manifest(args.split_version)
    split_hash = sha256_file(split_path) if split_path.is_file() else None
    queue = build_queue(
        candidates,
        episodes,
        config=config,
        feature_records=_feature_records_by_id(paths),
        candidate_manifest_hash=sha256_file(paths.candidates_manifest),
        split_manifest_hash=split_hash,
    )
    destination = _queue_destination(paths, args, queue.queue_version)
    write_queue(destination, queue, force=args.force)

    cov = queue.coverage
    print(
        f"Queue {queue.queue_version}: {queue.unique_count} unique candidate(s) "
        f"(pilot {len(queue.pilot_candidate_ids)}, primary "
        f"{len(queue.primary_candidate_ids)}, overlap "
        f"{len(queue.overlap_candidate_ids)}, consistency "
        f"{len(queue.consistency_candidate_ids)})."
    )
    print(f"  score strata: {queue.score_strata.to_dict()}")
    print(f"  by split: {cov['by_split']}")
    print(f"  by score stratum: {cov['by_score_stratum']}")
    print(f"  by series: {cov['by_series']}")
    print(
        f"  transcript-available {cov['transcript_available']}, "
        f"signal-disagreement {cov['signal_disagreement']}, "
        f"max per episode {cov['max_per_episode_selected']}"
    )
    print(f"Wrote {destination}")
    return 0


def _cmd_label_queue_summary(args: argparse.Namespace) -> int:
    """Reduce a queue artifact to the committable counts-and-hashes summary.

    The queue itself lists 2,400 candidate ids and lives under the uncommitted
    ``data/`` tree, so a fresh clone has no way to see that the labelling round
    was built at all. This writes the reduction that does ship, under
    ``artifacts/labelling/``.
    """
    paths = _paths(args)
    queue_path = (
        Path(args.queue)
        if args.queue
        else paths.data_root / "labels" / "queue_full-v2.json"
    )
    queue = read_queue(queue_path)
    destination = (
        Path(args.output)
        if args.output
        else paths.data_root.parent / "artifacts" / "labelling" / "queue_summary.json"
    )
    summary = queue_summary(queue)

    if args.check:
        if not destination.is_file():
            print(
                f"error: {destination} does not exist; run without --check first.",
                file=sys.stderr,
            )
            return 1
        committed = json.loads(destination.read_text(encoding="utf-8"))
        # generated_at is deliberately absent from the summary, so the two
        # dictionaries are comparable without excluding anything.
        if committed != summary:
            print(
                f"error: {destination} no longer agrees with {queue_path}.",
                file=sys.stderr,
            )
            return 1
        print(f"{destination} agrees with {queue_path}.")
        return 0

    write_queue_summary(destination, queue)
    print(
        f"Queue {summary['queue_version']}: {summary['unique_candidate_count']} "
        f"unique candidate(s) of a {summary['target_unique']} target, "
        f"{summary['presentation_count']} presentation(s) including "
        f"{summary['blind_repeat_presentations']} blind repeat(s)."
    )
    print(f"  by split: {summary['coverage'].get('by_split')}")
    print(f"Wrote {destination}")
    return 0


def stage_candidate_ids(queue: Any, stage: str) -> set[str]:
    """The candidate ids a serve session should expose for a queue stage.

    ``all`` is the whole unique set; ``pilot`` / ``primary`` restrict to that
    stage so a controlled pilot can be run and its progress counted against the
    right denominator. Consistency re-checks are never served here -- they are
    presentation-level repeats, not unique candidates.
    """
    if stage == "pilot":
        return set(queue.pilot_candidate_ids)
    if stage == "primary":
        return set(queue.primary_candidate_ids)
    if stage == "all":
        return set(queue.unique_candidate_ids)
    raise ValueError(f"Unknown serve stage {stage!r}")


def _load_serve_inputs(args: argparse.Namespace):
    """Resolve the candidates, episodes and queue one serve session will use."""
    paths = _paths(args)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible_for_labelling and not candidate.is_synthetic
    ]
    queue = None
    if getattr(args, "queue", None):
        queue = read_queue(Path(args.queue))
        stage = getattr(args, "stage", "all")
        wanted = stage_candidate_ids(queue, stage)
        eligible = [c for c in eligible if c.candidate_id in wanted]
        stage_label = "" if stage == "all" else f" ({stage} stage)"
        print(
            f"Restricted to labelling queue {queue.queue_version}{stage_label}: "
            f"{len(eligible)} of {len(wanted)} queued candidate(s) present."
        )
    return paths, episodes, eligible, queue


def _serve_labelling_app(
    args: argparse.Namespace,
    paths: DataPaths,
    episodes,
    eligible,
    queue,
    target_unique: int | None = None,
) -> int:
    from slotify_rank.labelling.service import LabellingSettings, create_app

    database = LabelDatabase(paths.label_database, args.acceptable_threshold)
    app = create_app(
        database,
        eligible,
        episodes,
        paths,
        LabellingSettings(
            context_before_ms=args.context_before_ms,
            context_after_ms=args.context_after_ms,
            reveal_hints=args.reveal_hints,
            target_unique=target_unique,
        ),
        queue=queue,
    )

    try:
        import uvicorn
    except ImportError:
        print(
            'error: uvicorn is not installed. Install with: pip install -e ".[label]"',
            file=sys.stderr,
        )
        return 1

    print(f"Serving {len(eligible)} candidate(s) from {len(episodes)} episode(s).")
    print(f"Labels: {paths.label_database}")
    print(f"Open http://{args.host}:{args.port}/ and enter an annotator id.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_label_serve(args: argparse.Namespace) -> int:
    paths, episodes, eligible, queue = _load_serve_inputs(args)
    if not eligible:
        print(
            "error: no candidates eligible for labelling. Run "
            "`candidates generate` first.",
            file=sys.stderr,
        )
        return 1
    return _serve_labelling_app(args, paths, episodes, eligible, queue)


def _cmd_label_run_experiment(args: argparse.Namespace) -> int:
    """One command that leaves nothing to do but the labelling itself.

    Everything that can be prepared mechanically is prepared here -- the queue is
    read, every clip is cut, the readiness of the corpus is checked -- and then
    the server starts. The only step left afterwards is a human forming
    judgements, which is the one step no command can do.
    """
    from slotify_rank.labelling.service import LabellingSettings, prerender_clips

    # The default queue lives under the data root, which moves with --data-root
    # and is not the working directory. Resolving it here rather than as an
    # argparse default means running from ml/ finds the same file as running
    # from the repository root.
    if not args.queue:
        args.queue = str(_paths(args).labels_dir / f"queue_{args.queue_version}.json")
    if not Path(args.queue).is_file():
        print(
            f"error: no labelling queue at {args.queue}. Build the corpus with:\n"
            "    slotify-rank dataset prepare-experiment\n"
            "or, if it is already prepared, just the queue:\n"
            "    slotify-rank label queue "
            "--config ml/configs/labelling_queue_full_v2.yaml --split-version v4",
            file=sys.stderr,
        )
        return 1

    paths, episodes, eligible, queue = _load_serve_inputs(args)
    if queue is None:
        print(f"error: {args.queue} is not a labelling queue.", file=sys.stderr)
        return 1
    if not eligible:
        print(
            "error: the queue names no candidate present in the manifest. Run "
            "`dataset prepare-experiment` first.",
            file=sys.stderr,
        )
        return 1

    target = args.target or len(queue.unique_candidate_ids)
    database = LabelDatabase(paths.label_database, args.acceptable_threshold)
    already = len(database.labelled_candidate_ids())

    print("Preparing the labelling session")
    print(f"  queue            : {args.queue} ({queue.queue_version})")
    print(f"  unique candidates: {len(queue.unique_candidate_ids)}")
    print(
        f"  blind repeats    : {len(queue.consistency_candidate_ids)} "
        "(interleaved; they measure whether you agree with yourself and are "
        "never exported as labels)"
    )
    print(f"  target           : {target}")
    print()
    print("Pre-cutting every clip so no rating waits on FFmpeg...")
    counts = prerender_clips(
        eligible,
        episodes,
        paths,
        LabellingSettings(
            context_before_ms=args.context_before_ms,
            context_after_ms=args.context_after_ms,
        ),
    )
    print(
        f"  clips: {counts['rendered']} rendered, {counts['cached']} already cached, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    # Via the same resolver the labelling service reads with. Spelling the
    # convention out here instead cost this banner its accuracy once already: it
    # looked for "<id>.json" where the cache writes "<id>.transcript.json", so it
    # reported 0/77 transcripts for a fully transcribed corpus and made the
    # session look far less ready than it was.
    from slotify_rank.transcription.cache import transcript_path

    transcribed = sum(
        1
        for episode in episodes
        if transcript_path(paths, episode.episode_id).is_file()
    )
    print(f"  transcripts: {transcribed}/{len(episodes)} episode(s) have one")
    print()
    remaining = max(0, target - already)
    print("=" * 72)
    print("EVERYTHING ELSE IS READY. The only remaining task is labelling.")
    print(f"  human labels so far : {already}")
    print(f"  still to label      : {remaining}")
    print(f"  open                : http://{args.host}:{args.port}/")
    print("  keys                : 1-5 rates and advances, S skips, U marks broken")
    print("=" * 72)
    print()

    if args.no_serve:
        return 0
    return _serve_labelling_app(
        args, paths, episodes, eligible, queue, target_unique=target
    )


def _cmd_label_check(args: argparse.Namespace) -> int:
    from slotify_rank.data.checksum import sha256_file
    from slotify_rank.labelling.quality import check_label_quality

    paths = _paths(args)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    episodes = manifests.read_episodes(paths.episodes_manifest)
    database = LabelDatabase(paths.label_database, args.acceptable_threshold)
    labels = database.all_labels()

    queue = read_queue(Path(args.queue)) if args.queue else None
    split_path = paths.split_manifest(args.split_version)
    report = check_label_quality(
        labels,
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
    destination = (
        Path(args.output)
        if args.output
        else paths.artifacts_dir / "label_quality_report.json"
    )
    report.write(destination)

    for finding in report.findings:
        prefix = {"error": "ERROR  ", "warning": "warning", "info": "info   "}[
            finding.severity
        ]
        subject = f" [{finding.subject}]" if finding.subject else ""
        print(f"  {prefix} {finding.check}{subject}: {finding.message}")
    print(
        f"{report.label_count} label(s) from {report.annotator_count} annotator(s): "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
    )
    print(f"Wrote {destination}")
    if not report.ok:
        print("Label quality check FAILED (errors must be fixed).")
        return 1
    return 0


def _cmd_label_export(args: argparse.Namespace) -> int:
    paths = _paths(args)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    database = LabelDatabase(paths.label_database, args.acceptable_threshold)
    destination = (
        Path(args.output)
        if args.output
        else paths.labels_dir / f"labels_{args.dataset_version}.jsonl"
    )
    result = export_labels(database, candidates, destination, args.dataset_version)
    print(
        f"Exported {result.row_count} human label(s) from {result.annotator_count} "
        f"annotator(s)."
    )
    if result.orphan_count:
        print(
            f"WARNING: {result.orphan_count} label(s) reference candidates that are no "
            "longer in the manifest and were not exported; see the metadata file."
        )
    print(f"Wrote {result.path}")
    print(f"Wrote {result.metadata_path}")
    return 0


def _cmd_label_weak(args: argparse.Namespace) -> int:
    """Generate the weak, heuristic-derived bootstrap labels.

    Loud on purpose. These are not human labels, they are not counted as human
    labels anywhere, and a model trained on them distills the baseline rather
    than beating it.
    """
    from slotify_rank.labelling.weak import (
        build_weak_label_rows,
        write_weak_label_export,
    )

    paths = _paths(args)
    candidates = manifests.read_candidates(paths.candidates_manifest)
    rows, summary = build_weak_label_rows(candidates)
    if not rows:
        print(
            "No eligible candidate carried a heuristic score, so no weak label "
            "could be derived.",
            file=sys.stderr,
        )
        return 1

    destination = (
        Path(args.output)
        if args.output
        else paths.labels_dir / f"weak_labels_{args.dataset_version}.jsonl"
    )
    write_weak_label_export(destination, rows, summary)

    print("WEAK LABELS - NOT HUMAN LABELS.")
    print(
        "  Derived by binning heuristic_offline_v1's own score into the 1-5 rubric"
    )
    print(
        "  within each episode. A model trained on these distills the baseline and"
    )
    print("  must never be compared against it, or quoted as a quality result.")
    print(f"  candidates: {summary.candidate_count}")
    print(f"  episodes:   {summary.episode_count}")
    print(f"  grades:     {dict(sorted(summary.grades.items()))}")
    if summary.episodes_without_spread:
        print(
            f"  {len(summary.episodes_without_spread)} episode(s) had no score "
            "spread and therefore express no preference."
        )
    print(f"Wrote {destination}")
    print(f"Wrote {destination.with_suffix(destination.suffix + '.meta.json')}")
    return 0


# ---------------------------------------------------------------------------
# parser wiring
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the data root (default: <repo>/data, or $SLOTIFY_DATA_ROOT).",
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the Phase 2 command groups to the Phase 1 parser."""
    default_sources = "ml/configs/sources.yaml"

    dataset = subparsers.add_parser(
        "dataset", help="Build and validate the offline dataset."
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    discover_parser = dataset_sub.add_parser(
        "discover",
        help=(
            "Resolve a corpus plan into a committed source registry by reading "
            "the Internet Archive's public metadata API."
        ),
    )
    discover_parser.add_argument(
        "--plan",
        default="ml/configs/corpus_v2.yaml",
        help="Corpus plan to resolve.",
    )
    discover_parser.add_argument(
        "--output",
        default="ml/configs/sources_v2.yaml",
        help="Source registry to write (committed).",
    )
    discover_parser.add_argument(
        "--report", default=None, help="Optional JSON discovery report."
    )
    discover_parser.add_argument("--timeout", type=float, default=60.0)
    discover_parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the registry would change.",
    )
    discover_parser.set_defaults(func=_cmd_discover)

    prepare_parser = dataset_sub.add_parser(
        "prepare-experiment",
        help=(
            "One command: acquire, generate, featurise, split, validate, count "
            "and queue the corpus for the benchmark experiment."
        ),
    )
    _add_common(prepare_parser)
    prepare_parser.add_argument(
        "--sources",
        action="append",
        default=None,
        help=(
            "Source registry to fetch (repeatable). Defaults to the two committed "
            "registries: sources_real_v1.yaml and sources_v2.yaml."
        ),
    )
    prepare_parser.add_argument(
        "--fixtures",
        default=None,
        help="Registry of local/fixture sources to import (default: none).",
    )
    prepare_parser.add_argument(
        "--dataset-config", default="ml/configs/dataset_v1.yaml"
    )
    prepare_parser.add_argument("--split-config", default="ml/configs/splits_v4.yaml")
    prepare_parser.add_argument("--split-version", default="v4")
    prepare_parser.add_argument(
        "--queue-config", default="ml/configs/labelling_queue_full_v2.yaml"
    )
    prepare_parser.add_argument("--queue-output", default=None)
    prepare_parser.add_argument(
        "--queue-version",
        default="full-v2",
        dest="queue_version",
        help=(
            "Must match `queue.version` in the queue config; it names the queue "
            "file that `label run-experiment` then opens."
        ),
    )
    prepare_parser.add_argument("--corpus-version", default="corpus-v2")
    prepare_parser.add_argument(
        "--target-labels",
        type=int,
        default=2400,
        help="Human labels the experiment targets; readiness is measured against it.",
    )
    prepare_parser.add_argument("--timeout", type=float, default=180.0)
    prepare_parser.add_argument(
        "--limit", type=int, default=None, help="Featurise at most N episodes."
    )
    prepare_parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Assume the audio is already downloaded (the only networked step).",
    )
    prepare_parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Stop before the model-loading stages (useful for a dry structure run).",
    )
    prepare_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Overwrite an existing split manifest of this version.",
    )
    prepare_parser.add_argument(
        "--force-queue",
        action="store_true",
        help=(
            "Overwrite an existing labelling queue of this version. Orphans every "
            "label already collected against it; prefer bumping queue.version."
        ),
    )
    prepare_parser.add_argument("--report", default=None)
    prepare_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the corpus cannot support the labelling target.",
    )
    prepare_parser.set_defaults(
        func=_cmd_prepare_experiment,
        sources_default=[
            "ml/configs/sources_real_v1.yaml",
            "ml/configs/sources_v2.yaml",
        ],
    )

    import_parser = dataset_sub.add_parser(
        "import-local", help="Register local files and repository fixtures."
    )
    _add_common(import_parser)
    import_parser.add_argument(
        "--sources", default=default_sources, help=f"Source registry (default {default_sources})."
    )
    import_parser.add_argument(
        "--source-id", action="append", help="Import only this source id (repeatable)."
    )
    import_parser.set_defaults(func=_cmd_import_local)

    fetch_parser = dataset_sub.add_parser(
        "fetch", help="Download direct_download sources (the only networked command)."
    )
    _add_common(fetch_parser)
    fetch_parser.add_argument("--sources", default=default_sources)
    fetch_parser.add_argument("--source-id", action="append")
    fetch_parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request timeout in seconds."
    )
    fetch_parser.set_defaults(func=_cmd_fetch)

    reconcile_parser = dataset_sub.add_parser(
        "reconcile",
        help=(
            "Remove episodes the committed registries no longer declare "
            "(fetch is additive, so a corpus version bump leaves strays)."
        ),
    )
    _add_common(reconcile_parser)
    reconcile_parser.add_argument(
        "--sources",
        action="append",
        default=None,
        help="Source registry that defines the corpus (repeatable).",
    )
    reconcile_parser.add_argument(
        "--drop-local",
        action="store_true",
        help=(
            "Also drop locally imported files and repository fixtures. Off by "
            "default: no remote registry can declare them."
        ),
    )
    reconcile_parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of removing anything.",
    )
    reconcile_parser.add_argument("--report", default=None)
    reconcile_parser.set_defaults(
        func=_cmd_reconcile,
        sources_default=[
            "ml/configs/sources_real_v1.yaml",
            "ml/configs/sources_v2.yaml",
        ],
    )

    orphans_parser = dataset_sub.add_parser(
        "orphans",
        help=(
            "Report generated files under data/ that no manifest episode "
            "references. Reports only; never deletes."
        ),
    )
    _add_common(orphans_parser)
    orphans_parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when unreferenced generated files exist.",
    )
    orphans_parser.add_argument("--report", default=None)
    orphans_parser.set_defaults(func=_cmd_orphans)

    probe_parser = dataset_sub.add_parser(
        "probe", help="Measure duration, sample rate, channels and format with ffprobe."
    )
    _add_common(probe_parser)
    probe_parser.add_argument("--episode-id", action="append")
    probe_parser.add_argument(
        "--force", action="store_true", help="Re-probe episodes that already have metadata."
    )
    probe_parser.set_defaults(func=_cmd_probe)

    normalize_parser = dataset_sub.add_parser(
        "normalize", help="Render audio to 16 kHz mono PCM WAV (cached)."
    )
    _add_common(normalize_parser)
    normalize_parser.add_argument("--episode-id", action="append")
    normalize_parser.add_argument(
        "--force", action="store_true", help="Re-render even when the cache is valid."
    )
    normalize_parser.set_defaults(func=_cmd_normalize)

    split_parser = dataset_sub.add_parser(
        "split", help="Create deterministic, leakage-safe splits."
    )
    _add_common(split_parser)
    split_parser.add_argument("--config", default=None, help="ml/configs/splits_v1.yaml")
    split_parser.add_argument("--seed", type=int, default=None)
    split_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest of the same version (invalidates results).",
    )
    split_parser.set_defaults(func=_cmd_split)

    validate_parser = dataset_sub.add_parser(
        "validate", help="Check dataset integrity. Non-zero exit on any error."
    )
    _add_common(validate_parser)
    validate_parser.add_argument(
        "--deep", action="store_true", help="Also re-verify every SHA-256 (slow)."
    )
    validate_parser.add_argument("--split-version", default="v4")
    validate_parser.add_argument("--output", default=None)
    validate_parser.set_defaults(func=_cmd_validate)

    stats_parser = dataset_sub.add_parser(
        "stats", help="Compute dataset, candidate, label and split statistics."
    )
    _add_common(stats_parser)
    stats_parser.add_argument("--split-version", default="v4")
    stats_parser.add_argument("--output-dir", default=None)
    stats_parser.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    stats_parser.set_defaults(func=_cmd_stats)

    candidates_parser = subparsers.add_parser(
        "candidates", help="Offline candidate-pool generation."
    )
    candidates_sub = candidates_parser.add_subparsers(
        dest="candidates_command", required=True
    )
    generate_parser = candidates_sub.add_parser(
        "generate", help="Generate the candidate pool from normalized audio."
    )
    _add_common(generate_parser)
    generate_parser.add_argument(
        "--config", default=None, help="ml/configs/dataset_v1.yaml"
    )
    generate_parser.add_argument("--episode-id", action="append")
    generate_parser.add_argument(
        "--include-product-padding",
        action="store_true",
        help=(
            "Also record the product's invented fallback slots for auditing. They are "
            "always is_synthetic=true and never eligible for labelling or evaluation."
        ),
    )
    generate_parser.set_defaults(func=_cmd_generate)

    label_parser = subparsers.add_parser("label", help="Local human labelling.")
    label_sub = label_parser.add_subparsers(dest="label_command", required=True)

    serve_parser = label_sub.add_parser("serve", help="Run the local labelling UI.")
    _add_common(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--context-before-ms", type=int, default=10_000)
    serve_parser.add_argument("--context-after-ms", type=int, default=10_000)
    serve_parser.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    serve_parser.add_argument(
        "--reveal-hints",
        action="store_true",
        help="Show the heuristic score and candidate sources (biases the annotator).",
    )
    serve_parser.add_argument(
        "--queue",
        default=None,
        help="Restrict the served candidates to a labelling queue artifact.",
    )
    serve_parser.add_argument(
        "--stage",
        choices=("all", "pilot", "primary"),
        default="all",
        help=(
            "With --queue, serve only one queue stage. 'pilot' runs the small "
            "diverse first pass (~24) as a controlled session; 'primary' runs the "
            "remainder. Ignored without --queue."
        ),
    )
    serve_parser.set_defaults(func=_cmd_label_serve)

    run_experiment_parser = label_sub.add_parser(
        "run-experiment",
        help=(
            "Prepare and open the labelling session for the benchmark experiment: "
            "pre-cut every clip, report what is left, and serve the UI."
        ),
    )
    _add_common(run_experiment_parser)
    run_experiment_parser.add_argument("--host", default="127.0.0.1")
    run_experiment_parser.add_argument("--port", type=int, default=8000)
    run_experiment_parser.add_argument("--context-before-ms", type=int, default=10_000)
    run_experiment_parser.add_argument("--context-after-ms", type=int, default=10_000)
    run_experiment_parser.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    run_experiment_parser.add_argument(
        "--queue",
        default=None,
        help=(
            "The labelling queue to work through. Defaults to "
            "<data-root>/labels/queue_<queue-version>.json."
        ),
    )
    run_experiment_parser.add_argument(
        "--queue-version", default="full-v2", dest="queue_version"
    )
    run_experiment_parser.add_argument(
        "--stage", choices=("all", "pilot", "primary"), default="all"
    )
    run_experiment_parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Unique human labels this round is aiming for (default: the queue size).",
    )
    run_experiment_parser.add_argument(
        "--reveal-hints",
        action="store_true",
        help="Show the heuristic score and candidate sources (biases the annotator).",
    )
    run_experiment_parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Prepare everything and report readiness without starting the server.",
    )
    run_experiment_parser.set_defaults(func=_cmd_label_run_experiment)

    queue_parser = label_sub.add_parser(
        "queue", help="Build a deterministic, stratified labelling queue."
    )
    _add_common(queue_parser)
    queue_parser.add_argument(
        "--config", default=None, help="ml/configs/labelling_queue_v1.yaml"
    )
    queue_parser.add_argument(
        "--split-version",
        default="v2",
        help="Split manifest version to stratify against (default v2, the real corpus).",
    )
    queue_parser.add_argument("--output", default=None, help="Queue artifact path.")
    queue_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing queue of the same version (orphans its labels).",
    )
    queue_parser.set_defaults(func=_cmd_label_queue)

    queue_summary_parser = label_sub.add_parser(
        "queue-summary",
        help=(
            "Write the committable counts-and-hashes reduction of a queue to "
            "artifacts/labelling/queue_summary.json."
        ),
    )
    _add_common(queue_summary_parser)
    queue_summary_parser.add_argument(
        "--queue", default=None, help="Queue artifact (default data/labels/queue_full-v2.json)."
    )
    queue_summary_parser.add_argument("--output", default=None)
    queue_summary_parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against the committed summary instead of writing it.",
    )
    queue_summary_parser.set_defaults(func=_cmd_label_queue_summary)

    check_parser = label_sub.add_parser(
        "check", help="Run label-quality controls (warnings only; never mutates)."
    )
    _add_common(check_parser)
    check_parser.add_argument("--queue", default=None, help="Assigned queue artifact.")
    check_parser.add_argument("--split-version", default="v4")
    check_parser.add_argument("--output", default=None)
    check_parser.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    check_parser.set_defaults(func=_cmd_label_check)

    export_parser = label_sub.add_parser(
        "export", help="Export human labels to versioned JSONL."
    )
    _add_common(export_parser)
    export_parser.add_argument("--output", default=None)
    export_parser.add_argument("--dataset-version", default="v1")
    export_parser.add_argument(
        "--acceptable-threshold", type=int, default=DEFAULT_ACCEPTABLE_THRESHOLD
    )
    export_parser.set_defaults(func=_cmd_label_export)

    weak_parser = label_sub.add_parser(
        "weak",
        help="Generate WEAK heuristic-derived labels for the bootstrap run "
        "(not human labels; see docs/evaluation-evidence.md).",
    )
    _add_common(weak_parser)
    weak_parser.add_argument("--output", default=None)
    weak_parser.add_argument("--dataset-version", default="v1")
    weak_parser.set_defaults(func=_cmd_label_weak)
