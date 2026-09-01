"""``transcribe``, ``features``, ``embeddings`` and ``pipeline`` commands.

Kept out of :mod:`slotify_rank.cli` and :mod:`slotify_rank.dataset_cli` so each
phase's commands stay readable, and wired in by :func:`register`. Conventions
match Phase 1 and 2: machine-readable JSON to files, a human summary on stdout,
exit 0 on success, 1 on a handled runtime failure, 2 on a usage error.

``--force`` re-runs a stage whose cache is valid. It deliberately does **not**
bypass validation: ``features validate`` still exits non-zero on an integrity
failure, because the point of the check is to catch exactly the corruption a
forced re-run might introduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.config.feature_settings import (
    load_embedding_config,
    load_feature_config,
    load_pipeline_config,
    load_transcription_config,
)
from slotify_rank.data import manifests
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.schema import DatasetCandidate, EpisodeRecord
from slotify_rank.data.splits import read_split_manifest
from slotify_rank.features.assemble import read_feature_manifest
from slotify_rank.features.stats import (
    compute_feature_statistics,
    write_feature_statistics,
)
from slotify_rank.features.validate import validate_features
from slotify_rank.pipeline.feature_pipeline import (
    SELECTABLE_STAGES,
    run_feature_pipeline,
)
from slotify_rank.pipeline.stages import StageContext, select_episodes
from slotify_rank.pipeline.state import STAGE_NAMES, StageLedger

__all__ = ["register"]

_STAGE_RUN_SUMMARY = "feature_run.json"


def _paths(args: argparse.Namespace) -> DataPaths:
    paths = DataPaths(data_root=Path(args.data_root) if args.data_root else None)
    paths.mkdirs()
    return paths


def _context(args: argparse.Namespace, paths: DataPaths) -> StageContext:
    return StageContext(
        paths=paths,
        transcription=load_transcription_config(getattr(args, "transcription_config", None)),
        features=load_feature_config(getattr(args, "features_config", None)),
        embeddings=load_embedding_config(getattr(args, "embeddings_config", None)),
        pipeline=load_pipeline_config(getattr(args, "pipeline_config", None)),
        force=bool(getattr(args, "force", False)),
        retry_failed=bool(getattr(args, "retry_failed", False)),
    )


def _split_lookup(paths: DataPaths, version: str) -> dict[str, str]:
    path = paths.split_manifest(version)
    if not path.is_file():
        return {}
    return dict(read_split_manifest(path).by_episode)


def _load_corpus(
    args: argparse.Namespace, paths: DataPaths
) -> tuple[list[EpisodeRecord], list[DatasetCandidate], dict[str, str]]:
    episodes = list(manifests.read_episodes(paths.episodes_manifest))
    candidates = list(manifests.read_candidates(paths.candidates_manifest))
    split_lookup = _split_lookup(paths, getattr(args, "split_version", "v4"))
    selected = select_episodes(
        episodes,
        episode_ids=getattr(args, "episode_id", None),
        content_types=getattr(args, "content_type", None),
        limit=getattr(args, "limit", None),
        splits=getattr(args, "split", None),
        split_lookup=split_lookup,
    )
    return selected, candidates, split_lookup


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _report_pipeline(result) -> None:
    for outcome in result.outcomes:
        print(
            f"  {outcome.stage:<16} processed={outcome.processed} "
            f"skipped={outcome.skipped} failed={outcome.failed} "
            f"cache_hits={outcome.cache_hits} ({outcome.seconds:.1f}s)"
        )
    if result.failures:
        print(f"  {len(result.failures)} candidate failure(s):")
        for candidate_id, reason in result.failures[:5]:
            print(f"    {candidate_id}: {reason}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_stages(args: argparse.Namespace, stages: Sequence[str]) -> int:
    paths = _paths(args)
    context = _context(args, paths)
    episodes, candidates, split_lookup = _load_corpus(args, paths)
    if not episodes:
        print("No normalized episodes selected; nothing to do.")
        return 0

    print(f"{len(episodes)} episode(s) selected.")
    result = run_feature_pipeline(
        context, episodes, candidates, split_lookup, stages=stages
    )
    _report_pipeline(result)
    _write_json(paths.feature_state_dir / _STAGE_RUN_SUMMARY, result.to_dict())

    failed = sum(outcome.failed for outcome in result.outcomes)
    if failed and context.pipeline.fail_fast:
        return 1
    return 0


def _cmd_transcribe_run(args: argparse.Namespace) -> int:
    return _run_stages(args, ["transcribe"])


def _cmd_transcribe_validate(args: argparse.Namespace) -> int:
    from slotify_rank.transcription.cache import read_transcript, transcript_path

    paths = _paths(args)
    episodes, _, _ = _load_corpus(args, paths)
    errors: list[str] = []
    checked = 0
    for episode in episodes:
        path = transcript_path(paths, episode.episode_id)
        if not path.is_file():
            continue
        try:
            transcript = read_transcript(path)
        except (ValueError, TypeError) as error:
            errors.append(f"{episode.episode_id}: {error}")
            continue
        checked += 1
        duration = episode.normalized_duration_ms or transcript.audio_duration_ms
        for segment in transcript.segments:
            if segment.end_ms > duration:
                errors.append(
                    f"{episode.episode_id}: segment {segment.segment_id} ends at "
                    f"{segment.end_ms} ms, past the episode end ({duration} ms)"
                )

    print(f"Validated {checked} transcript(s).")
    for message in errors[:20]:
        print(f"  error: {message}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} transcript error(s).", file=sys.stderr)
        return 1
    return 0


def _cmd_features_acoustic(args: argparse.Namespace) -> int:
    return _run_stages(args, ["acoustic"])


def _cmd_embeddings_audio(args: argparse.Namespace) -> int:
    return _run_stages(args, ["audio_embedding"])


def _cmd_embeddings_text(args: argparse.Namespace) -> int:
    return _run_stages(args, ["text_embedding"])


def _cmd_features_assemble(args: argparse.Namespace) -> int:
    return _run_stages(args, ["assemble"])


def _cmd_pipeline_features(args: argparse.Namespace) -> int:
    """The full workflow: validate inputs, run every stage, validate, report."""
    paths = _paths(args)
    context = _context(args, paths)
    episodes, candidates, split_lookup = _load_corpus(args, paths)
    if not episodes:
        print("No normalized episodes selected; nothing to do.")
        return 0

    stages = args.stage or list(SELECTABLE_STAGES)
    print(f"{len(episodes)} episode(s) selected; stages: {', '.join(stages)}")
    result = run_feature_pipeline(
        context, episodes, candidates, split_lookup, stages=stages
    )
    _report_pipeline(result)
    _write_json(paths.feature_state_dir / _STAGE_RUN_SUMMARY, result.to_dict())

    if "assemble" not in stages:
        return 0

    header, records = read_feature_manifest(paths.features_manifest)
    report = validate_features(
        paths=paths,
        header=header,
        records=records,
        candidates=candidates,
        episodes=list(manifests.read_episodes(paths.episodes_manifest)),
        split_lookup=split_lookup,
        deep=args.deep,
    )
    statistics = compute_feature_statistics(
        paths=paths,
        header=header,
        records=records,
        episodes=episodes,
        stage_outcomes=[outcome.to_dict() for outcome in result.outcomes],
    )
    written = write_feature_statistics(paths, statistics)

    print(
        f"\n{statistics['candidates']['processed']} candidate(s): "
        f"{statistics['candidates']['complete_multimodal']} complete multimodal, "
        f"{statistics['candidates']['audio_only']} audio-only, "
        f"{statistics['candidates']['text_only']} text-only, "
        f"{statistics['candidates']['handcrafted_only']} handcrafted-only."
    )
    print(f"Statistics: {written[0].parent}")

    if not report.ok:
        for issue in report.errors[:20]:
            print(f"  error [{issue.code}] {issue.message}", file=sys.stderr)
        print(f"{len(report.errors)} validation error(s).", file=sys.stderr)
        return 1
    for issue in report.warnings[:10]:
        print(f"  warning [{issue.code}] {issue.message}")
    print("Validation passed.")
    return 0


def _cmd_features_validate(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes = list(manifests.read_episodes(paths.episodes_manifest))
    candidates = list(manifests.read_candidates(paths.candidates_manifest))
    split_lookup = _split_lookup(paths, args.split_version)

    header, records = read_feature_manifest(paths.features_manifest)
    report = validate_features(
        paths=paths,
        header=header,
        records=records,
        candidates=candidates,
        episodes=episodes,
        split_lookup=split_lookup,
        deep=args.deep,
    )
    output = Path(args.output) if args.output else (
        paths.feature_artifacts_dir / "feature_validation_report.json"
    )
    _write_json(output, report.to_dict())

    print(
        f"Checked {len(records)} feature record(s): {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)."
    )
    print(f"Report: {output}")
    for issue in report.warnings[:10]:
        print(f"  warning [{issue.code}] {issue.message}")
    if not report.ok:
        for issue in report.errors[:20]:
            print(f"  error [{issue.code}] {issue.message}", file=sys.stderr)
        return 1
    return 0


def _cmd_features_stats(args: argparse.Namespace) -> int:
    paths = _paths(args)
    episodes, _, _ = _load_corpus(args, paths)
    header, records = read_feature_manifest(paths.features_manifest)
    statistics = compute_feature_statistics(
        paths=paths,
        header=header,
        records=records,
        episodes=episodes or list(manifests.read_episodes(paths.episodes_manifest)),
    )
    written = write_feature_statistics(
        paths, statistics, Path(args.output_dir) if args.output_dir else None
    )
    print(f"Wrote {len(written)} report(s):")
    for path in written:
        print(f"  {path}")
    return 0


def _cmd_pipeline_status(args: argparse.Namespace) -> int:
    """Per-stage state for every selected episode. Loads no model."""
    paths = _paths(args)
    context = _context(args, paths)
    episodes, candidates, _ = _load_corpus(args, paths)

    print(f"{len(episodes)} normalized episode(s) selected.\n")
    print(f"{'stage':<16} {'complete':>9} {'failed':>7} {'partial':>8} {'stale':>6}")
    for stage in STAGE_NAMES:
        ledger = StageLedger.read(paths.stage_ledger(stage), stage)
        counts = ledger.counts()
        print(
            f"{stage:<16} {counts['complete']:>9} {counts['failed']:>7} "
            f"{counts['partial']:>8} {counts['stale']:>6}"
        )

    failures: list[tuple[str, str, str]] = []
    for stage in STAGE_NAMES:
        ledger = StageLedger.read(paths.stage_ledger(stage), stage)
        for entry in ledger.entries():
            if entry.status == "failed":
                failures.append((stage, entry.episode_id, entry.failure_reason or ""))
    if failures:
        print(f"\n{len(failures)} recorded failure(s) (retry with --retry-failed):")
        for stage, episode_id, reason in failures[:10]:
            print(f"  {stage:<16} {episode_id}: {reason}")

    if paths.features_manifest.is_file():
        header, records = read_feature_manifest(paths.features_manifest)
        print(
            f"\nFeature manifest: {len(records)} record(s), "
            f"{header.get('handcrafted_feature_count')} handcrafted feature(s)."
        )
    else:
        print("\nFeature manifest: not yet assembled.")
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root", default=None, help="Override the data root (default <repo>/data)."
    )
    parser.add_argument(
        "--episode-id", action="append", help="Process only this episode (repeatable)."
    )
    parser.add_argument(
        "--content-type", action="append", help="Filter by content type (repeatable)."
    )
    parser.add_argument(
        "--split", action="append", help="Filter by dataset split (repeatable)."
    )
    # v4 is the split the committed experiment pins. This defaulted to v1,
    # so a bare `features stats` would recompute the by-split feature counts
    # against a superseded split and write them into the same
    # feature_statistics.json the evidence report reads.
    parser.add_argument("--split-version", default="v4")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N episodes."
    )


def _add_configs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transcription-config", default=None, help="ml/configs/transcription_v1.yaml"
    )
    parser.add_argument(
        "--features-config", default=None, help="ml/configs/features_v1.yaml"
    )
    parser.add_argument(
        "--embeddings-config", default=None, help="ml/configs/embeddings_v1.yaml"
    )
    parser.add_argument(
        "--pipeline-config", default=None, help="ml/configs/feature_pipeline_v1.yaml"
    )


def _add_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the cache is valid. Does NOT bypass validation.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry episodes previously recorded as failed.",
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach the Phase 3 command groups to the existing parser."""

    # -- transcribe --------------------------------------------------------
    transcribe = subparsers.add_parser(
        "transcribe", help="Local timestamped transcription."
    )
    transcribe_sub = transcribe.add_subparsers(dest="transcribe_command", required=True)

    run_parser = transcribe_sub.add_parser(
        "run", help="Transcribe eligible episodes with local Whisper (cached)."
    )
    _add_common(run_parser)
    _add_configs(run_parser)
    _add_execution(run_parser)
    run_parser.set_defaults(func=_cmd_transcribe_run)

    validate_transcripts = transcribe_sub.add_parser(
        "validate", help="Check stored transcripts. Non-zero exit on any error."
    )
    _add_common(validate_transcripts)
    validate_transcripts.set_defaults(func=_cmd_transcribe_validate)

    # -- features ----------------------------------------------------------
    features = subparsers.add_parser(
        "features", help="Handcrafted candidate features and their assembly."
    )
    features_sub = features.add_subparsers(dest="features_command", required=True)

    acoustic_parser = features_sub.add_parser(
        "acoustic", help="Extract acoustic, structural and text-scalar features."
    )
    _add_common(acoustic_parser)
    _add_configs(acoustic_parser)
    _add_execution(acoustic_parser)
    acoustic_parser.set_defaults(func=_cmd_features_acoustic)

    assemble_parser = features_sub.add_parser(
        "assemble", help="Join features and embeddings into candidate records."
    )
    _add_common(assemble_parser)
    _add_configs(assemble_parser)
    _add_execution(assemble_parser)
    assemble_parser.set_defaults(func=_cmd_features_assemble)

    validate_parser = features_sub.add_parser(
        "validate", help="Check feature integrity. Non-zero exit on any error."
    )
    _add_common(validate_parser)
    validate_parser.add_argument(
        "--deep", action="store_true", help="Also open every referenced array (slow)."
    )
    validate_parser.add_argument("--output", default=None)
    validate_parser.set_defaults(func=_cmd_features_validate)

    stats_parser = features_sub.add_parser(
        "stats", help="Feature, transcription, embedding and cache statistics."
    )
    _add_common(stats_parser)
    stats_parser.add_argument("--output-dir", default=None)
    stats_parser.set_defaults(func=_cmd_features_stats)

    # -- embeddings --------------------------------------------------------
    embeddings = subparsers.add_parser(
        "embeddings", help="Frozen speech and transcript representations."
    )
    embeddings_sub = embeddings.add_subparsers(
        dest="embeddings_command", required=True
    )

    audio_parser = embeddings_sub.add_parser(
        "audio", help="Frozen Whisper encoder representations (cached per episode)."
    )
    _add_common(audio_parser)
    _add_configs(audio_parser)
    _add_execution(audio_parser)
    audio_parser.set_defaults(func=_cmd_embeddings_audio)

    text_parser = embeddings_sub.add_parser(
        "text", help="Frozen MiniLM transcript-context embeddings."
    )
    _add_common(text_parser)
    _add_configs(text_parser)
    _add_execution(text_parser)
    text_parser.set_defaults(func=_cmd_embeddings_text)

    # -- pipeline ----------------------------------------------------------
    pipeline = subparsers.add_parser(
        "pipeline", help="End-to-end feature generation and status."
    )
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command", required=True)

    features_pipeline = pipeline_sub.add_parser(
        "features",
        help="Transcribe, extract, embed, assemble, validate and report.",
    )
    _add_common(features_pipeline)
    _add_configs(features_pipeline)
    _add_execution(features_pipeline)
    features_pipeline.add_argument(
        "--stage",
        action="append",
        choices=list(SELECTABLE_STAGES),
        help="Run only this stage (repeatable). Default: all, in order.",
    )
    features_pipeline.add_argument(
        "--deep", action="store_true", help="Deep validation of every array."
    )
    features_pipeline.set_defaults(func=_cmd_pipeline_features)

    status_parser = pipeline_sub.add_parser(
        "status", help="Per-stage progress. Loads no model."
    )
    _add_common(status_parser)
    _add_configs(status_parser)
    status_parser.set_defaults(func=_cmd_pipeline_status)
