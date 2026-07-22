"""Command-line interface for the Slotify ranking project.

Phase 1 -- the baseline and its metrics (offline, CPU-only, no model downloads,
no paid APIs)::

    python -m slotify_rank.cli version
    python -m slotify_rank.cli config show [--profile NAME] [--output PATH]
    python -m slotify_rank.cli heuristic rank --input EPISODES.json --output RANKINGS.json
    python -m slotify_rank.cli evaluate --predictions RANKINGS.json --labels LABELS.json [--output REPORT.json]

Phase 2 -- the dataset foundation. Only ``dataset fetch`` touches the network::

    python -m slotify_rank.cli dataset import-local --sources ml/configs/sources.yaml
    python -m slotify_rank.cli dataset fetch --sources ml/configs/sources.yaml
    python -m slotify_rank.cli dataset probe
    python -m slotify_rank.cli dataset normalize
    python -m slotify_rank.cli candidates generate --config ml/configs/dataset_v1.yaml
    python -m slotify_rank.cli dataset split --config ml/configs/splits_v1.yaml
    python -m slotify_rank.cli dataset validate [--deep]
    python -m slotify_rank.cli dataset stats
    python -m slotify_rank.cli label serve [--port 8000]
    python -m slotify_rank.cli label export

Phase 3 -- the multimodal feature pipeline. Local models only; the sole network
access is the one-time Hugging Face model download::

    python -m slotify_rank.cli transcribe run [--episode-id ID]
    python -m slotify_rank.cli transcribe validate
    python -m slotify_rank.cli features acoustic
    python -m slotify_rank.cli embeddings audio
    python -m slotify_rank.cli embeddings text
    python -m slotify_rank.cli features assemble
    python -m slotify_rank.cli features validate [--deep]
    python -m slotify_rank.cli features stats
    python -m slotify_rank.cli pipeline features [--stage NAME] [--limit N]
    python -m slotify_rank.cli pipeline status

The console script ``slotify-rank`` is equivalent to ``python -m slotify_rank.cli``.

Exit codes: ``0`` success, ``1`` runtime failure (message on stderr), ``2``
usage error (argparse). Machine-readable JSON goes to ``--output``; the human
summary goes to stdout, so the two never interleave.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from slotify_rank.candidates.heuristic import rank_episodes
from slotify_rank.candidates.schema import EpisodeInput
from slotify_rank.config.settings import load_heuristic_config
from slotify_rank.config.versions import CANDIDATE_SCHEMA_VERSION, PACKAGE_VERSION
from slotify_rank.evaluation.metrics import (
    EpisodeJudgements,
    EpisodePrediction,
    MetricConfig,
    evaluate_rankings,
)

__all__ = ["main", "build_parser"]


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Input file not found: {path}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _as_episode_list(raw: Any, source: Path) -> list[dict[str, Any]]:
    """Accept either a bare episode object or ``{"episodes": [...]}``."""
    if isinstance(raw, dict) and "episodes" in raw:
        episodes = raw["episodes"]
    elif isinstance(raw, dict):
        episodes = [raw]
    elif isinstance(raw, list):
        episodes = raw
    else:
        raise ValueError(
            f"{source} must contain an episode object, a list of episodes, or "
            f'an object with an "episodes" key; got {type(raw).__name__}'
        )
    if not isinstance(episodes, list):
        raise ValueError(f'{source}: "episodes" must be a list')
    if not episodes:
        raise ValueError(f"{source} contains no episodes")
    return episodes


def _cmd_version(args: argparse.Namespace) -> int:
    config = load_heuristic_config()
    payload = {
        "package_version": PACKAGE_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "heuristic_config_version": config.config_version,
        "canonical_baseline": config.canonical_profile_name,
        "python": sys.version.split()[0],
    }
    if args.output:
        _write_json(Path(args.output), payload)
    for key, value in payload.items():
        print(f"{key}: {value}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    config = load_heuristic_config()
    profile = config.profile(args.profile)
    payload = {
        "config_path": str(config.path),
        "config_version": config.config_version,
        "canonical_profile": config.canonical_profile_name,
        "profile": asdict(profile),
    }
    if args.output:
        _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_heuristic_rank(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    raw = _read_json(input_path)
    episodes = [
        EpisodeInput.from_mapping(entry) for entry in _as_episode_list(raw, input_path)
    ]
    config = load_heuristic_config()
    rankings = rank_episodes(episodes, config=config, profile_name=args.profile)

    payload = {
        "baseline_version": config.canonical_profile_name,
        "config_version": config.config_version,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "source": str(input_path),
        "episodes": [ranking.to_dict() for ranking in rankings],
    }
    _write_json(Path(args.output), payload)

    total_candidates = sum(len(r.candidates) for r in rankings)
    print(f"Ranked {len(rankings)} episode(s), {total_candidates} candidate record(s).")
    print(f"Baseline: {config.canonical_profile_name} ({config.config_version})")
    for ranking in rankings:
        synthetic = sum(1 for c in ranking.candidates if c.is_synthetic)
        points = ", ".join(f"{point:.3f}s" for point in ranking.points)
        note = f"  [{synthetic} synthetic padding slot(s)]" if synthetic else ""
        print(f"  {ranking.episode_id}: {points}{note}")
    print(f"Wrote {args.output}")
    return 0


def _load_predictions(path: Path) -> list[EpisodePrediction]:
    raw = _read_json(path)
    episodes = _as_episode_list(raw, path)
    predictions: list[EpisodePrediction] = []
    for entry in episodes:
        if "episode_id" not in entry:
            raise KeyError(f"{path}: an episode is missing 'episode_id'")
        if "ranked_candidate_ids" in entry:
            ranked = list(entry["ranked_candidate_ids"])
            scores = {
                candidate["candidate_id"]: float(candidate["total_score"])
                for candidate in entry.get("candidates", ())
                if candidate.get("rank") is not None
            }
        else:
            raise KeyError(
                f"{path}: episode {entry['episode_id']!r} has no 'ranked_candidate_ids'. "
                "Pass the output of `heuristic rank`."
            )
        predictions.append(
            EpisodePrediction(
                episode_id=str(entry["episode_id"]),
                ranked_candidate_ids=ranked,
                scores=scores or None,
            )
        )
    return predictions


def _load_judgements(path: Path) -> list[EpisodeJudgements]:
    raw = _read_json(path)
    episodes = _as_episode_list(raw, path)
    judgements: list[EpisodeJudgements] = []
    for entry in episodes:
        if "episode_id" not in entry:
            raise KeyError(f"{path}: an episode is missing 'episode_id'")
        if "relevance" not in entry:
            raise KeyError(
                f"{path}: episode {entry['episode_id']!r} is missing 'relevance' "
                "(a mapping of candidate_id -> graded label)"
            )
        relevance = entry["relevance"]
        if not isinstance(relevance, dict):
            raise TypeError(
                f"{path}: 'relevance' for episode {entry['episode_id']!r} must be an object"
            )
        judgements.append(
            EpisodeJudgements(
                episode_id=str(entry["episode_id"]),
                relevance={str(k): float(v) for k, v in relevance.items()},
            )
        )
    return judgements


def _cmd_evaluate(args: argparse.Namespace) -> int:
    predictions = _load_predictions(Path(args.predictions))
    judgements = _load_judgements(Path(args.labels))
    config = MetricConfig(
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        min_pair_gap=args.min_pair_gap,
    )
    report = evaluate_rankings(predictions, judgements, config)

    payload = {
        "package_version": PACKAGE_VERSION,
        "predictions": str(args.predictions),
        "labels": str(args.labels),
        **report.to_dict(),
    }
    if args.output:
        _write_json(Path(args.output), payload)

    print(f"Episodes evaluated: {report.n_episodes}")
    print(
        f"Relevance threshold: >= {config.relevance_threshold} | k = {config.k} | "
        f"recall denominator: {config.to_dict()['recall_denominator']}"
    )
    for name, value in report.aggregate.items():
        skipped = report.coverage[f"{name}_n_skipped"]
        rendered = "undefined" if value is None else f"{value:.4f}"
        suffix = f"  ({skipped} episode(s) undefined, excluded)" if skipped else ""
        print(f"  {name}: {rendered}{suffix}")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slotify-rank",
        description="Slotify multimodal breakpoint ranking - Phase 1 (offline heuristic baseline).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print version stamps.")
    version_parser.add_argument("--output", help="Also write the versions as JSON.")
    version_parser.set_defaults(func=_cmd_version)

    config_parser = subparsers.add_parser("config", help="Inspect configuration.")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    show_parser = config_sub.add_parser(
        "show", help="Print the resolved heuristic configuration."
    )
    show_parser.add_argument(
        "--profile", default=None, help="Profile name (default: the canonical baseline)."
    )
    show_parser.add_argument("--output", help="Also write the configuration as JSON.")
    show_parser.set_defaults(func=_cmd_config_show)

    heuristic_parser = subparsers.add_parser(
        "heuristic", help="Run the deterministic offline baseline."
    )
    heuristic_sub = heuristic_parser.add_subparsers(
        dest="heuristic_command", required=True
    )
    rank_parser = heuristic_sub.add_parser(
        "rank", help="Rank candidate breakpoints for one or more episodes."
    )
    rank_parser.add_argument(
        "--input", required=True, help="JSON file of episode inputs."
    )
    rank_parser.add_argument(
        "--output", required=True, help="Destination JSON file for the rankings."
    )
    rank_parser.add_argument(
        "--profile", default=None, help="Profile name (default: the canonical baseline)."
    )
    rank_parser.set_defaults(func=_cmd_heuristic_rank)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score saved rankings against labelled relevance."
    )
    evaluate_parser.add_argument(
        "--predictions", required=True, help="Output of `heuristic rank`."
    )
    evaluate_parser.add_argument(
        "--labels", required=True, help="JSON file of graded relevance labels."
    )
    evaluate_parser.add_argument("--output", help="Destination JSON file for the report.")
    evaluate_parser.add_argument("--k", type=int, default=3, help="Cutoff k (default 3).")
    evaluate_parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        dest="relevance_threshold",
        help="Binary relevance threshold for P/R/F1/MRR (default 4.0).",
    )
    evaluate_parser.add_argument(
        "--min-pair-gap",
        type=float,
        default=1.0,
        dest="min_pair_gap",
        help="Minimum label gap for a pair to count in pairwise accuracy (default 1.0).",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    # Phase 2: dataset / candidates / label. Registered from a separate module so
    # this file stays about the baseline and the metrics.
    from slotify_rank.dataset_cli import register as register_dataset_commands

    register_dataset_commands(subparsers)

    # Phase 3: transcribe / features / embeddings / pipeline. Imported lazily
    # for the same reason -- and additionally because the heavy ML extras are
    # optional, so `slotify-rank version` must not require torch to be present.
    from slotify_rank.features_cli import register as register_feature_commands

    register_feature_commands(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
