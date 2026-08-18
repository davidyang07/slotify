"""``slotify-rank infer`` -- the product-facing inference commands.

    slotify-rank infer describe --checkpoint PATH
    slotify-rank infer rank --audio PATH --checkpoint PATH --output RESULT.json

``describe`` reads a checkpoint's identity without scoring anything, which is
what the preflight check and the Express API's startup probe use. ``rank`` is
the request path: one audio file in, a ranked, provenance-stamped JSON document
out.

Machine-readable JSON goes to ``--output`` (or to stdout with
``--output -``); human progress goes to stderr. The two never interleave,
because the Node service parses stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["register"]


def _write_json(path: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _cmd_describe(args: argparse.Namespace) -> int:
    from slotify_rank.inference.predictor import RankerPredictor

    predictor = RankerPredictor.load(args.checkpoint, args.normalizer)
    payload = predictor.identity.to_dict()
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from slotify_rank.inference.episode import (
        PipelineConfigs,
        episode_examples,
        prepare_episode,
        workspace,
    )
    from slotify_rank.inference.predictor import RankerPredictor, score_examples
    from slotify_rank.inference.schema import ExcludedCandidate, InferenceResult

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    started = time.perf_counter()
    predictor = RankerPredictor.load(args.checkpoint, args.normalizer)
    load_seconds = round(time.perf_counter() - started, 3)
    log(
        f"  model: {predictor.identity.model_variant} "
        f"({predictor.identity.parameter_count} parameters), loaded in {load_seconds}s"
    )

    scratch = Path(args.workspace) if args.workspace else Path(
        tempfile.mkdtemp(prefix="slotify-infer-")
    )
    configs = PipelineConfigs.load(args.config_dir)

    with workspace(scratch, cleanup=not args.keep_workspace) as space:
        prepared = prepare_episode(
            space,
            args.audio,
            configs,
            episode_id=args.episode_id,
            transcribe=not args.no_transcribe,
            log=log,
        )
        examples = list(episode_examples(prepared))
        started = time.perf_counter()
        ranked = score_examples(
            predictor,
            examples,
            prepared.timestamps_ms,
            prepared.feature_status,
        )
        score_seconds = round(time.perf_counter() - started, 3)

        timings = dict(prepared.timings_seconds)
        timings["model_load"] = load_seconds
        timings["score"] = score_seconds

        if args.top is not None and args.top > 0:
            ranked = ranked[: args.top]

        result = InferenceResult(
            episode_id=prepared.episode.episode_id,
            duration_seconds=prepared.duration_seconds,
            ranked=tuple(ranked),
            excluded=tuple(
                ExcludedCandidate(
                    candidate_id=exclusion.candidate_id,
                    reason=exclusion.reason,
                    detail=exclusion.detail,
                )
                for exclusion in prepared.dataset.exclusions
            ),
            model=predictor.identity,
            candidate_count=len(prepared.candidates),
            warnings=tuple(prepared.warnings),
            timings_seconds=timings,
        )

    _write_json(args.output, result.to_dict())
    log(
        f"  ranked {len(result.ranked)} of {result.candidate_count} candidate(s) "
        f"in {sum(timings.values()):.1f}s"
    )
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "infer", help="Score candidates with a trained ranker (no training)."
    )
    sub = parser.add_subparsers(dest="infer_command", required=True)

    describe = sub.add_parser(
        "describe", help="Print a checkpoint's identity without scoring."
    )
    describe.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    describe.add_argument(
        "--normalizer",
        default=None,
        help="Path to normalizer.json (default: beside the checkpoint).",
    )
    describe.add_argument("--output", default=None, help="Write the identity as JSON.")
    describe.set_defaults(func=_cmd_describe)

    rank = sub.add_parser("rank", help="Rank one audio file's candidate breakpoints.")
    rank.add_argument("--audio", required=True, help="Audio file to analyse.")
    rank.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    rank.add_argument(
        "--normalizer",
        default=None,
        help="Path to normalizer.json (default: beside the checkpoint).",
    )
    rank.add_argument(
        "--output",
        default="-",
        help="Destination JSON file, or '-' for stdout (the default).",
    )
    rank.add_argument(
        "--episode-id",
        default="upload",
        dest="episode_id",
        help="Episode id used to build candidate ids (default: upload).",
    )
    rank.add_argument(
        "--top",
        type=int,
        default=None,
        help="Return only the top N. Omit to return every scored candidate.",
    )
    rank.add_argument(
        "--config-dir",
        default=None,
        dest="config_dir",
        help="Directory of pipeline YAML configs (default: the built-in values, "
        "which ml/configs/ currently matches exactly).",
    )
    rank.add_argument(
        "--no-transcribe",
        action="store_true",
        dest="no_transcribe",
        help="Skip Whisper transcription. Faster, but every candidate is then "
        "scored with its text modality masked, and the result says so.",
    )
    rank.add_argument(
        "--workspace",
        default=None,
        help="Scratch directory for the throwaway single-episode corpus.",
    )
    rank.add_argument(
        "--keep-workspace",
        action="store_true",
        dest="keep_workspace",
        help="Do not delete the scratch workspace (for debugging).",
    )
    rank.add_argument(
        "--quiet", action="store_true", help="Suppress progress output on stderr."
    )
    rank.set_defaults(func=_cmd_rank)
