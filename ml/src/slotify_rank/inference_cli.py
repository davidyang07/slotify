"""``slotify-rank infer`` -- the product-facing inference commands.

    slotify-rank infer describe --checkpoint PATH
    slotify-rank infer rank --audio PATH --checkpoint PATH --output RESULT.json [--work-dir DIR]
    slotify-rank infer serve --checkpoint PATH
    slotify-rank infer warm [--check]

``describe`` reads a checkpoint's identity without scoring anything, which is
what the preflight check and the Express API's startup probe use. ``rank`` is
the one-shot request path: one audio file in, a ranked, provenance-stamped JSON
document out. ``serve`` is the same request path as a long-lived process: it
loads the checkpoint, the normalizer and every encoder once, then answers
newline-delimited JSON requests on stdin with one JSON line each on stdout (see
:mod:`slotify_rank.inference.worker`). ``warm`` downloads exactly the model
snapshots the configs pin, so ``rank`` and ``serve`` can run offline.

Machine-readable JSON goes to ``--output`` (or to stdout with
``--output -``); human progress goes to stderr. The two never interleave,
because the Node service parses stdout. ``serve`` goes further and moves file
descriptor 1 onto stderr for everything but its own responses.

``rank`` and ``serve`` read the pipeline settings from the committed
``ml/configs/`` unless ``--config-dir`` names another directory, so serving
builds features with the same values the corpus pipeline does. The served
feature layout is compared with the checkpoint's by column name before anything
is scored (see :func:`slotify_rank.inference.predictor.align_examples`).

**Model snapshots.** When every model revision in those configs is a commit SHA
whose snapshot is in the local Hugging Face cache, both commands set
``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` before any Hugging Face
library loads, so a slow or unreachable Hub cannot slow or break ranking. When a
pinned snapshot is missing, ``serve`` refuses to start and names ``infer warm``;
``rank`` says the same on stderr and downloads that pinned revision, as it
always has.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TextIO

__all__ = ["register"]


def _write_json(path: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _stderr_logger(quiet: bool = False) -> Callable[[str], None]:
    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    return log


def _choose_hub_mode(configs: Any, log: Callable[[str], None], require_local: bool) -> Any:
    """Go offline when every pinned snapshot is local; otherwise say why not.

    Must run before anything imports ``huggingface_hub`` / ``transformers``.
    ``require_local`` (the worker) turns a missing pinned snapshot into a
    startup error naming ``infer warm``; without it (one-shot ``rank``) the
    pinned revision is downloaded on first use, as before.
    """
    from slotify_rank.inference.hub import SnapshotsMissing, apply_offline_env, plan_hub_access

    plan = plan_hub_access(configs)
    if plan.unpinned:
        log(f"  models: {plan.unpinned_message()}")
    if plan.missing:
        if require_local:
            raise SnapshotsMissing(plan.missing_message())
        log(f"  models: {plan.missing_message()} Downloading them for this run instead.")
    if plan.offline:
        apply_offline_env()
        log(
            "  models: every pinned snapshot is in the local cache; Hugging Face Hub "
            "access is off (HF_HUB_OFFLINE=1)"
        )
    return plan


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
    from slotify_rank.inference.episode import PipelineConfigs
    from slotify_rank.inference.predictor import RankerPredictor
    from slotify_rank.inference.service import (
        Terminated,
        rank_audio,
        request_scratch,
        terminate_on_sigterm,
    )

    log = _stderr_logger(args.quiet)
    configs = PipelineConfigs.load(args.config_dir)
    _choose_hub_mode(configs, log, require_local=False)

    started = time.perf_counter()
    predictor = RankerPredictor.load(args.checkpoint, args.normalizer)
    load_seconds = round(time.perf_counter() - started, 3)
    log(
        f"  model: {predictor.identity.model_variant} "
        f"({predictor.identity.parameter_count} parameters), loaded in {load_seconds}s"
    )

    options = dict(
        cleanup=not args.keep_workspace,
        episode_id=args.episode_id,
        transcribe=not args.no_transcribe,
        top=args.top,
        model_load_seconds=load_seconds,
        log=log,
    )
    # A SIGTERM mid-run unwinds through the workspace's cleanup instead of
    # killing the process with the user's upload still on disk.
    with terminate_on_sigterm():
        try:
            if args.work_dir:
                with request_scratch(args.work_dir) as scratch:
                    result = rank_audio(predictor, configs, args.audio, scratch, **options)
            else:
                scratch = (
                    Path(args.workspace)
                    if args.workspace
                    else Path(tempfile.mkdtemp(prefix="slotify-infer-"))
                )
                result = rank_audio(predictor, configs, args.audio, scratch, **options)
        except Terminated as stop:
            log("  stopped by SIGTERM; scratch files removed")
            return 128 + stop.signum

    _write_json(args.output, result.to_dict())
    log(
        f"  ranked {len(result.ranked)} of {result.candidate_count} candidate(s) "
        f"in {sum(result.timings_seconds.values()):.1f}s"
    )
    return 0


def _claim_stdout() -> TextIO:
    """Keep file descriptor 1 for the protocol; point everything else at stderr.

    After this, ``print()``, a library's progress bar, or a C extension writing
    to fd 1 all reach stderr, and only the returned stream reaches the caller's
    stdout. Done before any model library is imported.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n")


def _cmd_serve(args: argparse.Namespace) -> int:
    protocol = _claim_stdout()
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    from slotify_rank.embeddings.resident import resident_count, resident_models
    from slotify_rank.inference.episode import PipelineConfigs
    from slotify_rank.inference.hub import WARM_COMMAND, SnapshotsMissing
    from slotify_rank.inference.predictor import RankerPredictor
    from slotify_rank.inference.service import Terminated, terminate_on_sigterm
    from slotify_rank.inference.worker import (
        RankerWorker,
        last_meaningful_line,
        preload_models,
        serve,
    )

    log = _stderr_logger()
    with terminate_on_sigterm():
        try:
            started = time.perf_counter()
            configs = PipelineConfigs.load(args.config_dir)
            plan = _choose_hub_mode(configs, log, require_local=True)
            predictor = RankerPredictor.load(args.checkpoint, args.normalizer)
            with resident_models():
                try:
                    preload_models(configs)
                except OSError as error:
                    if not plan.offline:
                        raise
                    # Snapshot directory present but incomplete: the offline
                    # load names the missing file; the fix is the same.
                    raise SnapshotsMissing(
                        f"A pinned model could not be loaded offline "
                        f"({last_meaningful_line(str(error))}); run `{WARM_COMMAND}`."
                    ) from error
                identity = predictor.identity
                log(
                    f"  worker: ready in {time.perf_counter() - started:.1f}s -- "
                    f"{identity.model_variant} {identity.model_run_id} "
                    f"({identity.parameter_count} parameters), {resident_count()} "
                    "encoder(s) resident"
                )
                worker = RankerWorker(predictor=predictor, configs=configs, log=log)
                return serve(worker.handle, sys.stdin, protocol, log)
        except (Terminated, KeyboardInterrupt):
            log("  worker: stopping")
            return 0
        except BrokenPipeError:
            # The caller closed our stdout: nobody is left to answer.
            return 0


def _cmd_warm(args: argparse.Namespace) -> int:
    from slotify_rank.inference.episode import PipelineConfigs
    from slotify_rank.inference.hub import (
        WARM_COMMAND,
        plan_hub_access,
        snapshot_dir,
        warm_models,
    )

    configs = PipelineConfigs.load(args.config_dir)
    if args.check:
        plan = plan_hub_access(configs)
    else:
        plan = warm_models(configs, log=_stderr_logger())

    missing = set(plan.missing)
    for model in plan.models:
        if not model.is_pinned:
            state = f"NOT PINNED (revision {model.revision!r})"
        elif model in missing:
            state = "missing"
        else:
            state = f"present at {snapshot_dir(model)}"
        print(f"{model.describe()} [{', '.join(model.used_by)}]: {state}")

    if plan.offline:
        print("All pinned snapshots are local; `infer rank` and `infer serve` run offline.")
        return 0
    if plan.unpinned:
        print(plan.unpinned_message(), file=sys.stderr)
    if plan.missing:
        print(f"Run `{WARM_COMMAND}` to fetch the missing snapshot(s).", file=sys.stderr)
    return 1


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--normalizer",
        default=None,
        help="Path to normalizer.json (default: beside the checkpoint).",
    )


def _add_config_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-dir",
        default=None,
        dest="config_dir",
        help="Directory of pipeline YAML configs (default: the repository's "
        "committed ml/configs/, the same files the corpus pipeline reads).",
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "infer", help="Score candidates with a trained ranker (no training)."
    )
    sub = parser.add_subparsers(dest="infer_command", required=True)

    describe = sub.add_parser(
        "describe", help="Print a checkpoint's identity without scoring."
    )
    _add_model_arguments(describe)
    describe.add_argument("--output", default=None, help="Write the identity as JSON.")
    describe.set_defaults(func=_cmd_describe)

    rank = sub.add_parser("rank", help="Rank one audio file's candidate breakpoints.")
    rank.add_argument("--audio", required=True, help="Audio file to analyse.")
    _add_model_arguments(rank)
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
    _add_config_dir_argument(rank)
    rank.add_argument(
        "--no-transcribe",
        action="store_true",
        dest="no_transcribe",
        help="Skip Whisper transcription. Faster, but every candidate is then "
        "scored with its text modality masked, and the result says so.",
    )
    scratch = rank.add_mutually_exclusive_group()
    scratch.add_argument(
        "--work-dir",
        default=None,
        dest="work_dir",
        help="An existing directory the caller owns. The scratch corpus is built "
        "in a fresh subdirectory of it (and every temp file goes there too), so "
        "nothing is left in the system temp directory if this process is killed.",
    )
    scratch.add_argument(
        "--workspace",
        default=None,
        help="Scratch directory for the throwaway single-episode corpus "
        "(default: a new directory in the system temp directory).",
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

    serve = sub.add_parser(
        "serve",
        help="Long-lived ranker: load once, answer JSON-lines requests on stdin.",
        description="Load the checkpoint, its normalizer and every encoder once, "
        "then read one JSON request per line on stdin and write one JSON response "
        "per line on stdout. Progress goes to stderr. Exits 0 on end of input or "
        "SIGTERM. See slotify_rank.inference.worker for the protocol.",
    )
    _add_model_arguments(serve)
    _add_config_dir_argument(serve)
    serve.set_defaults(func=_cmd_serve)

    warm = sub.add_parser(
        "warm",
        help="Download exactly the model snapshots the configs pin.",
        description="Fetch the pinned Whisper and MiniLM snapshots into the local "
        "Hugging Face cache through the same loaders serving uses, so `infer rank` "
        "and `infer serve` can run offline. Exit 1 if anything is still missing.",
    )
    _add_config_dir_argument(warm)
    warm.add_argument(
        "--check",
        action="store_true",
        help="Only report whether every pinned snapshot is local; download nothing.",
    )
    warm.set_defaults(func=_cmd_warm)
