"""``training`` and ``models`` commands.

Conventions match Phases 1-3: machine-readable JSON to files, a human summary on
stdout, exit 0 on success, 1 on a handled runtime failure, 2 on a usage error.

``--smoke`` is a shorthand for a set of overrides, never a hidden mode. It sets
a small epoch budget and small data caps, and every value it changed is written
to the run's ``resolved_config.json`` under ``overrides`` and reproduced in the
Markdown summary. A run's numbers can therefore always be traced to the flags
that produced them.

Synthetic fixtures are opt-in and self-declaring: ``training synthesize`` writes
them, and any run against them is detected from the label export's metadata and
labelled ``synthetic`` throughout its reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slotify_rank.data import manifests
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.splits import read_split_manifest
from slotify_rank.training.reporting import (
    RunArtifacts,
    append_epoch_metrics,
    build_summary,
    write_json,
    write_pairs,
    write_training_summary,
)

# Everything that pulls in torch is imported inside the command functions, not
# here: `build_parser` runs on every CLI invocation, so a module-level torch
# import would make `slotify-rank version` require the heavy `features` extra.

__all__ = ["register"]

#: What --smoke actually does. Named here rather than scattered through the
#: command so it can be printed, recorded and reviewed.
SMOKE_OVERRIDES: dict[str, Any] = {
    "epochs": 3,
    "max_episodes": 4,
    "max_candidates": 24,
    "max_pairs": 96,
    "early_stopping_patience": 2,
}


def _paths(args: argparse.Namespace) -> DataPaths:
    paths = DataPaths(data_root=Path(args.data_root) if args.data_root else None)
    paths.mkdirs()
    return paths


def _label_export(args: argparse.Namespace, paths: DataPaths) -> Path:
    if args.labels:
        return Path(args.labels)
    default = paths.labels_dir / "labels_v1.jsonl"
    if not default.is_file():
        raise FileNotFoundError(
            f"No label export at {default}. Run `slotify-rank label export` first, "
            "or pass --labels, or use `training synthesize` for a synthetic "
            "fixture corpus."
        )
    return default


def _is_synthetic(label_export: Path) -> bool:
    """Synthetic provenance is read from the artifact, never assumed."""
    metadata_path = label_export.with_suffix(label_export.suffix + ".meta.json")
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(metadata.get("synthetic", False))


def _split_lookup(paths: DataPaths, version: str) -> dict[str, str]:
    path = paths.split_manifest(version)
    if not path.is_file():
        return {}
    return dict(read_split_manifest(path).by_episode)


def _dataset_candidates(paths: DataPaths):
    if not paths.candidates_manifest.is_file():
        return None
    return manifests.read_candidates(paths.candidates_manifest)


def _resolve_config(args: argparse.Namespace) -> tuple[Any, dict[str, Any], bool]:
    from slotify_rank.training.config import load_training_config

    config = load_training_config(args.training_config)
    overrides: dict[str, Any] = {}
    smoke = bool(getattr(args, "smoke", False))
    if smoke:
        config, applied = config.with_overrides(**SMOKE_OVERRIDES)
        overrides.update(applied)
    explicit = {
        "epochs": getattr(args, "epochs", None),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", None),
        "batch_size": getattr(args, "batch_size", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "max_episodes": getattr(args, "max_episodes", None),
        "max_candidates": getattr(args, "max_candidates", None),
        "max_pairs": getattr(args, "max_pairs", None),
        "num_threads": getattr(args, "num_threads", None),
    }
    config, applied = config.with_overrides(**explicit)
    overrides.update(applied)
    return config, overrides, smoke


def _prepare(args: argparse.Namespace, config: Any):
    from slotify_rank.training.prepare import prepare_dataset

    paths = _paths(args)
    label_export = _label_export(args, paths)
    dataset = prepare_dataset(
        paths=paths,
        config=config,
        label_export=label_export,
        candidates=_dataset_candidates(paths),
        split_lookup=_split_lookup(paths, getattr(args, "split_version", "v1")),
    )
    return paths, label_export, dataset


def _model_config(args: argparse.Namespace, config: Any):
    from slotify_rank.models.schema import ModelConfig, load_model_config

    if getattr(args, "model_config", None):
        model_config = load_model_config(args.model_config)
    elif getattr(args, "model", None):
        model_config = ModelConfig(variant=args.model)
    else:
        raise ValueError("Pass --model VARIANT or --model-config PATH")
    if model_config.auxiliary_head != config.auxiliary_head:
        # Two configs disagreeing about the auxiliary head silently trains a
        # head that is never optimized, or optimizes one that does not exist.
        raise ValueError(
            f"The model config says auxiliary_head={model_config.auxiliary_head} "
            f"but the training config says {config.auxiliary_head}. Make them agree."
        )
    return model_config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_synthesize(args: argparse.Namespace) -> int:
    from slotify_rank.datasets.synthetic import (
        SyntheticCorpusConfig,
        build_synthetic_corpus,
    )

    paths = _paths(args)
    corpus = build_synthetic_corpus(
        paths,
        SyntheticCorpusConfig(
            episode_count=args.episodes,
            candidates_per_episode=args.candidates_per_episode,
            seed=args.seed,
            text_missing_episode_stride=args.text_missing_stride,
        ),
    )
    print("Wrote a SYNTHETIC fixture corpus. These are generated numbers, not")
    print("real audio and not human labels. Any metric measured on them is")
    print("evidence that the training system runs, never evidence of quality.")
    print(f"  episodes:   {len(corpus.episode_ids)}")
    print(f"  candidates: {corpus.candidate_count}")
    print(f"  features:   {corpus.features_manifest}")
    print(f"  labels:     {corpus.label_export}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    config, overrides, _ = _resolve_config(args)
    _, label_export, dataset = _prepare(args, config)
    summary = dataset.summary()
    summary["synthetic_data"] = _is_synthetic(label_export)
    summary["overrides"] = overrides

    if args.output:
        write_json(Path(args.output), summary)

    print(f"Eligible candidates: {summary['eligibility']['candidate_count']}")
    for reason, count in sorted(summary["eligibility"]["exclusion_counts"].items()):
        if reason != "eligible":
            print(f"  excluded [{reason}]: {count}")
    print(
        f"Train: {summary['training_candidate_count']} candidate(s) across "
        f"{summary['training_episode_count']} episode(s)"
    )
    print(
        f"Validation: {summary['validation_candidate_count']} candidate(s) across "
        f"{summary['validation_episode_count']} episode(s)"
    )
    print(f"Training pairs: {summary['training_pair_count']}")
    if summary["episode_overlap_between_splits"]:
        print("ERROR: episodes appear in both splits", file=sys.stderr)
        return 1
    if summary["synthetic_data"]:
        print("NOTE: this dataset is SYNTHETIC.")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def _cmd_pairs(args: argparse.Namespace) -> int:
    config, _, _ = _resolve_config(args)
    _, _, dataset = _prepare(args, config)

    payload = {
        "train": dataset.train_pairs.summary(),
        "validation": dataset.validation_pairs.summary(),
    }
    if args.output:
        write_json(Path(args.output), payload)
    if args.pairs_output:
        write_pairs(Path(args.pairs_output), dataset.train_pairs.to_rows())

    for name in ("train", "validation"):
        stats = payload[name]
        print(f"{name}: {stats['pair_count']} pair(s) from "
              f"{stats['contributing_episode_count']} episode(s)")
        print(f"  per episode: min={stats['pairs_per_episode_min']} "
              f"max={stats['pairs_per_episode_max']} "
              f"mean={stats['pairs_per_episode_mean']:.1f}")
        print(f"  by score difference: {stats['score_difference_histogram']}")
        if stats["episodes_without_pairs"]:
            print(f"  {stats['episodes_without_pairs_count']} episode(s) produced "
                  f"no usable pair: {stats['episodes_without_pairs'][:5]}")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def _run_training(args: argparse.Namespace, resume_from: Path | None) -> int:
    from slotify_rank.datasets.normalizer import write_normalizer
    from slotify_rank.training.checkpoint import assert_compatible, load_checkpoint
    from slotify_rank.training.config import ResolvedConfig, compute_run_id, git_commit
    from slotify_rank.training.trainer import Trainer, build_seeded_model

    config, overrides, smoke = _resolve_config(args)
    paths, label_export, dataset = _prepare(args, config)
    model_config = _model_config(args, config)

    synthetic = _is_synthetic(label_export)
    commit = git_commit(paths.repo_root)
    run_id = compute_run_id(config, model_config, dataset.fingerprint, commit)

    # Seeded before construction so the initial weights are reproducible; see
    # build_seeded_model.
    model = build_seeded_model(model_config, dataset.schema, config.seed)
    run_directory = Path(args.run_dir) if args.run_dir else (
        paths.data_root.parent / config.report_dir / run_id
    )
    artifacts = RunArtifacts(run_directory)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        run_directory=artifacts.directory,
        run_id=run_id,
        model_config=model_config.to_dict(),
        git_commit=commit,
        label_manifest_hash=dataset.fingerprint,
        on_epoch=lambda metrics: _report_epoch(artifacts, metrics),
    )

    if resume_from is not None:
        payload = load_checkpoint(resume_from)
        assert_compatible(
            payload,
            model_variant=model_config.variant,
            schema=dataset.schema,
            normalizer=dataset.normalizer,
            dataset_split_hash=dataset.normalizer.training_split_hash,
        )
        trainer.resume_from(resume_from)
        print(
            f"Resuming {run_id} from epoch {trainer.start_epoch} "
            f"(best so far: {trainer.early_stopping.best_metric})"
        )
        if trainer.start_epoch > config.epochs:
            print(
                f"Nothing to do: the checkpoint is already at epoch "
                f"{trainer.start_epoch - 1} of {config.epochs}."
            )
            return 0

    resolved = ResolvedConfig(
        training=config,
        model=model_config,
        run_id=run_id,
        training_config_hash=config.digest,
        model_config_hash=model_config.digest,
        dataset_fingerprint=dataset.fingerprint,
        git_commit=commit,
        overrides=overrides,
        smoke=smoke,
        synthetic_data=synthetic,
    )
    write_json(artifacts.resolved_config, resolved.to_dict())
    write_json(artifacts.environment, trainer.environment())
    dataset_summary = dataset.summary()
    dataset_summary["synthetic_data"] = synthetic
    dataset_summary["class_weights"] = trainer.class_weight_metadata
    write_json(artifacts.dataset_summary, dataset_summary)
    write_normalizer(artifacts.normalizer, dataset.normalizer)
    write_pairs(artifacts.pairs, dataset.train_pairs.to_rows())

    if synthetic:
        print("*** SYNTHETIC SMOKE TRAINING - not a model-quality measurement ***")
    print(
        f"Run {run_id}: {model_config.experiment_name}, "
        f"{model.parameter_count():,} parameters on {trainer.device_name}"
    )

    result = trainer.train()

    summary = build_summary(
        run_id=run_id,
        result=result,
        dataset_summary=dataset_summary,
        model_description=model.describe(),
        environment=trainer.environment(),
        artifacts=artifacts,
        synthetic=synthetic,
        smoke=smoke,
        overrides=overrides,
        label_source="synthetic" if synthetic else "human",
    )
    write_training_summary(artifacts, summary)

    print(f"Epochs run: {result.epochs_run}  (stop reason: {result.stop_reason})")
    best = result.best_validation_metric
    print(
        f"Best validation NDCG@{config.metric_k}: "
        f"{'undefined' if best is None else f'{best:.4f}'} at epoch {result.best_epoch}"
    )
    print(f"Runtime: {result.seconds:.1f}s  Artifacts: {artifacts.directory}")
    if result.interrupted:
        print("Interrupted; resume with `training resume`.", file=sys.stderr)
        return 1
    if best is None:
        print(
            "No epoch produced a defined validation NDCG, so no best checkpoint "
            "was selected.",
            file=sys.stderr,
        )
        return 1
    return 0


def _report_epoch(artifacts: RunArtifacts, metrics) -> None:
    payload = metrics.to_dict()
    append_epoch_metrics(artifacts.epoch_metrics, payload)
    validation = payload["validation"]
    ndcg = validation.get(f"ndcg_at_{validation.get('k', 3)}")
    print(
        f"  epoch {payload['epoch']:>3}  loss={payload['train_loss']:.4f}  "
        f"pair_acc={payload['train_pairwise_accuracy']:.3f}  "
        f"val_ndcg={'n/a' if ndcg is None else f'{ndcg:.4f}'}  "
        f"({payload['seconds']:.1f}s, {payload['pairs_per_second']:.0f} pairs/s)"
        f"{'  *best*' if payload['is_best'] else ''}"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    return _run_training(args, resume_from=None)


def _cmd_resume(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return _run_training(args, resume_from=checkpoint)


def _cmd_validate(args: argparse.Namespace) -> int:
    import torch

    from slotify_rank.evaluation.metrics import MetricConfig
    from slotify_rank.models.registry import build_model
    from slotify_rank.models.schema import ModelConfig
    from slotify_rank.ranking.metrics import evaluate_validation
    from slotify_rank.training.checkpoint import assert_compatible, load_checkpoint

    config, _, _ = _resolve_config(args)
    _, _, dataset = _prepare(args, config)
    payload = load_checkpoint(Path(args.checkpoint))

    model_config = ModelConfig.from_mapping(payload["model_config"])
    assert_compatible(
        payload,
        model_variant=model_config.variant,
        schema=dataset.schema,
        normalizer=dataset.normalizer,
        dataset_split_hash=dataset.normalizer.training_split_hash,
        allow_split_mismatch=bool(args.allow_split_mismatch),
    )

    model = build_model(model_config, dataset.schema)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rows = list(dataset.validation_rows)
    scores = torch.full((len(dataset.features),), float("nan"))
    logits = torch.full((len(dataset.features),), float("nan"))
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            chunk = rows[start : start + 64]
            output = model(**dataset.features.features_at(chunk))
            scores[chunk] = output.ranking_score
            if output.acceptability_logit is not None:
                logits[chunk] = output.acceptability_logit

    metrics = evaluate_validation(
        dataset.validation_groups,
        scores,
        logits if bool(torch.isfinite(logits[rows]).all()) else None,
        config=MetricConfig(
            k=config.metric_k, relevance_threshold=config.relevance_threshold
        ),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "model_variant": model_config.variant,
        "run_id": payload.get("run_id"),
        "epoch": payload.get("epoch"),
        **metrics.to_dict(),
    }
    if args.output:
        write_json(Path(args.output), report)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation episodes: {metrics.episodes_scored} scored, "
          f"{metrics.episodes_excluded} excluded from NDCG")
    for key, value in metrics.to_dict().items():
        if isinstance(value, float) or value is None:
            rendered = "undefined" if value is None else f"{value:.4f}"
            print(f"  {key}: {rendered}")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from slotify_rank.training.checkpoint import inspect_checkpoint

    report = inspect_checkpoint(Path(args.checkpoint))
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    from slotify_rank.models.registry import list_models

    entries = list_models()
    if args.output:
        write_json(Path(args.output), entries)
    for entry in entries:
        print(f"{entry['variant']:<14} {entry['description']}")
    return 0


def _cmd_models_describe(args: argparse.Namespace) -> int:
    from slotify_rank.models.registry import describe_variant
    from slotify_rank.models.schema import ModelConfig, load_model_config
    from slotify_rank.datasets.schema import (
        NATIVE_EMBEDDING_DIMENSION,
        DatasetSchema,
        audio_dimension,
        text_dimension,
    )

    if args.model_config:
        model_config = load_model_config(args.model_config)
    else:
        model_config = ModelConfig(variant=args.model)

    native = args.native_dimension
    schema = DatasetSchema(
        handcrafted_feature_names=tuple(
            f"feature_{index}" for index in range(args.handcrafted_dimension)
        ),
        handcrafted_dimension=args.handcrafted_dimension,
        audio_dimension=audio_dimension(native),
        text_dimension=text_dimension(native),
        native_embedding_dimension=native,
    )
    report = describe_variant(model_config.variant, model_config, schema)
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if native != NATIVE_EMBEDDING_DIMENSION:
        print(
            f"(described against a {native}-dimensional embedding, not the "
            f"default {NATIVE_EMBEDDING_DIMENSION})"
        )
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", help="Override the data root.")
    parser.add_argument("--labels", help="Label export JSONL (default: labels_v1).")
    parser.add_argument("--training-config", help="Training config YAML.")
    parser.add_argument("--split-version", default="v1", help="Split manifest version.")
    parser.add_argument("--output", help="Write the machine-readable report here.")


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, help="Override the epoch budget.")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    parser.add_argument("--device", help="auto | cpu | cuda[:N].")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--max-episodes", type=int, dest="max_episodes")
    parser.add_argument("--max-candidates", type=int, dest="max_candidates")
    parser.add_argument("--max-pairs", type=int, dest="max_pairs")
    parser.add_argument("--num-threads", type=int, dest="num_threads")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Shorthand for a small run: "
            + ", ".join(f"{k}={v}" for k, v in SMOKE_OVERRIDES.items())
            + ". Every override is recorded in resolved_config.json."
        ),
    )


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Variant name (e.g. gated).")
    parser.add_argument("--model-config", dest="model_config", help="Model config YAML.")
    parser.add_argument("--run-dir", dest="run_dir", help="Override the run directory.")


def register(subparsers: argparse._SubParsersAction) -> None:
    training = subparsers.add_parser(
        "training", help="Phase 4: prepare, pair, train, resume, validate, inspect."
    )
    training_sub = training.add_subparsers(dest="training_command", required=True)

    synth = training_sub.add_parser(
        "synthesize",
        help="Write a SYNTHETIC fixture corpus (generated numbers, not real data).",
    )
    synth.add_argument("--data-root")
    synth.add_argument("--episodes", type=int, default=10)
    synth.add_argument(
        "--candidates-per-episode", type=int, default=8, dest="candidates_per_episode"
    )
    synth.add_argument("--seed", type=int, default=20260722)
    synth.add_argument(
        "--text-missing-stride",
        type=int,
        default=0,
        dest="text_missing_stride",
        help="Withhold text from every Nth episode (0 keeps all).",
    )
    synth.set_defaults(func=_cmd_synthesize)

    prepare = training_sub.add_parser(
        "prepare", help="Load features and labels; report eligibility and splits."
    )
    _add_common(prepare)
    _add_overrides(prepare)
    prepare.set_defaults(func=_cmd_prepare)

    pairs = training_sub.add_parser(
        "pairs", help="Generate within-episode pairs and report their distribution."
    )
    _add_common(pairs)
    _add_overrides(pairs)
    pairs.add_argument("--pairs-output", dest="pairs_output", help="Write pairs JSONL.")
    pairs.set_defaults(func=_cmd_pairs)

    run = training_sub.add_parser("run", help="Train a model.")
    _add_common(run)
    _add_overrides(run)
    _add_model(run)
    run.set_defaults(func=_cmd_run)

    resume = training_sub.add_parser(
        "resume", help="Continue an interrupted run from a checkpoint."
    )
    _add_common(resume)
    _add_overrides(resume)
    _add_model(resume)
    resume.add_argument("--checkpoint", required=True)
    resume.set_defaults(func=_cmd_resume)

    validate = training_sub.add_parser(
        "validate", help="Score a saved checkpoint on the validation split."
    )
    _add_common(validate)
    _add_overrides(validate)
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        dest="allow_split_mismatch",
        help="Inference-only: score a corpus the model was not trained against.",
    )
    validate.set_defaults(func=_cmd_validate)

    inspect = training_sub.add_parser("inspect", help="Print a checkpoint's metadata.")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--output")
    inspect.set_defaults(func=_cmd_inspect)

    models = subparsers.add_parser("models", help="Inspect the model registry.")
    models_sub = models.add_subparsers(dest="models_command", required=True)

    listing = models_sub.add_parser("list", help="List every registered variant.")
    listing.add_argument("--output")
    listing.set_defaults(func=_cmd_models_list)

    describe = models_sub.add_parser(
        "describe", help="Build a variant and report its real shape and size."
    )
    describe.add_argument("--model", help="Variant name.")
    describe.add_argument("--model-config", dest="model_config")
    describe.add_argument(
        "--handcrafted-dimension", type=int, default=110, dest="handcrafted_dimension"
    )
    describe.add_argument(
        "--native-dimension", type=int, default=384, dest="native_dimension"
    )
    describe.add_argument("--output")
    describe.set_defaults(func=_cmd_models_describe)
