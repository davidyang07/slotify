"""The training loop: explicit, CPU-first, resumable.

Structure per epoch: iterate pair batches, score both slots, apply the margin
ranking loss, add the auxiliary acceptability loss over the batch's *unique*
candidates, clip, step. Then score the whole validation split once, group by
episode, and compute NDCG@3. The best checkpoint is the one with the highest
validation NDCG@3; the last checkpoint is written every epoch so an interrupted
run can resume.

Determinism on CPU
------------------

Seeds are set for Python, NumPy and torch; the DataLoader shuffles through an
explicit ``torch.Generator`` whose state is checkpointed. Under these settings a
run is reproducible on the same machine and torch build. What is *not*
guaranteed: bit-identical results across torch versions, across BLAS thread
counts, or against a CUDA device -- floating-point reduction order differs and
no seed controls that. Resume equivalence is therefore asserted within a
tolerance rather than bitwise, and the tolerance is stated in the test.

Mixed precision is supported where the device supports it safely and is never
required. On CPU it is refused outright: float16 has no accelerated CPU kernels
here and would be slower than float32 while looking like an optimization.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from slotify_rank.datasets.collate import make_pair_collate
from slotify_rank.datasets.ranking_dataset import EpisodeGroups, PairDataset
from slotify_rank.embeddings.device import describe_device, resolve_device, resolve_dtype
from slotify_rank.evaluation.metrics import MetricConfig
from slotify_rank.models.base import BaseRanker
from slotify_rank.pipeline.identity import library_versions
from slotify_rank.ranking.losses import class_weight_from_training, combined_loss
from slotify_rank.ranking.metrics import ValidationMetrics, evaluate_validation
from slotify_rank.ranking.pairs import index_pairs
from slotify_rank.training.checkpoint import (
    RngState,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from slotify_rank.training.config import TrainingConfig, resolve_thread_count
from slotify_rank.training.early_stopping import EarlyStopping
from slotify_rank.training.prepare import PreparedDataset

__all__ = [
    "Trainer",
    "TrainingResult",
    "EpochMetrics",
    "set_global_seed",
    "build_seeded_model",
]

_DEPENDENCIES = ("torch", "numpy", "slotify-rank")


def set_global_seed(seed: int) -> None:
    """Seed every generator the training path draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_seeded_model(model_config: Any, schema: Any, seed: int) -> BaseRanker:
    """Seed, *then* construct, so the initial weights are reproducible.

    This is not the same seeding :class:`Trainer` does. The trainer seeds batch
    order and dropout, but by the time it exists the model has already been
    built and its weights already drawn -- from whatever global RNG state the
    process happened to be in. Two "identical" runs in one process would then
    start from different weights and diverge from the first step, which also
    breaks resume equivalence. Model construction must therefore be seeded by
    whoever performs it, which is what this helper is for.
    """
    from slotify_rank.models.registry import build_model

    set_global_seed(seed)
    return build_model(model_config, schema)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_ranking_loss: float
    train_auxiliary_loss: float | None
    train_pairwise_accuracy: float
    train_pair_count: int
    train_episode_count: int
    train_score_margin_mean: float
    train_score_margin_std: float
    validation: Mapping[str, Any]
    seconds: float
    pairs_per_second: float
    is_best: bool
    learning_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_ranking_loss": self.train_ranking_loss,
            "train_auxiliary_loss": self.train_auxiliary_loss,
            "train_pairwise_accuracy": self.train_pairwise_accuracy,
            "train_pair_count": self.train_pair_count,
            "train_contributing_episode_count": self.train_episode_count,
            "train_score_margin_mean": self.train_score_margin_mean,
            "train_score_margin_std": self.train_score_margin_std,
            "validation": dict(self.validation),
            "seconds": self.seconds,
            "pairs_per_second": self.pairs_per_second,
            "is_best": self.is_best,
            "learning_rate": self.learning_rate,
        }


@dataclass
class TrainingResult:
    run_id: str
    epochs_run: int
    best_epoch: int
    best_validation_metric: float | None
    best_metrics: Mapping[str, Any]
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    stop_reason: str = ""
    seconds: float = 0.0
    device: str = "cpu"
    dtype: str = "float32"
    parameter_count: int = 0
    interrupted: bool = False
    best_checkpoint_path: str = ""
    last_checkpoint_path: str = ""


class Trainer:
    """Owns one training run: its model, its data, its checkpoints."""

    def __init__(
        self,
        model: BaseRanker,
        dataset: PreparedDataset,
        config: TrainingConfig,
        run_directory: Path,
        run_id: str,
        model_config: Mapping[str, Any],
        git_commit: str = "unknown",
        label_manifest_hash: str = "",
        on_epoch: Callable[[EpochMetrics], None] | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.run_directory = Path(run_directory)
        self.run_id = run_id
        self.model_config = dict(model_config)
        self.git_commit = git_commit
        self.label_manifest_hash = label_manifest_hash
        self.on_epoch = on_epoch

        self.device_name = resolve_device(config.device)
        self.device = torch.device(self.device_name)
        self.dtype = resolve_dtype(config.dtype, self.device_name)
        self.use_amp = self._resolve_mixed_precision()

        threads = resolve_thread_count(config.num_threads)
        if config.num_threads > 0:
            torch.set_num_threads(config.num_threads)
        self.threads = threads

        set_global_seed(config.seed)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(config.seed)

        # The RNG state `train()` will start from. Captured here rather than
        # relied upon implicitly: dropout draws from the *global* torch
        # generator, so anything that consumes it between construction and
        # train() -- another run in the same process, a stray forward pass --
        # would otherwise change this run's results. `resume_from` replaces it
        # with the checkpointed state, which is what makes a resumed run
        # continue the same stream rather than restart it.
        self._entry_rng_state = torch.get_rng_state()

        self.global_step = 0
        self.start_epoch = 1
        self.epoch_metrics: list[EpochMetrics] = []

        self.positive_weight: torch.Tensor | None = None
        self.class_weight_metadata: dict[str, Any] = {}
        if config.use_class_weights and config.auxiliary_head:
            train_examples = [
                example
                for example in dataset.loaded.examples
                if example.split == config.train_split
            ]
            metadata = class_weight_from_training(train_examples, config.train_split)
            self.class_weight_metadata = metadata
            if not metadata["degenerate"]:
                self.positive_weight = torch.tensor(
                    metadata["positive_weight"], dtype=torch.float32, device=self.device
                )

        self.features = dataset.features
        if self.device_name != "cpu":
            self.features = _move_features(self.features, self.device)

        self.pair_dataset = PairDataset(
            index_pairs(dataset.train_pairs), self.features.index_of
        )
        self.validation_groups = EpisodeGroups(
            self.features, list(dataset.validation_rows)
        )
        self.metric_config = MetricConfig(
            k=config.metric_k, relevance_threshold=config.relevance_threshold
        )

    # -- setup helpers ------------------------------------------------------

    def _resolve_mixed_precision(self) -> bool:
        if not self.config.mixed_precision:
            return False
        if self.device_name == "cpu":
            raise ValueError(
                "mixed_precision was requested but the resolved device is CPU, "
                "where float16 has no accelerated kernels and would be slower "
                "than float32. Mixed precision is supported, never required."
            )
        return True

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.model.parameters()
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        return torch.optim.SGD(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            momentum=0.9,
        )

    def _loader(self) -> DataLoader:
        return DataLoader(
            self.pair_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self.loader_generator,
            num_workers=self.config.num_workers,
            collate_fn=make_pair_collate(self.features),
            drop_last=False,
        )

    # -- environment --------------------------------------------------------

    def environment(self) -> dict[str, Any]:
        described = describe_device(self.device_name)
        described.update(
            {
                "resolved_dtype": self.config.dtype,
                "mixed_precision": self.use_amp,
                "threads": self.threads,
                "num_workers": self.config.num_workers,
                "small_memory_mode": self.config.small_memory_mode,
                "dependency_versions": library_versions(*_DEPENDENCIES),
                "seed": self.config.seed,
            }
        )
        return described

    # -- the loop -----------------------------------------------------------

    def train(self) -> TrainingResult:
        torch.set_rng_state(self._entry_rng_state)
        started = time.perf_counter()
        interrupted = False
        stop_reason = "completed the configured epoch budget"

        best_metrics: dict[str, Any] = {}
        best_checkpoint = self.run_directory / "best_checkpoint.pt"
        last_checkpoint = self.run_directory / "last_checkpoint.pt"

        epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                metrics = self._run_epoch(epoch)
                self.epoch_metrics.append(metrics)
                if self.on_epoch is not None:
                    self.on_epoch(metrics)

                if metrics.is_best:
                    best_metrics = dict(metrics.validation)
                    save_checkpoint(best_checkpoint, self._checkpoint_payload(epoch))
                save_checkpoint(last_checkpoint, self._checkpoint_payload(epoch))

                if self.early_stopping.stopped:
                    stop_reason = self.early_stopping.stop_reason
                    break
        except KeyboardInterrupt:
            # A deliberate interruption is not a crash: the last checkpoint is
            # already on disk, so the run is resumable rather than lost.
            interrupted = True
            stop_reason = "interrupted by the user; resume from last_checkpoint.pt"

        return TrainingResult(
            run_id=self.run_id,
            epochs_run=len(self.epoch_metrics),
            best_epoch=self.early_stopping.best_epoch,
            best_validation_metric=self.early_stopping.best_metric,
            best_metrics=best_metrics,
            epoch_metrics=self.epoch_metrics,
            stop_reason=stop_reason,
            seconds=time.perf_counter() - started,
            device=self.device_name,
            dtype=self.config.dtype,
            parameter_count=self.model.parameter_count(),
            interrupted=interrupted,
            best_checkpoint_path=str(best_checkpoint) if best_metrics else "",
            last_checkpoint_path=str(last_checkpoint),
        )

    def _run_epoch(self, epoch: int) -> EpochMetrics:
        started = time.perf_counter()
        self.model.train()
        loss_config = self.config.loss_config()

        totals = {"total": 0.0, "ranking": 0.0, "auxiliary": 0.0}
        auxiliary_batches = 0
        pair_total = 0
        correct = 0.0
        margins: list[float] = []

        for batch in self._loader():
            self.optimizer.zero_grad(set_to_none=True)

            left = self._score(batch, "left")
            right = self._score(batch, "right")
            unique_logits = None
            unique_targets = None
            if loss_config.auxiliary_enabled:
                unique = self._forward(batch, "unique")
                unique_logits = unique.acceptability_logit
                unique_targets = batch["unique_is_acceptable"]

            components = combined_loss(
                left.ranking_score,
                right.ranking_score,
                batch["target"],
                pair_weights=batch["pair_weight"],
                unique_logits=unique_logits,
                unique_targets=unique_targets,
                config=loss_config,
                positive_weight=self.positive_weight,
            )
            components.total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            self.optimizer.step()
            self.global_step += 1

            batch_pairs = components.pair_count
            pair_total += batch_pairs
            totals["total"] += float(components.total.detach()) * batch_pairs
            totals["ranking"] += float(components.ranking.detach()) * batch_pairs
            if components.auxiliary is not None:
                totals["auxiliary"] += float(components.auxiliary.detach())
                auxiliary_batches += 1

            with torch.no_grad():
                signed = (
                    left.ranking_score - right.ranking_score
                ) * batch["target"]
                correct += float((signed > 0).float().sum()) + 0.5 * float(
                    (signed == 0).float().sum()
                )
                margins.extend(signed.detach().cpu().tolist())

        validation = self._validate()
        metric = validation.ndcg_at_k
        is_best = self.early_stopping.update(metric, epoch)

        seconds = time.perf_counter() - started
        margin_array = np.asarray(margins, dtype=np.float64) if margins else np.zeros(1)

        return EpochMetrics(
            epoch=epoch,
            train_loss=totals["total"] / pair_total if pair_total else float("nan"),
            train_ranking_loss=(
                totals["ranking"] / pair_total if pair_total else float("nan")
            ),
            train_auxiliary_loss=(
                totals["auxiliary"] / auxiliary_batches if auxiliary_batches else None
            ),
            train_pairwise_accuracy=correct / pair_total if pair_total else float("nan"),
            train_pair_count=pair_total,
            train_episode_count=self.dataset.train_pairs.contributing_episode_count,
            train_score_margin_mean=float(margin_array.mean()),
            train_score_margin_std=float(margin_array.std()),
            validation=validation.to_dict(),
            seconds=seconds,
            pairs_per_second=pair_total / seconds if seconds > 0 else 0.0,
            is_best=is_best,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def _forward(self, batch: Mapping[str, torch.Tensor], side: str):
        return self.model(
            handcrafted=batch[f"{side}_handcrafted"],
            handcrafted_missing_mask=batch[f"{side}_handcrafted_missing_mask"],
            audio=batch[f"{side}_audio"],
            audio_available=batch[f"{side}_audio_available"],
            text=batch[f"{side}_text"],
            text_available=batch[f"{side}_text_available"],
        )

    def _score(self, batch: Mapping[str, torch.Tensor], side: str):
        return self._forward(batch, side)

    def _validate(self) -> ValidationMetrics:
        """Score every validation candidate, then rank within each episode."""
        self.model.eval()
        rows = list(self.dataset.validation_rows)
        scores = torch.full((len(self.features),), float("nan"))
        logits = torch.full((len(self.features),), float("nan"))

        with torch.no_grad():
            for start in range(0, len(rows), max(self.config.batch_size, 1)):
                chunk = rows[start : start + max(self.config.batch_size, 1)]
                inputs = self.features.features_at(chunk)
                output = self.model(**inputs)
                scores[chunk] = output.ranking_score.detach().cpu()
                if output.acceptability_logit is not None:
                    logits[chunk] = output.acceptability_logit.detach().cpu()

        has_logits = bool(torch.isfinite(logits[rows]).all()) if rows else False
        return evaluate_validation(
            self.validation_groups,
            scores,
            logits if has_logits else None,
            config=self.metric_config,
        )

    # -- checkpointing ------------------------------------------------------

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            best_validation_metric=self.early_stopping.best_metric,
            best_epoch=self.early_stopping.best_epoch,
            training_config=self.config.to_dict(),
            training_config_hash=self.config.digest,
            model_config=self.model_config,
            model_variant=self.model.variant,
            schema=self.dataset.schema,
            normalizer=self.dataset.normalizer,
            dataset_split_hash=self.dataset.normalizer.training_split_hash,
            label_manifest_hash=self.label_manifest_hash,
            random_seed=self.config.seed,
            resolved_device=self.device_name,
            resolved_dtype=self.config.dtype,
            dependency_versions=library_versions(*_DEPENDENCIES),
            git_commit=self.git_commit,
            run_id=self.run_id,
            early_stopping_state=self.early_stopping.state_dict(),
            rng_state=RngState(
                torch_state=torch.get_rng_state(),
                loader_state=self.loader_generator.get_state(),
            ),
        )

    def resume_from(self, path: Path) -> dict[str, Any]:
        """Restore model, optimizer, epoch, step, early stopping and RNG state.

        Compatibility is the caller's job (see
        :func:`slotify_rank.training.checkpoint.assert_compatible`); by the time
        we are here the checkpoint has been accepted.
        """
        payload = load_checkpoint(Path(path))
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.start_epoch = int(payload.get("epoch", 0)) + 1

        state = payload.get("early_stopping_state")
        if isinstance(state, Mapping):
            self.early_stopping.load_state_dict(state)

        rng = payload.get("rng_state")
        if isinstance(rng, Mapping):
            # Restoring both generators is what makes a resumed run draw the
            # same batches as an uninterrupted one from this point on.
            if "torch" in rng:
                self._entry_rng_state = rng["torch"].to(torch.uint8)
                torch.set_rng_state(self._entry_rng_state)
            if "loader" in rng:
                self.loader_generator.set_state(rng["loader"].to(torch.uint8))
        return payload


def _move_features(features, device: torch.device):
    """Move the materialized block to a non-CPU device once, up front."""
    from dataclasses import replace as dataclass_replace

    moved = {}
    for name in (
        "handcrafted",
        "handcrafted_missing_mask",
        "audio",
        "audio_available",
        "text",
        "text_available",
        "quality_score",
        "is_acceptable",
    ):
        moved[name] = getattr(features, name).to(device)
    return dataclass_replace(features, **moved)
