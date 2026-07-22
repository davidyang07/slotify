"""Training configuration and the deterministic run identity.

Configuration lives in ``ml/configs/training_v1.yaml``; overrides arrive from the
CLI and are *recorded*, never applied silently. That is what makes ``--smoke``
safe: it changes epochs and data limits, and every value it changed appears in
the run's ``resolved_config.json`` alongside a list naming the overrides. A run
whose numbers look surprising can always be traced to the flags that produced
them.

Run identity is content-addressed rather than timestamped. Two runs over the
same data with the same configuration and the same code are the same logical
experiment and get the same ``run_id``; a timestamp would make them look like
two results and invite averaging them. Attempt directories keep the artifacts of
a rerun separate without pretending it is a different experiment.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.datasets.loader import EligibilityConfig
from slotify_rank.pipeline.identity import config_digest
from slotify_rank.ranking.losses import LossConfig
from slotify_rank.ranking.pairs import PairConfig

__all__ = [
    "TrainingConfig",
    "ResolvedConfig",
    "load_training_config",
    "compute_run_id",
    "git_commit",
]


@dataclass(frozen=True)
class TrainingConfig:
    """Everything about *how* training runs, as opposed to what it trains."""

    # -- optimization ------------------------------------------------------
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 30
    gradient_clip_norm: float = 1.0
    #: 0 disables early stopping. Patience counts epochs without an improvement
    #: in validation NDCG@3.
    early_stopping_patience: int = 8
    #: Minimum improvement that counts. Without it, floating-point noise in the
    #: fourth decimal resets the patience counter forever.
    early_stopping_min_delta: float = 1e-4

    # -- objective ---------------------------------------------------------
    margin: float = 0.2
    auxiliary_weight: float = 0.25
    auxiliary_head: bool = True
    use_class_weights: bool = False

    # -- pairs -------------------------------------------------------------
    minimum_score_difference: float = 1.0
    max_pairs_per_episode: int | None = 64
    pair_sampling_strategy: str = "balanced"
    pair_weight_scheme: str = "uniform"
    hard_pair_emphasis: float = 0.0
    shuffle_pair_direction: bool = True

    # -- data --------------------------------------------------------------
    require_complete_multimodal: bool = False
    required_feature_pipeline_version: str = ""
    required_feature_spec_version: str = ""
    train_split: str = "train"
    validation_split: str = "validation"
    #: Caps for smoke runs. None means "no limit".
    max_episodes: int | None = None
    max_candidates: int | None = None
    max_pairs: int | None = None

    # -- runtime -----------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    num_workers: int = 0
    #: Cap torch's intra-op threads. 0 leaves torch's own default in place.
    num_threads: int = 0
    #: Trades throughput for peak memory: materialize in float32 but avoid
    #: keeping a second copy of the feature block during batching.
    small_memory_mode: bool = False
    #: Mixed precision is *offered*, never required, and is refused on devices
    #: where it is not safe (see training.trainer).
    mixed_precision: bool = False

    # -- artifacts ---------------------------------------------------------
    checkpoint_dir: str = "artifacts/training"
    report_dir: str = "artifacts/training"
    metric_k: int = 3
    relevance_threshold: float = 4.0

    config_version: str = "training-config-v1"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive; disable clipping by setting "
                "it high rather than to zero, so the configured value always means "
                "the same thing"
            )
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ValueError(f"Unsupported optimizer {self.optimizer!r}")
        if self.train_split == self.validation_split:
            raise ValueError(
                f"train_split and validation_split are both {self.train_split!r}; "
                "validating on the training split reports a number that means "
                "nothing"
            )

    # -- derived sub-configs ------------------------------------------------
    def loss_config(self) -> LossConfig:
        return LossConfig(
            margin=self.margin,
            auxiliary_weight=self.auxiliary_weight,
            auxiliary_enabled=self.auxiliary_head,
            use_class_weights=self.use_class_weights,
        )

    def pair_config(self) -> PairConfig:
        return PairConfig(
            minimum_score_difference=self.minimum_score_difference,
            max_pairs_per_episode=self.max_pairs_per_episode,
            sampling_strategy=self.pair_sampling_strategy,  # type: ignore[arg-type]
            weight_scheme=self.pair_weight_scheme,  # type: ignore[arg-type]
            hard_pair_emphasis=self.hard_pair_emphasis,
            shuffle_direction=self.shuffle_pair_direction,
            seed=self.seed,
        )

    def eligibility_config(self) -> EligibilityConfig:
        return EligibilityConfig(
            required_feature_pipeline_version=self.required_feature_pipeline_version,
            required_feature_spec_version=self.required_feature_spec_version,
            require_complete_multimodal=self.require_complete_multimodal,
            require_labels=True,
            require_acceptability_label=self.auxiliary_head,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return config_digest(self.to_dict())

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], where: str = "training config"
    ) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"{where}: unknown setting(s) {unknown}. Known settings: {sorted(known)}"
            )
        return cls(**dict(raw))

    def with_overrides(self, **overrides: Any) -> tuple["TrainingConfig", dict[str, Any]]:
        """Apply CLI overrides, returning the new config and what changed.

        The second return value is the audit trail. A flag that changes a run's
        behaviour without appearing in its report is a flag that makes the
        report wrong.
        """
        applied: dict[str, Any] = {}
        changes: dict[str, Any] = {}
        for name, value in overrides.items():
            if value is None:
                continue
            if name not in self.__dataclass_fields__:
                raise ValueError(f"Unknown training override {name!r}")
            current = getattr(self, name)
            if current != value:
                applied[name] = {"from": current, "to": value}
            changes[name] = value
        return replace(self, **changes), applied


def load_training_config(path: Path | str | None) -> TrainingConfig:
    """Load ``ml/configs/training_v1.yaml``; ``None`` returns the defaults."""
    if path is None:
        return TrainingConfig()
    import yaml

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Training config not found: {config_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    payload = dict(loaded)
    section = payload.get("training")
    if isinstance(section, Mapping):
        payload = dict(section)
    return TrainingConfig.from_mapping(payload, where=str(config_path))


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit, or ``"unknown"`` outside a checkout.

    Recorded in every checkpoint so a saved model can be traced to the code that
    produced it. A missing git is not an error -- training must work from a
    source tarball -- but the absence is recorded rather than faked.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class ResolvedConfig:
    """Everything a run was actually configured with, plus its identity."""

    training: TrainingConfig
    model: Any
    run_id: str
    training_config_hash: str
    model_config_hash: str
    dataset_fingerprint: str
    git_commit: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    smoke: bool = False
    label_source: str = "human"
    synthetic_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "training_config_hash": self.training_config_hash,
            "model_config_hash": self.model_config_hash,
            "dataset_fingerprint": self.dataset_fingerprint,
            "git_commit": self.git_commit,
            "overrides": dict(self.overrides),
            "smoke": self.smoke,
            "label_source": self.label_source,
            "synthetic_data": self.synthetic_data,
            "training": self.training.to_dict(),
            "model": self.model.to_dict() if hasattr(self.model, "to_dict") else self.model,
        }


def compute_run_id(
    training: TrainingConfig,
    model_config: Any,
    dataset_fingerprint: str,
    commit: str,
) -> str:
    """Content-addressed run id over config, data and code.

    Not a timestamp: a rerun with identical inputs is the same logical
    experiment and should be recognizable as one.
    """
    payload = {
        "training": training.to_dict(),
        "model": model_config.to_dict() if hasattr(model_config, "to_dict") else model_config,
        "dataset": dataset_fingerprint,
        "commit": commit,
    }
    variant = getattr(model_config, "variant", "model")
    return f"{variant}-{config_digest(payload)[:16]}"


def resolve_thread_count(configured: int) -> int:
    """Torch intra-op threads. 0 keeps torch's own default."""
    if configured > 0:
        return configured
    try:
        import torch

        return int(torch.get_num_threads())
    except ImportError:  # pragma: no cover - torch is a training dependency
        return os.cpu_count() or 1
