"""Model configuration: one versioned YAML per variant, no hyperparameters in code.

A hyperparameter that lives in a source file is a hyperparameter that cannot be
recorded in a run's ``resolved_config.json``, cannot be diffed between two
experiments, and quietly changes the meaning of every checkpoint that predates
the edit. So the dataclass here holds the defaults, the YAML files hold the
values, and an unknown key is an error rather than a silently ignored line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.pipeline.identity import config_digest

__all__ = ["ModelConfig", "load_model_config", "DEFAULT_MODEL_CONFIG_DIR"]

DEFAULT_MODEL_CONFIG_DIR = "ml/configs/models"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of one ranker variant."""

    variant: str
    #: Width every modality is projected to before fusion. The gated variant
    #: requires these to agree, since it forms a weighted sum of them.
    handcrafted_projection: int = 128
    audio_projection: int = 128
    text_projection: int = 128
    #: Width of the shared post-fusion trunk.
    hidden_dimension: int = 128
    dropout: float = 0.1
    auxiliary_head: bool = True
    #: Feed the handcrafted scalar block's missing-mask alongside its values, so
    #: the model can tell "this pause was 0 ms" from "we could not measure it".
    include_missing_mask: bool = True
    #: Only meaningful for ``audio_only``. False is ``audio_embedding_only``
    #: (the learned speech representation alone); True is
    #: ``audio_plus_acoustic``, which also sees the handcrafted scalars. Two
    #: different experiments, named so a report cannot confuse them.
    include_handcrafted_features: bool = False
    #: ``strict`` rejects a dataset whose dimensions differ from the ones baked
    #: into the checkpoint. There is deliberately no "coerce" policy.
    dimension_policy: str = "strict"
    config_version: str = "model-config-v1"

    def __post_init__(self) -> None:
        if not self.variant:
            raise ValueError("A model config must name a variant")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        for name in (
            "handcrafted_projection",
            "audio_projection",
            "text_projection",
            "hidden_dimension",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.dimension_policy != "strict":
            raise ValueError(
                f"Unknown dimension_policy {self.dimension_policy!r}; only 'strict' "
                "exists, because silently reshaping an input is how a model ends "
                "up trained on mis-columned features"
            )

    @property
    def experiment_name(self) -> str:
        """The variant as a Phase 5 ablation would name it."""
        if self.variant == "audio_only":
            return (
                "audio_plus_acoustic"
                if self.include_handcrafted_features
                else "audio_embedding_only"
            )
        return self.variant

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return config_digest(self.to_dict())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "model config") -> "ModelConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"{where}: unknown setting(s) {unknown}. Known settings: {sorted(known)}"
            )
        return cls(**dict(raw))


def load_model_config(path: Path | str) -> ModelConfig:
    """Read one ``ml/configs/models/<variant>.yaml``."""
    import yaml

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Model config not found: {config_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    payload = dict(loaded)
    model_section = payload.get("model")
    if isinstance(model_section, Mapping):
        payload = dict(model_section)
    return ModelConfig.from_mapping(payload, where=str(config_path))
