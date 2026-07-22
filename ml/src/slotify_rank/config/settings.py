"""Thin loader for the canonical, language-neutral heuristic configuration.

``config/heuristic_offline_v1.json`` at the repository root is the single source
of truth for the baseline constants. It is also read by
``backend/src/lib/heuristic-config.ts`` and
``backend/ad_inserter/heuristic_config.py``, so no constant is ever duplicated
in code. This module validates the file aggressively: a missing key is a hard
failure, because a silently defaulted constant would invalidate every experiment
that compares against the baseline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "HeuristicProfile",
    "HeuristicConfig",
    "find_repo_root",
    "canonical_config_path",
    "load_heuristic_config",
    "load_run_config",
]

_CONFIG_RELATIVE_PATH = Path("config") / "heuristic_offline_v1.json"
_REPO_ROOT_ENV_VAR = "SLOTIFY_REPO_ROOT"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up for the canonical config file.

    ``SLOTIFY_REPO_ROOT`` overrides the search, which is what lets the package
    be installed and run from anywhere (including an editable install invoked
    from another working directory).
    """
    override = os.environ.get(_REPO_ROOT_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / _CONFIG_RELATIVE_PATH).is_file():
            raise FileNotFoundError(
                f"{_REPO_ROOT_ENV_VAR}={root} does not contain {_CONFIG_RELATIVE_PATH}"
            )
        return root

    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / _CONFIG_RELATIVE_PATH).is_file():
            return parent
    raise FileNotFoundError(
        f"Could not locate {_CONFIG_RELATIVE_PATH} in any parent of {current}. "
        f"Set {_REPO_ROOT_ENV_VAR} to the repository root."
    )


def canonical_config_path(start: Path | None = None) -> Path:
    """Absolute path of the canonical heuristic configuration file."""
    return find_repo_root(start) / _CONFIG_RELATIVE_PATH


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return mapping[key]


@dataclass(frozen=True)
class HeuristicProfile:
    """One named profile from the canonical configuration file."""

    name: str
    description: str
    is_canonical_baseline: bool
    silence_detection: Mapping[str, Any]
    merge: Mapping[str, Any]
    scoring: Mapping[str, Any]
    selection: Mapping[str, Any]
    route_fallback_candidates: Mapping[str, Any]
    slot_finalisation: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> "HeuristicProfile":
        where = f"profile '{name}'"
        return cls(
            name=name,
            description=str(_require(raw, "description", where)),
            is_canonical_baseline=bool(_require(raw, "is_canonical_baseline", where)),
            silence_detection=dict(_require(raw, "silence_detection", where)),
            merge=dict(_require(raw, "merge", where)),
            scoring=dict(_require(raw, "scoring", where)),
            selection=dict(_require(raw, "selection", where)),
            route_fallback_candidates=dict(
                _require(raw, "route_fallback_candidates", where)
            ),
            slot_finalisation=dict(_require(raw, "slot_finalisation", where)),
        )


@dataclass(frozen=True)
class HeuristicConfig:
    """The whole canonical configuration file."""

    path: Path
    config_version: str
    canonical_profile_name: str
    profiles: Mapping[str, HeuristicProfile]

    @property
    def canonical(self) -> HeuristicProfile:
        return self.profiles[self.canonical_profile_name]

    def profile(self, name: str | None = None) -> HeuristicProfile:
        if name is None:
            return self.canonical
        if name not in self.profiles:
            available = ", ".join(sorted(self.profiles))
            raise KeyError(f"Unknown heuristic profile '{name}'. Available: {available}")
        return self.profiles[name]


# Keys required on the canonical profile. Non-canonical profiles (e.g.
# `legacy_cli_v1`) are recorded for documentation and are not fully structured.
_REQUIRED_SCORING_KEYS = (
    "base_score",
    "pause_reward_max",
    "pause_reward_saturation_ms",
    "song_mode_bonus",
    "sentence_end_reward",
    "no_sentence_end_penalty",
    "unavailable_snippet_sentinel",
    "mid_episode_reward",
    "mid_episode_min_ratio",
    "mid_episode_max_ratio",
    "edge_penalty",
    "edge_guard_seconds",
    "score_min",
    "score_max",
)
_REQUIRED_SELECTION_KEYS = (
    "min_separation_seconds",
    "requested_count",
    "max_returned",
    "ratio_fallback_positions",
    "ratio_fallback_score",
    "spacing_fallback_score",
)
_REQUIRED_FINALISATION_KEYS = (
    "end_guard_ms",
    "confidence_base",
    "confidence_scale",
    "confidence_min",
    "confidence_max",
    "time_seconds_decimals",
)


def load_heuristic_config(path: Path | None = None) -> HeuristicConfig:
    """Load and validate the canonical heuristic configuration."""
    config_path = Path(path) if path is not None else canonical_config_path()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Canonical heuristic config not found at {config_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{config_path} is not valid JSON: {error}") from error

    where = str(config_path)
    config_version = str(_require(raw, "config_version", where))
    canonical_name = str(_require(raw, "canonical_profile", where))
    raw_profiles = _require(raw, "profiles", where)
    if canonical_name not in raw_profiles:
        raise KeyError(
            f"{where} declares canonical_profile '{canonical_name}' "
            "but no such profile exists"
        )

    profiles: dict[str, HeuristicProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if name != canonical_name and not raw_profile.get("is_canonical_baseline"):
            # Non-canonical profiles are documentation-only; keep them addressable
            # without forcing them to carry the full canonical structure.
            continue
        profiles[name] = HeuristicProfile.from_mapping(name, raw_profile)

    canonical = profiles[canonical_name]
    if not canonical.is_canonical_baseline:
        raise ValueError(
            f"Profile '{canonical_name}' is declared canonical but has "
            "is_canonical_baseline=false"
        )
    for key in _REQUIRED_SCORING_KEYS:
        _require(canonical.scoring, key, f"{where} -> {canonical_name}.scoring")
    for key in _REQUIRED_SELECTION_KEYS:
        _require(canonical.selection, key, f"{where} -> {canonical_name}.selection")
    for key in _REQUIRED_FINALISATION_KEYS:
        _require(
            canonical.slot_finalisation,
            key,
            f"{where} -> {canonical_name}.slot_finalisation",
        )

    return HeuristicConfig(
        path=config_path,
        config_version=config_version,
        canonical_profile_name=canonical_name,
        profiles=profiles,
    )


def load_run_config(path: Path) -> dict[str, Any]:
    """Load a YAML run configuration (e.g. ``ml/configs/heuristic_v1.yaml``).

    Run configs hold *run* settings -- which profile to use, evaluation
    thresholds, output paths. They must never restate baseline constants; those
    live only in the canonical JSON.
    """
    import yaml  # imported lazily so `config show` works without the extra

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
    return loaded
