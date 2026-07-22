"""Thin loader for the canonical heuristic configuration.

The constants live in ``config/heuristic_offline_v1.json`` at the repository
root and are shared with ``backend/src/lib/heuristic-config.ts`` and
``ml/src/slotify_rank/config/settings.py``. This module exists so the Python
pipeline reads the same numbers as the TypeScript product code and the ML
baseline, rather than keeping a third copy in source.

Two profiles are exposed:

``PRODUCT_V1``
    The canonical ``heuristic_offline_v1`` baseline. Used by
    :mod:`ad_inserter.analyze_cli`, which serves ``POST /api/insert-sections``.

``LEGACY_CLI_V1``
    The divergent constants of the standalone LLM-driven insertion path
    (:mod:`ad_inserter.cli`), whose silence detector uses a 500 ms minimum rather
    than 700 ms. It is **not** the baseline; it is kept distinct so the two are
    never accidentally merged.

Values are unchanged from the previous inline literals.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CONFIG_PATH",
    "PRODUCT_PROFILE",
    "LEGACY_CLI_PROFILE",
    "config_version",
    "load_profile",
    "silence_params",
    "PRODUCT_V1",
    "LEGACY_CLI_V1",
]

# backend/ad_inserter/heuristic_config.py -> backend/ -> repository root
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "heuristic_offline_v1.json"

PRODUCT_PROFILE = "heuristic_offline_v1"
LEGACY_CLI_PROFILE = "legacy_cli_v1"


@lru_cache(maxsize=1)
def _load() -> Mapping[str, Any]:
    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Canonical heuristic config not found at {CONFIG_PATH}. "
            "It is required: the ad-insertion baseline must not fall back to "
            "hard-coded defaults."
        ) from error
    return json.loads(raw)


def config_version() -> str:
    """Version stamp of the canonical configuration file."""
    return str(_load()["config_version"])


def load_profile(name: str) -> Mapping[str, Any]:
    """Return one named profile, raising if it is absent."""
    profiles = _load()["profiles"]
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown heuristic profile {name!r}. Available: {available}")
    return profiles[name]


def silence_params(profile_name: str) -> tuple[int, float, float]:
    """Return ``(min_silence_len_ms, threshold_offset_db, fallback_dbfs)``."""
    detection = load_profile(profile_name)["silence_detection"]
    return (
        int(detection["min_silence_len_ms"]),
        float(detection["silence_threshold_offset_db"]),
        float(detection["fallback_silence_threshold_dbfs"]),
    )


PRODUCT_V1 = load_profile(PRODUCT_PROFILE)
LEGACY_CLI_V1 = load_profile(LEGACY_CLI_PROFILE)
