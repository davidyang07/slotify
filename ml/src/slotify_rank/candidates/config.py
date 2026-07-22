"""Candidate-generation settings (``ml/configs/dataset_v1.yaml``).

These are *recall* controls for building a dataset pool, and they are separate
from the baseline constants in ``config/heuristic_offline_v1.json``. The product
wants three good slots; a training set wants a broad pool with plenty of
negatives, so the generators here run at much higher recall than the product
path and their tuning must never leak back into the baseline.

The one exception is silence detection, whose thresholds are *read from* the
canonical profile so the ``silence`` generator agrees with what the product
considers a pause. Overriding them is possible but explicit, and the override is
recorded in the run summary.

Every value that changes which candidates exist is written into
``artifacts/dataset/candidate_statistics.json``, so a change in candidate counts
can always be traced to a configuration change rather than to drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.config.settings import HeuristicProfile

__all__ = [
    "SilenceConfig",
    "PauseConfig",
    "RmsMinimumConfig",
    "FixedIntervalConfig",
    "TranscriptConfig",
    "GenerationConfig",
    "load_generation_config",
]


@dataclass(frozen=True)
class SilenceConfig:
    """Product-equivalent silence detection.

    ``min_silence_len_ms`` and ``threshold_offset_db`` default to ``None``,
    meaning "take the canonical profile's values", which is what keeps this
    generator aligned with ``heuristic_offline_v1``.
    """

    enabled: bool = True
    min_silence_len_ms: int | None = None
    threshold_offset_db: float | None = None
    fallback_threshold_dbfs: float | None = None


@dataclass(frozen=True)
class PauseConfig:
    """Shorter pauses than the product accepts, for candidate recall.

    A 300 ms breath is not a slot the product would offer, but it is exactly the
    kind of near-miss a ranker needs to learn to reject.
    """

    enabled: bool = True
    min_pause_ms: int = 250
    max_pause_ms: int = 700
    threshold_offset_db: float = -16.0


@dataclass(frozen=True)
class RmsMinimumConfig:
    """Local energy minima: quiet moments that are not full silences."""

    enabled: bool = True
    window_ms: int = 300
    hop_ms: int = 100
    #: Only windows quieter than this percentile of the episode are considered.
    percentile: float = 15.0
    #: Minimum spacing between two accepted minima.
    min_spacing_ms: int = 3000


@dataclass(frozen=True)
class FixedIntervalConfig:
    """Regular grid points, purely to guarantee negative coverage.

    Most grid points land mid-sentence, which is the point: without them a
    dataset built only from silences contains almost no clear negatives and any
    model trained on it looks better than it is.
    """

    enabled: bool = True
    interval_ms: int = 30_000


@dataclass(frozen=True)
class TranscriptConfig:
    """Segment/sentence endings from a supplied timestamped transcript.

    Phase 2 never *generates* transcripts. This consumes one if the operator
    supplies it; otherwise the generator contributes nothing and says so.
    """

    enabled: bool = True
    context_chars: int = 240


@dataclass(frozen=True)
class GenerationConfig:
    """Everything that determines which candidates exist for an episode."""

    dataset_version: str = "dataset_v1"
    #: No candidate within this distance of either end of the episode. Both a
    #: quality guard (nobody wants an ad 2 s in) and a safety guard for the
    #: labelling UI, which needs context either side of the timestamp.
    edge_guard_ms: int = 5_000
    #: Candidates closer together than this collapse into one.
    merge_tolerance_ms: int = 400
    #: Upper bound on post-merge density. ``None`` disables the cap. When it
    #: bites, the lowest-heuristic-score candidates are dropped and the fact is
    #: reported -- density is never trimmed silently.
    max_candidates_per_minute: float | None = 12.0
    #: Window used for ``local_energy_summary``.
    energy_summary_window_ms: int = 1_000
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    pause: PauseConfig = field(default_factory=PauseConfig)
    rms_minimum: RmsMinimumConfig = field(default_factory=RmsMinimumConfig)
    fixed_interval: FixedIntervalConfig = field(default_factory=FixedIntervalConfig)
    transcript: TranscriptConfig = field(default_factory=TranscriptConfig)

    def __post_init__(self) -> None:
        if self.edge_guard_ms < 0:
            raise ValueError("edge_guard_ms must be non-negative")
        if self.merge_tolerance_ms < 0:
            raise ValueError("merge_tolerance_ms must be non-negative")
        if self.max_candidates_per_minute is not None and (
            self.max_candidates_per_minute <= 0
        ):
            raise ValueError(
                "max_candidates_per_minute must be positive or null (null disables "
                "the density cap)"
            )
        if self.energy_summary_window_ms <= 0:
            raise ValueError("energy_summary_window_ms must be positive")
        if self.fixed_interval.enabled and self.fixed_interval.interval_ms <= 0:
            raise ValueError("fixed_interval.interval_ms must be positive")
        if self.rms_minimum.enabled and not 0 < self.rms_minimum.percentile < 100:
            raise ValueError("rms_minimum.percentile must be strictly between 0 and 100")
        if self.pause.enabled and self.pause.min_pause_ms > self.pause.max_pause_ms:
            raise ValueError("pause.min_pause_ms must not exceed pause.max_pause_ms")

    def resolved_silence(self, profile: HeuristicProfile) -> tuple[int, float, float]:
        """``(min_silence_len_ms, threshold_offset_db, fallback_dbfs)``.

        Falls back to the canonical profile for anything the dataset config
        leaves unset.
        """
        detection = profile.silence_detection
        min_len = self.silence.min_silence_len_ms
        offset = self.silence.threshold_offset_db
        fallback = self.silence.fallback_threshold_dbfs
        return (
            int(detection["min_silence_len_ms"] if min_len is None else min_len),
            float(
                detection["silence_threshold_offset_db"] if offset is None else offset
            ),
            float(
                detection["fallback_silence_threshold_dbfs"]
                if fallback is None
                else fallback
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return dict(value)


def _build(cls: type, payload: Mapping[str, Any], where: str):
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"{where}: unknown setting(s) {unknown}. Known settings: {sorted(known)}"
        )
    return cls(**payload)


def load_generation_config(path: Path | str | None = None) -> GenerationConfig:
    """Load ``dataset_v1.yaml``. With no path, returns the built-in defaults."""
    if path is None:
        return GenerationConfig()

    import yaml  # lazy, matching config.settings.load_run_config

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Candidate generation config not found: {config_path}"
        ) from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    generation = _section(loaded, "candidate_generation")
    where = f"{config_path} -> candidate_generation"
    nested = {
        "silence": _build(SilenceConfig, _section(generation, "silence"), f"{where}.silence"),
        "pause": _build(PauseConfig, _section(generation, "pause"), f"{where}.pause"),
        "rms_minimum": _build(
            RmsMinimumConfig, _section(generation, "rms_minimum"), f"{where}.rms_minimum"
        ),
        "fixed_interval": _build(
            FixedIntervalConfig,
            _section(generation, "fixed_interval"),
            f"{where}.fixed_interval",
        ),
        "transcript": _build(
            TranscriptConfig, _section(generation, "transcript"), f"{where}.transcript"
        ),
    }
    scalars = {
        key: value
        for key, value in generation.items()
        if key not in ("silence", "pause", "rms_minimum", "fixed_interval", "transcript")
    }
    payload: dict[str, Any] = {
        "dataset_version": str(loaded.get("dataset_version", "dataset_v1")),
        **scalars,
        **nested,
    }
    return _build(GenerationConfig, payload, where)
