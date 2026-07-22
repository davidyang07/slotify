"""Phase 3 configuration: transcription, features, embeddings, pipeline.

Every value that changes an artifact's bytes lives here and nowhere else. That
is not tidiness for its own sake -- each config object renders to a digest that
becomes part of the corresponding cache identity, so a setting that is hidden in
code is a setting that cannot invalidate a stale cache when it changes.

Four files, matching the four stages::

    ml/configs/transcription_v1.yaml
    ml/configs/features_v1.yaml
    ml/configs/embeddings_v1.yaml
    ml/configs/feature_pipeline_v1.yaml

Loading with no path returns the built-in defaults, which are the same values
the YAML files carry. The files exist so the defaults are reviewable and
overridable; the dataclasses exist so a typo is an error rather than a silently
ignored key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from slotify_rank.pipeline.identity import config_digest

__all__ = [
    "TranscriptionConfig",
    "WindowConfig",
    "AcousticConfig",
    "TranscriptContextConfig",
    "FeatureConfig",
    "AudioEmbeddingConfig",
    "TextEmbeddingConfig",
    "EmbeddingConfig",
    "PipelineConfig",
    "load_transcription_config",
    "load_feature_config",
    "load_embedding_config",
    "load_pipeline_config",
]

#: Default local models. Both run on CPU and neither requires a paid API.
DEFAULT_WHISPER_MODEL = "openai/whisper-tiny.en"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build(cls: type, payload: Mapping[str, Any], where: str):
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"{where}: unknown setting(s) {unknown}. Known settings: {sorted(known)}"
        )
    return cls(**payload)


def _section(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return dict(value)


def _load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml  # lazy, matching config.settings.load_run_config

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Config not found: {config_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return dict(loaded)


class _Digestible:
    """Mixin: a stable digest over the config's own values."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return config_digest(self.to_dict())


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionConfig(_Digestible):
    """Local Whisper transcription settings.

    ``chunk_ms`` defaults to Whisper's own 30-second receptive field. Going
    larger gains nothing (the model pads or truncates to 30 s regardless) and
    going much smaller costs accuracy, since the model loses context.

    ``overlap_ms`` of 5 s is roughly one spoken clause -- long enough that a word
    straddling a boundary is seen whole by at least one chunk, short enough that
    the duplicated region stays a small fraction of the work.
    """

    model_id: str = DEFAULT_WHISPER_MODEL
    #: Pinned Hugging Face revision. ``"main"`` means "whatever is current",
    #: which is honest rather than reproducible; pin a commit SHA for a corpus
    #: whose transcripts must be reproducible months later.
    model_revision: str = "main"
    language: str = "en"
    device: str = "auto"
    dtype: str = "float32"
    chunk_ms: int = 30_000
    overlap_ms: int = 5_000
    batch_size: int = 1
    #: Word timestamps roughly double transcription time and tiny.en's are not
    #: reliable enough to justify that by default. Nothing in Phase 3 requires
    #: them; they are stored when requested.
    word_timestamps: bool = False
    num_beams: int = 1
    #: Greedy decoding. Sampling would make transcripts non-reproducible, which
    #: would defeat the entire cache-identity scheme.
    temperature: float = 0.0
    max_new_tokens: int = 440

    def __post_init__(self) -> None:
        if self.chunk_ms <= 0:
            raise ValueError("transcription.chunk_ms must be positive")
        if self.overlap_ms < 0:
            raise ValueError("transcription.overlap_ms must be non-negative")
        if self.overlap_ms >= self.chunk_ms:
            raise ValueError(
                "transcription.overlap_ms must be strictly less than chunk_ms"
            )
        if self.batch_size < 1:
            raise ValueError("transcription.batch_size must be at least 1")
        if self.temperature != 0.0:
            raise ValueError(
                "transcription.temperature must be 0.0: sampled decoding is not "
                "reproducible, so a cached transcript could never be trusted"
            )
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unknown dtype {self.dtype!r}")


def load_transcription_config(path: Path | str | None = None) -> TranscriptionConfig:
    if path is None:
        return TranscriptionConfig()
    raw = _load_yaml(path)
    return _build(
        TranscriptionConfig, _section(raw, "transcription"), f"{path} -> transcription"
    )


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowConfig(_Digestible):
    """Candidate-centred analysis windows, in milliseconds either side.

    Three scales, because a breakpoint is a multi-scale phenomenon: ``short``
    catches the pause itself, ``medium`` catches the phrase either side of it,
    and ``context`` catches whether the topic actually changed.
    """

    short_ms: int = 1_000
    medium_ms: int = 3_000
    context_ms: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.short_ms <= self.medium_ms <= self.context_ms:
            raise ValueError(
                "windows must satisfy 0 < short_ms <= medium_ms <= context_ms; got "
                f"{self.short_ms}, {self.medium_ms}, {self.context_ms}"
            )

    def as_pairs(self) -> tuple[tuple[str, int], ...]:
        return (
            ("short", self.short_ms),
            ("medium", self.medium_ms),
            ("context", self.context_ms),
        )


@dataclass(frozen=True)
class AcousticConfig(_Digestible):
    """STFT and framing parameters for the handcrafted descriptors.

    ``frame_ms``/``hop_ms`` of 25/10 is the standard speech-analysis framing and
    matches Whisper's own 10 ms hop, so handcrafted features and encoder frames
    share a time base.
    """

    frame_ms: int = 25
    hop_ms: int = 10
    n_fft: int = 512
    #: Below this (relative to the episode's median frame energy, in dB) a frame
    #: counts as non-speech for the pause and silence-duration features.
    speech_threshold_db: float = -35.0
    #: Spectral contrast needs enough bands below Nyquist to be defined; at
    #: 16 kHz, 6 bands is the most that stays numerically stable.
    n_spectral_contrast_bands: int = 6
    roll_off_percent: float = 0.85

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("acoustic.frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError(
                "acoustic.hop_ms must not exceed frame_ms, otherwise frames skip "
                "audio and the descriptors miss transients"
            )
        if self.n_fft <= 0 or self.n_fft & (self.n_fft - 1):
            raise ValueError(f"acoustic.n_fft must be a power of two, got {self.n_fft}")
        if not 0 < self.roll_off_percent < 1:
            raise ValueError("acoustic.roll_off_percent must be in (0, 1)")


@dataclass(frozen=True)
class TranscriptContextConfig(_Digestible):
    """Bounds on the text taken either side of a candidate."""

    max_chars_before: int = 600
    max_chars_after: int = 600
    max_segments_before: int = 4
    max_segments_after: int = 4
    #: A transcript segment further than this from the candidate is not context.
    #: Without the bound, a candidate in a long silence would pull in text from
    #: a completely different part of the episode and call it "the sentence
    #: before".
    max_gap_ms: int = 15_000

    def __post_init__(self) -> None:
        if self.max_chars_before <= 0 or self.max_chars_after <= 0:
            raise ValueError("transcript_context max_chars_* must be positive")
        if self.max_segments_before <= 0 or self.max_segments_after <= 0:
            raise ValueError("transcript_context max_segments_* must be positive")
        if self.max_gap_ms <= 0:
            raise ValueError("transcript_context.max_gap_ms must be positive")


@dataclass(frozen=True)
class FeatureConfig(_Digestible):
    """Everything that determines a candidate's handcrafted feature values."""

    feature_version: str = "features_v1"
    windows: WindowConfig = field(default_factory=WindowConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    transcript_context: TranscriptContextConfig = field(
        default_factory=TranscriptContextConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "windows": self.windows.to_dict(),
            "acoustic": self.acoustic.to_dict(),
            "transcript_context": self.transcript_context.to_dict(),
        }


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    if path is None:
        return FeatureConfig()
    raw = _load_yaml(path)
    features = _section(raw, "features")
    where = f"{path} -> features"
    payload: dict[str, Any] = {
        "feature_version": str(raw.get("feature_version", "features_v1")),
        "windows": _build(
            WindowConfig, _section(features, "windows"), f"{where}.windows"
        ),
        "acoustic": _build(
            AcousticConfig, _section(features, "acoustic"), f"{where}.acoustic"
        ),
        "transcript_context": _build(
            TranscriptContextConfig,
            _section(features, "transcript_context"),
            f"{where}.transcript_context",
        ),
    }
    scalars = {
        key: value
        for key, value in features.items()
        if key not in ("windows", "acoustic", "transcript_context")
    }
    return _build(FeatureConfig, {**payload, **scalars}, where)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioEmbeddingConfig(_Digestible):
    """Frozen Whisper encoder settings.

    ``chunk_ms`` is fixed at Whisper's 30-second window. The encoder always
    produces a fixed number of frames for that window regardless of how much
    real audio it contains, so anything other than 30 s would mean pooling over
    padding without knowing it.
    """

    model_id: str = DEFAULT_WHISPER_MODEL
    model_revision: str = "main"
    device: str = "auto"
    dtype: str = "float32"
    chunk_ms: int = 30_000
    overlap_ms: int = 5_000
    batch_size: int = 1
    #: Mean pooling over the frames inside each candidate window. Mean is the
    #: only pooling here that is defined for a one-frame window and is unaffected
    #: by how many frames the window happens to contain.
    pooling: str = "mean"
    #: Store `context_before - context_after` as a fourth vector. It is exactly
    #: derivable from the other two, so it is redundant for a model that sees
    #: both -- but it is cheap and makes the "discontinuity across the break"
    #: signal available to linear probes.
    include_difference: bool = True

    def __post_init__(self) -> None:
        if self.pooling != "mean":
            raise ValueError(
                f"Unsupported pooling {self.pooling!r}; only 'mean' is implemented"
            )
        if self.chunk_ms != 30_000:
            raise ValueError(
                "audio.chunk_ms must be 30000: the Whisper encoder's receptive "
                "field is fixed at 30 s and any other value would pool over "
                "padding frames"
            )
        if self.overlap_ms < 0 or self.overlap_ms >= self.chunk_ms:
            raise ValueError("audio.overlap_ms must be in [0, chunk_ms)")
        if self.batch_size < 1:
            raise ValueError("audio.batch_size must be at least 1")


@dataclass(frozen=True)
class TextEmbeddingConfig(_Digestible):
    """Frozen MiniLM settings."""

    model_id: str = DEFAULT_TEXT_MODEL
    model_revision: str = "main"
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 32
    max_seq_length: int = 256
    #: MiniLM is trained for cosine similarity on normalized vectors, and the
    #: cosine-similarity scalar feature assumes it.
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("text.batch_size must be at least 1")
        if self.max_seq_length < 8:
            raise ValueError("text.max_seq_length must be at least 8")


@dataclass(frozen=True)
class EmbeddingConfig(_Digestible):
    embedding_version: str = "embeddings_v1"
    audio: AudioEmbeddingConfig = field(default_factory=AudioEmbeddingConfig)
    text: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_version": self.embedding_version,
            "audio": self.audio.to_dict(),
            "text": self.text.to_dict(),
        }


def load_embedding_config(path: Path | str | None = None) -> EmbeddingConfig:
    if path is None:
        return EmbeddingConfig()
    raw = _load_yaml(path)
    embeddings = _section(raw, "embeddings")
    where = f"{path} -> embeddings"
    payload: dict[str, Any] = {
        "embedding_version": str(raw.get("embedding_version", "embeddings_v1")),
        "audio": _build(
            AudioEmbeddingConfig, _section(embeddings, "audio"), f"{where}.audio"
        ),
        "text": _build(
            TextEmbeddingConfig, _section(embeddings, "text"), f"{where}.text"
        ),
    }
    scalars = {
        key: value for key, value in embeddings.items() if key not in ("audio", "text")
    }
    return _build(EmbeddingConfig, {**payload, **scalars}, where)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig(_Digestible):
    """Orchestration policy. Deliberately *not* part of any cache identity.

    Nothing here changes an artifact's bytes -- worker counts and retry limits
    change how long the work takes, not what it produces. Folding them into the
    cache key would invalidate a whole corpus because someone changed
    ``--workers``.
    """

    #: 1 by default. The heavy stages are already internally parallel through
    #: BLAS/torch threading, and running several torch processes on a 4-core
    #: laptop makes everything slower while multiplying peak memory.
    workers: int = 1
    max_retries: int = 1
    #: Release each model before the next stage loads its own. On a laptop,
    #: Whisper and MiniLM resident simultaneously is the difference between
    #: fitting in RAM and swapping.
    release_models_between_stages: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("pipeline.workers must be at least 1")
        if self.max_retries < 0:
            raise ValueError("pipeline.max_retries must be non-negative")


def load_pipeline_config(path: Path | str | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    raw = _load_yaml(path)
    return _build(PipelineConfig, _section(raw, "pipeline"), f"{path} -> pipeline")
