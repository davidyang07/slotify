"""The stored transcript artifact: segments, words and their provenance.

Canonical timestamps are **integer milliseconds**, matching every other
timestamp in the project (candidates, episode durations, split manifests).
Whisper reports seconds as floats; the conversion happens once, here, at the
boundary -- so nothing downstream ever has to decide whether ``12.34`` means
seconds or something else, and a candidate at 12_340 ms joins a segment on
exactly the same integer scale.

Three invariants are enforced rather than documented:

1. **Segments lie inside the episode.** A segment ending after the audio does
   is a symptom of chunk offsets being applied twice, which is otherwise almost
   invisible -- the text still reads correctly.
2. **Segments are ordered and non-overlapping** after reconciliation.
3. **A successful transcript is non-empty.** An episode that produced no
   segments is a *failure*, recorded as one. Silently storing an empty
   transcript would make every candidate on that episode "text-missing" and
   quietly shrink the dataset.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from slotify_rank.config.versions import (
    FEATURE_PIPELINE_VERSION,
    SUPPORTED_TRANSCRIPTION_VERSIONS,
    TRANSCRIPTION_VERSION,
)

__all__ = [
    "TerminalPunctuation",
    "TranscriptWord",
    "TranscriptSegment",
    "EpisodeTranscript",
    "make_segment_id",
]

#: Category of the segment's final punctuation mark. ``none`` means the segment
#: ended mid-thought, which is a genuinely different signal from ``period`` when
#: deciding whether a breakpoint is natural -- so it is a category, not a null.
TerminalPunctuation = Literal[
    "period", "question", "exclamation", "ellipsis", "comma", "dash", "none"
]

#: Closing punctuation ignored when looking for the terminal mark. Includes the
#: typographic quotes Whisper actually emits, not just the ASCII ones.
_CLOSING_CHARACTERS = "\"')]}’”»"

_ELLIPSIS_RE = re.compile(rf"(\.\.\.|…)[{re.escape(_CLOSING_CHARACTERS)}]?$")
_TERMINAL_MAP: tuple[tuple[str, TerminalPunctuation], ...] = (
    (".", "period"),
    ("?", "question"),
    ("!", "exclamation"),
    (",", "comma"),
    ("-", "dash"),
    ("–", "dash"),
    ("—", "dash"),
)


def make_segment_id(episode_id: str, index: int, start_ms: int) -> str:
    """Deterministic segment identity: ``<episode>:seg<index>@<start_ms>``.

    Both the index and the start time are included on purpose. The index alone
    would renumber every downstream reference when one boundary segment is
    dropped by overlap reconciliation; the start time alone is not unique when a
    model emits two zero-length segments at the same instant.
    """
    if index < 0:
        raise ValueError(f"Segment index must be non-negative, got {index}")
    if start_ms < 0:
        raise ValueError(f"start_ms must be non-negative, got {start_ms}")
    return f"{episode_id}:seg{index:05d}@{start_ms:09d}"


@dataclass(frozen=True)
class TranscriptWord:
    """One word with its own timing.

    Word timings are optional throughout Phase 3: ``whisper-tiny.en`` via the
    Transformers pipeline returns them only when word-level timestamps are
    requested, and they are not always well-formed. Nothing in the feature
    pipeline *requires* them; they are stored when available because they make
    a much tighter pause estimate possible later.
    """

    word: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError(f"word {self.word!r}: start_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"word {self.word!r}: end_ms {self.end_ms} precedes start_ms "
                f"{self.start_ms}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TranscriptWord":
        return cls(
            word=str(raw["word"]),
            start_ms=int(raw["start_ms"]),
            end_ms=int(raw["end_ms"]),
        )


@dataclass(frozen=True)
class TranscriptSegment:
    """One timestamped span of speech.

    ``text`` is the model's output verbatim; ``normalized_text`` is the
    lowercased, punctuation-stripped, whitespace-collapsed form. Both are
    stored: embeddings and human review want the raw text (punctuation carries
    the boundary signal), while matching and word counts want the normalized
    form. Deriving one from the other at read time would mean every consumer
    re-implementing the same normalization slightly differently.
    """

    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    normalized_text: str
    sentence_end: bool
    terminal_punctuation: TerminalPunctuation
    words: tuple[TranscriptWord, ...] = ()

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError(
                f"{self.segment_id}: start_ms must be non-negative, got {self.start_ms}"
            )
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"{self.segment_id}: end_ms {self.end_ms} precedes start_ms "
                f"{self.start_ms}"
            )
        if self.terminal_punctuation not in (
            "period",
            "question",
            "exclamation",
            "ellipsis",
            "comma",
            "dash",
            "none",
        ):
            raise ValueError(
                f"{self.segment_id}: unknown terminal_punctuation "
                f"{self.terminal_punctuation!r}"
            )

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def word_count(self) -> int:
        return len(self.normalized_text.split()) if self.normalized_text else 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["words"] = [word.to_dict() for word in self.words]
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=str(raw["segment_id"]),
            start_ms=int(raw["start_ms"]),
            end_ms=int(raw["end_ms"]),
            text=str(raw["text"]),
            normalized_text=str(raw["normalized_text"]),
            sentence_end=bool(raw["sentence_end"]),
            terminal_punctuation=str(raw["terminal_punctuation"]),  # type: ignore[arg-type]
            words=tuple(
                TranscriptWord.from_mapping(word) for word in raw.get("words", ())
            ),
        )


@dataclass(frozen=True)
class EpisodeTranscript:
    """A complete transcript plus everything needed to invalidate it.

    ``audio_sha256`` is the checksum of the **normalized** render that was
    actually transcribed, not the original: re-normalizing an episode changes
    the samples the model saw, and a transcript keyed on the original checksum
    would survive that change while no longer describing the audio.
    """

    episode_id: str
    language: str
    model_id: str
    model_revision: str
    audio_sha256: str
    audio_duration_ms: int
    segments: tuple[TranscriptSegment, ...]
    transcription_config: dict[str, Any] = field(default_factory=dict)
    #: Full cache-identity digest this transcript was produced under.
    identity_digest: str = ""
    has_word_timestamps: bool = False
    transcription_version: str = TRANSCRIPTION_VERSION
    created_by_pipeline_version: str = FEATURE_PIPELINE_VERSION

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(
                f"{self.episode_id}: refusing to store an empty transcript as a "
                "success. An episode that produced no segments is a transcription "
                "failure and must be recorded as one."
            )
        if self.audio_duration_ms <= 0:
            raise ValueError(
                f"{self.episode_id}: audio_duration_ms must be positive, got "
                f"{self.audio_duration_ms}"
            )
        if len(self.audio_sha256) != 64:
            raise ValueError(
                f"{self.episode_id}: audio_sha256 must be 64 hex characters"
            )
        previous_end = -1
        for segment in self.segments:
            if segment.start_ms < previous_end:
                raise ValueError(
                    f"{self.episode_id}: segment {segment.segment_id} starts at "
                    f"{segment.start_ms} ms, before the previous segment ended at "
                    f"{previous_end} ms. Segments must be ordered and "
                    "non-overlapping after reconciliation."
                )
            if segment.end_ms > self.audio_duration_ms:
                raise ValueError(
                    f"{self.episode_id}: segment {segment.segment_id} ends at "
                    f"{segment.end_ms} ms but the audio is only "
                    f"{self.audio_duration_ms} ms long. This usually means a chunk "
                    "offset was applied twice."
                )
            previous_end = segment.end_ms

    # -- derived quantities ------------------------------------------------
    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def transcribed_duration_ms(self) -> int:
        """Total speech covered by segments (not the episode duration).

        Reported separately from the episode duration because the gap between
        them is the silence the model found, which is exactly what the resume
        claim "transcribed audio hours" must not overstate.
        """
        return sum(segment.duration_ms for segment in self.segments)

    @property
    def word_count(self) -> int:
        return sum(segment.word_count for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "language": self.language,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "audio_sha256": self.audio_sha256,
            "audio_duration_ms": self.audio_duration_ms,
            "transcription_config": dict(self.transcription_config),
            "identity_digest": self.identity_digest,
            "has_word_timestamps": self.has_word_timestamps,
            "transcription_version": self.transcription_version,
            "created_by_pipeline_version": self.created_by_pipeline_version,
            "segment_count": self.segment_count,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EpisodeTranscript":
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Transcript must be a mapping, got {type(raw).__name__}"
            )
        version = str(raw.get("transcription_version", ""))
        if version not in SUPPORTED_TRANSCRIPTION_VERSIONS:
            raise ValueError(
                f"Unsupported transcription_version {version!r}; this build reads "
                f"{sorted(SUPPORTED_TRANSCRIPTION_VERSIONS)}"
            )
        return cls(
            episode_id=str(raw["episode_id"]),
            language=str(raw["language"]),
            model_id=str(raw["model_id"]),
            model_revision=str(raw["model_revision"]),
            audio_sha256=str(raw["audio_sha256"]),
            audio_duration_ms=int(raw["audio_duration_ms"]),
            segments=tuple(
                TranscriptSegment.from_mapping(item) for item in raw["segments"]
            ),
            transcription_config=dict(raw.get("transcription_config") or {}),
            identity_digest=str(raw.get("identity_digest", "")),
            has_word_timestamps=bool(raw.get("has_word_timestamps", False)),
            transcription_version=version,
            created_by_pipeline_version=str(
                raw.get("created_by_pipeline_version", "")
            ),
        )


def classify_terminal_punctuation(text: str) -> TerminalPunctuation:
    """Category of the last punctuation mark in ``text``.

    Checked before the single-character marks so ``"..."`` is an ellipsis (a
    trailing-off, weak boundary) rather than a period (a firm one).
    """
    stripped = str(text or "").strip()
    if not stripped:
        return "none"
    if _ELLIPSIS_RE.search(stripped):
        return "ellipsis"
    # Ignore closing quotes and brackets, matching ends_with_sentence_boundary.
    # Whisper emits typographic quotes ("smart quotes"), so the curly forms must
    # be stripped too or `He said "stop."` is misread as having no terminal
    # punctuation -- turning a clean sentence end into a mid-thought one.
    while stripped and stripped[-1] in _CLOSING_CHARACTERS:
        stripped = stripped[:-1]
    if not stripped:
        return "none"
    for mark, category in _TERMINAL_MAP:
        if stripped.endswith(mark):
            return category
    return "none"


def sort_segments(
    segments: Sequence[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Canonical ordering: start time, then end time, then id."""
    return sorted(
        segments,
        key=lambda segment: (segment.start_ms, segment.end_ms, segment.segment_id),
    )
