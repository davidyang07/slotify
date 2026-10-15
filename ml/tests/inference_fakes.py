"""Deterministic stand-ins for Whisper, the Whisper encoder and MiniLM.

The stage code, cache identities, stores, assembly, the checkpoint and the
scoring all run for real; only the three pretrained models are fake. That is the
boundary that keeps the inference tests offline and fast.

Usable two ways: :func:`install` with a pytest ``monkeypatch`` (undone after the
test), or :func:`install` with no argument from the bootstrap of a subprocess
that then runs the real ``slotify_rank.cli`` -- see :data:`BOOTSTRAP`.

Two hooks for subprocess tests, both environment variables so they cross the
process boundary: ``SLOTIFY_FAKE_STDOUT_NOISE=1`` makes the transcriber
``print()`` to stdout (the worker must keep that off its protocol stream), and
``SLOTIFY_FAKE_HANG_MARKER=<path>`` makes it create ``<path>`` and then sleep,
so a test can signal a worker that is provably mid-request.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from slotify_rank.config.settings import find_repo_root

SAMPLE_RATE = 16_000
DIMENSION = 384

#: Run as ``python -c BOOTSTRAP <cli args...>`` with the ml/ directory as cwd:
#: the real CLI, with the fakes installed first.
BOOTSTRAP = (
    "import sys\n"
    "from tests.inference_fakes import install\n"
    "install()\n"
    "from slotify_rank.cli import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n"
)

REPO_ROOT = Path(find_repo_root())
#: The committed checkpoint the product serves, and a real repository fixture.
SHIPPED_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "training" / "gated-d8ed976101aa4c3b" / "best_checkpoint.pt"
)
FIXTURE_AUDIO = REPO_ROOT / "backend" / "audio_tests" / "ryan-test1.mp3"


def serving_fixtures_missing() -> str:
    """Why the real-audio serving tests cannot run here, or ``""`` when they can."""
    from slotify_rank.data.ffmpeg import FFmpegNotFound, require_ffmpeg

    if not SHIPPED_CHECKPOINT.is_file():
        return "the committed checkpoint is absent"
    if not FIXTURE_AUDIO.is_file():
        return "the repository audio fixture is absent"
    try:
        require_ffmpeg()
    except FFmpegNotFound:
        return "FFmpeg is not installed"
    return ""


#: How many times each fake's ``load()`` ran, by class name.
LOADS: dict[str, int] = {}


def _count(name: str) -> None:
    LOADS[name] = LOADS.get(name, 0) + 1


class FakeTranscriber:
    def __init__(self, config: Any) -> None:
        self.config = config

    def load(self) -> "FakeTranscriber":
        _count("transcriber")
        return self

    @property
    def device(self) -> str:
        return "cpu"

    def release(self) -> None:
        return None

    def transcribe(self, episode_id, audio, audio_sha256, identity_digest, sample_rate=SAMPLE_RATE):
        from slotify_rank.transcription.schema import EpisodeTranscript
        from slotify_rank.transcription.segments import build_segments

        if os.environ.get("SLOTIFY_FAKE_STDOUT_NOISE") == "1":
            print("fake transcriber chatter on stdout")
        marker = os.environ.get("SLOTIFY_FAKE_HANG_MARKER")
        if marker:
            Path(marker).write_text("transcribing", encoding="utf-8")
            time.sleep(120)

        duration_ms = int(audio.size * 1000 / sample_rate)
        spans = [
            (0, 4_000, "The first topic finishes here."),
            (4_500, 9_000, "A second thought begins now."),
            (9_500, 15_000, "And a third one after that."),
            (15_500, 21_000, "Finally we wrap up."),
        ]
        segments = build_segments(
            episode_id,
            [(start, min(end, duration_ms), text, ()) for start, end, text in spans if start < duration_ms],
            duration_ms,
        )
        return EpisodeTranscript(
            episode_id=episode_id,
            language="en",
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            audio_sha256=audio_sha256,
            audio_duration_ms=duration_ms,
            segments=tuple(segments),
            transcription_config=self.config.to_dict(),
            identity_digest=identity_digest,
        )


class FakeAudioEncoder:
    def __init__(self, config: Any) -> None:
        self.config = config

    def load(self) -> "FakeAudioEncoder":
        _count("audio_encoder")
        return self

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dimension(self) -> int:
        return DIMENSION

    def release(self) -> None:
        return None

    def encode_episode(self, episode_id, audio, sample_rate=SAMPLE_RATE):
        from slotify_rank.embeddings.pooling import FrameGrid
        from slotify_rank.embeddings.whisper_audio import EpisodeAudioStates

        duration_ms = int(audio.size * 1000 / sample_rate)
        frames = max(1, duration_ms // 20)
        states = np.tile(
            (np.arange(frames, dtype=np.float32) / frames).reshape(frames, 1),
            (1, DIMENSION),
        )
        grid = FrameGrid(
            chunk_start_ms=0,
            chunk_end_ms=duration_ms,
            frame_duration_ms=20.0,
            total_frames=frames,
        )
        return EpisodeAudioStates(
            episode_id=episode_id,
            chunks=((grid, states),),
            dimension=DIMENSION,
            frame_duration_ms=20.0,
        )


class FakeTextEncoder:
    def __init__(self, config: Any) -> None:
        self.config = config

    def load(self) -> "FakeTextEncoder":
        _count("text_encoder")
        return self

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dimension(self) -> int:
        return DIMENSION

    def release(self) -> None:
        return None

    def encode(self, texts):
        if not texts:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        rows = []
        for text in texts:
            # Seeded by content, not by hash(): stable across processes.
            seed = sum(ord(character) for character in text) % (2**32)
            vector = np.random.default_rng(seed).normal(size=DIMENSION).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


def install(monkeypatch: Any = None) -> None:
    """Replace the three model wrappers where the stages look them up."""
    from slotify_rank.embeddings import minilm_text, whisper_audio
    from slotify_rank.transcription import local_whisper

    LOADS.clear()
    replacements = (
        (local_whisper, "WhisperTranscriber", FakeTranscriber),
        (whisper_audio, "WhisperAudioEncoder", FakeAudioEncoder),
        (minilm_text, "MiniLmTextEncoder", FakeTextEncoder),
    )
    for module, name, fake in replacements:
        if monkeypatch is None:
            setattr(module, name, fake)
        else:
            monkeypatch.setattr(module, name, fake)
