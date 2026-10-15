from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import which


# Song-mode valley picking. Valleys closer together than this are treated as
# one, and frames this close to either end of the track are ignored, so the
# candidates spread through the song instead of collapsing into a quiet intro
# or outro. The values match the heuristic scorer's 6 s slot separation and 5 s
# edge guard, so valleys are not spent on spots the slot selection would skip
# as too close to another or the scorer would penalise as too near an edge.
SONG_VALLEY_MIN_SEPARATION_MS = 6000.0
SONG_VALLEY_EDGE_GUARD_MS = 5000.0

# Analysis -- silence detection, Whisper snippets, the duration -- runs on a
# mono 16 kHz copy, the format Whisper itself transcribes. Silence thresholds
# are relative to the clip's own dBFS, so they behave the same on it. Rendering
# keeps the full-quality ``standardize_audio`` format.
ANALYSIS_FRAME_RATE = 16000
# Song analysis is mono too, but keeps 44.1 kHz: librosa's beat tracker works
# in frames of a fixed number of samples, and at 22.05 kHz the tempo and beats
# of the audio_tests clips change (one goes from 123 to 191 BPM).
SONG_FRAME_RATE = 44100

# Whisper models take seconds and hundreds of MB to load; each process loads a
# given model once and reuses it for every snippet.
_WHISPER_MODELS: Dict[str, Any] = {}


@dataclass
class SongAnalysis:
    candidates_ms: List[int]
    tempo: Optional[float]
    beat_times_ms: List[int]


@dataclass
class SpeakerSegment:
    speaker: str
    start_ms: int
    end_ms: int


def check_ffmpeg() -> None:
    if which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it is in your PATH."
        )


def load_audio(path: Path) -> AudioSegment:
    """Decode all of ``path`` at its own rate and channels (for rendering)."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return AudioSegment.from_file(path)


def standardize_audio(audio: AudioSegment) -> AudioSegment:
    return audio.set_frame_rate(44100).set_channels(2)


@contextmanager
def _decoded_mono(path: Path, frame_rate: int, sample_format: str) -> Iterator[Path]:
    """Decode ``path`` to raw mono PCM in a temp file; yield the file's path.

    ffmpeg downmixes and resamples while it decodes, and the caller reads the
    file back in one allocation of exactly its size, so the process never
    holds a full-rate or stereo copy, nor the transient second copy that
    collecting a pipe's output costs. ``sample_format`` is ``s16le`` or
    ``f32le``. The downmix is the channel mean, as librosa's: without
    ``rematrix_maxval`` ffmpeg mixes stereo to float mono at +3 dB. The file
    is removed on exit.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    # tempfile honours the CLI session's private dir (inside --work-dir).
    fd, raw_name = tempfile.mkstemp(suffix=f".{sample_format}")
    os.close(fd)
    raw_path = Path(raw_name)
    try:
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(frame_rate), "-rematrix_maxval", "1.0",
            "-acodec", f"pcm_{sample_format}", "-f", sample_format,
            str(raw_path),
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"Could not decode {path.name} (ffmpeg exited "
                f"{result.returncode}): {detail[-2000:]}"
            )
        yield raw_path
    finally:
        raw_path.unlink(missing_ok=True)


def load_analysis_audio(path: Path) -> AudioSegment:
    """Decode ``path`` to mono 16-bit PCM at ``ANALYSIS_FRAME_RATE``, for analysis.

    About 115 MB per hour of audio, held once; ``standardize_audio(load_audio())``
    holds about 635 MB per hour at 44.1 kHz stereo, plus transient copies.
    """
    with _decoded_mono(path, ANALYSIS_FRAME_RATE, "s16le") as raw_path:
        data = raw_path.read_bytes()
    # An odd byte count can only be a truncated trailing sample.
    data = data[: len(data) - len(data) % 2]
    return AudioSegment(
        data=data, sample_width=2, frame_rate=ANALYSIS_FRAME_RATE, channels=1
    )


def load_song_samples(path: Path) -> np.ndarray:
    """Decode ``path`` to mono float32 samples at ``SONG_FRAME_RATE``, for analysis."""
    with _decoded_mono(path, SONG_FRAME_RATE, "f32le") as raw_path:
        return np.fromfile(raw_path, dtype="<f4")


def load_audio_window(path: Path, start_ms: int, end_ms: int) -> AudioSegment:
    """Decode only ``[start_ms, end_ms)`` of ``path``, in the rendering format.

    For paths that need a few seconds at full quality (loudness context, room
    tone) without decoding the whole upload at 44.1 kHz stereo.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms, int(end_ms))
    window = AudioSegment.from_file(
        path, start_second=start_ms / 1000.0, duration=(end_ms - start_ms) / 1000.0
    )
    return standardize_audio(window)


def detect_podcast_candidates(audio: AudioSegment) -> List[int]:
    """Silence candidates for the legacy LLM-driven CLI path.

    NOTE: this uses the `legacy_cli_v1` profile (500 ms minimum silence), which
    deliberately differs from the canonical `heuristic_offline_v1` baseline used
    by /api/insert-sections (700 ms). The divergence is pre-existing and is
    recorded rather than reconciled.
    """
    from ad_inserter import heuristic_config

    min_silence_len, threshold_offset, fallback_dbfs = heuristic_config.silence_params(
        heuristic_config.LEGACY_CLI_PROFILE
    )
    silence_thresh = (
        audio.dBFS + threshold_offset if audio.dBFS != float("-inf") else fallback_dbfs
    )
    silence_ranges = detect_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    candidates = [(start + end) // 2 for start, end in silence_ranges]
    return sorted(set(candidates))


def _rms_minima_times(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    top_k: int = 12,
    min_separation_ms: float = SONG_VALLEY_MIN_SEPARATION_MS,
    edge_guard_ms: float = SONG_VALLEY_EDGE_GUARD_MS,
) -> List[float]:
    """Times (seconds, chronological) of up to ``top_k`` low-energy valleys.

    Frames are taken quietest first, skipping any within ``min_separation_ms``
    of a valley already picked, so a quiet stretch yields one valley rather
    than a run of adjacent frames. Frames within ``edge_guard_ms`` of either end
    are excluded, unless the clip is too short to leave any.
    """
    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)
    if y.size < frame_length or frame_length <= 0 or hop_length <= 0:
        return [0.0]
    windows = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    if windows.size == 0:
        return [0.0]
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    if rms.size == 0:
        return []

    # Frame centres, so the edge guard and separation measure the valley itself.
    times = (np.arange(rms.size) * hop_length + frame_length / 2.0) / float(sr)
    duration = y.size / float(sr)
    guard = edge_guard_ms / 1000.0
    eligible = (times >= guard) & (times <= duration - guard)
    if not eligible.any():
        eligible = np.ones(rms.size, dtype=bool)

    separation = min_separation_ms / 1000.0
    picked: List[float] = []
    for idx in np.argsort(rms, kind="stable"):
        if not eligible[idx]:
            continue
        t = float(times[idx])
        if all(abs(t - other) >= separation for other in picked):
            picked.append(t)
            if len(picked) >= top_k:
                break
    return sorted(picked)


def analyze_song(path: Path) -> SongAnalysis:
    import librosa

    # Replaces librosa.load(path, sr=44100, mono=True) without its full-rate,
    # all-channel float copy: a WAV gives identical samples, and the
    # audio_tests MP3s the same tempo, beats and candidates.
    sr = SONG_FRAME_RATE
    y = load_song_samples(path)
    if y.size == 0:
        return SongAnalysis(candidates_ms=[], tempo=None, beat_times_ms=[])

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times_ms = [int(t * 1000) for t in beat_times]

    minima_times = _rms_minima_times(y, sr)
    candidates_ms = []
    seen = set()
    for t in minima_times:
        t_ms = int(t * 1000)
        if beat_times_ms:
            nearest = min(beat_times_ms, key=lambda b: abs(b - t_ms))
            if nearest not in seen:
                candidates_ms.append(nearest)
                seen.add(nearest)
        else:
            if t_ms not in seen:
                candidates_ms.append(t_ms)
                seen.add(t_ms)
    # librosa >= 0.10 returns tempo as a 1-element array, which NumPy 2 refuses
    # to convert with float() directly.
    tempo_values = np.atleast_1d(tempo) if tempo is not None else np.empty(0)
    return SongAnalysis(
        candidates_ms=candidates_ms,
        tempo=float(tempo_values[0]) if tempo_values.size else None,
        beat_times_ms=beat_times_ms,
    )


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401

        return True
    except Exception:
        return False


def transcribe_snippet(
    audio: AudioSegment, start_ms: int, end_ms: int, model_name: str = "base"
) -> str:
    try:
        import whisper
    except Exception:
        return "TRANSCRIPT_UNAVAILABLE"

    start_ms = max(0, start_ms)
    end_ms = max(start_ms, end_ms)
    snippet = audio[start_ms:end_ms]
    if len(snippet) == 0:
        return ""

    # tempfile honours the CLI session's private dir (inside --work-dir).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        snippet.export(tmp_path, format="wav")
        model = _WHISPER_MODELS.get(model_name)
        if model is None:
            model = whisper.load_model(model_name)
            _WHISPER_MODELS[model_name] = model
        result = model.transcribe(str(tmp_path), fp16=False)
        return str(result.get("text", "")).strip()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def choose_default_insertion(candidates: List[int], min_offset_ms: int = 30000) -> int:
    if not candidates:
        return min_offset_ms
    for cand in sorted(candidates):
        if cand >= min_offset_ms:
            return cand
    return sorted(candidates)[-1]


def build_candidate_payload(
    candidates: List[int], snippets: Dict[int, str]
) -> List[Dict[str, Any]]:
    payload = []
    for cand in candidates:
        payload.append({"ms": cand, "snippet": snippets.get(cand, "")})
    return payload


def build_snippets(
    audio: AudioSegment, candidates: List[int], max_snippets: int = 5
) -> Dict[int, str]:
    snippets: Dict[int, str] = {}
    for cand in candidates[:max_snippets]:
        snippet = transcribe_snippet(audio, cand - 15000, cand)
        snippets[cand] = snippet
    return snippets


def filter_candidates(
    candidates: List[int],
    audio_len_ms: int,
    min_offset_ms: int,
    end_buffer_ms: int,
) -> List[int]:
    if audio_len_ms <= 0:
        return candidates
    max_allowed = max(0, audio_len_ms - end_buffer_ms)
    return [cand for cand in candidates if min_offset_ms <= cand <= max_allowed]


def diarization_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401

        return True
    except Exception:
        return False


def diarize_speakers(path: Path) -> Optional[List[SpeakerSegment]]:
    """Run optional diarization and map the top two speakers to A/B."""
    try:
        from pyannote.audio import Pipeline
    except Exception:
        return None

    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("PYANNOTE_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    if not token:
        return None

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=token
    )
    diarization = pipeline(str(path))

    raw_segments: List[tuple[str, int, int]] = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        if end_ms > start_ms:
            raw_segments.append((str(label), start_ms, end_ms))

    if not raw_segments:
        return []

    totals: Dict[str, int] = {}
    for label, start_ms, end_ms in raw_segments:
        totals[label] = totals.get(label, 0) + (end_ms - start_ms)

    top_labels = sorted(totals.keys(), key=lambda l: totals[l], reverse=True)[:2]
    if not top_labels:
        return []

    label_map = {top_labels[0]: "A"}
    if len(top_labels) > 1:
        label_map[top_labels[1]] = "B"

    segments: List[SpeakerSegment] = []
    for label, start_ms, end_ms in raw_segments:
        speaker = label_map.get(label)
        if speaker:
            segments.append(
                SpeakerSegment(speaker=speaker, start_ms=start_ms, end_ms=end_ms)
            )

    return segments
