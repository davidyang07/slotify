from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import which


@dataclass
class SongAnalysis:
    candidates_ms: List[int]
    tempo: Optional[float]
    beat_times_ms: List[int]


def check_ffmpeg() -> None:
    if which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it is in your PATH."
        )


def load_audio(path: Path) -> AudioSegment:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return AudioSegment.from_file(path)


def standardize_audio(audio: AudioSegment) -> AudioSegment:
    return audio.set_frame_rate(44100).set_channels(2)


def detect_podcast_candidates(audio: AudioSegment) -> List[int]:
    silence_thresh = audio.dBFS - 16 if audio.dBFS != float("-inf") else -40
    silence_ranges = detect_silence(
        audio, min_silence_len=500, silence_thresh=silence_thresh
    )
    candidates = [(start + end) // 2 for start, end in silence_ranges]
    return sorted(set(candidates))


def _rms_minima_times(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    top_k: int = 12,
) -> List[float]:
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
    idx = np.argsort(rms)[:top_k]
    times = (idx * hop_length) / float(sr)
    return times.tolist()


def analyze_song(path: Path) -> SongAnalysis:
    import librosa

    y, sr = librosa.load(path, sr=44100, mono=True)
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
    return SongAnalysis(
        candidates_ms=candidates_ms,
        tempo=float(tempo) if tempo is not None else None,
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        snippet.export(tmp_path, format="wav")
        model = whisper.load_model(model_name)
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
    return candidates[0]


def build_candidate_payload(
    candidates: List[int], snippets: Dict[int, str]
) -> List[Dict[str, Any]]:
    payload = []
    for cand in candidates:
        payload.append({"ms": cand, "snippet": snippets.get(cand, "")})
    return payload
