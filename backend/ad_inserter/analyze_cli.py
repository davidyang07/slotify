from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment
from pydub.silence import detect_silence

from ad_inserter import analysis, heuristic_config


def _detect_podcast_silences(audio: AudioSegment) -> List[Dict[str, int]]:
    # Constants come from the canonical `heuristic_offline_v1` profile in
    # config/heuristic_offline_v1.json, shared with the TypeScript scorer and
    # the ML baseline port. Values are unchanged (700 ms, dBFS - 16, -40).
    min_silence_len, threshold_offset, fallback_dbfs = heuristic_config.silence_params(
        heuristic_config.PRODUCT_PROFILE
    )
    silence_thresh = (
        audio.dBFS + threshold_offset if audio.dBFS != float("-inf") else fallback_dbfs
    )
    silence_ranges = detect_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    silences: List[Dict[str, int]] = []
    for start, end in silence_ranges:
        mid = (start + end) // 2
        silences.append(
            {
                "start_ms": int(start),
                "end_ms": int(end),
                "mid_ms": int(mid),
                "silence_ms": int(end - start),
            }
        )
    silences.sort(key=lambda entry: entry["mid_ms"])
    return silences


def _longest_silences(
    silences: List[Dict[str, int]], count: int
) -> List[Dict[str, int]]:
    """The ``count`` longest silences (ties go to the earlier one), in time order.

    Selecting by length rather than taking the first ``count`` in time keeps
    candidates spread across a long episode instead of bunched at its start.
    """
    if count <= 0:
        return []
    longest = sorted(silences, key=lambda entry: (-entry["silence_ms"], entry["mid_ms"]))
    return sorted(longest[:count], key=lambda entry: entry["mid_ms"])


def _build_snippets(
    audio: AudioSegment, candidates: List[Dict[str, int]], max_snippets: int
) -> Dict[int, str]:
    """Transcribe the context before the ``max_snippets`` longest-silence candidates."""
    if not analysis.whisper_available():
        return {}
    snippets: Dict[int, str] = {}
    for entry in _longest_silences(candidates, max_snippets):
        cand = entry["mid_ms"]
        text = analysis.transcribe_snippet(audio, cand - 15000, cand)
        snippets[cand] = text
    return snippets


def run() -> None:
    default_max_candidates, default_snippet_count = heuristic_config.candidate_limits(
        heuristic_config.PRODUCT_PROFILE
    )
    parser = argparse.ArgumentParser(description="Analyze audio for insertion points.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--mode", choices=["podcast", "song"], default="podcast")
    parser.add_argument("--max-candidates", type=int, default=default_max_candidates)
    parser.add_argument("--snippet-count", type=int, default=default_snippet_count)
    args = parser.parse_args()

    analysis.check_ffmpeg()
    audio = analysis.standardize_audio(analysis.load_audio(args.audio))

    candidates: List[Dict[str, int]] = []
    snippets: Dict[int, str] = {}
    tempo: Optional[float] = None
    beat_times_ms: List[int] = []

    if args.mode == "podcast":
        silences = _detect_podcast_silences(audio)
        candidates = _longest_silences(silences, args.max_candidates)
        snippets = _build_snippets(audio, candidates, args.snippet_count)
    else:
        song = analysis.analyze_song(args.audio)
        tempo = song.tempo
        beat_times_ms = song.beat_times_ms
        candidates = [
            {"mid_ms": int(ms), "silence_ms": 0, "start_ms": int(ms), "end_ms": int(ms)}
            for ms in song.candidates_ms[: max(0, args.max_candidates)]
        ]

    payload = {
        "mode": args.mode,
        "duration_ms": int(len(audio)),
        "candidates": candidates,
        "snippets": {str(key): value for key, value in snippets.items()},
        "tempo": tempo,
        "beat_times_ms": beat_times_ms,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    run()
