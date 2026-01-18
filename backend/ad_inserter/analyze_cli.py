from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment
from pydub.silence import detect_silence

from ad_inserter import analysis


def _detect_podcast_silences(audio: AudioSegment) -> List[Dict[str, int]]:
    silence_thresh = audio.dBFS - 16 if audio.dBFS != float("-inf") else -40
    silence_ranges = detect_silence(
        audio, min_silence_len=700, silence_thresh=silence_thresh
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


def _build_snippets(
    audio: AudioSegment, candidates_ms: List[int], max_snippets: int
) -> Dict[int, str]:
    if not analysis.whisper_available():
        return {}
    snippets: Dict[int, str] = {}
    for cand in candidates_ms[:max_snippets]:
        text = analysis.transcribe_snippet(audio, cand - 15000, cand)
        snippets[cand] = text
    return snippets


def run() -> None:
    parser = argparse.ArgumentParser(description="Analyze audio for insertion points.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--mode", choices=["podcast", "song"], default="podcast")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--snippet-count", type=int, default=5)
    args = parser.parse_args()

    analysis.check_ffmpeg()
    audio = analysis.standardize_audio(analysis.load_audio(args.audio))

    candidates: List[Dict[str, int]] = []
    snippets: Dict[int, str] = {}
    tempo: Optional[float] = None
    beat_times_ms: List[int] = []

    if args.mode == "podcast":
        silences = _detect_podcast_silences(audio)
        candidates = silences[: max(0, args.max_candidates)]
        candidate_ms = [entry["mid_ms"] for entry in candidates]
        snippets = _build_snippets(audio, candidate_ms, args.snippet_count)
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
