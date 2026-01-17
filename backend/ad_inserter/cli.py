from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment

from ad_inserter import analysis, llm, mix


def _default_model(provider: str) -> str:
    return "gpt-4o-mini"


def _build_snippets(
    audio: AudioSegment, candidates: List[int], max_snippets: int = 5
) -> Dict[int, str]:
    snippets: Dict[int, str] = {}
    for cand in candidates[:max_snippets]:
        snippet = analysis.transcribe_snippet(audio, cand - 15000, cand)
        snippets[cand] = snippet
    return snippets


def _ensure_debug_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run() -> None:
    parser = argparse.ArgumentParser(description="Insert a promo into audio.")
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--promo-audio", required=True, type=Path)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--product-desc", required=True)
    parser.add_argument("--product-url", default=None)
    parser.add_argument("--mode", choices=["podcast", "song"], default="podcast")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--llm-provider", choices=["openai"], default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--duck-db", type=float, default=0.0)

    args = parser.parse_args()

    analysis.check_ffmpeg()

    main_audio = analysis.standardize_audio(analysis.load_audio(args.main))
    promo_audio = analysis.standardize_audio(analysis.load_audio(args.promo_audio))

    if len(promo_audio) > 20000:
        raise RuntimeError("Promo audio is longer than 20 seconds; please shorten it.")

    candidates: List[int] = []
    tempo: Optional[float] = None
    beat_times_ms: List[int] = []
    snippets: Dict[int, str] = {}

    if args.mode == "podcast":
        candidates = analysis.detect_podcast_candidates(main_audio)
        if analysis.whisper_available() and candidates:
            snippets = _build_snippets(main_audio, candidates)
    else:
        song = analysis.analyze_song(args.main)
        candidates = song.candidates_ms
        tempo = song.tempo
        beat_times_ms = song.beat_times_ms

    chosen_ms = analysis.choose_default_insertion(candidates)
    if len(main_audio) > 0:
        chosen_ms = min(chosen_ms, len(main_audio) - 1)
    llm_result = llm.generate_promo_and_choice(
        provider=args.llm_provider,
        model=args.llm_model or _default_model(args.llm_provider),
        product_name=args.product_name,
        product_desc=args.product_desc,
        product_url=args.product_url,
        mode=args.mode,
        candidates=analysis.build_candidate_payload(candidates, snippets),
    )

    if args.mode == "podcast" and llm_result.chosen_index is not None:
        if 0 <= llm_result.chosen_index < len(candidates):
            chosen_ms = candidates[llm_result.chosen_index]

    transcript_before = snippets.get(chosen_ms, "TRANSCRIPT_UNAVAILABLE")
    if analysis.whisper_available() and not transcript_before and args.mode == "podcast":
        transcript_before = analysis.transcribe_snippet(
            main_audio, chosen_ms - 15000, chosen_ms
        )

    context_audio = mix.context_window(main_audio, chosen_ms, window_ms=4000)
    target_lufs = mix.measure_lufs(context_audio)
    loudness_match = mix.match_loudness(promo_audio, target_lufs)

    promo_processed = loudness_match.matched.fade_in(250).fade_out(250)

    room_tone: Optional[AudioSegment] = None
    if args.mode == "podcast":
        tone_start = max(0, chosen_ms - 600)
        room_tone = main_audio[tone_start:chosen_ms]

    promo_processed = mix.apply_room_tone(promo_processed, room_tone)

    merged = mix.insert_with_crossfade(
        main_audio, promo_processed, chosen_ms, duck_db=args.duck_db
    )

    if args.debug_dir:
        _ensure_debug_dir(args.debug_dir)
        context_audio.export(args.debug_dir / "context.wav", format="wav")
        promo_audio.export(args.debug_dir / "promo_raw.wav", format="wav")
        promo_processed.export(args.debug_dir / "promo_matched.wav", format="wav")
        preview_start = max(0, chosen_ms - 5000)
        preview_end = min(len(merged), chosen_ms + 5000)
        merged[preview_start:preview_end].export(
            args.debug_dir / "merged_preview.wav", format="wav"
        )
        chosen_beat_index = None
        if beat_times_ms and chosen_ms in beat_times_ms:
            chosen_beat_index = beat_times_ms.index(chosen_ms)

        debug_payload = {
            "chosen_insertion_ms": chosen_ms,
            "candidates_ms": candidates,
            "mode": args.mode,
            "tempo": tempo,
            "beat_times_count": len(beat_times_ms),
            "beat_times_ms": beat_times_ms,
            "chosen_beat_index": chosen_beat_index,
            "loudness_target_lufs": loudness_match.target_lufs,
            "loudness_promo_before": loudness_match.promo_before_lufs,
            "loudness_promo_after": loudness_match.promo_after_lufs,
            "transcript_snippet_before": transcript_before,
            "llm_prompt": llm_result.prompt[:500],
            "llm_output_text": llm_result.raw_text,
        }
        (args.debug_dir / "debug.json").write_text(
            json.dumps(debug_payload, indent=2)
        )

    if args.dry_run:
        print(
            f"Dry run: insertion at {chosen_ms}ms. Rationale: {llm_result.rationale}"
        )
        print(f"Promo text: {llm_result.promo_text}")
        return

    merged.export(args.out, format="mp3")
    print(f"Wrote output: {args.out}")
    print(f"Promo text: {llm_result.promo_text}")


if __name__ == "__main__":
    run()
