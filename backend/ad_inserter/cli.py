from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydub import AudioSegment

from ad_inserter import analysis, llm, mix, tts


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


def _encode_multipart(
    fields: Dict[str, str],
    files: List[Dict[str, str | bytes]],
) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    body = bytearray()

    def _add_line(line: str) -> None:
        body.extend(line.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in fields.items():
        _add_line(f"--{boundary}")
        _add_line(f'Content-Disposition: form-data; name="{name}"')
        _add_line("")
        _add_line(value)

    for file_info in files:
        name = str(file_info["name"])
        filename = str(file_info["filename"])
        content_type = str(file_info["content_type"])
        data = bytes(file_info["data"])

        _add_line(f"--{boundary}")
        _add_line(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
        )
        _add_line(f"Content-Type: {content_type}")
        _add_line("")
        body.extend(data)
        body.extend(b"\r\n")

    _add_line(f"--{boundary}--")
    return boundary, bytes(body)


def _merge_with_api(
    merge_url: str,
    main_path: Path,
    insert_audio: AudioSegment,
    insert_at_ms: int,
    crossfade: float,
    pause: float,
    out_path: Path,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        insert_audio.export(tmp_path, format="mp3")
        boundary, body = _encode_multipart(
            fields={
                "insertAt": f"{insert_at_ms / 1000.0:.3f}",
                "crossfade": str(crossfade),
                "pause": str(pause),
            },
            files=[
                {
                    "name": "audio",
                    "filename": main_path.name,
                    "content_type": "audio/mpeg",
                    "data": main_path.read_bytes(),
                },
                {
                    "name": "insert",
                    "filename": tmp_path.name,
                    "content_type": "audio/mpeg",
                    "data": tmp_path.read_bytes(),
                },
            ],
        )
        request = Request(
            merge_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Merge request failed with status {response.status}."
                )
            out_path.write_bytes(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Merge request failed with status {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Merge request failed: {error.reason}") from error
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

def run() -> None:
    parser = argparse.ArgumentParser(description="Insert a promo into audio.")
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--product-desc", required=True)
    parser.add_argument("--product-url", default=None)
    parser.add_argument("--mode", choices=["podcast", "song"], default="podcast")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--llm-provider", choices=["openai", "none"], default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--tts-url", default="http://localhost:3001/api/tts")
    parser.add_argument("--tts-model-id", default=None)
    parser.add_argument("--tts-output-format", default=None)
    parser.add_argument("--merge-url", default="http://localhost:3001/api/merge")
    parser.add_argument("--merge-crossfade", type=float, default=0.08)
    parser.add_argument("--merge-pause", type=float, default=0.2)
    parser.add_argument("--duck-db", type=float, default=0.0)

    args = parser.parse_args()

    analysis.check_ffmpeg()

    main_audio = analysis.standardize_audio(analysis.load_audio(args.main))

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

    min_insert_ms = 30000
    chosen_ms = analysis.choose_default_insertion(
        candidates, min_offset_ms=min_insert_ms
    )
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

    promo_audio = analysis.standardize_audio(
        tts.synthesize_audio(
            tts.TTSRequest(
                voice_id=args.voice_id,
                text=llm_result.promo_text,
                url=args.tts_url,
                model_id=args.tts_model_id,
                output_format=args.tts_output_format,
            )
        )
    )

    if len(promo_audio) > 20000:
        raise RuntimeError("Promo audio is longer than 20 seconds; please shorten it.")

    if args.mode == "podcast" and llm_result.chosen_index is not None:
        if 0 <= llm_result.chosen_index < len(candidates):
            chosen_ms = candidates[llm_result.chosen_index]
            if chosen_ms < min_insert_ms:
                chosen_ms = analysis.choose_default_insertion(
                    candidates, min_offset_ms=min_insert_ms
                )

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

    if args.debug_dir:
        _ensure_debug_dir(args.debug_dir)
        context_audio.export(args.debug_dir / "context.wav", format="wav")
        promo_audio.export(args.debug_dir / "promo_raw.wav", format="wav")
        promo_processed.export(args.debug_dir / "promo_matched.wav", format="wav")
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

    _merge_with_api(
        merge_url=args.merge_url,
        main_path=args.main,
        insert_audio=promo_processed,
        insert_at_ms=chosen_ms,
        crossfade=args.merge_crossfade,
        pause=args.merge_pause,
        out_path=args.out,
    )

    if args.debug_dir:
        merged_audio = analysis.standardize_audio(analysis.load_audio(args.out))
        preview_start = max(0, chosen_ms - 5000)
        preview_end = min(len(merged_audio), chosen_ms + 5000)
        merged_audio[preview_start:preview_end].export(
            args.debug_dir / "merged_preview.wav", format="wav"
        )

    print(f"Wrote output: {args.out}")
    print(f"Promo text: {llm_result.promo_text}")


if __name__ == "__main__":
    run()
