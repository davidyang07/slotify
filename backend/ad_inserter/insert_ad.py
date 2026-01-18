from __future__ import annotations

"""CLI workflow for inserting two-speaker ads into conversation audio."""

import argparse
import json
import os
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydub import AudioSegment

from ad_inserter import analysis, llm, mix, tts


def _default_model(provider: str) -> str:
    return "gpt-4o-mini"


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


def _clone_voice(
    clone_url: str, name: str, samples: List[Path]
) -> Optional[str]:
    """Call the cloning endpoint and return the new voice ID."""
    files: List[Dict[str, str | bytes]] = []
    for sample in samples:
        if sample.exists():
            files.append(
                {
                    "name": "files",
                    "filename": sample.name,
                    "content_type": "audio/wav",
                    "data": sample.read_bytes(),
                }
            )

    if not files:
        return None

    boundary, body = _encode_multipart(fields={"name": name}, files=files)
    request = Request(
        clone_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Clone request failed with status {response.status}."
                )
            payload = json.loads(response.read().decode("utf-8"))
            voice_id = payload.get("voiceId")
            return str(voice_id) if voice_id else None
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Clone request failed with status {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Clone request failed: {error.reason}") from error


def _build_speaker_sample(
    audio: AudioSegment,
    segments: List[analysis.SpeakerSegment],
    speaker: str,
    max_ms: int = 30000,
) -> AudioSegment:
    """Concatenate speaker segments into a short sample clip."""
    combined = AudioSegment.empty()
    for segment in segments:
        if segment.speaker != speaker:
            continue
        combined += audio[segment.start_ms : segment.end_ms]
        if len(combined) >= max_ms:
            break
    return combined[:max_ms]


def _resolve_voice_id(
    speaker: str,
    explicit: Optional[str],
    env_key: str,
    default_voice: Optional[str],
) -> str:
    """Resolve the ElevenLabs voice ID for a speaker."""
    if explicit:
        return explicit
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    if default_voice:
        return default_voice
    raise RuntimeError(f"Missing ElevenLabs voice ID for speaker {speaker}.")


def _concat_ad_segments(
    segments: List[llm.AdSegment],
    voice_map: Dict[str, str],
    tts_url: str,
    tts_model_id: Optional[str],
    tts_output_format: Optional[str],
    pause_ms: int = 180,
) -> AudioSegment:
    """Generate and concatenate TTS audio for all ad segments."""
    combined = AudioSegment.empty()
    for idx, segment in enumerate(segments):
        voice_id = voice_map[segment.speaker]
        audio = analysis.standardize_audio(
            tts.synthesize_audio(
                tts.TTSRequest(
                    voice_id=voice_id,
                    text=segment.text,
                    url=tts_url,
                    model_id=tts_model_id,
                    output_format=tts_output_format,
                )
            )
        )
        if idx > 0 and pause_ms > 0:
            combined += AudioSegment.silent(duration=pause_ms, frame_rate=audio.frame_rate)
        combined += audio
    return combined


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Insert a two-speaker ad into a conversation."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--product-blurb", required=True)
    parser.add_argument("--ad-style", choices=["casual", "serious", "funny"], required=True)
    parser.add_argument(
        "--ad-mode", choices=["A_ONLY", "B_ONLY", "DUO"], required=True
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--voice-id-a")
    parser.add_argument("--voice-id-b")
    parser.add_argument("--clone-voices", action="store_true")
    parser.add_argument("--tts-url", default="http://localhost:3001/api/tts")
    parser.add_argument("--clone-url", default="http://localhost:3001/api/clone")
    parser.add_argument("--tts-model-id", default=None)
    parser.add_argument("--tts-output-format", default=None)
    parser.add_argument("--llm-provider", choices=["openai", "none"], default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--min-offset-ms", type=int, default=30000)
    parser.add_argument("--end-buffer-ms", type=int, default=15000)
    parser.add_argument("--duck-db", type=float, default=0.0)
    parser.add_argument("--crossfade-ms", type=int, default=250)
    parser.add_argument("--debug-dir", type=Path)

    args = parser.parse_args()

    analysis.check_ffmpeg()

    main_audio = analysis.standardize_audio(analysis.load_audio(args.input))
    diarization = analysis.diarize_speakers(args.input)
    diarization_available = bool(diarization)

    if not diarization_available and args.ad_mode == "DUO":
        raise RuntimeError(
            "Speaker diarization unavailable; DUO mode requires diarization."
        )

    candidates = analysis.detect_podcast_candidates(main_audio)
    candidates_for_prompt = analysis.filter_candidates(
        candidates,
        audio_len_ms=len(main_audio),
        min_offset_ms=args.min_offset_ms,
        end_buffer_ms=args.end_buffer_ms,
    )
    if not candidates_for_prompt:
        candidates_for_prompt = candidates

    snippets: Dict[int, str] = {}
    if analysis.whisper_available() and candidates_for_prompt:
        snippets = analysis.build_snippets(main_audio, candidates_for_prompt)

    chosen_ms = analysis.choose_default_insertion(
        candidates_for_prompt, min_offset_ms=args.min_offset_ms
    )
    if len(main_audio) > 0:
        chosen_ms = min(chosen_ms, len(main_audio) - 1)

    llm_choice = llm.generate_promo_and_choice(
        provider=args.llm_provider,
        model=args.llm_model or _default_model(args.llm_provider),
        product_name=args.product_name,
        product_desc=args.product_blurb,
        product_url=None,
        mode="podcast",
        candidates=analysis.build_candidate_payload(candidates_for_prompt, snippets),
    )
    if llm_choice.chosen_index is not None:
        if 0 <= llm_choice.chosen_index < len(candidates_for_prompt):
            chosen_ms = candidates_for_prompt[llm_choice.chosen_index]
            if chosen_ms < args.min_offset_ms:
                chosen_ms = analysis.choose_default_insertion(
                    candidates_for_prompt, min_offset_ms=args.min_offset_ms
                )
            max_allowed = max(0, len(main_audio) - args.end_buffer_ms)
            if chosen_ms > max_allowed:
                chosen_ms = analysis.choose_default_insertion(
                    candidates_for_prompt, min_offset_ms=args.min_offset_ms
                )

    ad_script = llm.generate_ad_script(
        provider=args.llm_provider,
        model=args.llm_model or _default_model(args.llm_provider),
        product_name=args.product_name,
        product_blurb=args.product_blurb,
        ad_style=args.ad_style,
        ad_mode=args.ad_mode,
    )

    segments = ad_script.segments
    if args.ad_mode == "A_ONLY":
        segments = [seg for seg in segments if seg.speaker == "A"] or [
            llm.AdSegment(speaker="A", text=args.product_blurb)
        ]
    elif args.ad_mode == "B_ONLY":
        segments = [seg for seg in segments if seg.speaker == "B"] or [
            llm.AdSegment(speaker="B", text=args.product_blurb)
        ]
    else:
        segments = [seg for seg in segments if seg.speaker in {"A", "B"}]
        if len(segments) < 2:
            segments = [
                llm.AdSegment(speaker="A", text=f"Quick shout-out to {args.product_name}."),
                llm.AdSegment(speaker="B", text=args.product_blurb),
            ]

    default_voice = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID")
    voice_id_a = _resolve_voice_id("A", args.voice_id_a, "ELEVENLABS_VOICE_ID_A", default_voice)
    voice_id_b = _resolve_voice_id("B", args.voice_id_b, "ELEVENLABS_VOICE_ID_B", default_voice)

    if args.clone_voices:
        if not diarization_available or not diarization:
            raise RuntimeError("Voice cloning requires speaker diarization.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            samples: Dict[str, Path] = {}
            for speaker in ["A", "B"]:
                sample_audio = _build_speaker_sample(main_audio, diarization, speaker)
                if len(sample_audio) == 0:
                    continue
                sample_path = tmp_path / f"speaker_{speaker.lower()}.wav"
                sample_audio.export(sample_path, format="wav")
                samples[speaker] = sample_path

            if "A" in samples:
                cloned = _clone_voice(args.clone_url, "Speaker A Clone", [samples["A"]])
                if cloned:
                    voice_id_a = cloned
            if "B" in samples:
                cloned = _clone_voice(args.clone_url, "Speaker B Clone", [samples["B"]])
                if cloned:
                    voice_id_b = cloned

    voice_map = {"A": voice_id_a, "B": voice_id_b}

    ad_audio = _concat_ad_segments(
        segments=segments,
        voice_map=voice_map,
        tts_url=args.tts_url,
        tts_model_id=args.tts_model_id,
        tts_output_format=args.tts_output_format,
    )

    if len(ad_audio) > 25000:
        raise RuntimeError("Ad audio is longer than 25 seconds; shorten the script.")

    context_audio = mix.context_window(main_audio, chosen_ms, window_ms=4000)
    target_lufs = mix.measure_lufs(context_audio)
    loudness_match = mix.match_loudness(ad_audio, target_lufs)

    ad_processed = loudness_match.matched.fade_in(200).fade_out(200)

    room_tone: Optional[AudioSegment] = None
    tone_start = max(0, chosen_ms - 600)
    room_tone = main_audio[tone_start:chosen_ms]
    ad_processed = mix.apply_room_tone(ad_processed, room_tone)

    merged = mix.insert_with_crossfade(
        main=main_audio,
        promo=ad_processed,
        insert_ms=chosen_ms,
        duck_db=args.duck_db,
        crossfade_ms=args.crossfade_ms,
    )

    if args.debug_dir:
        _ensure_debug_dir(args.debug_dir)
        context_audio.export(args.debug_dir / "context.wav", format="wav")
        ad_audio.export(args.debug_dir / "ad_raw.wav", format="wav")
        ad_processed.export(args.debug_dir / "ad_processed.wav", format="wav")
        debug_payload = {
            "chosen_insertion_ms": chosen_ms,
            "candidates_ms": candidates,
            "candidates_for_prompt_ms": candidates_for_prompt,
            "llm_prompt": ad_script.prompt[:500],
            "llm_output_text": ad_script.raw_text,
            "loudness_target_lufs": loudness_match.target_lufs,
            "loudness_ad_before": loudness_match.promo_before_lufs,
            "loudness_ad_after": loudness_match.promo_after_lufs,
            "diarization_available": diarization_available,
            "segments": [asdict(seg) for seg in segments],
        }
        (args.debug_dir / "debug.json").write_text(
            json.dumps(debug_payload, indent=2)
        )

    merged.export(args.out, format="mp3")
    print(f"Wrote output: {args.out}")


if __name__ == "__main__":
    run()
