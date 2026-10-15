from __future__ import annotations

"""CLI workflow for inserting two-speaker ads into conversation audio."""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from pydub import AudioSegment

from ad_inserter import analysis, http_client, llm, mix, runtime, tts

ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
VOICE_DELETE_TIMEOUT_S = 30.0
# Once a stop signal has arrived the deletes must fit in the caller's grace
# period before SIGKILL (5 s by default), two voices included.
VOICE_DELETE_SHUTDOWN_TIMEOUT_S = 2.0
DIARIZATION_HINT = (
    "install pyannote.audio and set HUGGINGFACE_TOKEN, PYANNOTE_TOKEN or HF_TOKEN"
)


def _ensure_debug_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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

    body = http_client.post_multipart(
        clone_url, fields={"name": name}, files=files, label="Clone"
    )
    payload = json.loads(body.decode("utf-8"))
    voice_id = payload.get("voiceId") if isinstance(payload, dict) else None
    return str(voice_id) if voice_id else None


def _delete_cloned_voice(voice_id: str) -> None:
    """Delete a voice this run cloned, so clones do not use up voice slots.

    ``/api/clone`` creates a permanent ElevenLabs voice, so the cleanup goes to
    the ElevenLabs API directly with the inherited ``ELEVENLABS_API_KEY``.
    Failures are logged, never raised: the ad has already been rendered.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        print(
            f"warning: ELEVENLABS_API_KEY not set; cloned voice {voice_id} "
            "was not deleted.",
            file=sys.stderr,
        )
        return
    try:
        http_client.send(
            f"{ELEVENLABS_VOICES_URL}/{quote(voice_id, safe='')}",
            label="Voice delete",
            headers={"xi-api-key": api_key},
            method="DELETE",
            internal_auth=False,
            timeout_s=(
                VOICE_DELETE_SHUTDOWN_TIMEOUT_S
                if runtime.termination_requested()
                else VOICE_DELETE_TIMEOUT_S
            ),
        )
    except Exception as error:  # noqa: BLE001 - cleanup must not fail the run
        print(
            f"warning: failed to delete cloned voice {voice_id}: {error}",
            file=sys.stderr,
        )


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
    parser.add_argument("--crossfade-ms", type=int, default=250)
    parser.add_argument("--debug-dir", type=Path)
    runtime.add_work_dir_argument(parser)

    args = parser.parse_args()
    try:
        args.llm_model = llm.resolve_model(args.llm_provider, args.llm_model)
    except llm.ModelNotAllowed as error:
        parser.error(str(error))
    if args.clone_voices and not os.getenv("ELEVENLABS_API_KEY", "").strip():
        # Without it the permanent voices this run clones could not be deleted.
        parser.error(
            "--clone-voices requires ELEVENLABS_API_KEY to delete the cloned voices."
        )

    with runtime.cli_session(args.work_dir):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    analysis.check_ffmpeg()

    # Full quality: this copy is what the ad is spliced into.
    main_audio = analysis.standardize_audio(analysis.load_audio(args.input))
    diarization = analysis.diarize_speakers(args.input)
    diarization_available = bool(diarization)

    if not diarization_available and args.ad_mode == "DUO":
        raise RuntimeError(
            "Speaker diarization unavailable; DUO mode requires diarization "
            f"({DIARIZATION_HINT})."
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
        model=args.llm_model,
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
        model=args.llm_model,
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
    # The fallbacks above carry the product text, which is not length-capped.
    segments = llm.cap_segments(segments)

    # Only the speakers the ad mode actually voices need a voice (A_ONLY must
    # not fail for want of a speaker B voice), and only they are cloned.
    needed_speakers = sorted({seg.speaker for seg in segments})
    explicit_voices = {"A": args.voice_id_a, "B": args.voice_id_b}
    default_voice = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID")
    voice_map: Dict[str, str] = {}

    # Voices cloned by this run; deleted once the ad is rendered (or on failure).
    created_voice_ids: List[str] = []
    try:
        if args.clone_voices:
            if not diarization_available or not diarization:
                raise RuntimeError(
                    f"Voice cloning requires speaker diarization ({DIARIZATION_HINT})."
                )
            # tempfile honours the CLI session's private dir (inside --work-dir).
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                for speaker in needed_speakers:
                    sample_audio = _build_speaker_sample(
                        main_audio, diarization, speaker
                    )
                    if len(sample_audio) == 0:
                        continue
                    sample_path = tmp_path / f"speaker_{speaker.lower()}.wav"
                    sample_audio.export(sample_path, format="wav")
                    # Shielded: a stop signal during the request waits for its
                    # voice ID, so the voice is recorded and deleted below.
                    with runtime.shielded():
                        cloned = _clone_voice(
                            args.clone_url, f"Speaker {speaker} Clone", [sample_path]
                        )
                        if cloned:
                            created_voice_ids.append(cloned)
                            voice_map[speaker] = cloned

        # Speakers not cloned (cloning off, or no usable sample) use the
        # configured voices.
        for speaker in needed_speakers:
            if speaker not in voice_map:
                voice_map[speaker] = _resolve_voice_id(
                    speaker,
                    explicit_voices[speaker],
                    f"ELEVENLABS_VOICE_ID_{speaker}",
                    default_voice,
                )

        ad_audio = _concat_ad_segments(
            segments=segments,
            voice_map=voice_map,
            tts_url=args.tts_url,
            tts_model_id=args.tts_model_id,
            tts_output_format=args.tts_output_format,
        )
    finally:
        # Shielded: a stop signal must not cut the deletes short.
        with runtime.shielded():
            for voice_id in created_voice_ids:
                _delete_cloned_voice(voice_id)

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
