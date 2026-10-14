from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from pydub import AudioSegment

from ad_inserter import http_client


@dataclass
class TTSRequest:
    voice_id: str
    text: str
    url: str
    model_id: Optional[str] = None
    output_format: Optional[str] = None


def synthesize_audio(request: TTSRequest) -> AudioSegment:
    payload = {
        "voiceId": request.voice_id,
        "text": request.text,
    }
    if request.model_id:
        payload["modelId"] = request.model_id
    if request.output_format:
        payload["outputFormat"] = request.output_format

    audio_bytes = http_client.post_json(request.url, payload, label="TTS")

    if not audio_bytes:
        raise RuntimeError("TTS request returned empty audio.")

    return AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
