from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydub import AudioSegment


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

    data = json.dumps(payload).encode("utf-8")
    http_request = Request(
        request.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(http_request) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"TTS request failed with status {response.status}."
                )
            audio_bytes = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"TTS request failed with status {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"TTS request failed: {error.reason}") from error

    if not audio_bytes:
        raise RuntimeError("TTS request returned empty audio.")

    return AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
