import json
from typing import Optional, Tuple
from urllib import request as urlrequest

from .config import (
    LLM_API_BASE,
    LLM_API_KEY,
    TTS_API_BASE,
    TTS_API_KEY,
    TTS_MODEL,
    TTS_VOICE,
    TTS_RESPONSE_FORMAT,
    TTS_SPEED,
    TTS_ENDPOINT,
    TTS_TIMEOUT,
)


_FORMAT_MEDIA = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _resolve_endpoint() -> Tuple[str, str]:
    api_base = TTS_API_BASE or LLM_API_BASE
    api_key = TTS_API_KEY or LLM_API_KEY

    if not api_base:
        raise RuntimeError("TTS_API_BASE가 설정되지 않았습니다.")
    if not api_key:
        raise RuntimeError("TTS_API_KEY가 설정되지 않았습니다.")

    if TTS_ENDPOINT:
        endpoint = TTS_ENDPOINT
    else:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            endpoint = f"{base}/audio/speech"
        else:
            endpoint = f"{base}/v1/audio/speech"

    return endpoint, api_key


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
) -> Tuple[bytes, str]:
    endpoint, api_key = _resolve_endpoint()
    model = TTS_MODEL or "tts-1"
    voice_value = (voice or TTS_VOICE or "alloy").strip()
    fmt = (response_format or TTS_RESPONSE_FORMAT or "mp3").strip().lower()
    speed_value = speed if speed is not None else TTS_SPEED

    payload = {
        "model": model,
        "input": text,
        "voice": voice_value,
        "response_format": fmt,
    }
    if speed_value:
        payload["speed"] = speed_value

    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    req = urlrequest.Request(endpoint, data=data, headers=headers)
    with urlrequest.urlopen(req, timeout=TTS_TIMEOUT) as resp:
        audio_bytes = resp.read()

    media_type = _FORMAT_MEDIA.get(fmt, "application/octet-stream")
    return audio_bytes, media_type
