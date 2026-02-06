import json
import mimetypes
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError

from .config import (
    LLM_API_BASE,
    LLM_API_KEY,
    STT_API_BASE,
    STT_API_KEY,
    STT_LANGUAGE,
    STT_MODEL,
    STT_ENDPOINT,
    STT_TIMEOUT,
    STT_MAX_BYTES,
    STT_AUTO_COMPRESS,
    STT_COMPRESS_FORMAT,
    STT_COMPRESS_BITRATE,
    STT_COMPRESS_SAMPLE_RATE,
    STT_COMPRESS_CHANNELS,
)


def _resolve_endpoint() -> Tuple[str, str, str, str]:
    api_base = STT_API_BASE or LLM_API_BASE
    api_key = STT_API_KEY or LLM_API_KEY
    model = STT_MODEL or "whisper-1"
    language = STT_LANGUAGE or "en"

    if not api_base:
        raise RuntimeError("STT_API_BASE가 설정되지 않았습니다.")
    if not api_key:
        raise RuntimeError("STT_API_KEY가 설정되지 않았습니다.")

    if STT_ENDPOINT:
        endpoint = STT_ENDPOINT
    else:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            endpoint = f"{base}/audio/transcriptions"
        else:
            endpoint = f"{base}/v1/audio/transcriptions"

    return endpoint, api_key, model, language


def _build_multipart(fields: dict, file_field: str, filename: str, content_type: str, payload: bytes):
    boundary = f"----stt-{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(str(value).encode())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(payload)
    body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())
    return boundary, bytes(body)


def _compress_audio(payload: bytes, filename: str) -> Tuple[bytes, str]:
    suffix = Path(filename).suffix or ".bin"
    out_suffix = f".{STT_COMPRESS_FORMAT}"

    in_fd, in_path = tempfile.mkstemp(suffix=suffix)
    out_fd, out_path = tempfile.mkstemp(suffix=out_suffix)
    os.close(in_fd)
    os.close(out_fd)

    try:
        with open(in_path, "wb") as infile:
            infile.write(payload)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_path,
            "-ac",
            str(STT_COMPRESS_CHANNELS),
            "-ar",
            str(STT_COMPRESS_SAMPLE_RATE),
            "-b:a",
            STT_COMPRESS_BITRATE,
            out_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            error_msg = result.stderr.decode(errors="ignore").strip()
            raise RuntimeError(f"오디오 자동 압축 실패: {error_msg or 'ffmpeg 오류'}")

        with open(out_path, "rb") as outfile:
            compressed = outfile.read()
    except FileNotFoundError as exc:
        raise RuntimeError("서버에 ffmpeg가 없어 자동 압축을 수행할 수 없습니다.") from exc
    finally:
        for path in (in_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass

    return compressed, f"audio/{STT_COMPRESS_FORMAT}"


def _request_transcription(
    endpoint: str,
    api_key: str,
    model: str,
    language: str,
    payload: bytes,
    filename: str,
    content_type: str,
) -> str:
    boundary, body = _build_multipart(
        fields={"model": model, "language": language},
        file_field="file",
        filename=filename,
        content_type=content_type,
        payload=payload,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    req = urlrequest.Request(endpoint, data=body, headers=headers)
    with urlrequest.urlopen(req, timeout=STT_TIMEOUT) as resp:
        return resp.read().decode()


def transcribe_audio(payload: bytes, filename: str, content_type: Optional[str] = None) -> str:
    endpoint, api_key, model, language = _resolve_endpoint()
    compressed = False

    if STT_MAX_BYTES and len(payload) > STT_MAX_BYTES and STT_AUTO_COMPRESS:
        payload, content_type = _compress_audio(payload, filename)
        compressed = True

    if STT_MAX_BYTES and len(payload) > STT_MAX_BYTES:
        max_mb = STT_MAX_BYTES / (1024 * 1024)
        raise RuntimeError(f"오디오 파일이 너무 큽니다. {max_mb:.0f}MB 이하로 줄여주세요.")

    guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    while True:
        try:
            data = _request_transcription(endpoint, api_key, model, language, payload, filename, guessed_type)
            break
        except HTTPError as exc:
            # API가 용량 초과를 반환하면 자동 압축 후 1회 재시도
            if exc.code == 413 and STT_AUTO_COMPRESS and not compressed:
                payload, guessed_type = _compress_audio(payload, filename)
                compressed = True
                continue
            try:
                error_body = exc.read().decode()
            except Exception:
                error_body = ""
            if exc.code == 413:
                raise RuntimeError("오디오 파일이 너무 큽니다. 25MB 이하로 줄여주세요.") from exc
            if exc.code == 401:
                raise RuntimeError("STT API 인증에 실패했습니다. API 키를 확인해주세요.") from exc
            if exc.code == 429:
                raise RuntimeError("STT API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.") from exc
            message = error_body.strip() or "STT API 요청에 실패했습니다."
            raise RuntimeError(f"STT 요청 실패 ({exc.code}): {message}") from exc
    except Exception as exc:
        raise RuntimeError("STT 요청 중 오류가 발생했습니다.") from exc

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return data.strip()

    if isinstance(parsed, dict) and "text" in parsed:
        return parsed["text"].strip()

    return data.strip()
