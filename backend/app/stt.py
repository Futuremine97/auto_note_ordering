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
    STT_AUTO_SPLIT,
    STT_CHUNK_SECONDS,
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


def _compress_audio(
    payload: bytes,
    filename: str,
    bitrate: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> Tuple[bytes, str]:
    suffix = Path(filename).suffix or ".bin"
    out_suffix = f".{STT_COMPRESS_FORMAT}"

    bitrate = bitrate or STT_COMPRESS_BITRATE
    sample_rate = sample_rate or STT_COMPRESS_SAMPLE_RATE
    channels = channels or STT_COMPRESS_CHANNELS

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
            str(channels),
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
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


def _parse_bitrate(bitrate: str) -> int:
    value = bitrate.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    return int(value)


def _estimate_segment_seconds(bitrate: str) -> int:
    if not STT_MAX_BYTES:
        return max(15, STT_CHUNK_SECONDS)
    bps = _parse_bitrate(bitrate)
    if bps <= 0:
        return max(15, STT_CHUNK_SECONDS)
    max_seconds = int((STT_MAX_BYTES * 0.9 * 8) / bps)
    return max(15, min(STT_CHUNK_SECONDS, max_seconds))


def _compress_to_limit(payload: bytes, filename: str) -> Tuple[bytes, str, bool]:
    if not STT_MAX_BYTES:
        return payload, mimetypes.guess_type(filename)[0] or "application/octet-stream", True

    bitrates = [
        STT_COMPRESS_BITRATE,
        "24k",
        "16k",
        "12k",
        "8k",
    ]
    sample_rates = [
        STT_COMPRESS_SAMPLE_RATE,
        12000,
        8000,
    ]

    best_payload = payload
    best_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    tried = set()
    for bitrate in bitrates:
        for sample_rate in sample_rates:
            key = (bitrate, sample_rate)
            if key in tried:
                continue
            tried.add(key)
            compressed, content_type = _compress_audio(
                payload,
                filename,
                bitrate=bitrate,
                sample_rate=sample_rate,
            )
            if len(compressed) < len(best_payload):
                best_payload = compressed
                best_type = content_type
            if len(compressed) <= STT_MAX_BYTES:
                return compressed, content_type, True

    return best_payload, best_type, len(best_payload) <= STT_MAX_BYTES


def _split_audio_to_chunks(
    payload: bytes,
    filename: str,
    bitrate: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> List[Tuple[bytes, str, str]]:
    bitrate = bitrate or STT_COMPRESS_BITRATE
    sample_rate = sample_rate or STT_COMPRESS_SAMPLE_RATE
    channels = channels or STT_COMPRESS_CHANNELS
    segment_seconds = _estimate_segment_seconds(bitrate)

    suffix = Path(filename).suffix or ".bin"
    out_suffix = f".{STT_COMPRESS_FORMAT}"
    temp_dir = tempfile.TemporaryDirectory()
    in_fd, in_path = tempfile.mkstemp(suffix=suffix)
    os.close(in_fd)

    try:
        with open(in_path, "wb") as infile:
            infile.write(payload)

        out_pattern = os.path.join(temp_dir.name, f"chunk_%03d{out_suffix}")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_path,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            out_pattern,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            error_msg = result.stderr.decode(errors="ignore").strip()
            raise RuntimeError(f"오디오 분할 실패: {error_msg or 'ffmpeg 오류'}")

        chunks = []
        for path in sorted(Path(temp_dir.name).glob(f"chunk_*{out_suffix}")):
            data = path.read_bytes()
            if not data:
                continue
            chunks.append((data, f"audio/{STT_COMPRESS_FORMAT}", path.name))
    except FileNotFoundError as exc:
        raise RuntimeError("서버에 ffmpeg가 없어 오디오 분할을 수행할 수 없습니다.") from exc
    finally:
        try:
            os.remove(in_path)
        except OSError:
            pass
        temp_dir.cleanup()

    if not chunks:
        raise RuntimeError("오디오 분할 결과가 없습니다.")
    return chunks


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
        payload, content_type, fits = _compress_to_limit(payload, filename)
        compressed = True
        if not fits:
            max_mb = STT_MAX_BYTES / (1024 * 1024)
            raise RuntimeError(
                f"오디오 파일이 너무 큽니다. {max_mb:.0f}MB 이하로 줄여주세요."
            )

    if STT_MAX_BYTES and len(payload) > STT_MAX_BYTES:
        if STT_AUTO_SPLIT:
            chunks = _split_audio_to_chunks(payload, filename)
            results = []
            for chunk_payload, chunk_type, chunk_name in chunks:
                data = _request_transcription(
                    endpoint,
                    api_key,
                    model,
                    language,
                    chunk_payload,
                    chunk_name,
                    chunk_type,
                )
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    results.append(data.strip())
                else:
                    if isinstance(parsed, dict) and "text" in parsed:
                        results.append(parsed["text"].strip())
                    else:
                        results.append(data.strip())
            return "\n".join([line for line in results if line])
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
                payload, guessed_type, fits = _compress_to_limit(payload, filename)
                compressed = True
                if not fits:
                    max_mb = STT_MAX_BYTES / (1024 * 1024)
                    raise RuntimeError(
                        f"오디오 파일이 너무 큽니다. {max_mb:.0f}MB 이하로 줄여주세요."
                    ) from exc
                continue
            if exc.code == 413 and STT_AUTO_SPLIT:
                chunks = _split_audio_to_chunks(payload, filename)
                results = []
                for chunk_payload, chunk_type, chunk_name in chunks:
                    chunk_data = _request_transcription(
                        endpoint,
                        api_key,
                        model,
                        language,
                        chunk_payload,
                        chunk_name,
                        chunk_type,
                    )
                    try:
                        parsed = json.loads(chunk_data)
                    except json.JSONDecodeError:
                        results.append(chunk_data.strip())
                    else:
                        if isinstance(parsed, dict) and "text" in parsed:
                            results.append(parsed["text"].strip())
                        else:
                            results.append(chunk_data.strip())
                return "\n".join([line for line in results if line])
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
