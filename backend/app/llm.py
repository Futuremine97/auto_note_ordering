import base64
import io
import json
import urllib.request
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps

from .config import (
    LLM_PROVIDER,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_CHAT_ENDPOINT,
    LLM_MAX_CONTEXT_CHARS,
    LLM_MAX_IMAGE_CHARS,
    LLM_TIMEOUT,
    LLM_MAX_IMAGES,
    LLM_IMAGE_MAX_BYTES,
    LLM_IMAGE_MAX_DIM,
)


def build_ocr_context(records: Iterable, max_chars: int | None = None) -> Tuple[str, int]:
    limit = max_chars or LLM_MAX_CONTEXT_CHARS
    chunks: List[str] = []
    used = 0
    total = 0

    for record in records:
        text = (record.ocr_text or "").strip()
        if not text:
            continue
        if LLM_MAX_IMAGE_CHARS > 0 and len(text) > LLM_MAX_IMAGE_CHARS:
            text = text[:LLM_MAX_IMAGE_CHARS]
        page = record.page_number if record.page_number is not None else "미인식"
        header = f"[id:{record.id} page:{page}]\n"
        chunk = f"{header}{text}\n"
        remaining = limit - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
        used += 1
        if total >= limit:
            break

    return "\n".join(chunks).strip(), used


def build_image_payloads(records: Iterable, max_images: int | None = None) -> Tuple[List[dict], int]:
    limit = max_images or LLM_MAX_IMAGES
    payloads: List[dict] = []
    count = 0

    for record in records:
        if count >= limit:
            break
        if not getattr(record, "stored_filename", None):
            continue
        try:
            data_url = _encode_image_data_url(record)
        except Exception:
            continue
        payloads.append({"type": "image_url", "image_url": {"url": data_url}})
        count += 1

    return payloads, count


def _encode_image_data_url(record) -> str:
    data = _encode_image_bytes(record)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def _encode_image_bytes(record) -> bytes:
    from .config import UPLOAD_DIR

    path = UPLOAD_DIR / record.stored_filename
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _resize_to_limit(image, LLM_IMAGE_MAX_DIM)

    data = _encode_jpeg(image, quality=82)
    if len(data) > LLM_IMAGE_MAX_BYTES:
        data = _shrink_to_bytes(image, LLM_IMAGE_MAX_BYTES)
    return data


def _resize_to_limit(image: Image.Image, max_dim: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_dim / max(width, height))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BICUBIC)


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _shrink_to_bytes(image: Image.Image, target_bytes: int) -> bytes:
    quality = 80
    current = image
    data = _encode_jpeg(current, quality)
    while len(data) > target_bytes and min(current.size) > 256:
        width, height = current.size
        current = current.resize((int(width * 0.85), int(height * 0.85)), Image.BICUBIC)
        quality = max(60, quality - 5)
        data = _encode_jpeg(current, quality)
    return data


def call_llm(messages: List[dict]) -> str:
    provider = (LLM_PROVIDER or "").lower().strip()
    if not provider:
        raise RuntimeError("LLM_PROVIDER not configured")

    if provider == "mock":
        return _mock_response(messages)

    if provider not in {"openai", "openai_compatible"}:
        raise RuntimeError("Unsupported LLM_PROVIDER")

    if not LLM_API_KEY or not LLM_MODEL:
        raise RuntimeError("LLM_API_KEY or LLM_MODEL not configured")

    endpoint = LLM_CHAT_ENDPOINT
    if not endpoint:
        if not LLM_API_BASE:
            raise RuntimeError("LLM_API_BASE not configured")
        endpoint = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
        body = response.read().decode("utf-8")
        parsed = json.loads(body)

    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("LLM response missing content")
    return content


def _mock_response(messages: List[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    return (
        "LLM 설정이 아직 없어서 모의 응답을 반환합니다.\n\n"
        "요청 요약:\n"
        f"{last[:1000]}"
    )
