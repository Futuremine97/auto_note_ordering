import re
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    # HEIF/HEIC support is optional; if unavailable, OCR will fail for those files.
    pass
import pytesseract

from .config import TESSERACT_CMD, TESSERACT_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

PAGE_NUMBER_PATTERNS = [
    re.compile(r"(?:page|p)\s*[:\-]?\s*(\d{1,4})", re.IGNORECASE),
    re.compile(r"(?:페이지|쪽)\s*[:\-]?\s*(\d{1,4})", re.IGNORECASE),
    re.compile(r"[-–—]\s*(\d{1,4})\s*[-–—]"),
    re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE),
]

STRICT_PAGE_PATTERNS = [
    re.compile(r"(?:page|p)\s*[:\-]?\s*(\d{1,4})", re.IGNORECASE),
    re.compile(r"(?:페이지|쪽)\s*[:\-]?\s*(\d{1,4})", re.IGNORECASE),
    re.compile(r"[-–—]\s*(\d{1,4})\s*[-–—]"),
    re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE),
]

TOP_BIAS = 4.0
CENTER_BONUS = 1.0


def _open_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image)


def _otsu_threshold(image: Image.Image) -> int:
    histogram = image.histogram()
    total = sum(histogram)
    sum_total = 0
    for idx, count in enumerate(histogram):
        sum_total += idx * count

    sum_back = 0.0
    weight_back = 0.0
    max_variance = -1.0
    threshold = 180
    for idx, count in enumerate(histogram):
        weight_back += count
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += idx * count
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = idx
    return int(threshold)


def _preprocess_variants(image: Image.Image) -> List[Image.Image]:
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.MedianFilter(3))
    width, height = gray.size
    gray = gray.resize((width * 2, height * 2), Image.BICUBIC)

    threshold = _otsu_threshold(gray)
    binary = gray.point(lambda p: 255 if p > threshold else 0)
    binary = binary.filter(ImageFilter.SHARPEN)

    inverted = ImageOps.invert(gray)
    inv_threshold = _otsu_threshold(inverted)
    inverted_binary = inverted.point(lambda p: 255 if p > inv_threshold else 0)
    inverted_binary = inverted_binary.filter(ImageFilter.SHARPEN)

    return [binary, inverted_binary, gray]


def _crop_bands(
    image: Image.Image, ratios: Iterable[float]
) -> List[Tuple[str, str, Image.Image]]:
    width, height = image.size
    crops: List[Tuple[str, str, Image.Image]] = []
    for ratio in ratios:
        band = int(height * ratio)
        center_left = int(width * 0.2)
        center_right = int(width * 0.8)
        corner_width = int(width * 0.3)

        crops.append(("top", "full", image.crop((0, 0, width, band))))
        crops.append(("bottom", "full", image.crop((0, height - band, width, height))))
        crops.append(("top", "center", image.crop((center_left, 0, center_right, band))))
        crops.append(("bottom", "center", image.crop((center_left, height - band, center_right, height))))
        crops.append(("top", "left", image.crop((0, 0, corner_width, band))))
        crops.append(("top", "right", image.crop((width - corner_width, 0, width, band))))
        crops.append(("bottom", "left", image.crop((0, height - band, corner_width, height))))
        crops.append(("bottom", "right", image.crop((width - corner_width, height - band, width, height))))
    return crops


def _extract_best_number(data: dict) -> Tuple[Optional[int], float, str]:
    tokens: List[str] = []
    best_number = None
    best_score = 0.0
    for idx, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        if not word:
            continue
        tokens.append(word)
        digits = re.sub(r"\D", "", word)
        if not digits or len(digits) > 4:
            continue
        try:
            conf = float(data.get("conf", [])[idx])
        except (TypeError, ValueError, IndexError):
            continue
        if conf < 0:
            continue
        try:
            number = int(digits)
        except ValueError:
            continue
        if conf > best_score:
            best_score = conf
            best_number = number

    return best_number, best_score, " ".join(tokens)


def _ocr_page_candidate(image: Image.Image, psm: int, whitelist_digits: bool) -> Tuple[Optional[int], float]:
    config = f"--psm {psm}"
    if whitelist_digits:
        config += " -c tessedit_char_whitelist=0123456789"

    best_number = None
    best_score = 0.0
    for processed in _preprocess_variants(image):
        data = pytesseract.image_to_data(
            processed,
            lang=TESSERACT_LANG,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        number, score, text = _extract_best_number(data)
        if number is None:
            extracted = extract_page_number(text)
            if extracted is not None:
                number = extracted
                score = max(score, 50.0)
        if number is None:
            continue
        if score > best_score:
            best_score = score
            best_number = number

    return best_number, best_score


def run_ocr(image_path: str) -> str:
    image = _open_image(image_path)
    processed = _preprocess_variants(image)[0]
    text = pytesseract.image_to_string(
        processed, lang=TESSERACT_LANG, config="--psm 6"
    )
    return text


def detect_page_number_from_regions(image_path: str) -> Optional[int]:
    image = _open_image(image_path)
    crops = _crop_bands(image, ratios=[0.08, 0.12, 0.18])
    top_best = None
    bottom_best = None

    for position, region, crop in crops:
        for psm, whitelist in [(7, True), (8, True), (7, False)]:
            page, score = _ocr_page_candidate(crop, psm=psm, whitelist_digits=whitelist)
            if page is None:
                continue
            bonus = 0.0
            if position == "top":
                bonus += TOP_BIAS
            if region == "center":
                bonus += CENTER_BONUS
            candidate = {"page": page, "score": score + bonus}
            if position == "top":
                if top_best is None or candidate["score"] > top_best["score"]:
                    top_best = candidate
            else:
                if bottom_best is None or candidate["score"] > bottom_best["score"]:
                    bottom_best = candidate

    if top_best and bottom_best:
        return top_best["page"] if top_best["score"] >= bottom_best["score"] else bottom_best["page"]
    if top_best:
        return top_best["page"]
    if bottom_best:
        return bottom_best["page"]
    return None


def extract_page_number(text: str, strict: bool = False) -> Optional[int]:
    if strict:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        window = lines[:10] + lines[-10:] if lines else []
        text = "\n".join(window)
        patterns = STRICT_PAGE_PATTERNS
    else:
        patterns = PAGE_NUMBER_PATTERNS

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None
