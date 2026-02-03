import re
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps
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


def _preprocess(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(3))
    width, height = image.size
    image = image.resize((width * 2, height * 2))
    # Simple binarization to make digits pop.
    image = image.point(lambda p: 255 if p > 180 else 0)
    return image


def _crop_bands(
    image: Image.Image, ratios: Iterable[float]
) -> List[Tuple[str, Image.Image]]:
    width, height = image.size
    crops: List[Tuple[str, Image.Image]] = []
    for ratio in ratios:
        band = int(height * ratio)
        crops.append(("top", image.crop((0, 0, width, band))))
        crops.append(("bottom", image.crop((0, height - band, width, height))))
    return crops


def _ocr_with_confidence(image: Image.Image, psm: int) -> Tuple[str, float]:
    processed = _preprocess(image)
    data = pytesseract.image_to_data(
        processed,
        lang=TESSERACT_LANG,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    tokens: List[str] = []
    confidences: List[float] = []
    digit_confidences: List[float] = []
    for idx, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        if not word:
            continue
        tokens.append(word)
        try:
            conf = float(data.get("conf", [])[idx])
        except (TypeError, ValueError, IndexError):
            continue
        if conf < 0:
            continue
        confidences.append(conf)
        if any(ch.isdigit() for ch in word):
            digit_confidences.append(conf)

    text = " ".join(tokens)
    if digit_confidences:
        score = sum(digit_confidences) / len(digit_confidences)
    elif confidences:
        score = sum(confidences) / len(confidences)
    else:
        score = 0.0
    return text, score


def run_ocr(image_path: str) -> str:
    image = Image.open(image_path)
    processed = _preprocess(image)
    text = pytesseract.image_to_string(
        processed, lang=TESSERACT_LANG, config="--psm 6"
    )
    return text


def detect_page_number_from_regions(image_path: str) -> Optional[int]:
    image = Image.open(image_path)
    crops = _crop_bands(image, ratios=[0.12, 0.18])
    top_best = None
    bottom_best = None

    for position, crop in crops:
        text, score = _ocr_with_confidence(crop, psm=7)
        page = extract_page_number(text)
        if page is None:
            continue
        candidate = {"page": page, "score": score}
        if position == "top":
            if top_best is None or score > top_best["score"]:
                top_best = candidate
        else:
            if bottom_best is None or score > bottom_best["score"]:
                bottom_best = candidate

    if top_best and bottom_best:
        # Slightly bias toward top to match typical book layout.
        top_score = top_best["score"] + 2.0
        return top_best["page"] if top_score >= bottom_best["score"] else bottom_best["page"]
    if top_best:
        return top_best["page"]
    if bottom_best:
        return bottom_best["page"]
    return None
    return text


def extract_page_number(text: str) -> Optional[int]:
    for pattern in PAGE_NUMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None
