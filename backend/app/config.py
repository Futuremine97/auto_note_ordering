import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/page_ocr",
)

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng+kor+jpn")
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
