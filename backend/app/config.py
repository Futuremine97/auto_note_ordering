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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_CHAT_ENDPOINT = os.getenv("LLM_CHAT_ENDPOINT", "")
LLM_MAX_CONTEXT_CHARS = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "80000"))
LLM_MAX_IMAGE_CHARS = int(os.getenv("LLM_MAX_IMAGE_CHARS", "2000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_IMAGES = int(os.getenv("LLM_MAX_IMAGES", "12"))
LLM_IMAGE_MAX_BYTES = int(os.getenv("LLM_IMAGE_MAX_BYTES", "800000"))
LLM_IMAGE_MAX_DIM = int(os.getenv("LLM_IMAGE_MAX_DIM", "1024"))

PHOTO_PASSWORD = os.getenv("PHOTO_PASSWORD", "")
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_TTL_HOURS = int(os.getenv("AUTH_COOKIE_TTL_HOURS", "72"))
