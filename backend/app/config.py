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

STT_API_BASE = os.getenv("STT_API_BASE", "")
STT_API_KEY = os.getenv("STT_API_KEY", "")
STT_MODEL = os.getenv("STT_MODEL", "")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
STT_ENDPOINT = os.getenv("STT_ENDPOINT", "")
STT_TIMEOUT = int(os.getenv("STT_TIMEOUT", "60"))
STT_MAX_BYTES = int(os.getenv("STT_MAX_BYTES", str(25 * 1024 * 1024)))
STT_AUTO_COMPRESS = os.getenv("STT_AUTO_COMPRESS", "true").lower() == "true"
STT_AUTO_SPLIT = os.getenv("STT_AUTO_SPLIT", "true").lower() == "true"
STT_CHUNK_SECONDS = int(os.getenv("STT_CHUNK_SECONDS", "180"))
STT_COMPRESS_FORMAT = os.getenv("STT_COMPRESS_FORMAT", "mp3")
STT_COMPRESS_BITRATE = os.getenv("STT_COMPRESS_BITRATE", "32k")
STT_COMPRESS_SAMPLE_RATE = int(os.getenv("STT_COMPRESS_SAMPLE_RATE", "16000"))
STT_COMPRESS_CHANNELS = int(os.getenv("STT_COMPRESS_CHANNELS", "1"))

TTS_API_BASE = os.getenv("TTS_API_BASE", "")
TTS_API_KEY = os.getenv("TTS_API_KEY", "")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "mp3")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "60"))

PHOTO_PASSWORD = os.getenv("PHOTO_PASSWORD", "")
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_TTL_HOURS = int(os.getenv("AUTH_COOKIE_TTL_HOURS", "72"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small"
)
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", str(BASE_DIR / "cache"))
EMBEDDING_TRAIN_EPOCHS = int(os.getenv("EMBEDDING_TRAIN_EPOCHS", "25"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_LEARNING_RATE = float(os.getenv("EMBEDDING_LEARNING_RATE", "0.001"))
