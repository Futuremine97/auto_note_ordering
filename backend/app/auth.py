import hmac
import hashlib
import time

from fastapi import HTTPException, Request

from .config import AUTH_SECRET, AUTH_COOKIE_TTL_HOURS, PHOTO_PASSWORD

AUTH_COOKIE_NAME = "page_ocr_auth"


def _sign(value: str) -> str:
    return hmac.new(AUTH_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()


def create_auth_cookie() -> str:
    issued = str(int(time.time()))
    signature = _sign(issued)
    return f"{issued}.{signature}"


def verify_auth_cookie(token: str) -> bool:
    if not token or "." not in token:
        return False
    issued, signature = token.split(".", 1)
    if not issued.isdigit():
        return False
    expected = _sign(issued)
    if not hmac.compare_digest(signature, expected):
        return False
    ttl = AUTH_COOKIE_TTL_HOURS * 3600
    if time.time() - int(issued) > ttl:
        return False
    return True


def require_auth(request: Request) -> None:
    if not PHOTO_PASSWORD:
        return
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token or not verify_auth_cookie(token):
        raise HTTPException(status_code=401, detail="Authentication required")
