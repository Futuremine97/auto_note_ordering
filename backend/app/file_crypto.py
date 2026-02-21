import base64
import hashlib
import hmac
import os
from typing import Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .config import AUTH_SECRET, FILE_ENC_MASTER_KEY, FILE_ENC_ALLOW_PLAINTEXT

MAGIC = b"ENCF"
IV_LEN = 16
HMAC_LEN = 32
HEADER_LEN = len(MAGIC) + IV_LEN + HMAC_LEN


def encrypt_file_in_place(path, file_id: str) -> None:
    data = path.read_bytes()
    if data.startswith(MAGIC):
        return
    encrypted = encrypt_bytes(data, file_id)
    path.write_bytes(encrypted)


def read_decrypted_file(path, file_id: str) -> bytes:
    data = path.read_bytes()
    return decrypt_bytes(data, file_id, allow_plaintext=FILE_ENC_ALLOW_PLAINTEXT)


def encrypt_bytes(plaintext: bytes, file_id: str) -> bytes:
    enc_key, mac_key = _derive_keys(file_id)
    iv = os.urandom(IV_LEN)
    cipher = Cipher(algorithms.AES(enc_key), modes.CTR(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    tag = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    return MAGIC + iv + tag + ciphertext


def decrypt_bytes(data: bytes, file_id: str, *, allow_plaintext: bool = False) -> bytes:
    if not data.startswith(MAGIC):
        if allow_plaintext:
            return data
        raise ValueError("File is not encrypted or has invalid header")

    if len(data) < HEADER_LEN:
        raise ValueError("Encrypted file is too small")

    iv_start = len(MAGIC)
    tag_start = iv_start + IV_LEN
    ciphertext_start = tag_start + HMAC_LEN

    iv = data[iv_start:tag_start]
    tag = data[tag_start:ciphertext_start]
    ciphertext = data[ciphertext_start:]

    enc_key, mac_key = _derive_keys(file_id)
    expected = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Integrity check failed")

    cipher = Cipher(algorithms.AES(enc_key), modes.CTR(iv))
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def _derive_keys(file_id: str) -> Tuple[bytes, bytes]:
    master = _load_master_key()
    file_id_bytes = file_id.encode("utf-8")
    enc_key = hmac.new(master, b"enc:" + file_id_bytes, hashlib.sha256).digest()[:16]
    mac_key = hmac.new(master, b"mac:" + file_id_bytes, hashlib.sha256).digest()
    return enc_key, mac_key


def _load_master_key() -> bytes:
    raw = FILE_ENC_MASTER_KEY or AUTH_SECRET
    if not raw:
        raise RuntimeError("FILE_ENC_MASTER_KEY is required for file encryption")
    key_bytes = _decode_key(raw)
    if len(key_bytes) != 32:
        key_bytes = hashlib.sha256(key_bytes).digest()
    return key_bytes


def _decode_key(raw: str) -> bytes:
    try:
        decoded = base64.b64decode(raw, validate=True)
        if decoded:
            return decoded
    except Exception:
        pass
    return raw.encode("utf-8")
