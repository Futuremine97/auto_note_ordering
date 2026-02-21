#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import FILE_ENC_MASTER_KEY, UPLOAD_DIR
from app.file_crypto import MAGIC, encrypt_file_in_place


def iter_files(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            yield entry


def is_encrypted(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False
    return data.startswith(MAGIC)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encrypt existing upload files in place (AES-CTR + HMAC)."
    )
    parser.add_argument(
        "--dir",
        default=str(UPLOAD_DIR),
        help="Uploads directory (default: UPLOAD_DIR env or backend/uploads)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be encrypted without modifying them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encrypt even if file already has encryption header",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N files (0 = no limit)",
    )
    args = parser.parse_args()

    if not FILE_ENC_MASTER_KEY:
        raise SystemExit("FILE_ENC_MASTER_KEY must be set before migration")

    root = Path(args.dir).resolve()
    if not root.exists():
        raise SystemExit(f"Upload directory not found: {root}")

    total = 0
    encrypted = 0
    skipped = 0

    for path in iter_files(root):
        if args.limit and total >= args.limit:
            break
        total += 1

        already = is_encrypted(path)
        if already and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            print(path.name)
            continue

        encrypt_file_in_place(path, path.name)
        encrypted += 1

    print(
        f"Done. scanned={total} encrypted={encrypted} skipped={skipped} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
