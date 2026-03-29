from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.db.mongo import get_database


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "law_dataset" / "data"

FILES_TO_DELETE = [
    DATA_DIR / "raw_links.json",
    DATA_DIR / "raw_documents.jsonl",
    DATA_DIR / "cleaned_documents.jsonl",
    DATA_DIR / "documents.jsonl",
    DATA_DIR / "chunks.jsonl",
    DATA_DIR / "parse_errors.jsonl",
    DATA_DIR / "chunk_errors.jsonl",
]

DIRS_TO_CLEAR = [
    DATA_DIR / "docs" / "raw",
    DATA_DIR / "docs" / "cleaned",
    DATA_DIR / "chunks",
    DATA_DIR / "logs",
]


def delete_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()
        print(f"[DELETED FILE] {path}")


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            print(f"[DELETED DIR ] {item}")
        else:
            item.unlink()
            print(f"[DELETED FILE] {item}")


def recreate_required_dirs() -> None:
    required_dirs = [
        DATA_DIR,
        DATA_DIR / "docs",
        DATA_DIR / "docs" / "raw",
        DATA_DIR / "docs" / "cleaned",
        DATA_DIR / "chunks",
        DATA_DIR / "logs",
    ]

    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)


def drop_mongodb_database() -> None:
    db = get_database()
    db_name = db.name
    db.client.drop_database(db_name)
    print(f"[DROPPED DB  ] {db_name}")


def confirm_or_exit(force: bool) -> None:
    if force:
        return

    print("Bạn sắp xóa toàn bộ database MongoDB và toàn bộ file crawl/chunk/log.")
    answer = input("Gõ DELETE để xác nhận: ").strip()
    if answer != "DELETE":
        print("Đã hủy thao tác.")
        raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset toàn bộ database và dữ liệu crawl/chunk của project."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bỏ qua bước xác nhận."
    )
    args = parser.parse_args()

    confirm_or_exit(force=args.yes)

    drop_mongodb_database()

    for file_path in FILES_TO_DELETE:
        delete_file(file_path)

    for dir_path in DIRS_TO_CLEAR:
        clear_directory(dir_path)

    recreate_required_dirs()
    print("Reset hoàn tất.")


if __name__ == "__main__":
    main()