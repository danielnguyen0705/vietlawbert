# law_dataset/utils/io_utils.py
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, Any


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def slugify_filename(text: str, max_length: int = 150) -> str:
    """
    Chuyển doc_id thành tên file an toàn.
    Ví dụ:
    'Quyết định 1472/QĐ-TCĐBVN'
    -> 'quyet_dinh_1472_qd_tcdbvn'
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_length] if text else "document"


def write_text_file(file_path: str | Path, text: str) -> None:
    ensure_parent_dir(file_path)
    Path(file_path).write_text(text, encoding="utf-8")


def append_jsonl(file_path: str | Path, record: Dict[str, Any]) -> None:
    ensure_parent_dir(file_path)
    with Path(file_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)