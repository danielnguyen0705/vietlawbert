from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JSONLineError:
    filename: str
    line_number: int
    reason: str


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_jsonl_file(path: Path, required_keys: tuple[str, ...] = ()) -> tuple[list[dict[str, Any]], list[JSONLineError]]:
    records: list[dict[str, Any]] = []
    errors: list[JSONLineError] = []
    if not path.exists():
        return records, [JSONLineError(str(path), 0, "file_not_found")]
    if path.stat().st_size == 0:
        return records, [JSONLineError(str(path), 0, "empty_file")]

    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(JSONLineError(str(path), line_number, f"invalid_json:{exc.msg}"))
                continue
            if not isinstance(value, dict):
                errors.append(JSONLineError(str(path), line_number, "record_not_object"))
                continue
            for key in required_keys:
                if key not in value:
                    errors.append(JSONLineError(str(path), line_number, f"missing_required_key:{key}"))
            text = value.get("text")
            if "text" in value and not isinstance(text, str):
                errors.append(JSONLineError(str(path), line_number, "text_not_string"))
            chunk_index = value.get("chunk_index")
            if "chunk_index" in value and (not isinstance(chunk_index, int) or chunk_index < 0):
                errors.append(JSONLineError(str(path), line_number, "invalid_chunk_index"))
            if "text" in value and isinstance(text, str) and not text.strip():
                errors.append(JSONLineError(str(path), line_number, "empty_text"))
            records.append(value)
    return records, errors
