"""Ghép các lần crawl retry vào checkpoint JSONL theo ``item_id``.

Record trong overlay thay thế record cùng ID ở base. Thứ tự của base được giữ
nguyên; ID mới trong overlay được nối vào cuối theo thứ tự xuất hiện.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: JSONL lỗi ở dòng {line_number}: {exc}") from exc
            item_id = str(record.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"{path}: dòng {line_number} không có item_id")
            yield record


def merge_records(base: Iterable[dict], overlays: Iterable[Iterable[dict]]) -> list[dict]:
    records: dict[str, dict] = {}
    order: list[str] = []

    def put(record: dict) -> None:
        item_id = str(record["item_id"]).strip()
        if item_id not in records:
            order.append(item_id)
        records[item_id] = record

    for record in base:
        put(record)
    for overlay in overlays:
        for record in overlay:
            put(record)
    return [records[item_id] for item_id in order]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay retry artifacts vào crawl checkpoint")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = merge_records(
        read_jsonl(args.base),
        (read_jsonl(path) for path in args.overlay),
    )
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "base": str(args.base),
                "overlays": [str(path) for path in args.overlay],
                "output": str(args.output),
                "records": len(records),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
