"""
merge.py - Ghép nối overlay vào checkpoint JSONL.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Iterable

from .canonical import read_jsonl, write_jsonl, extract_item_id

logger = logging.getLogger("VietLawBERT_ArtifactMerge")


def merge_records(base: Iterable[dict], overlays: Iterable[Iterable[dict]]) -> List[dict]:
    records: Dict[str, dict] = {}
    order: List[str] = []

    def put(record: dict) -> None:
        raw_id = record.get("item_id") or record.get("doc_id") or record.get("id")
        item_id = str(raw_id or "").strip()
        if not item_id:
            raise ValueError("record không có item_id")
        if item_id not in records:
            order.append(item_id)
        records[item_id] = record

    for record in base:
        put(record)
    for overlay in overlays:
        for record in overlay:
            put(record)
    return [records[item_id] for item_id in order]


def merge_records_streaming(
    base_path: Path | str,
    overlay_paths: List[Path | str],
    output_path: Path | str
) -> Dict[str, int]:
    b_path = Path(base_path)
    out_path = Path(output_path)

    overlay_map: Dict[str, Dict[str, Any]] = {}
    overlay_new_order: List[str] = []

    for o_path in overlay_paths:
        p = Path(o_path)
        if not p.exists():
            continue
        for record in read_jsonl(p, require_item_id=True):
            item_id = extract_item_id(record)
            if item_id not in overlay_map and item_id not in overlay_new_order:
                overlay_new_order.append(item_id)
            overlay_map[item_id] = record

    overwritten_count = 0
    base_count = 0
    consumed_overlay_ids = set()

    def record_generator():
        nonlocal overwritten_count, base_count
        for base_record in read_jsonl(b_path, require_item_id=True):
            base_count += 1
            item_id = extract_item_id(base_record)
            if item_id in overlay_map:
                yield overlay_map[item_id]
                consumed_overlay_ids.add(item_id)
                overwritten_count += 1
            else:
                yield base_record

        for item_id in overlay_new_order:
            if item_id not in consumed_overlay_ids:
                yield overlay_map[item_id]

    total_written = write_jsonl(out_path, record_generator())
    return {
        "base_records": base_count,
        "overwritten_records": overwritten_count,
        "appended_records": total_written - base_count,
        "total_final_records": total_written,
    }
