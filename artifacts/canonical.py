"""
canonical.py - Công cụ chuẩn hóa Artifacts và đóng gói Kafka Envelope.
"""

from __future__ import annotations

import os
import gzip
import json
import base64
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Iterable


def read_jsonl(path: Path | str, require_item_id: bool = True) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp: {p}")

    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            clean_line = line.strip()
            if not clean_line:
                continue
            try:
                record = json.loads(clean_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Lỗi JSON tại {p.name} dòng {line_no}: {exc}") from exc

            if require_item_id:
                raw_id = record.get("item_id") or record.get("doc_id") or record.get("id")
                if not str(raw_id or "").strip():
                    raise ValueError(f"{p}: dòng {line_no} không có item_id")

            yield record


def write_jsonl(path: Path | str, records: Iterable[Dict[str, Any]]) -> int:
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f"{dest_path.name}.tmp_{os.getpid()}")

    opener = gzip.open if dest_path.suffix == ".gz" else open
    count = 0
    try:
        with opener(temp_path, "wt", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        temp_path.replace(dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return count


def canonical_artifacts(input_dir: Path | str, expected_shards: Optional[int] = None) -> List[Path]:
    dir_path = Path(input_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Đường dẫn không phải thư mục: {dir_path}")

    bases = sorted(
        path for path in dir_path.glob("crawl_pages_*.jsonl.gz")
        if ".rescued." not in path.name
    )

    selected: List[Path] = []
    for base in bases:
        rescued = base.with_name(base.name.replace(".jsonl.gz", ".rescued.jsonl.gz"))
        selected.append(rescued if rescued.exists() else base)

    if expected_shards is not None and len(selected) != expected_shards:
        raise ValueError(f"Kỳ vọng {expected_shards} shards, tìm thấy {len(selected)}")
    return selected


def encode_payload(record: Dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_item_id(record: Dict[str, Any]) -> str:
    raw_id = record.get("item_id") or record.get("doc_id") or record.get("id")
    item_id = str(raw_id or "").strip()
    if not item_id:
        raise KeyError(f"Bản ghi thiếu item_id/doc_id: {record}")
    return item_id


def encode_kafka_envelope(record: Dict[str, Any]) -> bytes:
    item_id = extract_item_id(record)
    payload = encode_payload(record)
    sha256_sum = payload_hash(payload)

    compressed_bytes = gzip.compress(payload, compresslevel=6, mtime=0)
    b64_payload = base64.b64encode(compressed_bytes).decode("ascii")

    envelope = {
        "schema": "vietlaw.raw.gzip.v1",
        "item_id": item_id,
        "payload_sha256": sha256_sum,
        "uncompressed_bytes": len(payload),
        "payload_gzip_b64": b64_payload,
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decode_kafka_envelope(value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")

    envelope = json.loads(value.decode("utf-8"))
    if envelope.get("schema") != "vietlaw.raw.gzip.v1":
        raise ValueError(f"Schema không hỗ trợ: {envelope.get('schema')}")

    payload = gzip.decompress(base64.b64decode(envelope["payload_gzip_b64"]))
    if len(payload) != int(envelope["uncompressed_bytes"]):
        raise ValueError("Sai uncompressed_bytes")

    if payload_hash(payload) != envelope["payload_sha256"]:
        raise ValueError("Mã SHA-256 không khớp")

    return payload


def artifact_manifest(paths: List[Path]) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for path in paths:
        for record in read_jsonl(path, require_item_id=True):
            item_id = extract_item_id(record)
            if item_id in manifest:
                raise ValueError(f"Trùng lặp item_id '{item_id}' tại {path.name}")
            manifest[item_id] = payload_hash(encode_payload(record))
    return manifest
