"""Shared helpers cho canonical crawl artifacts và payload Kafka."""

from __future__ import annotations

import hashlib
import base64
import gzip
import json
from pathlib import Path

from audit_pilot import read_jsonl


def canonical_artifacts(input_dir: Path, expected_shards: int | None = None) -> list[Path]:
    bases = sorted(
        path
        for path in input_dir.glob("crawl_pages_*.jsonl.gz")
        if ".rescued." not in path.name
    )
    selected = []
    for base in bases:
        rescued = base.with_name(base.name.replace(".jsonl.gz", ".rescued.jsonl.gz"))
        selected.append(rescued if rescued.exists() else base)
    if expected_shards is not None and len(selected) != expected_shards:
        raise ValueError(f"expected {expected_shards} shards, found {len(selected)}")
    return selected


def encode_payload(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_kafka_envelope(record: dict) -> bytes:
    payload = encode_payload(record)
    envelope = {
        "schema": "vietlaw.raw.gzip.v1",
        "item_id": str(record["item_id"]),
        "payload_sha256": payload_hash(payload),
        "uncompressed_bytes": len(payload),
        "payload_gzip_b64": base64.b64encode(gzip.compress(payload, compresslevel=6, mtime=0)).decode("ascii"),
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decode_kafka_envelope(value: bytes) -> bytes:
    envelope = json.loads(value)
    if envelope.get("schema") != "vietlaw.raw.gzip.v1":
        raise ValueError("Kafka envelope schema không hỗ trợ")
    payload = gzip.decompress(base64.b64decode(envelope["payload_gzip_b64"]))
    if len(payload) != int(envelope["uncompressed_bytes"]):
        raise ValueError("Kafka envelope sai uncompressed_bytes")
    if payload_hash(payload) != envelope["payload_sha256"]:
        raise ValueError("Kafka envelope SHA-256 mismatch")
    return payload


def artifact_manifest(paths: list[Path]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in paths:
        for record in read_jsonl(path):
            item_id = str(record.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"record không có item_id trong {path}")
            if item_id in manifest:
                raise ValueError(f"duplicate item_id xuyên shard: {item_id}")
            manifest[item_id] = payload_hash(encode_payload(record))
    return manifest
