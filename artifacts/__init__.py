"""
vietlawbert.artifacts
~~~~~~~~~~~~~~~~~~~~~
Gói module quản lý tính toàn vẹn (Data Provenance & Lineage) của kho ngữ liệu pháp luật.
"""

from .canonical import (
    read_jsonl,
    write_jsonl,
    encode_payload,
    payload_hash,
    extract_item_id,
    encode_kafka_envelope,
    decode_kafka_envelope,
    canonical_artifacts,
    artifact_manifest,
)
from .merge import merge_records, merge_records_streaming

__all__ = [
    "read_jsonl",
    "write_jsonl",
    "encode_payload",
    "payload_hash",
    "extract_item_id",
    "encode_kafka_envelope",
    "decode_kafka_envelope",
    "canonical_artifacts",
    "artifact_manifest",
    "merge_records",
    "merge_records_streaming",
]
