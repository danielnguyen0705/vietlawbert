"""
pipeline_verifier.py - Bộ đối soát toàn vẹn dòng dữ liệu lớn (End-to-End Lineage Verifier).
Xác thực tính tương đồng tuyệt đối: Shard Artifacts == Kafka Messages == Neo4j Archive Nodes.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from confluent_kafka import Consumer, TopicPartition
from neo4j import GraphDatabase

from configs.paths import ROOT_DIR, ARTIFACTS_DIR
from configs.config import config
from artifacts.canonical import (
    artifact_manifest,
    canonical_artifacts,
    decode_kafka_envelope,
    payload_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VietLawBERT_LineageVerifier")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp_{os.getpid()}")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_kafka_manifest(broker: str, topic: str) -> Tuple[Dict[str, str], Dict[int, int], int]:
    """
    Quét toàn bộ bản tin trên Topic Kafka. Tính toán chính xác dung lượng dựa trên khoảng cách (high - low).
    """
    consumer = Consumer({
        "bootstrap.servers": broker,
        "group.id": f"lineage-verifier-{uuid.uuid4()}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })

    result: Dict[str, str] = {}
    high_watermarks: Dict[int, int] = {}
    duplicate_keys_count = 0

    try:
        metadata = consumer.list_topics(topic, timeout=15)
        if topic not in metadata.topics:
            raise ValueError(f"Topic [{topic}] không tồn tại trên Kafka Broker!")

        partitions = sorted(metadata.topics[topic].partitions.keys())
        assignments = []
        total_expected_messages = 0

        for p in partitions:
            tp = TopicPartition(topic, p)
            low, high = consumer.get_watermark_offsets(tp, timeout=10)
            high_watermarks[p] = high
            assignments.append(TopicPartition(topic, p, low))
            total_expected_messages += max(0, high - low)

        consumer.assign(assignments)
        logger.info(f"Tổng số bản tin khả dụng trên topic [{topic}]: {total_expected_messages} (trên {len(partitions)} phân vùng)")

        messages_read = 0
        while messages_read < total_expected_messages:
            msg = consumer.poll(3.0)
            if msg is None:
                logger.warning("Chạm giới hạn poll timeout, hoàn tất đọc các bản tin hiện hữu.")
                break
            if msg.error():
                logger.error(f"Lỗi phân vùng Kafka: {msg.error()}")
                continue

            doc_id = msg.key().decode("utf-8") if msg.key() else f"unknown_offset_{msg.offset()}"
            try:
                payload = decode_kafka_envelope(msg.value())
                calc_hash = payload_hash(payload)
            except Exception:
                calc_hash = payload_hash(msg.value())

            if doc_id in result:
                duplicate_keys_count += 1
            else:
                result[doc_id] = calc_hash

            messages_read += 1

    finally:
        consumer.close()

    return result, high_watermarks, duplicate_keys_count


def verify_pipeline_lineage(
    input_dir: Path,
    topic: str,
    dataset_id: str,
    expect_documents: int,
    expect_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tiến hành đối soát mã băm SHA-256 đối xứng và tập định danh ID giữa:
    1. Disk Shards (Artifacts)
    2. Kafka Topic Messages
    3. Neo4j RawLawDocument Nodes
    """
    broker = getattr(config, "KAFKA_BROKER", "localhost:9092")
    neo_uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    neo_user = getattr(config, "NEO4J_USER", "neo4j")
    neo_pwd = getattr(config, "NEO4J_PASSWORD", "vietlawbert")

    logger.info("1. Đang tính toán Deterministic Manifest từ Disk Shards...")
    paths = canonical_artifacts(input_dir.resolve(), expect_shards)
    artifacts = artifact_manifest(paths)
    logger.info(f"✓ Shards Manifest: {len(artifacts)} văn bản duy nhất.")

    logger.info(f"2. Đang kiểm toán toàn bộ bản tin trên Kafka [{topic}]...")
    kafka_manifest, high_watermarks, dup_keys = read_kafka_manifest(broker, topic)
    logger.info(f"✓ Kafka Manifest: {len(kafka_manifest)} văn bản duy nhất (Trùng lặp: {dup_keys}).")

    logger.info(f"3. Đang đối soát kho lưu trữ đồ thị Neo4j (Dataset: {dataset_id})...")
    driver = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pwd))
    database_manifest: Dict[str, str] = {}
    html_statuses: Dict[str, int] = {}

    try:
        with driver.session() as session:
            for rec in session.run(
                "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) "
                "RETURN d.doc_id AS doc_id, d.payload_sha256 AS payload_sha256",
                dataset_id=dataset_id,
            ):
                database_manifest[str(rec["doc_id"])] = str(rec["payload_sha256"])

            for rec in session.run(
                "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) "
                "RETURN coalesce(d.html_status, 'UNKNOWN') AS status, count(d) AS total",
                dataset_id=dataset_id,
            ):
                html_statuses[str(rec["status"])] = int(rec["total"])
    finally:
        driver.close()
    logger.info(f"✓ Neo4j Manifest: {len(database_manifest)} văn bản.")

    art_keys = set(artifacts.keys())
    kafka_keys = set(kafka_manifest.keys())
    db_keys = set(database_manifest.keys())

    missing_in_kafka = list(art_keys - kafka_keys)[:10]
    extra_in_kafka = list(kafka_keys - art_keys)[:10]
    missing_in_db = list(art_keys - db_keys)[:10]
    extra_in_db = list(db_keys - art_keys)[:10]

    hash_diff_kafka = [k for k in art_keys & kafka_keys if artifacts[k] != kafka_manifest[k]][:10]
    hash_diff_db = [k for k in art_keys & db_keys if artifacts[k] != database_manifest[k]][:10]

    id_match_kafka = art_keys == kafka_keys
    id_match_db = art_keys == db_keys
    hash_match_kafka = (len(hash_diff_kafka) == 0) and id_match_kafka
    hash_match_db = (len(hash_diff_db) == 0) and id_match_db

    passed = (
        len(artifacts) == expect_documents
        and len(kafka_manifest) == expect_documents
        and len(database_manifest) == expect_documents
        and hash_match_kafka
        and hash_match_db
    )

    report = {
        "dataset_id": dataset_id,
        "topic": topic,
        "expected_documents": expect_documents,
        "artifact_documents": len(artifacts),
        "kafka_unique_documents": len(kafka_manifest),
        "kafka_duplicate_messages": dup_keys,
        "database_documents": len(database_manifest),
        "kafka_high_watermarks": high_watermarks,
        "database_html_statuses": html_statuses,
        "matches": {
            "artifact_kafka_id_match": id_match_kafka,
            "artifact_database_id_match": id_match_db,
            "artifact_kafka_hash_match": hash_match_kafka,
            "artifact_database_hash_match": hash_match_db,
        },
        "diagnostics": {
            "missing_in_kafka_samples": missing_in_kafka,
            "extra_in_kafka_samples": extra_in_kafka,
            "missing_in_db_samples": missing_in_db,
            "extra_in_db_samples": extra_in_db,
            "hash_mismatch_kafka_samples": hash_diff_kafka,
            "hash_mismatch_db_samples": hash_diff_db,
        },
        "passed": passed,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bộ đối soát toàn vẹn Shards = Kafka = Neo4j")
    parser.add_argument("--input-dir", type=Path, default=ARTIFACTS_DIR / "full_crawl", help="Thư mục chứa Shards")
    parser.add_argument("--topic", default=getattr(config, "KAFKA_TOPIC", "law-documents-v5"), help="Tên Kafka Topic")
    parser.add_argument("--dataset-id", default="vietlaw_2026_q1", help="Mã định danh Dataset")
    parser.add_argument("--expect-documents", type=int, required=True, help="Số lượng văn bản kỳ vọng")
    parser.add_argument("--expect-shards", type=int, help="Số lượng file Shards kỳ vọng")
    parser.add_argument("--output", type=Path, help="Đường dẫn lưu tệp báo cáo JSON")
    args = parser.parse_args()

    try:
        report = verify_pipeline_lineage(
            input_dir=args.input_dir,
            topic=args.topic,
            dataset_id=args.dataset_id,
            expect_documents=args.expect_documents,
            expect_shards=args.expect_shards,
        )

        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)

        out_path = args.output or (args.input_dir / "pipeline_lineage_report.json")
        write_json_atomic(out_path, report)
        logger.info(f"✓ Đã lưu biên bản đối soát tại: {out_path}")

        return 0 if report["passed"] else 1
    except Exception as exc:
        logger.error(f"Thất bại trong quá trình đối soát dữ liệu: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
