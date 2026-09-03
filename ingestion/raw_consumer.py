"""
raw_consumer.py - Consumer lưu trữ nhanh văn bản thô vào Neo4j (Raw Document Manifest).
Hoạt động độc lập không cần load model Deep Learning để tối ưu hóa tốc độ ghi nhận diện rộng.
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import logging
import argparse
from typing import List, Dict, Any

from confluent_kafka import Consumer
from neo4j import GraphDatabase

from configs.config import config
from artifacts.canonical import decode_kafka_envelope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VietLawBERT_RawArchive")


def write_batch(session, rows: List[Dict[str, Any]]):
    """
    Nạp dữ liệu vào Neo4j qua thuộc tính khóa đơn archive_uid.
    Tương thích 100% với cả bản Neo4j Community lẫn Enterprise.
    """
    query = """
    UNWIND $rows AS row
    MERGE (d:RawLawDocument {archive_uid: row.archive_uid})
    SET d.dataset_id = row.dataset_id,
        d.doc_id = row.doc_id,
        d.payload_sha256 = row.payload_sha256,
        d.payload_bytes = row.payload_bytes,
        d.html_status = row.html_status,
        d.content_source = row.content_source,
        d.rescue_status = row.rescue_status,
        d.doc_number = row.doc_number,
        d.title = row.title,
        d.archived_at = datetime()
    """
    session.run(query, parameters={"rows": rows}).consume()


class RawArchiveConsumer:
    def __init__(self, dataset_id: str, topic: str):
        self.dataset_id = dataset_id
        self.topic = topic
        self.broker = getattr(config, "KAFKA_BROKER", "localhost:9092")
        self.group = os.getenv("KAFKA_RAW_GROUP_ID", f"raw-archive-{self.dataset_id}")

        self.consumer = Consumer({
            "bootstrap.servers": self.broker,
            "group.id": self.group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 900000,
        })
        self.driver = GraphDatabase.driver(
            getattr(config, "NEO4J_URI", "bolt://localhost:7687"),
            auth=(getattr(config, "NEO4J_USER", "neo4j"), getattr(config, "NEO4J_PASSWORD", "vietlawbert")),
        )

    def run(self, expect_documents: int, batch_size: int = 500, idle_exit_seconds: float = 10.0) -> dict:
        started = time.monotonic()
        last_message = started
        consumed = 0
        batch = []

        try:
            with self.driver.session() as session:
                # Sử dụng ràng buộc khóa đơn archive_uid để đảm bảo tương thích Neo4j Community
                session.run(
                    "CREATE CONSTRAINT raw_law_doc_uid IF NOT EXISTS "
                    "FOR (d:RawLawDocument) REQUIRE d.archive_uid IS UNIQUE"
                ).consume()
                session.run(
                    "CREATE INDEX raw_law_doc_dataset IF NOT EXISTS "
                    "FOR (d:RawLawDocument) ON (d.dataset_id)"
                ).consume()

                self.consumer.subscribe([self.topic])
                logger.info(f"Đang lắng nghe topic [{self.topic}] để lưu trữ raw manifest vào Neo4j...")

                while True:
                    message = self.consumer.poll(1.0)
                    if message is None:
                        if time.monotonic() - last_message >= idle_exit_seconds:
                            logger.info(f"Hàng đợi rỗng trong {idle_exit_seconds}s. Hoàn tất quá trình lưu trữ.")
                            break
                        continue

                    if message.error():
                        raise RuntimeError(f"Lỗi Kafka: {message.error()}")

                    raw_val = message.value()
                    try:
                        payload = decode_kafka_envelope(raw_val)
                        record = json.loads(payload.decode("utf-8"))
                    except Exception:
                        payload = raw_val
                        record = json.loads(payload.decode("utf-8"))

                    doc_id = str(record.get("item_id") or record.get("id") or "")
                    detail = record.get("metadata_detail") or {}
                    if not isinstance(detail, dict):
                        detail = {}

                    batch.append({
                        "archive_uid": f"{self.dataset_id}#{doc_id}",
                        "dataset_id": self.dataset_id,
                        "doc_id": doc_id,
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "payload_bytes": len(payload),
                        "html_status": str(record.get("html_status") or ""),
                        "content_source": str(record.get("content_source") or ""),
                        "rescue_status": str(record.get("rescue_status") or ""),
                        "doc_number": str(record.get("doc_number") or ""),
                        "title": str(detail.get("title") or "")[:2000],
                    })
                    consumed += 1
                    last_message = time.monotonic()

                    if len(batch) >= batch_size:
                        write_batch(session, batch)
                        self.consumer.commit(asynchronous=False)
                        batch.clear()

                if batch:
                    write_batch(session, batch)
                    self.consumer.commit(asynchronous=False)
                    batch.clear()

                stored = session.run(
                    "MATCH (d:RawLawDocument {dataset_id: $dataset_id}) RETURN count(d) AS total",
                    parameters={"dataset_id": self.dataset_id},
                ).single()["total"]
        finally:
            self.consumer.close()
            self.driver.close()

        elapsed = time.monotonic() - started
        report = {
            "topic": self.topic,
            "group": self.group,
            "consumed": consumed,
            "stored": stored,
            "elapsed_seconds": round(elapsed, 3),
            "documents_per_second": round(consumed / elapsed, 3) if elapsed > 0 else 0,
            "passed": stored == expect_documents,
        }
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Lưu trữ bản kê khai tài liệu thô vào Neo4j")
    parser.add_argument("--topic", default=getattr(config, "KAFKA_TOPIC", "law-documents-v5"), help="Tên Kafka Topic")
    parser.add_argument("--dataset-id", default="vietlaw_2026_q1", help="Mã định danh tập dữ liệu")
    parser.add_argument("--expect-documents", type=int, required=True, help="Số lượng văn bản kỳ vọng")
    parser.add_argument("--batch-size", type=int, default=500, help="Số văn bản nạp trong 1 batch")
    parser.add_argument("--idle-exit-seconds", type=float, default=10.0, help="Thời gian ngắt khi cạn hàng đợi")
    args = parser.parse_args()

    archiver = RawArchiveConsumer(dataset_id=args.dataset_id, topic=args.topic)
    result = archiver.run(
        expect_documents=args.expect_documents,
        batch_size=args.batch_size,
        idle_exit_seconds=args.idle_exit_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())