"""
kafka_publisher.py - Bộ phát dữ liệu lớn Shards lên Kafka Broker.
Bảo đảm ngữ nghĩa phát Exactly-Once (Idempotent Producer) và đóng gói phong bì Gzip.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from configs.paths import ROOT_DIR, ARTIFACTS_DIR
from configs.config import config
from artifacts.canonical import (
    artifact_manifest,
    canonical_artifacts,
    encode_kafka_envelope,
    read_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VietLawBERT_KafkaPublisher")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp_{os.getpid()}")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def topic_message_count(broker: str, topic: str) -> int:
    """Đếm tổng số bản tin hiện có trên mọi phân vùng của Topic."""
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer({
        "bootstrap.servers": broker,
        "group.id": "topic-preflight-verifier",
        "enable.auto.commit": False,
    })
    try:
        metadata = consumer.list_topics(topic, timeout=10)
        if topic not in metadata.topics or metadata.topics[topic].error is not None:
            return 0
        total_msgs = 0
        for partition in metadata.topics[topic].partitions:
            watermarks = consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=10)
            total_msgs += watermarks[1]  # High watermark
        return total_msgs
    finally:
        consumer.close()


class KafkaPublisher:
    def __init__(self, broker: Optional[str] = None):
        self.broker = broker or getattr(config, "KAFKA_BROKER", "localhost:9092")
        self.producer = Producer({
            "bootstrap.servers": self.broker,
            "enable.idempotence": True,
            "acks": "all",
            "compression.type": "zstd",
            "linger.ms": 20,
            "batch.num.messages": 10000,
            "max.in.flight.requests.per.connection": 5,
        })
        self.delivered_count = 0
        self.delivery_errors: List[str] = []

    def _delivery_report(self, err, msg):
        if err is not None:
            self.delivery_errors.append(str(err))
            logger.error(f"Phát bản tin thất bại: {err}")
        else:
            self.delivered_count += 1

    def publish_shards(
        self,
        input_dir: Path,
        topic: str,
        expect_documents: int,
        expect_shards: Optional[int] = None,
        partitions: int = 8,
        manifest_output: Optional[Path] = None,
    ) -> dict:
        paths = canonical_artifacts(input_dir.resolve(), expect_shards)
        manifest = artifact_manifest(paths)

        if len(manifest) != expect_documents:
            raise RuntimeError(f"Kỳ vọng {expect_documents} tài liệu, nhưng manifest chỉ tìm thấy {len(manifest)} IDs!")

        # Khởi tạo topic nếu chưa tồn tại
        admin = AdminClient({"bootstrap.servers": self.broker})
        new_topic = NewTopic(topic, num_partitions=partitions, replication_factor=1)
        res = admin.create_topics([new_topic])
        try:
            res[topic].result(timeout=15)
            logger.info(f"✓ Đã tạo mới topic [{topic}] với {partitions} phân vùng.")
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(exc).lower():
                raise

        existing_count = topic_message_count(self.broker, topic)
        if existing_count > 0:
            logger.warning(f"Topic [{topic}] hiện đã chứa {existing_count} bản tin.")

        logger.info(f"Bắt đầu phát {len(manifest)} văn bản từ {len(paths)} shards lên topic [{topic}]...")

        for path in paths:
            logger.info(f"Đang phát shard: {path.name}")
            for record in read_jsonl(path):
                payload = encode_kafka_envelope(record)

                if len(payload) >= 950_000:
                    raise RuntimeError(f"Kích thước phong bì vượt quá ngưỡng 1MB: {record.get('item_id')} = {len(payload)} bytes")

                while True:
                    try:
                        self.producer.produce(
                            topic,
                            key=str(record.get("item_id") or "").encode("utf-8"),
                            value=payload,
                            callback=self._delivery_report,
                        )
                        break
                    except BufferError:
                        self.producer.poll(0.5)

                self.producer.poll(0)

        remaining = self.producer.flush(120)
        if remaining > 0 or self.delivery_errors or self.delivered_count != expect_documents:
            raise RuntimeError(
                f"Quá trình phát Kafka thất bại: đã nhận={self.delivered_count}, tồn đọng={remaining}, lỗi={len(self.delivery_errors)}"
            )

        report = {
            "topic": topic,
            "broker": self.broker,
            "total_documents": len(manifest),
            "delivered_documents": self.delivered_count,
            "shards": [str(p) for p in paths],
        }

        out_path = manifest_output or (input_dir / "kafka_publish_manifest.json")
        write_json_atomic(out_path, report)
        logger.info(f"✓ Phát Kafka hoàn tất 100%! Đã lưu biên bản tại: {out_path}")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phát dữ liệu Shards lên Kafka đạt chuẩn Big Data")
    parser.add_argument("--input-dir", type=Path, default=ARTIFACTS_DIR / "full_crawl", help="Thư mục chứa Shards")
    parser.add_argument("--expect-documents", type=int, required=True, help="Số lượng văn bản kỳ vọng")
    parser.add_argument("--expect-shards", type=int, help="Số lượng file shard kỳ vọng")
    parser.add_argument("--topic", default=getattr(config, "KAFKA_TOPIC", "law-documents-v5"), help="Tên Kafka Topic")
    parser.add_argument("--partitions", type=int, default=8, help="Số phân vùng")
    parser.add_argument("--manifest-output", type=Path, help="Đường dẫn lưu file báo cáo kết quả")
    args = parser.parse_args()

    publisher = KafkaPublisher()
    try:
        report = publisher.publish_shards(
            input_dir=args.input_dir,
            topic=args.topic,
            expect_documents=args.expect_documents,
            expect_shards=args.expect_shards,
            partitions=args.partitions,
            manifest_output=args.manifest_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logger.error(f"Thất bại trong quá trình phát Kafka: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())