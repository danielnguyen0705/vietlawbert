"""
producer.py - Event Streaming Producer đẩy dữ liệu văn bản pháp luật vào Kafka/Redpanda.
Chuẩn hóa phong bì Gzip v1 (Deterministic SHA-256) và hỗ trợ cơ chế Idempotent Delivery.
"""

from __future__ import annotations

import os
import sys
import socket
import logging
from typing import Dict, Any, Optional, List

from confluent_kafka import Producer

from configs.config import config
from artifacts.canonical import encode_kafka_envelope

logger = logging.getLogger("VietLawBERT_StreamingProducer")


class LawEventProducer:
    """Producer chuyên trách phát sự kiện văn bản pháp lý lên Kafka Broker đạt chuẩn Big Data."""

    def __init__(self, bootstrap_servers: Optional[str] = None, topic: Optional[str] = None):
        self.broker = bootstrap_servers or getattr(config, "KAFKA_BROKER", "localhost:9092")
        self.topic = topic or getattr(config, "KAFKA_TOPIC", "law-documents-v5")

        conf = {
            "bootstrap.servers": self.broker,
            "client.id": socket.gethostname(),
            "enable.idempotence": True,
            "acks": "all",
            "compression.type": "zstd",
            "linger.ms": 20,
            "batch.size": 65536,
            "max.in.flight.requests.per.connection": 5,
        }
        self.producer = Producer(conf)
        self.delivered_count = 0
        self.delivery_errors: List[str] = []
        logger.info(f"Khởi tạo LawEventProducer kết nối tới {self.broker} (Topic: {self.topic})")

    def _delivery_report(self, err, msg):
        if err is not None:
            err_msg = f"Phát tin nhắn thất bại: {err}"
            self.delivery_errors.append(err_msg)
            logger.error(err_msg)
        else:
            self.delivered_count += 1

    def send_document(
        self,
        item_id: str,
        doc_number: str,
        html_raw: str,
        metadata_api: Optional[Dict[str, Any]] = None,
        metadata_detail: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Đóng gói văn bản theo chuẩn Kafka Envelope và đẩy bất đồng bộ vào hàng đợi."""
        clean_item_id = str(item_id).strip()
        record = {
            "item_id": clean_item_id,
            "doc_number": str(doc_number).strip(),
            "html_raw": html_raw or "",
            "html_status": "VALID" if len(html_raw.strip()) >= 100 else "EMPTY",
            "metadata_api": metadata_api or {},
            "metadata_detail": metadata_detail or {},
            "relationships": relationships or [],
        }

        try:
            payload = encode_kafka_envelope(record)
        except Exception as exc:
            logger.error(f"Lỗi đóng gói phong bì Envelope cho document {clean_item_id}: {exc}")
            return False

        while True:
            try:
                self.producer.produce(
                    self.topic,
                    key=clean_item_id.encode("utf-8"),
                    value=payload,
                    callback=self._delivery_report,
                )
                break
            except BufferError:
                # Bộ đệm RAM của Producer đầy, xả tạm trong 0.5s rồi thử lại
                self.producer.poll(0.5)

        self.producer.poll(0)
        return True

    def flush(self, timeout: float = 30.0) -> int:
        """Xả sạch toàn bộ bộ đệm Producer ra Kafka broker."""
        remaining = self.producer.flush(timeout)
        logger.info(f"Đã xả bộ đệm Producer. Số bản tin còn tồn đọng: {remaining}")
        return remaining

    def close(self):
        self.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
    p = LawEventProducer()
    logger.info("✓ LawEventProducer sẵn sàng hoạt động.")