"""
producer.py - Module Streaming Producer đẩy dữ liệu văn bản thô vào Redpanda/Kafka.
"""

import json
import logging
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("KafkaProducer")

class LawEventProducer:
    def __init__(self, bootstrap_servers="localhost:9092"):
        conf = {'bootstrap.servers': bootstrap_servers}
        self.producer = Producer(conf)
        self.topic = "law-documents"

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Gửi tin nhắn thất bại: {err}")
        else:
            logger.info(f"Đã gửi thành công {msg.topic()} [{msg.partition()}]")

    def send_document(self, item_id: str, doc_number: str, html_raw: str, metadata_api: dict):
        """Đẩy văn bản mới vào hàng đợi Kafka để xử lý bất đồng bộ."""
        payload = {
            "item_id": item_id,
            "doc_number": doc_number,
            "html_raw": html_raw,
            "metadata_api": metadata_api
        }
        self.producer.produce(
            self.topic,
            key=str(item_id),
            value=json.dumps(payload, ensure_ascii=False),
            callback=self._delivery_report
        )
        self.producer.flush()

if __name__ == "__main__":
    # Test nhanh kết nối
    p = LawEventProducer()
    logger.info("Kafka Producer ready.")
