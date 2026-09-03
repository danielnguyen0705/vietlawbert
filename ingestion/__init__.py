"""
vietlawbert.ingestion
~~~~~~~~~~~~~~~~~~~~~
Phân hệ điều phối luồng dữ liệu sự kiện (Event-Driven Ingestion Engine).
Đảm bảo tính toàn vẹn At-Least-Once Delivery và đồng bộ giao dịch ACID giữa Kafka, Milvus và Neo4j.
"""

from .embedding_consumer import LawEventConsumer
from .kafka_publisher import KafkaPublisher, main as publish_main
from .raw_consumer import RawArchiveConsumer, main as raw_archive_main

__all__ = [
    "LawEventConsumer",
    "KafkaPublisher",
    "RawArchiveConsumer",
    "publish_main",
    "raw_archive_main",
]