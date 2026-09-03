"""
vietlawbert.streaming
~~~~~~~~~~~~~~~~~~~~~
Phân hệ điều phối luồng dữ liệu sự kiện thời gian thực (Real-time Event Streaming).
Đảm bảo tính tương thích với chuẩn Kafka Envelope v1 và tích hợp đồng bộ với Milvus/Neo4j.
"""

from .producer import LawEventProducer
from .consumer import LawEventConsumer

__all__ = [
    "LawEventProducer",
    "LawEventConsumer",
]