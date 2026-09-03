"""
consumer.py - Production Streaming Consumer cho VietLawBERT.
Đảm bảo At-Least-Once Delivery, cam kết offset thủ công, phân đoạn AST và tương thích Kafka Envelope v1.
"""

from __future__ import annotations

import os
import sys
import time
import json
import signal
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple

from confluent_kafka import Consumer, KafkaError, TopicPartition
from bs4 import BeautifulSoup
import html2text

from configs.config import config
from artifacts.canonical import decode_kafka_envelope
from preprocess.legal_chunker import chunk_legal_document
from preprocess.text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
)
from database.milvus_client import MilvusClientWrapper, EmbeddingEngine
from database.neo4j_client import Neo4jClient

logger = logging.getLogger("VietLawBERT_StreamingConsumer")


class LawEventConsumer:
    """Consumer xử lý luồng sự kiện pháp luật, cắt đoạn AST và nạp đồng thời vào Milvus và Neo4j."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        self.broker = bootstrap_servers or getattr(config, "KAFKA_BROKER", "localhost:9092")
        self.group_id = group_id or getattr(config, "KAFKA_GROUP_ID", "vietlawbert-consumers-v5-bounded")
        self.topic = topic or getattr(config, "KAFKA_TOPIC", "law-documents-v5")

        self.doc_batch_size = int(getattr(config, "CONSUMER_DOC_BATCH_SIZE", 10))
        self.chunk_batch_size = int(getattr(config, "CONSUMER_CHUNK_BATCH_SIZE", 64))
        self.flush_interval = float(getattr(config, "CONSUMER_FLUSH_INTERVAL_SECONDS", 5.0))

        conf = {
            "bootstrap.servers": self.broker,
            "group.id": self.group_id,
            "enable.auto.commit": False,  # Bắt buộc tắt auto-commit bảo đảm tính toàn vẹn At-Least-Once
            "auto.offset.reset": "earliest",
            "session.timeout.ms": 60000,
            "max.poll.interval.ms": 1800000,
        }
        self.consumer = Consumer(conf)
        self.consumer.subscribe([self.topic])

        # Khởi tạo kết nối lưu trữ phân tán
        self.encoder = EmbeddingEngine.get_instance()
        self.milvus_client = MilvusClientWrapper()
        self.neo_client = Neo4jClient()

        # Bộ đệm Micro-batching trên RAM
        self.pending_milvus_rows: List[Dict[str, Any]] = []
        self.pending_milvus_texts: List[str] = []
        self.pending_neo_batch: List[Dict[str, Any]] = []
        self.pending_relations_batch: List[Dict[str, Any]] = []
        self.pending_messages: List[Any] = []

        self.last_flush_time = time.monotonic()
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"LawEventConsumer đã khởi tạo: Topic [{self.topic}] | Group [{self.group_id}]")

    def _signal_handler(self, sig, frame):
        logger.info(f"[DỪNG TIẾN TRÌNH] Bắt tín hiệu {sig}. Chuẩn bị xả bộ đệm và ngắt kết nối an toàn...")
        self.running = False

    def process_message(self, msg: Any, data: Dict[str, Any]):
        """Xử lý phân luồng: bản ghi đã phân đoạn sẵn hoặc văn bản thô cần phân tích cú pháp AST."""
        if "chunk_id" in data and ("contextualized_text" in data or "original_text" in data):
            self._process_chunk_record(data)
        else:
            self._process_raw_document(data)

        self.pending_messages.append(msg)

        now = time.monotonic()
        if len(self.pending_messages) >= self.doc_batch_size or (now - self.last_flush_time >= self.flush_interval):
            self._safe_flush_and_commit()

    def _process_chunk_record(self, data: Dict[str, Any]):
        chunk_id = str(data["chunk_id"])
        meta = data.get("metadata") or {}
        hierarchy = meta.get("hierarchy_path") or {}
        text = data.get("contextualized_text") or data.get("original_text", "")
        original_text = data.get("original_text", "")

        doc_id = str(meta.get("doc_id") or data.get("doc_id") or "")
        doc_number = str(meta.get("doc_number") or data.get("doc_number") or "N/A")
        doc_type = str(meta.get("doc_type") or data.get("doc_type") or "")
        effective_date = str(meta.get("effective_date") or data.get("effective_date") or "Chưa xác định")

        row = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_number": doc_number,
            "effective_date": effective_date,
            "source_doc": f"{doc_type} {doc_number}".strip(),
            "hierarchy": json.dumps(hierarchy, ensure_ascii=False),
            "original_text": original_text,
        }
        self.pending_milvus_rows.append(row)
        self.pending_milvus_texts.append(text)

        self.pending_neo_batch.append({
            "chunk_id": chunk_id,
            "original_text": original_text,
            "doc_id": doc_id,
            "doc_number": doc_number,
            "effective_date": effective_date,
            "source_doc": row["source_doc"],
            "chuong": str(hierarchy.get("chương") or "Chương N/A"),
            "dieu": str(hierarchy.get("điều") or "Điều N/A"),
        })

        if len(self.pending_milvus_rows) >= self.chunk_batch_size:
            self._flush_chunks_storage()

    def _process_raw_document(self, data: Dict[str, Any]):
        item_id = str(data.get("item_id") or data.get("id") or "")
        doc_number = str(data.get("doc_number") or item_id)
        html_raw = data.get("html_raw") or ""
        meta_api = data.get("metadata_api") or {}
        meta_detail = data.get("metadata_detail") or {}

        if not html_raw:
            return

        # Làm sạch mã HTML thô
        soup = BeautifulSoup(html_raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "button", "iframe"]):
            tag.decompose()

        main_content = soup.find("div", class_="fulltext") or soup.find("body")
        cleaned_html = str(main_content) if main_content else html_raw

        h2t = html2text.HTML2Text()
        h2t.ignore_links = True
        h2t.ignore_images = True
        h2t.body_width = 0
        raw_md = h2t.handle(cleaned_html)

        doc_type = extract_doc_type(raw_md) or str((meta_api.get("docType") or {}).get("name") or "")
        doc_num = extract_doc_number(raw_md) or doc_number
        effective_date = (
            extract_effective_date(raw_md)
            or meta_detail.get("effFrom")
            or meta_api.get("effFrom")
            or "Chưa xác định"
        )

        cleaned_md = clean_boilerplate(raw_md)
        chunks = chunk_legal_document(cleaned_md, item_id)

        for chunk in chunks:
            chunk_type = chunk.metadata.get("type", "")
            if chunk_type == "preamble" or len(chunk.text.strip()) < 50:
                continue

            hierarchy_dict = chunk.hierarchy.to_dict() if hasattr(chunk.hierarchy, "to_dict") else {}
            row = {
                "chunk_id": chunk.chunk_id,
                "doc_id": item_id,
                "doc_number": doc_num,
                "effective_date": str(effective_date),
                "source_doc": f"{doc_type} {doc_num}".strip(),
                "hierarchy": json.dumps(hierarchy_dict, ensure_ascii=False),
                "original_text": chunk.text,
            }
            self.pending_milvus_rows.append(row)
            self.pending_milvus_texts.append(chunk.text)

            self.pending_neo_batch.append({
                "chunk_id": row["chunk_id"],
                "original_text": row["original_text"],
                "doc_id": row["doc_id"],
                "doc_number": row["doc_number"],
                "effective_date": row["effective_date"],
                "source_doc": row["source_doc"],
                "chuong": str(hierarchy_dict.get("chương") or "Chương N/A"),
                "dieu": str(hierarchy_dict.get("điều") or "Điều N/A"),
            })

            if len(self.pending_milvus_rows) >= self.chunk_batch_size:
                self._flush_chunks_storage()

        if data.get("relationships"):
            self.pending_relations_batch.append({
                "item_id": item_id,
                "doc_number": doc_number,
                "metadata_api": meta_api,
                "relationships": data["relationships"],
            })

    def _flush_chunks_storage(self):
        """Tính toán vector và nạp nguyên tử vào Milvus và Neo4j."""
        if not self.pending_milvus_rows:
            return

        batch_size = len(self.pending_milvus_rows)
        vectors = self.encoder.encode_texts(self.pending_milvus_texts)

        milvus_records = []
        for row, vec in zip(self.pending_milvus_rows, vectors):
            milvus_records.append({**row, "embedding": vec})

        # Nạp Idempotent vào Milvus và Neo4j
        self.milvus_client.insert_batch(milvus_records)
        self.neo_client.insert_structural_batch(self.pending_neo_batch)

        self.pending_milvus_rows.clear()
        self.pending_milvus_texts.clear()
        self.pending_neo_batch.clear()

        logger.info(f"[BATCH UPSERT] Đã nạp thành công {batch_size} chunks vào Milvus & Neo4j.")

    def _safe_flush_and_commit(self):
        """Xả sạch toàn bộ hàng đợi và commit offset thủ công trên Kafka."""
        if not self.pending_milvus_rows and not self.pending_relations_batch and not self.pending_messages:
            return

        try:
            while self.pending_milvus_rows:
                self._flush_chunks_storage()

            if self.pending_relations_batch:
                inserted_rels = self.neo_client.insert_semantic_relations_batch(self.pending_relations_batch)
                logger.info(f"[GRAPH RELATIONS] Đã nạp {inserted_rels} cạnh quan hệ vào Neo4j.")
                self.pending_relations_batch.clear()

            # Cam kết offset thủ công có xác nhận
            if self.pending_messages:
                highest_offsets: Dict[Tuple[str, int], int] = {}
                for msg in self.pending_messages:
                    key = (msg.topic(), msg.partition())
                    highest_offsets[key] = max(highest_offsets.get(key, -1), msg.offset() + 1)

                offsets = [
                    TopicPartition(topic, partition, offset)
                    for (topic, partition), offset in highest_offsets.items()
                ]
                self.consumer.commit(offsets=offsets, asynchronous=False)
                self.pending_messages.clear()
                self.last_flush_time = time.monotonic()
                logger.info(f"[KAFKA COMMIT] Đã commit offset thành công cho {len(offsets)} partitions.")

        except Exception as exc:
            logger.error(f"[GIAO DỊCH THẤT BẠI] Lỗi nạp dữ liệu: {exc}. Giữ nguyên offset để retry!", exc_info=True)
            raise

    def run(self, idle_exit_seconds: float = 0.0):
        logger.info(f"Bắt đầu lắng nghe sự kiện từ topic [{self.topic}]...")
        idle_started_at: Optional[float] = None

        try:
            while self.running:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    now = time.monotonic()
                    if self.pending_messages and (now - self.last_flush_time >= self.flush_interval):
                        self._safe_flush_and_commit()

                    if idle_exit_seconds > 0:
                        if idle_started_at is None:
                            idle_started_at = now
                        elif now - idle_started_at >= idle_exit_seconds:
                            logger.info(f"[IDLE TIMEOUT] Không có bản tin sau {idle_exit_seconds}s. Tự động thoát.")
                            break
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Lỗi Kafka Consumer: {msg.error()}")
                    break

                idle_started_at = None
                raw_val = msg.value()

                # Giải mã phong bì nén Gzip v1 hoặc chuỗi JSON thuần
                try:
                    payload = decode_kafka_envelope(raw_val)
                    data = json.loads(payload.decode("utf-8"))
                except Exception:
                    try:
                        data = json.loads(raw_val.decode("utf-8"))
                    except Exception as err:
                        logger.error(f"Lỗi phân tích JSON tại offset {msg.offset()}: {err}")
                        continue

                self.process_message(msg, data)

        finally:
            logger.info("[SHUTDOWN] Xả nốt đệm cuối cùng và giải phóng socket...")
            try:
                self._safe_flush_and_commit()
            except Exception:
                logger.exception("Không thể hoàn tất xả đệm trước khi thoát.")
            finally:
                self.close()

    def close(self):
        try:
            self.consumer.close()
            self.milvus_client.close()
            self.neo_client.close()
            logger.info("✓ Đã giải phóng hoàn toàn tài nguyên Streaming Consumer.")
        except Exception as exc:
            logger.warning(f"Lỗi giải phóng kết nối: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Real-time Streaming Consumer")
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=float(os.getenv("CONSUMER_IDLE_EXIT_SECONDS", "0")),
        help="Thoát sau N giây nếu hàng đợi rỗng (0 nghĩa là chạy daemon liên tục)",
    )
    args = parser.parse_args()

    consumer = LawEventConsumer()
    consumer.run(idle_exit_seconds=args.idle_exit_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())