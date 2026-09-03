"""
embedding_consumer.py - Kafka Streaming Consumer nạp đồng bộ Vector & Graph RAG.
Tối ưu hóa thông lượng với Bulk Upsert, phân đoạn văn bản AST và cam kết offset nguyên tử.
"""

from __future__ import annotations

import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from confluent_kafka import Consumer, KafkaError, TopicPartition
from bs4 import BeautifulSoup
import html2text

from configs.paths import ROOT_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from database.milvus_client import MilvusClientWrapper, EmbeddingEngine
from database.neo4j_client import Neo4jClient
from artifacts.canonical import decode_kafka_envelope
from preprocess.legal_chunker import chunk_legal_document
from preprocess.text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
)

logger = get_subsystem_logger("streaming", "ingestion")


class LawEventConsumer:
    """Consumer xử lý luồng văn bản pháp luật, phân đoạn và nạp song song vào Milvus và Neo4j."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or getattr(config, "KAFKA_BROKER", "localhost:9092")
        self.group_id = group_id or getattr(config, "KAFKA_GROUP_ID", "vietlawbert-consumers-v5-bounded")
        self.topic = topic or getattr(config, "KAFKA_TOPIC", "law-documents-v5")

        self.doc_commit_batch_size = int(getattr(config, "CONSUMER_DOC_BATCH_SIZE", 10))
        self.chunk_write_batch_size = int(getattr(config, "CONSUMER_CHUNK_BATCH_SIZE", 64))
        self.flush_interval_seconds = float(getattr(config, "CONSUMER_FLUSH_INTERVAL_SECONDS", 5.0))

        conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "session.timeout.ms": 60000,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "1800000")),
        }
        self.consumer = Consumer(conf)
        self.consumer.subscribe([self.topic])

        logger.info(f"Kết nối Kafka Ingestion Consumer tới topic [{self.topic}] (Group: {self.group_id})")

        self.milvus_client = MilvusClientWrapper()
        self.encoder = EmbeddingEngine.get_instance()
        self.neo_client = Neo4jClient()

        self.pending_milvus_rows: List[Dict[str, Any]] = []
        self.pending_milvus_texts: List[str] = []
        self.pending_neo_batch: List[Dict[str, Any]] = []
        self.pending_relations_batch: List[Dict[str, Any]] = []
        self.pending_messages: List[Any] = []

        self.last_flush_at = time.monotonic()
        self.documents_processed = 0
        self.chunks_written = 0

    def process_message(self, msg: Any, data: Dict[str, Any]):
        """Phân loại dữ liệu đầu vào: bản ghi đã chunk sẵn hoặc văn bản thô cần phân rã AST."""
        if "chunk_id" in data and ("contextualized_text" in data or "original_text" in data):
            self._process_chunk_record(data)
        else:
            self._process_raw_document(data)

        self.pending_messages.append(msg)
        self.documents_processed += 1

        if len(self.pending_messages) >= self.doc_commit_batch_size:
            self._flush_batch()

    def _process_chunk_record(self, data: Dict[str, Any]):
        chunk_id = str(data["chunk_id"])
        meta = data.get("metadata") or {}
        hierarchy = meta.get("hierarchy_path") or {}
        contextualized_text = data.get("contextualized_text") or data.get("original_text", "")
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
        self.pending_milvus_texts.append(contextualized_text)

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
        self._flush_chunks_if_full()

    def _process_raw_document(self, data: Dict[str, Any]):
        item_id = str(data.get("item_id") or data.get("id") or "")
        doc_number = str(data.get("doc_number") or item_id)
        html_raw = data.get("html_raw") or ""
        meta_api = data.get("metadata_api") or {}
        meta_detail = data.get("metadata_detail") or {}

        if not html_raw:
            logger.warning(f"[BỎ QUA] Văn bản {doc_number} (ID: {item_id}) không có nội dung HTML.")
            return

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
            self._flush_chunks_if_full()

        if data.get("relationships"):
            self.pending_relations_batch.append({
                "item_id": item_id,
                "doc_number": doc_number,
                "metadata_api": meta_api,
                "relationships": data["relationships"],
            })

    def _flush_chunks_if_full(self):
        if len(self.pending_milvus_rows) >= self.chunk_write_batch_size:
            self._flush_chunk_storage(self.chunk_write_batch_size)

    def _flush_chunk_storage(self, count: Optional[int] = None):
        if not self.pending_milvus_rows:
            return

        batch_count = min(count or len(self.pending_milvus_rows), len(self.pending_milvus_rows))
        rows = self.pending_milvus_rows[:batch_count]
        texts = self.pending_milvus_texts[:batch_count]
        neo_rows = self.pending_neo_batch[:batch_count]

        started_at = time.monotonic()
        vectors = self.encoder.encode_texts(texts)

        milvus_data = []
        for row, vec in zip(rows, vectors):
            milvus_data.append({**row, "embedding": vec})

        self.milvus_client.insert_batch(milvus_data)
        self.neo_client.insert_structural_batch(neo_rows)

        del self.pending_milvus_rows[:batch_count]
        del self.pending_milvus_texts[:batch_count]
        del self.pending_neo_batch[:batch_count]

        self.chunks_written += batch_count
        elapsed = time.monotonic() - started_at
        rate = batch_count / elapsed if elapsed > 0 else 0.0
        logger.info(f"[CHUNK BULK] Đã nạp {batch_count} chunks (Tốc độ: {rate:.2f} chunks/s, Tổng: {self.chunks_written})")

    def _commit_pending_messages(self):
        if not self.pending_messages:
            return

        highest_offsets: Dict[Tuple[str, int], int] = {}
        for msg in self.pending_messages:
            key = (msg.topic(), msg.partition())
            highest_offsets[key] = max(highest_offsets.get(key, -1), msg.offset() + 1)

        offsets = [
            TopicPartition(topic, partition, offset)
            for (topic, partition), offset in highest_offsets.items()
        ]
        self.consumer.commit(offsets=offsets, asynchronous=False)
        logger.info(f"[KAFKA COMMIT] Đã commit thành công {len(self.pending_messages)} messages trên {len(offsets)} partitions.")

    def _flush_batch(self):
        """Xả sạch toàn bộ bộ đệm RAM và chỉ commit Kafka offset khi mọi DB đã hoàn tất."""
        if not self.pending_milvus_rows and not self.pending_neo_batch and not self.pending_relations_batch:
            if self.pending_messages:
                self._commit_pending_messages()
                self.pending_messages = []
                self.last_flush_at = time.monotonic()
            return

        try:
            while self.pending_milvus_rows:
                self._flush_chunk_storage(self.chunk_write_batch_size)

            if self.pending_relations_batch:
                inserted_rels = self.neo_client.insert_semantic_relations_batch(self.pending_relations_batch)
                logger.info(f"[ĐỒ THỊ] Đã nạp {inserted_rels} quan hệ ngữ nghĩa mới vào Neo4j.")

            if self.pending_messages:
                self._commit_pending_messages()

        except Exception as exc:
            logger.error(f"[LỖI BULK INSERT] Ghi CSDL thất bại: {exc}. Giữ nguyên đệm để thử lại, KHÔNG commit offset!", exc_info=True)
            raise
        else:
            self.pending_milvus_rows.clear()
            self.pending_milvus_texts.clear()
            self.pending_neo_batch.clear()
            self.pending_relations_batch.clear()
            self.pending_messages.clear()
            self.last_flush_at = time.monotonic()

    def run(self, idle_exit_seconds: float = 0.0):
        logger.info("Kafka Ingestion Consumer đang hoạt động và chờ nhận dữ liệu...")
        idle_started_at: Optional[float] = None
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            logger.info(f"[DỪNG TIẾN TRÌNH] Bắt tín hiệu {signum}. Đang xả nốt bộ đệm...")

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        try:
            while not stop_requested:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    now = time.monotonic()
                    if self.pending_messages and (now - self.last_flush_at >= self.flush_interval_seconds):
                        logger.info("[ĐẾM GIỜ XẢ ĐỆM] Kafka tạm hết thông điệp, xả đệm tồn dư.")
                        self._flush_batch()

                    if idle_exit_seconds > 0:
                        if idle_started_at is None:
                            idle_started_at = now
                        elif now - idle_started_at >= idle_exit_seconds:
                            logger.info(f"[HÀNG ĐỢI RỖNG] Không có thêm message sau {idle_exit_seconds:.1f}s. Tự động thoát Consumer.")
                            break
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Lỗi phân vùng Kafka: {msg.error()}")
                    raise RuntimeError(str(msg.error()))

                idle_started_at = None
                raw_val = msg.value()

                try:
                    payload = decode_kafka_envelope(raw_val)
                    data = json.loads(payload.decode("utf-8"))
                except Exception:
                    try:
                        data = json.loads(raw_val.decode("utf-8"))
                    except Exception as err:
                        logger.error(f"Không thể giải mã bản tin tại offset {msg.offset()}: {err}")
                        continue

                self.process_message(msg, data)
        finally:
            try:
                self._flush_batch()
            except Exception:
                logger.exception("Không thể hoàn tất xả đệm trước khi tắt; offset chưa cam kết sẽ được replay an toàn.")
            finally:
                self.close()

    def close(self):
        """Giải phóng toàn bộ tài nguyên kết nối socket."""
        try:
            self.consumer.close()
            self.milvus_client.close()
            self.neo_client.close()
            logger.info("✓ Đã đóng toàn bộ kết nối Consumer, Milvus và Neo4j an toàn.")
        except Exception as e:
            logger.warning(f"Cảnh báo khi giải phóng tài nguyên: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Kafka Embedding Consumer")
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=float(os.getenv("CONSUMER_IDLE_EXIT_SECONDS", "0")),
        help="Thời gian tự ngắt khi cạn message (0 là chế độ daemon)",
    )
    args = parser.parse_args()

    consumer = LawEventConsumer()
    consumer.run(idle_exit_seconds=args.idle_exit_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())