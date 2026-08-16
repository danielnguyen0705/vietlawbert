"""
consumer.py - Streaming Consumer chạy nền xử lý sự kiện vi phân, nạp Milvus & Neo4j thời gian thực.
Bỏ qua JSONL trung gian, sử dụng Bulk Insert trực tiếp để tối ưu throughput.
"""

import json
import logging
import os
import signal
import sys
import time
from confluent_kafka import Consumer, KafkaError, TopicPartition

# Đảm bảo PYTHONPATH đúng
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.legal_chunker import chunk_legal_document
from preprocess.text_cleaner import clean_boilerplate, extract_doc_type, extract_doc_number, extract_effective_date
from database.milvus_client import load_encoder, encode_texts, setup_milvus
from database.neo4j_client import Neo4jManager

# Khai báo logger toàn cục
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("KafkaConsumer")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "vietlawbert-consumers-v5-bounded")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "law-documents-v5")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "vietlaw_chunks")


class LawEventConsumer:
    DOC_COMMIT_BATCH_SIZE = int(os.getenv("CONSUMER_DOC_BATCH_SIZE", "10"))
    CHUNK_WRITE_BATCH_SIZE = int(os.getenv("CONSUMER_CHUNK_BATCH_SIZE", "64"))
    FLUSH_INTERVAL_SECONDS = float(os.getenv("CONSUMER_FLUSH_INTERVAL_SECONDS", "5"))

    def __init__(self, bootstrap_servers=KAFKA_BROKER, group_id=KAFKA_GROUP_ID):
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'session.timeout.ms': 60000,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
            'max.poll.interval.ms': int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "1800000")),
        }
        self.consumer = Consumer(conf)
        self.topic = KAFKA_TOPIC
        self.consumer.subscribe([self.topic])

        self.tokenizer, self.model = load_encoder()
        self.milvus_client = setup_milvus()
        self.neo_manager = Neo4jManager()

        self.pending_milvus_rows = []
        self.pending_milvus_texts = []
        self.pending_neo_batch = []
        self.pending_relations_batch = []
        self.pending_messages = []
        self.last_flush_at = time.monotonic()
        self.documents_processed = 0
        self.chunks_written = 0

    def process_message(self, msg, data):
        if "chunk_id" in data and "contextualized_text" in data:
            self._process_chunk_record(data)
        else:
            self._process_raw_document(data)

        # Chỉ đánh dấu message hoàn tất sau khi parse/chunk không phát sinh lỗi.
        # Nếu có exception, process dừng để lần chạy sau replay từ offset đã commit.
        self.pending_messages.append(msg)
        self.documents_processed += 1

        if len(self.pending_messages) >= self.DOC_COMMIT_BATCH_SIZE:
            self._flush_batch()

    def _process_chunk_record(self, data):
        chunk_id = data["chunk_id"]
        meta = data.get("metadata", {})
        hierarchy = meta.get("hierarchy_path", {})
        contextualized_text = data.get("contextualized_text") or data.get("original_text", "")
        original_text = data.get("original_text", "")
        doc_id = meta.get("doc_id", "")
        doc_number = meta.get("doc_number", "N/A")
        doc_type = meta.get("doc_type", "")
        effective_date = meta.get("effective_date", "Chua xac dinh")

        logger.info(f"[CHUNK] {chunk_id} - Doc: {doc_number}")

        row = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_number": doc_number,
            "effective_date": effective_date,
            "source_doc": f"{doc_type} {doc_number}",
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
            "source_doc": f"{doc_type} {doc_number}",
            "chuong": hierarchy.get("chương") or "Chương N/A",
            "dieu": hierarchy.get("điều") or "Điều N/A"
        })
        self._flush_chunks_if_full()

    def _process_raw_document(self, data):
        item_id = data.get("item_id", "")
        doc_number = data.get("doc_number", item_id)
        html_raw = data.get("html_raw", "")
        meta_api = data.get("metadata_api", {})

        logger.info(f"[DOC] {doc_number} (ID: {item_id})")

        if not html_raw:
            logger.warning(f"[DOC] {doc_number} (ID: {item_id}) khong co HTML, bo qua.")
            return

        from bs4 import BeautifulSoup
        import html2text
        soup = BeautifulSoup(html_raw, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'button', 'iframe']):
            tag.decompose()
        main_content = soup.find('div', class_='fulltext') or soup.find('body')
        cleaned_html = str(main_content) if main_content else html_raw

        h2t = html2text.HTML2Text()
        h2t.ignore_links = True
        h2t.ignore_images = True
        h2t.body_width = 0
        raw_md = h2t.handle(cleaned_html)

        doc_type = extract_doc_type(raw_md)
        doc_num = extract_doc_number(raw_md) or doc_number
        effective_date = extract_effective_date(raw_md) or meta_api.get('effFrom', 'Chua xac dinh')

        cleaned_md = clean_boilerplate(raw_md)
        chunks = chunk_legal_document(cleaned_md, item_id)

        for chunk in chunks:
            if chunk.metadata.get('type') == 'preamble':
                continue
            if len(chunk.text.strip()) < 50:
                continue

            # Bỏ sinh context bằng LLM để tăng tốc độ (chỉ dùng LLM cho OCR PDF sau này)
            contextualized_text = chunk.text

            hierarchy_dict = chunk.hierarchy.to_dict()
            row = {
                "chunk_id": chunk.chunk_id,
                "doc_id": item_id,
                "doc_number": doc_num,
                "effective_date": effective_date,
                "source_doc": f"{doc_type} {doc_num}",
                "hierarchy": json.dumps(hierarchy_dict, ensure_ascii=False),
                "original_text": chunk.text,
            }
            self.pending_milvus_rows.append(row)
            self.pending_milvus_texts.append(contextualized_text)

            self.pending_neo_batch.append({
                "chunk_id": row["chunk_id"],
                "original_text": row["original_text"],
                "doc_id": row["doc_id"],
                "doc_number": row["doc_number"],
                "effective_date": row["effective_date"],
                "source_doc": row["source_doc"],
                "chuong": hierarchy_dict.get("chương") or "Chương N/A",
                "dieu": hierarchy_dict.get("điều") or "Điều N/A"
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
        if len(self.pending_milvus_rows) >= self.CHUNK_WRITE_BATCH_SIZE:
            self._flush_chunk_storage(self.CHUNK_WRITE_BATCH_SIZE)

    def _flush_chunk_storage(self, count=None):
        if not self.pending_milvus_rows:
            return

        count = min(count or len(self.pending_milvus_rows), len(self.pending_milvus_rows))
        rows = self.pending_milvus_rows[:count]
        texts = self.pending_milvus_texts[:count]
        neo_rows = self.pending_neo_batch[:count]
        started_at = time.monotonic()

        vectors = encode_texts(self.tokenizer, self.model, texts)
        milvus_data = []
        for row, vec in zip(rows, vectors):
            milvus_data.append({**row, "embedding": vec})

        # Hai thao tác đều idempotent. Nếu Neo4j lỗi sau Milvus, replay/upsert là an toàn.
        self.milvus_client.upsert(collection_name=MILVUS_COLLECTION, data=milvus_data)
        self.neo_manager._insert_structural_batch(neo_rows)

        del self.pending_milvus_rows[:count]
        del self.pending_milvus_texts[:count]
        del self.pending_neo_batch[:count]
        self.chunks_written += count
        elapsed = time.monotonic() - started_at
        logger.info(
            "[CHUNK BATCH] Đã upsert %d chunks trong %.2fs (%.2f chunks/s, tổng=%d).",
            count,
            elapsed,
            count / elapsed if elapsed else 0.0,
            self.chunks_written,
        )

    def _commit_pending_messages(self):
        if not self.pending_messages:
            return
        highest_offsets = {}
        for message in self.pending_messages:
            key = (message.topic(), message.partition())
            highest_offsets[key] = max(highest_offsets.get(key, -1), message.offset() + 1)
        offsets = [
            TopicPartition(topic, partition, offset)
            for (topic, partition), offset in highest_offsets.items()
        ]
        self.consumer.commit(offsets=offsets, asynchronous=False)
        logger.info(
            "[KAFKA COMMIT] Đã commit %d message trên %d partition.",
            len(self.pending_messages),
            len(offsets),
        )

    def _flush_batch(self):
        if not self.pending_milvus_rows and not self.pending_neo_batch and not self.pending_relations_batch:
            if self.pending_messages:
                self._commit_pending_messages()
                self.pending_messages = []
                self.last_flush_at = time.monotonic()
            return

        try:
            while self.pending_milvus_rows:
                self._flush_chunk_storage(self.CHUNK_WRITE_BATCH_SIZE)

            if self.pending_relations_batch:
                self.neo_manager.insert_semantic_relations_batch(self.pending_relations_batch)
                logger.info(f"[BULK RELATIONS] Đã nạp {len(self.pending_relations_batch)} semantic relations.")

            # CHỈ COMMIT KHI CẢ MILVUS VÀ NEO4J ĐÃ THÀNH CÔNG HÀN HOÀN
            if self.pending_messages:
                self._commit_pending_messages()

        except Exception:
            logger.exception("[BULK INSERT ERROR]")
            raise
        else:
            self.pending_milvus_rows = []
            self.pending_milvus_texts = []
            self.pending_neo_batch = []
            self.pending_relations_batch = []
            self.pending_messages = []
            self.last_flush_at = time.monotonic()

    def run(self, idle_exit_seconds=0):
        logger.info("Kafka Consumer đang chờ tin nhắn...")
        idle_started_at = None
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            logger.info("[SHUTDOWN] Nhận signal %s; sẽ dừng sau record hiện tại.", signum)

        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
        try:
            while not stop_requested:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    now = time.monotonic()
                    if self.pending_messages and now - self.last_flush_at >= self.FLUSH_INTERVAL_SECONDS:
                        logger.info("[IDLE FLUSH] Kafka tạm hết message, flush batch đang chờ.")
                        self._flush_batch()
                    if idle_exit_seconds > 0:
                        idle_started_at = idle_started_at or now
                        if now - idle_started_at >= idle_exit_seconds:
                            logger.info("[IDLE EXIT] Không còn message trong %.1fs, dừng consumer.", idle_exit_seconds)
                            break
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF: continue
                    logger.error(f"Lỗi Consumer: {msg.error()}")
                    raise RuntimeError(str(msg.error()))

                idle_started_at = None
                data = json.loads(msg.value().decode('utf-8'))
                self.process_message(msg, data)
        finally:
            try:
                self._flush_batch()
            except Exception:
                logger.exception("Không thể flush batch còn lại; offset chưa được commit")
            finally:
                self.consumer.close()
                self.neo_manager.close()
                signal.signal(signal.SIGINT, previous_sigint)
                signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VietLawBERT Kafka ingestion consumer")
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=float(os.getenv("CONSUMER_IDLE_EXIT_SECONDS", "0")),
        help="Thoát sau N giây Kafka không có message; 0 nghĩa là chạy daemon.",
    )
    args = parser.parse_args()
    c = LawEventConsumer()
    c.run(idle_exit_seconds=args.idle_exit_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
