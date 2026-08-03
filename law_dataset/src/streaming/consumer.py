"""
consumer.py - Streaming Consumer chạy nền xử lý sự kiện vi phân, nạp Milvus & Neo4j thời gian thực.
Bỏ qua JSONL trung gian, sử dụng Bulk Insert trực tiếp để tối ưu throughput.
"""

import json
import logging
import os
import sys
from confluent_kafka import Consumer, KafkaError

# Đảm bảo PYTHONPATH đúng
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from preprocess.legal_chunker import chunk_legal_document
from preprocess.text_cleaner import clean_boilerplate, extract_doc_type, extract_doc_number, extract_effective_date
from preprocess.contextualizer import build_prompt, call_ollama
from database.milvus_client import load_encoder, encode_texts, setup_milvus
from database.neo4j_client import Neo4jManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("KafkaConsumer")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")


class LawEventConsumer:
    BATCH_SIZE = 200  # Tăng batch size để tận dụng Bulk Insert

    def __init__(self, bootstrap_servers=KAFKA_BROKER):
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': "vietlawbert-consumers",
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(conf)
        self.topic = "law-documents"
        self.consumer.subscribe([self.topic])

        # Khởi tạo encoder và kết nối DB
        self.tokenizer, self.model = load_encoder()
        self.milvus_client = setup_milvus()
        self.neo_manager = Neo4jManager()

        # Buffer cho bulk insert
        self.pending_milvus_rows = []
        self.pending_milvus_texts = []
        self.pending_neo_batch = []
        self.pending_relations_batch = []

    def process_message(self, data):
        # Hỗ trợ 2 dạng message: từ contextualizer (đã có chunk) hoặc từ crawler (HTML thô)
        if "chunk_id" in data and "contextualized_text" in data:
            self._process_chunk_record(data)
        else:
            self._process_raw_document(data)

        if len(self.pending_milvus_rows) >= self.BATCH_SIZE:
            self._flush_batch()

    def _process_chunk_record(self, data):
        """Xử lý record đã được contextualize (từ contextualizer.py)."""
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

    def _process_raw_document(self, data):
        """Xử lý document thô từ crawler (HTML). Giữ backward-compatible."""
        item_id = data.get("item_id", "")
        doc_number = data.get("doc_number", item_id)
        html_raw = data.get("html_raw", "")
        meta_api = data.get("metadata_api", {})

        logger.info(f"[DOC] {doc_number} (ID: {item_id})")

        if not html_raw:
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

            prompt = build_prompt(chunk, doc_num, doc_type, effective_date)
            context = call_ollama(prompt, max_tokens=150)
            contextualized_text = f"Context: {context}\n\nContent:\n{chunk.text}" if context else chunk.text

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

        # Gom quan hệ đồ thị ngữ nghĩa từ pipeline crawler
        if data.get("relationships"):
            self.pending_relations_batch.append({
                "item_id": item_id,
                "doc_number": doc_number,
                "metadata_api": meta_api,
                "relationships": data["relationships"],
            })

    def _flush_batch(self):
        """Bulk insert dữ liệu vào Milvus, Neo4j và quan hệ ngữ nghĩa."""
        if not self.pending_milvus_rows:
            return

        try:
            vectors = encode_texts(self.tokenizer, self.model, self.pending_milvus_texts)
            milvus_data = []
            for row, vec in zip(self.pending_milvus_rows, vectors):
                row["embedding"] = vec
                milvus_data.append(row)
            self.milvus_client.insert(collection_name="vietlaw_chunks", data=milvus_data)
            logger.info(f"[BULK MILVUS] Đã nạp {len(milvus_data)} chunks.")

            self.neo_manager._insert_structural_batch(self.pending_neo_batch)
            logger.info(f"[BULK NEO4J] Đã nạp {len(self.pending_neo_batch)} nodes.")

            if self.pending_relations_batch:
                self.neo_manager.insert_semantic_relations_batch(self.pending_relations_batch)
                logger.info(f"[BULK RELATIONS] Đã nạp {len(self.pending_relations_batch)} semantic relations.")

        except Exception as e:
            logger.error(f"[BULK INSERT ERROR] {e}")
        finally:
            self.pending_milvus_rows = []
            self.pending_milvus_texts = []
            self.pending_neo_batch = []
            self.pending_relations_batch = []

    def run(self):
        logger.info("Kafka Consumer đang chờ tin nhắn từ topic 'law-documents'...")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Lỗi Consumer: {msg.error()}")
                        break

                data = json.loads(msg.value().decode('utf-8'))
                try:
                    self.process_message(data)
                except Exception as e:
                    logger.error(f"Lỗi xử lý tin nhắn: {e}")
        finally:
            self._flush_batch()
            self.consumer.close()
            self.neo_manager.close()


if __name__ == "__main__":
    c = LawEventConsumer()
    c.run()
