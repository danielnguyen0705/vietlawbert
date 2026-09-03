"""
milvus_client.py - Tầng lưu trữ vector phân tán Milvus phục vụ Dense Retrieval.
Tích hợp kiểm soát Schema mở rộng, HNSW Indexing và Singleton Embedding Engine.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from pymilvus import MilvusClient, DataType

from configs.paths import get_log_path
from configs.config import config

logger = logging.getLogger("VietLawBERT_MilvusClient")


class EmbeddingEngine:
    """Singleton Embedding Engine tối ưu hóa CPU Inference và phòng chống rò rỉ RAM."""
    _instance: Optional[EmbeddingEngine] = None

    def __init__(self):
        self.model_name = getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.device = getattr(config, "EMBED_DEVICE", "cpu").lower()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = int(getattr(config, "EMBED_BATCH_SIZE", 16))
        self.max_length = int(getattr(config, "MAX_SEQ_LENGTH", 512))
        self.vector_dim = int(getattr(config, "EMBEDDING_DIM", 1024))

        logger.info(f"Đang khởi tạo Encoder [{self.model_name}] trên thiết bị: {self.device}")
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)

        if self.device == "cuda" and os.getenv("EMBED_FP16", "1") == "1":
            self.model.half()
        elif self.device == "cpu" and getattr(config, "EMBED_CPU_INT8", True):
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✓ Đã kích hoạt dynamic INT8 quantization cho CPU Inference.")
            except Exception as e:
                logger.warning(f"Không thể áp dụng INT8 quantization: {e}")

        self.model.eval()

    @classmethod
    def get_instance(cls) -> EmbeddingEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        ordered_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        for i in range(0, len(ordered_indices), self.batch_size):
            batch_idx = ordered_indices[i : i + self.batch_size]
            batch_texts = [texts[idx] for idx in batch_idx]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                outputs = self.model(**encoded)
                # CLS token representation kết hợp L2 normalization
                cls_rep = outputs.last_hidden_state[:, 0, :]
                norm_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)

            vectors = norm_rep.float().cpu().tolist()
            for orig_idx, vec in zip(batch_idx, vectors):
                ordered_embeddings[orig_idx] = vec

        return [v for v in ordered_embeddings if v is not None]


class MilvusClientWrapper:
    """Lớp giao tiếp chính với Milvus, hỗ trợ quản lý Schema, lập chỉ mục và tìm kiếm."""

    def __init__(
        self,
        uri: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        self.uri = uri or getattr(config, "MILVUS_URI", "http://localhost:19530")
        self.collection_name = collection_name or getattr(config, "MILVUS_COLLECTION_NAME", "vietlawbert_chunks")
        self.vector_dim = vector_dim or int(getattr(config, "EMBEDDING_DIM", 1024))
        
        logger.info(f"Kết nối Milvus RPC tại {self.uri} (Collection: {self.collection_name})")
        self.client = MilvusClient(uri=self.uri)
        self._ensure_collection()

    def _ensure_collection(self):
        """Khởi tạo Schema 8 trường metadata chuẩn hóa, mở rộng tối đa độ dài chuỗi tránh lỗi 1100."""
        if self.client.has_collection(collection_name=self.collection_name):
            logger.info(f"Collection '{self.collection_name}' đã sẵn sàng.")
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_number", DataType.VARCHAR, max_length=256)
        schema.add_field("effective_date", DataType.VARCHAR, max_length=128)
        # Mở rộng giới hạn lên 2048 ký tự giải quyết triệt để lỗi độ dài tên văn bản
        schema.add_field("source_doc", DataType.VARCHAR, max_length=2048)
        schema.add_field("hierarchy", DataType.VARCHAR, max_length=2048)
        schema.add_field("original_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.vector_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 256},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"✓ Đã tạo mới Collection '{self.collection_name}' với HNSW Cosine Index.")

    def insert_batch(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0

        sanitized_records = []
        for r in records:
            sanitized_records.append({
                "chunk_id": str(r.get("chunk_id", ""))[:250],
                "doc_id": str(r.get("doc_id", ""))[:250],
                "doc_number": str(r.get("doc_number", "N/A"))[:250],
                "effective_date": str(r.get("effective_date", "Chưa xác định"))[:120],
                "source_doc": str(r.get("source_doc", ""))[:2000],
                "hierarchy": str(r.get("hierarchy", ""))[:2000],
                "original_text": str(r.get("original_text", ""))[:65000],
                "embedding": r["embedding"],
            })

        self.client.upsert(collection_name=self.collection_name, data=sanitized_records)
        return len(sanitized_records)

    def search_dense(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=["chunk_id", "doc_id", "doc_number", "effective_date", "source_doc", "hierarchy", "original_text"],
            search_params=search_params,
        )
        if not results or not results[0]:
            return []

        hits = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            entity["score"] = hit.get("distance", 0.0)
            hits.append(entity)
        return hits

    def drop_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            logger.warning(f"Đã xóa hoàn toàn Collection '{self.collection_name}'.")

    def close(self):
        self.client.close()


# Định danh tương thích ngược
MilvusStore = MilvusClientWrapper


def ingest_from_kafka_consumer(consumer_batch_texts: List[str], consumer_batch_rows: List[Dict[str, Any]]) -> int:
    """Nạp mini-batch trực tiếp từ Kafka Consumer qua Singleton Embedding Engine."""
    if not consumer_batch_texts:
        return 0

    client_wrapper = MilvusClientWrapper()
    encoder = EmbeddingEngine.get_instance()

    try:
        vectors = encoder.encode_texts(consumer_batch_texts)
        records = []
        for row, vec in zip(consumer_batch_rows, vectors):
            row["embedding"] = vec
            records.append(row)
        inserted = client_wrapper.insert_batch(records)
        logger.info(f"[MILVUS INGESTION] Đã nạp thành công {inserted} vector chunks.")
        return inserted
    except Exception as exc:
        logger.error(f"[MILVUS INGESTION ERROR] Thất bại khi nạp batch: {exc}", exc_info=True)
        return 0