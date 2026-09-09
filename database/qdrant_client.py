"""
qdrant_client.py - Lớp giao tiếp Qdrant Vector Engine phục vụ Dense Retrieval (d=256).
Hỗ trợ HNSW Cosine Index, Deterministic UUIDv5, Timeout 120s và Exponential Backoff Retry.
"""

from __future__ import annotations

import time
import uuid
import logging
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from configs.config import config

logger = logging.getLogger("VietLawBERT_QdrantClient")


class QdrantClientWrapper:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
    ):
        self.host = host or getattr(config, "QDRANT_HOST", "localhost")
        self.port = port or getattr(config, "QDRANT_PORT", 6333)
        self.default_collection = getattr(config, "QDRANT_COLLECTION_NAME", "vietlawbert_chunks")
        self.vector_dim = getattr(config, "QDRANT_VECTOR_DIM", 256)

        # Nâng timeout cơ sở lên 120.0 giây
        if url:
            self.client = QdrantClient(url=url, timeout=120.0)
        else:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=120.0)

        logger.info("Kết nối Qdrant Engine thành công tại %s:%s (Timeout: 120s).", self.host, self.port)

    def init_collection(
        self,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        """Khởi tạo collection và cấu hình Payload Index cho các trường lọc vĩ mô."""
        col_name = collection_name or self.default_collection
        dim = vector_dim or self.vector_dim

        existing = [c.name for c in self.client.get_collections().collections]
        if col_name not in existing:
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            for field, schema in [
                ("is_effective", PayloadSchemaType.BOOL),
                ("doc_id", PayloadSchemaType.KEYWORD),
                ("doc_number", PayloadSchemaType.KEYWORD),
                ("macro_label", PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=col_name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            logger.info("Đã tạo mới Collection [%s] (dim=%d) kèm Payload Indices.", col_name, dim)
        else:
            logger.info("Collection [%s] đã tồn tại và sẵn sàng.", col_name)

    def upsert_batch(
        self,
        records: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        batch_size: int = 128,
        max_retries: int = 3,
    ) -> int:
        """Đẩy dữ liệu vector d=256 kèm metadata với cơ chế Retry tự phục hồi chống sập luồng."""
        if not records:
            return 0

        col_name = collection_name or self.default_collection
        total_upserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            points: List[PointStruct] = []

            for item in batch:
                vec = item.get("embedding")
                if vec is None:
                    continue

                vec_slice = vec[: self.vector_dim]
                meta = item.get("metadata", {})
                chunk_id = str(item.get("chunk_id") or meta.get("chunk_id") or uuid.uuid4().hex)
                doc_id = str(item.get("doc_id") or meta.get("doc_id") or "")
                doc_number = str(item.get("doc_number") or meta.get("doc_number") or "N/A")
                content = str(item.get("content") or item.get("text") or "")
                hierarchy_path = str(item.get("hierarchy_path") or meta.get("hierarchy_path") or "")
                macro_label = str(item.get("macro_label") or meta.get("macro_label") or "CHUNG")

                status_raw = str(item.get("status") or meta.get("status") or "")
                is_effective = bool(item.get("is_effective", "hết hiệu lực" not in status_raw.lower()))

                payload = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "hierarchy_path": hierarchy_path,
                    "macro_label": macro_label,
                    "is_effective": is_effective,
                    "content": content,
                }

                # Bảo toàn tính lũy thừa bằng UUIDv5 xác định
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                points.append(PointStruct(id=point_id, vector=vec_slice, payload=payload))

            if not points:
                continue

            # Vòng lặp Retry phòng vệ: Tự phục hồi khi gặp Timeout hoặc nghẽn I/O
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=col_name,
                        points=points,
                        wait=False,
                    )
                    total_upserted += len(points)
                    success = True
                    break
                except Exception as exc:
                    logger.warning(
                        "Batch Qdrant gặp độ trễ lớn (Thử lại %d/%d): %s. Tạm dừng %ds...",
                        attempt,
                        max_retries,
                        exc,
                        attempt * 3,
                    )
                    time.sleep(attempt * 3)

            if not success:
                logger.error("Bỏ qua batch %d điểm sau %d lần thử thất bại.", len(points), max_retries)

        logger.info("Đã nạp thành công %d bản ghi vào Qdrant [%s].", total_upserted, col_name)
        return total_upserted

    def search_dense(
        self,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 50,
        must_be_effective: bool = True,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm tương đồng ngữ nghĩa lát cắt d=256 kết hợp lọc hiệu lực."""
        col_name = collection_name or self.default_collection
        query_slice = query_vector[: self.vector_dim]

        query_filter = None
        if must_be_effective:
            query_filter = Filter(
                must=[FieldCondition(key="is_effective", match=MatchValue(value=True))]
            )

        results = self.client.search(
            collection_name=col_name,
            query_vector=query_slice,
            limit=top_k,
            query_filter=query_filter,
        )

        hits = []
        for hit in results:
            data = dict(hit.payload or {})
            data["score"] = float(hit.score)
            hits.append(data)
        return hits

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()