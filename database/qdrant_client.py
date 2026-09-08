"""
qdrant_client.py - Lớp giao tiếp Qdrant Vector Engine phục vụ Dense Retrieval (d=256).
Hỗ trợ HNSW Cosine Index và Native Metadata Payload Filtering theo chuẩn MRL 2026.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

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

        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=10.0)
        logger.info("Kết nối Qdrant Engine thành công tại %s:%s.", self.host, self.port)

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
            # Khởi tạo Payload index tăng tốc độ lọc thời gian thực
            self.client.create_payload_index(col_name, "is_effective", PayloadSchemaType.BOOL)
            self.client.create_payload_index(col_name, "doc_id", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(col_name, "macro_label", PayloadSchemaType.KEYWORD)
            logger.info("Đã tạo mới Collection [%s] (dim=%d) kèm Payload Indices.", col_name, dim)
        else:
            logger.info("Collection [%s] đã tồn tại và sẵn sàng.", col_name)

    def upsert_batch(
        self,
        records: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        batch_size: int = 256,
    ) -> int:
        """Đẩy dữ liệu vector d=256 kèm metadata theo từng batch nhỏ."""
        if not records:
            return 0

        col_name = collection_name or self.default_collection
        total_upserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            points = []
            for item in batch:
                vec = item["embedding"][:self.vector_dim]
                payload = {
                    "chunk_id": item.get("chunk_id", ""),
                    "doc_id": item.get("doc_id", ""),
                    "doc_number": item.get("doc_number", "N/A"),
                    "hierarchy_path": item.get("hierarchy_path", ""),
                    "macro_label": item.get("macro_label", "CHUNG"),
                    "is_effective": bool(item.get("is_effective", True)),
                    "content": item.get("text", "") or item.get("content", ""),
                }
                point_id = abs(hash(str(item["chunk_id"]))) % (2**63 - 1)
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            self.client.upsert(collection_name=col_name, points=points)
            total_upserted += len(points)

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
        query_slice = query_vector[:self.vector_dim]

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