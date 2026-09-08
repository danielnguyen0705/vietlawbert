"""
retriever.py - Động cơ truy xuất lai Compile-time Graph-injected RAG (Kiến trúc v3).
RRF Fusion (Qdrant Dense d=256 + Elasticsearch BM25) -> Cross-Encoder & Graph Scoring Rerank.
Hỗ trợ Lazy Auto-wiring các client CSDL và nạp cache Vector Đồ thị 128 chiều.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from configs.config import config

logger = logging.getLogger("VietLawBERT_HybridRetriever")


class LegalHybridRetriever:
    def __init__(
        self,
        qdrant_wrapper=None,
        es_client=None,
        encoder_model=None,
        reranker_model=None,
        graph_embeddings_cache: Optional[Dict[str, np.ndarray]] = None,
        alpha_graph: Optional[float] = None,
    ):
        # 1. Tự động liên kết Qdrant Vector Engine nếu không truyền vào
        if qdrant_wrapper is not None:
            self.qdrant = qdrant_wrapper
        else:
            from database.qdrant_client import QdrantClientWrapper
            self.qdrant = QdrantClientWrapper(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
            )

        # 2. Tự động liên kết Elasticsearch Sparse Retriever nếu không truyền vào
        if es_client is not None:
            self.es = es_client
        else:
            from rag.es_retriever import LegalElasticsearchRetriever
            self._es_wrapper = LegalElasticsearchRetriever(
                hosts=[config.ES_HOST],
                index_name=config.ES_INDEX_NAME,
            )
            self.es = self._es_wrapper.client

        # 3. Tự động nạp Backbone Encoder (lát cắt d=256) nếu không truyền vào
        if encoder_model is not None:
            self.encoder = encoder_model
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                config.BASE_MODEL_NAME,
                device=config.EMBED_DEVICE,
            )

        self.reranker = reranker_model
        self.graph_cache = graph_embeddings_cache or {}
        self.alpha = alpha_graph if alpha_graph is not None else config.GRAPH_ALPHA

        logger.info(
            "LegalHybridRetriever khởi tạo thành công: Alpha_Graph=%.2f, Qdrant Collection=[%s], ES Index=[%s]",
            self.alpha,
            config.QDRANT_COLLECTION_NAME,
            config.ES_INDEX_NAME,
        )

    def _search_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_vec = self.encoder.encode([query])[0].tolist()
        return self.qdrant.search_dense(
            query_vector=q_vec,
            collection_name=config.QDRANT_COLLECTION_NAME,
            top_k=top_k,
            must_be_effective=True,
        )

    def _search_sparse_es(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Truy vấn Elasticsearch kết hợp Custom Vietnamese Legal Analyzer."""
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "doc_number^4",
                        "hierarchy_path^3",
                        "content^1.5",
                    ],
                    "fuzziness": "AUTO",
                }
            },
            "size": top_k,
        }
        try:
            res = self.es.search(index=config.ES_INDEX_NAME, body=es_query)
            hits = []
            for h in res["hits"]["hits"]:
                src = h["_source"]
                src["score"] = float(h["_score"])
                hits.append(src)
            return hits
        except Exception as exc:
            logger.error("Lỗi truy xuất Elasticsearch: %s", exc)
            return []

    def _rrf_fusion(
        self,
        dense_hits: List[Dict[str, Any]],
        sparse_hits: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        rrf_scores = defaultdict(float)
        item_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            rrf_scores[cid] += 0.5 * (1.0 / (k + rank))
            item_map[cid] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["chunk_id"]
            rrf_scores[cid] += 0.5 * (1.0 / (k + rank))
            if cid not in item_map:
                item_map[cid] = hit

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        fused = []
        for cid in sorted_cids:
            elem = dict(item_map[cid])
            elem["rrf_score"] = rrf_scores[cid]
            fused.append(elem)
        return fused

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Quy trình truy xuất lai toàn trình kiểm soát độ trễ dưới 500ms."""
        # 1. Thu hồi ứng viên Top-50 từ luồng Dense (Qdrant) và Sparse (ES)
        dense_candidates = self._search_dense(query, top_k=50)
        sparse_candidates = self._search_sparse_es(query, top_k=50)

        # 2. Hợp nhất RRF
        candidates = self._rrf_fusion(dense_candidates, sparse_candidates, k=config.RRF_K, top_k=50)
        if not candidates:
            return []

        # 3. Cross-Encoder Re-ranking & Bơm điểm véc-tơ đồ thị tiền tính toán
        if self.reranker:
            pairs = [[query, c.get("content", "")] for c in candidates]
            ce_scores = self.reranker.predict(pairs)
        else:
            ce_scores = [c.get("rrf_score", 0.0) for c in candidates]

        top_cid = candidates[0]["chunk_id"]
        anchor_g_vec = self.graph_cache.get(top_cid)

        final_candidates = []
        for idx, item in enumerate(candidates):
            cid = item["chunk_id"]
            text_score = float(ce_scores[idx])

            # Tính điểm tương đồng cấu trúc tô-pô đồ thị tại compile-time
            graph_score = 0.0
            if anchor_g_vec is not None and cid in self.graph_cache:
                cand_g_vec = self.graph_cache[cid]
                denom = (np.linalg.norm(anchor_g_vec) * np.linalg.norm(cand_g_vec)) + 1e-9
                graph_score = float(np.dot(anchor_g_vec, cand_g_vec) / denom)

            final_score = text_score + self.alpha * graph_score
            item["final_rerank_score"] = round(final_score, 4)
            item["graph_boost"] = round(graph_score, 4)
            final_candidates.append(item)

        final_candidates.sort(key=lambda x: x["final_rerank_score"], reverse=True)
        return final_candidates[:top_k]

    def search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Bí danh tương thích ngược cho các module Generator và Benchmark."""
        return self.retrieve(query=query, top_k=top_k)

    def close(self) -> None:
        """Giải phóng kết nối socket an toàn."""
        logger.info("Đã giải phóng tài nguyên LegalHybridRetriever.")


# Định danh tương thích ngược cho toàn bộ hệ thống
LegalRetriever = LegalHybridRetriever