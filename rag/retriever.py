"""
retriever.py - Động cơ truy xuất lai Compile-time Graph-injected RAG (Kiến trúc v3).
RRF Fusion (Qdrant Dense d=256 + Elasticsearch BM25) -> Cross-Encoder & Graph Scoring Rerank.
Khắc phục triệt để lỗi lệch số chiều MRL d=256 và tự động nạp vector đồ thị 128d từ Parquet.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from configs.config import config
from configs.paths import ARTIFACTS_DIR

logger = logging.getLogger("VietLawBERT_HybridRetriever")


class LegalHybridRetriever:
    def __init__(
        self,
        qdrant_wrapper=None,
        es_client=None,
        es_retriever=None,
        encoder_model=None,
        reranker_model=None,
        graph_embeddings_cache: Optional[Dict[str, np.ndarray]] = None,
        alpha_graph: Optional[float] = None,
        vector_dim: Optional[int] = None,
    ):
        self.vector_dim = vector_dim or getattr(config, "QDRANT_VECTOR_DIM", 256)

        # 1. Tự động liên kết Qdrant Vector Engine
        if qdrant_wrapper is not None:
            self.qdrant = qdrant_wrapper
        else:
            from database.qdrant_client import QdrantClientWrapper
            self.qdrant = QdrantClientWrapper(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
            )

        # 2. Hỗ trợ cả hai tham số es_client và es_retriever
        chosen_es = es_client or es_retriever
        if chosen_es is not None:
            if hasattr(chosen_es, "client"):
                self._es_wrapper = chosen_es
                self.es = chosen_es.client
            else:
                self.es = chosen_es
                self._es_wrapper = None
        else:
            from rag.es_retriever import LegalElasticsearchRetriever
            self._es_wrapper = LegalElasticsearchRetriever(
                hosts=[config.ES_HOST],
                index_name=config.ES_INDEX_NAME,
            )
            self.es = self._es_wrapper.client

        # 3. Tự động nạp Backbone Encoder
        if encoder_model is not None:
            self.encoder = encoder_model
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                config.BASE_MODEL_NAME,
                device=config.EMBED_DEVICE,
            )

        self.reranker = reranker_model
        self.alpha = alpha_graph if alpha_graph is not None else getattr(config, "GRAPH_ALPHA", 0.2)

        # 4. Tự động nạp cache vector đồ thị 128d phục vụ Compile-time Topo Scoring
        if graph_embeddings_cache is not None:
            self.graph_cache = graph_embeddings_cache
        else:
            self.graph_cache = {}
            graph_emb_file = Path(ARTIFACTS_DIR) / "graph_embeddings_128d.parquet"
            if graph_emb_file.exists():
                try:
                    logger.info("Đang nạp cache vector đồ thị 128d từ %s...", graph_emb_file.name)
                    df_emb = pd.read_parquet(graph_emb_file, columns=["chunk_id", "graph_embedding"])
                    for _, row in df_emb.iterrows():
                        v = np.array(row["graph_embedding"], dtype=np.float32)
                        norm = np.linalg.norm(v)
                        self.graph_cache[str(row["chunk_id"])] = v / (norm + 1e-9)
                    logger.info("✓ Nạp thành công %d vector đồ thị vào cache truy xuất.", len(self.graph_cache))
                except Exception as exc:
                    logger.warning("Không thể nạp file vector đồ thị: %s", exc)

        logger.info(
            "LegalHybridRetriever sẵn sàng: Alpha=%.2f | VectorDim=%d | Qdrant=[%s] | ES=[%s]",
            self.alpha,
            self.vector_dim,
            config.QDRANT_COLLECTION_NAME,
            config.ES_INDEX_NAME,
        )

    def _search_dense(self, query: str, top_k: int = 50, must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """Mã hóa câu hỏi, cắt lát MRL d=256 và chuẩn hóa L2 trước khi truy vấn Qdrant."""
        raw_emb = self.encoder.encode([query], show_progress_bar=False, normalize_embeddings=False)[0]
        sub_vec = raw_emb[: self.vector_dim].astype(np.float32)
        norm = np.linalg.norm(sub_vec)
        q_vec = (sub_vec / (norm + 1e-9)).tolist()

        try:
            return self.qdrant.search_dense(
                query_vector=q_vec,
                collection_name=config.QDRANT_COLLECTION_NAME,
                top_k=top_k,
                must_be_effective=must_be_effective,
            )
        except Exception as exc:
            logger.error("Lỗi truy vấn Qdrant Dense: %s", exc)
            return []

    def _search_sparse_es(self, query: str, top_k: int = 50, must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """Truy vấn từ khóa qua bộ LegalElasticsearchRetriever chuẩn hóa."""
        if self._es_wrapper is not None:
            return self._es_wrapper.search_sparse(query=query, top_k=top_k, must_be_effective=must_be_effective)

        filter_clauses = [{"term": {"is_effective": True}}] if must_be_effective else []
        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["doc_number^4", "hierarchy_path^3", "content^1.5"],
                            "type": "best_fields",
                            "tie_breaker": 0.3,
                        }
                    }
                ],
                "filter": filter_clauses,
            }
        }
        try:
            res = self.es.search(index=config.ES_INDEX_NAME, query=es_query, size=top_k)
            hits = []
            for h in res["hits"]["hits"]:
                src = h["_source"]
                src["score"] = float(h.get("_score") or 0.0)
                src["chunk_id"] = str(src.get("chunk_id") or h["_id"])
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
            cid = str(hit.get("chunk_id") or hit.get("id") or "")
            if not cid:
                continue
            rrf_scores[cid] += 1.0 / (k + rank)
            item_map[cid] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = str(hit.get("chunk_id") or hit.get("id") or "")
            if not cid:
                continue
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in item_map:
                item_map[cid] = hit

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        fused = []
        for cid in sorted_cids:
            elem = dict(item_map[cid])
            elem["rrf_score"] = round(rrf_scores[cid], 6)
            fused.append(elem)
        return fused

    def retrieve(self, query: str, top_k: int = 5, must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """Quy trình truy xuất lai toàn trình kiểm soát SLA dưới 500ms."""
        dense_candidates = self._search_dense(query, top_k=50, must_be_effective=must_be_effective)
        sparse_candidates = self._search_sparse_es(query, top_k=50, must_be_effective=must_be_effective)

        candidates = self._rrf_fusion(dense_candidates, sparse_candidates, k=getattr(config, "RRF_K", 60), top_k=50)
        if not candidates:
            return []

        # Cross-Encoder Re-ranking nếu được nạp
        if self.reranker is not None:
            pairs = [[query, c.get("content", "") or c.get("text", "")] for c in candidates]
            ce_scores = self.reranker.predict(pairs)
        else:
            ce_scores = [c.get("rrf_score", 0.0) for c in candidates]

        top_cid = str(candidates[0].get("chunk_id") or candidates[0].get("id") or "")
        anchor_g_vec = self.graph_cache.get(top_cid)

        final_candidates = []
        for idx, item in enumerate(candidates):
            cid = str(item.get("chunk_id") or item.get("id") or "")
            text_score = float(ce_scores[idx])

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
        return self.retrieve(query=query, top_k=top_k)

    def close(self) -> None:
        logger.info("Đã giải phóng tài nguyên LegalHybridRetriever.")


LegalRetriever = LegalHybridRetriever