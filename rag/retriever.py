"""
retriever.py - Động cơ truy xuất lai Compile-time Graph-injected RAG (Kiến trúc v3).
RRF Fusion (Qdrant Dense d=256 + Elasticsearch BM25) -> Cross-Encoder & Graph Scoring Rerank.
Khắc phục triệt để lỗi lệch số chiều MRL d=256 và tự động nạp vector đồ thị 128d từ Parquet.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from configs.config import config
from configs.paths import ARTIFACTS_DIR, ROOT_DIR

logger = logging.getLogger("VietLawBERT_HybridRetriever")


def _resolve_model_source(name: str) -> str:
    """
    Phân giải nguồn mô hình BỀN VỮNG theo gốc repo (độc lập thư mục làm việc - cwd).

    - HF Hub id (vd 'BAAI/bge-m3') không tồn tại cục bộ  -> giữ nguyên (tải/offline-cache).
    - Checkpoint cục bộ (vd 'checkpoints/BAAI_bge-m3' hay './checkpoints/...') -> phân giải
      về đường dẫn TUYỆT ĐỐI theo ROOT_DIR, tránh lỗi 'Path ./checkpoints/... not found'
      khi promptfoo/pytest chạy từ thư mục con (tests/system/configs/...).
    """
    if not name:
        return name
    cand = Path(name)
    try:
        if cand.exists():
            return str(cand.resolve())
    except OSError:
        pass
    rel = name[2:] if name.startswith("./") else name
    anchored = (Path(ROOT_DIR) / rel)
    if anchored.exists():
        return str(anchored.resolve())
    return name  # coi như HF Hub id


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
                _resolve_model_source(config.BASE_MODEL_NAME),
                device=config.EMBED_DEVICE,
            )

        self.reranker = reranker_model
        self.alpha = alpha_graph if alpha_graph is not None else getattr(config, "GRAPH_ALPHA", 0.2)

        # 4. Tự động nạp cache vector đồ thị 128d phục vụ Compile-time Topo Scoring (Tối ưu hóa Vectorized Lazy Loading)
        self._graph_emb_file = Path(ARTIFACTS_DIR) / "graph_embeddings_128d.parquet"
        self._graph_cache_loaded = False

        if graph_embeddings_cache is not None:
            self.graph_cache = graph_embeddings_cache
            self._graph_cache_loaded = True
        else:
            self.graph_cache = {}

        logger.info(
            "LegalHybridRetriever sẵn sàng: Alpha=%.2f | VectorDim=%d | Qdrant=[%s] | ES=[%s]",
            self.alpha,
            self.vector_dim,
            config.QDRANT_COLLECTION_NAME,
            config.ES_INDEX_NAME,
        )

    @staticmethod
    def _graph_key(cid: str) -> str:
        """
        Chuẩn hóa chunk_id về đúng cấp của VECTOR ĐỒ THỊ để tra graph_cache.
        Đồ thị keyed ở cấp ĐIỀU ({doc}_art_N); chunk truy xuất ở cấp KHOẢN ({doc}_art_N_cl_M / _root).
        → tước hậu tố khoản về cấp điều. Khắc phục lệch granularity khiến graph_boost=0 (nhánh graph chết).
        """
        m = re.match(r"^(.*_art_\d+)", cid)
        if m:
            return m.group(1)
        return re.split(r"_(?:art|preamble)_", cid, maxsplit=1)[0]

    def _ensure_graph_cache(self) -> None:
        """Nạp lười (Lazy Load) cache vector đồ thị 128d theo cơ chế vector hóa tốc độ cao khi thực sự cần dùng."""
        if self._graph_cache_loaded:
            return

        if self._graph_emb_file.exists():
            try:
                logger.info("Đang nạp cache vector đồ thị 128d từ %s (Vectorized Fast Load)...", self._graph_emb_file.name)
                df_emb = pd.read_parquet(self._graph_emb_file, columns=["chunk_id", "graph_embedding"])
                chunk_ids = df_emb["chunk_id"].astype(str).values
                # Chuyển đổi mảng một lần thay vì duyệt từng dòng bằng iterrows()
                raw_mat = np.array(df_emb["graph_embedding"].tolist(), dtype=np.float32)
                norms = np.linalg.norm(raw_mat, axis=1, keepdims=True) + 1e-9
                norm_mat = raw_mat / norms
                self.graph_cache = dict(zip(chunk_ids, norm_mat))
                self._graph_cache_loaded = True
                logger.info("✓ Nạp thành công %d vector đồ thị vào cache truy xuất.", len(self.graph_cache))
            except Exception as exc:
                logger.warning("Không thể nạp file vector đồ thị: %s", exc)
                self._graph_cache_loaded = True
        else:
            self._graph_cache_loaded = True

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

    # Quan hệ BỔ TRỢ (Hierarchical + Operational): mở rộng để LÀM GIÀU ngữ cảnh.
    # KHÔNG mở rộng qua quan hệ đối kháng (THAY_THE/BAI_BO/...) → tránh kéo VB hết hiệu lực.
    BO_TRO_RELATIONS = {
        "CAN_CU_BAN_HANH", "QUY_DINH_CHI_TIET_HUONG_DAN", "HUONG_DAN_AP_DUNG",
        "DAN_CHIEU", "GIAI_THICH", "HOP_NHAT", "SUA_DOI_BO_SUNG",
    }

    def _get_neo4j(self):
        """Kết nối Neo4j lười cho graph-expansion (fail-soft: lỗi → bỏ qua mở rộng, không làm hỏng truy xuất)."""
        drv = getattr(self, "_neo4j_driver", None)
        if drv is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, getattr(config, "NEO4J_PASSWORD", "")),
                    connection_acquisition_timeout=10.0,
                )
                self._neo4j_driver.verify_connectivity()
            except Exception as exc:
                logger.warning("Graph-expansion: không kết nối Neo4j (%s) → bỏ qua mở rộng.", exc)
                self._neo4j_driver = False
        return self._neo4j_driver or None

    def _search_sparse_in_docs(self, query: str, doc_ids: List[str], top_k: int = 20,
                               must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """ES search GIỚI HẠN trong tập doc_id (các VB liên quan qua đồ thị) — lấy điều khớp câu hỏi."""
        if not doc_ids:
            return []
        filt: List[Dict[str, Any]] = [{"terms": {"doc_id": list(doc_ids)[:200]}}]
        if must_be_effective:
            filt.append({"term": {"is_effective": True}})
        es_query = {
            "bool": {
                "must": [{"multi_match": {
                    "query": query,
                    "fields": ["hierarchy_path^3", "content^1.5"],
                    "type": "best_fields", "tie_breaker": 0.3,
                }}],
                "filter": filt,
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
            logger.warning("Graph-expansion ES filter lỗi: %s", exc)
            return []

    def _graph_expand(self, candidates: List[Dict[str, Any]], query: str, top_m: int = 5,
                      must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """
        MỞ RỘNG pool (tăng recall): từ top-M hit tin cậy → lấy VB liên quan qua quan hệ BỔ TRỢ
        (Cypher GET_NEIGHBORS) → fetch điều khớp câu hỏi trong các VB đó (ES filter doc_id) → ứng viên bổ sung.
        Đây là hiện thực đúng "theo vết 22 quan hệ để surface điều liên quan" của Đề cương (thay re-score).
        """
        driver = self._get_neo4j()
        if driver is None:
            return []
        seed_docs, seen = [], set()
        for c in candidates[:top_m]:
            did = str(c.get("doc_id") or "")
            if did and did not in seen:
                seen.add(did); seed_docs.append(did)
        if not seed_docs:
            return []
        try:
            from database.cypher_templates import GET_NEIGHBORS
            neighbor_docs = set()
            with driver.session() as sess:
                for did in seed_docs:
                    for rec in sess.run(GET_NEIGHBORS, doc_id=did, limit=25):
                        rtype = str(rec.get("rel_type") or "").upper()
                        nid = str(rec.get("neighbor_id") or "")
                        if nid and nid not in seen and rtype in self.BO_TRO_RELATIONS:
                            neighbor_docs.add(nid)
        except Exception as exc:
            logger.warning("Graph-expansion traversal lỗi: %s", exc)
            return []
        if not neighbor_docs:
            return []
        exp = self._search_sparse_in_docs(query, list(neighbor_docs), top_k=20, must_be_effective=must_be_effective)
        for h in exp:
            h["from_expansion"] = True
        return exp

    def retrieve(self, query: str, top_k: int = 5, must_be_effective: bool = True) -> List[Dict[str, Any]]:
        """
        Truy xuất lai + GRAPH EXPANSION (Đề cương: theo vết 22 quan hệ để surface điều liên quan).
        Pipeline: Hybrid-RRF → mở rộng pool bằng điều từ VB liên quan (quan hệ BỔ TRỢ) → rerank text.
        Cơ chế graph-cosine re-score CŨ chỉ bật khi GRAPH_ALPHA>0 và GRAPH_EXPANSION=0 (mặc định: expansion).
        """
        dense_candidates = self._search_dense(query, top_k=50, must_be_effective=must_be_effective)
        sparse_candidates = self._search_sparse_es(query, top_k=50, must_be_effective=must_be_effective)
        candidates = self._rrf_fusion(dense_candidates, sparse_candidates, k=getattr(config, "RRF_K", 60), top_k=50)
        if not candidates:
            return []

        use_expansion = bool(getattr(config, "GRAPH_EXPANSION", True))
        if use_expansion:
            exp_hits = self._graph_expand(
                candidates, query,
                top_m=int(getattr(config, "GRAPH_EXPANSION_TOP_M", 5)),
                must_be_effective=must_be_effective,
            )
            if exp_hits:
                candidates = self._rrf_fusion(candidates, exp_hits, k=getattr(config, "RRF_K", 60), top_k=60)

        # Rerank bằng text score (cross-encoder nếu có, else RRF)
        if self.reranker is not None:
            pairs = [[query, c.get("content", "") or c.get("text", "")] for c in candidates]
            ce_scores = self.reranker.predict(pairs)
        else:
            ce_scores = [c.get("rrf_score", 0.0) for c in candidates]

        # (Mặc định TẮT) graph-cosine re-score cũ — chỉ khi alpha>0 và KHÔNG expansion.
        use_cosine = (self.alpha > 0) and (not use_expansion)
        anchor_g_vec = None
        if use_cosine:
            if not self._graph_cache_loaded:
                self._ensure_graph_cache()
            anchor_g_vec = self.graph_cache.get(
                self._graph_key(str(candidates[0].get("chunk_id") or candidates[0].get("id") or "")))

        final_candidates = []
        for idx, item in enumerate(candidates):
            text_score = float(ce_scores[idx]) if idx < len(ce_scores) else float(item.get("rrf_score", 0.0))
            graph_score = 0.0
            if use_cosine and anchor_g_vec is not None:
                cid = self._graph_key(str(item.get("chunk_id") or item.get("id") or ""))
                if cid in self.graph_cache:
                    cand = self.graph_cache[cid]
                    denom = (np.linalg.norm(anchor_g_vec) * np.linalg.norm(cand)) + 1e-9
                    graph_score = float(np.dot(anchor_g_vec, cand) / denom)
            item["final_rerank_score"] = round(text_score + self.alpha * graph_score, 4)
            item["graph_boost"] = round(graph_score, 4)
            final_candidates.append(item)

        final_candidates.sort(key=lambda x: x["final_rerank_score"], reverse=True)
        return final_candidates[:top_k]

    def search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(query=query, top_k=top_k)

    def close(self) -> None:
        logger.info("Đã giải phóng tài nguyên LegalHybridRetriever.")


LegalRetriever = LegalHybridRetriever