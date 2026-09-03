"""
retriever.py - Động cơ truy xuất lai 3 tầng (Dense MRL + Sparse BM25 + Neo4j Graph).
Hợp nhất thứ hạng qua thuật toán Reciprocal Rank Fusion (RRF k=60) và Re-ranking bằng Cross-Encoder.
"""

from __future__ import annotations

import os
import sys
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from neo4j import GraphDatabase
from pymilvus import MilvusClient

from configs.paths import DATA_STORAGE_ROOT, ARTIFACTS_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from database.milvus_client import EmbeddingEngine

logger = get_subsystem_logger("rag", "rag_engine")


class LegalReranker:
    """Tầng tái xếp hạng chính xác bằng mô hình Cross-Encoder (BGE-Reranker-v2-M3)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.device = device or getattr(config, "EMBED_DEVICE", "cpu")
        self.model_name = model_name
        self.model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Đang nạp mô hình Re-ranker Cross-Encoder [{self.model_name}] trên {self.device}...")
                self.model = CrossEncoder(self.model_name, device=self.device)
                self._initialized = True
                logger.info("✓ Re-ranker Cross-Encoder đã sẵn sàng.")
            except Exception as exc:
                logger.warning(f"Không thể khởi tạo Cross-Encoder ({exc}). Tự động bỏ qua bước Re-ranking.")
                self.model = None
                self._initialized = True

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        self._lazy_init()
        if not self.model or not candidates:
            return candidates[:top_k]

        pairs = [[query, c.get("content", "")] for c in candidates]
        try:
            scores = self.model.predict(pairs)
            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked[:top_k]
        except Exception as exc:
            logger.error(f"Lỗi trong quá trình tính điểm Re-ranking: {exc}")
            return candidates[:top_k]


class LegalRetriever:
    """Động cơ truy xuất tri thức pháp luật lai phục vụ bài toán tra cứu chính xác cao."""

    def __init__(self, use_reranker: bool = False):
        logger.info("Khởi tạo Động cơ Truy xuất Lai VietLawBERT...")
        self.collection_name = getattr(config, "MILVUS_COLLECTION_NAME", "vietlawbert_chunks")
        self.milvus_uri = getattr(config, "MILVUS_URI", "http://localhost:19530")

        self.encoder = EmbeddingEngine.get_instance()
        self.milvus_client = MilvusClient(uri=self.milvus_uri)
        self.neo4j_driver = GraphDatabase.driver(
            getattr(config, "NEO4J_URI", "bolt://localhost:7687"),
            auth=(getattr(config, "NEO4J_USER", "neo4j"), getattr(config, "NEO4J_PASSWORD", "vietlawbert")),
        )

        self.use_reranker = use_reranker
        self.reranker = LegalReranker() if use_reranker else None

        self._cached_bm25 = None
        self._cached_chunks = None

        logger.info("✓ LegalRetriever đã sẵn sàng tiếp nhận truy vấn.")

    def _search_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            query_vectors = self.encoder.encode_texts([query])
            if not query_vectors:
                return []

            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=query_vectors,
                limit=top_k,
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                output_fields=[
                    "chunk_id",
                    "doc_id",
                    "doc_number",
                    "effective_date",
                    "source_doc",
                    "hierarchy",
                    "original_text",
                ],
            )

            hits = []
            if results and len(results) > 0:
                for hit in results[0]:
                    meta = hit.get("entity", {})
                    hierarchy_raw = meta.get("hierarchy", "{}")
                    try:
                        hierarchy_dict = json.loads(hierarchy_raw) if isinstance(hierarchy_raw, str) else hierarchy_raw
                    except Exception:
                        hierarchy_dict = {}

                    dieu = hierarchy_dict.get("điều") or "Điều N/A"
                    hits.append({
                        "chunk_id": str(meta.get("chunk_id", "")),
                        "doc_id": str(meta.get("doc_id", "")),
                        "doc_number": str(meta.get("doc_number", "N/A")),
                        "source_doc": str(meta.get("source_doc", "")),
                        "effective_date": str(meta.get("effective_date", "Chưa xác định")),
                        "article": dieu,
                        "content": str(meta.get("original_text", "")),
                        "retrieval_source": "dense_milvus",
                    })
            return hits
        except Exception as exc:
            logger.error(f"Lỗi truy vấn Dense Milvus: {exc}")
            return []

    def _init_bm25_corpus(self):
        if self._cached_bm25 is not None:
            return

        corpus_candidates = [
            DATA_STORAGE_ROOT / "processed" / "contextual_chunks.jsonl",
            ARTIFACTS_DIR / "contextual_chunks.jsonl",
        ]

        target_file = None
        for cand in corpus_candidates:
            if cand.exists():
                target_file = cand
                break

        if not target_file:
            return

        try:
            chunks = []
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        try:
                            chunks.append(json.loads(clean_line))
                        except Exception:
                            continue

            if chunks:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [(c.get("contextualized_text") or c.get("original_text", "")).lower().split() for c in chunks]
                self._cached_bm25 = BM25Okapi(tokenized_corpus)
                self._cached_chunks = chunks
                logger.info(f"✓ Đã xây dựng bộ chỉ mục BM25 từ {len(chunks)} chunks.")
        except Exception as exc:
            logger.warning(f"Không thể khởi tạo BM25 từ tệp đĩa: {exc}")

    def _search_sparse_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._init_bm25_corpus()
        if self._cached_bm25 is None or not self._cached_chunks:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._cached_bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]

            hits = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                ck = self._cached_chunks[idx]
                meta = ck.get("metadata", {})
                hierarchy = meta.get("hierarchy_path", {})

                hits.append({
                    "chunk_id": str(ck.get("chunk_id", "")),
                    "doc_id": str(meta.get("doc_id", "")),
                    "doc_number": str(meta.get("doc_number", "N/A")),
                    "source_doc": str(meta.get("source_doc", "")),
                    "effective_date": str(meta.get("effective_date", "Chưa xác định")),
                    "article": str(hierarchy.get("điều") or "Điều N/A"),
                    "content": str(ck.get("original_text", "")),
                    "retrieval_source": "sparse_bm25",
                })
            return hits
        except Exception as exc:
            logger.error(f"Lỗi truy vấn BM25: {exc}")
            return []

    def _search_exact_neo4j(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        match_doc = re.search(r"(\d+/\d+/[A-ZĐa-z0-9\-]+)", query)
        match_art = re.search(r"Điều\s+(\d+[a-zA-Z]?)", query, re.IGNORECASE)

        params: Dict[str, Any] = {"limit": top_k}

        if match_doc and match_art:
            cypher = """
            MATCH (doc:LawDocument)
            WHERE doc.doc_number CONTAINS $doc_num OR doc.name CONTAINS $doc_num
            MATCH (doc)-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(art:Article)
            WHERE art.name = $art_name
            MATCH (art)-[:HAS_CHUNK]->(ck:Chunk)
            RETURN doc.doc_id AS DocId,
                   doc.name AS DocName,
                   doc.doc_number AS DocNum,
                   doc.effective_date AS EffDate,
                   art.name AS Article,
                   ck.chunk_id AS ChunkId,
                   ck.text AS Content
            LIMIT $limit
            """
            params["doc_num"] = match_doc.group(1).upper()
            params["art_name"] = f"Điều {match_art.group(1)}"

        elif match_art:
            cypher = """
            MATCH (doc:LawDocument)-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(art:Article)
            WHERE art.name = $art_name
            MATCH (art)-[:HAS_CHUNK]->(ck:Chunk)
            RETURN doc.doc_id AS DocId,
                   doc.name AS DocName,
                   doc.doc_number AS DocNum,
                   doc.effective_date AS EffDate,
                   art.name AS Article,
                   ck.chunk_id AS ChunkId,
                   ck.text AS Content
            LIMIT $limit
            """
            params["art_name"] = f"Điều {match_art.group(1)}"

        elif match_doc:
            cypher = """
            MATCH (doc:LawDocument)
            WHERE doc.doc_number CONTAINS $doc_num OR doc.name CONTAINS $doc_num
            MATCH (doc)-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(art:Article)-[:HAS_CHUNK]->(ck:Chunk)
            RETURN doc.doc_id AS DocId,
                   doc.name AS DocName,
                   doc.doc_number AS DocNum,
                   doc.effective_date AS EffDate,
                   art.name AS Article,
                   ck.chunk_id AS ChunkId,
                   ck.text AS Content
            LIMIT $limit
            """
            params["doc_num"] = match_doc.group(1).upper()
        else:
            return []

        hits = []
        try:
            with self.neo4j_driver.session() as session:
                results = session.run(cypher, parameters=params).data()
                for record in results:
                    hits.append({
                        "chunk_id": str(record.get("ChunkId", "")),
                        "doc_id": str(record.get("DocId", "")),
                        "doc_number": str(record.get("DocNum", "N/A")),
                        "source_doc": str(record.get("DocName", "")),
                        "effective_date": str(record.get("EffDate", "Chưa xác định")),
                        "article": str(record.get("Article", "Điều N/A")),
                        "content": str(record.get("Content", "")),
                        "retrieval_source": "graph_neo4j",
                    })
        except Exception as exc:
            logger.error(f"Lỗi truy vấn Neo4j Graph: {exc}")

        return hits

    @staticmethod
    def _rrf_fusion(
        result_lists: List[List[Dict[str, Any]]],
        weights: Tuple[float, ...] = (0.5, 0.25, 0.25),
        k_param: int = 60,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        rrf_scores = defaultdict(float)
        docs_cache: Dict[str, Dict[str, Any]] = {}

        for list_idx, hits in enumerate(result_lists):
            w = weights[list_idx] if list_idx < len(weights) else 1.0
            for rank, hit in enumerate(hits, start=1):
                cid = hit.get("chunk_id")
                if not cid:
                    continue
                rrf_scores[cid] += w * (1.0 / (float(k_param) + float(rank)))
                if cid not in docs_cache:
                    docs_cache[cid] = hit

        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]

        fused_results = []
        for cid in sorted_cids:
            doc_item = dict(docs_cache[cid])
            doc_item["rrf_score"] = round(rrf_scores[cid], 5)
            doc_item["doc_info"] = f"{doc_item.get('source_doc', '')} (Số: {doc_item.get('doc_number', 'N/A')})"
            fused_results.append(doc_item)

        return fused_results

    def search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        candidate_k = max(top_k * 5, 20)

        dense_hits = self._search_dense(clean_query, top_k=candidate_k)
        sparse_hits = self._search_sparse_bm25(clean_query, top_k=candidate_k)
        exact_hits = self._search_exact_neo4j(clean_query, top_k=candidate_k)

        fused_candidates = self._rrf_fusion(
            [dense_hits, sparse_hits, exact_hits],
            weights=(0.5, 0.25, 0.25),
            k_param=int(getattr(config, "RRF_K", 60)),
            top_k=candidate_k,
        )

        if self.use_reranker and self.reranker:
            return self.reranker.rerank(clean_query, fused_candidates, top_k=top_k)

        return fused_candidates[:top_k]

    def close(self):
        try:
            self.milvus_client.close()
            self.neo4j_driver.close()
            logger.info("✓ Đã đóng kết nối Milvus và Neo4j an toàn.")
        except Exception:
            pass