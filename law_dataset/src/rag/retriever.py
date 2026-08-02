"""
retriever.py - Truy xuất Hybrid (Vector + Graph)

Dùng BGE-M3 embedding + Milvus + Neo4j.
"""

import os
import sys
import logging
import re
import json
from collections import defaultdict
from datetime import datetime
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from config import config

from paths import BASE_DIR, get_log_path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = get_log_path(CURRENT_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(CURRENT_FILENAME.capitalize())


def get_device() -> str:
    import torch
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LegalRetriever:
    def __init__(self):
        logger.info("Khoi tao Retriever...")
        self.device = get_device()
        logger.info(f"Su dung device: {self.device}")

        self.encoder = SentenceTransformer('BAAI/bge-m3', device=self.device)

        # Su dung MilvusClient v3 dong bo voi milvus_client.py
        self.milvus_client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")

        self.neo4j_driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        logger.info("Retriever da san sang!")

    def _search_dense(self, query: str, top_k: int) -> list[dict]:
        """Tầng 1: Dense Semantic Search (Milvus)."""
        try:
            # Encode query
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

            results = self.milvus_client.search(
                collection_name="vietlaw_chunks",
                data=[query_vector],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                output_fields=["chunk_id", "original_text", "doc_number", "effective_date", "hierarchy"]
            )

            hits = []
            for hit in results[0]:
                meta = hit.get("entity", {})
                hierarchy_dict = json.loads(meta.get("hierarchy", "{}"))
                dieu = hierarchy_dict.get("điều") or "Điều N/A"
                hits.append({
                    "chunk_id": meta.get("chunk_id"),
                    "doc_info": f"Văn bản số {meta.get('doc_number')}",
                    "effective_date": meta.get("effective_date", "Chưa xác định"),
                    "article": dieu,
                    "content": meta.get("original_text")
                })
            return hits
        except Exception as e:
            logger.error(f"Lỗi Milvus Dense search: {e}")
            return []

    def _search_sparse_bm25(self, query: str, top_k: int) -> list[dict]:
        """Tầng 2: Sparse Semantic Search (BM25 từ file local JSONL để tránh setup Elasticsearch phức tạp)."""
        from paths import CONTEXTUAL_CHUNKS_FILE
        if not os.path.exists(CONTEXTUAL_CHUNKS_FILE):
            return []

        try:
            # Tối ưu hóa: Cache corpus trong RAM để tránh I/O bottleneck
            if not hasattr(self, '_cached_bm25'):
                chunks = []
                with open(CONTEXTUAL_CHUNKS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            chunks.append(json.loads(line))
                self._cached_chunks = chunks
                tokenized_corpus = [c["contextualized_text"].split() for c in chunks]
                from rank_bm25 import BM25Okapi
                self._cached_bm25 = BM25Okapi(tokenized_corpus)

            scores = self._cached_bm25.get_scores(query.split())
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:top_k]

            hits = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                ck = self._cached_chunks[idx]
                meta = ck.get("metadata", {})
                dieu = (meta.get("hierarchy_path") or {}).get("điều") or "Điều N/A"
                hits.append({
                    "chunk_id": ck["chunk_id"],
                    "doc_info": f"Văn bản số {meta.get('doc_number')}",
                    "effective_date": meta.get("effective_date", "Chưa xác định"),
                    "article": dieu,
                    "content": ck["original_text"]
                })
            return hits
        except Exception as e:
            logger.error(f"Lỗi BM25 Sparse search: {e}")
            return []

    def _search_exact_neo4j(self, query: str, top_k: int) -> list[dict]:
        """Tầng 3: Exact & Structural Search (Neo4j)."""
        # Phân tích thực thể số hiệu (VD: 100/2019/NĐ-CP hoặc Điều 12)
        match_doc = re.search(r'\d+/\d+/[A-ZĐ\-]+', query)
        match_art = re.search(r'Điều\s+(\d+[a-zA-Z]?)', query, re.IGNORECASE)

        cypher = ""
        params = {"top_k": top_k}

        if match_doc and match_art:
            cypher = """
            MATCH (doc:LawDocument) WHERE doc.doc_number CONTAINS $doc_num
            MATCH (doc)-[:HAS_CHAPTER]->(ch)-[:HAS_ARTICLE]->(art:Article)
            WHERE art.name = $art_name
            MATCH (art)-[:HAS_VERSION]->(ctv)-[:HAS_CHUNK]->(ck)
            RETURN doc.name AS DocName, doc.doc_number AS DocNum, doc.effective_date AS EffDate, art.name AS Article, ck.chunk_id AS ChunkId, ck.text AS Content
            LIMIT $top_k
            """
            params["doc_num"] = match_doc.group(0)
            params["art_name"] = f"Điều {match_art.group(1)}"
        elif match_art:
            cypher = """
            MATCH (art:Article) WHERE art.name = $art_name
            MATCH (art)-[:HAS_VERSION]->(ctv)-[:HAS_CHUNK]->(ck)
            MATCH (doc:LawDocument)-[:HAS_CHAPTER]->(ch)-[:HAS_ARTICLE]->(art)
            RETURN doc.name AS DocName, doc.doc_number AS DocNum, doc.effective_date AS EffDate, art.name AS Article, ck.chunk_id AS ChunkId, ck.text AS Content
            LIMIT $top_k
            """
            params["art_name"] = f"Điều {match_art.group(1)}"
        else:
            # Fallback sang Contains search
            cypher = """
            MATCH (doc:LawDocument)
            WHERE doc.name CONTAINS $query OR doc.doc_number CONTAINS $query
            MATCH (doc)-[:HAS_CHAPTER]->(ch)-[:HAS_ARTICLE]->(art)-[:HAS_VERSION]->(ctv)-[:HAS_CHUNK]->(ck)
            RETURN doc.name AS DocName, doc.doc_number AS DocNum, doc.effective_date AS EffDate, art.name AS Article, ck.chunk_id AS ChunkId, ck.text AS Content
            LIMIT $top_k
            """
            params["query"] = query

        hits = []
        with self.neo4j_driver.session() as session:
            try:
                results = session.run(cypher, **params).data()
                for meta in results:
                    hits.append({
                        "chunk_id": meta.get("ChunkId"),
                        "doc_info": f"{meta['DocName']} (Số hiệu: {meta['DocNum']})",
                        "effective_date": meta.get('EffDate', 'Chưa xác định'),
                        "article": meta.get('Article', 'Không xác định'),
                        "content": meta['Content']
                    })
            except Exception as e:
                logger.error(f"Lỗi query Neo4j: {e}")
        return hits

    def _rrf_fusion(self, dense_results, sparse_results, exact_results, weights=(0.4, 0.3, 0.3), top_k=3):
        """
        Reciprocal Rank Fusion (RRF) - Phép kết hợp thứ hạng ngược.
        Độ phức tạp: O(M log M) với M là tổng số hits thu được, tối ưu Big O.
        """
        rrf_scores = defaultdict(float)
        docs_cache = {}
        k_param = 60.0  # Hằng số smoothing tiêu chuẩn cho RRF

        # Gom nhóm kết quả
        for list_idx, hits in enumerate([dense_results, sparse_results, exact_results]):
            w = weights[list_idx]
            for rank, hit in enumerate(hits):
                cid = hit["chunk_id"]
                if not cid:
                    continue
                rrf_scores[cid] += w * (1.0 / (k_param + rank + 1))
                if cid not in docs_cache:
                    docs_cache[cid] = hit

        # Sắp xếp theo score giảm dần
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [docs_cache[cid] for cid in sorted_cids]

    def search_context(self, query: str, top_k: int = 3):
        logger.info(f"Tìm kiếm Hybrid RRF (3-layer) cho: '{query}'")

        # 3 tầng tìm kiếm song song hoặc tuần tự
        dense_hits = self._search_dense(query, top_k * 3)
        sparse_hits = self._search_sparse_bm25(query, top_k * 3)
        exact_hits = self._search_exact_neo4j(query, top_k * 3)

        # Hợp nhất thứ hạng RRF
        final_context = self._rrf_fusion(
            dense_hits, sparse_hits, exact_hits,
            weights=(0.4, 0.3, 0.3), top_k=top_k
        )

        logger.info(f"Lọc ra {len(final_context)} chunks tốt nhất qua RRF.")
        return final_context

    def close(self):
        """Đóng kết nối."""
        try:
            self.neo4j_driver.close()
            logger.info("Da dong ket noi Neo4j.")
        except Exception:
            pass


if __name__ == "__main__":
    pass