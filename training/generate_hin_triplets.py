"""
training/generate_hin_triplets.py - Động cơ khai phá mẫu khó đối lập dựa trên mạng thông tin dị thể (HIN).
Triển khai giải thuật GG-SLM (Graph-Guided Semantic Learning):
- Kết hợp tô-pô HIN 128d (Random Walk) và độ tương đồng từ vựng BM25 (Elasticsearch & Neo4j).
- Nạp toàn diện 1.28M vector đồ thị để tính toán độ tương đồng tô-pô s_p chuẩn xác cho mọi ứng viên.
- Chuẩn hóa nhãn phân cấp và loại trừ triệt để False Negatives theo tiêu chuẩn Q1.
"""

from __future__ import annotations

import os
import re
import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from configs.config import config
from configs.paths import ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_HINTripletMiner")

LEGAL_STOP_WORDS = {
    "căn", "cứ", "luật", "số", "ngày", "tháng", "năm", "của", "về", "việc",
    "quy", "định", "chính", "phủ", "bộ", "ban", "hành", "và", "các", "có",
    "được", "tại", "theo", "khoản", "điều", "như", "sau", "đây", "này"
}


class HINTripletMiner:
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        es_host: Optional[str] = None,
        es_index: Optional[str] = None,
        graph_emb_path: Optional[Path | str] = None,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI") or config.NEO4J_URI
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or config.NEO4J_USER
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD") or getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")
        self.es_host = es_host or os.getenv("ES_HOST") or getattr(config, "ES_HOST", "http://localhost:9200")
        self.es_index = es_index or getattr(config, "ES_INDEX_NAME", "vietlaw_sparse_idx")
        self.graph_emb_file = Path(graph_emb_path) if graph_emb_path else (ARTIFACTS_DIR / "graph_embeddings_128d.parquet")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            connection_acquisition_timeout=15.0,
        )
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j Engine thành công tại %s.", self.neo4j_uri)

        self._ensure_neo4j_fulltext_index()

        self.es = None
        self.es_available = False
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch(hosts=[self.es_host], request_timeout=3.0)
            if self.es.ping() and self.es.indices.exists(index=self.es_index):
                self.es_available = True
                logger.info("✓ Elasticsearch index [%s] sẵn sàng.", self.es_index)
        except Exception:
            pass

        self.graph_embeddings: Dict[str, np.ndarray] = {}

    def _ensure_neo4j_fulltext_index(self):
        """Đảm bảo chỉ mục toàn văn trên Neo4j bám sát trường content thực tế."""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content]"
                ).consume()
            except Exception as e:
                logger.debug("Thông tin Fulltext Index: %s", e)

    def close(self):
        self.driver.close()
        if self.es:
            try:
                self.es.close()
            except Exception:
                pass

    def fetch_positive_pairs_from_neo4j(self, limit: int = 50000) -> List[Dict[str, str]]:
        """
        Trích xuất các cặp Chunks liên kết tích cực.
        - LOẠI BỎ quan hệ bãi bỏ/thay thế (BAI_BO, THAY_THE) khỏi tập Positive.
        - Duy trì biến d1 trong scope Cypher để gán nhãn macro nhất quán.
        """
        logger.info("Truy xuất cặp Chunks liên kết tích cực từ Neo4j...")
        pairs: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        half_limit = max(1000, limit // 2)

        # 1. Cặp nội văn bản
        intra_query = """
        MATCH (d:LawDocument)-[:HAS_CHUNK]->(c1:Chunk)
        MATCH (d)-[:HAS_CHUNK]->(c2:Chunk)
        WHERE c1.chunk_id < c2.chunk_id
          AND c1.chunk_id IS NOT NULL AND c2.chunk_id IS NOT NULL
        RETURN c1.chunk_id AS a_id,
               coalesce(c1.content, '') AS a_text,
               c2.chunk_id AS p_id,
               coalesce(c2.content, '') AS p_text,
               coalesce(d.doc_number, 'CHUNG') AS macro
        LIMIT $limit
        """

        # 2. Cặp liên văn bản hướng dẫn/dẫn chiếu (giữ nguyên d1 trong scope)
        inter_query = """
        MATCH (d1:LawDocument)-[r:LEGAL_RELATION]->(d2:LawDocument)
        WHERE r.type IN [
            'CAN_CU_BAN_HANH', 'DAN_CHIEU', 'QUY_DINH_CHI_TIET_HUONG_DAN', 'HOP_NHAT', 'HUONG_DAN'
        ]
        WITH d1, d2, r LIMIT 5000
        MATCH (d1)-[:HAS_CHUNK]->(c1:Chunk)
        WITH d1, d2, c1, r LIMIT 25000
        MATCH (d2)-[:HAS_CHUNK]->(c2:Chunk)
        WHERE c1.chunk_id <> c2.chunk_id
          AND c1.chunk_id IS NOT NULL AND c2.chunk_id IS NOT NULL
        RETURN c1.chunk_id AS a_id,
               coalesce(c1.content, '') AS a_text,
               c2.chunk_id AS p_id,
               coalesce(c2.content, '') AS p_text,
               coalesce(d1.doc_number, 'LIEN_VAN_BAN') AS macro
        LIMIT $limit
        """

        with self.driver.session() as session:
            for rec in session.run(intra_query, limit=half_limit):
                u, v = str(rec["a_id"]), str(rec["p_id"])
                if u and v and u != v and (u, v) not in seen:
                    seen.add((u, v))
                    pairs.append({
                        "anchor_id": u, "anchor_text": str(rec["a_text"]),
                        "positive_id": v, "positive_text": str(rec["p_text"]),
                        "hierarchy_label": str(rec["macro"]),
                    })

            rem = limit - len(pairs)
            if rem > 0:
                for rec in session.run(inter_query, limit=rem):
                    u, v = str(rec["a_id"]), str(rec["p_id"])
                    if u and v and u != v and (u, v) not in seen:
                        seen.add((u, v))
                        pairs.append({
                            "anchor_id": u, "anchor_text": str(rec["a_text"]),
                            "positive_id": v, "positive_text": str(rec["p_text"]),
                            "hierarchy_label": str(rec["macro"]),
                        })

        logger.info("✓ Trích xuất thành công %d cặp Positive liên kết từ Neo4j.", len(pairs))
        return pairs

    def load_graph_embeddings(self) -> None:
        """Nạp toàn bộ 1.28M vector đồ thị vào RAM và chuẩn hóa L2 (chỉ tốn ~620MB RAM)."""
        if not self.graph_emb_file.exists():
            logger.warning("Không tìm thấy tệp vector đồ thị tại %s. Sẽ dùng BM25 thuần túy.", self.graph_emb_file)
            return

        logger.info("Đang nạp toàn bộ vector đồ thị 128d từ %s...", self.graph_emb_file.name)
        try:
            df_emb = pd.read_parquet(self.graph_emb_file, columns=["chunk_id", "graph_embedding"])
            chunk_ids = df_emb["chunk_id"].astype(str).values
            raw_embs = np.vstack(df_emb["graph_embedding"].values).astype(np.float32)

            # Chuẩn hóa L2 tốc độ cao qua C/BLAS
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-9
            norm_embs = raw_embs / norms

            self.graph_embeddings = {cid: norm_embs[i] for i, cid in enumerate(chunk_ids)}
            del df_emb, raw_embs, norm_embs
            logger.info("✓ Đã nạp thành công %d vector tô-pô HIN vào RAM.", len(self.graph_embeddings))
        except Exception as exc:
            logger.warning("Bỏ qua nạp vector đồ thị (%s). Sẽ dùng BM25 thuần túy.", exc)

    def extract_salient_query_terms(self, text: str) -> str:
        tokens = re.findall(r"\w+", text.lower())
        salient = [t for t in tokens if t not in LEGAL_STOP_WORDS and len(t) > 1]
        if len(salient) < 3:
            return " ".join(tokens[:10])
        return " ".join(salient[:12])

    def _query_candidate_negatives(self, query_tokens: str, a_id: str, p_id: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        candidates = []
        if self.es_available:
            try:
                res = self.es.search(
                    index=self.es_index,
                    query={
                        "bool": {
                            "must": [{"match": {"content": {"query": query_tokens, "operator": "or"}}}],
                            "must_not": [{"ids": {"values": [a_id, p_id]}}],
                        }
                    },
                    size=top_k,
                    source=["content"],
                )
                for h in res.get("hits", {}).get("hits", []):
                    c_id = str(h["_id"])
                    c_text = h.get("_source", {}).get("content") or ""
                    score = float(h.get("_score") or 1.0)
                    if len(c_text.strip()) >= 30:
                        candidates.append((c_id, c_text, score))
                if candidates:
                    return candidates
            except Exception:
                pass

        try:
            clean_q = re.sub(r"[\+\-\&\|\!\(\)\{\}\[\]\^\"~*\?:\/\\]", " ", query_tokens)
            clean_q = " ".join(clean_q.split())
            if not clean_q:
                return []
            cypher = """
            CALL db.index.fulltext.queryNodes("chunk_fulltext", $q) YIELD node, score
            WHERE node.chunk_id <> $a_id AND node.chunk_id <> $p_id
            RETURN node.chunk_id AS cand_id, coalesce(node.content, '') AS cand_text, score
            LIMIT $limit
            """
            with self.driver.session() as session:
                records = session.run(cypher, q=clean_q, a_id=a_id, p_id=p_id, limit=top_k)
                for r in records:
                    txt = str(r["cand_text"] or "")
                    if len(txt.strip()) >= 30:
                        candidates.append((str(r["cand_id"]), txt, float(r["score"] or 1.0)))
        except Exception:
            pass
        return candidates

    def mine_triplets(
        self,
        pairs: List[Dict[str, str]],
        beta: float = 0.5,
        top_neg_candidates: int = 10,
        max_triplets: int = 50000,
    ) -> pd.DataFrame:
        triplets: List[Dict[str, Any]] = []
        validated_pool = [p["positive_text"] for p in pairs if len(p["positive_text"]) > 50]

        logger.info("Bắt đầu khai phá Hard Negatives trên %d cặp...", min(len(pairs), max_triplets))
        pbar = tqdm(pairs[:max_triplets], desc="Khai phá HIN Triplets", unit=" pair")

        for item in pbar:
            a_id = item["anchor_id"]
            a_text = item["anchor_text"]
            p_id = item["positive_id"]
            p_text = item["positive_text"]
            macro = item["hierarchy_label"]

            query_tokens = self.extract_salient_query_terms(a_text)
            best_neg_text = None
            best_neg_id = "N/A"
            max_hardness = -float("inf")

            candidates = self._query_candidate_negatives(query_tokens, a_id, p_id, top_k=top_neg_candidates)

            for cand_id, cand_text, bm25_score in candidates:
                # LOẠI TRỪ FALSE NEGATIVE: Không được trùng anchor và không được trùng positive
                if cand_text.strip() == p_text.strip() or cand_text.strip() == a_text.strip():
                    continue

                s_p = 0.0
                if p_id in self.graph_embeddings and cand_id in self.graph_embeddings:
                    s_p = max(0.0, float(np.dot(self.graph_embeddings[p_id], self.graph_embeddings[cand_id])))

                norm_bm25 = min(bm25_score / 25.0, 1.0)
                # Hardness cao: Lexical match cao (dễ nhầm chữ) nhưng Graph similarity thấp (khác chế tài/nhánh luật)
                hardness = beta * norm_bm25 + (1.0 - beta) * (1.0 - s_p)

                if hardness > max_hardness:
                    max_hardness = hardness
                    best_neg_text = cand_text
                    best_neg_id = cand_id

            if best_neg_text is None and validated_pool:
                safe_fallbacks = [
                    t for t in validated_pool 
                    if t.strip() != p_text.strip() and t.strip() != a_text.strip()
                ]
                if safe_fallbacks:
                    best_neg_text = random.choice(safe_fallbacks)
                    best_neg_id = "random_fallback"
                    max_hardness = 0.5

            if best_neg_text is not None:
                triplets.append({
                    "anchor": a_text,
                    "positive": p_text,
                    "negative": best_neg_text,
                    "query": a_text,
                    "hard_negative": best_neg_text,
                    "anchor_id": a_id,
                    "positive_id": p_id,
                    "negative_id": best_neg_id,
                    "hierarchy_label": macro,
                    "hardness_score": round(float(max_hardness), 4),
                })

        df_triplets = pd.DataFrame(triplets)
        logger.info("✓ Khai phá thành công %d HIN-Guided Triplets đạt chuẩn Q1.", len(df_triplets))
        return df_triplets


def mine_and_export_hin_triplets(output_path: str, limit: int = 50000, beta: float = 0.5):
    miner = HINTripletMiner()
    try:
        raw_pairs = miner.fetch_positive_pairs_from_neo4j(limit=limit)
        valid_pairs = [p for p in raw_pairs if len(p["anchor_text"].strip()) >= 30 and len(p["positive_text"].strip()) >= 30]
        miner.load_graph_embeddings()
        df_triplets = miner.mine_triplets(pairs=valid_pairs, beta=beta, max_triplets=limit)

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_triplets.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
        logger.info("✓ Toàn bộ tập dữ liệu Triplet HIN đã được xuất tại: %s", out_file.resolve())
    finally:
        miner.close()


def main():
    parser = argparse.ArgumentParser(description="Khai phá mẫu khó đối lập HIN-Guided Triplets cho VietLawBERT")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"))
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    mine_and_export_hin_triplets(output_path=args.output, limit=args.limit, beta=args.beta)


if __name__ == "__main__":
    main()