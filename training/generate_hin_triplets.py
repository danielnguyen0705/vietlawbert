"""
generate_hin_triplets.py - Động cơ khai phá mẫu khó đối lập dựa trên mạng thông tin dị thể (HIN).
Kết hợp giải thuật Random Walk with Restart (RWR) trên Neo4j và Elasticsearch BM25.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
import pandas as pd
from neo4j import GraphDatabase

from configs.config import config
from configs.paths import ARTIFACTS_DIR
from rag.es_retriever import LegalElasticsearchRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_HINTripletMiner")


class HINTripletMiner:
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 40, tol: float = 1e-5):
        # alpha = 0.85 là xác suất chuyển bước ngẫu nhiên (transition probability)
        self.alpha = damping_factor
        self.max_iter = max_iter
        self.tol = tol

    def compute_rwr(self, W_norm: sp.csr_matrix, seed_idx: int) -> np.ndarray:
        """
        Giải thuật RWR chuẩn xác theo Gleich (2015):
        r^(k+1) = alpha * W_norm * r^(k) + (1 - alpha) * e_seed
        """
        n_nodes = W_norm.shape[0]
        e = np.zeros(n_nodes, dtype=np.float32)
        e[seed_idx] = 1.0

        r = e.copy()
        alpha = self.alpha

        for _ in range(self.max_iter):
            r_new = alpha * (W_norm.dot(r)) + (1.0 - alpha) * e
            if np.linalg.norm(r_new - r, ord=1) < self.tol:
                break
            r = r_new

        return r

    def mine_triplets(
        self,
        node_ids: List[str],
        node_texts: List[str],
        hierarchy_labels: List[str],
        positive_pairs: List[Tuple[str, str]],
        adj_matrix: sp.csr_matrix,
        es_retriever: LegalElasticsearchRetriever,
        beta: float = 0.6,
        top_neg_candidates: int = 15,
    ) -> pd.DataFrame:
        """Khai phá Hard Negatives: Dùng Elasticsearch lấy ứng viên BM25 cao và phạt topo HIN qua RWR."""
        logger.info("Chuẩn hóa ma trận kề sang ma trận ngẫu nhiên cột (Column-stochastic)...")
        deg = np.array(adj_matrix.sum(axis=0)).flatten()
        deg[deg == 0] = 1.0
        inv_deg = sp.diags(1.0 / deg)
        W_norm = adj_matrix.dot(inv_deg).tocsr()

        id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
        triplets = []

        logger.info("Bắt đầu khai phá mẫu khó trên %d cặp Positive liên kết pháp lý...", len(positive_pairs))

        for anchor_id, pos_id in positive_pairs:
            if anchor_id not in id2idx or pos_id not in id2idx:
                continue

            a_idx = id2idx[anchor_id]
            p_idx = id2idx[pos_id]
            anchor_text = node_texts[a_idx]

            # 1. Tính toán xác suất tiệm cận cấu trúc tô-pô HIN từ nút Positive
            rwr_proximity = self.compute_rwr(W_norm, seed_idx=p_idx)

            # 2. Truy xuất nhanh ứng viên từ vựng tương đồng qua Elasticsearch index
            es_candidates = es_retriever.search_sparse(
                query=anchor_text[:300],
                top_k=top_neg_candidates,
                must_be_effective=True,
            )

            best_hard_neg_idx = None
            max_hardness = -float("inf")

            for cand in es_candidates:
                cand_id = cand.get("chunk_id")
                if not cand_id or cand_id not in id2idx:
                    continue

                cand_idx = id2idx[cand_id]
                if cand_idx == a_idx or cand_idx == p_idx:
                    continue

                s_p = float(rwr_proximity[cand_idx])
                # BM25 điểm số chuẩn hóa mềm
                bm25_val = float(cand.get("score", 1.0))
                norm_bm25 = min(bm25_val / 20.0, 1.0)

                # Công thức Hardness: Từ khóa gây nhầm lẫn nhưng cấu trúc HIN xa rời
                hardness = beta * norm_bm25 + (1.0 - beta) * (1.0 - s_p)

                if hardness > max_hardness:
                    max_hardness = hardness
                    best_hard_neg_idx = cand_idx

            if best_hard_neg_idx is not None:
                triplets.append({
                    "anchor": node_texts[a_idx],
                    "positive": node_texts[p_idx],
                    "negative": node_texts[best_hard_neg_idx],
                    "hierarchy_label": hierarchy_labels[a_idx],
                    "anchor_id": anchor_id,
                    "positive_id": pos_id,
                    "negative_id": node_ids[best_hard_neg_idx],
                    "hardness_score": round(float(max_hardness), 4),
                })

        df_triplets = pd.DataFrame(triplets)
        logger.info("✓ Khai phá thành công %d HIN-Guided Triplets đạt chuẩn Q1.", len(df_triplets))
        return df_triplets


def extract_hin_from_neo4j(
    uri: str, user: str, password: str, limit_pairs: int = 50000
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]], sp.csr_matrix]:
    logger.info("Đang kết nối Neo4j (%s) để trích xuất cấu trúc HIN...", uri)
    driver = GraphDatabase.driver(uri, auth=(user, password))

    chunk_query = """
    MATCH (c:Chunk)
    WHERE (c.content IS NOT NULL AND size(c.content) > 30) OR (c.text IS NOT NULL AND size(c.text) > 30)
    RETURN c.chunk_id AS id, coalesce(c.content, c.text) AS text, coalesce(c.macro_label, 'CHUNG') AS macro
    LIMIT 200000
    """
    pairs_query = """
    MATCH (d1:LawDocument)-[r:LEGAL_RELATION]->(d2:LawDocument)
    MATCH (d1)-[:HAS_CHUNK]->(c1:Chunk)
    MATCH (d2)-[:HAS_CHUNK]->(c2:Chunk)
    WHERE c1.chunk_id <> c2.chunk_id
    RETURN c1.chunk_id AS a_id, c2.chunk_id AS p_id
    LIMIT $limit
    """
    edges_query = """
    MATCH (c1:Chunk)<-[:HAS_CHUNK]-(:LawDocument)-[:LEGAL_RELATION]-(:LawDocument)-[:HAS_CHUNK]->(c2:Chunk)
    RETURN c1.chunk_id AS src, c2.chunk_id AS tgt
    LIMIT 300000
    """

    with driver.session() as session:
        chunk_records = session.run(chunk_query).data()
        pos_records = session.run(pairs_query, parameters={"limit": limit_pairs}).data()
        edge_records = session.run(edges_query).data()

    driver.close()

    node_ids = [r["id"] for r in chunk_records]
    node_texts = [r["text"] for r in chunk_records]
    hierarchy_labels = [r["macro"] for r in chunk_records]
    pos_pairs = [(r["a_id"], r["p_id"]) for r in pos_records]

    id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    row_indices = []
    col_indices = []
    for edge in edge_records:
        u, v = edge["src"], edge["tgt"]
        if u in id2idx and v in id2idx:
            row_indices.extend([id2idx[u], id2idx[v]])
            col_indices.extend([id2idx[v], id2idx[u]])

    data = np.ones(len(row_indices), dtype=np.float32)
    adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))

    logger.info("✓ Trích xuất hoàn tất: %d Chunks, %d Positive Pairs, Kích thước ma trận kề: %s", n_nodes, len(pos_pairs), adj_matrix.shape)
    return node_ids, node_texts, hierarchy_labels, pos_pairs, adj_matrix


def mine_and_export_hin_triplets(
    output_path: str,
    limit: int = 50000,
    beta: float = 0.6,
):
    node_ids, node_texts, hierarchy_labels, pos_pairs, adj_matrix = extract_hin_from_neo4j(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        limit_pairs=limit,
    )

    es_retriever = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)
    miner = HINTripletMiner(damping_factor=0.85, max_iter=30)
    df_triplets = miner.mine_triplets(
        node_ids=node_ids,
        node_texts=node_texts,
        hierarchy_labels=hierarchy_labels,
        positive_pairs=pos_pairs,
        adj_matrix=adj_matrix,
        es_retriever=es_retriever,
        beta=beta,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_triplets.to_parquet(out_file, engine="pyarrow", compression="snappy")
    logger.info("✓ Toàn bộ tập dữ liệu Triplet HIN đã được xuất tại: %s", out_file.resolve())


def main():
    parser = argparse.ArgumentParser(description="Khai phá mẫu khó đối lập HIN-Guided Triplets cho VietLawBERT")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"))
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--beta", type=float, default=0.6)
    args = parser.parse_args()

    mine_and_export_hin_triplets(output_path=args.output, limit=args.limit, beta=args.beta)


if __name__ == "__main__":
    main()