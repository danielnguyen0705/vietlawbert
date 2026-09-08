"""
generate_hin_triplets.py - Động cơ khai phá mẫu khó đối lập dựa trên mạng thông tin dị thể (HIN).
Hiện thực hóa giải thuật RWR trên ma trận thưa CSR kết hợp BM25.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
import pandas as pd
from rank_bm25 import BM25Okapi
from neo4j import GraphDatabase

from configs.config import config
from configs.paths import ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_HINTripletMiner")


class HINTripletMiner:
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 40, tol: float = 1e-5):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol

    def compute_rwr(self, W_norm: sp.csr_matrix, seed_idx: int) -> np.ndarray:
        """
        Tính toán xác suất Random Walk with Restart (RWR) từ nút seed:
        r = (1 - c) * W_norm * r + c * e
        """
        n_nodes = W_norm.shape[0]
        e = np.zeros(n_nodes, dtype=np.float32)
        e[seed_idx] = 1.0

        r = e.copy()
        c = self.damping_factor

        for _ in range(self.max_iter):
            r_new = (1.0 - c) * (W_norm.dot(r)) + c * e
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
        beta: float = 0.6,
        top_neg_candidates: int = 30,
    ) -> pd.DataFrame:
        """Khai phá các mẫu Hard Negatives: từ khóa gây nhiễu (BM25 cao) nhưng topo HIN xa rời."""
        logger.info("Chuẩn hóa ma trận kề thưa sang ma trận chuyển trạng thái ngẫu nhiên...")
        deg = np.array(adj_matrix.sum(axis=0)).flatten()
        deg[deg == 0] = 1.0
        inv_deg = sp.diags(1.0 / deg)
        W_norm = adj_matrix.dot(inv_deg).tocsr()

        id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
        tokenized_corpus = [text.lower().split() for text in node_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        triplets = []
        logger.info("Bắt đầu khai phá trên %d cặp Positive liên kết pháp lý...", len(positive_pairs))

        for anchor_id, pos_id in positive_pairs:
            if anchor_id not in id2idx or pos_id not in id2idx:
                continue

            a_idx = id2idx[anchor_id]
            p_idx = id2idx[pos_id]

            query_tokens = node_texts[a_idx].lower().split()
            bm25_scores = np.array(bm25.get_scores(query_tokens))

            # 1. Tính toán khoảng cách cấu trúc bằng RWR từ nút Positive
            rwr_proximity = self.compute_rwr(W_norm, seed_idx=p_idx)

            # 2. Thu hồi các ứng viên BM25 cao nhất (từ vựng tương đồng)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_neg_candidates]

            best_hard_neg_idx = None
            max_hardness = -float("inf")

            sub_bm25 = bm25_scores[top_bm25_indices]
            denom = (sub_bm25.max() - sub_bm25.min()) + 1e-9
            norm_bm25 = (sub_bm25 - sub_bm25.min()) / denom

            for i, cand_idx in enumerate(top_bm25_indices):
                if cand_idx == a_idx or cand_idx == p_idx:
                    continue

                # Hardness: Từ khóa tương tự nhưng khoảng cách cấu trúc HIN xa rời
                s_p = rwr_proximity[cand_idx]
                hardness = beta * norm_bm25[i] + (1.0 - beta) * (1.0 - s_p)

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
        logger.info("✓ Khai phá thành công %d HIN-Guided Triplets đạt chuẩn.", len(df_triplets))
        return df_triplets


def extract_hin_from_neo4j(
    uri: str, user: str, password: str, limit_pairs: int = 50000
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]], sp.csr_matrix]:
    """Trích xuất danh sách Chunk, cạnh liên kết pháp lý và xây dựng ma trận kề thưa từ Neo4j."""
    logger.info("Đang kết nối Neo4j (%s) để trích xuất dữ liệu HIN...", uri)
    driver = GraphDatabase.driver(uri, auth=(user, password))

    chunk_query = """
    MATCH (c:Chunk)
    WHERE c.content IS NOT NULL AND size(c.content) > 50
    RETURN c.chunk_id AS id, c.content AS text, coalesce(c.macro_label, 'CHUNG') AS macro
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
    LIMIT 500000
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
            row_indices.append(id2idx[u])
            col_indices.append(id2idx[v])
            row_indices.append(id2idx[v])
            col_indices.append(id2idx[u])

    data = np.ones(len(row_indices), dtype=np.float32)
    adj_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))

    logger.info("✓ Đã trích xuất %d Chunks, %d Positive Pairs, Ma trận kề: %s", n_nodes, len(pos_pairs), adj_matrix.shape)
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

    miner = HINTripletMiner(damping_factor=0.85, max_iter=40)
    df_triplets = miner.mine_triplets(
        node_ids=node_ids,
        node_texts=node_texts,
        hierarchy_labels=hierarchy_labels,
        positive_pairs=pos_pairs,
        adj_matrix=adj_matrix,
        beta=beta,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_triplets.to_parquet(out_file, engine="pyarrow", compression="snappy")
    logger.info("✓ Toàn bộ tập dữ liệu Triplet HIN đã được lưu tại: %s", out_file.resolve())


def main():
    parser = argparse.ArgumentParser(description="Khai phá mẫu khó đối lập HIN-Guided Triplets cho VietLawBERT")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"), help="Đường dẫn lưu tệp Parquet")
    parser.add_argument("--limit", type=int, default=50000, help="Số lượng cặp positive tối đa")
    parser.add_argument("--beta", type=float, default=0.6, help="Trọng số BM25 so với khoảng cách đồ thị HIN")
    args = parser.parse_args()

    mine_and_export_hin_triplets(output_path=args.output, limit=args.limit, beta=args.beta)


if __name__ == "__main__":
    main()