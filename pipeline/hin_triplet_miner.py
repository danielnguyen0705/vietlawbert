"""
hin_triplet_miner.py - Khai phá bộ ba mẫu khó đối lập (HIN-Guided Hard Negative Mining).
Sử dụng cấu trúc tô-pô Neo4j và kho tệp Parquet để sinh tập huấn luyện VietLawBERT-MRL.
Triệt tiêu lỗi tích Descartes bằng cơ chế phân đoạn truy vấn Cypher và lấy mẫu giới hạn.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

import pyarrow as pa
import pyarrow.parquet as pq
from neo4j import GraphDatabase

from configs.config import config
from configs.paths import PROCESSED_DIR, TRIPLETS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_TripletMiner")

PARQUET_DIR = PROCESSED_DIR / "parquet"
OUTPUT_DIR = TRIPLETS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIPLET_SCHEMA = pa.schema([
    ("anchor_id", pa.string()),
    ("anchor_text", pa.string()),
    ("positive_id", pa.string()),
    ("positive_text", pa.string()),
    ("negative_id", pa.string()),
    ("negative_text", pa.string()),
    ("mining_strategy", pa.string()),
])


def load_chunk_lookup(parquet_files: List[Path], max_chunks: int = 400000) -> Dict[str, str]:
    logger.info("Đang nạp bảng băm Chunks từ Parquet vào bộ nhớ...")
    lookup = {}
    for f in parquet_files:
        table = pq.read_table(str(f), columns=["chunk_id", "content"]).to_pydict()
        for cid, text in zip(table["chunk_id"], table["content"]):
            if text and len(text) >= 50:
                lookup[cid] = text
        if len(lookup) >= max_chunks:
            break
    logger.info("✓ Đã nạp %d Chunks hợp lệ vào RAM.", len(lookup))
    return lookup


def mine_triplets_from_graph(
    driver,
    chunk_lookup: Dict[str, str],
    target_triplets: int = 100000,
) -> List[Dict[str, Any]]:
    logger.info("Bắt đầu khai phá %d Triplets dựa trên mạng HIN Neo4j...", target_triplets)
    triplets = []
    seen_pairs: Set[tuple] = set()

    # Truy vấn phân đoạn triệt tiêu Cartesian Product
    cypher_inter_doc = """
    MATCH (d_src:LawDocument)-[r:LEGAL_RELATION]->(d_tgt:LawDocument)
    WHERE toUpper(r.type) IN ["CAN_CU_BAN_HANH", "CAN_CU", "SUA_DOI_BO_SUNG", "DAN_CHIEU", "HUONG_DAN_CHI_TIET", "HUONG_DAN"]
    WITH d_src, d_tgt LIMIT $doc_limit
    MATCH (d_src)-[:HAS_CHUNK]->(c_anchor:Chunk)
    WITH d_src, d_tgt, c_anchor LIMIT $chunk_limit
    MATCH (d_tgt)-[:HAS_CHUNK]->(c_pos:Chunk)
    WITH d_src, c_anchor, c_pos LIMIT $chunk_limit
    MATCH (d_src)-[:HAS_CHUNK]->(c_neg:Chunk)
    WHERE c_neg.chunk_id <> c_anchor.chunk_id
    RETURN c_anchor.chunk_id AS anchor_id,
           c_pos.chunk_id AS positive_id,
           c_neg.chunk_id AS negative_id
    LIMIT $chunk_limit
    """

    cypher_intra_doc = """
    MATCH (a:Article)-[:HAS_CHUNK]->(c1:Chunk)
    MATCH (a)-[:HAS_CHUNK]->(c2:Chunk)
    WHERE elementId(c1) < elementId(c2)
    WITH a, c1, c2 LIMIT $limit
    MATCH (d_other:LawDocument)-[:HAS_CHUNK]->(c_neg:Chunk)
    WHERE c_neg.macro_label = c1.macro_label AND d_other.doc_id <> c1.doc_id
    RETURN c1.chunk_id AS anchor_id,
           c2.chunk_id AS positive_id,
           c_neg.chunk_id AS negative_id
    LIMIT $limit
    """

    with driver.session() as session:
        logger.info("1/2 Đang khai phá bộ ba liên văn bản (Inter-Document)...")
        results_inter = session.run(cypher_inter_doc, doc_limit=1000, chunk_limit=target_triplets // 2)
        for row in results_inter:
            a_id, p_id, n_id = row["anchor_id"], row["positive_id"], row["negative_id"]
            if a_id in chunk_lookup and p_id in chunk_lookup and n_id in chunk_lookup:
                pair_key = (a_id, p_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    triplets.append({
                        "anchor_id": a_id,
                        "anchor_text": chunk_lookup[a_id],
                        "positive_id": p_id,
                        "positive_text": chunk_lookup[p_id],
                        "negative_id": n_id,
                        "negative_text": chunk_lookup[n_id],
                        "mining_strategy": "HIN_INTER_DOCUMENT",
                    })

        remaining = target_triplets - len(triplets)
        if remaining > 0:
            logger.info("2/2 Đang khai phá bộ ba nội văn bản (Intra-Article Hard Negatives)...")
            results_intra = session.run(cypher_intra_doc, limit=remaining * 2)
            for row in results_intra:
                if len(triplets) >= target_triplets:
                    break
                a_id, p_id, n_id = row["anchor_id"], row["positive_id"], row["negative_id"]
                if a_id in chunk_lookup and p_id in chunk_lookup and n_id in chunk_lookup:
                    pair_key = (a_id, p_id)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        triplets.append({
                            "anchor_id": a_id,
                            "anchor_text": chunk_lookup[a_id],
                            "positive_id": p_id,
                            "positive_text": chunk_lookup[p_id],
                            "negative_id": n_id,
                            "negative_text": chunk_lookup[n_id],
                            "mining_strategy": "HIN_INTRA_ARTICLE",
                        })

    logger.info("✓ Tổng số Triplets khai phá thành công: %d mẫu.", len(triplets))
    return triplets


def main():
    parquet_files = sorted(list(PARQUET_DIR.glob("*.parquet")))
    if not parquet_files:
        logger.error("Không tìm thấy Parquet files tại %s", PARQUET_DIR)
        return 1

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        chunk_lookup = load_chunk_lookup(parquet_files)
        triplets_data = mine_triplets_from_graph(driver, chunk_lookup, target_triplets=100000)

        if not triplets_data:
            logger.warning("Không trích xuất được triplet nào. Vui lòng kiểm tra lại đồ thị.")
            return 1

        output_file = OUTPUT_DIR / "hin_triplets_100k.parquet"
        table = pa.Table.from_pylist(triplets_data, schema=TRIPLET_SCHEMA)
        pq.write_table(table, output_file, compression="snappy")

        logger.info("=" * 65)
        logger.info("✓ HOÀN TẤT: Đã đóng gói %d Triplets tại %s", len(triplets_data), output_file)
        logger.info("=" * 65)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())